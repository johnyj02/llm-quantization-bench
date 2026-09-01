#!/usr/bin/env python3
"""Score a generation run through the full ladder. CPU only, no GPU, no queue.

    .venv/bin/python scripts/score_pilot.py                      # run_id=pilot-v1
    .venv/bin/python scripts/score_pilot.py --run-id pilot-v1 --out artifacts/scores

This is the second half of "generate once, score many times" (README §1). Generation
is GPU-bound, nondeterministic and expensive; scoring is deterministic, cheap, and WILL
be revised repeatedly. Keeping them separate is what makes a rung change or a parser fix
cost seconds instead of a queue cycle.

Emits `scores_<run_id>.json` with per-cell accuracy at every rung, the compliance
breakdown, termination classes, and the H1 truncation signal --- plus the two contrasts
job 3 was built to enable:

  * **the rung-1 vs rung-2 gap per arm** (§2.2's headline contrast), and
  * **the AWQ reproducibility replicate** (R12.1), where two byte-different checkpoints
    of the same recipe are scored on the same items.

⚠️ **Nothing here decides anything about H3/H4.** Those are pre-registered GLMM
specifications on the FULL reference runs (`analysis/preregistration.md` §1--§2). This
script reports descriptive per-cell numbers on a 300-item pilot, which is a
sanity-and-plumbing instrument. Treating a pilot number as a result is exactly the
error the prereg exists to prevent, so every consumer of this file should read
`n_items` first.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantartifact.scoring import compliance, extract  # noqa: E402

# math_verify is rung 3 and needs an optional dependency. Absence must DEGRADE, not
# crash: rungs 1/1b/2 are the headline and are pure-python.
try:  # noqa: SIM105
    import math_verify  # noqa: F401
    MATH_VERIFY = True
except ImportError:
    MATH_VERIFY = False

# Rung 2 is primary per §2.2's headline contrast (R17.4); rung 1 reported alongside.
PRIMARY_RUNG = "permissive"

RUNGS = ("strict", "strict_last", "permissive")
if MATH_VERIFY:
    RUNGS = (*RUNGS, "math_verify")


# 🚨 GOLD MUST BE REDUCED BEFORE COMPARISON, AND ONLY FOR SOME TASKS.
#
# `materialize.py` records `gold = str(harness_task.doc_to_target(doc))`, and lm-eval's
# answer for that is NOT uniform across task configs:
#
#   gsm8k      doc_to_target = "{{answer}}"
#              -> "Janet sells 16 - 3 - 4 = ... $18 every day\n#### 18"   FULL SOLUTION
#   gsm8k_cot  doc_to_target = "{{answer.split('####')[-1].strip() ...}}"
#              -> "18"                                                   ALREADY BARE
#
# These are two deliberately different levels of our format gradient (`answer_marker`
# `hash` vs `phrase`, `configs/tasks.yaml`), so both golds are correct as recorded --- but
# a scorer that compares an extracted "18" against the full solution string scores every
# GSM8K item WRONG, silently, and reports a plausible-looking aggregate because ARC and
# BBH still work. ARC gold is already "C" and BBH gold already "8"/"False"/word lists, so
# `gsm8k` is the ONLY affected config and the failure would look like a real result.
#
# Job 2 used `gsm8k_cot` and is therefore unaffected; its accuracy numbers stand.
GOLD_MARKER_REGEX: dict[str, str] = {"gsm8k": r"####\s*(-?[0-9.,]+)"}


def reduce_gold(gold: str, task: str) -> str:
    """Apply the task's own answer marker to the gold. Identity when none applies.

    Fails LOUD rather than silently: a task registered here whose gold does not match
    its own declared marker means the recorded gold is not what this expects, and
    quietly returning the raw string would restore the exact bug this guards.
    """
    pattern = GOLD_MARKER_REGEX.get(task)
    if pattern is None:
        return gold
    import re  # noqa: PLC0415 - only needed for the registered tasks
    match = re.search(pattern, gold)
    if match is None:
        raise ValueError(
            f"gold for task {task!r} does not match its declared answer marker "
            f"{pattern!r}: {gold[-80:]!r}. Either the task config changed or the "
            f"generations were materialised with a different lm_eval_task."
        )
    return match.group(1).strip()


def read_records(root: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(root.rglob("*.jsonl.gz")):
        with gzip.open(path, "rt") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def score_one(record: dict) -> dict:
    """Score a single generation. Returns the flat row the aggregates read."""
    generation = record["generation_text"]
    task, subtask = record["task"], record.get("subtask")
    gold = reduce_gold(record.get("gold") or "", task)

    scored = extract.score_generation(
        generation, gold, task, subtask,
        rungs=RUNGS, math_verify_available=MATH_VERIFY,
    )
    # `score_generation` returns per-rung dicts of extracted/correct/rule, not the
    # `Extraction` objects `label_compliance` needs, so the two extractors are called
    # directly. They are pure functions of (generation, task, subtask), so this cannot
    # disagree with what was just scored.
    strict = extract.extract_strict(generation, task, subtask)
    permissive = extract.extract_permissive(generation, task, subtask)

    labels = compliance.label_compliance(
        generation, task, subtask,
        finish_reason=record.get("finish_reason"),
        stop_reason=record.get("stop_reason"),
        strict=strict, permissive=permissive,
    )

    row = {
        "cell_id": record["cell_id"],
        # 🚨 K=2 (prereg §9f). Carried onto every row because `aggregate()` MUST group by it:
        # ten cells of the reference run are generated TWICE, in two separate engine loads,
        # to estimate the invocation-noise floor. Without this the two invocations are pooled
        # into one accuracy for exactly those ten cells and not the other five -- so the
        # K=2 cells would report a two-invocation mean and the rest a one-invocation mean,
        # which is not comparable and looks entirely plausible.
        "repeat_idx": record.get("repeat_idx", 0),
        # Provenance only, never a grouping key (R21.3): which engine load produced this.
        "invocation_id": record.get("invocation_id", ""),
        "task": task,
        "subtask": subtask,
        "item_id": record["item_id"],
        "n_gen_tokens": record.get("n_gen_tokens"),
        "max_gen_toks": record.get("max_gen_toks"),
        "finish_reason": record.get("finish_reason"),
        "compliance": labels.compliance,
        "termination": labels.termination,
    }
    for rung in RUNGS:
        row[f"correct_{rung}"] = bool(scored[rung]["correct"])
        row[f"found_{rung}"] = scored[rung]["extracted"] is not None
    return row


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def cell_key(cell_id: str, repeat_idx: int) -> str:
    """JSON-safe display key. `cell_id` for the first invocation, `cell_id#k1` after.

    ⚠️ The suffix is added ONLY for repeat_idx != 0, deliberately. Every run before K=2 has
    repeat_idx 0 throughout, so their keys are unchanged and existing saved scores stay
    readable and comparable. The cost of that choice is that the key alone does not tell you
    the repeat index -- so `repeat_idx` is ALSO stored inside every entry, and consumers must
    read the field rather than parse the key. A value that has to be recovered by string
    parsing is the shape of bug this project keeps finding.
    """
    return cell_id if repeat_idx == 0 else f"{cell_id}#k{repeat_idx}"


def aggregate(rows: list[dict]) -> dict:
    """Per-cell statistics, keyed by (cell_id, repeat_idx).

    🚨 THE KEY INCLUDES `repeat_idx`, AND THAT IS NOT OPTIONAL. Grouping by `cell_id` alone
    silently averaged the two K=2 invocations of the ten designated reference cells into a
    single accuracy, while the other five cells were single-invocation means. The two are not
    comparable, and the resulting table looks completely normal.

    ⚠️ Why the join-key uniqueness check below does NOT catch it, which is the instructive
    part: `repeat_idx` IS in `records.JOIN_KEY`, so the two invocations are correctly distinct
    records and the duplicate check passes. The guard that exists to catch double-counting
    confirms the records are fine and then the aggregator pools them anyway. A uniqueness
    constraint on the *records* says nothing about the grouping applied to them.

    Eleventh instance of this project's recurring family, one layer downstream of the ten
    already recorded: a value that determines an outcome was not part of that outcome's
    identity -- here the outcome is an aggregate rather than a cached cell.
    """
    by_cell: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[(row["cell_id"], row.get("repeat_idx", 0))].append(row)

    cells = {}
    for (cell_id, repeat_idx), group in sorted(by_cell.items()):
        # Both stored explicitly so no consumer has to parse them back out of the key.
        entry: dict = {"cell_id": cell_id, "repeat_idx": repeat_idx,
                       "n_items": len(group),
                       "invocation_ids": sorted({r["invocation_id"] for r in group
                                                 if r.get("invocation_id")})}
        for rung in RUNGS:
            entry[f"acc_{rung}"] = mean(r[f"correct_{rung}"] for r in group)
            entry[f"found_{rung}"] = mean(r[f"found_{rung}"] for r in group)
        # §2.2's headline contrast: what leniency buys, per arm. If this gap differs
        # ACROSS arms, the measured quantization delta is scorer-dependent, which is
        # the paper's thesis; if it is constant, extraction is a level effect only.
        entry["rung_gap_permissive_minus_strict"] = (
            entry["acc_permissive"] - entry["acc_strict"]
        )
        entry["compliance"] = dict(Counter(r["compliance"] for r in group))
        entry["termination"] = dict(Counter(r["termination"] for r in group))
        # H1's raw signal. `TRUNC_BUDGET` means the generation hit max_gen_toks, so a
        # differential rate BY PRECISION is the truncation channel; `TRUNC_STOPSEQ` is
        # the pre-registered differential test of §4.
        entry["trunc_budget_rate"] = mean(
            r["termination"] == compliance.TRUNC_BUDGET for r in group)
        entry["trunc_stopseq_rate"] = mean(
            r["termination"] == compliance.TRUNC_STOPSEQ for r in group)
        lengths = [r["n_gen_tokens"] for r in group if r["n_gen_tokens"] is not None]
        entry["n_gen_tokens_mean"] = mean(lengths)
        entry["n_gen_tokens_median"] = statistics.median(lengths) if lengths else None
        cells[cell_key(cell_id, repeat_idx)] = entry

    # Per (cell, task) as well: the format gradient is a per-task claim, and a blended
    # number over gsm8k + arc + six BBH subtasks would hide it entirely.
    by_task: dict[str, dict] = {}
    for (cell_id, repeat_idx), group in sorted(by_cell.items()):
        per_task: dict[str, dict] = {}
        buckets: dict[str, list[dict]] = defaultdict(list)
        for row in group:
            key = row["task"] if not row["subtask"] else f"{row['task']}/{row['subtask']}"
            buckets[key].append(row)
        for key, bucket in sorted(buckets.items()):
            per_task[key] = {
                "n_items": len(bucket),
                **{f"acc_{rung}": mean(r[f"correct_{rung}"] for r in bucket)
                   for rung in RUNGS},
                "trunc_budget_rate": mean(
                    r["termination"] == compliance.TRUNC_BUDGET for r in bucket),
            }
        by_task[cell_key(cell_id, repeat_idx)] = per_task
    return {"cells": cells, "by_task": by_task}


def awq_replicate(cells: dict) -> dict | None:
    """R12.1: two byte-different checkpoints of ONE recipe, scored on the same items.

    The irreproducibility finding so far rests on an S3 byte comparison, which shows the
    weights differ but not whether it MATTERS. This is the accuracy consequence, and it
    is the whole reason cells 16--17 exist.
    """
    # Both are repeat_idx 0 (neither is a K=2 cell), so the bare cell_id IS the key --
    # asserted rather than assumed, because if either ever joins the K=2 set the bare
    # lookup would silently miss and this contrast would vanish from the report.
    a, b = "qwen2.5-3b|awq-4bit-offload", "qwen2.5-3b|awq-4bit-nooffload"
    if a not in cells or b not in cells:
        return None
    for key in (a, b):
        if cells[key].get("repeat_idx", 0) != 0:
            raise SystemExit(
                f"awq_replicate: {key} has repeat_idx "
                f"{cells[key]['repeat_idx']}; this contrast assumes a single invocation "
                f"per arm and must be re-specified if the arm is now run at K=2"
            )
    out: dict = {"arms": [a, b], "n_items": cells[a]["n_items"]}
    for rung in RUNGS:
        out[f"acc_{rung}"] = [cells[a][f"acc_{rung}"], cells[b][f"acc_{rung}"]]
        out[f"delta_{rung}"] = cells[b][f"acc_{rung}"] - cells[a][f"acc_{rung}"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a run through the full ladder")
    parser.add_argument("--run-id", default="pilot-v1")
    parser.add_argument("--generations", default=None,
                        help="default: artifacts/generations/<run-id>")
    parser.add_argument("--out", default="artifacts/scores")
    args = parser.parse_args()

    root = Path(args.generations) if args.generations else (
        REPO / "artifacts" / "generations" / args.run_id)
    if not root.is_dir():
        raise SystemExit(f"no generations at {root} — sync them from S3 first")

    records = read_records(root)
    if not records:
        raise SystemExit(f"no *.jsonl.gz records under {root}")
    print(f"records      : {len(records):,} from {root}")
    print(f"rungs        : {list(RUNGS)}"
          f"{'' if MATH_VERIFY else '   (math_verify UNAVAILABLE — rung 3 skipped)'}")

    # ⚠️ The join key is a UNIQUENESS CONSTRAINT, not a convention (README §10). A
    # duplicate means two generations are being scored as one item, which silently
    # doubles that item's weight in every aggregate. Assert, don't hope.
    keys = Counter(
        (r["run_id"], r["cell_id"], r["task"], r.get("subtask"), r["item_id"],
         r.get("budget_level"), r.get("prompt_variant"), r.get("sample_idx"),
         r.get("repeat_idx"))
        for r in records
    )
    duplicates = [k for k, n in keys.items() if n > 1]
    if duplicates:
        raise SystemExit(
            f"JOIN KEY VIOLATED: {len(duplicates)} duplicate keys, e.g. {duplicates[0]}"
        )
    print(f"join key     : {len(keys):,} unique, no duplicates")

    rows = [score_one(r) for r in records]
    report = aggregate(rows)
    report["run_id"] = args.run_id
    report["n_records"] = len(records)
    report["rungs"] = list(RUNGS)
    report["math_verify_available"] = MATH_VERIFY
    report["awq_replicate"] = awq_replicate(report["cells"])

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scores_{args.run_id}.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    print()
    header = f"{'cell':36} {'n':>4}"
    for rung in RUNGS:
        header += f" {rung[:9]:>9}"
    print(header + f" {'perm-str':>9} {'trunc':>7} {'tok':>6}")
    print("-" * (len(header) + 25))
    for cell_id, entry in report["cells"].items():
        line = f"{cell_id:36} {entry['n_items']:>4}"
        for rung in RUNGS:
            line += f" {entry[f'acc_{rung}']:>8.1%}"
        line += (f" {entry['rung_gap_permissive_minus_strict']:>+8.1%}"
                 f" {entry['trunc_budget_rate']:>6.1%}"
                 f" {entry['n_gen_tokens_median']:>6.0f}")
        print(line)

    # 🚨 Report the K=2 pairs as PAIRS. The whole point of the second invocation is that the
    # two numbers are compared, not averaged -- printing them adjacently makes an accidental
    # pooling visible at a glance instead of hiding inside one row.
    k2 = {}
    for entry in report["cells"].values():
        if entry.get("repeat_idx", 0) != 0 or any(
            e.get("cell_id") == entry.get("cell_id") and e.get("repeat_idx", 0) != 0
            for e in report["cells"].values()
        ):
            k2.setdefault(entry["cell_id"], {})[entry.get("repeat_idx", 0)] = entry
    if k2:
        print(f"\n=== K=2 pairs (prereg §9f): {len(k2)} cells, two invocations each ===")
        print(f"{'cell':32} {'acc k0':>8} {'acc k1':>8} {'|delta|':>8}   invocations")
        for cell_id, pair in sorted(k2.items()):
            a0, a1 = pair.get(0), pair.get(1)
            if not (a0 and a1):
                print(f"{cell_id:32}  INCOMPLETE — only repeat_idx "
                      f"{sorted(pair)} present")
                continue
            v0 = a0[f"acc_{PRIMARY_RUNG}"]
            v1 = a1[f"acc_{PRIMARY_RUNG}"]
            ids = len(set(a0.get("invocation_ids", []) + a1.get("invocation_ids", [])))
            print(f"{cell_id:32} {v0:>7.2%} {v1:>7.2%} {abs(v1 - v0):>7.2%}   "
                  f"{ids} distinct")
        print("  (the per-arm invocation SD in §9f is estimated from these differences)")

    print(f"\nreport -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
