#!/usr/bin/env python3
"""H1: does measured 4-bit degradation shrink as the generation budget rises? `A15`, prereg §3.

    .venv/bin/python scripts/analyze_h1.py

**Why this is a separate script.** `analyze_run.py` takes one `--run-id` and reads one
directory, so it cannot see both runs. H1's curve needs three points, and they live in two
runs: the 1x level is in `main-v1` (the reference condition, by the prefix-nesting design in
§2.2) and 2x and 4x are in `sweep-v1`. Without a cross-run join H1 has two points, which is a
line rather than a curve and cannot show shrinkage.

The join is on `(cell_id, task, subtask, item_id)`, deliberately **not** `JOIN_KEY`, whose
first field is `run_id` so a 1x and a 2x record can never share one. That is correct for
uniqueness and wrong for this, because here they are the same item at different budgets.

🚨 **The scored prefix is the primary quantity, not the full generation** (§5b.1). R35.Q4 found
H1's sign FLIPS on Llama depending on which you use: measured on the full generation
quantization looks like it shortens chain-of-thought, measured on the scored answer it
lengthens it. Both are reported here so the reader sees the choice rather than inheriting it.
"""

from __future__ import annotations

import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantartifact.analysis.results import Effect, NoiseClass, holm_bonferroni  # noqa: E402
from quantartifact.scoring import extract  # noqa: E402

#: 1x comes from the reference run by the §2.2 nesting; 2x and 4x from the sweep.
SOURCES = {"1x": "main-v1", "2x": "sweep-v1", "4x": "sweep-v1"}
LEVELS = ("1x", "2x", "4x")
#: §2.2 excludes bnb-NF4 and GPTQ-8bit from the expensive arms, so the sweep has three.
ARMS = ("bf16", "gptq-4bit", "awq-4bit")
#: `gsm8k` (hash marker) is excluded from interaction estimates by §9c and is not swept.
TASKS = ("gsm8k_cot", "bbh")
RUNG = "permissive"   # §2.2: rung 2 primary, rung 1 alongside



def shrinkage_effect(at_1x: list[int], at_4x: list[int], name: str) -> Effect:
    """Change in degradation from 1x to 4x, with a cluster bootstrap over items.

    Each input is one signed per-item contribution (bf16 correct minus arm correct), so the
    mean is the degradation in accuracy points at that budget and the difference of means is
    H1's quantity: a NEGATIVE value means degradation SHRANK as the budget rose, which is
    H1's registered direction.

    10,000 draws with a fixed seed, matching the convention the rest of the analysis layer
    uses, so the interval is reproducible from the repository alone.

    ⚠️ Class 3 (compounding) under §9f: it contrasts two arms, so two independent invocation
    noises enter and their variances add.
    """
    import random  # noqa: PLC0415

    rng = random.Random(20260908)
    point = 100 * (sum(at_4x) / len(at_4x) - sum(at_1x) / len(at_1x))
    draws = []
    n1, n4 = len(at_1x), len(at_4x)
    for _ in range(10_000):
        a = sum(at_1x[rng.randrange(n1)] for _ in range(n1)) / n1
        b = sum(at_4x[rng.randrange(n4)] for _ in range(n4)) / n4
        draws.append(100 * (b - a))
    draws.sort()
    return Effect(
        name=name, estimate=point,
        ci_low=draws[int(0.025 * len(draws))], ci_high=draws[int(0.975 * len(draws))],
        unit="pp", n=min(n1, n4),
        interval_method="cluster bootstrap over items, 10000 percentile draws",
        noise_class=NoiseClass.COMPOUNDING,
        context={"seed": 20260908, "n_1x": n1, "n_4x": n4,
                 "direction": "negative = degradation shrank as the budget rose (H1)"},
    )

def item_key(r: dict) -> tuple:
    """The cross-run identity of an item. NOT `JOIN_KEY`: that starts with `run_id`."""
    return (r["cell_id"], r["task"], r.get("subtask"), r["item_id"])


def load(run: str, levels: set[str]) -> dict:
    out: dict[tuple, dict] = {}
    root = REPO / "artifacts" / "generations" / run
    if not root.is_dir():
        raise SystemExit(f"missing {root.relative_to(REPO)}; sync it from S3")
    for path in root.rglob("*.jsonl.gz"):
        with gzip.open(path, "rt") as handle:
            for line in handle:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["budget_level"] not in levels or r["task"] not in TASKS:
                    continue
                # K=2 repeats are two measurements of one cell (§9f). Keep the first so the
                # curve is one observation per item, not a silently doubled one.
                if r.get("repeat_idx", 0) != 0:
                    continue
                out[(r["budget_level"], *item_key(r))] = r
    return out


def correct(r: dict, rung: str) -> bool:
    s = extract.score_generation(extract.scored_text_for(r), r.get("gold") or "",
                                 r["task"], r.get("subtask"),
                                 rungs=(rung,), math_verify_available=False)
    return bool(s[rung]["correct"])


def main() -> int:
    records: dict[tuple, dict] = {}
    for run, levels in {"main-v1": {"1x"}, "sweep-v1": {"2x", "4x"}}.items():
        got = load(run, levels)
        print(f"{run:10} {len(got):>7,} records at {sorted(levels)}")
        records.update(got)

    # ---- the join, asserted rather than assumed ---------------------------------------
    # If the item sets do not actually match across runs, H1 silently loses its 1x level and
    # the curve becomes two points. That must fail loudly, not quietly.
    by_level = defaultdict(set)
    for (level, *key) in records:
        by_level[level].add(tuple(key))
    common = set.intersection(*(by_level[lv] for lv in LEVELS))
    print("\njoin on (cell_id, task, subtask, item_id):")
    for lv in LEVELS:
        print(f"  {lv}: {len(by_level[lv]):>6,} items")
    print(f"  common to all three levels: {len(common):,}")
    if not common:
        raise SystemExit(
            "JOIN FAILED: no item appears at all three budget levels. H1 cannot be computed. "
            "Check that the sweep's item selection is a prefix-nested subset of the "
            "reference condition's, per §2.2."
        )
    lost = {lv: len(by_level[lv] - common) for lv in LEVELS}
    if any(lost.values()):
        print(f"  ⚠️ dropped as not common to all levels: {lost}")

    report: dict = {"n_items_joined": len(common), "rung": RUNG, "levels": list(LEVELS),
                    "accuracy": {}, "length": {}, "degradation": {}, "effects": []}
    effects: list[Effect] = []

    # ---- accuracy and length per (arm, level) -----------------------------------------
    for task in TASKS:
        items = sorted(k for k in common if k[1] == task)
        for arm in ARMS:
            for lv in LEVELS:
                got = [records[(lv, *k)] for k in items
                       if k[0].endswith(f"|{arm}") and (lv, *k) in records]
                if not got:
                    continue
                tag = f"{task}|{arm}|{lv}"
                report["accuracy"][tag] = statistics.mean(correct(r, RUNG) for r in got)
                # §5b.1: scored length is primary. Full length is reported beside it because
                # R35.Q4 found H1's SIGN flips between the two on Llama.
                report["length"][tag] = {
                    "n": len(got),
                    "scored_mean": statistics.mean(
                        r.get("n_gen_tokens_scored") or r["n_gen_tokens"] for r in got),
                    "full_mean": statistics.mean(r["n_gen_tokens"] for r in got),
                    "hit_cap_rate": statistics.mean(
                        r["finish_reason"] == "length" for r in got),
                }

    # ---- H1 itself: does the bf16-minus-4bit gap SHRINK as the budget rises? -----------
    print(f"\nH1, rung {RUNG}: measured degradation against bf16, per budget level")
    print(f"  {'task':10} {'arm':11} {'1x':>8} {'2x':>8} {'4x':>8}  {'1x->4x':>9}")
    for task in TASKS:
        items = [k for k in common if k[1] == task]
        for arm in ARMS:
            if arm == "bf16":
                continue
            row, paired = {}, {}
            for lv in LEVELS:
                pairs = []
                for k in items:
                    ref = records.get((lv, f"{k[0].split('|')[0]}|bf16", *k[1:]))
                    alt = records.get((lv, f"{k[0].split('|')[0]}|{arm}", *k[1:]))
                    if ref is not None and alt is not None:
                        pairs.append((correct(ref, RUNG), correct(alt, RUNG)))
                if not pairs:
                    continue
                # Degradation in accuracy points: bf16 minus the quantized arm.
                row[lv] = 100 * statistics.mean(a - b for a, b in pairs)
                paired[lv] = pairs
            if len(row) < 2:
                continue
            delta = row.get("4x", 0) - row.get("1x", 0)
            report["degradation"][f"{task}|{arm}"] = row
            print(f"  {task:10} {arm:11} " +
                  " ".join(f"{row.get(lv, float('nan')):>7.2f}" for lv in LEVELS) +
                  f"  {delta:>+8.2f}")

            # The registered test: is the 1x-to-4x change in degradation non-zero?
            if "1x" in paired and "4x" in paired:
                effects.append(shrinkage_effect(
                    [int(a) - int(b) for a, b in paired["1x"]],
                    [int(a) - int(b) for a, b in paired["4x"]],
                    f"H1_degradation_shrinkage[{task}|{arm}]"))

    report["effects"] = [e.as_dict() for e in effects]
    # holm_bonferroni returns a LIST of Effect objects, not a dict; serialise them.
    report["holm"] = [r.as_dict() if isinstance(r, Effect) else r
                      for r in holm_bonferroni(effects, family="H1")]

    out = REPO / "artifacts" / "analysis" / "analysis_h1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # `default=` rather than hunting nested Effects: holm_bonferroni returns them inside
    # its rows and the shape is not ours to depend on.
    out.write_text(json.dumps(report, indent=2, sort_keys=True,
                              default=lambda o: o.as_dict() if hasattr(o, "as_dict")
                              else str(o)))

    print("\nscored vs full generation length (§5b.1: scored is primary)")
    print(f"  {'cell':28} {'scored':>8} {'full':>8} {'discarded':>10} {'hit cap':>8}")
    for tag, v in sorted(report["length"].items()):
        disc = v["full_mean"] - v["scored_mean"]
        print(f"  {tag:28} {v['scored_mean']:>8.1f} {v['full_mean']:>8.1f} "
              f"{disc:>10.1f} {v['hit_cap_rate']:>7.1%}")

    print(f"\nreport -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
