#!/usr/bin/env python3
"""Apply prereg §3b-bis's DECISION RULE to the template probe. CPU only.

    .venv/bin/python scripts/score_template.py

The rule was registered before the probe ran (planner ruling R19.2), and it is the whole
reason this is a test rather than a defence:

  1. the interaction holds in **sign and rough magnitude** under both template levels
     -> `chat` primary, `raw` robustness, and the self-infliction objection is answered
     **empirically** rather than argued;
  2. they **diverge** -> the divergence **is the finding**: the prompt template is a
     **moderator** of the measured quantization effect. Stronger, on-thesis, and reported
     as the primary framing rather than buried.

Both branches publishable, neither preferred, and §3b-bis pre-commits us to **not**
relabelling branch 2 as a limitation.

⚠️ **What "the interaction" means here.** The paper's headline is `strict` vs `permissive`
(rungs 1 vs 2, §2.2). The quantity that must be template-invariant is therefore not an
accuracy LEVEL but the **quantization delta**, and specifically how that delta differs
between rungs:

    delta(arm, rung)   = acc(arm, rung) - acc(bf16, rung)
    interaction(arm)   = delta(arm, permissive) - delta(arm, strict)

`interaction` is the scorer-dependence of the measured quantization effect -- the paper's
subject. The test is whether it survives the template change.
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
sys.path.insert(0, str(REPO / "scripts"))

from quantartifact.scoring import extract  # noqa: E402
from score_pilot import reduce_gold  # noqa: E402

RUN = "template-v1"
RUNGS = ("strict", "permissive")
ARMS = ("bf16", "gptq-4bit", "awq-4bit")
LEVELS = ("chat", "raw")
# "Rough magnitude" needs a number registered rather than eyeballed. The interaction is a
# difference of differences on ~1,000 paired items, so its SE is a few points; 5 pp is a
# deliberately generous band -- wide enough that ordinary sampling noise cannot trip it,
# narrow enough that a real moderation shows through.
MAGNITUDE_TOLERANCE_PP = 5.0


def score_run(root: Path) -> dict:
    """(level, model, arm, task) -> list of per-rung correctness."""
    rows: dict[tuple, list[dict]] = defaultdict(list)
    for path in sorted(root.rglob("*.jsonl.gz")):
        cell = path.parent.name
        model, arm, level = cell.split("__")
        if arm not in ARMS or level not in LEVELS:
            continue
        with gzip.open(path, "rt") as handle:
            for line in handle:
                record = json.loads(line)
                task = record["task"]
                scored = extract.score_generation(
                    record["generation_text"],
                    reduce_gold(record.get("gold") or "", task),
                    task, record.get("subtask"),
                    rungs=RUNGS, math_verify_available=False,
                )
                rows[(level, model, arm)].append(
                    {rung: bool(scored[rung]["correct"]) for rung in RUNGS})
    return rows


def main() -> int:
    root = REPO / "artifacts" / "generations" / RUN
    if not root.is_dir():
        raise SystemExit(f"no generations at {root} — sync from S3 first")

    rows = score_run(root)
    acc = {key: {rung: statistics.mean(r[rung] for r in group) for rung in RUNGS}
           for key, group in rows.items()}
    models = sorted({m for _, m, _ in acc})

    print(f"{'':28} {'STRICT':>17}   {'PERMISSIVE':>17}")
    print(f"{'model / arm':28} {'chat':>7} {'raw':>8}   {'chat':>7} {'raw':>8}   "
          f"{'n':>5}")
    print("-" * 78)
    for model in models:
        for arm in ARMS:
            cells = {lv: acc.get((lv, model, arm)) for lv in LEVELS}
            if not all(cells.values()):
                continue
            n = len(rows[("chat", model, arm)])
            print(f"  {model+'/'+arm:26} "
                  f"{cells['chat']['strict']:>6.1%} {cells['raw']['strict']:>7.1%}   "
                  f"{cells['chat']['permissive']:>6.1%} "
                  f"{cells['raw']['permissive']:>7.1%}   {n:>5}")
    print()

    print("=== THE DECISION RULE: is the INTERACTION template-invariant? ===")
    print("interaction(arm) = [acc(arm,perm) - acc(bf16,perm)] - "
          "[acc(arm,strict) - acc(bf16,strict)]")
    print()
    print(f"{'model / arm':28} {'chat':>9} {'raw':>9} {'|shift|':>9}   verdict")
    print("-" * 78)
    verdicts = []
    for model in models:
        for arm in ARMS:
            if arm == "bf16":
                continue
            try:
                inter = {}
                for lv in LEVELS:
                    d_perm = acc[(lv, model, arm)]["permissive"] - \
                        acc[(lv, model, "bf16")]["permissive"]
                    d_strict = acc[(lv, model, arm)]["strict"] - \
                        acc[(lv, model, "bf16")]["strict"]
                    inter[lv] = d_perm - d_strict
            except KeyError:
                continue
            shift = abs(inter["chat"] - inter["raw"]) * 100
            same_sign = (inter["chat"] >= 0) == (inter["raw"] >= 0)
            ok = same_sign and shift <= MAGNITUDE_TOLERANCE_PP
            verdicts.append(ok)
            note = "INVARIANT" if ok else (
                "SIGN FLIP" if not same_sign else f"shift > {MAGNITUDE_TOLERANCE_PP} pp")
            print(f"  {model+'/'+arm:26} {inter['chat']:>+8.1%} {inter['raw']:>+8.1%} "
                  f"{shift:>8.1f}   {note}")

    print()
    n_ok = sum(verdicts)
    print(f"invariant in {n_ok}/{len(verdicts)} (model x arm) comparisons "
          f"at a {MAGNITUDE_TOLERANCE_PP} pp tolerance")
    if n_ok == len(verdicts):
        print("=> BRANCH 1: the interaction is template-invariant. `chat` stays primary,")
        print("   `raw` is reported as robustness, and the self-infliction objection is")
        print("   ANSWERED EMPIRICALLY rather than argued.")
    else:
        print("=> BRANCH 2: the interaction is NOT template-invariant. Per §3b-bis the")
        print("   divergence IS THE FINDING -- the template MODERATES the measured")
        print("   quantization effect -- and is reported as the primary framing, NOT as a")
        print("   limitation. We pre-committed to this.")

    out = REPO / "artifacts" / "scores" / f"scores_{RUN}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"run_id": RUN, "tolerance_pp": MAGNITUDE_TOLERANCE_PP,
         "accuracy": {f"{lv}|{m}|{a}": v for (lv, m, a), v in acc.items()},
         "n_invariant": n_ok, "n_comparisons": len(verdicts)}, indent=2, sort_keys=True))
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
