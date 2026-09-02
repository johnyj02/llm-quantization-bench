#!/usr/bin/env python3
"""Anchor A's verdict: gate G5a. CPU only, over stored generations.

    .venv/bin/python scripts/score_anchor.py --run-id anchor-v1

Computed here rather than inside the job, because a gate decision inside a billable process cannot
be re-run without a queue slot, and because the verdict must be recomputable if the criterion
metric ever changes.

## The criterion, per planner ruling R30

**Two-tier, and the tiers are deliberately not symmetric.**

* **`gsm8k_cot` permissive within ±2.0 pp BLOCKS.** Anchor A exists to detect a calibration or
  serving error, and those damage **competence**. Rung 1 conflates competence with marker
  compliance, so gating on it would let a real compliance effect spuriously fail a blocking gate.
* **A strict discrepancy is a documented diagnostic requiring explanation, not a block.** A large
  strict gap alongside a small permissive one points at chat-template or BOS handling rather than
  calibration, which is informative rather than disqualifying.

🚨 **The verdict is evaluated against the INTERVAL, not the point estimate alone** (R30). The gate
passes on the point difference, but a paired bootstrap CI that **straddles ±2.0 pp** is recorded as
`INCONCLUSIVE, investigate` rather than as a clean pass. The reasoning is the one the paper itself
levels at other people: making a blocking decision on a number whose resolution has not been stated
is exactly the criticism H7 makes of the literature, so we state ours.

⚠️ **What this tests.** The official checkpoints were calibrated on **Qwen's** data, not our
`ultrachat_200k` 512×2048, so this is **not** a reproduction test. It tests that our pipeline
produces a checkpoint of *comparable quality*, and ±2.0 pp is a quality-comparability band.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from quantartifact.scoring import extract  # noqa: E402
from score_pilot import MATH_VERIFY, RUNGS, read_records, reduce_gold  # noqa: E402

CRITERION_METRIC = ("gsm8k_cot", "permissive")
CRITERION_PP = 2.0
DIAGNOSTIC_METRICS = (("gsm8k_cot", "strict"), ("gsm8k", "permissive"), ("gsm8k", "strict"))

#: A closed set, for the same reason `holm_block`'s status is one: the verdict is the field a reader
#: trusts, so a caller must not be able to write anything into it.
VERDICTS = ("PASS", "FAIL", "INCONCLUSIVE", "NOT COMPUTED")


def paired_difference(
    ours: dict[str, bool], official: dict[str, bool], *,
    n_boot: int = 10_000, seed: int = 20260902,
) -> dict:
    """`ours - official` in percentage points, with a paired bootstrap CI over items.

    The comparison is **paired**: the same items are scored under both checkpoints, so the standard
    error is driven by item-level **discordance** rather than by the marginal accuracies. Resampling
    items keeps each item's pair of outcomes together, which is what makes the interval reflect that.
    """
    import numpy as np  # noqa: PLC0415

    shared = sorted(set(ours) & set(official))
    if not shared:
        raise SystemExit(
            "no shared items between our checkpoint and the official one, so the paired "
            "difference is undefined rather than zero"
        )
    contrib = np.array([int(ours[i]) - int(official[i]) for i in shared], dtype=float)
    point = float(contrib.mean()) * 100.0

    # Discordant pairs are the only items that carry information about the difference. Reported
    # because the interval's width is a function of them, so a reader can check the resolution.
    n_discordant = int((contrib != 0).sum())

    if contrib.min() == contrib.max():
        # Same degeneracy R25.1 registered: nothing to resample. Wilson upper on 0 of n, times the
        # per-item magnitude of 1, since this is a difference of two per-item indicators.
        from quantartifact.analysis.results import zero_variance_bound  # noqa: PLC0415
        ci_low, ci_high, method = zero_variance_bound(
            len(shared), max_abs_contribution=1, centre=point)
    else:
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(shared), size=(n_boot, len(shared)))
        boot = contrib[draws].mean(axis=1) * 100.0
        ci_low, ci_high = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
        method = f"paired bootstrap over items, {n_boot} percentile draws"

    return {
        "difference_pp": point,
        "ci_low": ci_low, "ci_high": ci_high, "interval_method": method,
        "n": len(shared), "n_discordant": n_discordant,
        "accuracy_ours_pp": sum(ours[i] for i in shared) / len(shared) * 100.0,
        "accuracy_official_pp": sum(official[i] for i in shared) / len(shared) * 100.0,
        "n_unpaired": len(set(ours) ^ set(official)),
    }


def verdict_for(result: dict) -> tuple[str, str]:
    """`(verdict, why)` for the blocking metric, per R30's interval rule."""
    inside = abs(result["difference_pp"]) <= CRITERION_PP
    # Does the interval straddle the band edge? If either limit lies outside while the point is
    # inside, the data does not resolve the question at the stated tolerance.
    straddles = (abs(result["ci_low"]) > CRITERION_PP) or (abs(result["ci_high"]) > CRITERION_PP)
    if inside and not straddles:
        return "PASS", (f"|{result['difference_pp']:+.2f}| <= {CRITERION_PP} pp and the interval "
                        f"[{result['ci_low']:+.2f}, {result['ci_high']:+.2f}] lies inside the band")
    if inside and straddles:
        return "INCONCLUSIVE", (
            f"point {result['difference_pp']:+.2f} pp is inside the band but the interval "
            f"[{result['ci_low']:+.2f}, {result['ci_high']:+.2f}] straddles +/-{CRITERION_PP}, so "
            f"the data does not resolve the question at this tolerance. R30: record as "
            f"INCONCLUSIVE and investigate, do NOT report a clean pass")
    return "FAIL", (f"|{result['difference_pp']:+.2f}| > {CRITERION_PP} pp. A9: a larger gap means "
                    f"our calibration or serving config is wrong. Debug before proceeding")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Anchor A's verdict (gate G5a)")
    parser.add_argument("--run-id", default="anchor-v1")
    parser.add_argument("--out", default="artifacts/tables")
    args = parser.parse_args()

    root = REPO / "artifacts" / "generations" / args.run_id
    if not root.is_dir():
        raise SystemExit(f"no generations at {root}; sync them from S3 first")
    records = read_records(root)
    if not records:
        raise SystemExit(f"no records under {root}")

    # (task, rung, arm, side) -> item -> correct
    scored: dict[tuple, dict[str, bool]] = defaultdict(dict)
    arms: set[str] = set()
    for record in records:
        cell_id = record["cell_id"]
        parts = cell_id.split("|")
        if len(parts) != 3:
            raise SystemExit(
                f"cell_id {cell_id!r} is not '<model>|<arm>|<side>'. Refusing to guess which side "
                f"a record came from: mislabelling one would invert the comparison."
            )
        _, arm, side = parts
        if side not in ("ours", "official"):
            raise SystemExit(f"cell_id {cell_id!r}: side must be 'ours' or 'official'")
        arms.add(arm)
        task = record["task"]
        gold = reduce_gold(record.get("gold") or "", task)
        result = extract.score_generation(
            record["generation_text"], gold, task, record.get("subtask"),
            rungs=RUNGS, math_verify_available=MATH_VERIFY)
        for rung in RUNGS:
            scored[(task, rung, arm, side)][record["item_id"]] = bool(result[rung]["correct"])

    report: dict = {
        "gate": "G5a", "anchor": "A", "run_id": args.run_id,
        "criterion_metric": list(CRITERION_METRIC), "criterion_pp": CRITERION_PP,
        "what_this_tests": (
            "comparable QUALITY, not reproduction: the official checkpoints were calibrated on "
            "Qwen's data, not our ultrachat_200k 512x2048"),
        "blocking": {}, "diagnostic": {},
    }

    gate_task, gate_rung = CRITERION_METRIC
    verdicts: list[str] = []
    print(f"Anchor A, gate G5a. Blocking metric: {gate_task} {gate_rung}, "
          f"+/-{CRITERION_PP} pp\n")
    print(f"{'arm':11} {'ours':>8} {'official':>9} {'diff':>8} "
          f"{'95% CI':>18} {'disc':>5}  verdict")
    print("-" * 84)
    for arm in sorted(arms):
        ours = scored.get((gate_task, gate_rung, arm, "ours"))
        official = scored.get((gate_task, gate_rung, arm, "official"))
        if not ours or not official:
            report["blocking"][arm] = {"verdict": "NOT COMPUTED",
                                       "why": "one side produced no records"}
            verdicts.append("NOT COMPUTED")
            print(f"{arm:11} {'--':>8} {'--':>9}  NOT COMPUTED (one side missing)")
            continue
        result = paired_difference(ours, official)
        verdict, why = verdict_for(result)
        verdicts.append(verdict)
        report["blocking"][arm] = {**result, "verdict": verdict, "why": why}
        print(f"{arm:11} {result['accuracy_ours_pp']:>7.2f}% "
              f"{result['accuracy_official_pp']:>8.2f}% {result['difference_pp']:>+7.2f} "
              f"[{result['ci_low']:>+6.2f}, {result['ci_high']:>+6.2f}] "
              f"{result['n_discordant']:>5}  {verdict}")

    for task, rung in DIAGNOSTIC_METRICS:
        for arm in sorted(arms):
            ours = scored.get((task, rung, arm, "ours"))
            official = scored.get((task, rung, arm, "official"))
            if not ours or not official:
                continue
            result = paired_difference(ours, official)
            result["exceeds_band"] = abs(result["difference_pp"]) > CRITERION_PP
            report["diagnostic"][f"{task}|{rung}|{arm}"] = result

    print()
    print("DIAGNOSTIC metrics (never block; a discrepancy here requires explanation, per R30):")
    for key, result in report["diagnostic"].items():
        flag = "  <-- exceeds the band, EXPLAIN IT" if result["exceeds_band"] else ""
        print(f"  {key:34} {result['difference_pp']:>+7.2f} pp "
              f"[{result['ci_low']:>+6.2f}, {result['ci_high']:>+6.2f}]{flag}")

    overall = ("FAIL" if "FAIL" in verdicts
               else "NOT COMPUTED" if "NOT COMPUTED" in verdicts
               else "INCONCLUSIVE" if "INCONCLUSIVE" in verdicts
               else "PASS")
    report["overall_verdict"] = overall
    assert overall in VERDICTS

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "anchors.csv").write_text(
        "anchor,arm,metric,ours_pp,official_pp,difference_pp,ci_low,ci_high,"
        "n,n_discordant,verdict\n"
        + "".join(
            f"A,{arm},{gate_task}|{gate_rung},{r.get('accuracy_ours_pp', '')},"
            f"{r.get('accuracy_official_pp', '')},{r.get('difference_pp', '')},"
            f"{r.get('ci_low', '')},{r.get('ci_high', '')},{r.get('n', '')},"
            f"{r.get('n_discordant', '')},{r['verdict']}\n"
            for arm, r in sorted(report["blocking"].items())))
    (out_dir / "anchor_a.json").write_text(json.dumps(report, indent=2))

    print()
    print(f"GATE G5a: {overall}")
    for arm, r in sorted(report["blocking"].items()):
        print(f"  {arm:11} {r['verdict']:13} {r['why']}")
    print(f"\nreport -> {(out_dir / 'anchors.csv').relative_to(REPO)}, "
          f"{(out_dir / 'anchor_a.json').relative_to(REPO)}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
