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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from sm_anchor import ANCHOR_MODEL as ANCHOR_MODEL_KEY  # noqa: E402
from sm_anchor import ANCHOR_TASKS  # noqa: E402

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
    n_boot: int = 10_000, seed: int = 20260902, gate: bool = False,
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
        # 🚨 On the GATE metric this is a RED FLAG, not a pass (R31 Q2). Zero variance here requires
        # the two CHECKPOINTS to agree on every single item, which more plausibly means we compared
        # a checkpoint to itself, or resolved both sides to the same S3 path, than that two
        # independently quantized models agree perfectly. The caller passes `gate=True` for the
        # blocking metric so the two cases are not conflated: on a diagnostic metric the bound is
        # legitimate, on the gate it is evidence of a wiring error.
        if gate:
            raise SystemExit(
                f"GATE METRIC HAS ZERO VARIANCE over {len(shared)} items: the two sides agree on "
                f"every item. That is far more likely a wiring error (both sides resolved to the "
                f"same checkpoint path) than two independently quantized models agreeing "
                f"perfectly. Refusing to emit a verdict; check the cell_ids and model_path of "
                f"both sides in anchor_report.json."
            )
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
    """`(verdict, why)` for the blocking metric. SYMMETRIC on the point estimate (R31).

    🚨 **Rewritten 2026-09-02 (R31), and the reason is arithmetic rather than taste.** The first
    version required the whole CI inside the band for a PASS, which the planner showed makes the
    gate unusable: at n=1319 with 10-20% discordance the paired SE is 0.9-1.2 pp, so the CI is about
    3.6-4.8 pp wide against a band that is 4.0 pp wide. Requiring containment means PASS needs
    `|point| + half-width <= 2.0`, i.e. `|point|` under roughly **0.2 pp**, a near-exact match. So
    almost every outcome would have landed in INCONCLUSIVE, and *"a gate that always defers to a
    human is not a gate."* The threshold and the achievable resolution were never matched, and the
    CODE REVIEWER's asymmetry example is what exposed it.

    **The rule:**

    | verdict | condition | blocks `main-v1`? |
    |---|---|---|
    | **PASS** | `|point| <= 2.0 pp` | no |
    | **FAIL** | `|point| > 2.0 pp` | **yes**, debug the checkpoints |
    | **INCONCLUSIVE** | the CI cannot distinguish `|diff| <= 2.0` from `|diff| >= 2 x 2.0`, i.e. the run says nothing about *gross* error | **yes**, the measurement is too weak to decide |

    ⚠️ **INCONCLUSIVE BLOCKS, exactly as FAIL does**, and that is what makes symmetry free: being
    honest about the evidence costs nothing in safety because both non-pass verdicts stop the run.
    They differ only in what to do next. R30 said *"investigate"* where it should have said
    *"blocks"*, and the planner recorded that gap as its own drafting error.

    **On false-pass safety:** a broken calibration produces a *large* gap, not a 2.1 pp gap. Gating
    the point estimate at ±2.0 pp catches gross error reliably, which is what Anchor A is for.
    Certifying near-equality to 2 pp was never achievable at this n, and pretending otherwise is the
    thing H7 criticises in other people.
    """
    point = result["difference_pp"]
    low, high = result["ci_low"], result["ci_high"]
    half_width = (high - low) / 2.0
    gross = 2.0 * CRITERION_PP

    # ⚠️ INCONCLUSIVE fires on the RESOLUTION, not on where the interval happens to fall, and the
    # encoding is pinned by R31's own worked examples rather than by my reading of its prose.
    #
    # R31 says INCONCLUSIVE is for "when the CI cannot distinguish |diff| <= 2.0 from |diff| >= 4.0".
    # My first encoding took that as "the interval overlaps the band AND reaches beyond twice it",
    # which fired on R31's own FAIL example (-2.1, CI [-4.0, +0.5]) and would have made INCONCLUSIVE
    # common. Since R31 explicitly resolves that case as FAIL and says INCONCLUSIVE is "rare, and it
    # means something when it fires", the operative condition must be about the MEASUREMENT's power:
    # if the half-width alone reaches the gross threshold, the run cannot see gross error wherever
    # the point sits.
    #
    # Checked against every example R31 gives: (0.5, [-1.8, 2.8]) -> PASS, (-1.9, [-3.5, -0.3]) ->
    # PASS, (-2.1, [-4.0, +0.5]) -> FAIL, (3.0, [1.2, 4.8]) -> FAIL, and a genuinely useless
    # (0.5, [-4.5, 5.5]) -> INCONCLUSIVE.
    if half_width >= gross:
        return "INCONCLUSIVE", (
            f"resolution is +/-{half_width:.2f} pp, which reaches the gross-error threshold of "
            f"+/-{gross} pp, so this run cannot distinguish fine agreement from gross error "
            f"wherever the point estimate sits. CI [{low:+.2f}, {high:+.2f}]. BLOCKS, and the "
            f"action is a stronger measurement rather than debugging the checkpoints")

    if abs(point) <= CRITERION_PP:
        return "PASS", (
            f"|{point:+.2f}| <= {CRITERION_PP} pp. Resolution +/-{half_width:.2f} pp, CI "
            f"[{low:+.2f}, {high:+.2f}], so we can rule out a gap larger than "
            f"{max(abs(low), abs(high)):.2f} pp. Anchor A tests comparable QUALITY, not "
            f"reproduction, and certifying equality to {CRITERION_PP} pp is not achievable at this n")
    return "FAIL", (
        f"|{point:+.2f}| > {CRITERION_PP} pp, CI [{low:+.2f}, {high:+.2f}]. A9: a larger gap means "
        f"our calibration or serving config is wrong. BLOCKS, debug before proceeding")


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

    # ⚠️ (MODEL, task, rung, arm, side) -> item -> correct. The model is IN the key even though
    # this stage is single-model, because `item_id` is doc-scoped and keying on it alone is exactly
    # what produced a CRITICAL in `interaction._collect`, where three models and six BBH subtasks
    # shared an id namespace. Here the precondition holds today (one model, no subtasks, one budget
    # level, one prompt variant) but it is a set of facts about the stage rather than a property of
    # the code, and one tuple element removes the precondition entirely.
    scored: dict[tuple, dict[str, bool]] = defaultdict(dict)
    # 🚨 R32.4(1): budget truncation PER SIDE, on the gate task. Truncation is SERVED behaviour,
    # so the two sides' different kernel paths (R32.1) can move it -- and unlike a symmetric rate
    # it would then enter the gate metric DIRECTLY rather than cancelling in the paired difference.
    # The smoke measured 6 of 16 per side, symmetric, but n=8 establishes nothing about n=1319.
    truncated: dict[tuple, int] = defaultdict(int)
    n_records: dict[tuple, int] = defaultdict(int)
    arms: set[str] = set()
    for record in records:
        cell_id = record["cell_id"]
        parts = cell_id.split("|")
        if len(parts) != 3:
            raise SystemExit(
                f"cell_id {cell_id!r} is not '<model>|<arm>|<side>'. Refusing to guess which side "
                f"a record came from: mislabelling one would invert the comparison."
            )
        model_key, arm, side = parts
        if side not in ("ours", "official"):
            raise SystemExit(f"cell_id {cell_id!r}: side must be 'ours' or 'official'")
        arms.add(arm)
        task = record["task"]
        gold = reduce_gold(record.get("gold") or "", task)
        result = extract.score_generation(
            extract.scored_text_for(record), gold, task, record.get("subtask"),
            rungs=RUNGS, math_verify_available=MATH_VERIFY)
        for rung in RUNGS:
            scored[(model_key, task, rung, arm, side)][record["item_id"]] = bool(
                result[rung]["correct"])
        # `finish_reason == "length"` is the budget cap, distinct from a stop sequence firing.
        # Keyed by (task, arm, side) so the gate task can be read out on its own.
        n_records[(task, arm, side)] += 1
        if record.get("finish_reason") == "length":
            truncated[(task, arm, side)] += 1

    # 🚨 The cross-side engine check, which is what makes the "any pipeline error is common to
    # both and cancels" claim checked rather than asserted. Compares what the two sides' engines
    # actually REPORTED, not what we asked for.
    engine_mismatches: list[str] = []
    # 🚨 Scoped by run_id, not a single fixed filename. A shared path would let the smoke's
    # report be read as the real gate's, which is the isolation every other stage gets from its
    # own `run_id` and which `anchors.csv` was missing until R32 exposed it. It also means the
    # rehearsal's engine-state check actually RUNS, which is how the kernel-path confound was
    # found at all.
    job_report = REPO / "artifacts" / f"anchor_report_{args.run_id}.json"
    if job_report.is_file():
        cells = {c["cell_id"]: c for c in json.loads(job_report.read_text()).get("cells", [])}
        for arm in sorted({c["arm"] for c in cells.values() if "arm" in c}):
            sides = {}
            for side in ("ours", "official"):
                cell = cells.get(f"{ANCHOR_MODEL_KEY}|{arm}|{side}")
                state = (cell or {}).get("engine_state") or {}
                sides[side] = {k: state.get(k) for k in
                               ("num_gpu_blocks", "quantization", "max_num_seqs",
                                "enable_prefix_caching")}
            if sides["ours"] != sides["official"] and all(
                    any(v is not None for v in s.values()) for s in sides.values()):
                engine_mismatches.append(
                    f"{arm}: engine state differs between sides. ours={sides['ours']} "
                    f"official={sides['official']}")
    else:
        engine_mismatches.append(
            f"{job_report.name} not found, so the two sides' engine states were NOT compared. "
            f"So the two sides' served configurations were NOT compared for this run, and "
            f"R32.1's kernel-path finding could not be checked here.")

    # Derived from the run_id rather than passed as a flag, so a caller cannot label a smoke as
    # a real gate by forgetting an argument. `sm_anchor.py` sets RUN_ID = "anchor-v1-smoke" when
    # SMOKE, so the suffix IS the provenance.
    is_smoke = args.run_id.endswith("-smoke")

    report: dict = {
        "gate": "G5a", "anchor": "A", "run_id": args.run_id, "is_smoke": is_smoke,
        "engine_state_mismatches": engine_mismatches,
        "criterion_metric": list(CRITERION_METRIC), "criterion_pp": CRITERION_PP,
        "what_this_tests": (
            "comparable end-to-end QUALITY, not reproduction and NOT the weights in isolation. "
            "The official checkpoints were calibrated on Qwen's data rather than our "
            "ultrachat_200k 512x2048, AND vLLM dispatches each checkpoint through the kernel its "
            "own config selects, so the two sides take different kernel paths (R32.1). H8 puts "
            "the served-path contribution at 0.2-1.6 pp, so the +/-2.0 pp band is largely "
            "consumed by it and a PASS means only 'no gross calibration error at this "
            "resolution' (R31)."),
        "blocking": {}, "diagnostic": {},
    }

    gate_task, gate_rung = CRITERION_METRIC
    models = {key[0] for key in scored}
    if len(models) != 1:
        raise SystemExit(
            f"expected exactly one anchor model, found {sorted(models)}. Anchor A is defined on "
            f"the one model with official quantized checkpoints; more than one means the pairing "
            f"spans models and must be keyed per model before a verdict is computed."
        )
    model = next(iter(models))

    # 🚨 MODERATE 3: the registered item count is part of the CRITERION, not an implementation
    # detail. R30 chose 1319 over 500 because at 500 the +/-2.0 pp band is about one paired SE and
    # a perfect pipeline would fail this blocking gate roughly one time in six.
    #
    # The direction of a short run is SAFE: smaller n widens the interval, which makes `straddles`
    # more likely, which yields INCONCLUSIVE. So a truncated cell cannot produce a spurious PASS.
    # But it can go unmet with no signal, and an INCONCLUSIVE caused by a short cell reads
    # identically to one caused by a real discrepancy -- sending us to debug the checkpoints when
    # the answer is "re-run the cell".
    registered_n = {task: n for _, task, n in ANCHOR_TASKS}[gate_task]
    verdicts: list[str] = []
    print(f"Anchor A, gate G5a. Blocking metric: {gate_task} {gate_rung}, "
          f"+/-{CRITERION_PP} pp\n")
    print(f"registered n for the gate task: {registered_n}")
    print(f"verdict rule (R31): PASS |diff| <= {CRITERION_PP} pp; FAIL otherwise; INCONCLUSIVE "
          f"only when the CI cannot separate <= {CRITERION_PP} from >= {2 * CRITERION_PP} pp.")
    print("⚠️ BOTH non-pass verdicts BLOCK main-v1; they differ in what to do next.\n")
    print(f"{'arm':11} {'ours':>8} {'official':>9} {'diff':>8} "
          f"{'95% CI':>18} {'n':>6} {'disc':>5}  verdict")
    print("-" * 92)
    for arm in sorted(arms):
        ours = scored.get((model, gate_task, gate_rung, arm, "ours"))
        official = scored.get((model, gate_task, gate_rung, arm, "official"))
        if not ours or not official:
            report["blocking"][arm] = {"verdict": "NOT COMPUTED",
                                       "why": "one side produced no records"}
            verdicts.append("NOT COMPUTED")
            print(f"{arm:11} {'--':>8} {'--':>9}  NOT COMPUTED (one side missing)")
            continue
        result = paired_difference(ours, official, gate=True)
        verdict, why = verdict_for(result)
        if result["n"] < registered_n:
            verdict = "INCONCLUSIVE"
            why = (f"only {result['n']} of the registered {registered_n} items were scored, a "
                   f"shortfall of {registered_n - result['n']}. R30 chose {registered_n} because "
                   f"the band is about one paired SE at 500, so a short cell does not meet the "
                   f"registered resolution. RE-RUN THE CELL rather than debugging the checkpoints")
        verdicts.append(verdict)
        report["blocking"][arm] = {**result, "verdict": verdict, "why": why}
        short = "!" if result["n"] < registered_n else " "
        print(f"{arm:11} {result['accuracy_ours_pp']:>7.2f}% "
              f"{result['accuracy_official_pp']:>8.2f}% {result['difference_pp']:>+7.2f} "
              f"[{result['ci_low']:>+6.2f}, {result['ci_high']:>+6.2f}] "
              f"{result['n']:>6}{short}{result['n_discordant']:>4}  {verdict}")
        # ⚠️ Which OBJECT the verdict was evaluated against. R26.1 withdrew the H6 overlap framing
        # because "a worst-case bound and a bootstrap percentile interval are different objects and
        # must not be read against each other", and the same `straddles` test is applied to both
        # here, so the reader is told which one produced this row.
        print(f"{'':11} interval: {result['interval_method']}")

    for task, rung in DIAGNOSTIC_METRICS:
        for arm in sorted(arms):
            ours = scored.get((model, task, rung, arm, "ours"))
            official = scored.get((model, task, rung, arm, "official"))
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

    # ---- R32.4(1): truncation comparability on the gate task -----------------------------
    gate_task_only = CRITERION_METRIC[0]
    report["truncation"] = {}
    for arm in sorted(arms):
        counts, rates = {}, {}
        for side in ("ours", "official"):
            n = n_records.get((gate_task_only, arm, side), 0)
            k = truncated.get((gate_task_only, arm, side), 0)
            counts[side] = {"n": n, "truncated": k}
            rates[side] = (100.0 * k / n) if n else None
        entry = {"task": gate_task_only, "counts": counts,
                 "rate_ours_pct": rates["ours"], "rate_official_pct": rates["official"]}
        if rates["ours"] is not None and rates["official"] is not None:
            gap = rates["ours"] - rates["official"]
            entry["difference_pp"] = gap
            # 5 pp is a REPORTING trigger, not a gate: nothing about the verdict changes, but a
            # divergence this size means the two kernel paths produce materially different
            # generation lengths, which enters `permissive` directly since it scans the trace.
            entry["comparable"] = abs(gap) <= 5.0
        else:
            entry["difference_pp"] = None
            entry["comparable"] = None
        report["truncation"][arm] = entry

    print()
    print(f"BUDGET TRUNCATION on the gate task ({gate_task_only}), per side (R32.4):")
    for arm, e in sorted(report["truncation"].items()):
        if e["difference_pp"] is None:
            print(f"  {arm:11} not computable: "
                  f"ours n={e['counts']['ours']['n']}, official n={e['counts']['official']['n']}")
            continue
        flag = "" if e["comparable"] else "  <-- DIVERGENT, the verdict carries this caveat"
        print(f"  {arm:11} ours {e['rate_ours_pct']:5.1f}%  official "
              f"{e['rate_official_pct']:5.1f}%  diff {e['difference_pp']:+5.1f} pp{flag}")
    print("  Truncation is served behaviour, so different kernel paths can move it. A symmetric")
    print("  rate cancels in the paired difference; a divergent one enters permissive directly.")

    overall = ("FAIL" if "FAIL" in verdicts
               else "NOT COMPUTED" if "NOT COMPUTED" in verdicts
               else "INCONCLUSIVE" if "INCONCLUSIVE" in verdicts
               else "PASS")
    report["overall_verdict"] = overall
    # A real check, not an `assert`: an assert vanishes under `python -O`, and this is the field a
    # reader trusts. Per-arm verdicts are checked too, which the assert never did.
    for label, value in [("overall", overall)] + [
            (arm, row["verdict"]) for arm, row in report["blocking"].items()]:
        if value not in VERDICTS:
            raise SystemExit(f"{label}: verdict {value!r} is not one of {VERDICTS}")

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    # 🚨 `run_id` and `is_smoke` are the FIRST columns, and they are here because the smoke
    # exposed the gap. `anchors.csv` is A9's NAMED evidence artifact for gate G5a, and the
    # rehearsal wrote an 8-item INCONCLUSIVE row into it carrying no indication of which run it
    # came from. The `n` column hints at it, but a reader of the gate's evidence file should not
    # have to infer provenance from a sample size. Same family as the project's standing rule
    # that a smoke result must never be readable as a real one, which every other stage honours
    # by writing to a separate `run_id` -- this file was the one place that isolation did not
    # reach, because both runs write to the same path.
    (out_dir / "anchors.csv").write_text(
        "run_id,is_smoke,anchor,arm,metric,ours_pp,official_pp,difference_pp,ci_low,ci_high,"
        "n,n_discordant,verdict\n"
        + "".join(
            f"{args.run_id},{str(is_smoke).lower()},"
            f"A,{arm},{gate_task}|{gate_rung},{r.get('accuracy_ours_pp', '')},"
            f"{r.get('accuracy_official_pp', '')},{r.get('difference_pp', '')},"
            f"{r.get('ci_low', '')},{r.get('ci_high', '')},{r.get('n', '')},"
            f"{r.get('n_discordant', '')},{r['verdict']}\n"
            for arm, r in sorted(report["blocking"].items())))
    (out_dir / "anchor_a.json").write_text(json.dumps(report, indent=2))

    if engine_mismatches:
        print()
        print("⚠️ ENGINE STATE, cross-side:")
        for message in engine_mismatches:
            print(f"  - {message}")
        print("   Anchor A is an END-TO-END comparability check and never isolated the weights")
        print("   (R32.1): vLLM dispatches each checkpoint through the kernel its own config")
        print("   selects. A difference here is expected and is a finding to REPORT, not a")
        print("   reason to discard the run. H8 puts the served-path contribution at 0.2-1.6 pp.")

    print()
    print(f"GATE G5a: {overall}")
    for arm, r in sorted(report["blocking"].items()):
        print(f"  {arm:11} {r['verdict']:13} {r['why']}")
    print(f"\nreport -> {(out_dir / 'anchors.csv').relative_to(REPO)}, "
          f"{(out_dir / 'anchor_a.json').relative_to(REPO)}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
