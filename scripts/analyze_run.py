#!/usr/bin/env python3
"""Run the pre-registered analysis over a scored run. CPU only, no GPU, no queue.

    .venv/bin/python scripts/analyze_run.py --run-id pilot-v2
    .venv/bin/python scripts/analyze_run.py --run-id pilot-v2 --task bbh

🚨 WHY THIS EXISTS. `src/quantartifact/analysis/` was built and tested and then **used on
nothing**, which the user asked about directly on 2026-09-02 and was right to. The gap was
concrete rather than a matter of sequencing: `scripts/score_pilot.py` builds per-item rows in
memory and writes only per-cell aggregates, so no artifact on disk was in the shape the analysis
layer consumes. A library with no caller produces no number, and a statistic nobody has computed
is indistinguishable from one that does not work.

This is that caller. It reads stored generations, scores every rung, reshapes into the per-item
rows the analysis layer takes, and emits every registered quantity:

* §1.1 the always-compliant stratum, with both §9g sizes
* §1.2 the compliance-transition matrices with exact McNemar
* §1.3 the Manski bounds
* §2  H3's interaction, both the marginal effect and the registered rung companion
* §4  H4's four components and the `TRUNC_STOPSEQ` differentiality test
* §9  Holm-Bonferroni within each family

⚠️ **TWO HONEST LIMITATIONS, stated because a number that looks finished is worse than a
missing one.**

1. **Compliance is generation-level here, while §1.1 quantifies over rungs.**
   `scoring.compliance.label_compliance` returns ONE label per generation, derived from the
   strict and permissive extractions together, so `RECOVERABLE` already means "permissive found
   what strict missed". §1.1 writes the stratum as a conjunction over `s in rungs 1-4`. On this
   data that conjunction is degenerate: the same label is used for every rung. That is recorded
   in the output as `compliance_is_generation_level: true` and **flagged for the planner**, since
   deciding what per-rung compliance means is a science question rather than an implementation
   detail.
2. **`pilot-v2` predates schema 1.3.0**, so its records carry no `until`, `template_level` or
   `invocation_id`, and no `model_key` (it is derived from `cell_id`). `classify_termination`
   therefore takes its documented `tasks.yaml` fallback for `until`. Fine for a pilot, and the
   reason the reference run should be analysed from 1.3.0 records instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from quantartifact.analysis import (  # noqa: E402
    EXCLUDED_FROM_H3_INTERACTION,
    EXCLUDED_FROM_H4_PRIMARY,
    bounds_effects,
    build_transitions,
    compute_bounds,
    compute_stratum,
    decompose,
    holm_bonferroni,
    interaction_marginal,
    rung_gap_decomposition,
    rung_mcnemar,
    stopseq_differential,
    study_wide_magnitude,
    transition_effects,
)
from quantartifact.scoring import compliance as compliance_module  # noqa: E402
from quantartifact.scoring import extract  # noqa: E402
from score_pilot import MATH_VERIFY, RUNGS, read_records, reduce_gold  # noqa: E402

#: §1.1's three contrasted arms are what the stratum and H4 use; the interaction is reported for
#: every quantized arm against bf16, since §1.2 ranges over all four.
ALT_ARMS = ("gptq-8bit", "gptq-4bit", "awq-4bit", "bnb-nf4")


def to_analysis_row(record: dict) -> dict:
    """Reshape one generation record into the per-item row the analysis layer consumes.

    `cell_id` is `"<model>|<arm>"`, so `model_key` and `arm` are split out of it rather than
    read, because records before schema 1.3.0 do not carry `model_key` separately.
    """
    cell_id = record["cell_id"]
    if "|" not in cell_id:
        raise ValueError(
            f"cell_id {cell_id!r} is not '<model>|<arm>', so model and arm cannot be recovered. "
            f"Refusing to guess: an arm label silently wrong would mis-assign every contrast."
        )
    model_key, arm = cell_id.split("|", 1)

    task, subtask = record["task"], record.get("subtask")
    generation = record["generation_text"]
    gold = reduce_gold(record.get("gold") or "", task)

    scored = extract.score_generation(generation, gold, task, subtask,
                                      rungs=RUNGS, math_verify_available=MATH_VERIFY)
    strict = extract.extract_strict(generation, task, subtask)
    permissive = extract.extract_permissive(generation, task, subtask)
    labels = compliance_module.label_compliance(
        generation, task, subtask,
        finish_reason=record.get("finish_reason"),
        stop_reason=record.get("stop_reason"),
        strict=strict, permissive=permissive,
        until=record.get("until"),
    )
    return {
        "model_key": model_key,
        "arm": arm,
        "task": task,
        "subtask": subtask,
        "item_id": record["item_id"],
        "budget_level": record.get("budget_level", "1x"),
        "prompt_variant": record.get("prompt_variant", "fewshot_cot"),
        "repeat_idx": record.get("repeat_idx", 0),
        # See limitation 1 in the module docstring: one generation-level label, applied to every
        # rung, because that is what the scorer produces.
        "compliance": dict.fromkeys(RUNGS, labels.compliance),
        "correct": {r: bool(scored[r]["correct"]) for r in RUNGS},
        "termination": labels.termination,
    }


def holm_block(effects: list, *, family: str, absent_reason: str) -> dict:
    """A Holm-Bonferroni result that carries its own STATUS, never a bare list.

    🚨 WHY THIS EXISTS, and it is the third appearance of one shape.

    `holm_bonferroni` raises on an empty family, and the raise carries the reason: *"returning an
    empty list would let an upstream skip read as a completed correction."* The first version of
    this script wrote `[...] if effects else []`, which performs exactly the pre-check the guard
    exists to catch, so the guard became unreachable and the skip flowed straight through. The
    review before this one found the same shape with `allow_excluded=`, and before that
    `score_template.py` declared BRANCH 1 from an empty verdict list.

    **A guard is not a guard if the caller may pre-check its condition.**

    The second half is that a bare `[]` collapses three different meanings into one serialised
    value: no registered tests implemented in this family yet, the family resolved to nothing so
    the correction was skipped, and the correction ran and rejected nothing. This project already
    refuses that collapse in three places, each with a comment saying why: `Effect.claimable`
    returns `None` rather than `False`, `StratumResult.fraction` returns `None` rather than `0.0`,
    and `mcnemar_exact` returns `None` rather than `1.0`. Same argument, same fix.

    So the block always carries `family`, `m`, `status` and `rows`, and `status` distinguishes the
    cases. `holm_bonferroni` is called if and only if there is something to correct, and when there
    is not, the reason is recorded rather than implied by an empty list.
    """
    if effects:
        rows = [{k: v for k, v in r.items() if k != "effect"}
                for r in holm_bonferroni(effects, family=family)]
        return {"family": family, "m": rows[0]["m"], "status": "corrected", "rows": rows}
    return {"family": family, "m": 0, "status": absent_reason, "rows": []}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the pre-registered analysis")
    parser.add_argument("--run-id", default="pilot-v2")
    parser.add_argument("--task", default=None, help="restrict to one task")
    parser.add_argument("--out", default="artifacts/analysis")
    args = parser.parse_args()

    root = REPO / "artifacts" / "generations" / args.run_id
    if not root.is_dir():
        raise SystemExit(f"no generations at {root}; sync them from S3 first")

    records = read_records(root)
    if not records:
        raise SystemExit(f"no records under {root}")
    rows = [to_analysis_row(r) for r in records]
    if args.task:
        rows = [r for r in rows if r["task"] == args.task]
        if not rows:
            raise SystemExit(f"no rows for task {args.task!r}")

    # 🚨 §9h's magnitude is STUDY-WIDE, so it is computed once over every row before any cell is
    # estimated, and passed down. R26.1 also requires monotonicity be asserted on the real data
    # rather than only on the G3 fixtures, and that is the same computation.
    magnitude, n_violations, violations = study_wide_magnitude(rows)

    tasks = sorted({r["task"] for r in rows})
    print(f"run_id       : {args.run_id}")
    print(f"records      : {len(records):,}  rows after filter: {len(rows):,}")
    print(f"rungs        : {list(RUNGS)}"
          f"{'' if MATH_VERIFY else '   (math_verify UNAVAILABLE)'}")
    print(f"tasks        : {tasks}")
    print(f"§9h magnitude: {magnitude}  (rung monotonicity violations on real data: "
          f"{n_violations})")
    if n_violations:
        print("   🚨 MONOTONICITY VIOLATED on real data. This is a finding about the scorers, "
              "not a wider interval to absorb quietly. G3's fixtures forbid it.")

    report: dict = {
        "run_id": args.run_id,
        "n_rows": len(rows),
        "rungs": list(RUNGS),
        "math_verify_available": MATH_VERIFY,
        "compliance_is_generation_level": True,
        # §9h / R26.1: the study-wide magnitude, and monotonicity asserted on the real data.
        "study_wide_max_abs_contribution": magnitude,
        "n_rung_monotonicity_violations": n_violations,
        "rung_monotonicity_violations": violations[:50],
        "limitations": [
            "compliance is one generation-level label applied to every rung; §1.1 quantifies "
            "over rungs. Flagged for the planner.",
            f"{args.run_id} predates schema 1.3.0: no until/template_level/invocation_id, so "
            "classify_termination uses the tasks.yaml fallback for until.",
        ],
    }

    # ---- §1.1 the stratum -------------------------------------------------------------
    strata = compute_stratum(rows, rungs=RUNGS)
    report["stratum"] = [s.as_dict() for s in strata.values()]
    print(f"\n§1.1 STRATUM  : {len(strata)} cells")
    print(f"{'cell':52} {'size':>5} {'of paired':>10} {'unpaired':>9} {'K':>3}")
    for cell, s in sorted(strata.items(), key=lambda kv: str(kv[0])):
        label = f"{cell[0]}|{cell[1]}" + (f"/{cell[2]}" if cell[2] else "")
        frac = s.fraction_of_paired
        k = max(s.invocations_by_arm.values()) if s.invocations_by_arm else 0
        print(f"  {label:50} {s.size:>5} "
              f"{'n/a' if frac is None else f'{frac:>9.1%}'} {s.n_items_unpaired:>9} {k:>3}")

    # ---- §1.2 transitions -------------------------------------------------------------
    tables = build_transitions(rows, rungs=("strict",), alt_arms=ALT_ARMS)
    trans = transition_effects(tables)
    report["transitions"] = [t.as_dict() for t in tables]
    report["transition_effects"] = [e.as_dict() for e in trans]
    print(f"\n§1.2 TRANSITIONS: {len(tables)} tables, {len(trans)} effects")

    # ---- §1.3 Manski bounds -----------------------------------------------------------
    bounds = compute_bounds(rows, rung="strict", alt_arms=ALT_ARMS)
    report["bounds"] = [b.as_dict() for b in bounds]
    report["bounds_effects"] = [e.as_dict() for e in bounds_effects(bounds)]
    print(f"§1.3 BOUNDS   : {len(bounds)} difference bounds")

    # ---- §2 H3 --------------------------------------------------------------------------
    # 🚨 TWO SEPARATE LISTS, and the split is the whole point. §9c excludes the `gsm8k`
    # hash-marker arm from the H3 interaction, and MODERATE 2 of the 2026-09-02 review made the
    # library refuse it. The first version of this runner then passed `allow_excluded=True` for
    # exactly that task and appended the result to the SAME list that goes into
    # `holm_bonferroni(family="H3")` -- which is the inflation of `m` the refusal exists to
    # prevent, reintroduced one layer up. A guard in the library is not a guard if the caller is
    # allowed to hand it a permission slip.
    #
    # So: excluded tasks are computed (§9c retains the arm as a DEMONSTRATION, and the numbers are
    # informative) but they are reported under their own key and are NOT in the Holm family.
    h3: list = []
    h3_demonstration: list = []
    companions: list = []
    gap_parts: list = []
    for task in tasks:
        excluded = task in EXCLUDED_FROM_H3_INTERACTION
        task_rows = [r for r in rows if r["task"] == task]
        arms_here = {r["arm"] for r in task_rows}
        sink = h3_demonstration if excluded else h3
        for alt in ALT_ARMS:
            if alt not in arms_here or "bf16" not in arms_here:
                continue
            sink.append(interaction_marginal(
                task_rows, task=task, alt_arm=alt, allow_excluded=excluded,
                max_abs_contribution=magnitude))
        companions.extend(rung_mcnemar(task_rows, task=task))
        # R27.1: the gap is a net of two opposing effects, so both directions are reported.
        gap_parts.extend(rung_gap_decomposition(task_rows, task=task))
    report["h3_interaction"] = [e.as_dict() for e in h3]
    report["h3_interaction_demonstration_only"] = [e.as_dict() for e in h3_demonstration]
    report["h3_excluded_tasks"] = sorted(EXCLUDED_FROM_H3_INTERACTION & set(tasks))
    report["h3_rung_companion"] = [e.as_dict() for e in companions]
    report["rung_gap_decomposition"] = [e.as_dict() for e in gap_parts]
    # §9: Holm-Bonferroni within the H3 family only, and the demonstration arm is not in it.
    report["h3_holm"] = holm_block(
        h3, family="H3",
        absent_reason="no H3 interaction resolved: a task had no bf16 or no alt arm")
    def show(effects: list, heading: str) -> None:
        if not effects:
            return
        print(f"\n{heading}")
        for e in effects:
            star = "" if e.excludes_null else "  (interval spans the null)"
            print(f"  {e.name:44} {e.estimate:+7.2f} "
                  f"[{e.ci_low:+6.2f}, {e.ci_high:+6.2f}] n={e.n:>5}{star}")

    show(h3, "§2 H3 INTERACTION (marginal, permissive minus strict, pp) - IN the Holm family:")
    show(h3_demonstration,
         "§9c DEMONSTRATION ONLY, NOT an H3 interaction level and NOT in the Holm family:")

    # R27.1's headline: leniency does not only recover, it also DESTROYS.
    destroyed = [e for e in gap_parts if e.context["direction"] == "destroyed"]
    if destroyed:
        worst = max(destroyed, key=lambda e: e.estimate)
        total_destroyed = sum(e.context["count"] for e in destroyed)
        print(f"\nR27.1 RUNG GAP, both directions. Total DESTROYED by rung 2: "
              f"{total_destroyed} items")
        print(f"  worst arm: {worst.name} {worst.estimate:.2f} pp "
              f"[{worst.ci_low:.2f}, {worst.ci_high:.2f}]")

    # ---- §4 H4 --------------------------------------------------------------------------
    # 🚨 THE SAME SPLIT AS H3, because I reintroduced the identical permission-slip defect here
    # eight lines below the place I had just fixed it (code review, 2026-09-02). Two instances:
    # `allow_excluded=allow` was passed for gsm8k and the result appended to the SAME `decomps`
    # list under one key, so a row R24.3 excludes from H4's primary table sat in it unlabelled;
    # and `stopseq_differential` has no exclusion parameter at all, so gsm8k's tests entered
    # `holm_bonferroni(family="H4")` and inflated `m`.
    #
    # The reviewer was precise about direction and it is worth keeping: `m` inflation
    # OVER-corrects, so it is conservative and cannot manufacture a finding. It makes the
    # differentiality test harder to declare. The substantive half is the unlabelled row in the
    # primary decomposition table.
    #
    # 🚨 R26.2 SETTLED THIS AND REVERSED MY OWN CHOICE. I had excluded gsm8k from the stop-sequence
    # test too, as the "conservative" reading, and the CODE REVIEWER correctly flagged that the
    # code was answering a registration question by accident. It was, and the answer was wrong on
    # two counts:
    #
    #   1. TRUNC_STOPSEQ differentiality is a SEPARATE registered test, not part of the
    #      decomposition, so R24.3 does not reach it. R24.3's exclusion was grounded in the
    #      *stratum* being ~1% on the hash arm, i.e. the COMPETENCE contrast having no items. The
    #      stopseq test is a rate over ALL items, needs no stratum, and its indicator has variance
    #      whether or not the model emits `####`. The rationale simply does not transfer.
    #   2. It belongs in **H1's** Holm family, not H4's, because an earlier ruling folded
    #      TRUNC_STOPSEQ into H1 as the second mechanism for the "answer never emitted" construct.
    #
    # So my conservative reading was over-correcting `m` in the WRONG FAMILY, which is worse than
    # over-correcting in the right one: it inflated H4 while leaving H1 under-counted.
    #
    # The decomposition itself IS still split, because that exclusion is real (R24.3).
    stratum_ids = {cell: s.item_ids for cell, s in strata.items()}
    decomps, decomps_demo = [], []
    stopseq = []
    for task in tasks:
        task_rows = [r for r in rows if r["task"] == task]
        excluded = task in EXCLUDED_FROM_H4_PRIMARY
        sub = {c: v for c, v in stratum_ids.items() if c[1] == task}
        (decomps_demo if excluded else decomps).extend(
            decompose(task_rows, sub, rung="strict",
                      alt_arms=ALT_ARMS, allow_excluded=excluded))
        # Every task, gsm8k included (R26.2).
        stopseq.extend(stopseq_differential(task_rows, alt_arms=ALT_ARMS))
    report["h4_decomposition"] = [d.as_dict() for d in decomps]
    report["h4_decomposition_demonstration_only"] = [d.as_dict() for d in decomps_demo]
    report["h4_excluded_tasks"] = sorted(EXCLUDED_FROM_H4_PRIMARY & set(tasks))
    # §4's stopseq test, in H1's family per R26.2, over all tasks.
    report["h1_stopseq_differential"] = [e.as_dict() for e in stopseq]
    report["h1_holm"] = holm_block(
        stopseq, family="H1",
        absent_reason="no stopseq test resolved: no cell had both bf16 and an alt arm")
    report["h4_holm"] = holm_block(
        [], family="H4",
        absent_reason="H4's registered tests are not yet implemented as hypothesis tests")
    print(f"\n§4 H4 DECOMPOSITION: {len(decomps)} cell-arm pairs in the primary table")
    if decomps_demo:
        print(f"   plus DEMONSTRATION ONLY (R24.3): {len(decomps_demo)} pairs")
    print(f"§4 TRUNC_STOPSEQ: {len(stopseq)} tests, ALL tasks, in H1's Holm family (R26.2)")

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"analysis_{args.run_id}.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"\nreport -> {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
