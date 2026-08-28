"""Determinism-audit analysis tests — gate G4's arithmetic, checked without a GPU.

The two tests that matter most:

* ``test_select_max_num_seqs_is_a_min_over_formats`` — R8 requires ONE
  ``max_num_seqs`` identical across all arms, so a value stable for bf16 but not
  GPTQ is ineligible. Taking the max of per-format maxima would silently pick a
  value that is unstable for some arm, which is the mistake this guards.
* ``test_differential_noise_floor_detects_gptq_only_nondeterminism`` — the
  scenario H8 predicts from vLLM's own source (``auto_gptq.py`` has no
  batch-invariance guard while ``auto_awq.py`` does). If it happens, that rate is
  a floor on every contrast involving GPTQ.
"""

from __future__ import annotations

import pytest

from quantartifact.determinism import (
    MAX_NUM_SEQS_SWEEP,
    ORDERING_SEEDS,
    PROBE_COMPOSITION,
    DisagreementResult,
    RepeatOutputs,
    compute_disagreement,
    differential_noise_floor,
    distinct_output_histogram,
    divergence_by_level,
    next_sweep_level,
    non_monotonic_levels,
    probe_fingerprint,
    select_max_num_seqs,
    summarise,
    sweep_plan,
)


def repeats_from(
    texts_per_repeat: list[dict[str, str]], fmt: str = "gptq-4bit", mns: int = 8
) -> list[RepeatOutputs]:
    return [
        RepeatOutputs(fmt=fmt, max_num_seqs=mns, repeat_idx=i, outputs=outputs)
        for i, outputs in enumerate(texts_per_repeat)
    ]


# ---------------------------------------------------------------------------
# compute_disagreement
# ---------------------------------------------------------------------------


def test_identical_repeats_are_bitwise_stable():
    outputs = {"a": "answer A", "b": "answer B"}
    result = compute_disagreement(repeats_from([outputs, dict(outputs), dict(outputs)]))
    assert result.bitwise_stable
    assert result.n_disagreeing == 0
    assert result.disagreement_rate == 0.0
    assert result.max_distinct_outputs == 1


def test_one_diverging_item_is_counted():
    result = compute_disagreement(
        repeats_from(
            [
                {"a": "same", "b": "first"},
                {"a": "same", "b": "second"},
                {"a": "same", "b": "first"},
            ]
        )
    )
    assert not result.bitwise_stable
    assert result.n_disagreeing == 1
    assert result.disagreement_rate == pytest.approx(0.5)
    assert result.max_distinct_outputs == 2


def test_three_distinct_outputs_reproduces_the_51187_shape():
    """vLLM #51187: 44 co-resident sequences gave 3 distinct outputs in 5 repeats."""
    result = compute_disagreement(
        repeats_from([{"a": "x"}, {"a": "y"}, {"a": "z"}])
    )
    assert result.max_distinct_outputs == 3
    assert result.disagreement_rate == 1.0


def test_missing_item_counts_as_disagreement_not_skipped():
    """A dropped generation IS a reproducibility failure; skipping it would flatter."""
    result = compute_disagreement(
        repeats_from([{"a": "x", "b": "y"}, {"a": "x"}, {"a": "x", "b": "y"}])
    )
    assert result.n_disagreeing == 1
    assert result.n_items == 2


def test_fewer_than_two_repeats_is_rejected():
    with pytest.raises(ValueError, match="at least 2 repeats"):
        compute_disagreement(repeats_from([{"a": "x"}]))


def test_distinct_output_histogram():
    hist = distinct_output_histogram(
        repeats_from(
            [
                {"a": "1", "b": "1", "c": "1"},
                {"a": "1", "b": "2", "c": "1"},
                {"a": "1", "b": "3", "c": "1"},
            ]
        )
    )
    assert hist == {1: 2, 3: 1}


def test_probe_fingerprint_is_order_independent_but_content_sensitive():
    assert probe_fingerprint({"a": "x", "b": "y"}) == probe_fingerprint({"b": "y", "a": "x"})
    assert probe_fingerprint({"a": "x"}) != probe_fingerprint({"a": "z"})


# ---------------------------------------------------------------------------
# R8 — max_num_seqs selection
# ---------------------------------------------------------------------------


def make_result(fmt: str, mns: int, stable: bool, rate: float = 0.0) -> DisagreementResult:
    return DisagreementResult(
        fmt=fmt,
        max_num_seqs=mns,
        n_items=500,
        n_disagreeing=0 if stable else int(rate * 500),
        disagreement_rate=0.0 if stable else rate,
        max_distinct_outputs=1 if stable else 2,
        bitwise_stable=stable,
    )


def test_select_max_num_seqs_is_a_min_over_formats():
    """THE R8 rule. bf16 stable to 64, GPTQ only to 16 => the answer is 16, not 64.

    §3.3 requires one value identical across all arms, so a value stable for bf16
    but not GPTQ is INELIGIBLE. Taking the max of per-format maxima would pick 64
    and silently run GPTQ nondeterministically.
    """
    results = [
        make_result("bf16", 8, True), make_result("bf16", 16, True),
        make_result("bf16", 32, True), make_result("bf16", 64, True),
        make_result("gptq-4bit", 8, True), make_result("gptq-4bit", 16, True),
        make_result("gptq-4bit", 32, False, 0.02), make_result("gptq-4bit", 64, False, 0.05),
    ]
    selection = select_max_num_seqs(results)
    assert selection["selected_max_num_seqs"] == 16
    assert selection["per_format_max_stable"] == {"bf16": 64, "gptq-4bit": 16}


def test_a_format_with_no_stable_value_blocks_selection():
    """If any format is never stable, there is no shared setting -- and that is a finding."""
    results = [
        make_result("bf16", 8, True),
        make_result("gptq-4bit", 8, False, 0.03),
    ]
    selection = select_max_num_seqs(results)
    assert selection["selected_max_num_seqs"] is None
    assert selection["formats_with_no_stable_value"] == ["gptq-4bit"]
    assert "reportable H8 result" in selection["finding_if_none"]


def test_all_formats_stable_at_the_floor_only():
    results = [
        make_result("bf16", 8, True), make_result("bf16", 16, False, 0.01),
        make_result("awq-4bit", 8, True), make_result("awq-4bit", 16, False, 0.01),
    ]
    assert select_max_num_seqs(results)["selected_max_num_seqs"] == 8


def test_sweep_values_are_recorded_so_the_frontier_is_reportable():
    selection = select_max_num_seqs([make_result("bf16", 8, True)])
    assert selection["swept"] == list(MAX_NUM_SEQS_SWEEP)


# ---------------------------------------------------------------------------
# The differential noise floor — what makes H8 consequential
# ---------------------------------------------------------------------------


def test_differential_noise_floor_detects_gptq_only_nondeterminism():
    """The scenario H8 predicts from vLLM's source.

    ``auto_gptq.py`` has NO batch-invariance guard while ``auto_awq.py`` does, so
    GPTQ keeps using Marlin, which vLLM's own comment says is not batch invariant.
    If bf16 and AWQ are clean and GPTQ is not, differential noise exists BETWEEN
    ARMS and floors every contrast involving GPTQ.
    """
    results = [
        make_result("bf16", 8, True),
        make_result("awq-4bit", 8, True),
        make_result("gptq-4bit", 8, False, 0.024),
    ]
    floor = differential_noise_floor(results)
    assert floor["differential_noise_exists"] is True
    assert floor["bf16_disagreement"] == 0.0
    assert floor["differential_vs_bf16"]["gptq-4bit"] == pytest.approx(0.024)
    assert floor["differential_vs_bf16"]["awq-4bit"] == pytest.approx(0.0)


def test_no_differential_when_everything_is_clean():
    results = [make_result(f, 8, True) for f in ("bf16", "gptq-4bit", "awq-4bit")]
    assert differential_noise_floor(results)["differential_noise_exists"] is False


def test_bf16_nondeterminism_does_not_create_a_negative_differential_claim():
    """If bf16 itself is noisy, a quieter quantized arm must not read as 'better'."""
    results = [
        make_result("bf16", 8, False, 0.05),
        make_result("gptq-4bit", 8, False, 0.01),
    ]
    floor = differential_noise_floor(results)
    assert floor["differential_vs_bf16"]["gptq-4bit"] < 0
    assert floor["differential_noise_exists"] is False, (
        "a negative differential is not evidence of extra noise in the quantized arm"
    )


# ---------------------------------------------------------------------------
# The G4 evidence object
# ---------------------------------------------------------------------------


def test_summarise_produces_the_g4_evidence_shape():
    results = [
        make_result("bf16", 8, True),
        make_result("gptq-4bit", 8, False, 0.02),
    ]
    summary = summarise(results)
    assert summary["gate"] == "G4"
    assert summary["any_nondeterminism"] is True
    assert summary["formats_tested"] == ["bf16", "gptq-4bit"]
    assert sum(summary["probe_composition"].values()) == 500
    assert "max_num_seqs_selection" in summary
    assert "differential_noise_floor" in summary


def test_probe_is_500_items_across_all_three_answer_formats():
    """A format-dependent nondeterminism must not be able to hide in the probe."""

    assert sum(PROBE_COMPOSITION.values()) == 500
    assert set(PROBE_COMPOSITION) == {"gsm8k_cot", "bbh", "arc_challenge_chat"}


# ---------------------------------------------------------------------------
# Descending sweep with early exit — the cost fix
# ---------------------------------------------------------------------------


def test_sweep_plan_reports_best_and_worst_case():
    """The launch request needs an honest range, not one optimistic number."""

    plan = sweep_plan(["bf16", "gptq-4bit", "awq-4bit", "gptq-8bit", "bnb-nf4"], 500, 3)
    assert plan["generations_per_level"] == 7500
    # TWO levels best case (R13.2): the stable level plus the one below it for the
    # non-monotonicity check. The floor is skipped when something above it is stable.
    assert plan["best_case_generations"] == 15000
    assert plan["worst_case_generations"] == 30000
    assert plan["exhaustive_would_cost"] == 30000
    assert plan["order"] == [64, 32, 16, 8], "must be DESCENDING"


def test_sweep_starts_at_the_largest_value():

    assert next_sweep_level([], ["bf16", "gptq-4bit"]) == 64


def test_sweep_stops_early_when_the_top_level_is_stable_everywhere():
    """64 stable everywhere => still measure 32 (non-monotonicity) and the floor.

    Planner ruling R12.3.1: "stable at 64, unstable at 32" would invalidate the
    frontier framing, so the level immediately below the stable one is always
    measured. Skipping it would also leave the frontier as two points, which makes
    the H8 contribution -- telling others what they can get away with -- thin.
    """

    tested = [make_result("bf16", 64, True), make_result("gptq-4bit", 64, True)]
    assert next_sweep_level(tested, ["bf16", "gptq-4bit"]) == 32, "R12.3.1 check"

    tested += [make_result("bf16", 32, True), make_result("gptq-4bit", 32, True)]
    # R13.2: the floor is NOT measured once something above it is stable -- its only
    # purpose was to guarantee SOME value is stable, and that is now known.
    assert next_sweep_level(tested, ["bf16", "gptq-4bit"]) is None, "floor skipped"


def test_floor_is_still_reached_when_nothing_above_it_is_stable():
    """R13.2's other half: if 64/32/16 all fail, the walk-down must reach 8."""

    tested = []
    for level in (64, 32, 16):
        tested += [make_result("bf16", level, False, 0.02),
                   make_result("gptq-4bit", level, False, 0.02)]
    assert next_sweep_level(tested, ["bf16", "gptq-4bit"]) == 8


def test_sweep_descends_when_one_format_fails():
    """One unstable format at 64 must drive the whole sweep down, not just its own."""

    tested = [make_result("bf16", 64, True), make_result("gptq-4bit", 64, False, 0.02)]
    assert next_sweep_level(tested, ["bf16", "gptq-4bit"]) == 32


def test_sweep_completes_a_partially_tested_level_before_descending():

    tested = [make_result("bf16", 64, True)]
    assert next_sweep_level(tested, ["bf16", "gptq-4bit"]) == 64


# ---------------------------------------------------------------------------
# R12.3.1 — the frontier framing assumes stability is monotone in batch size.
# Fewer co-resident sequences cannot be WORSE. If that is violated, "the largest
# stable value" is not a well-defined threshold and must be reported as a set.
# ---------------------------------------------------------------------------

def test_monotone_frontier_reports_no_violations():
    """Unstable at 64, stable at 32 and 8 -- the expected, well-behaved shape."""

    results = [
        make_result("bf16", 64, False, 0.02),
        make_result("bf16", 32, True),
        make_result("bf16", 8, True),
    ]
    assert non_monotonic_levels(results, ["bf16"]) == []


def test_non_monotonicity_is_detected():
    """Stable at 64 but UNSTABLE at 32 is the violation worth ruling out."""

    results = [
        make_result("bf16", 64, True),
        make_result("bf16", 32, False, 0.01),
    ]
    violations = non_monotonic_levels(results, ["bf16"])
    assert len(violations) == 1
    assert violations[0]["unstable_at"] == 32
    assert violations[0]["stable_at"] == 64
    assert "not well defined" in violations[0]["implication"].replace("\n", " ") or \
        "not \nwell defined" in violations[0]["implication"]


def test_violation_is_reported_per_format():
    """One format violating must not be hidden by another behaving."""

    results = [
        make_result("bf16", 64, True), make_result("bf16", 32, True),
        make_result("gptq-4bit", 64, True), make_result("gptq-4bit", 32, False, 0.03),
    ]
    violations = non_monotonic_levels(results, ["bf16", "gptq-4bit"])
    assert [v["format"] for v in violations] == ["gptq-4bit"]


def test_marlin_fallback_cells_are_excluded():
    """The fallback runs a DIFFERENT engine config; comparing it corrupts the frontier."""

    fallback = make_result("gptq-4bit", 8, False, 0.05)
    fallback.disabled_kernels = "MarlinLinearKernel"
    results = [make_result("gptq-4bit", 64, True), fallback]
    assert non_monotonic_levels(results, ["gptq-4bit"]) == []


def test_confirmation_varies_ordering_not_repeat_count():
    """R13.1: batch COMPOSITION is the mechanism, so vary ORDER, not repeats.

    Ten repeats of the same items in the same order re-present the scheduler with
    nearly the same batches. A different order means different co-residency, which
    is what vLLM #51187 actually reported.
    """
    assert len(ORDERING_SEEDS) == 3
    # Ordering 0 is the natural probe order, already run by the sweep, so it is
    # reused rather than regenerated -- that is what makes this cost 5,000 not 25,000.
    assert ORDERING_SEEDS[0] == 0
    assert len(set(ORDERING_SEEDS)) == 3, "seeds must be distinct or orderings repeat"


def test_confirmation_reuses_the_sweep_pass():
    """Only the extra orderings are charged: 5 arms x 2 passes x 500 = 5,000."""

    plan = sweep_plan(["bf16", "gptq-4bit", "awq-4bit", "gptq-8bit", "bnb-nf4"], 500, 3)
    assert plan["confirmation_generations"] == 5000


# ---------------------------------------------------------------------------
# A crashed cell is recorded as maximally unstable so the selection fails closed.
# That is right. Printing its synthetic 1.0 rate as a noise floor is not -- the first
# determinism smoke reported "bf16 100.0000%" when nothing had been measured at all.
# ---------------------------------------------------------------------------

def test_unmeasured_cells_are_excluded_from_the_noise_floor():
    crashed = make_result("bf16", 8, False, 1.0)
    crashed.measured = False
    real = make_result("gptq-4bit", 8, False, 0.02)
    floor = differential_noise_floor([crashed, real])
    assert floor["n_unmeasured_cells_excluded"] == 1
    # bf16 never ran, so it must not appear with a 100% rate.
    assert floor["bf16_disagreement"] == 0.0
    assert floor["per_format_worst_disagreement"].get("bf16") is None


def test_unmeasured_cells_still_fail_the_selection_closed():
    """Excluding them from the noise floor must NOT make them look eligible."""
    crashed = make_result("bf16", 64, False, 1.0)
    crashed.measured = False
    ok = make_result("gptq-4bit", 64, True)
    selection = select_max_num_seqs([crashed, ok])
    assert selection["selected_max_num_seqs"] is None
    assert "bf16" in selection["formats_with_no_stable_value"]


def test_measured_defaults_true_and_reaches_the_row():
    result = make_result("bf16", 8, True)
    assert result.measured is True
    assert result.to_row()["measured"] is True


# ---------------------------------------------------------------------------
# R15.3 — three levels, because they disagree sharply. The smoke measured
# gptq-4bit at 65% text / 10% strict answer, and awq-4bit at 75% text / 0% strict
# answer: the cascade largely collapses by the time a scorer reads the answer.
# ---------------------------------------------------------------------------

def test_divergence_collapses_from_text_to_answer():
    """Different traces, identical committed answer -- awq-4bit's observed shape."""
    items = {"a": {"gold": "42", "task": "gsm8k_cot", "subtask": None}}
    passes = [
        RepeatOutputs(fmt="awq-4bit", max_num_seqs=8, repeat_idx=0, ordering_seed=0,
                      outputs={"a": "First I add. The answer is 42."}),
        RepeatOutputs(fmt="awq-4bit", max_num_seqs=8, repeat_idx=1,
                      ordering_seed=20260827,
                      outputs={"a": "Let me reason differently. The answer is 42."}),
    ]
    result = divergence_by_level(passes, items)
    assert result["text_divergence"] == 1.0, "the traces differ"
    assert result["answer_divergence"]["strict"] == 0.0, "the answer does not"
    assert result["correctness_divergence"]["strict"] == 0.0


def test_divergence_that_reaches_correctness_is_flagged():
    """The only level that moves a reported accuracy delta."""
    items = {"a": {"gold": "42", "task": "gsm8k_cot", "subtask": None}}
    passes = [
        RepeatOutputs(fmt="gptq-4bit", max_num_seqs=8, repeat_idx=0, ordering_seed=0,
                      outputs={"a": "The answer is 42."}),
        RepeatOutputs(fmt="gptq-4bit", max_num_seqs=8, repeat_idx=1,
                      ordering_seed=20260827,
                      outputs={"a": "The answer is 41."}),
    ]
    result = divergence_by_level(passes, items)
    assert result["answer_divergence"]["strict"] == 1.0
    assert result["correctness_divergence"]["strict"] == 1.0
    assert result["escalation_metric_strict_correctness"] == 1.0


def test_items_without_gold_are_counted_not_dropped():
    """Silently dropping them would understate the denominator and flatter the rate."""
    items = {"a": {"gold": "42", "task": "gsm8k_cot", "subtask": None}}
    passes = [
        RepeatOutputs(fmt="bf16", max_num_seqs=8, repeat_idx=0, ordering_seed=0,
                      outputs={"a": "The answer is 42.", "b": "The answer is 1."}),
        RepeatOutputs(fmt="bf16", max_num_seqs=8, repeat_idx=1, ordering_seed=20260827,
                      outputs={"a": "The answer is 42.", "b": "The answer is 2."}),
    ]
    result = divergence_by_level(passes, items)
    assert result["n_items_text"] == 2
    assert result["n_items_scored"] == 1
    assert result["n_items_unscored_no_gold"] == 1
    # "b" diverges at text level but has no gold, so it cannot enter correctness.
    assert result["text_divergence"] == 0.5
    assert result["correctness_divergence"]["strict"] == 0.0


def test_one_ordering_is_refused():
    passes = [RepeatOutputs(fmt="bf16", max_num_seqs=8, repeat_idx=0, ordering_seed=0,
                            outputs={"a": "x"})]
    assert "error" in divergence_by_level(passes, {})
