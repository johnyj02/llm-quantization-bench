"""Compliance, termination-class, and H2 trace-scan tests. Part of GATE G3.

The termination-class tests pin down the correction in
``notes/harness-configs.md`` §4: defining truncation as
``finish_reason == "length"`` alone undercounts it, because the harness ``until``
strings cut generations mid-reasoning and report ``"stop"``. BBH's ``until``
contains a **bare capital Q** and a blank line; ARC-chat's contains a **single
period** with a 100-token cap.

The trace-scan tests pin the false-positive guard. Gold ``18`` matching a trace
about "18 apples" that concluded ``24`` is the failure mode that would make H2
worthless, and H2 is the hypothesis where we claim another paper's authors did
not ask the obvious question -- so it has to be airtight.
"""

from __future__ import annotations

import pytest

from quantartifact.scoring.compliance import (
    ABSENT,
    COMPLETE,
    COMPLIANT,
    RECOVERABLE,
    TRUNC_BUDGET,
    TRUNC_STOPSEQ,
    classify_termination,
    incidental_match_floor,
    label_compliance,
    scan_trace,
)

# ---------------------------------------------------------------------------
# Compliance 2x2 -- orthogonal to correctness
# ---------------------------------------------------------------------------


def test_compliant_when_canonical_format_present():
    labels = label_compliance(
        "Ann buys apples.\n#### 18", "gsm8k", finish_reason="stop", stop_reason="</s>"
    )
    assert labels.compliance == COMPLIANT


def test_recoverable_when_only_permissive_finds_it():
    labels = label_compliance(
        "Ann buys apples. The answer is 18.",
        "gsm8k",
        finish_reason="stop",
        stop_reason="</s>",
    )
    assert labels.compliance == RECOVERABLE


def test_absent_when_no_identifiable_answer():
    labels = label_compliance(
        "I cannot determine that.", "gsm8k", finish_reason="stop", stop_reason="</s>"
    )
    assert labels.compliance == ABSENT


def test_compliance_is_independent_of_correctness():
    """A confidently-wrong answer in the right slot is COMPLIANT.

    This orthogonality is what breaks the circularity objection: compliance
    asks whether the answer is in the expected PLACE, competence asks whether
    it is right.
    """
    labels = label_compliance(
        "Ann buys apples.\n#### 99", "gsm8k", finish_reason="stop", stop_reason="</s>"
    )
    assert labels.compliance == COMPLIANT  # gold is 18; still compliant


def test_missing_finish_reason_is_flagged_not_silently_defaulted():
    """``finish_reason`` is unreconstructable, so its absence must be loud."""
    labels = label_compliance("#### 18", "gsm8k", finish_reason=None)
    assert labels.note is not None
    assert "finish_reason" in labels.note


# ---------------------------------------------------------------------------
# Termination classes
# ---------------------------------------------------------------------------


def test_budget_truncation_detected():
    assert (
        classify_termination("gsm8k", finish_reason="length", stop_reason=None, committed=False)
        == TRUNC_BUDGET
    )


def test_budget_truncation_takes_precedence_even_when_committed():
    """Hitting the cap is the class the budget sweep manipulates.

    A generation can contain an answer AND still have been cut off; H1 needs the
    cap-hit recorded as such.
    """
    assert (
        classify_termination("gsm8k", finish_reason="length", stop_reason=None, committed=True)
        == TRUNC_BUDGET
    )


def test_bbh_bare_capital_q_stop_with_no_answer_is_stopseq_truncation():
    """The finding: BBH's ``until`` contains a BARE CAPITAL "Q".

    Any chain of thought containing one is cut mid-reasoning and reported as
    ``"stop"``, which the naive ``finish_reason == "length"`` definition misses
    entirely.
    """
    assert (
        classify_termination("bbh", finish_reason="stop", stop_reason="Q", committed=False)
        == TRUNC_STOPSEQ
    )


def test_bbh_blank_line_stop_with_no_answer_is_stopseq_truncation():
    assert (
        classify_termination("bbh", finish_reason="stop", stop_reason="\n\n", committed=False)
        == TRUNC_STOPSEQ
    )


def test_arc_chat_single_period_stop_with_no_answer_is_stopseq_truncation():
    """ARC-chat stops on a single period, with a 100-token cap."""
    assert (
        classify_termination(
            "arc_challenge_chat", finish_reason="stop", stop_reason=".", committed=False
        )
        == TRUNC_STOPSEQ
    )


def test_stop_sequence_with_committed_answer_is_complete_not_truncated():
    """The stop sequence did its job -- this is not an artifact."""
    assert (
        classify_termination("bbh", finish_reason="stop", stop_reason="\n\n", committed=True)
        == COMPLETE
    )


def test_eos_with_no_answer_is_complete_not_stopseq():
    """Model chose to stop with no answer: degeneration or refusal.

    That is a model behaviour, not a harness artifact, so it must not be
    counted as stop-sequence truncation -- otherwise TRUNC_STOPSEQ stops being
    a clean internal control.
    """
    assert (
        classify_termination("bbh", finish_reason="stop", stop_reason="</s>", committed=False)
        == COMPLETE
    )


def test_truncated_flag_tracks_budget_class_only():
    """§7.2's ``truncated`` means the BUDGET class specifically."""
    budget = label_compliance("Ann buys", "gsm8k", finish_reason="length")
    stopseq = label_compliance("Ann buys", "bbh", finish_reason="stop", stop_reason="Q")
    assert budget.truncated is True
    assert stopseq.truncated is False
    assert stopseq.termination == TRUNC_STOPSEQ


def test_truncated_and_absent_can_co_occur():
    """They are orthogonal; conflating them loses the mechanism."""
    labels = label_compliance("Ann has 3 apples and buys", "gsm8k", finish_reason="length")
    assert labels.compliance == ABSENT
    assert labels.truncated is True


def test_positional_fallback_scores_but_does_not_count_as_compliant():
    """`found` (for scoring) vs `committed` (for compliance). Load-bearing for §4.0.

    "Ann has 3 apples and buys" is truncated with no committed answer, but rung
    2's last_number fallback yields "3". It must still be SCOREABLE (that is
    faithful to the harness's own flexible-extract) while NOT counting as
    compliant -- otherwise truncated generations enter the always-compliant
    stratum and corrupt the identified competence contrast.
    """
    labels = label_compliance("Ann has 3 apples and buys", "gsm8k", finish_reason="length")

    assert labels.permissive_found is True, "rung 2 still extracts, for scoring"
    assert labels.permissive_committed is False, "but it is not commitment evidence"
    assert labels.permissive_rule == "last_number"
    assert labels.compliance == ABSENT


def test_commitment_rules_do_count_as_compliant():
    """Boxed / answer-phrase / #### are positive commitment evidence."""
    for generation, expected_rule in [
        (r"So \boxed{18}.", "boxed"),
        ("Hence the answer is 18.", "commitment_phrase"),
    ]:
        labels = label_compliance(generation, "gsm8k", finish_reason="stop", stop_reason="</s>")
        assert labels.compliance == RECOVERABLE, generation
        assert labels.permissive_committed is True, generation
        assert labels.permissive_rule == expected_rule


def test_arc_chat_canonical_letter_is_compliant():
    """The task ships gen_prefix "The best answer is", so " C" is the canonical output."""
    labels = label_compliance(" C", "arc_challenge_chat", finish_reason="stop", stop_reason=".")
    assert labels.compliance == COMPLIANT


def test_arc_chat_rambling_output_is_not_trivially_compliant():
    """arc_challenge_chat's rung-1 filter has NO regex, so it accepts anything.

    Taken literally that would make EVERY ARC-chat generation COMPLIANT, the
    compliance column vacuous, and the H4 decomposition meaningless on exactly
    the task that is our structural control (H6). Canonical shape is therefore
    required for the COMPLIANCE label, while rung-1 SCORING stays
    harness-verbatim.
    """
    labels = label_compliance(
        "Well, option A looks plausible, and B might also work",
        "arc_challenge_chat",
        finish_reason="length",
    )
    assert labels.strict_found is True, "rung 1 still yields a value, per the harness"
    assert labels.compliance != COMPLIANT, "but it is not canonical format"
    # Rung 2's last_option_letter is a positional fallback, not commitment.
    assert labels.permissive_committed is False
    assert labels.compliance == ABSENT


# ---------------------------------------------------------------------------
# H2 trace scan
# ---------------------------------------------------------------------------


def test_gold_in_trace_but_not_extracted():
    """The H2 target: correct answer present in reasoning, wrong answer committed."""
    generation = "We get 18 at the midpoint. Then I erred. The answer is 24."
    result = scan_trace(generation, gold="18", extracted="24")
    assert result.gold_in_trace


def test_incidental_substring_does_not_match_numerically():
    """Gold 18 must NOT match inside 180 or 1.18 -- the top source of false hits."""
    assert not scan_trace("There were 180 apples.", gold="18", extracted="24").gold_in_trace
    assert not scan_trace("The ratio is 1.18 exactly.", gold="18", extracted="24").gold_in_trace


def test_conservative_variant_requires_late_position_or_commitment_marker():
    early = "18 is where we start. " + ("filler words here. " * 60) + "The answer is 24."
    result = scan_trace(early, gold="18", extracted="24")
    assert result.gold_in_trace, "permissive variant should still see it"
    assert not result.gold_in_trace_conservative, "but it is early and uncommitted"


def test_conservative_variant_accepts_gold_in_final_span():
    generation = "Long reasoning here. " * 20 + "Therefore 18."
    result = scan_trace(generation, gold="18", extracted="24")
    assert result.gold_in_trace_conservative


def test_conservative_variant_excludes_already_correct_extraction():
    """If rung 2 already extracted the gold, this is not a RECOVERABLE failure."""
    generation = "So the answer is 18."
    result = scan_trace(generation, gold="18", extracted="18")
    assert not result.gold_in_trace_conservative


def test_small_gold_stratum_flagged():
    """Single-digit golds have a high incidental match rate; reported separately."""
    assert scan_trace("We had 7 items. The answer is 9.", gold="7", extracted="9").small_gold
    assert not scan_trace("We had 42 items.", gold="42", extracted="9").small_gold


def test_empty_gold_is_handled():
    result = scan_trace("anything at all", gold="", extracted=None)
    assert not result.gold_in_trace


# ---------------------------------------------------------------------------
# The false-positive floor -- MANDATORY, or H2 means nothing
# ---------------------------------------------------------------------------


def test_incidental_match_floor_computed_over_extracted_and_wrong_only():
    """The floor's denominator is items where extraction SUCCEEDED and was WRONG.

    In those items the gold cannot be the model's committed answer, so any
    trace appearance is incidental by construction.
    """
    records = [
        # Eligible: extracted, wrong.
        {"extracted": "24", "correct": False, "gold_in_trace": True},
        {"extracted": "24", "correct": False, "gold_in_trace": False},
        {"extracted": "31", "correct": False, "gold_in_trace": False},
        {"extracted": "12", "correct": False, "gold_in_trace": False},
        # Ineligible: correct.
        {"extracted": "18", "correct": True, "gold_in_trace": True},
        # Ineligible: nothing extracted.
        {"extracted": None, "correct": False, "gold_in_trace": True},
    ]
    floor = incidental_match_floor(records)
    assert floor["n_eligible"] == 4
    assert floor["floor_permissive"] == pytest.approx(0.25)


def test_incidental_match_floor_handles_empty_input():
    floor = incidental_match_floor([])
    assert floor["n_eligible"] == 0
    assert floor["floor_permissive"] != floor["floor_permissive"]  # NaN


def test_incidental_match_floor_reports_small_gold_separately():
    records = [
        {"extracted": "9", "correct": False, "gold_in_trace": True, "small_gold": True},
        {"extracted": "9", "correct": False, "gold_in_trace": True, "small_gold": True},
        {"extracted": "99", "correct": False, "gold_in_trace": False, "small_gold": False},
    ]
    floor = incidental_match_floor(records)
    assert floor["n_small_gold"] == 2
    assert floor["floor_permissive_small_gold"] == pytest.approx(1.0)
    assert floor["floor_permissive"] == pytest.approx(2 / 3)
