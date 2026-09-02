"""Scorer unit tests — the highest-value tests in this repo. GATE G3.

`EXECUTOR-TASKS.md` A5 specifies the minimum fixture set and G3 is defined by
it passing. The scorers ARE the instrument: if they are wrong, every number in
the paper is wrong, and unlike a generation bug this would not announce itself.

Two tests here assert behaviour that looks like a defect and is deliberate:

* ``test_bbh_capitalized_answer_phrase_fails_strict`` -- the harness lookbehind
  is case-sensitive lowercase, so "The answer is X" scores ZERO on a correct
  answer. A5 requires asserting this explicitly because **it is a finding**,
  not a bug to fix.
* ``test_gsm8k_cot_uses_different_strict_regex_than_gsm8k`` -- the two GSM8K
  variants carry different strict regexes. Conflating them is the error
  ``notes/harness-configs.md`` §3 exists to prevent.
"""

from __future__ import annotations

import pytest

import quantartifact.scoring.extract as extract_module
from quantartifact.scoring.extract import (
    RUNG_ANCHORED,
    RUNGS,
    answers_match,
    extract_permissive,
    extract_permissive_anchored,
    extract_strict,
    extract_strict_last,
    normalise_answer,
    score_generation,
)

# ---------------------------------------------------------------------------
# RUNG 1 -- strict, per task, harness-verbatim
# ---------------------------------------------------------------------------


def test_gsm8k_hash_marker_present():
    generation = "Ann has 3 apples and buys 15 more.\n#### 18"
    result = extract_strict(generation, "gsm8k")
    assert result.value == "18"
    assert result.found


def test_gsm8k_hash_marker_absent_collapses_to_no_answer():
    """The core premise of the paper's GSM8K arm.

    Instruction-tuned models essentially never emit ``####``, so strict-match
    collapses toward zero for chat models as a matter of course.
    """
    generation = "Ann has 3 apples and buys 15 more. The answer is 18."
    result = extract_strict(generation, "gsm8k")
    assert result.value is None
    assert not result.found


def test_gsm8k_cot_uses_different_strict_regex_than_gsm8k():
    """The mutually-exclusive-requirements finding, pinned as a test.

    ``gsm8k`` wants ``####``; ``gsm8k_cot`` wants "The answer is N.". The SAME
    generation therefore scores differently under the two harness task configs,
    which is the whole reason decision D6 runs both.
    """
    generation = "Ann has 3 apples and buys 15 more. The answer is 18."
    assert extract_strict(generation, "gsm8k").value is None
    assert extract_strict(generation, "gsm8k_cot").value == "18"


def test_gsm8k_negative_number():
    result = extract_strict("Net change.\n#### -42", "gsm8k")
    assert result.value == "-42"


def test_gsm8k_comma_thousands_survives_strict_then_normalises():
    result = extract_strict("Total cost.\n#### 1,234", "gsm8k")
    assert result.value == "1,234"
    assert normalise_answer(result.value) == "1234"
    assert answers_match(result.value, "1234", "gsm8k")


def test_bbh_lowercase_answer_phrase_passes_strict():
    generation = "Evaluating left to right. So the answer is False."
    result = extract_strict(generation, "bbh", "boolean_expressions")
    assert result.value == "False"


def test_bbh_capitalized_answer_phrase_fails_strict():
    """A5 requires asserting this EXPLICITLY -- it is a finding, not a bug.

    The harness regex is ``(?<=the answer is )(.*)(?=.)``: case-sensitive,
    lowercase ``the``. A model that writes "The answer is False." gets zero on a
    correct answer. Rung 2 must recover it, and the gap between the two IS the
    measured artifact.
    """
    generation = "Evaluating left to right. The answer is False."
    strict = extract_strict(generation, "bbh", "boolean_expressions")

    assert strict.value is None, "capitalized phrase must FAIL rung 1"
    assert strict.capitalization_failure, "and must be flagged as a capitalization failure"

    recovered = extract_permissive(generation, "bbh", "boolean_expressions")
    assert recovered.value == "False", "rung 2 must recover it"


@pytest.mark.parametrize(
    ("generation", "expected"),
    [
        # (?=.) demands a character AFTER the capture, forcing a one-character
        # backtrack. So when the text ends exactly at the answer, the harness
        # silently truncates it.
        ("the answer is 42", "4"),        # <-- drops the "2"
        ("So the answer is False", "Fals"),  # <-- drops the "e"
        ("the answer is (B)", "(B"),      # <-- drops the ")"
        # A trailing period supplies the lookahead character, so these are fine.
        ("the answer is 42.", "42"),
        ("So the answer is False.", "False"),
        ("the answer is (B).", "(B)"),
    ],
)
def test_bbh_get_answer_drops_final_character_without_trailing_punctuation(generation, expected):
    """VERIFIED harness defect, predicted by the planner and confirmed here.

    ``(?<=the answer is )(.*)(?=.)`` is greedy with a mandatory trailing
    lookahead, so a generation ending exactly at its answer loses the last
    character and is scored WRONG on a correct answer.

    This is not hypothetical: BBH's ``until`` is ``["</s>", "Q", "\\n\\n"]``, so a
    stop sequence can terminate the text right at the answer and trigger it.
    Squarely on-thesis, plausibly differential by precision, and rung 1 stays
    verbatim regardless -- we measure the defect, we do not patch it.
    """
    assert extract_strict(generation, "bbh", "boolean_expressions").value == expected


def test_rung2_recovers_the_dropped_final_character():
    """The defect must be recoverable at rung 2, or it is not an extraction artifact."""
    for generation, gold in [
        ("So the answer is False", "False"),
        ("the answer is 42", "42"),
    ]:
        strict = extract_strict(generation, "bbh", "boolean_expressions")
        assert not answers_match(strict.value, gold, "bbh", "boolean_expressions")

        permissive = extract_permissive(generation, "bbh", "boolean_expressions")
        assert answers_match(permissive.value, gold, "bbh", "boolean_expressions"), (
            f"rung 2 must recover {gold!r} from {generation!r}"
        )


def test_self_correction_first_vs_last_match_on_phrase_arm():
    """Rung 1 takes the FIRST match; flexible-extract takes the LAST.

    Neither strict regex sets ``group_select``, so both default to the first
    match, while ``flexible-extract`` sets ``group_select: -1``. Self-correction
    is common in CoT, so part of any strict/flexible gap is a POSITION-selection
    artifact rather than a strictness artifact. R1.4 adds rung 1b to separate
    them; this fixture pins the underlying behaviour.
    """
    generation = "The answer is 6. Wait, I made an error. The answer is 8."
    strict = extract_strict(generation, "gsm8k_cot")
    # "6." not "6": the regex is `The answer is (\-?[0-9\.\,]+).` and `\.` sits
    # INSIDE the character class, so the capture greedily swallows the sentence
    # period. The harness tolerates that at metric time via
    # `regexes_to_ignore: ["\\.$"]`; we tolerate it via numeric comparison.
    assert strict.value == "6.", "rung 1 takes the FIRST match, period included"
    assert answers_match(strict.value, "6", "gsm8k_cot"), "trailing period must not fail a match"

    permissive = extract_permissive(generation, "gsm8k_cot")
    assert permissive.value == "8", "rung 2 takes the model's LAST commitment"

    # The consequence that matters: on a self-corrected generation whose final
    # answer is right, rung 1 scores it WRONG purely on match position.
    assert not answers_match(strict.value, "8", "gsm8k_cot")
    assert answers_match(permissive.value, "8", "gsm8k_cot")


def test_rung1b_selects_the_last_match_and_decomposes_the_gap():
    """Rung 1b (R1.4) separates POSITION selection from STRICTNESS.

    1 -> 1b is the position effect; 1b -> 2 is the strictness effect. Without
    1b, a self-correction gap would be misattributed to strictness, which is the
    paper's headline contrast.
    """
    generation = "The answer is 6. Wait, I erred. The answer is 8."

    first = extract_strict(generation, "gsm8k_cot")
    last = extract_strict_last(generation, "gsm8k_cot")

    # "6." keeps the period (a space follows, satisfying the trailing `.`),
    # while "8" loses it (the string ends, so the capture backtracks to give the
    # trailing `.` a character). Same greedy-plus-mandatory-lookahead mechanism
    # as the BBH last-character defect, and harmless here only because we
    # compare numerically.
    assert first.value == "6.", "rung 1 takes the FIRST match"
    assert last.value == "8", "rung 1b takes the LAST match"
    assert not answers_match(first.value, "8", "gsm8k_cot")
    assert answers_match(last.value, "8", "gsm8k_cot")


def test_rung1b_is_identical_to_rung1_when_there_is_one_match():
    generation = "So the answer is 42."
    assert (
        extract_strict(generation, "gsm8k_cot").value
        == extract_strict_last(generation, "gsm8k_cot").value
    )


def test_rung1b_never_loses_what_rung1_found():
    """Monotonicity along 1 -> 1b. Selecting a different match must not demote."""
    # Rung 1 (first match) is correct here; rung 1b picks a later, wrong one.
    generation = "The answer is 8. For contrast, the answer is 6."
    results = score_generation(generation, "8", "gsm8k_cot", math_verify_available=False)
    assert results["strict"]["correct"] is True
    assert results["strict_last"]["correct"] is True, "1b must not lose rung 1's hit"


def test_rung1b_is_a_noop_for_arc_chat():
    """No regex in the harness filter means no match position to select."""
    generation = " C"
    assert (
        extract_strict(generation, "arc_challenge_chat").value
        == extract_strict_last(generation, "arc_challenge_chat").value
    )


def test_bbh_capitalization_failure_flag_not_set_when_phrase_absent_entirely():
    """Distinguish "wrong case" from "no answer phrase at all" -- different findings."""
    strict = extract_strict("I am not sure.", "bbh", "boolean_expressions")
    assert strict.value is None
    assert not strict.capitalization_failure


def test_arc_chat_strict_is_whitespace_strip_not_regex():
    result = extract_strict("  The best answer is C  ", "arc_challenge_chat")
    assert result.value == "The best answer is C"
    assert answers_match(result.value, "C", "arc_challenge_chat")


# ---------------------------------------------------------------------------
# RUNG 2 -- permissive
# ---------------------------------------------------------------------------


def test_boxed_answer():
    result = extract_permissive(r"Thus \boxed{18} apples.", "gsm8k")
    assert result.value == "18"
    assert result.rule == "boxed"


def test_boxed_answer_with_nested_braces():
    """Brace counting, not a regex -- a non-greedy regex truncates to ``\\frac{1``."""
    result = extract_permissive(r"So \boxed{\frac{1}{2}}.", "gsm8k")
    assert result.value == r"\frac{1}{2}"


def test_currency_and_trailing_period_stripped():
    result = extract_permissive("The answer is $18.", "gsm8k")
    assert result.value == "18"


def test_comma_thousands_stripped_but_list_commas_preserved():
    assert normalise_answer("1,234") == "1234"
    assert normalise_answer("12,345,678") == "12345678"
    # A genuine word-list answer (BBH word_sorting) must NOT be mangled.
    assert normalise_answer("arson, aspect, census") == "arson, aspect, census"


def test_multiple_candidate_answers_takes_the_committed_one():
    """CoT restates numbers constantly; the LAST commitment is what is scored."""
    generation = (
        "First I thought the answer is 12. Then I recomputed. "
        "Actually the answer is 18."
    )
    result = extract_permissive(generation, "gsm8k")
    assert result.value == "18"


def test_last_number_fallback_for_numeric_task():
    generation = "5 times 4 is 20, minus 2 leaves 18"
    result = extract_permissive(generation, "gsm8k")
    assert result.value == "18"
    assert result.rule == "last_number"


def test_empty_generation_yields_no_candidate():
    for task in ("gsm8k", "gsm8k_cot", "arc_challenge_chat"):
        assert extract_strict("", task).value is None
        assert extract_permissive("", task).value is None


def test_whitespace_only_generation_yields_no_candidate():
    assert extract_permissive("   \n\n  ", "gsm8k").value is None


def test_degenerate_repetition_does_not_crash_and_finds_no_commitment():
    generation = "the the the " * 400
    result = extract_permissive(generation, "gsm8k")
    # No number anywhere, and no commitment phrase followed by content.
    assert result.value is None


def test_degenerate_numeric_repetition_is_not_credited_as_correct():
    generation = "7 " * 500
    result = extract_permissive(generation, "gsm8k")
    assert result.value == "7"
    assert not answers_match(result.value, "18", "gsm8k")


def test_truncated_mid_answer_yields_no_committed_answer():
    """A generation cut off before stating the answer must NOT be credited."""
    generation = "Ann has 3 apples and buys 15 more, so the total is"
    result = extract_permissive(generation, "gsm8k")
    assert result.value is None or not answers_match(result.value, "18", "gsm8k")


def test_truncated_after_commitment_phrase_with_nothing_following():
    result = extract_permissive("Therefore the answer is", "gsm8k")
    assert result.value is None


def test_mc_option_letter_extraction():
    generation = "Given the options, The best answer is C"
    result = extract_permissive(generation, "arc_challenge_chat")
    assert result.value == "C"


def test_mc_does_not_fall_back_to_stray_numbers():
    """A multiple-choice task must not extract a number from the reasoning."""
    generation = "There are 4 options and 3 look wrong. The best answer is B"
    result = extract_permissive(generation, "arc_challenge_chat")
    assert result.value == "B"


# ---------------------------------------------------------------------------
# Numeric comparison
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("extracted", "gold"),
    [("18", "18"), ("18.0", "18"), ("18.00", "18"), ("1,234", "1234"), ("$18", "18")],
)
def test_numeric_equivalence(extracted, gold):
    """A formatting difference is not a competence difference.

    Treating "18.0" as wrong for gold "18" would inflate precisely the effect
    this paper exists to measure honestly.
    """
    assert answers_match(extracted, gold, "gsm8k")


def test_numeric_inequality_still_detected():
    assert not answers_match("19", "18", "gsm8k")
    assert not answers_match("180", "18", "gsm8k")


# ---------------------------------------------------------------------------
# MONOTONICITY -- the ladder contract. Asserted on FIXTURES only.
# ---------------------------------------------------------------------------

# (generation, gold, task, subtask)
MONOTONICITY_FIXTURES = [
    ("Ann buys apples.\n#### 18", "18", "gsm8k", None),
    ("The answer is 18.", "18", "gsm8k_cot", None),
    (r"Thus \boxed{18}.", "18", "gsm8k", None),
    ("Total is $1,234.", "1234", "gsm8k", None),
    ("Net is -42, so #### -42", "-42", "gsm8k", None),
    ("So the answer is False.", "False", "bbh", "boolean_expressions"),
    ("The answer is False.", "False", "bbh", "boolean_expressions"),
    ("The best answer is C", "C", "arc_challenge_chat", None),
    ("", "18", "gsm8k", None),
    ("the the the " * 50, "18", "gsm8k", None),
    ("I cannot answer that.", "18", "gsm8k", None),
    ("First the answer is 12. Actually the answer is 18.", "18", "gsm8k", None),
]


@pytest.mark.parametrize(("generation", "gold", "task", "subtask"), MONOTONICITY_FIXTURES)
def test_rungs_are_monotone_on_fixtures(generation, gold, task, subtask):
    """Anything rung n marks correct, rung n+1 must also mark correct.

    A violation is either a bug or a genuine finding and must be triaged, never
    ignored. Asserted on fixtures ONLY -- an empirical violation on real
    generations is a REPORTABLE RESULT, not a test failure.
    """
    results = score_generation(
        generation, gold, task, subtask, rungs=RUNGS, math_verify_available=False
    )

    ladder = [results[rung]["correct"] for rung in RUNGS if rung in results]
    for lower, higher in zip(ladder, ladder[1:], strict=False):
        if lower is None or higher is None:
            continue  # math-verify unavailable locally; covered by its own test
        assert not (lower and not higher), (
            f"monotonicity violated for {task}/{subtask}: "
            f"rung marked correct but the next rung did not. ladder={ladder}"
        )


def test_monotonicity_fixture_set_is_not_vacuous():
    """Guard the guard: at least one fixture must actually be scored correct.

    An all-incorrect fixture set would make the monotonicity test pass
    trivially and hide a broken ladder.
    """
    any_correct = False
    for generation, gold, task, subtask in MONOTONICITY_FIXTURES:
        results = score_generation(
            generation, gold, task, subtask, math_verify_available=False
        )
        if any(results[rung]["correct"] for rung in results):
            any_correct = True
            break
    assert any_correct


def test_permissive_is_at_least_as_permissive_as_strict_on_fixtures():
    """Rung 2 must recover everything rung 1 finds. This is the headline contrast."""
    for generation, gold, task, subtask in MONOTONICITY_FIXTURES:
        strict = extract_strict(generation, task, subtask)
        if not strict.found or not answers_match(strict.value, gold, task, subtask):
            continue
        permissive = extract_permissive(generation, task, subtask)
        assert answers_match(permissive.value, gold, task, subtask), (
            f"rung 2 lost an answer rung 1 found: {generation!r}"
        )


# ---------------------------------------------------------------------------
# RUNG 3 -- math-verify. Timeout path is required by A5.
# ---------------------------------------------------------------------------


def test_math_verify_unavailable_is_recorded_not_crashed():
    results = score_generation(
        "The answer is 18.", "18", "gsm8k_cot", math_verify_available=False
    )
    assert results["math_verify"]["note"] == "math_verify_unavailable"
    assert results["math_verify"]["correct"] is None


def test_math_verify_timeout_path_records_incorrect_with_note(monkeypatch):
    """A5 requires a math-verify timeout path.

    A timeout must be recorded as incorrect WITH a note, never silently
    dropped, so the timeout count can be reported.
    """
    fake = type(
        "FakeMathVerify",
        (),
        {
            "parse": staticmethod(lambda text, **kwargs: text),
            "verify": staticmethod(
                lambda gold, prediction, **kwargs: (_ for _ in ()).throw(TimeoutError())
            ),
        },
    )
    monkeypatch.setitem(
        __import__("sys").modules, "math_verify", fake
    )

    correct, note = extract_module.verify_math("18", "18")
    assert correct is False
    assert note == "math_verify_timeout"


def test_math_verify_parse_error_records_note(monkeypatch):
    def boom(text, **kwargs):
        raise ValueError("unparseable")

    fake = type("FakeMathVerify", (), {"parse": staticmethod(boom), "verify": staticmethod(lambda *a, **k: True)})
    monkeypatch.setitem(__import__("sys").modules, "math_verify", fake)

    correct, note = extract_module.verify_math("18", "!!!")
    assert correct is False
    assert note is not None and note.startswith("math_verify_parse_error")


def test_math_verify_never_loses_a_rung2_correct(monkeypatch):
    """math-verify sits at 45.6% F1 in independent evaluation.

    It is a fast path, not an oracle, so it must never demote an answer rung 2
    already scored correct -- otherwise the ladder is not monotone.
    """
    fake = type(
        "FakeMathVerify",
        (),
        {
            "parse": staticmethod(lambda text, **kwargs: text),
            "verify": staticmethod(lambda gold, prediction, **kwargs: False),
        },
    )
    monkeypatch.setitem(__import__("sys").modules, "math_verify", fake)

    results = score_generation("The answer is 18.", "18", "gsm8k_cot")
    assert results["permissive"]["correct"] is True
    assert results["math_verify"]["correct"] is True


# ---------------------------------------------------------------------------
# R27.2: trailing-continuation fixtures. THE PROCESS FIX.
# ---------------------------------------------------------------------------
#
# 🚨 Rung monotonicity failed on real generations while passing on fixtures, and the reason is
# that the fixture set contained NO generation that continues past its committed answer. The
# fixtures were unrepresentative of the failure mode, so the ladder's two-sided error was
# invisible to the tests. The planner's process fix: add cases for each shape traced on
# `pilot-v2`, "otherwise this class recurs".
#
# ⚠️ These are DELIBERATELY EXCLUDED from `MONOTONICITY_FIXTURES`. They violate monotonicity by
# construction, which is the point: the gate asserts monotonicity on generations that should have
# it, while these pin the KNOWN behaviour on generations that should not, so it cannot drift
# silently in either direction.
#
# Each tuple is (label, generation, gold, task, subtask, rung2_destroys, rung2a_recovers).
TRAILING_CONTINUATION_FIXTURES = [
    # CLASS A: leniency reaches into trailing prose after a committed marker. Genuine destruction.
    (
        "A: hash marker then prose restatement",
        "She has 28 sweets, so 28 / 2 = <<28/2=14>>14.\n#### 14\n\n"
        "Final Answer: Jean can fill 14 bags of lollipops.\n#### 14",
        "14", "gsm8k", None, True, True,
    ),
    (
        "A: hash marker then a formatting note",
        "1300+130=<<1300+130=1430>>1430.\n#### 1430\n\n"
        "Note: The answer is in dollar format with two decimal points, "
        "even if it represents a whole number.",
        "1430", "gsm8k", None, True, True,
    ),
    # CLASS B: several commitments, model self-corrects to a WRONG one. Strict takes the first
    # (harness `take_first`), rung 2 takes the last. Genuine destruction, and rung 2a recovers it
    # because anchoring on the FIRST commitment is exactly the counterfactual.
    (
        "B: two commitments, self-corrected to the wrong one",
        "not not False and not not not False. So the answer is False.\n\n"
        "Alternatively, let me re-evaluate. not not False = True, "
        "not not not False = True, so the answer is True. "
        "It seems there was an error in my initial reasoning.",
        "False", "bbh", "boolean_expressions", True, True,
    ),
    (
        "B: four commitments, last one a non-answer",
        "Amy finished third, so the answer is (A).\n\n"
        "The orange book is second, so the answer is (A).\n\n"
        "The red book is fourth, so the answer is (C).\n\n"
        "Therefore the answer is not (A), (B), or (E).",
        "(A)", "bbh", "logical_deduction_five_objects", True, True,
    ),
]

# 🚨 CLASS C IS SEPARATE AND RUNS THE OTHER WAY. Recorded because lumping it in with the above
# would misstate the finding.
#
# On `llama dyck_languages idx-0` the model's stated answer is `] ] ]` (three brackets) and the
# gold is `] ]` (two). So the model is WRONG. Rung 1 nevertheless scores it correct, because BBH's
# harness regex `(?<=the answer is )(.*)(?=.)` DROPS THE FINAL CHARACTER -- a defect already pinned
# as a test in this file and recorded in `notes/gates.md` under G3. Rung 2 reports `] ] ]`, which is
# faithful to what the model said.
#
# So this is NOT leniency destroying a correct answer. It is **strict being accidentally correct
# via a known harness defect**, and rung 2 is the rung telling the truth. Two of the six measured
# "monotonicity violations" are this shape, which is why the destruction claim rests on four cases
# rather than six.
STRICT_ACCIDENTALLY_CORRECT_FIXTURES = [
    (
        "C: strict lucky via the drops-final-char defect",
        'The final stack is "[[".\nWe will need to pop out "[", "[", "[", '
        'one by one in that order.\nSo, we need "]", "]", "]". So the answer is ] ] ]',
        "] ]", "bbh", "dyck_languages",
    ),
]



@pytest.mark.parametrize(
    ("label", "generation", "gold", "task", "subtask", "destroys", "anchored_recovers"),
    TRAILING_CONTINUATION_FIXTURES,
    ids=[f[0] for f in TRAILING_CONTINUATION_FIXTURES],
)
def test_trailing_continuation_behaviour_is_pinned(
    label, generation, gold, task, subtask, destroys, anchored_recovers
):
    """Pin rung 2's DESTRUCTION and rung 2a's partial recovery on each traced shape.

    Rung 2 is lm-eval's `flexible-extract` with `group_select: -1`, i.e. published practice, so
    its destruction here is the measured property being reported (prereg §5a), not a bug to fix.
    Rung 2a is the anchored counterfactual.
    """
    strict = extract_strict(generation, task, subtask)
    permissive = extract_permissive(generation, task, subtask)
    anchored = extract_permissive_anchored(generation, task, subtask)

    strict_right = strict.value is not None and answers_match(
        strict.value, gold, task, subtask)
    permissive_right = permissive.value is not None and answers_match(
        permissive.value, gold, task, subtask)
    anchored_right = anchored.value is not None and answers_match(
        anchored.value, gold, task, subtask)

    assert strict_right, (
        f"{label}: the fixture is only meaningful if rung 1 gets it RIGHT, since destruction is "
        f"defined as rung 1 right and rung 2 wrong. strict extracted {strict.value!r}"
    )
    if destroys:
        assert not permissive_right, (
            f"{label}: rung 2 was expected to DESTROY this answer and did not. If the extractor "
            f"changed, prereg §5a's measured destruction rate changed with it."
        )
    assert anchored_right is anchored_recovers, (
        f"{label}: rung 2a recovery expected {anchored_recovers}, got {anchored_right} "
        f"(extracted {anchored.value!r}). Rung 2a reduces destruction but does not eliminate it, "
        f"and the dyck case is why: the model's only commitment phrase is the wrong one."
    )


def test_trailing_continuation_fixtures_are_excluded_from_the_monotonicity_gate():
    """The gate must keep asserting monotonicity on fixtures that should have it.

    If these leaked into `MONOTONICITY_FIXTURES` the gate would start failing for a reason that
    is a reported finding rather than a defect, and the signal would be lost.
    """
    monotone_generations = {f[0] for f in MONOTONICITY_FIXTURES}
    for fixture in TRAILING_CONTINUATION_FIXTURES:
        assert fixture[1] not in monotone_generations


def test_rung_2a_is_not_in_the_registered_rung_tuple():
    """R27.1: rung 2a is ADDITIVE, so it must not enter a family specified over RUNGS."""
    assert RUNG_ANCHORED not in RUNGS


@pytest.mark.parametrize(
    ("label", "generation", "gold", "task", "subtask"),
    STRICT_ACCIDENTALLY_CORRECT_FIXTURES,
    ids=[f[0] for f in STRICT_ACCIDENTALLY_CORRECT_FIXTURES],
)
def test_strict_can_be_accidentally_correct_via_the_dropped_final_char(
    label, generation, gold, task, subtask
):
    """Class C. Rung 1 scores a WRONG model answer as correct, and rung 2 is the honest one.

    BBH's harness regex drops the final character, so a model that says `] ] ]` (three) has `] ]`
    (two) extracted, which happens to match the gold. Two of the six measured monotonicity
    violations are this shape, so they must NOT be counted as leniency destroying a correct answer:
    the destruction claim rests on classes A and B only.
    """
    strict = extract_strict(generation, task, subtask)
    permissive = extract_permissive(generation, task, subtask)

    assert strict.value is not None and answers_match(strict.value, gold, task, subtask), (
        f"{label}: rung 1 is expected to score this CORRECT, by accident")
    assert permissive.value is not None, f"{label}: rung 2 should extract something"
    assert not answers_match(permissive.value, gold, task, subtask), (
        f"{label}: rung 2 is expected to report what the model actually said, which is wrong")
    assert permissive.value.count("]") > strict.value.count("]"), (
        f"{label}: the tell is that rung 2 has MORE of the repeated token than rung 1, because "
        f"rung 1 lost one to the dropped final character")
