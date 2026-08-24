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
    RUNGS,
    answers_match,
    extract_permissive,
    extract_strict,
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
