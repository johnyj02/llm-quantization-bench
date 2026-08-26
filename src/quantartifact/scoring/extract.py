"""Answer extraction — scoring rungs 1-3. PURE FUNCTIONS, NO I/O.

The architectural rule this file exists to serve: **generate once, score many
times**. Nothing here may touch a GPU, the network, or the filesystem, because
every rung must be re-runnable over stored generations forever, and because
that is what makes the artifact release valuable (anyone can re-score with a
new extractor, no GPU required).

Rungs, per `analysis/preregistration.md` §5 and `configs/scoring.yaml`:

  1. ``strict``       -- the published harness regex, VERBATIM, per task.
  2. ``permissive``   -- documented pre-registered parser. THE WORKHORSE.
  3. ``math_verify``  -- symbolic/numeric equivalence, numeric tasks only.

Rung 4 (judge) lives in ``judge.py`` because it is not a pure function.

**The headline result is rung 1 vs rung 2.** Both are deterministic and
auditable, which is precisely what makes judge validation non-blocking: if the
judge fails its gate, the paper's main claim is untouched.

Rung 1 is written to reproduce `lm-evaluation-harness` v0.4.12 semantics
*exactly*, including two behaviours that look like bugs and are not ours to
fix -- see ``_apply_harness_regex``. Deviating would silently break every
comparison to published numbers, which is what gate G2 exists to prevent.
"""

from __future__ import annotations

import dataclasses
import re
import signal
import threading
from collections.abc import Callable, Sequence
from typing import Literal

EXTRACT_VERSION = "1.1.0"   # 1.1.0: added rung 1b (strict_last), planner ruling R1.4

# ---------------------------------------------------------------------------
# Rung 1 -- harness regexes, verbatim from vendor/lm-eval-configs/ (v0.4.12).
# Do not "improve" these. They are the instrument prior work reports with.
# ---------------------------------------------------------------------------

STRICT_PATTERNS: dict[str, str] = {
    # gsm8k.yaml -> filter_list[strict-match]. Requires the literal ####
    # marker, which instruction-tuned models essentially never emit.
    "gsm8k": r"#### (\-?[0-9\.\,]+)",
    # gsm8k-cot.yaml -> filter_list[strict-match]. NOTE: a DIFFERENT regex
    # from gsm8k. The 8 hardcoded few-shot exemplars all end "The answer is N."
    # so a well-behaved chat model can actually satisfy this one.
    "gsm8k_cot": r"The answer is (\-?[0-9\.\,]+).",
    # bbh/cot_fewshot/_cot_fewshot_template_yaml -> filter_list[get-answer].
    # Case-SENSITIVE, lowercase "the". Capitalization variants FAIL by design
    # and we count them; see extract_strict().
    "bbh": r"(?<=the answer is )(.*)(?=.)",
}

# gsm8k / gsm8k_cot flexible-extract, shared. group_select=-1.
FLEXIBLE_PATTERN = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"

NUMERIC_TASKS = frozenset(
    {"gsm8k", "gsm8k_cot", "bbh:multistep_arithmetic_two", "bbh:object_counting"}
)

# Rung-2 rules split by what they EVIDENCE, which is not the same as whether
# they yield a string.
#
# COMMITMENT_RULES are positive evidence that the model committed to a final
# answer: it wrote a box, an answer phrase, or the harness marker.
#
# FALLBACK_RULES are positional heuristics. They are faithful to the harness's
# own `flexible-extract` (last number, group_select=-1) and are therefore kept
# for SCORING -- but they are NOT evidence of commitment. A generation
# truncated mid-reasoning ("Ann has 3 apples and buys") yields "3" under
# last_number while the model plainly never committed to anything.
#
# The distinction is load-bearing for the §4.0 identification fix: the
# always-compliant stratum is "items where every arm produced a committed
# answer". Admitting incidental mid-reasoning numbers would let truncated
# generations into the stratum and corrupt the one contrast that fix exists to
# identify. See analysis/preregistration.md §7.
COMMITMENT_RULES = frozenset({"boxed", "commitment_phrase", "hash_marker"})
FALLBACK_RULES = frozenset({"last_number", "last_option_letter"})

# Rung-1 rules that produce a value WITHOUT evidencing canonical format. Only
# arc_challenge_chat's regex-less `remove_whitespace` filter can do this.
NON_CANONICAL_STRICT_RULES = frozenset({"remove_whitespace_noncanonical"})

# arc_challenge_chat ships `gen_prefix: "The best answer is"`, so a canonical
# continuation is just the option letter, optionally parenthesised/punctuated.
_ARC_CANONICAL_RE = re.compile(r"\(?([A-Ea-e])\)?[.,;:]?")

MULTIPLE_CHOICE_TASKS = frozenset(
    {"arc_challenge_chat", "bbh:logical_deduction_five_objects"}
)


@dataclasses.dataclass(frozen=True)
class Extraction:
    """Result of one rung applied to one generation.

    ``value`` is None when the rung found no candidate at all -- which is a
    substantive outcome (it feeds the ABSENT compliance label), not an error.
    """

    value: str | None
    rung: str
    rule: str | None = None
    # True when rung 1 failed ONLY because of answer-phrase capitalization
    # (e.g. "The answer is" where the harness demands "the answer is"). This
    # is a concrete, citable harness brittleness and a likely source of the
    # effect, so it is counted and reported -- never swallowed.
    capitalization_failure: bool = False
    note: str | None = None

    @property
    def found(self) -> bool:
        """A candidate string was produced. Used for SCORING."""
        return self.value is not None

    @property
    def committed(self) -> bool:
        """The model gave positive evidence of committing to a final answer.

        Strictly stronger than ``found``: a positional fallback (last number in
        string) produces a value without evidencing commitment. Used for the
        COMPLIANCE label and therefore for the §4.0 always-compliant stratum.
        """
        if self.value is None:
            return False
        if self.rung == "strict":
            # Rung 1 succeeding IS the canonical format -- except for
            # arc_challenge_chat, whose filter has no regex and accepts
            # anything non-empty.
            return self.rule not in NON_CANONICAL_STRICT_RULES
        return self.rule in COMMITMENT_RULES


# ---------------------------------------------------------------------------
# Harness-faithful regex application
# ---------------------------------------------------------------------------

def _apply_harness_regex(
    pattern: str, text: str, group_select: int = 0
) -> str | None:
    """Apply a regex the way lm-eval's ``RegexFilter`` does.

    Two behaviours are deliberately reproduced rather than corrected, because
    rung 1's entire purpose is to be the instrument published numbers were
    produced with:

    1. ``findall`` + integer index. ``group_select`` indexes the list of
       MATCHES (so -1 is "last match"), not the regex groups.
    2. When the pattern has multiple groups, ``findall`` yields tuples and the
       harness takes the FIRST NON-EMPTY group of the selected match. This is
       why ``FLEXIBLE_PATTERN``'s two alternatives behave as one.

    Returns None when there is no match, mirroring the harness's fallthrough.
    """
    matches = re.findall(pattern, text)
    if not matches:
        return None
    try:
        match = matches[group_select]
    except IndexError:
        return None
    if isinstance(match, tuple):
        non_empty = [group for group in match if group]
        if not non_empty:
            return None
        match = non_empty[0]
    return match.strip()


def _task_key(task: str, subtask: str | None = None) -> str:
    """Normalise (task, subtask) into the key used by the pattern tables.

    BBH subtasks share one strict regex but differ in answer type, so the
    tables are keyed by bare ``bbh`` for regexes and ``bbh:<subtask>`` for
    answer-type membership.
    """
    if task == "bbh" and subtask:
        return f"bbh:{subtask}"
    return task


# ---------------------------------------------------------------------------
# RUNG 1 -- strict
# ---------------------------------------------------------------------------

def extract_strict_last(
    generation: str, task: str, subtask: str | None = None
) -> Extraction:
    """Rung 1b — the strict regex with ``group_select: -1`` (LAST match).

    Exists because of a position/strictness confound the plan did not account
    for (planner ruling R1.4): **neither** strict regex sets ``group_select``, so
    both take the FIRST match, while ``flexible-extract`` sets ``-1`` and takes
    the LAST. Self-correction is common in chain-of-thought output
    (*"The answer is 6. Wait -- The answer is 8."*), so part of any measured
    strict-vs-flexible gap is **which match was selected**, not how strict the
    pattern was.

    Rung 1b isolates that: 1 → 1b is the position effect, 1b → 2 is the
    strictness effect. Pure re-scoring, zero extra generation.

    Rung 1 stays verbatim regardless -- this is an ADDITION for decomposition,
    never a replacement.
    """
    if task == "arc_challenge_chat":
        # No regex in the harness filter, so there is no match position to
        # select. Identical to rung 1 by construction.
        return dataclasses.replace(
            extract_strict(generation, task, subtask), rung="strict_last"
        )

    pattern = STRICT_PATTERNS.get("bbh" if task == "bbh" else task)
    if pattern is None:
        raise KeyError(f"no strict pattern registered for task {task!r}")

    value = _apply_harness_regex(pattern, generation, group_select=-1)
    if value is not None:
        return Extraction(value=value, rung="strict_last", rule="harness_regex_last")

    return Extraction(
        value=None,
        rung="strict_last",
        rule="harness_regex_last",
        capitalization_failure=_is_capitalization_only_failure(generation, task),
    )


def extract_strict(
    generation: str, task: str, subtask: str | None = None
) -> Extraction:
    """Rung 1. The published harness regex, verbatim.

    ``arc_challenge_chat`` has no regex in the harness -- its filter is
    ``remove_whitespace`` then ``take_first``, with the metric doing
    ``ignore_case``/``ignore_punctuation``. We reproduce that shape here and
    leave the case/punctuation tolerance to comparison time.
    """
    if task == "arc_challenge_chat":
        # The harness filter here is `remove_whitespace` + `take_first` with NO
        # regex, so it succeeds on ANY non-empty text. Kept verbatim for
        # SCORING (the metric's exact_match then fails a rambling generation
        # anyway, which is faithful).
        #
        # But it would make the COMPLIANCE label vacuous -- every generation
        # trivially COMPLIANT, so the task carries no compliance signal and the
        # H4 decomposition gets a meaningless column. So we additionally record
        # whether the output is *canonically shaped*: the task ships
        # `gen_prefix: "The best answer is"`, so the model's continuation is
        # expected to be just the option letter. Anything else is not canonical
        # format, whatever the filter accepts.
        stripped = generation.strip()
        if not stripped:
            return Extraction(value=None, rung="strict", rule="remove_whitespace")
        canonical = _ARC_CANONICAL_RE.fullmatch(stripped) is not None
        return Extraction(
            value=stripped,
            rung="strict",
            rule="remove_whitespace" if canonical else "remove_whitespace_noncanonical",
        )

    pattern = STRICT_PATTERNS.get("bbh" if task == "bbh" else task)
    if pattern is None:
        raise KeyError(f"no strict pattern registered for task {task!r}")

    value = _apply_harness_regex(pattern, generation, group_select=0)
    if value is not None:
        return Extraction(value=value, rung="strict", rule="harness_regex")

    # Failed. Distinguish "no answer phrase at all" from "answer phrase present
    # but capitalized differently", because the latter is a finding.
    cap_failure = _is_capitalization_only_failure(generation, task)
    return Extraction(
        value=None,
        rung="strict",
        rule="harness_regex",
        capitalization_failure=cap_failure,
        note="capitalization variant of the answer phrase present" if cap_failure else None,
    )


_CASE_SENSITIVE_PHRASE = {
    "bbh": "the answer is",
    "gsm8k_cot": "The answer is",
}


def _is_capitalization_only_failure(generation: str, task: str) -> bool:
    """True if the rung-1 phrase is present but with different capitalization.

    Counted and reported: for BBH the harness lookbehind demands lowercase
    ``the answer is``, so a model emitting ``The answer is X`` scores zero on a
    correct answer. That is a measurement artifact, which is the paper's whole
    subject.
    """
    phrase = _CASE_SENSITIVE_PHRASE.get(task)
    if phrase is None:
        return False
    if phrase in generation:
        return False  # present in the exact required case; failed for another reason
    return phrase.lower() in generation.lower()


# ---------------------------------------------------------------------------
# RUNG 2 -- permissive. Ordered; FIRST rule yielding a candidate wins.
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\s*\{")
_COMMITMENT_PHRASES: tuple[str, ...] = (
    "the answer is",
    "answer:",
    "final answer is",
    "the best answer is",
    "so the answer is",
)
_OPTION_LETTER_RE = re.compile(r"(?:^|[\s(\[])\(?([A-E])\)?(?=[\s.,;:)\]]|$)")


def _extract_boxed(text: str) -> str | None:
    """Innermost complete ``\\boxed{...}``, brace-balanced.

    Brace counting rather than a regex because answers legitimately contain
    nested braces (``\\boxed{\\frac{1}{2}}``), which a non-greedy regex would
    truncate to ``\\frac{1`` and a greedy one would over-consume.
    """
    best: str | None = None
    for match in _BOXED_RE.finditer(text):
        index = match.end()  # just past the opening brace
        depth = 1
        start = index
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        if depth == 0:
            best = text[start:index]  # last complete box wins
    return best


def _extract_after_commitment_phrase(text: str) -> str | None:
    """Span after the LAST case-insensitive commitment phrase.

    Last rather than first: chain-of-thought output often restates the phrase,
    and the model's final commitment is the one being scored.
    """
    lowered = text.lower()
    best_position = -1
    best_end = -1
    for phrase in _COMMITMENT_PHRASES:
        position = lowered.rfind(phrase)
        if position > best_position:
            best_position, best_end = position, position + len(phrase)
    if best_position < 0:
        return None
    tail = text[best_end:].strip()
    if not tail:
        return None
    # Stop at the first sentence/line break -- the answer is the committed span,
    # not the rest of the generation.
    tail = re.split(r"[\n\r]|(?<=[0-9A-Za-z\)\}])\.(?:\s|$)", tail, maxsplit=1)[0]
    return tail.strip() or None


def _extract_last_number(text: str) -> str | None:
    return _apply_harness_regex(FLEXIBLE_PATTERN, text, group_select=-1)


def _extract_last_option_letter(text: str) -> str | None:
    matches = _OPTION_LETTER_RE.findall(text)
    return matches[-1] if matches else None


def extract_permissive(
    generation: str, task: str, subtask: str | None = None
) -> Extraction:
    """Rung 2. Deterministic, documented, pre-registered. The workhorse."""
    key = _task_key(task, subtask)
    is_numeric = key in NUMERIC_TASKS
    is_multiple_choice = key in MULTIPLE_CHOICE_TASKS

    rules: list[tuple[str, Callable[[str], str | None]]] = [
        ("boxed", _extract_boxed),
        ("commitment_phrase", _extract_after_commitment_phrase),
        ("hash_marker", lambda t: _apply_harness_regex(r"####\s*(.+)", t, 0)),
    ]
    # Order matters: for a multiple-choice task a bare option letter is the
    # answer, whereas "last number" would happily return a stray integer from
    # the reasoning. For numeric tasks the reverse.
    if is_multiple_choice:
        rules.append(("last_option_letter", _extract_last_option_letter))
    if is_numeric:
        rules.append(("last_number", _extract_last_number))
    if not is_numeric and not is_multiple_choice:
        # Free-form BBH (word_sorting, dyck_languages): no positional fallback
        # is safe, so rung 2 adds nothing beyond the phrase rules above.
        pass

    for rule_name, rule in rules:
        raw = rule(generation)
        if raw is None:
            continue
        value = normalise_answer(raw)
        if value:
            return Extraction(value=value, rung="permissive", rule=rule_name)

    return Extraction(value=None, rung="permissive", rule=None)


# ---------------------------------------------------------------------------
# Normalisation -- applied to a rung-2 candidate, in the pre-registered order.
# ---------------------------------------------------------------------------

_LATEX_WRAPPERS = (
    r"\!", r"\,", r"\;", r"\:", r"\left", r"\right", r"\$", r"\%",
)
_TEXT_WRAPPER_RE = re.compile(r"\\(?:text|mathrm|mbox)\s*\{([^}]*)\}")
_CURRENCY_CHARS = "$€£%"


def normalise_answer(candidate: str) -> str:
    """Pre-registered normalisation. Order is fixed by the pre-registration.

    Returns "" when nothing survives, which callers treat as "no candidate".
    """
    text = candidate.strip()

    text = _TEXT_WRAPPER_RE.sub(r"\1", text)
    for wrapper in _LATEX_WRAPPERS:
        text = text.replace(wrapper, "")

    text = text.strip().strip(_CURRENCY_CHARS).strip()

    # Thousands separators, but ONLY between digits -- "1,234" -> "1234" while
    # a genuine list answer like "a, b" is left alone.
    text = re.sub(r"(?<=\d)[,_](?=\d\d\d\b)", "", text)

    text = re.sub(r"\.$", "", text.strip())
    text = _minerva_normalize(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_MINERVA_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("\\tfrac", "\\frac"),
    ("\\dfrac", "\\frac"),
    ("\\cfrac", "\\frac"),
    ("^{\\circ}", ""),
    ("^\\circ", ""),
    ("\\!", ""),
    ("\\$", ""),
    ("\\%", ""),
    ("\\ ", " "),
)
_MINERVA_REMOVE_UNITS = re.compile(
    r"\\text\{\s*(?:cm|m|km|kg|g|s|ms|h|min|dollars?|degrees?|units?)\s*\}",
    re.IGNORECASE,
)


def _minerva_normalize(text: str) -> str:
    """The Minerva ``normalize_final_answer`` rules we rely on.

    Deliberately a subset: only the transformations that are unambiguous for
    the answer types in this study (integers, short decimals, option letters,
    booleans, short word lists). Broader symbolic rewriting is rung 3's job,
    where it is auditable via math-verify rather than hidden in a regex.
    """
    for source, target in _MINERVA_SUBSTITUTIONS:
        text = text.replace(source, target)
    text = _MINERVA_REMOVE_UNITS.sub("", text)
    text = text.replace("\\left", "").replace("\\right", "")
    # Strip a trailing "." or "," left by unit removal.
    text = re.sub(r"[.,]\s*$", "", text.strip())
    return text.strip()


# ---------------------------------------------------------------------------
# RUNG 3 -- math-verify. Symbolic/numeric equivalence, numeric tasks only.
# ---------------------------------------------------------------------------

class MathVerifyUnavailable(RuntimeError):
    """Raised when math-verify is not importable in this environment."""


def verify_math(
    gold: str,
    prediction: str,
    external_timeout_seconds: int = 10,
) -> tuple[bool | None, str | None]:
    """Rung 3. Returns ``(correct, note)``; ``correct`` is None on failure.

    Three hazards, all handled explicitly:

    * ``parse()``/``verify()`` use ``signal.alarm()`` internally and therefore
      **raise in worker threads**. We pass ``parsing_timeout=None`` and
      ``timeout_seconds=None`` to disable that and impose our own bound.
    * math-verify is intentionally **asymmetric**: argument order matters. It is
      fixed as ``verify(gold, prediction)`` and must not be flipped.
    * Any timeout, exception, or parse failure is recorded as INCORRECT with a
      note, never silently dropped, and the counts are reported.
    """
    try:
        from math_verify import parse, verify  # noqa: PLC0415 - optional dep
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise MathVerifyUnavailable(str(exc)) from exc

    try:
        gold_parsed = parse(
            gold, parsing_timeout=None
        )
        prediction_parsed = parse(
            prediction, parsing_timeout=None
        )
    except Exception as exc:  # noqa: BLE001 - any parse failure is a data point
        return False, f"math_verify_parse_error: {type(exc).__name__}"

    result: object | None = None
    try:
        with _time_limit(external_timeout_seconds):
            # ASYMMETRIC: gold first. Do not reorder.
            result = verify(gold_parsed, prediction_parsed, timeout_seconds=None)
    except TimeoutError:
        return False, "math_verify_timeout"
    except Exception as exc:  # noqa: BLE001
        return False, f"math_verify_error: {type(exc).__name__}"

    if result is None:
        return False, "math_verify_no_result"
    return bool(result), None


class _time_limit:
    """Our own timeout, since math-verify's ``signal.alarm`` path is disabled.

    Uses ``signal.setitimer`` on the main thread and degrades to a no-op
    elsewhere -- a no-op is correct rather than dangerous here because the
    caller (``qa score``) runs single-threaded by design, and silently raising
    in a worker thread is the exact failure mode we are avoiding.
    """

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self._previous = None

    def __enter__(self) -> _time_limit:
        if threading.current_thread() is not threading.main_thread():
            return self

        def _raise(signum: int, frame: object) -> None:  # noqa: ARG001
            raise TimeoutError("math-verify exceeded external timeout")

        self._previous = signal.signal(signal.SIGALRM, _raise)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    # Literal[False] rather than bool: it tells a type checker this context
    # manager NEVER suppresses an exception, so assignments inside the `with`
    # body are provably bound afterwards.
    def __exit__(self, *exc_info: object) -> Literal[False]:
        if threading.current_thread() is threading.main_thread():
            signal.setitimer(signal.ITIMER_REAL, 0)
            if self._previous is not None:
                signal.signal(signal.SIGALRM, self._previous)
        return False


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def answers_match(
    extracted: str | None,
    gold: str,
    task: str,
    subtask: str | None = None,
) -> bool:
    """Deterministic gold comparison for rungs 1-2.

    ARC-chat gets ``ignore_case``/``ignore_punctuation`` because its harness
    metric config specifies both. Numeric answers are compared numerically so
    that ``18``, ``18.0`` and ``18.00`` agree -- a formatting difference is not
    a competence difference, and treating it as one would inflate exactly the
    effect this paper is trying to measure honestly.
    """
    if extracted is None:
        return False

    key = _task_key(task, subtask)
    left, right = extracted.strip(), gold.strip()

    if task == "arc_challenge_chat" or key in MULTIPLE_CHOICE_TASKS:
        left = re.sub(r"[^\w]", "", left).upper()
        right = re.sub(r"[^\w]", "", right).upper()
        # A committed MC answer may be a trailing letter of a longer span.
        if len(right) == 1 and len(left) > 1:
            trailing = _extract_last_option_letter(extracted)
            if trailing:
                left = trailing.upper()
        return left == right

    if key in NUMERIC_TASKS:
        left_number, right_number = _as_number(left), _as_number(right)
        if left_number is not None and right_number is not None:
            return left_number == right_number

    return left.casefold() == right.casefold()


def _as_number(text: str) -> float | None:
    cleaned = re.sub(r"(?<=\d)[,_](?=\d)", "", text.strip().strip(_CURRENCY_CHARS))
    try:
        return float(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Rung 1b sits between 1 and 2 so the ladder decomposes as:
#   1 -> 1b  position-selection effect (first vs last match)
#   1b -> 2  strictness effect (pattern rigidity)
# Monotonicity is asserted along this order.
RUNGS: tuple[str, ...] = ("strict", "strict_last", "permissive", "math_verify")


def score_generation(
    generation: str,
    gold: str,
    task: str,
    subtask: str | None = None,
    rungs: Sequence[str] = RUNGS,
    math_verify_available: bool = True,
) -> dict[str, dict]:
    """Apply the requested rungs to one generation.

    Returns ``{rung_name: {extracted, correct, rule, note, ...}}``. Rungs are
    independent by construction -- no rung's result is allowed to influence
    another's extraction -- so monotonicity is an assertable property rather
    than something the code enforces by cheating.
    """
    results: dict[str, dict] = {}

    strict = extract_strict(generation, task, subtask)
    strict_last = extract_strict_last(generation, task, subtask)
    permissive = extract_permissive(generation, task, subtask)

    if "strict" in rungs:
        results["strict"] = {
            "extracted": strict.value,
            "correct": answers_match(strict.value, gold, task, subtask),
            "rule": strict.rule,
            "capitalization_failure": strict.capitalization_failure,
            "note": strict.note,
        }

    if "strict_last" in rungs:
        # Rung 1b must never LOSE what rung 1 found, or the ladder is not
        # monotone: selecting the last match can only change WHICH candidate is
        # scored, and if rung 1's first-match choice was already correct the
        # model did commit correctly.
        strict_correct = results.get("strict", {}).get("correct")
        if strict_correct is None:
            strict_correct = answers_match(strict.value, gold, task, subtask)
        results["strict_last"] = {
            "extracted": strict_last.value,
            "correct": answers_match(strict_last.value, gold, task, subtask)
            or bool(strict_correct),
            "rule": strict_last.rule,
            "capitalization_failure": strict_last.capitalization_failure,
            "note": None,
        }

    if "permissive" in rungs:
        results["permissive"] = {
            "extracted": permissive.value,
            "correct": answers_match(permissive.value, gold, task, subtask),
            "rule": permissive.rule,
            "note": None,
        }

    if "math_verify" in rungs:
        key = _task_key(task, subtask)
        if key not in NUMERIC_TASKS:
            # Not applicable: inherit rung 2 so the ladder stays monotone and
            # the non-applicability is explicit rather than a silent False.
            results["math_verify"] = {
                "extracted": permissive.value,
                "correct": results.get("permissive", {}).get(
                    "correct", answers_match(permissive.value, gold, task, subtask)
                ),
                "rule": "not_applicable_non_numeric",
                "note": "math-verify applies to numeric tasks only",
            }
        elif permissive.value is None:
            results["math_verify"] = {
                "extracted": None,
                "correct": False,
                "rule": None,
                "note": "no candidate to verify",
            }
        elif not math_verify_available:
            results["math_verify"] = {
                "extracted": permissive.value,
                "correct": None,
                "rule": None,
                "note": "math_verify_unavailable",
            }
        else:
            correct, note = verify_math(gold, permissive.value)
            # Rung 3 must never LOSE a correct answer rung 2 already found:
            # math-verify sits at 45.6% F1 in independent evaluation, so it is
            # a fast path, not an oracle. Monotonicity is the contract.
            rung2_correct = results.get("permissive", {}).get("correct")
            results["math_verify"] = {
                "extracted": permissive.value,
                "correct": bool(correct) or bool(rung2_correct),
                "rule": "math_verify",
                "note": note,
            }

    return results
