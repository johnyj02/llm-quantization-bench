"""Compliance labelling, termination classes, and the H2 trace scan.

PURE FUNCTIONS, with ONE exception, declared here because purity is what makes this layer
re-runnable by a third party and a false claim of it is worse than the exception:

⚠️ ``_harness_until_from_config()`` reads ``configs/tasks.yaml`` at IMPORT time to build the
``HARNESS_UNTIL`` fallback. It exists only for records written before schema 1.2.0, which do
not carry their own ``until``. Every scoring function is otherwise a pure function of its
arguments, and callers holding a record should pass ``until=record["until"]``, which bypasses
the fallback entirely. Once no pre-1.2.0 records are in play, the read can go.

Three things live here, and they are the paper's mechanism measurements rather
than bookkeeping:

* **The compliance 2×2** (`EXECUTOR-PLAN.md` §2.3). Compliance is *orthogonal*
  to correctness: it asks whether the answer is in the expected **place**, not
  whether it is right. Compliance rate is a dependent variable in its own
  right -- if it drops under quantization that is a real, novel, reportable
  effect, and it is exactly what prior work has been folding into "reasoning
  accuracy".

* **Termination classes.** `EXECUTOR-PLAN.md` §2.3 and `EXECUTOR-TASKS.md`
  §7.2 define ``TRUNCATED`` as ``finish_reason == "length"``. Verification of
  the v0.4.12 configs showed that **undercounts truncation substantially** --
  BBH's ``until`` list contains a bare capital ``Q`` and a blank line, and
  ARC-chat stops on a single period with a 100-token cap, so chains get cut
  mid-reasoning and reported as ``"stop"``. See ``notes/harness-configs.md``
  §4. We therefore classify three ways, pre-registered before any generations
  existed (``analysis/preregistration.md`` §4).

* **The H2 trace scan**, with the false-positive floor that makes it mean
  anything.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from pathlib import Path

from quantartifact.scoring.extract import (
    Extraction,
    _as_number,
    apply_until,
    extract_permissive,
    extract_strict,
    normalise_answer,
)


class UnclassifiableTerminationError(RuntimeError):
    """The terminators that produced a generation are unknown, so it cannot be classified.

    Raised rather than defaulting to ``COMPLETE``. "Unrecognised" and "the model chose to
    stop" must never be the same answer -- that conflation is what let BLOCKER 1 (BBH's
    ``trunc_stopseq_rate`` reading 0.0%) survive several scoring runs.
    """


COMPLIANCE_VERSION = "1.0.0"
TRACE_SCAN_VERSION = "1.0.0"

# --- compliance labels -----------------------------------------------------
COMPLIANT = "COMPLIANT"
RECOVERABLE = "RECOVERABLE"
ABSENT = "ABSENT"

# --- termination classes ---------------------------------------------------
TRUNC_BUDGET = "TRUNC_BUDGET"
TRUNC_STOPSEQ = "TRUNC_STOPSEQ"
COMPLETE = "COMPLETE"

# 🚨 FALLBACK ONLY. Pass the ACTUAL `until` list instead wherever you have it.
#
# This table drifted and produced a wrong number. It held BBH's PRE-R18.1 raw-level set
# `("</s>", "Q", "\n\n")` while `configs/tasks.yaml` had been corrected to the chat-level set
# `["</s>", "Q:", "<|im_end|>"]`. Consequence: `"Q:"` -- the only non-EOS terminator that can
# fire at the chat level -- was not in the table, so every BBH generation cut mid-reasoning by
# it was classified `COMPLETE`, and `trunc_stopseq_rate` for BBH read **0.0% in every arm**.
# That is §4's pre-registered TRUNC_STOPSEQ differential-by-precision test reporting a stale
# literal as a measurement. The tests locked the drift in: they asserted with `"Q"` and
# `"\n\n"` and never with `"Q:"`.
#
# ⚠️ AND THE TABLE CANNOT BE RIGHT IN PRINCIPLE. `tasks.yaml` gives BBH two terminator sets --
# `until_by_template` -- one per template level, because the terminator set is part of each
# level's DEFINITION (prereg §3b-bis). `template-v1` runs both levels in one run, so no
# single hardcoded tuple can classify both. A lookup keyed on `task` alone is the wrong shape.
#
# Hence `classify_termination(..., until=...)`: the caller passes the list that actually
# produced the generation, recorded on the record itself. This table remains only for callers
# that genuinely have no `until` -- records written before the field existed -- and is
# deliberately loaded from `tasks.yaml` rather than retyped, so it cannot drift again.
def _harness_until_from_config() -> dict[str, tuple[str, ...]]:
    """Every declared `until`, per task, read from `configs/tasks.yaml`.

    Includes BOTH template levels' sets where a task declares `until_by_template`, because a
    fallback that has to serve both levels can only be the union. That is imprecise by
    construction -- it cannot tell a `raw`-level kill from a `chat`-level one -- which is
    exactly why passing the real list is preferred and this is the fallback.
    """
    import yaml  # noqa: PLC0415 - config read, only needed on the fallback path

    # 🚨 NO `except`. This used to swallow OSError/KeyError and `return {}` -- which restores
    # BLOCKER 1 exactly: an empty table means `HARNESS_UNTIL.get(task, ())` is `()`, no
    # stop_reason ever matches, and every stop-sequence kill is reported COMPLETE. A guard
    # against a silent zero must not itself fail to a silent zero.
    #
    # And this is the LIVE path for the data already on disk: every `pilot-v2` and
    # `template-v1` record predates schema 1.2.0, so `record.get("until")` is None and this
    # fallback is what produced the re-scored 16.0% / 48.7%. Those numbers must not rest on an
    # unasserted file read.
    #
    # The path is right in-repo and at `/opt/ml/code` (submit_job stages `configs/`), but would
    # be wrong for a `pip install`ed copy -- so let it raise there rather than silently
    # reporting zeros.
    path = Path(__file__).resolve().parents[3] / "configs" / "tasks.yaml"
    tasks = yaml.safe_load(path.read_text())["tasks"]

    out: dict[str, tuple[str, ...]] = {}
    for name, config in tasks.items():
        if not isinstance(config, dict):
            continue
        strings: list[str] = list(config.get("until") or [])
        for level_until in (config.get("until_by_template") or {}).values():
            strings.extend(level_until or [])
        if strings:
            out[name] = tuple(dict.fromkeys(strings))   # dedupe, keep order
    return out


HARNESS_UNTIL: dict[str, tuple[str, ...]] = _harness_until_from_config()

# EOS-ish stop reasons that indicate the model chose to stop, not that the
# harness cut it off.
_EOS_STOP_REASONS = frozenset({"</s>", "<|im_end|>", "<|eot_id|>", "<|endoftext|>", None})


@dataclasses.dataclass(frozen=True)
class ComplianceLabels:
    compliance: str
    termination: str
    truncated: bool  # orthogonal to `compliance`; kept for the §7.2 record
    strict_found: bool
    permissive_found: bool
    # Distinct from `permissive_found`: True only when rung 2 found positive
    # evidence of commitment rather than falling back to a positional
    # heuristic. This is the field the §4.0 always-compliant stratum is built
    # from, so it is recorded explicitly rather than re-derived downstream.
    permissive_committed: bool = False
    permissive_rule: str | None = None
    note: str | None = None


def label_compliance(
    generation: str,
    task: str,
    subtask: str | None = None,
    finish_reason: str | None = None,
    stop_reason: str | None = None,
    strict: Extraction | None = None,
    permissive: Extraction | None = None,
    until: Sequence[str] | None = None,
    until_applied: str | None = None,
) -> ComplianceLabels:
    """Assign the compliance label and the termination class.

    ``finish_reason`` is REQUIRED for a meaningful result. It is emitted on
    every generation record precisely because it is unreconstructable after the
    fact, and H1 is impossible without it -- so a missing value is flagged in
    ``note`` rather than quietly defaulted.
    """
    # 🚨 UNDER THE `"scoring"` REGIME, EXTRACT FROM THE TRUNCATED TEXT (planner R34).
    #
    # `generation` now holds the FULL output, including whatever the model produced after a
    # terminator. Extracting from all of it would credit the study with answers the harness
    # would never have seen, which is the mirror image of the bug R34 fixed and just as wrong:
    # the old regime lost real answers, and this would invent ones. The truncated prefix is
    # byte-identical to what a generation-time `stop=` produced, so this keeps published
    # semantics exactly.
    #
    # `termination` below is deliberately classified against the FULL text, because deciding
    # whether a terminator fired at all requires seeing past it.
    scored_text = generation
    if until_applied == "scoring":
        scored_text, _ = apply_until(generation, _resolve_terminators(task, until))

    if strict is None:
        strict = extract_strict(scored_text, task, subtask)
    if permissive is None:
        permissive = extract_permissive(scored_text, task, subtask)

    # NOTE the asymmetry, and it is deliberate: `found` for scoring, `committed`
    # for compliance. A positional fallback (last number in string) yields a
    # value without evidencing that the model committed to an answer, so it
    # scores but does not count as compliant. See Extraction.committed and
    # analysis/preregistration.md §7.
    if strict.committed:
        compliance = COMPLIANT
    elif permissive.committed:
        compliance = RECOVERABLE
    else:
        compliance = ABSENT

    termination = classify_termination(
        task=task,
        finish_reason=finish_reason,
        stop_reason=stop_reason,
        committed=permissive.committed,
        until=until,
        generation_text=generation,
        until_applied=until_applied,
    )

    note = None
    if finish_reason is None:
        note = "finish_reason missing -- termination class is unreliable"

    return ComplianceLabels(
        compliance=compliance,
        termination=termination,
        # `truncated` stays keyed to the BUDGET class specifically, because that
        # is what §7.2 means and what the budget sweep manipulates. The wider
        # notion lives in `termination`.
        truncated=(termination == TRUNC_BUDGET),
        strict_found=strict.found,
        permissive_found=permissive.found,
        permissive_committed=permissive.committed,
        permissive_rule=permissive.rule,
        note=note,
    )


def _resolve_terminators(task: str, until: Sequence[str] | None) -> tuple[str, ...]:
    """The ACTUAL list when the caller has it, the config-derived union otherwise.

    🚨 Never a hardcoded literal: that is what drifted and became BLOCKER 1 (see
    ``HARNESS_UNTIL`` above). And it REFUSES rather than assuming, because *"I cannot know
    this task's terminators"* and *"the model chose to stop"* are different answers, and
    collapsing them into ``COMPLETE`` is how that blocker stayed hidden for weeks: 1,106 BBH
    records took the catch-all and were reported complete.
    """
    if until is not None:
        return tuple(until)
    if task in HARNESS_UNTIL:
        return HARNESS_UNTIL[task]
    raise UnclassifiableTerminationError(
        f"cannot classify termination for task {task!r}: the record carries no `until` "
        f"(pre-schema-1.2.0) and {task!r} is not in the config-derived HARNESS_UNTIL "
        f"{sorted(HARNESS_UNTIL)}.\n"
        f"  Pass `until=` from the record, or add the task to configs/tasks.yaml. Returning "
        f"COMPLETE here would silently under-count TRUNC_STOPSEQ."
    )


def classify_termination(
    task: str,
    finish_reason: str | None,
    stop_reason: str | None,
    committed: bool,
    until: Sequence[str] | None = None,
    *,
    generation_text: str | None = None,
    until_applied: str | None = None,
) -> str:
    """Three mutually exclusive termination classes.

    * ``TRUNC_BUDGET``  -- hit the token cap. **Primary for H1**, since it is
      the quantity the budget sweep manipulates.
    * ``TRUNC_STOPSEQ`` -- killed by a harness ``until`` string with no
      committed answer. A second, distinct measurement artifact on the same
      causal pathway, and a useful **internal control**: it should NOT respond
      to the budget sweep.
    * ``COMPLETE``      -- EOS, or a stop sequence with an answer present.

    🚨 **TWO REGIMES, selected by the record's own ``until_applied`` field** (planner R34).
    The regime has to come from the record rather than from ``schema_version``, because a
    reader should not need to know which release changed the generation path.

    ``"generation"`` (every record before R34: ``pilot-v1``, ``pilot-v2``, ``template-v1``,
    ``template-v2``, and ``None``, which is treated as this regime because it is what those
    records actually did) -- ``until`` was passed to vLLM as ``stop=``, so the engine halted
    there and ``stop_reason`` names the string that fired. Read the class off ``stop_reason``.

    ``"scoring"`` -- nothing was passed as ``stop=``. The generation ran to EOS or the budget,
    so ``stop_reason`` is ``None`` or an EOS token and **never** an ``until`` string. The class
    must therefore be **DERIVED FROM THE TEXT**: if a terminator appears in
    ``generation_text``, the harness would have cut it there, so the same record that was
    ``TRUNC_STOPSEQ`` under the old regime is ``TRUNC_STOPSEQ`` under this one, and the
    published semantics are preserved. ``committed`` must already have been computed on the
    **truncated** text by the caller, via ``scoring.extract.apply_until``.

    ⚠️ Order matters and ``TRUNC_BUDGET`` is checked FIRST in both regimes, but it means
    something subtly different under ``"scoring"``: ``finish_reason == "length"`` now describes
    the FULL generation. A generation that emitted a terminator early and then ran on to the
    cap is a stop-sequence kill in published terms, not a budget truncation, so the terminator
    check has to precede the budget check in that regime. Getting this backwards would move
    records from ``TRUNC_STOPSEQ`` into ``TRUNC_BUDGET`` and corrupt §4's differential test,
    whose whole design is that the two respond differently to the budget sweep.
    """
    scoring_regime = until_applied == "scoring"

    if scoring_regime:
        # Resolve the terminator set once, with the same refuse-rather-than-assume rule as
        # the legacy branch below.
        terms = _resolve_terminators(task, until)
        if generation_text is None:
            raise UnclassifiableTerminationError(
                f"task {task!r}: until_applied='scoring' but no `generation_text` was passed. "
                f"Under that regime the termination class is DERIVED FROM THE TEXT, so "
                f"classifying without it would silently report every record COMPLETE."
            )
        _, matched = apply_until(generation_text, terms)
        # 🚨 THE REGISTERED DEFINITIONS, on the SCORED PREFIX (planner R35.Q4):
        #
        #   TRUNC_STOPSEQ  a stop string fired, with no committed answer before it
        #   TRUNC_BUDGET   NO stop string fired *and* the full generation hit the cap
        #   COMPLETE       EOS, or a stop string with an answer before it
        #
        # The ordering is not a preference, it is what keeps §4's classes meaning what they
        # were registered to mean. R34 made `finish_reason` describe the FULL generation, so
        # checking the budget first would label a generation that ran past its stop string to
        # the cap as `TRUNC_BUDGET` even though its scored prefix ended complete and early --
        # over-counting the class H1's budget sweep manipulates, and leaving H1's actual
        # estimand (was the SCORED answer cut off?) unanswerable.
        if matched is not None:
            return COMPLETE if committed else TRUNC_STOPSEQ
        if finish_reason == "length":
            # No stop string anywhere, so the scored prefix IS the full generation and the cap
            # genuinely bound it.
            return TRUNC_BUDGET
        return COMPLETE

    if finish_reason == "length":
        return TRUNC_BUDGET

    if finish_reason == "stop":
        if committed:
            return COMPLETE
        if stop_reason in _EOS_STOP_REASONS:
            # The model emitted EOS with no answer: degeneration or refusal,
            # not a harness artifact. Not our stop-sequence class.
            return COMPLETE
        # 🚨 A NUMERIC `stop_reason` IS A TOKEN ID, NOT A STOP STRING, and it gets its own
        # branch rather than falling through.
        #
        # vLLM's contract: `None` for EOS-via-eos_token_id or length, a `str` when a `stop`
        # string matched, an `int` when a stop TOKEN fired. This study passes `stop=until` and
        # NO `stop_token_ids`, so a numeric value is always an EOS-family token -- the model
        # chose to stop. Observed in real data: `32007` (Phi `<|end|>`, 1,227 records) and
        # `151643` (Qwen `<|endoftext|>`, 472).
        #
        # ⚠️ Deliberately NOT fixed by adding those two ids to `_EOS_STOP_REASONS`. Putting
        # model-specific magic numbers in a scorer is precisely how `HARNESS_UNTIL` drifted:
        # it fixes the instance and keeps the class.
        if stop_reason is not None and str(stop_reason).lstrip("-").isdigit():
            return COMPLETE

        terminators = _resolve_terminators(task, until)
        if stop_reason in terminators:
            return TRUNC_STOPSEQ
        return COMPLETE

    return COMPLETE


# ---------------------------------------------------------------------------
# H2 -- trace scan. Both variants computed, BOTH reported.
# ---------------------------------------------------------------------------

_COMMITMENT_MARKERS: tuple[str, ...] = (
    "the answer is",
    "therefore",
    "so",
    "\\boxed{",
)

# 🚨 MARKERS MATCH ON WORD BOUNDARIES. `"so"` used to be matched as a bare SUBSTRING with
# `rfind`, so it hit inside `also`, `reason`, `reasoning`, `person`, `resolve`, `absolutely`,
# `consonant` -- words that appear constantly in chain-of-thought text. An incidental hit late
# in the generation pushed the last-marker position past the gold, `after_marker` became
# `False`, and `conservative = (in_final_span or after_marker) and ...` collapsed to
# `in_final_span and ...`. Half of H2's PRE-REGISTERED conservative definition was inert, and
# the size of the under-count scales with generation length and vocabulary -- both of which
# differ BY ARM, so the bias was differential.
#
# `\boxed{` gets no trailing boundary because `{` is not a word character, and no leading one
# because `\` is not either -- `\b` there would never match.
_MARKER_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(re.escape(marker) if marker.startswith("\\")
               else rf"\b{re.escape(marker)}\b")
    for marker in _COMMITMENT_MARKERS
)


@dataclasses.dataclass(frozen=True)
class TraceScan:
    gold_in_trace: bool               # permissive variant
    gold_in_trace_conservative: bool  # conservative variant
    small_gold: bool                  # |gold| < 10 -> high incidental match rate
    matched_position: float | None = None  # relative position of the match


def scan_trace(
    generation: str,
    gold: str,
    extracted: str | None,
    final_fraction: float = 0.20,
) -> TraceScan:
    """H2: is the correct answer present in the trace but not in answer position?

    Naively searching for the gold string is broken -- gold ``18`` matches a
    trace about "18 apples" that concluded ``24``. So we compute two variants
    and report both:

    * **permissive**   -- normalised gold appears anywhere.
    * **conservative** -- normalised gold appears in the final
      ``final_fraction`` of the generation, OR after a commitment marker,
      **AND** the rung-2 extraction is not already the gold.

    ⚠️ Neither number means anything without the false-positive floor from
    ``incidental_match_floor()``. The headline H2 figure is always reported net
    of that floor, because "how much of the 52% does an extractor recover?" is
    the exact question we claim arXiv:2606.00206 did not ask -- so it has to be
    airtight.
    """
    normalised_gold = normalise_answer(gold)
    if not normalised_gold:
        return TraceScan(False, False, False, None)

    gold_number = _as_number(normalised_gold)
    small_gold = gold_number is not None and abs(gold_number) < 10

    positions = _find_gold_positions(generation, normalised_gold, gold_number)
    if not positions:
        return TraceScan(False, False, small_gold, None)

    length = max(len(generation), 1)
    last_position = positions[-1]
    relative = last_position / length

    already_extracted = (
        extracted is not None and normalise_answer(extracted) == normalised_gold
    )

    in_final_span = relative >= (1.0 - final_fraction)
    after_marker = _appears_after_commitment_marker(generation, positions)

    conservative = (in_final_span or after_marker) and not already_extracted

    return TraceScan(
        gold_in_trace=True,
        gold_in_trace_conservative=conservative,
        small_gold=small_gold,
        matched_position=relative,
    )


def _find_gold_positions(
    generation: str, normalised_gold: str, gold_number: float | None
) -> list[int]:
    """Start offsets of gold occurrences, matched on word boundaries.

    Numeric golds are matched numerically-but-literally with boundaries so that
    gold ``18`` does not match inside ``180`` or ``1.18`` -- the single largest
    source of spurious hits in this measurement.
    """
    if gold_number is not None:
        pattern = re.compile(
            r"(?<![\d.,])" + re.escape(normalised_gold) + r"(?![\d.,]*\d)"
        )
    else:
        pattern = re.compile(r"\b" + re.escape(normalised_gold) + r"\b", re.IGNORECASE)
    return [match.start() for match in pattern.finditer(generation)]


def _appears_after_commitment_marker(generation: str, positions: Sequence[int]) -> bool:
    """Does the gold appear AFTER the last commitment marker?

    ⚠️ Word-boundary matched, not substring matched. See `_MARKER_PATTERNS` for why: `"so"`
    inside `also`/`reason`/`resolve` used to set the last-marker position near the end of the
    text and make this return `False` almost always.
    """
    lowered = generation.lower()
    last_marker = -1
    for pattern in _MARKER_PATTERNS:
        # The LAST occurrence of each marker; `re` has no rfind, so scan and keep the last.
        for match in pattern.finditer(lowered):
            last_marker = max(last_marker, match.start())
    if last_marker < 0:
        return False
    return any(position > last_marker for position in positions)


def incidental_match_floor(records: Sequence[dict]) -> dict[str, float]:
    """The H2 false-positive floor. MANDATORY -- without it H2 is meaningless.

    Defined (``analysis/preregistration.md`` §8) as the rate of incidental gold
    appearance among items where **extraction succeeded and was wrong**. In
    those items the gold cannot be the model's committed answer, so any
    appearance in the trace is by construction incidental. That rate is the
    floor the headline H2 number must be reported net of.

    Each record needs ``extracted``, ``correct`` and ``gold_in_trace``
    (and optionally ``gold_in_trace_conservative``, ``small_gold``).
    """
    eligible = [
        record
        for record in records
        if record.get("extracted") is not None and not record.get("correct")
    ]
    if not eligible:
        return {
            "n_eligible": 0,
            "floor_permissive": float("nan"),
            "floor_conservative": float("nan"),
            "floor_permissive_small_gold": float("nan"),
        }

    def rate(subset: Sequence[dict], key: str) -> float:
        if not subset:
            return float("nan")
        return sum(bool(record.get(key)) for record in subset) / len(subset)

    small = [record for record in eligible if record.get("small_gold")]

    return {
        "n_eligible": len(eligible),
        "floor_permissive": rate(eligible, "gold_in_trace"),
        "floor_conservative": rate(eligible, "gold_in_trace_conservative"),
        # Reported separately because single-digit golds have a much higher
        # incidental match rate and would otherwise inflate the pooled floor.
        "n_small_gold": len(small),
        "floor_permissive_small_gold": rate(small, "gold_in_trace"),
    }
