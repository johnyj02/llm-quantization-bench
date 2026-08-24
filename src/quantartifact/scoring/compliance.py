"""Compliance labelling, termination classes, and the H2 trace scan.

PURE FUNCTIONS, NO I/O -- same contract as ``extract.py``.

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

from quantartifact.scoring.extract import (
    Extraction,
    _as_number,
    extract_permissive,
    extract_strict,
    normalise_answer,
)

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

# Harness `until` strings per task, from vendor/lm-eval-configs/ (v0.4.12).
# Needed to tell a PREMATURE stop-sequence kill from a legitimate EOS.
HARNESS_UNTIL: dict[str, tuple[str, ...]] = {
    "gsm8k": ("Question:", "</s>", "<|im_end|>"),
    "gsm8k_cot": ("Q:", "</s>", "<|im_end|>"),
    # A BARE CAPITAL "Q" and a blank line. Any chain of thought containing
    # either is cut mid-reasoning.
    "bbh": ("</s>", "Q", "\n\n"),
    # A SINGLE PERIOD, with max_gen_toks=100 and a required closing
    # "The best answer is X". A live failure mode, not a hypothetical.
    "arc_challenge_chat": ("\n\n", "."),
}

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
) -> ComplianceLabels:
    """Assign the compliance label and the termination class.

    ``finish_reason`` is REQUIRED for a meaningful result. It is emitted on
    every generation record precisely because it is unreconstructable after the
    fact, and H1 is impossible without it -- so a missing value is flagged in
    ``note`` rather than quietly defaulted.
    """
    if strict is None:
        strict = extract_strict(generation, task, subtask)
    if permissive is None:
        permissive = extract_permissive(generation, task, subtask)

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


def classify_termination(
    task: str,
    finish_reason: str | None,
    stop_reason: str | None,
    committed: bool,
) -> str:
    """Three mutually exclusive termination classes.

    * ``TRUNC_BUDGET``  -- hit the token cap. **Primary for H1**, since it is
      the quantity the budget sweep manipulates.
    * ``TRUNC_STOPSEQ`` -- killed by a harness ``until`` string with no
      committed answer. A second, distinct measurement artifact on the same
      causal pathway, and a useful **internal control**: it should NOT respond
      to the budget sweep.
    * ``COMPLETE``      -- EOS, or a stop sequence with an answer present.
    """
    if finish_reason == "length":
        return TRUNC_BUDGET

    if finish_reason == "stop":
        if committed:
            return COMPLETE
        if stop_reason in _EOS_STOP_REASONS:
            # The model emitted EOS with no answer: degeneration or refusal,
            # not a harness artifact. Not our stop-sequence class.
            return COMPLETE
        if stop_reason in HARNESS_UNTIL.get(task, ()):
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
    lowered = generation.lower()
    marker_positions = [
        lowered.rfind(marker) for marker in _COMMITMENT_MARKERS
    ]
    last_marker = max(marker_positions)
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
