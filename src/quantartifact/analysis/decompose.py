"""H4, the decomposition: §1 and §4 of the pre-registration.

The identity the paper is built on, in the four-component form §4 registers as `H4'`:

    measured_strict_delta = competence_delta + compliance_delta + truncation_delta
                            + stopseq_delta

§4 adds the fourth term and pre-registers its test so it is not a post-hoc addition:

> **Pre-specified test:** is the `TRUNC_STOPSEQ` rate differential by precision (bf16 vs each
> quantized arm), tested by exact McNemar on the paired item-level indicator?
>
> **If yes**, quantized models emit bare `Q` / blank lines / premature periods more often, and
> this is a genuine compliance-delta mechanism that belongs in the decomposition table with a
> number and its own subsection. **If no**, it is a static configuration defect affecting all
> arms equally, reported as a measured rate and a limitation of the benchmark rather than as an
> effect of quantization.

⚠️ §4 also pre-commits an emphasis this module must not be used to undermine: *"H1 remains the
lead regardless of how large `TRUNC_STOPSEQ` turns out to be"*, because H1 responds to a
manipulation we control and has a dose-response across three budget levels while
`TRUNC_STOPSEQ` has neither. That was recorded specifically to stop a large stop-sequence rate
tempting a restructure of the paper around whichever number came out biggest.

⚠️ **R24.3 excludes the `gsm8k` hash-marker arm from H4's primary decomposition**, exactly as
§9c excludes it from H3: a 1 to 27 percent stratum has too few items for a competence contrast.
It is retained as the demonstration arm. `EXCLUDED_FROM_H4_PRIMARY` names it, and
`decompose()` refuses it unless the caller passes `allow_excluded=True`, so the exclusion cannot
be forgotten by omission.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .results import Effect, NoiseClass
from .stratum import ABSENT_LABEL, COMMITTED_LABELS

#: R24.3. Keyed by task name as it appears on the records.
EXCLUDED_FROM_H4_PRIMARY: frozenset[str] = frozenset({"gsm8k"})

TRUNC_BUDGET = "TRUNC_BUDGET"
TRUNC_STOPSEQ = "TRUNC_STOPSEQ"


class DecompositionError(RuntimeError):
    """Raised when the identity cannot be formed, rather than formed with a silent gap."""


@dataclass(frozen=True)
class Decomposition:
    """The four registered components for one (cell, arm pair), all in percentage points."""

    cell: tuple
    reference_arm: str
    alt_arm: str
    n: int
    #: The quantity being decomposed: strict accuracy, alt minus reference.
    measured_strict_delta: float
    #: Difference in P(correct | committed) restricted to the always-compliant stratum, so the
    #: contrast is identified per §1.1. `None` when the stratum is empty for this cell, which
    #: is a distinct answer from zero.
    competence_delta: float | None
    #: Difference in the rate of committing at all.
    compliance_delta: float
    #: Difference in the budget-truncation rate (`finish_reason == "length"`).
    truncation_delta: float
    #: Difference in the premature stop-sequence rate. §4's fourth component.
    stopseq_delta: float
    #: Items in the always-compliant stratum for this cell.
    n_stratum: int

    @property
    def residual(self) -> float | None:
        """`measured - (competence + compliance + truncation + stopseq)`.

        The identity is **conceptual, not arithmetic**: the four components are separately
        estimated quantities on overlapping item sets, so they are not guaranteed to sum to the
        measured delta. Reporting the residual rather than forcing closure is the honest choice,
        and a large residual is itself informative because it says the four named mechanisms do
        not account for the observed change.

        `None` when the competence term is unavailable, since the sum is then undefined.
        """
        if self.competence_delta is None:
            return None
        return self.measured_strict_delta - (
            self.competence_delta + self.compliance_delta
            + self.truncation_delta + self.stopseq_delta)

    def as_dict(self) -> dict:
        return {
            "cell": list(self.cell),
            "reference_arm": self.reference_arm, "alt_arm": self.alt_arm,
            "n": self.n, "n_stratum": self.n_stratum,
            "measured_strict_delta": self.measured_strict_delta,
            "competence_delta": self.competence_delta,
            "compliance_delta": self.compliance_delta,
            "truncation_delta": self.truncation_delta,
            "stopseq_delta": self.stopseq_delta,
            "residual": self.residual,
        }


def _cell_of(row: dict) -> tuple:
    return (row["model_key"], row["task"], row.get("subtask"),
            row.get("budget_level", "1x"), row.get("prompt_variant", "fewshot_cot"))


def decompose(
    rows: list[dict],
    stratum_by_cell: dict[tuple, frozenset[str]],
    *,
    rung: str = "strict",
    reference_arm: str = "bf16",
    alt_arms: tuple[str, ...] = ("gptq-8bit", "gptq-4bit", "awq-4bit", "bnb-nf4"),
    allow_excluded: bool = False,
) -> list[Decomposition]:
    """Compute the four registered components per (cell, alt arm).

    `stratum_by_cell` maps a cell tuple to its always-compliant item ids, i.e. the
    `item_ids` field of `stratum.StratumResult`. It is passed in rather than recomputed so the
    stratum used by H4 is provably the same set §1.1 reported, and cannot drift from it.

    Raises `DecompositionError` on a task excluded by R24.3 unless `allow_excluded=True`, which
    exists so the demonstration-arm reporting path can still call this deliberately.

    ⚠️ **`rung` applies to two of the four components only.** `measured_strict_delta`,
    `compliance_delta` and the competence term follow `rung`, but `truncation_delta` and
    `stopseq_delta` are read from `row["termination"]`, which §4 defines at **rung 2** regardless
    of what `rung` is set to. Calling this with `rung="permissive"` therefore changes three
    numbers and leaves two identical, which is correct and is stated here so a reader does not
    take the two unchanged values as a finding about rung-invariance.
    """
    index: dict[tuple, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row.get("repeat_idx", 0) != 0:
            continue
        index[_cell_of(row)][row["arm"]][row["item_id"]] = row

    out: list[Decomposition] = []
    for cell, by_arm in sorted(index.items(), key=lambda kv: str(kv[0])):
        task = cell[1]
        if task in EXCLUDED_FROM_H4_PRIMARY and not allow_excluded:
            raise DecompositionError(
                f"{cell}: task {task!r} is excluded from H4's primary decomposition by R24.3 "
                f"(a floored metric and a 1-27% stratum). Pass allow_excluded=True to report "
                f"it as the demonstration arm instead of the primary."
            )
        if reference_arm not in by_arm:
            raise DecompositionError(
                f"{cell}: reference arm {reference_arm!r} absent, so no delta is defined")
        if cell not in stratum_by_cell:
            raise DecompositionError(
                f"{cell}: no stratum was supplied. H4's competence term is defined on the "
                f"always-compliant stratum (§1.1), so an absent stratum is undefined rather "
                f"than empty."
            )
        stratum = stratum_by_cell[cell]
        ref_items = by_arm[reference_arm]

        for alt_arm in alt_arms:
            if alt_arm not in by_arm:
                continue
            alt_items = by_arm[alt_arm]
            shared = sorted(set(ref_items) & set(alt_items))
            if not shared:
                raise DecompositionError(
                    f"{cell} {alt_arm}: no shared items with {reference_arm}")

            def rate(items: dict, ids: list[str], predicate) -> float:
                return sum(1 for i in ids if predicate(items[i])) / len(ids) * 100.0

            def correct_at(row: dict) -> bool:
                if rung not in row["correct"]:
                    raise DecompositionError(
                        f"item {row['item_id']}: no correctness for rung {rung!r}")
                return bool(row["correct"][rung])

            def committed(row: dict) -> bool:
                label = row["compliance"][rung]
                if label in COMMITTED_LABELS:
                    return True
                if label == ABSENT_LABEL:
                    return False
                raise DecompositionError(f"unrecognised compliance label {label!r}")

            def termination_is(row: dict, wanted: str) -> bool:
                term = row.get("termination")
                if term is None:
                    raise DecompositionError(
                        f"item {row['item_id']} arm {row['arm']}: no termination class. §4 "
                        f"defines two of the four components on it, so it cannot be inferred."
                    )
                return term == wanted

            measured = (rate(alt_items, shared, correct_at)
                        - rate(ref_items, shared, correct_at))
            compliance = (rate(alt_items, shared, committed)
                          - rate(ref_items, shared, committed))
            truncation = (rate(alt_items, shared, lambda r: termination_is(r, TRUNC_BUDGET))
                          - rate(ref_items, shared, lambda r: termination_is(r, TRUNC_BUDGET)))
            stopseq = (rate(alt_items, shared, lambda r: termination_is(r, TRUNC_STOPSEQ))
                       - rate(ref_items, shared, lambda r: termination_is(r, TRUNC_STOPSEQ)))

            in_stratum = [i for i in shared if i in stratum]
            competence = None
            if in_stratum:
                competence = (rate(alt_items, in_stratum, correct_at)
                              - rate(ref_items, in_stratum, correct_at))

            out.append(Decomposition(
                cell=cell, reference_arm=reference_arm, alt_arm=alt_arm,
                n=len(shared), n_stratum=len(in_stratum),
                measured_strict_delta=measured,
                competence_delta=competence,
                compliance_delta=compliance,
                truncation_delta=truncation,
                stopseq_delta=stopseq,
            ))
    return out


def stopseq_differential(
    rows: list[dict],
    *,
    reference_arm: str = "bf16",
    alt_arms: tuple[str, ...] = ("gptq-8bit", "gptq-4bit", "awq-4bit", "bnb-nf4"),
) -> list[Effect]:
    """§4's pre-specified test: is `TRUNC_STOPSEQ` differential by precision?

    Exact McNemar on the paired item-level indicator, per arm pair per cell. A positive
    estimate means the quantized arm was stop-sequence killed more often than bf16, which is
    the direction that would make this a genuine compliance mechanism rather than a static
    configuration defect.

    🚨 **There is deliberately no `rung` parameter.** `TRUNC_STOPSEQ` is defined at **rung 2** by
    registration: §4 defines it as a stop-sequence match where *"rung-2 extraction finds no
    committed answer"*, and `classify_termination` is called with `committed` taken from the
    permissive extraction. So the class does not vary by rung, and this function reads
    `row["termination"]` directly.

    An earlier version accepted `rung` and used it only as a context label. Calling it with
    `rung="permissive"` therefore returned numbers **identical** to the strict call, labelled
    differently, and two rows in a pre-registered table with the same values and different labels
    would read as evidence that the differential is rung-invariant when it had never been computed
    per rung. That is the charter's "parameter accepted then ignored" pattern, and here it could
    have put a duplicated number into a registered table. Removed rather than wired up, because
    wiring it up would invent a quantity the registration does not define.

    The verdict itself is deliberately not returned. §4 specifies what each branch *means*, and
    that reading belongs in the paper next to the effect size and its interval, not in a boolean
    this function hands back.
    """
    from statsmodels.stats.proportion import proportion_confint  # noqa: PLC0415

    from .transitions import mcnemar_exact  # noqa: PLC0415

    index: dict[tuple, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row.get("repeat_idx", 0) != 0:
            continue
        index[_cell_of(row)][row["arm"]][row["item_id"]] = row

    # 🚨 These raise rather than `continue`, matching `decompose()`. An earlier version skipped a
    # missing reference arm and an empty intersection, which meant a failed cell in `main-v1`
    # dropped silently out of §4's PRE-SPECIFIED test, shrinking the family handed to
    # `holm_bonferroni` and therefore UNDER-correcting the tests that remain, which makes a
    # differential easier to declare. A silent `continue` on the path into a multiplicity
    # correction is the same shape as `score_template.py`'s vacuous BRANCH 1.
    #
    # An arm genuinely absent from a condition (the 3-arm sweep cells) is still skipped, because
    # that is a design fact rather than a failure, and it is the only skip left.
    effects: list[Effect] = []
    for cell, by_arm in sorted(index.items(), key=lambda kv: str(kv[0])):
        if reference_arm not in by_arm:
            raise DecompositionError(
                f"{cell}: reference arm {reference_arm!r} produced no rows, so §4's paired "
                f"differentiality test is undefined here. Score it or exclude the cell "
                f"explicitly; skipping it silently under-corrects the rest of the family."
            )
        ref_items = by_arm[reference_arm]
        for alt_arm in alt_arms:
            if alt_arm not in by_arm:
                continue
            alt_items = by_arm[alt_arm]
            shared = sorted(set(ref_items) & set(alt_items))
            if not shared:
                raise DecompositionError(
                    f"{cell} {alt_arm} vs {reference_arm}: no shared items, so the paired "
                    f"indicator has nothing to pair"
                )

            def killed(row: dict) -> bool:
                term = row.get("termination")
                if term is None:
                    raise DecompositionError(
                        f"item {row['item_id']} arm {row['arm']}: no termination class")
                return term == TRUNC_STOPSEQ

            # b: killed under alt only. c: killed under reference only.
            b = sum(1 for i in shared if killed(alt_items[i]) and not killed(ref_items[i]))
            c = sum(1 for i in shared if killed(ref_items[i]) and not killed(alt_items[i]))
            n = len(shared)
            estimate = (b - c) / n * 100.0
            p_value, test = mcnemar_exact(b, c)

            if b + c == 0:
                low, high = proportion_confint(0, n, alpha=0.05, method="wilson")
                ci_low, ci_high = -float(high) * 100.0, float(high) * 100.0
            else:
                low, high = proportion_confint(b, b + c, alpha=0.05, method="wilson")
                scale = (b + c) / n * 100.0
                ci_low = (2.0 * float(low) - 1.0) * scale
                ci_high = (2.0 * float(high) - 1.0) * scale

            effects.append(Effect(
                name=f"stopseq_differential[{alt_arm} vs {reference_arm}]",
                estimate=estimate, ci_low=ci_low, ci_high=ci_high,
                unit="pp", n=n, p_value=p_value, test=test,
                noise_class=NoiseClass.COMPOUNDING,
                context={"cell": list(cell),
                         "termination_defined_at_rung": "permissive (rung 2), per §4",
                         "n_killed_alt_only": b, "n_killed_reference_only": c,
                         "registered_reading":
                             "§4: differential => a genuine compliance-delta mechanism with "
                             "its own subsection; not differential => a static configuration "
                             "defect affecting all arms equally, reported as a rate and a "
                             "benchmark limitation. H1 remains the lead either way."},
            ))
    return effects
