"""Manski bounds on the full-sample competence effect: §1.3 of the pre-registration.

§1.1 restricts the primary competence contrast to the always-compliant stratum, which buys
identification at the cost of generality. §1.3 buys the generality back, honestly: it reports
what the data can and cannot rule out **without** the stratum restriction, by bounding the
unobserved outcomes rather than imputing them.

Construction, as registered: *"for items `ABSENT` under an arm, competence is unobserved. Bound
it by imputing all such items **correct** (upper) and all **incorrect** (lower), for each arm
independently, and propagate to the difference."*

Propagating to a **difference** is where this is easy to get wrong, and the error is not
visible in the output. The widest difference is not (upper minus upper); it is:

    delta_upper = alt_upper - ref_lower
    delta_lower = alt_lower - ref_upper

because the bounds on the two arms are attained independently. Taking `alt_upper - ref_upper`
would report an interval narrower than the data supports, which is the one direction of error
that matters here: §1.3's whole purpose is to be honest about what is not identified.

R24.3 changes the status of this function rather than its mathematics. The 50% stratum
threshold is dissolved, so these bounds are no longer a fallback that fires for weak cells;
they are reported **for every cell, always**, next to the point estimate, with stratum size as
a first-class number. §1.3 pre-commits to *"reporting the bounds even when they are
uninformative, and to not substituting a narrower but unidentified estimate in the abstract."*
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .results import Effect, NoiseClass
from .stratum import ABSENT_LABEL, COMMITTED_LABELS


class BoundsError(RuntimeError):
    """Raised when bounds cannot be constructed, rather than constructed on partial data."""


@dataclass(frozen=True)
class ArmBounds:
    """Worst-case and best-case competence for one arm on one cell."""

    arm: str
    #: Items committed under this arm, i.e. where correctness is observed.
    n_committed: int
    #: Items ABSENT under this arm, i.e. where competence is unobserved.
    n_absent: int
    #: Correct among the committed.
    n_correct: int

    @property
    def n(self) -> int:
        return self.n_committed + self.n_absent

    @property
    def lower(self) -> float:
        """All absent items imputed INCORRECT."""
        if self.n == 0:
            raise BoundsError(f"{self.arm}: no items, so no bound is defined")
        return self.n_correct / self.n

    @property
    def upper(self) -> float:
        """All absent items imputed CORRECT."""
        if self.n == 0:
            raise BoundsError(f"{self.arm}: no items, so no bound is defined")
        return (self.n_correct + self.n_absent) / self.n

    @property
    def width(self) -> float:
        """Bound width, which equals the absent fraction. Uninformative bounds are wide."""
        return self.upper - self.lower

    def as_dict(self) -> dict:
        return {"arm": self.arm, "n": self.n, "n_committed": self.n_committed,
                "n_absent": self.n_absent, "n_correct": self.n_correct,
                "lower": self.lower, "upper": self.upper, "width": self.width}


@dataclass(frozen=True)
class DifferenceBounds:
    """Manski bounds on the competence DIFFERENCE between two arms on one cell."""

    cell: tuple
    rung: str
    reference_arm: str
    alt_arm: str
    reference: ArmBounds
    alt: ArmBounds
    #: Items excluded because one arm had no record for them.
    n_unpaired: int

    @property
    def lower(self) -> float:
        """Widest plausible LOW value of (alt - reference).

        Attained when the alt arm is at its worst and the reference at its best, because the
        two arms' unobserved outcomes are bounded independently.
        """
        return self.alt.lower - self.reference.upper

    @property
    def upper(self) -> float:
        """Widest plausible HIGH value of (alt - reference)."""
        return self.alt.upper - self.reference.lower

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def excludes_zero(self) -> bool:
        """Whether the bounds themselves rule out a null effect.

        Rarely true, and that is the honest result §1.3 anticipates. When it IS true the claim
        is strong, because it holds under every possible imputation of the unobserved items.
        """
        return self.lower > 0.0 or self.upper < 0.0

    def as_dict(self) -> dict:
        return {
            "cell": list(self.cell), "rung": self.rung,
            "reference_arm": self.reference_arm, "alt_arm": self.alt_arm,
            "lower": self.lower, "upper": self.upper, "width": self.width,
            "excludes_zero": self.excludes_zero, "n_unpaired": self.n_unpaired,
            "reference": self.reference.as_dict(), "alt": self.alt.as_dict(),
        }


def _arm_bounds(arm: str, items: dict[str, tuple[bool, bool]]) -> ArmBounds:
    """`items` maps item_id -> (committed, correct)."""
    n_committed = sum(1 for committed, _ in items.values() if committed)
    n_absent = len(items) - n_committed
    n_correct = sum(1 for committed, correct in items.values() if committed and correct)
    return ArmBounds(arm=arm, n_committed=n_committed, n_absent=n_absent,
                     n_correct=n_correct)


def compute_bounds(
    rows: list[dict],
    *,
    rung: str,
    reference_arm: str = "bf16",
    alt_arms: tuple[str, ...] = ("gptq-8bit", "gptq-4bit", "awq-4bit", "bnb-nf4"),
) -> list[DifferenceBounds]:
    """Manski bounds per (cell, alt arm) at one rung.

    Rows carry `compliance` (rung -> label) and `correct` (rung -> bool), matching what the
    scorers already emit. Only `repeat_idx == 0` is used, for the same reason as the transition
    tables: the bound is defined on one measurement per item per arm.

    ⚠️ An item that is ABSENT contributes to `n_absent` **regardless of its `correct` flag**.
    A scorer may still record `correct=False` for an absent item, since no answer cannot match
    the gold, and treating that False as an observed incorrect answer would silently collapse
    the upper bound onto the point estimate and make the bounds look far more informative than
    they are.
    """
    index: dict[tuple, dict[str, dict[str, tuple[bool, bool]]]] = defaultdict(
        lambda: defaultdict(dict))
    for row in rows:
        if row.get("repeat_idx", 0) != 0:
            continue
        if rung not in row["compliance"]:
            raise BoundsError(
                f"item {row['item_id']} arm {row['arm']}: no compliance label for rung {rung!r}"
            )
        if rung not in row["correct"]:
            raise BoundsError(
                f"item {row['item_id']} arm {row['arm']}: no correctness for rung {rung!r}"
            )
        label = row["compliance"][rung]
        if label in COMMITTED_LABELS:
            committed = True
        elif label == ABSENT_LABEL:
            committed = False
        else:
            raise BoundsError(
                f"unrecognised compliance label {label!r}; bounds are defined on "
                f"committed versus absent only"
            )
        cell = (row["model_key"], row["task"], row.get("subtask"),
                row.get("budget_level", "1x"), row.get("prompt_variant", "fewshot_cot"))
        index[cell][row["arm"]][row["item_id"]] = (committed, bool(row["correct"][rung]))

    out: list[DifferenceBounds] = []
    for cell, by_arm in sorted(index.items(), key=lambda kv: str(kv[0])):
        if reference_arm not in by_arm:
            raise BoundsError(
                f"{cell}: reference arm {reference_arm!r} absent, so no difference bound is "
                f"defined"
            )
        for alt_arm in alt_arms:
            if alt_arm not in by_arm:
                continue
            ref_items, alt_items = by_arm[reference_arm], by_arm[alt_arm]
            shared = set(ref_items) & set(alt_items)
            if not shared:
                raise BoundsError(
                    f"{cell} {alt_arm} vs {reference_arm}: no shared items, so the paired "
                    f"difference is undefined"
                )
            out.append(DifferenceBounds(
                cell=cell, rung=rung,
                reference_arm=reference_arm, alt_arm=alt_arm,
                reference=_arm_bounds(reference_arm,
                                      {i: ref_items[i] for i in shared}),
                alt=_arm_bounds(alt_arm, {i: alt_items[i] for i in shared}),
                n_unpaired=len(set(ref_items) ^ set(alt_items)),
            ))
    return out


def bounds_effects(bounds: list[DifferenceBounds]) -> list[Effect]:
    """Represent each bound as an `Effect` whose interval IS the bound.

    The midpoint is used as `estimate` purely so the type is satisfiable, and it is labelled
    `manski_midpoint` in the name so nobody mistakes it for a point estimate. §1.3 forbids
    substituting a narrower unidentified estimate, and a midpoint is not an estimator of
    anything; the interval is the result.
    """
    effects: list[Effect] = []
    for bound in bounds:
        effects.append(Effect(
            name=f"manski_midpoint[{bound.alt_arm} vs {bound.reference_arm}]",
            estimate=(bound.lower + bound.upper) / 2.0 * 100.0,
            ci_low=bound.lower * 100.0, ci_high=bound.upper * 100.0,
            unit="pp (Manski bound, NOT a confidence interval)",
            n=bound.reference.n,
            noise_class=NoiseClass.COMPOUNDING,
            context={"cell": list(bound.cell), "rung": bound.rung,
                     "width_pp": bound.width * 100.0,
                     "absent_fraction_reference": bound.reference.width,
                     "absent_fraction_alt": bound.alt.width,
                     "n_unpaired": bound.n_unpaired},
        ))
    return effects
