"""The result types every analysis in this package returns.

`analysis/preregistration.md` §9 requires *"effect sizes with CIs for everything"* and
**"never a bare significance flag"**. That is enforced here structurally rather than by
discipline: `Effect` cannot be constructed without both an estimate and an interval, and it
has no attribute that reports significance on its own. A caller that wants to know whether an
effect is distinguishable from zero has to look at the interval or the p-value together with
the estimate, which is the reporting behaviour the pre-registration asks for.

Two further rules from the pre-registration are encoded in this module because they are easy
to state and easy to forget at the call site:

* §9's **noise-floor pre-commitment**: *"any effect smaller than the measured noise floor will
  not be claimed by us either."* `Effect.floor` records the floor an effect was compared
  against and `Effect.claimable` applies it. An effect below its floor is not an error and is
  not hidden; it is reported with `claimable=False` so the paper can state the measurement and
  decline the claim.
* §9f's **three invocation-noise classes**. A class-3 quantity (anything contrasting two arms)
  carries a larger floor than a class-1 quantity (re-scoring one fixed generation set), so the
  class travels with the effect rather than being remembered separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


# §9f. Which floor an estimand binds to depends on what crosses an invocation boundary.
class NoiseClass(Enum):
    """Invocation-noise class, per `analysis/preregistration.md` §9f."""

    #: Computed by re-scoring ONE fixed set of generations: rung monotonicity, compliance
    #: labels, the trace scan, per-item rung agreement. No invocation boundary is crossed,
    #: so the floor is the repetition floor (measured 0.000%).
    NONE = "none"
    #: A within-arm difference of two rungs, i.e. the strict-permissive gap. The two rungs
    #: score the same text, so the noise is common and largely cancels.
    PARTIALLY_CANCELLING = "partially_cancelling"
    #: Anything contrasting arms: precision main effects, the competence contrast, the Manski
    #: bounds, H4's components, and H3's interaction. Two independent invocations, so the
    #: variances add and the floor carries a factor of sqrt(2).
    COMPOUNDING = "compounding"


@dataclass(frozen=True)
class Effect:
    """One estimated quantity, with its interval. Both are mandatory.

    `estimate`, `ci_low` and `ci_high` are in the units named by `unit`, which is spelled out
    rather than assumed because the pre-registration asks for the H3 interaction in **two**
    units (an odds ratio and accuracy points) and mixing them silently would be a wrong number
    that looks plausible.
    """

    name: str
    estimate: float
    ci_low: float
    ci_high: float
    unit: str
    n: int
    #: Two-sided p-value where a test was run. Optional, and deliberately not sufficient on
    #: its own: nothing in this package branches on `p_value` without also reporting `estimate`.
    p_value: float | None = None
    #: The test that produced `p_value`, so a reader never has to infer it.
    test: str | None = None
    #: §9's noise floor in the units of `unit`, if one applies to this quantity.
    floor: float | None = None
    noise_class: NoiseClass | None = None
    #: Free-form provenance: which cells, which rungs, which model.
    context: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ci_low > self.ci_high:
            raise ValueError(
                f"{self.name}: ci_low {self.ci_low} exceeds ci_high {self.ci_high}"
            )
        if self.n < 0:
            raise ValueError(f"{self.name}: n is negative ({self.n})")
        for label, value in (("estimate", self.estimate), ("ci_low", self.ci_low),
                             ("ci_high", self.ci_high)):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                raise ValueError(
                    f"{self.name}: {label} is not a number. An effect without an interval is "
                    f"exactly what §9 forbids reporting, so it cannot be constructed."
                )
        if self.p_value is not None and not 0.0 <= self.p_value <= 1.0:
            raise ValueError(f"{self.name}: p_value {self.p_value} outside [0, 1]")
        if self.p_value is not None and self.test is None:
            raise ValueError(
                f"{self.name}: a p_value was given with no `test` named. An unnamed p-value "
                f"is a bare significance flag in disguise."
            )
        if self.floor is not None and self.floor < 0:
            raise ValueError(f"{self.name}: floor {self.floor} is negative")

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval excludes zero. Not a significance flag on its own.

        Deliberately named for what it describes rather than for a verdict, and deliberately
        not called `significant`. It is only ever reported next to `estimate` and the interval.
        """
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def claimable(self) -> bool | None:
        """§9's pre-commitment: is `|estimate|` at or above the floor it binds to?

        Returns `None` when no floor has been attached, which is a distinct answer from
        `False`. "We have not measured the floor for this quantity" and "this effect is
        smaller than its floor" must not collapse into one value; that conflation is the shape
        of several bugs this project has already paid for.
        """
        if self.floor is None:
            return None
        return abs(self.estimate) >= self.floor

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "unit": self.unit,
            "n": self.n,
            "p_value": self.p_value,
            "test": self.test,
            "floor": self.floor,
            "noise_class": self.noise_class.value if self.noise_class else None,
            "excludes_zero": self.excludes_zero,
            "claimable": self.claimable,
            "context": dict(self.context),
        }

    def describe(self) -> str:
        """One line carrying the estimate, the interval and the floor verdict together."""
        text = (f"{self.name}: {self.estimate:+.4g} {self.unit} "
                f"[{self.ci_low:+.4g}, {self.ci_high:+.4g}] n={self.n}")
        if self.p_value is not None:
            text += f" {self.test} p={self.p_value:.4g}"
        if self.floor is not None:
            verdict = "AT/ABOVE floor" if self.claimable else "BELOW FLOOR, not claimed"
            text += f" (floor {self.floor:.4g}: {verdict})"
        return text


def holm_bonferroni(effects: list[Effect], alpha: float = 0.05,
                    family: str | None = None) -> list[dict]:
    """Holm-Bonferroni step-down within one hypothesis family (§9).

    §9: *"Holm-Bonferroni within each hypothesis family (H1, H3, H4, H5, H6 are separate
    families)."* The family is therefore a required part of the call in practice and is
    recorded on every row, because pooling two families would silently over-correct and
    splitting one would under-correct, and neither is visible in the output otherwise.

    Returns one row per input effect, **in the input order**, each carrying its adjusted
    threshold and whether it was rejected. Effects with no p-value are passed through with
    `rejected=None`: an estimate with an interval and no test is a legitimate result here
    (§9 asks for effect sizes everywhere, tests only where specified), and dropping it would
    silently shrink the family and weaken the correction for everything else.

    ⚠️ The correction is applied over the effects that HAVE p-values, and `m` is reported so
    a reader can check the family size rather than trust it.
    """
    if not effects:
        raise ValueError(
            "holm_bonferroni received no effects. An empty family cannot be corrected, and "
            "returning an empty list would let an upstream skip read as a completed "
            "correction. See notes/reviews/ for the score_template.py instance of this."
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    tested = [(i, e) for i, e in enumerate(effects) if e.p_value is not None]
    m = len(tested)
    rows: list[dict] = [
        {"name": e.name, "family": family, "p_value": e.p_value, "m": m,
         "adjusted_alpha": None, "rejected": None, "effect": e}
        for e in effects
    ]
    if m == 0:
        return rows

    # Step-down: sort ascending by p, compare p_(k) against alpha / (m - k), and stop
    # rejecting at the first failure. Everything after a failure is retained regardless of
    # its own p-value, which is the part a naive per-test comparison gets wrong.
    # `tested` is filtered on `p_value is not None`, so the key cannot be None. Spelled with
    # an explicit float() so a static reader sees the invariant rather than inferring it.
    order = sorted(tested, key=lambda pair: float(pair[1].p_value or 0.0))
    still_rejecting = True
    for k, (index, effect) in enumerate(order):
        adjusted = alpha / (m - k)
        rows[index]["adjusted_alpha"] = adjusted
        p_value = float(effect.p_value or 0.0)
        if still_rejecting and p_value <= adjusted:
            rows[index]["rejected"] = True
        else:
            still_rejecting = False
            rows[index]["rejected"] = False
    return rows
