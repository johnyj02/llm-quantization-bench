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
    #: 🚨 How the interval was constructed. Required in the reported table, not only in prose
    #: (R25.1): a column that silently mixes bootstrap percentile intervals with zero-event
    #: Wilson bounds is exactly what a careful reviewer flags, because the two mean different
    #: things and only one of them is an estimate of sampling variability.
    interval_method: str | None = None
    #: 🚨 The value this quantity takes under the null, in the units of `unit`. It is 0 for a
    #: difference and **1 for a ratio**, and getting it wrong manufactures a finding: an odds
    #: ratio is `exp(coefficient)` and therefore strictly positive, so a check written against 0
    #: is True for every odds ratio ever constructed regardless of whether its interval contains
    #: the null. Found by the code reviewer on 2026-09-02 and confirmed by execution: an OR of
    #: 1.01 with a 95% interval of [0.87, 1.17] reported `excludes_zero: true` in `as_dict()`.
    #: The null now travels with the unit, which is the same argument the class docstring already
    #: makes for spelling `unit` out rather than assuming it.
    null_value: float = 0.0
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
        # 🚨 The interval must CONTAIN its own point estimate. Obvious, and it was not checked:
        # the code reviewer showed on 2026-09-02 that `Effect(estimate=100.0, ci_low=-3.15,
        # ci_high=3.15)` constructed happily. That is the exact shape a naive extension of
        # `zero_variance_bound` to the all-equal-nonzero case would produce, since the bound is
        # symmetric about zero while the estimate is not. Structural version of the check, so it
        # catches the class rather than the one instance.
        # ⚠️ A tolerance is required and is NOT a loosening of the check. When every discordant
        # pair falls one way, Wilson's upper limit on the split is exactly 1.0, so the rescaled
        # bound equals the point estimate analytically, and floating point puts it a hair below:
        # this guard fired on real `pilot-v2` data with estimate 17.5 against ci_high
        # 17.499999999999996. The tolerance is scaled to the magnitudes involved, and the defect
        # it exists to catch was off by two orders of magnitude (an estimate of 100 pp inside a
        # +/-3.15 pp bound), so it remains comfortably detectable.
        span = max(abs(self.ci_low), abs(self.ci_high), abs(self.estimate), 1.0)
        tolerance = 1e-9 * span
        if not (self.ci_low - tolerance) <= self.estimate <= (self.ci_high + tolerance):
            raise ValueError(
                f"{self.name}: estimate {self.estimate} lies outside its own interval "
                f"[{self.ci_low}, {self.ci_high}]. An interval that excludes its point estimate "
                f"is not an interval for it."
            )
        # A ratio cannot be centred on a null of 0, and a difference is not usually centred on
        # 1. This catches the construction site that forgot to set `null_value` for a ratio,
        # which is how CRITICAL 2 arose in the first place.
        if "ratio" in self.unit.lower() and self.null_value == 0.0:
            raise ValueError(
                f"{self.name}: unit {self.unit!r} looks like a ratio but null_value is 0.0. "
                f"A ratio's null is 1.0; leaving it at 0 makes `excludes_null` True for every "
                f"such effect. Set null_value=1.0 explicitly."
            )

    @property
    def excludes_null(self) -> bool:
        """Whether the interval excludes `null_value`. Not a significance flag on its own.

        Deliberately named for what it describes rather than for a verdict, and deliberately
        not called `significant`. It is only ever reported next to `estimate` and the interval.

        ⚠️ Compares against `null_value`, not against 0. See that field for why: the previous
        version hard-coded 0 and was therefore unconditionally True for every odds ratio.
        """
        return self.ci_low > self.null_value or self.ci_high < self.null_value

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
            "interval_method": self.interval_method,
            "floor": self.floor,
            "null_value": self.null_value,
            "noise_class": self.noise_class.value if self.noise_class else None,
            "excludes_null": self.excludes_null,
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


def holm_bonferroni(effects: list[Effect], *, family: str,
                    alpha: float = 0.05) -> list[dict]:
    """Holm-Bonferroni step-down within one hypothesis family (§9).

    §9: *"Holm-Bonferroni within each hypothesis family (H1, H3, H4, H5, H6 are separate
    families)."* `family` is therefore **keyword-only and has no default**: pooling two families
    into one call would over-correct and splitting one across two calls would under-correct, and
    neither is visible in the output. A parameter the docstring calls required while the
    signature defaults it to `None` is the declared-versus-enforced pattern this project has paid
    for repeatedly, so the signature now enforces what §9 requires.

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


#: 🚨 R26.1. The largest absolute per-item contribution to a difference of differences, and it is
#: **DERIVED, not chosen**. This replaces the value 2 that R25.1 registered.
#:
#: `d_i = (alt_perm - alt_strict) - (ref_perm - ref_strict)`. Under the **rung monotonicity that
#: G3 already gates on** (anything rung *n* marks correct, rung *n+1* also marks correct),
#: permissive is a superset of strict, so each parenthesised gap lies in {0, +1} and never -1.
#: Therefore `d_i` lies in {-1, 0, +1} and **|d_i| <= 1**.
#:
#: So 2 described a range monotonicity **forbids**. I had registered 2 as the "conservative"
#: choice and reported that it weakened H6's contrast, on the reasoning that picking 1 because it
#: flattered the comparison would be a specification search. The planner's answer dissolves that
#: worry rather than overruling it: 1 is not the convenient option, it follows from a
#: pre-registered, gate-tested property of the scoring ladder.
#:
#: ⚠️ Used only as the FALLBACK. R26.1 registers the magnitude as the maximum |d_i| **observed
#: anywhere in the study**, which under monotonicity is 1, so the bound rests on a fact about the
#: data under a pre-registered rule rather than on an assumption in either direction. See
#: `observed_max_contribution()`.
MAX_ABS_ITEM_CONTRIBUTION: int = 1


def observed_max_contribution(contributions) -> tuple[int, str, int]:
    """The registered magnitude, its PROVENANCE, and the count of monotonicity violations.

    Returns `(magnitude, source, n_violations)` where `source` is `"observed"` or
    `"monotonicity_floor"`.

    🚨 **The provenance is returned because the magnitude is not always observed, and the one case
    where the bound is actually USED is the case where it is not.** In the all-zero (ARC) case
    `max |d_i|` is 0, the floor substitutes 1, and a field named `observed_max_abs_contribution`
    would then assert that the data supports a magnitude the data is silent on. R26.1's framing is
    that the magnitude is *observed rather than assumed*, so the artifact must distinguish the two
    or the audit trail claims something false. Found by the CODE REVIEWER, 2026-09-02.

    R26.1 registers the per-item magnitude as **the maximum absolute contribution observed
    anywhere in the study**, rather than as an assumed bound. Under the rung monotonicity G3 gates
    on, every `d_i` lies in {-1, 0, +1}, so the observed maximum is 1 and the bound is the tight
    one. If monotonicity is **violated** in the real data, `|d_i|` can reach 2, the observed
    maximum reflects that, and the bound widens to match. Either way the number is a measurement
    under a pre-specified rule rather than a choice.

    A "violation" here is any `|d_i| > 1`, which can only arise if a rung marks an item correct
    that a more permissive rung marks incorrect. That is precisely what G3's monotonicity fixture
    tests forbid, so a non-zero count is a finding about the scorers and must be reported, not
    silently absorbed into a wider interval.

    ⚠️ Monotonicity has only ever been asserted on FIXTURES. R26.1 requires asserting it on the
    real data, which is what this function makes possible at every call site that needs the bound.
    """
    import numpy as np  # noqa: PLC0415

    values = np.abs(np.asarray(contributions, dtype=float))
    if values.size == 0:
        raise ValueError(
            "observed_max_contribution needs at least one contribution; an empty array would "
            "make the magnitude vacuous and the bound arbitrarily tight."
        )
    n_violations = int((values > 1).sum())
    # ⚠️ `ceil`, not `int()`. `int()` truncates, so a float 1.9999999 arising from any future
    # non-integer contribution would report 1 and UNDERSTATE the magnitude, which is the one
    # direction of error that matters for a bound. The CODE REVIEWER asked about this directly.
    import math  # noqa: PLC0415
    observed = int(math.ceil(float(values.max())))
    if observed >= 1:
        return observed, "observed", n_violations
    # Zero events. The floor is the correct statistical behaviour, since a magnitude cannot be
    # estimated from no observations and `zero_variance_bound` would reject 0, but it is DERIVED
    # from rung monotonicity rather than measured, and the caller must be able to tell.
    return MAX_ABS_ITEM_CONTRIBUTION, "monotonicity_floor", n_violations


def zero_variance_bound(n: int, *, max_abs_contribution: int = MAX_ABS_ITEM_CONTRIBUTION,
                        alpha: float = 0.05,
                        centre: float = 0.0) -> tuple[float, float, str]:
    """A bound in percentage points when the observed variance is exactly zero (R25.1, §9h).

    Returns `(low, high, method)`, the interval centred on `centre`.

    ⚠️ **`centre` exists because the degenerate case is not only the all-zero one.** The bootstrap
    distribution is degenerate whenever every per-item contribution is **equal**, which includes a
    non-zero constant: if permissive fixes every item for the alt arm and none for bf16, every
    contribution is +1, the resample mean is always 1.0, and a percentile interval is `[100, 100]`
    pp. Verified by execution on 2026-09-02, and the most reachable version is **n = 1**, where a
    single paired item makes the bootstrap degenerate whatever the contribution is. §9h registers
    the rule for *"the observed variance ... exactly zero"*, which covers all of these, so the
    bound must be able to sit around a non-zero point estimate rather than always around zero.

    **Why a bound rather than a bootstrap.** When every per-item contribution is identically
    zero, a percentile bootstrap resamples zeros and returns `[0, 0]`. That is arithmetically
    correct and substantively misleading: with `n` items and no observed variation the honest
    claim is that the effect is **bounded near zero**, not **known to be zero**. Zero observed
    events is a standard problem with a standard answer, and a bootstrap cannot solve it because
    there is nothing to resample.

    **Construction, registered as a generic rule rather than an ARC-specific patch.** Take the
    Wilson upper limit on 0 successes out of `n`, which bounds the *rate* of items that could
    differ, then convert to accuracy points by multiplying by the largest contribution a single
    differing item can make (`MAX_ABS_ITEM_CONTRIBUTION`). At n=240 the Wilson upper limit is
    about 1.6%, giving roughly ±3.2 pp at the conservative per-item magnitude.

    ⚠️ **The bounded version is the stronger result, not a hedge.** "+3.33 pp on BBH against
    bounded within a few points on the structurally constrained task" is a real contrast that
    supports H6, whereas "[0, 0] against +3.33" invites the objection that the control is
    degenerate.

    The same rule applies everywhere observed variance is exactly zero, which is what makes it
    consistent with the zero-discordance path in `transitions.py`.
    """
    if n <= 0:
        raise ValueError(f"zero_variance_bound needs n > 0, got {n}")
    if max_abs_contribution <= 0:
        raise ValueError(
            f"max_abs_contribution must be positive, got {max_abs_contribution}")
    from statsmodels.stats.proportion import proportion_confint  # noqa: PLC0415

    _, rate_high = proportion_confint(0, n, alpha=alpha, method="wilson")
    half_width = float(rate_high) * max_abs_contribution * 100.0
    method = (f"zero-variance bound: Wilson upper on 0/{n} x max|contribution|="
              f"{max_abs_contribution} (R25.1/§9h), NOT a bootstrap interval")
    if centre:
        method += f", centred on the constant point estimate {centre:+.4g}"
    return centre - half_width, centre + half_width, method
