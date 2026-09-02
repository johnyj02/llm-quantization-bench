"""H3, the precision x scoring interaction: §2 of the pre-registration.

§2 registers the model, and it is the paper's headline quantity:

    correct ~ precision * scoring + model + (1 | item)          # BBH, ARC
    correct ~ precision * scoring * answer_marker + model + ... # GSM8K, three-way

with `model` as a **fixed** effect because three levels cannot support a variance estimate, and
item as the only random effect. The interaction is to be reported **as an odds ratio and as a
marginal effect in accuracy points with a 95% CI**, the headline contrast is rung 1 (`strict`)
versus rung 2 (`permissive`), and **McNemar on discordant pairs** is the registered
assumption-light companion.

§2 also pre-commits the interpretation, which this module must not undermine: *"a null
interaction is a publishable, confirmatory result and will be reported as one ... It will not be
reported as a failed experiment."* Nothing here returns a verdict; it returns estimates with
intervals, and `Effect` has no `significant` attribute by design.

---

## 🚨 The estimator question, stated rather than hidden

**`statsmodels` has no frequentist logistic mixed model.** Its options are
`BinomialBayesMixedGLM` (variational Bayes, giving posterior means and SDs) and `GEE` with an
exchangeable working correlation (population-averaged, cluster-robust). Neither is the
maximum-likelihood conditional GLMM that `(1 | item)` conventionally denotes in `lme4`.

Rather than pick one silently, this module computes the interaction **two ways** and labels both:

1. `interaction_marginal()` is the **primary** estimator for the accuracy-points quantity. It
   takes the difference of differences directly and gets its interval from a **cluster bootstrap
   over items**. That is not a workaround; for the headline contrast it is arguably the better
   instrument, because §2 itself notes the extraction axis re-scores *identical generations*, so
   strict-versus-permissive is perfectly paired at the generation level, and resampling items
   respects exactly the clustering the `(1 | item)` term exists to absorb. It assumes nothing
   about a link function.
2. `interaction_glmm()` fits the registered formula to obtain the **odds ratio**, and records
   which fitter produced it in the returned context so a reader is never misled about the
   inference.

⚠️ **Whether the registered `(1 | item)` is satisfied by a variational-Bayes fit is a science
question, not mine.** It is flagged for the planner. It does not block the marginal estimate,
the McNemar companion, or anything else in this package.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .results import Effect, NoiseClass, zero_variance_bound

#: §2's headline contrast, fixed here so no caller can quietly choose a flattering pair.
HEADLINE_RUNGS: tuple[str, str] = ("strict", "permissive")

#: §9c excludes the `gsm8k` hash-marker arm from the H3 interaction, and R24.3 describes its own
#: H4 exclusion as *"exactly parallel"* to it. `decompose.py` enforced its side with a refusal
#: while this side enforced nothing, so an interaction on a metric floored at 1.0% strict could be
#: produced, enter the Holm family, and inflate `m` for the legitimate tests. Now symmetrical.
EXCLUDED_FROM_H3_INTERACTION: frozenset[str] = frozenset({"gsm8k"})


def _item_key(row: dict) -> str:
    """The identity of one evaluation item, at the scope `records.JOIN_KEY` gives it.

    🚨 `item_id` alone is NOT unique. `materialize._doc_id` derives it from the dataset document
    (a natural `id`/`idx`/`example_id`, else `f"idx-{index}"`), so it is unique only within a
    `(model, task, subtask)`. Three models score the identical `item_id` set, and all six BBH
    subtasks restart at `idx-0`.

    Keying on `(arm, item_id)` therefore collapsed three models and six subtasks into one
    namespace, last-writer-wins, and the difference of differences was computed over an arbitrary
    mixture determined by row order. Confirmed by execution on 2026-09-02: a fixture with two
    models x two subtasks sharing `idx-0` reported **n=1** where four items existed, while
    `n_items_dropped_unpaired` read **0**, because the collision happens before the pairing check
    and so bypasses the one safeguard that would have flagged it.

    Pooling ACROSS models is intended: §2 registers the model fit per task with `model` as a
    fixed effect. The bug was never the pooling, it was that the key did not match the data.

    Hoisted into one function used by both estimators, because `interaction_glmm` built this key
    correctly twenty lines below while `_collect` did not, and two copies of an identity are two
    chances to disagree.
    """
    return f"{row['model_key']}|{row['task']}|{row.get('subtask')}|{row['item_id']}"


class InteractionError(RuntimeError):
    """Raised when the interaction is not estimable, rather than estimated on partial data."""


@dataclass(frozen=True)
class PairedCell:
    """Item-level correctness for one arm at two rungs, on a shared item set."""

    arm: str
    #: item_id -> correctness at the low rung (strict).
    low: dict[str, bool]
    #: item_id -> correctness at the high rung (permissive).
    high: dict[str, bool]

    @property
    def items(self) -> set[str]:
        return set(self.low) & set(self.high)

    def gap(self, items: list[str]) -> float:
        """(high - low) accuracy over `items`, in percentage points.

        This is the within-arm rung gap, which §9f classifies as noise class 2: the two rungs
        score the same text, so invocation noise is common to both and largely cancels.
        """
        if not items:
            raise InteractionError(f"{self.arm}: gap over an empty item list is undefined")
        high = sum(1 for i in items if self.high[i]) / len(items)
        low = sum(1 for i in items if self.low[i]) / len(items)
        return (high - low) * 100.0


def _collect(rows: list[dict], task: str, rungs: tuple[str, str]) -> dict[str, PairedCell]:
    low_rung, high_rung = rungs
    low: dict[str, dict[str, bool]] = defaultdict(dict)
    high: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        if row["task"] != task or row.get("repeat_idx", 0) != 0:
            continue
        correct = row["correct"]
        for rung, sink in ((low_rung, low), (high_rung, high)):
            if rung not in correct:
                raise InteractionError(
                    f"{task} item {row['item_id']} arm {row['arm']}: no correctness for rung "
                    f"{rung!r}. The interaction is a difference of differences, so a missing "
                    f"rung makes it undefined rather than zero."
                )
            sink[row["arm"]][_item_key(row)] = bool(correct[rung])
    return {arm: PairedCell(arm=arm, low=low[arm], high=high[arm]) for arm in low}


def interaction_marginal(
    rows: list[dict],
    *,
    task: str,
    reference_arm: str = "bf16",
    alt_arm: str,
    rungs: tuple[str, str] = HEADLINE_RUNGS,
    n_boot: int = 10_000,
    seed: int = 20260901,
    allow_excluded: bool = False,
) -> Effect:
    """The interaction as a marginal effect in accuracy points, with a cluster-bootstrap CI.

    The estimand is the difference of differences:

        interaction = [acc(alt, high) - acc(alt, low)] - [acc(ref, high) - acc(ref, low)]

    i.e. how much more (or less) the permissive rung buys the quantized arm than it buys bf16.
    A positive value means leniency recovers more for the quantized arm, which is the direction
    the compliance mechanism predicts.

    The interval comes from resampling **items** with replacement, keeping all four
    (arm, rung) observations of a sampled item together. That respects both the pairing across
    rungs and the clustering within items. `n_boot` is 10,000 by default, which puts the Monte
    Carlo error on a 95% percentile interval well below the reporting precision.

    ⚠️ Items are restricted to those present in **all four** (arm, rung) combinations, and the
    number dropped is reported. A difference of differences computed over four different item
    sets is a wrong number that no test would catch.
    """
    import numpy as np  # noqa: PLC0415

    if task in EXCLUDED_FROM_H3_INTERACTION and not allow_excluded:
        raise InteractionError(
            f"task {task!r} is EXCLUDED from the H3 interaction by §9c: a metric floored at "
            f"1.0% strict has no variance for an interaction to appear in, and `gsm8k_cot` "
            f"carries the primary interaction. Pass allow_excluded=True only to report it as "
            f"the demonstration arm, and keep it out of the Holm family either way."
        )

    cells = _collect(rows, task, rungs)
    for arm in (reference_arm, alt_arm):
        if arm not in cells:
            raise InteractionError(
                f"{task}: arm {arm!r} has no rows, so the interaction is not estimable"
            )
    ref, alt = cells[reference_arm], cells[alt_arm]
    shared = sorted(ref.items & alt.items)
    if not shared:
        raise InteractionError(
            f"{task}: {alt_arm} and {reference_arm} share no items at rungs {rungs}, so the "
            f"paired difference of differences is undefined"
        )
    dropped = len((ref.items | alt.items) - set(shared))

    point = alt.gap(shared) - ref.gap(shared)

    # Per-item contribution, so the bootstrap is a mean over items rather than four separate
    # rate computations. For item i the contribution is
    #   (alt_high - alt_low) - (ref_high - ref_low)
    # each term 0 or 1, so the contribution is an integer in [-2, 2] and its mean over items is
    # exactly the difference of differences.
    contrib = np.array(
        [(int(alt.high[i]) - int(alt.low[i])) - (int(ref.high[i]) - int(ref.low[i]))
         for i in shared], dtype=float)
    assert abs(contrib.mean() * 100.0 - point) < 1e-9, "contribution decomposition disagrees"

    n = len(shared)
    if not contrib.any():
        # 🚨 R25.1. Every per-item contribution is identically zero, so there is nothing for the
        # bootstrap to resample and it would return [0, 0]: arithmetically correct and
        # substantively misleading, because with n items and no observed variation the effect is
        # BOUNDED near zero rather than known to be zero. Measured on `arc_challenge_chat`, where
        # strict and permissive agree on every item, so all four arms reported [0.00, 0.00].
        #
        # The planner ruled the bounded form is the STRONGER result, not a hedge: "+3.33 pp on BBH
        # against bounded within a few points on the structurally constrained task" supports H6,
        # whereas "[0, 0] against +3.33" invites the objection that the control is degenerate.
        ci_low, ci_high, interval_method = zero_variance_bound(n)
    else:
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, n, size=(n_boot, n))
        boot = contrib[draws].mean(axis=1) * 100.0
        ci_low, ci_high = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
        interval_method = f"cluster bootstrap over items, {n_boot} percentile draws"

    # McNemar on the discordant items: an item is discordant when the two arms differ in whether
    # the permissive rung changed their correctness. This is the test appropriate to an
    # INTERACTION, since the interaction is itself an arm-level contrast.
    #
    # ⚠️ It is NOT the companion §2 registers. §2 registers McNemar on **strict versus
    # permissive**, which is a within-arm test, on the grounds that the extraction axis re-scores
    # identical generations and is therefore perfectly paired at the generation level. Both are
    # worth having and they answer different questions, so the test is named for the comparison
    # it ran rather than left as a bare "mcnemar_exact" that a reader would take for the
    # registered one. `rung_mcnemar()` below returns the registered companion.
    alt_changed = {i for i in shared if alt.high[i] != alt.low[i]}
    ref_changed = {i for i in shared if ref.high[i] != ref.low[i]}
    from .transitions import mcnemar_exact  # noqa: PLC0415
    p_value, _ = mcnemar_exact(len(alt_changed - ref_changed),
                               len(ref_changed - alt_changed))
    test = "mcnemar_exact[interaction: arm contrast of rung-change indicator]"

    return Effect(
        name=f"H3_interaction_marginal[{alt_arm} vs {reference_arm}]",
        estimate=point, ci_low=ci_low, ci_high=ci_high,
        unit="pp", n=n, p_value=p_value, test=test,
        interval_method=interval_method,
        noise_class=NoiseClass.COMPOUNDING,
        context={
            "task": task, "rungs": list(rungs),
            "zero_variance": not bool(contrib.any()),
            "estimator": "difference of differences, cluster bootstrap over items",
            "n_boot": n_boot, "seed": seed, "n_items_dropped_unpaired": dropped,
            "gap_alt_pp": alt.gap(shared), "gap_reference_pp": ref.gap(shared),
            "n_changed_alt": len(alt_changed), "n_changed_reference": len(ref_changed),
        },
    )


def interaction_glmm(
    rows: list[dict],
    *,
    task: str,
    rungs: tuple[str, str] = HEADLINE_RUNGS,
    fitter: str = "gee",
) -> list[Effect]:
    """Fit §2's registered formula and return the interaction terms as odds ratios.

    🚨 **`fitter` defaults to `"gee"` per R25.2(b)**, and the choice is not arbitrary. GEE is
    population-averaged, which **matches the marginal estimand R25.2(a) made primary**, so the
    odds ratio and the accuracy-point effect describe the same contrast on two scales and their
    cluster-robust standard errors respect the same clustering as the bootstrap. VB's interval is
    anti-conservative (variational Bayes understates posterior variance), and presenting an
    optimistic interval where a reader expects a confidence interval is the worse error; VB also
    yields a **conditional** odds ratio, which is numerically larger, so a reader comparing it
    against the marginal effect would find them inconsistent without knowing why.

    VB is retained for the appendix as a robustness fit that demonstrates the registered model
    was actually fitted, explicitly labelled as a posterior interval rather than a CI.

    `fitter` selects the approximation, and the choice is recorded on every returned effect:

    * `"vb"` uses `BinomialBayesMixedGLM`, which fits the registered conditional model with a
      random intercept for item. Its interval is a **posterior** interval from a variational
      approximation, not a frequentist confidence interval, and the unit string says so.
    * `"gee"` uses generalised estimating equations with an exchangeable working correlation
      and cluster-robust standard errors by item. That gives a frequentist interval for a
      **population-averaged** coefficient, which is a different target from the conditional
      one, and the unit string says that too.

    Neither is the maximum-likelihood conditional GLMM `(1 | item)` normally denotes. Reporting
    both, labelled, is the honest response to a registered specification the available tooling
    cannot fit exactly. Whether the registration is satisfied is a planner question.
    """
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    records = []
    for row in rows:
        if row["task"] != task or row.get("repeat_idx", 0) != 0:
            continue
        for rung in rungs:
            if rung not in row["correct"]:
                raise InteractionError(
                    f"{task} item {row['item_id']}: no correctness for rung {rung!r}")
            records.append({
                "correct": int(bool(row["correct"][rung])),
                "precision": row["arm"],
                "scoring": rung,
                "model": row["model_key"],
                "item": _item_key(row),
            })
    if not records:
        raise InteractionError(
            f"{task}: no rows, so the model cannot be fit. An empty frame would let an "
            f"upstream filter read as a fitted null."
        )
    frame = pd.DataFrame.from_records(records)
    if frame["precision"].nunique() < 2 or frame["scoring"].nunique() < 2:
        raise InteractionError(
            f"{task}: the interaction needs at least two precision levels and two rungs, got "
            f"{frame['precision'].nunique()} and {frame['scoring'].nunique()}"
        )

    # `model` enters as a fixed effect per §2. With one model in the frame the term is dropped
    # rather than silently aliased onto the intercept.
    fixed = "correct ~ precision * scoring"
    if frame["model"].nunique() > 1:
        fixed += " + model"

    effects: list[Effect] = []
    if fitter == "vb":
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM  # noqa: PLC0415
        fit = BinomialBayesMixedGLM.from_formula(
            fixed, {"item": "0 + C(item)"}, frame).fit_vb(verbose=False)
        names = list(fit.model.exog_names)
        means, sds = np.asarray(fit.fe_mean), np.asarray(fit.fe_sd)
        unit = ("odds ratio (VARIATIONAL BAYES posterior mean, 95% POSTERIOR interval, "
                "NOT a confidence interval; VB understates posterior variance)")
        estimator = "BinomialBayesMixedGLM.fit_vb, random intercept (1|item)"
    elif fitter == "gee":
        import statsmodels.formula.api as smf  # noqa: PLC0415
        from statsmodels.genmod.cov_struct import Exchangeable  # noqa: PLC0415
        from statsmodels.genmod.families import Binomial  # noqa: PLC0415
        fit = smf.gee(fixed, groups="item", data=frame,
                      family=Binomial(), cov_struct=Exchangeable()).fit()
        names = list(fit.params.index)
        means, sds = np.asarray(fit.params), np.asarray(fit.bse)
        unit = ("population-averaged odds ratio (GEE, exchangeable working correlation, "
                "95% cluster-robust CI by item)")
        estimator = "GEE exchangeable, cluster-robust by item"
    else:
        raise ValueError(f"unknown fitter {fitter!r}; use 'vb' or 'gee'")

    n_items = frame["item"].nunique()
    for name, mean, sd in zip(names, means, sds, strict=False):
        if ":" not in name:
            continue  # interaction terms only; main effects are reported separately
        effects.append(Effect(
            name=f"H3_interaction_OR[{name}]",
            estimate=float(np.exp(mean)),
            ci_low=float(np.exp(mean - 1.96 * sd)),
            ci_high=float(np.exp(mean + 1.96 * sd)),
            # 🚨 An odds ratio's null is 1, not 0. `exp()` is strictly positive, so an
            # `excludes_null` written against 0 would be True for every one of these.
            null_value=1.0,
            unit=unit, n=int(len(frame)),
            interval_method=estimator,
            noise_class=NoiseClass.COMPOUNDING,
            context={"task": task, "term": name, "formula": fixed,
                     "estimator": estimator, "fitter": fitter,
                     "log_odds": float(mean), "log_odds_se": float(sd),
                     "n_items": n_items, "rungs": list(rungs),
                     "registered_formula_note":
                         "§2's (1|item) denotes a conditional ML GLMM, which was NOT fit: "
                         "statsmodels has no frequentist logistic mixed model and an R "
                         "dependency was rejected (R25.2c) as a reproducibility liability in a "
                         "paper about reproducibility. §2 is amended to name the estimator "
                         "actually used (R25.2a): the marginal difference-of-differences is "
                         "primary and this OR is secondary. This note is the required "
                         "disclosure, and the paper must carry a sentence of it."},
        ))
    if not effects:
        raise InteractionError(
            f"{task}: the fit produced no interaction terms. Coefficient names were {names}, "
            f"which means the design matrix did not contain the crossed term."
        )
    return effects


def rung_mcnemar(
    rows: list[dict],
    *,
    task: str,
    arms: tuple[str, ...] | None = None,
    rungs: tuple[str, str] = HEADLINE_RUNGS,
) -> list[Effect]:
    """§2's REGISTERED companion: exact McNemar on strict versus permissive, within each arm.

    §2: *"Companion test: **McNemar on discordant pairs**, reported alongside the GLMM as an
    assumption-light check. The extraction axis re-scores *identical generations*, so the
    strict-vs-permissive comparison is perfectly paired at the generation level."*

    This is a **within-arm** test and is therefore noise class 2 under §9f: the two rungs score
    the same text, so invocation noise is common to both and largely cancels. That is a materially
    better position than the interaction itself, which is class 3.

    ⚠️ Distinct from the p-value attached to `interaction_marginal`'s effect, which tests the
    arm-level contrast of the rung-change indicator. Both are reported, and each names the
    comparison it ran, because a reader who sees one p-value beside an interaction estimate will
    otherwise assume it is the registered one.

    The estimate is the rung gap in percentage points, `permissive - strict`, per arm.
    """
    from statsmodels.stats.proportion import proportion_confint  # noqa: PLC0415

    from .transitions import mcnemar_exact  # noqa: PLC0415

    if task in EXCLUDED_FROM_H3_INTERACTION:
        # No `allow_excluded` here: the per-arm rung gap on the hash arm is exactly the
        # demonstration §9c retains, so this one is informative rather than misleading. It is
        # still labelled, so it cannot be mistaken for an interaction level.
        pass

    cells = _collect(rows, task, rungs)
    wanted = arms if arms is not None else tuple(sorted(cells))
    effects: list[Effect] = []
    for arm in wanted:
        if arm not in cells:
            raise InteractionError(f"{task}: arm {arm!r} has no rows")
        cell = cells[arm]
        items = sorted(cell.items)
        if not items:
            raise InteractionError(
                f"{task} {arm}: no items scored at both rungs, so the paired comparison is "
                f"undefined rather than null"
            )
        # b: correct only under permissive (leniency recovered it).
        # c: correct only under strict (leniency lost it, which should be rare and is a finding).
        b = sum(1 for i in items if cell.high[i] and not cell.low[i])
        c = sum(1 for i in items if cell.low[i] and not cell.high[i])
        n = len(items)
        p_value, _ = mcnemar_exact(b, c)
        estimate = (b - c) / n * 100.0
        if b + c == 0:
            ci_low, ci_high, interval_method = zero_variance_bound(
                n, max_abs_contribution=1)
        else:
            low, high = proportion_confint(b, b + c, alpha=0.05, method="wilson")
            scale = (b + c) / n * 100.0
            ci_low = (2.0 * float(low) - 1.0) * scale
            ci_high = (2.0 * float(high) - 1.0) * scale
            interval_method = ("Wilson on the discordant split, rescaled to the net rate "
                               "(conditional-on-discordants McNemar interval)")
        effects.append(Effect(
            name=f"rung_gap[{arm}]",
            estimate=estimate, ci_low=ci_low, ci_high=ci_high,
            unit="pp", n=n, p_value=p_value,
            test="mcnemar_exact[registered companion: strict vs permissive, within arm]",
            interval_method=interval_method,
            noise_class=NoiseClass.PARTIALLY_CANCELLING,
            context={"task": task, "arm": arm, "rungs": list(rungs),
                     "n_recovered_by_permissive": b, "n_lost_by_permissive": c,
                     "registered_in": "§2 companion test"},
        ))
    return effects
