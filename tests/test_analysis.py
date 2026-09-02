"""Tests for the analysis layer (task `A19`).

These are written against hand-computable fixtures rather than against the code's own output,
because the failure mode this layer has is a **wrong number that looks plausible**. Every
expected value below is derived by hand in the test's own comment or docstring, so a test that
passes is evidence about the statistic and not merely about the code being self-consistent.

Three properties are tested deliberately rather than incidentally, each because the
pre-registration turns on it:

* the Manski difference bound is the WIDEST one, `alt.lower - ref.upper` to `alt.upper -
  ref.lower`, not the same-side subtraction that would understate what is unidentified (§1.3);
* the stratum conjunction runs over **all invocations** of the **three contrasted arms** only
  (§1.1 as amended, §9g);
* missing data raises rather than silently reading as an absent answer, since that conflation is
  the shape of the `classify_termination` bug in `notes/reviews/2026-09-01T020424Z.md`.
"""

from __future__ import annotations

import math

import pytest

from quantartifact.analysis import (
    Effect,
    NoiseClass,
    StratumError,
    bounds_effects,
    build_transitions,
    compute_bounds,
    compute_stratum,
    decompose,
    holm_bonferroni,
    interaction_marginal,
    mcnemar_exact,
    stopseq_differential,
    transition_effects,
)
from quantartifact.analysis.decompose import DecompositionError
from quantartifact.analysis.stratum import ABSENT_LABEL, COMMITTED_LABELS
from quantartifact.scoring import compliance as compliance_module

RUNGS = ("strict", "permissive")


def row(item: str, arm: str, *,
        correct: bool | dict[str, bool],
        committed: bool | dict[str, str] = True,
        termination: str | None = "COMPLETE",
        repeat_idx: int = 0,
        task: str = "bbh", subtask: str | None = "word_sorting",
        model: str = "qwen2.5-3b") -> dict:
    """One scoring row. `correct` and `committed` may be a bool or a per-rung dict.

    The union types are deliberate: most tests want "correct everywhere", while the tests that
    exercise the conjunction over rungs need to set one rung differently from another.
    """
    if isinstance(correct, bool):
        correct = dict.fromkeys(RUNGS, correct)
    if isinstance(committed, bool):
        label = "COMPLIANT" if committed else ABSENT_LABEL
        committed = dict.fromkeys(RUNGS, label)
    return {"model_key": model, "task": task, "subtask": subtask, "item_id": item,
            "arm": arm, "repeat_idx": repeat_idx, "correct": correct,
            "compliance": committed, "termination": termination}


class TestLabelsAgreeWithTheScorer:
    """The stratum's labels must be the scorer's, not a second copy that can drift.

    `HARNESS_UNTIL` drifting from `tasks.yaml` is a CRITICAL bug this project already had
    (`notes/reviews/2026-09-01T012037Z.md`). The same shape is available here, so it is pinned.
    """

    def test_committed_labels_match_compliance_module(self):
        assert {compliance_module.COMPLIANT,
                                    compliance_module.RECOVERABLE} == COMMITTED_LABELS

    def test_absent_label_matches_compliance_module(self):
        assert ABSENT_LABEL == compliance_module.ABSENT


class TestEffectRefusesToBeABareFlag:
    """§9: effect sizes with CIs for everything, never a bare significance flag."""

    def test_inverted_interval_is_rejected(self):
        with pytest.raises(ValueError, match="exceeds ci_high"):
            Effect(name="x", estimate=0.0, ci_low=1.0, ci_high=-1.0, unit="pp", n=10)

    def test_nan_estimate_is_rejected(self):
        with pytest.raises(ValueError, match="not a number"):
            Effect(name="x", estimate=math.nan, ci_low=0.0, ci_high=1.0, unit="pp", n=10)

    def test_p_value_without_a_named_test_is_rejected(self):
        with pytest.raises(ValueError, match="unnamed p-value"):
            Effect(name="x", estimate=0.0, ci_low=-1.0, ci_high=1.0, unit="pp", n=10,
                   p_value=0.01)

    def test_no_significance_attribute_exists(self):
        effect = Effect(name="x", estimate=1.0, ci_low=0.5, ci_high=1.5, unit="pp", n=10)
        assert not hasattr(effect, "significant")
        assert effect.excludes_zero is True

    def test_claimable_distinguishes_unmeasured_from_below_floor(self):
        """`None` and `False` must not collapse: they are different findings."""
        no_floor = Effect(name="x", estimate=0.1, ci_low=0.0, ci_high=0.2, unit="pp", n=10)
        assert no_floor.claimable is None
        below = Effect(name="x", estimate=0.1, ci_low=0.0, ci_high=0.2, unit="pp", n=10,
                       floor=0.5, noise_class=NoiseClass.COMPOUNDING)
        assert below.claimable is False
        above = Effect(name="x", estimate=0.9, ci_low=0.5, ci_high=1.3, unit="pp", n=10,
                       floor=0.5)
        assert above.claimable is True


class TestHolmBonferroni:
    def test_step_down_stops_at_the_first_failure(self):
        """With m=3 and alpha=0.05 the thresholds are 0.0167, 0.025, 0.05.

        p = (0.001, 0.03, 0.04). The smallest clears 0.05/3 = 0.0167 and is rejected. The next
        is compared against 0.05/2 = 0.025 and fails, so it is retained, and 0.04 is retained
        too even though it would clear 0.05 on its own. That last part is what a naive
        per-test comparison gets wrong.
        """
        effects = [
            Effect(name="a", estimate=1.0, ci_low=0.5, ci_high=1.5, unit="pp", n=10,
                   p_value=0.001, test="mcnemar_exact"),
            Effect(name="b", estimate=1.0, ci_low=-0.5, ci_high=2.5, unit="pp", n=10,
                   p_value=0.03, test="mcnemar_exact"),
            Effect(name="c", estimate=1.0, ci_low=-0.5, ci_high=2.5, unit="pp", n=10,
                   p_value=0.04, test="mcnemar_exact"),
        ]
        rows = holm_bonferroni(effects, family="H3")
        assert [r["rejected"] for r in rows] == [True, False, False]
        assert [r["name"] for r in rows] == ["a", "b", "c"], "input order must be preserved"
        assert all(r["m"] == 3 for r in rows)
        assert all(r["family"] == "H3" for r in rows)

    def test_effects_without_p_values_stay_in_the_family_count(self):
        """They are passed through with rejected=None and do not shrink m for the others."""
        effects = [
            Effect(name="tested", estimate=1.0, ci_low=0.5, ci_high=1.5, unit="pp", n=10,
                   p_value=0.02, test="mcnemar_exact"),
            Effect(name="estimate_only", estimate=1.0, ci_low=-1.0, ci_high=3.0,
                   unit="pp", n=10),
        ]
        rows = holm_bonferroni(effects, family="H4")
        assert rows[1]["rejected"] is None
        assert rows[0]["m"] == 1, "only tested effects enter the correction"

    def test_empty_family_raises(self):
        """An empty family must not read as a completed correction (score_template.py's bug)."""
        with pytest.raises(ValueError, match="no effects"):
            holm_bonferroni([], family="H3")


class TestMcNemarExact:
    def test_no_discordant_pairs_returns_none_not_one(self):
        p, test = mcnemar_exact(0, 0)
        assert p is None, "zero discordant pairs is no data, not p=1.0"
        assert test == "mcnemar_exact"

    def test_all_discordance_in_one_direction(self):
        """10 vs 0 is the two-sided binomial p = 2 * 0.5**10 = 0.001953125."""
        p, _ = mcnemar_exact(10, 0)
        assert p == pytest.approx(2 * 0.5 ** 10)

    def test_symmetric_discordance_is_not_significant(self):
        p, _ = mcnemar_exact(5, 5)
        assert p == pytest.approx(1.0)


class TestStratum:
    def test_conjunction_over_arms_and_rungs(self):
        """i1 committed everywhere; i2 absent under awq at permissive only."""
        rows = [
            row("i1", "bf16", correct=True), row("i1", "gptq-4bit", correct=True),
            row("i1", "awq-4bit", correct=True),
            row("i2", "bf16", correct=True), row("i2", "gptq-4bit", correct=True),
            row("i2", "awq-4bit", correct=False,
                committed={"strict": "COMPLIANT", "permissive": ABSENT_LABEL}),
        ]
        result = next(iter(compute_stratum(rows, rungs=RUNGS).values()))
        assert result.item_ids == {"i1"}
        assert result.n_items_total == 2
        assert result.fraction == 0.5
        assert result.excluded_by_arm["awq-4bit"] == 1

    def test_non_contrasted_arms_are_ignored(self):
        """R18.4.1: bnb-nf4 and gptq-8bit must not shrink the stratum."""
        rows = [
            row("i1", "bf16", correct=True), row("i1", "gptq-4bit", correct=True),
            row("i1", "awq-4bit", correct=True),
            row("i1", "bnb-nf4", correct=False, committed=False),
            row("i1", "gptq-8bit", correct=False, committed=False),
        ]
        result = next(iter(compute_stratum(rows, rungs=RUNGS).values()))
        assert result.item_ids == {"i1"}

    def test_all_invocations_is_stricter_than_any(self):
        """§9g: absent in ONE invocation excludes the item, and the cost is reported."""
        rows = [
            row("i1", "bf16", correct=True, repeat_idx=0),
            row("i1", "bf16", correct=True, repeat_idx=1, committed=False),
            row("i1", "gptq-4bit", correct=True), row("i1", "awq-4bit", correct=True),
        ]
        result = next(iter(compute_stratum(rows, rungs=RUNGS).values()))
        assert result.item_ids == frozenset()
        assert result.item_ids_any_invocation == {"i1"}
        assert result.invocation_cost == 1
        assert result.invocations_by_arm["bf16"] == 2

    def test_a_missing_contrasted_arm_raises(self):
        """Undefined, not empty. A data gap must not read as an abstention."""
        rows = [row("i1", "bf16", correct=True), row("i1", "gptq-4bit", correct=True)]
        with pytest.raises(StratumError, match="produced no rows"):
            compute_stratum(rows, rungs=RUNGS)

    def test_a_missing_rung_raises(self):
        rows = [
            row("i1", "bf16", correct=True, committed={"strict": "COMPLIANT"}),
            row("i1", "gptq-4bit", correct=True), row("i1", "awq-4bit", correct=True),
        ]
        with pytest.raises(StratumError, match="no compliance label"):
            compute_stratum(rows, rungs=RUNGS)

    def test_empty_rungs_would_be_vacuously_true_so_it_raises(self):
        with pytest.raises(ValueError, match="at least one rung"):
            compute_stratum([], rungs=())

    def test_truncated_after_commitment_stays_in_the_stratum(self):
        """§1.1's explicit trap: committed and TRUNC_BUDGET are not exclusive."""
        rows = [
            row("i1", a, correct=True, termination="TRUNC_BUDGET")
            for a in ("bf16", "gptq-4bit", "awq-4bit")
        ]
        result = next(iter(compute_stratum(rows, rungs=RUNGS).values()))
        assert result.item_ids == {"i1"}


class TestTransitions:
    def test_two_by_two_and_asymmetry(self):
        """bf16 committed on i1,i2,i3; gptq-4bit committed on i1 only.

        So n11=1 (i1), n12=2 (i2,i3 lost), n21=0, n22=0, asymmetry=+2.
        """
        rows = [row(i, "bf16", correct=True) for i in ("i1", "i2", "i3")]
        rows += [row("i1", "gptq-4bit", correct=True)]
        rows += [row(i, "gptq-4bit", correct=False, committed=False) for i in ("i2", "i3")]
        tables = build_transitions(rows, rungs=("strict",))
        table = next(t for t in tables if t.alt_arm == "gptq-4bit")
        assert (table.n11, table.n12, table.n21, table.n22) == (1, 2, 0, 0)
        assert table.asymmetry == 2
        assert table.n_discordant == 2

    def test_net_shift_is_signed_toward_loss(self):
        """Losing 2 of 3 commitments is a net shift of (0 - 2) / 3 * 100 = -66.67 pp."""
        rows = [row(i, "bf16", correct=True) for i in ("i1", "i2", "i3")]
        rows += [row("i1", "gptq-4bit", correct=True)]
        rows += [row(i, "gptq-4bit", correct=False, committed=False) for i in ("i2", "i3")]
        effects = transition_effects(build_transitions(rows, rungs=("strict",)))
        effect = next(e for e in effects if "gptq-4bit" in e.name)
        assert effect.estimate == pytest.approx(-200.0 / 3.0)
        assert effect.noise_class is NoiseClass.COMPOUNDING
        assert effect.ci_low <= effect.estimate <= effect.ci_high

    def test_zero_discordance_gives_a_nonzero_width_interval(self):
        """A precise-looking null is exactly the wrong report for a small n."""
        rows = [row(i, a, correct=True)
                for i in ("i1", "i2") for a in ("bf16", "gptq-4bit")]
        effects = transition_effects(build_transitions(rows, rungs=("strict",)))
        effect = effects[0]
        assert effect.estimate == 0.0
        assert effect.ci_high > 0.0, "zero-width interval would overstate precision"
        assert effect.p_value is None

    def test_k2_second_invocation_is_not_double_counted(self):
        rows = [row("i1", a, correct=True) for a in ("bf16", "gptq-4bit")]
        rows += [row("i1", a, correct=True, repeat_idx=1) for a in ("bf16", "gptq-4bit")]
        table = build_transitions(rows, rungs=("strict",))[0]
        assert table.n == 1, "repeat_idx 1 must not add a second pair for the same item"


class TestManskiBounds:
    def test_widest_difference_uses_opposite_sides(self):
        """§1.3, and the one place a same-side subtraction would silently understate.

        bf16: 4 items, 2 committed and both correct, 2 absent.
          lower = 2/4 = 0.5, upper = (2+2)/4 = 1.0
        gptq-4bit: 4 items, 4 committed, 1 correct, 0 absent.
          lower = upper = 0.25
        difference bounds = (0.25 - 1.0, 0.25 - 0.5) = (-0.75, -0.25)
        A same-side subtraction would give (0.25-0.5, 0.25-1.0) = (-0.25, -0.75), inverted,
        or (-0.25, -0.25) if both used `upper`, which is a point estimate masquerading as one.
        """
        rows = []
        for i in ("i1", "i2"):
            rows.append(row(i, "bf16", correct=True))
        for i in ("i3", "i4"):
            rows.append(row(i, "bf16", correct=False, committed=False))
        rows.append(row("i1", "gptq-4bit", correct=True))
        for i in ("i2", "i3", "i4"):
            rows.append(row(i, "gptq-4bit", correct=False))

        bound = next(b for b in compute_bounds(rows, rung="strict")
                     if b.alt_arm == "gptq-4bit")
        assert bound.reference.lower == pytest.approx(0.5)
        assert bound.reference.upper == pytest.approx(1.0)
        assert bound.alt.lower == pytest.approx(0.25)
        assert bound.alt.upper == pytest.approx(0.25)
        assert bound.lower == pytest.approx(-0.75)
        assert bound.upper == pytest.approx(-0.25)
        assert bound.width == pytest.approx(0.5)

    def test_bound_width_equals_the_absent_fraction(self):
        rows = [row("i1", a, correct=True) for a in ("bf16", "gptq-4bit")]
        rows += [row("i2", "bf16", correct=False, committed=False),
                 row("i2", "gptq-4bit", correct=True)]
        bound = compute_bounds(rows, rung="strict")[0]
        assert bound.reference.width == pytest.approx(0.5), "1 of 2 items absent"
        assert bound.alt.width == pytest.approx(0.0)

    def test_absent_items_are_bounded_not_counted_as_incorrect(self):
        """An ABSENT item with correct=False must not collapse the upper bound."""
        rows = [row("i1", a, correct=False, committed=False)
                for a in ("bf16", "gptq-4bit")]
        bound = compute_bounds(rows, rung="strict")[0]
        assert bound.reference.upper == pytest.approx(1.0)
        assert bound.reference.lower == pytest.approx(0.0)

    def test_effect_labels_the_interval_as_a_bound(self):
        rows = [row("i1", a, correct=True) for a in ("bf16", "gptq-4bit")]
        effect = bounds_effects(compute_bounds(rows, rung="strict"))[0]
        assert "Manski" in effect.unit
        assert "NOT a confidence interval" in effect.unit
        assert "manski_midpoint" in effect.name


class TestInteraction:
    def test_difference_of_differences(self):
        """Four items. Permissive recovers 2 of 4 for gptq-4bit and 0 of 4 for bf16.

        gap(gptq-4bit) = +50 pp, gap(bf16) = 0 pp, so the interaction is +50 pp.
        """
        rows = []
        for i in ("i1", "i2", "i3", "i4"):
            rows.append(row(i, "bf16", correct={"strict": True, "permissive": True}))
        for i in ("i1", "i2"):
            rows.append(row(i, "gptq-4bit",
                            correct={"strict": False, "permissive": True}))
        for i in ("i3", "i4"):
            rows.append(row(i, "gptq-4bit",
                            correct={"strict": True, "permissive": True}))
        effect = interaction_marginal(rows, task="bbh", alt_arm="gptq-4bit", n_boot=2000)
        assert effect.estimate == pytest.approx(50.0)
        assert effect.context["gap_alt_pp"] == pytest.approx(50.0)
        assert effect.context["gap_reference_pp"] == pytest.approx(0.0)
        assert effect.ci_low <= 50.0 <= effect.ci_high
        assert effect.noise_class is NoiseClass.COMPOUNDING

    def test_a_true_null_is_reported_as_an_estimate_not_a_failure(self):
        """§2 pre-commits that a null is publishable, so it must produce a normal Effect."""
        rows = []
        for i in ("i1", "i2", "i3", "i4"):
            for arm in ("bf16", "gptq-4bit"):
                rows.append(row(i, arm, correct={"strict": False, "permissive": True}))
        effect = interaction_marginal(rows, task="bbh", alt_arm="gptq-4bit", n_boot=2000)
        assert effect.estimate == pytest.approx(0.0)
        assert effect.ci_low <= 0.0 <= effect.ci_high

    def test_bootstrap_is_reproducible_under_a_fixed_seed(self):
        rows = []
        for i in ("i1", "i2", "i3", "i4"):
            rows.append(row(i, "bf16", correct={"strict": False, "permissive": True}))
            rows.append(row(i, "gptq-4bit", correct={"strict": False, "permissive": i != "i1"}))
        a = interaction_marginal(rows, task="bbh", alt_arm="gptq-4bit", n_boot=500, seed=7)
        b = interaction_marginal(rows, task="bbh", alt_arm="gptq-4bit", n_boot=500, seed=7)
        assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)

    def test_unshared_items_are_dropped_and_counted(self):
        rows = [row(i, "bf16", correct=True) for i in ("i1", "i2")]
        rows.append(row("i1", "gptq-4bit", correct=True))
        effect = interaction_marginal(rows, task="bbh", alt_arm="gptq-4bit", n_boot=200)
        assert effect.n == 1
        assert effect.context["n_items_dropped_unpaired"] == 1

    def test_no_shared_items_raises(self):
        rows = [row("i1", "bf16", correct=True), row("i2", "gptq-4bit", correct=True)]
        with pytest.raises(Exception, match="share no items"):
            interaction_marginal(rows, task="bbh", alt_arm="gptq-4bit", n_boot=200)


class TestDecomposition:
    def _rows(self):
        rows = []
        for i in ("i1", "i2", "i3", "i4"):
            rows.append(row(i, "bf16", correct=True))
        # gptq-4bit: i1 correct, i2 wrong, i3 absent, i4 stop-sequence killed and absent.
        rows.append(row("i1", "gptq-4bit", correct=True))
        rows.append(row("i2", "gptq-4bit", correct=False))
        rows.append(row("i3", "gptq-4bit", correct=False, committed=False))
        rows.append(row("i4", "gptq-4bit", correct=False, committed=False,
                        termination="TRUNC_STOPSEQ"))
        return rows

    def test_components_and_residual(self):
        """measured = 25 - 100 = -75 pp; compliance = 50 - 100 = -50 pp; stopseq = +25 pp.

        The stratum is {i1, i2}, on which bf16 is 100% and gptq-4bit 50%, so competence is
        -50 pp. Truncation is 0. The residual is therefore
        -75 - (-50 + -50 + 0 + 25) = -75 + 75 = 0.
        """
        rows = self._rows()
        cell = ("qwen2.5-3b", "bbh", "word_sorting", "1x", "fewshot_cot")
        out = decompose(rows, {cell: frozenset({"i1", "i2"})}, rung="strict")
        d = next(x for x in out if x.alt_arm == "gptq-4bit")
        assert d.measured_strict_delta == pytest.approx(-75.0)
        assert d.compliance_delta == pytest.approx(-50.0)
        assert d.stopseq_delta == pytest.approx(25.0)
        assert d.truncation_delta == pytest.approx(0.0)
        assert d.competence_delta == pytest.approx(-50.0)
        assert d.residual == pytest.approx(0.0)
        assert d.n_stratum == 2

    def test_empty_stratum_gives_none_not_zero_competence(self):
        cell = ("qwen2.5-3b", "bbh", "word_sorting", "1x", "fewshot_cot")
        out = decompose(self._rows(), {cell: frozenset()}, rung="strict")
        d = next(x for x in out if x.alt_arm == "gptq-4bit")
        assert d.competence_delta is None
        assert d.residual is None, "the identity is undefined without the competence term"

    def test_missing_stratum_raises(self):
        with pytest.raises(DecompositionError, match="no stratum was supplied"):
            decompose(self._rows(), {}, rung="strict")

    def test_gsm8k_is_refused_unless_explicitly_allowed(self):
        """R24.3 excludes the hash-marker arm from H4's primary decomposition."""
        rows = [row(i, a, correct=True, task="gsm8k", subtask=None)
                for i in ("i1",) for a in ("bf16", "gptq-4bit")]
        cell = ("qwen2.5-3b", "gsm8k", None, "1x", "fewshot_cot")
        with pytest.raises(DecompositionError, match="excluded from H4"):
            decompose(rows, {cell: frozenset({"i1"})}, rung="strict")
        out = decompose(rows, {cell: frozenset({"i1"})}, rung="strict",
                        allow_excluded=True)
        assert len(out) == 1

    def test_missing_termination_raises(self):
        rows = self._rows()
        del rows[0]["termination"]
        cell = ("qwen2.5-3b", "bbh", "word_sorting", "1x", "fewshot_cot")
        with pytest.raises(DecompositionError, match="no termination class"):
            decompose(rows, {cell: frozenset({"i1"})}, rung="strict")


class TestStopseqDifferential:
    def test_differential_by_precision(self):
        """gptq-4bit killed on 2 of 4, bf16 on none, so +50 pp with b=2, c=0."""
        rows = [row(i, "bf16", correct=True) for i in ("i1", "i2", "i3", "i4")]
        rows += [row(i, "gptq-4bit", correct=False, committed=False,
                     termination="TRUNC_STOPSEQ") for i in ("i1", "i2")]
        rows += [row(i, "gptq-4bit", correct=True) for i in ("i3", "i4")]
        effect = next(e for e in stopseq_differential(rows) if "gptq-4bit" in e.name)
        assert effect.estimate == pytest.approx(50.0)
        assert effect.context["n_killed_alt_only"] == 2
        assert effect.context["n_killed_reference_only"] == 0
        assert effect.p_value == pytest.approx(0.5), "two-sided binomial on 2 of 2"

    def test_not_differential_when_both_arms_are_killed_equally(self):
        rows = [row(i, a, correct=False, committed=False, termination="TRUNC_STOPSEQ")
                for i in ("i1", "i2") for a in ("bf16", "gptq-4bit")]
        effect = next(e for e in stopseq_differential(rows) if "gptq-4bit" in e.name)
        assert effect.estimate == pytest.approx(0.0)
        assert effect.p_value is None, "no discordant pairs is no data"
        assert effect.ci_high > 0.0
