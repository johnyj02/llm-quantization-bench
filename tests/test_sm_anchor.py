"""Tests for the `anchor` stage (task A9, gate G5a) and its scorer.

Anchor A is **blocking**, so the highest-value tests are the ones that stop it reporting a verdict
it has not earned: a gate that cannot fail, a comparison built on the wrong pairing, or a side
label that could silently invert the difference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import sm_anchor  # noqa: E402

import score_anchor  # noqa: E402


class TestCriterionCannotBeUnfalsifiable:
    """The metric choice decides whether the gate can fail at all (planner R30)."""

    def test_gate_metric_is_gsm8k_cot_permissive(self):
        """Not the hash arm at rung 1, where the metric is floored near 1%.

        A +/-2.0 pp band around a 1.0% value is wider than the whole measurable range, so that
        combination would make a BLOCKING gate incapable of failing.
        """
        assert sm_anchor.CRITERION_METRIC == ("gsm8k_cot", "permissive")
        assert score_anchor.CRITERION_METRIC == sm_anchor.CRITERION_METRIC

    def test_strict_is_diagnostic_and_never_blocks(self):
        """R30: a strict discrepancy requires explanation, it does not block."""
        assert ("gsm8k_cot", "strict") in sm_anchor.DIAGNOSTIC_METRICS
        assert sm_anchor.CRITERION_METRIC not in sm_anchor.DIAGNOSTIC_METRICS

    def test_the_stage_and_the_scorer_agree_on_the_band(self):
        """Two copies of a threshold are two chances to disagree."""
        assert sm_anchor.CRITERION_PP == score_anchor.CRITERION_PP == 2.0

    def test_gate_task_runs_at_full_1319_items(self):
        """R30: at n=500 a perfect pipeline fails this gate roughly one time in six."""
        by_task = {task: n for _, task, n in sm_anchor.ANCHOR_TASKS}
        gate_task = sm_anchor.CRITERION_METRIC[0]
        assert by_task[gate_task] == 1319, (
            "the BLOCKING metric needs gate-grade resolution; the paired SE at n=500 makes the "
            "+/-2.0 pp band about one SE wide")


class TestVerdict:
    """R30 requires the verdict be evaluated against the INTERVAL, not the point alone."""

    def _result(self, diff, low, high):
        return {"difference_pp": diff, "ci_low": low, "ci_high": high}

    def test_clean_pass_needs_the_whole_interval_inside_the_band(self):
        verdict, why = score_anchor.verdict_for(self._result(0.5, -1.2, 1.4))
        assert verdict == "PASS"
        assert "inside the band" in why

    def test_point_inside_but_interval_straddling_is_INCONCLUSIVE_not_a_pass(self):
        """The whole reason R30 asked for the interval: this must not read as a clean pass."""
        verdict, why = score_anchor.verdict_for(self._result(1.5, -0.5, 3.4))
        assert verdict == "INCONCLUSIVE"
        assert "does not resolve" in why

    def test_point_outside_the_band_fails(self):
        verdict, why = score_anchor.verdict_for(self._result(4.0, 2.5, 5.5))
        assert verdict == "FAIL"
        assert "calibration or serving config is wrong" in why

    def test_verdict_is_from_a_closed_set(self):
        for diff, low, high in ((0.5, -1.0, 1.0), (1.5, -0.5, 3.4), (4.0, 2.5, 5.5)):
            verdict, _ = score_anchor.verdict_for(self._result(diff, low, high))
            assert verdict in score_anchor.VERDICTS


class TestPairedDifference:
    def test_difference_is_ours_minus_official(self):
        """Sign matters: getting it backwards would invert every verdict."""
        ours = {f"i{i}": i < 6 for i in range(10)}       # 60%
        official = {f"i{i}": i < 5 for i in range(10)}   # 50%
        result = score_anchor.paired_difference(ours, official, n_boot=500)
        assert result["difference_pp"] == pytest.approx(10.0)
        assert result["accuracy_ours_pp"] == pytest.approx(60.0)
        assert result["accuracy_official_pp"] == pytest.approx(50.0)

    def test_identical_sides_give_a_nonzero_width_bound_not_a_point(self):
        """Same degeneracy R25.1 registered: nothing to resample, so a bound rather than [0,0]."""
        same = {f"i{i}": i % 2 == 0 for i in range(40)}
        result = score_anchor.paired_difference(same, dict(same), n_boot=500)
        assert result["difference_pp"] == pytest.approx(0.0)
        assert result["n_discordant"] == 0
        assert result["ci_high"] > 0.0, "a zero-width interval would overstate resolution"
        assert "zero-variance bound" in result["interval_method"]

    def test_discordant_count_is_reported_because_it_drives_the_width(self):
        ours = {f"i{i}": True for i in range(20)}
        official = {f"i{i}": i >= 3 for i in range(20)}
        result = score_anchor.paired_difference(ours, official, n_boot=500)
        assert result["n_discordant"] == 3

    def test_no_shared_items_raises_rather_than_returning_zero(self):
        with pytest.raises(SystemExit, match="paired difference is undefined"):
            score_anchor.paired_difference({"a": True}, {"b": True}, n_boot=100)


class TestCellConstruction:
    def test_six_cells_three_arms_two_sides(self):
        models = yaml.safe_load((REPO / "configs" / "models.yaml").read_text())["models"]
        from quantartifact.quantize import load_specs
        specs, calibration = load_specs(REPO / "configs" / "quant.yaml")
        try:
            cells = sm_anchor.anchor_cells(models, specs, calibration)
        except SystemExit as exc:
            pytest.skip(f"checkpoints not present locally: {exc}")
        assert len(cells) == 6
        assert {c["side"] for c in cells} == {"ours", "official"}
        assert len({c["arm"] for c in cells}) == 3

    def test_official_cells_carry_no_recipe_hash(self):
        """We did not build them, so a null hash stops an official record reading as one of ours."""
        source = (REPO / "sm_anchor.py").read_text()
        assert '"quant_config_hash": None' in source

    def test_both_sides_use_the_base_repo_tokenizer(self):
        """The one thing that must be identical, or the comparison confounds weights with prompts."""
        assert 'AutoTokenizer.from_pretrained(cell["base_repo"]' in sm_anchor.CELL_SCRIPT

    def test_anchor_a_is_marked_blocking_in_config(self):
        cfg = yaml.safe_load((REPO / "configs" / "models.yaml").read_text())
        assert cfg["anchor_checkpoints"]["anchor_a"]["blocking"] is True


class TestSmokeCoversTheUntestedPath:
    def test_smoke_keeps_both_sides(self):
        """A rehearsal running only our checkpoint would skip the official-repo download, which is
        the only part of this stage that has never run anywhere."""
        source = (REPO / "sm_anchor.py").read_text()
        assert 'c["arm"] == "gptq-4bit"' in source
        assert "both sides" in source
