"""Tests for the `template` stage (prereg §3b-bis, planner rulings R19/R20).

These target the failures that would be SILENT. A template probe that runs one level
twice, or skips the second level entirely, still prints "18/18 cells" and still writes a
report — and the resulting "the delta is template-invariant" claim would be a comparison of
a level against itself. That is the worst available outcome for this stage, so the join-key
and level-coverage invariants are pinned here rather than trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

sm_template = pytest.importorskip(
    "sm_template", reason="entry point imports the serving stack"
)


class TestTemplateLevels:
    def test_exactly_two_levels_named_chat_and_raw(self):
        """§3b-bis defines a two-level factor. A third level would be unregistered."""
        assert sm_template.TEMPLATE_LEVELS == ("chat", "raw")

    def test_chat_is_the_primary_level_and_comes_first(self):
        """`chat` is primary (lm-eval recommends it for instruct variants, R19.1)."""
        assert sm_template.TEMPLATE_LEVELS[0] == "chat"

    def test_run_id_is_separate_from_the_annotation_pool(self):
        """Pooling into pilot-v2 would contaminate the population A11 samples blind."""
        assert "template" in sm_template.RUN_ID
        assert "pilot" not in sm_template.RUN_ID


class TestContrastedArms:
    def test_three_contrasted_arms_matching_the_amended_stratum(self):
        """R18.4.1 narrowed the stratum to these three; the probe must match it."""
        assert sm_template.TEMPLATE_ARMS == ("bf16", "gptq-4bit", "awq-4bit")

    def test_dose_response_arms_are_excluded(self):
        """gptq-8bit and bnb-nf4 exist for dose-response and method generality,
        neither of which a template-invariance probe tests."""
        assert "gptq-8bit" not in sm_template.TEMPLATE_ARMS
        assert "bnb-nf4" not in sm_template.TEMPLATE_ARMS

    def test_bf16_control_is_present(self):
        """Without bf16 there is no control and no delta to check invariance of."""
        assert "bf16" in sm_template.TEMPLATE_ARMS


class TestScope:
    def test_composition_is_the_r20_default_tier(self):
        """R20.1: measured 7,099 gen/h selects FULL by rate, but the 15%-of-remaining
        cap binds, so the probe runs the DEFAULT tier at {500, 500}."""
        assert sm_template.TEMPLATE_COMPOSITION == {"gsm8k_cot": 500, "bbh": 500}

    def test_both_tasks_present_because_dropping_either_guts_the_claim(self):
        """R20.1's floor: BBH carries the largest template effect, gsm8k_cot the
        primary interaction. Neither is optional."""
        assert set(sm_template.TEMPLATE_COMPOSITION) == {"gsm8k_cot", "bbh"}

    def test_total_generations_match_the_registered_18000(self):
        n_cells = len(sm_template.TEMPLATE_LEVELS) * 3 * len(sm_template.TEMPLATE_ARMS)
        per_cell = sum(sm_template.TEMPLATE_COMPOSITION.values())
        assert n_cells == 18
        assert n_cells * per_cell == 18_000


class TestPlan:
    def test_bbh_spreads_over_all_six_subtasks(self):
        """A single-subtask probe could not show the format gradient, which is where
        the template effect is expected to vary most."""
        tasks = {"bbh": {"lm_eval_task_prefix": "bbh_cot_fewshot_",
                         "subtasks": [{"name": f"s{i}"} for i in range(6)]}}
        plan = sm_template.template_plan(tasks, 500, 500)
        bbh = [row for row in plan if row[1] == "bbh"]
        assert len(bbh) == 6
        assert sum(row[3] for row in bbh) == 500, "every BBH item must be allocated"

    def test_plan_totals_match_the_composition_exactly(self):
        tasks = {"bbh": {"lm_eval_task_prefix": "bbh_cot_fewshot_",
                         "subtasks": [{"name": f"s{i}"} for i in range(6)]}}
        plan = sm_template.template_plan(tasks, 500, 500)
        assert sum(row[3] for row in plan) == 1000

    def test_gsm8k_cot_not_gsm8k(self):
        """R18.2: the hash arm is a floored metric with no variance for an interaction
        to appear in, so the probe must use the PHRASE-marker task."""
        tasks = {"bbh": {"lm_eval_task_prefix": "p_", "subtasks": [{"name": "s0"}]}}
        plan = sm_template.template_plan(tasks, 10, 10)
        assert plan[0][1] == "gsm8k_cot"


class TestCellIdCarriesTheLevel:
    """🚨 The silent-failure guard. Both levels sharing a cell_id means they share a
    join key, so `run_cell` treats the second level's items as already written and skips
    every one -- printing success while generating nothing, exactly as logged against
    `pilot-smoke-r2`. The resulting claim would compare a level against itself.
    """

    def test_cell_id_template_matches_the_declared_levels(self):
        """Checked on the source rather than by building cells, which needs weights."""
        source = (REPO / "sm_template.py").read_text()
        assert '"cell_id": f"{model_key}|{precision}|{level}"' in source, (
            "cell_id must include the template level or the two levels collide"
        )

    def test_level_is_recorded_on_every_cell(self):
        source = (REPO / "sm_template.py").read_text()
        assert '"template_level": level' in source

    def test_payload_forwards_the_level_to_the_cell_script(self):
        """A cell_id that differs while the RENDERING does not would be worse than a
        collision: two distinct-looking populations generated identically."""
        source = (REPO / "sm_template.py").read_text()
        assert '"template_level": cell["template_level"]' in source

    def test_cell_script_is_imported_not_copied(self):
        """Copying it would let the two levels drift apart in some third respect."""
        source = (REPO / "sm_template.py").read_text()
        assert "from sm_pilot import CELL_SCRIPT" in source


class TestCellScriptHonoursTheLevel:
    def test_apply_chat_template_is_driven_by_the_level(self):
        assert 'apply_chat_template=(payload["template_level"] == "chat")' \
            in sm_template.CELL_SCRIPT

    def test_until_is_selected_per_level_with_a_fallback(self):
        """Tasks without a per-level override fall back to their single `until`."""
        assert 'until_by_template' in sm_template.CELL_SCRIPT
        assert 'config["until"]' in sm_template.CELL_SCRIPT


class TestFailureIsLoadBearing:
    def test_any_failed_cell_fails_the_job(self):
        """Unlike the determinism sweep, there are no corroborating cells here: the
        claim is a PAIRED contrast, so a missing cell breaks the pairing."""
        source = (REPO / "sm_template.py").read_text()
        assert "load-bearing" in source
        assert "return 1" in source


class TestRunIdWasBumpedForTheFrozenBatchSize:
    """🚨 `template-v1` ran at max_num_seqs=64; R24.1 moved the freeze to 32.

    Re-running as `template-v1` would now ABORT, because `assert_cell_reusable` refuses to
    resume a cell whose stored `max_num_seqs` differs. Even without that guard it would have
    mixed 64-batch and 32-batch generations inside one cell file. `template-v1` is retained as
    the measurement taken at 64, and the two runs are never pooled.
    """

    SOURCE = (Path(__file__).resolve().parents[1] / "sm_template.py").read_text()

    def test_run_id_is_v2(self):
        assert sm_template.RUN_ID in ("template-v2", "template-v2-smoke")

    def test_the_superseded_id_is_not_reused(self):
        assert "template-v1" not in sm_template.RUN_ID

    def test_the_reason_is_recorded_next_to_it(self):
        """A bump without its reason gets reverted by the next person."""
        assert "max_num_seqs=64" in self.SOURCE
        assert "assert_cell_reusable" in self.SOURCE

    def test_the_scorer_defaults_to_the_new_run_but_can_read_the_old(self):
        source = (Path(__file__).resolve().parents[1]
                  / "scripts" / "score_template.py").read_text()
        assert 'RUN = "template-v2"' in source
        assert '"--run"' in source, "the earlier measurement must stay readable"
