"""Tests for the pilot entry point's decision logic (job 3, task A8).

The pilot produces the first study DATA rather than instruments, and two of its
decisions are unrecoverable if wrong:

* Its generations become the judge-validation annotation population (override O11),
  and `A11` draws that sample **blind** — so anything that pollutes the population
  cannot be caught downstream by inspection.
* It runs at the `A7`-frozen `max_num_seqs`. A guessed value would make the pilot
  incomparable to every main run, silently.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import sm_pilot
import yaml
from sm_pilot import PILOT_COMPOSITION, PRECISIONS, pilot_plan

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
TASKS = yaml.safe_load((CONFIGS / "tasks.yaml").read_text())["tasks"]


class TestPilotPlan:
    def test_totals_300_items(self):
        """A8 specifies 300 items per cell; the annotation stratification assumes it."""
        assert sum(limit for *_, limit in pilot_plan(TASKS)) == 300
        assert sum(PILOT_COMPOSITION.values()) == 300

    def test_composition_matches_a8(self):
        assert PILOT_COMPOSITION == {"gsm8k": 100, "bbh": 120, "arc_challenge_chat": 80}

    def test_bbh_spread_over_every_subtask(self):
        """All three answer-format classes must reach the annotation pool.

        A pool drawn from one BBH subtask would make the judge-reliability claim
        narrower than the population it is supposed to describe.
        """
        plan = pilot_plan(TASKS)
        bbh = {subtask: limit for _, task, subtask, limit in plan if task == "bbh"}
        expected = {s["name"] for s in TASKS["bbh"]["subtasks"]}
        assert set(bbh) == expected
        assert all(limit > 0 for limit in bbh.values())
        assert sum(bbh.values()) == PILOT_COMPOSITION["bbh"]

    def test_bbh_uses_the_configured_task_prefix(self):
        """Wrong prefix => lm-eval loads a different task => wrong prompts."""
        plan = pilot_plan(TASKS)
        prefix = TASKS["bbh"]["lm_eval_task_prefix"]
        for lm_eval_task, task, _, _ in plan:
            if task == "bbh":
                assert lm_eval_task.startswith(prefix)

    def test_plan_is_deterministic(self):
        assert pilot_plan(TASKS) == pilot_plan(TASKS)


class TestFrozenMaxNumSeqs:
    def test_refuses_while_the_audit_is_unresolved(self):
        """Fails CLOSED. models.yaml currently flags the value as unaudited.

        Running the pilot at §3.3's pre-audit 8 -- a figure from vLLM #51187 measured
        on other hardware -- would make its generations incomparable to every main
        run, with nothing downstream to reveal it.
        """
        serving = yaml.safe_load((CONFIGS / "models.yaml").read_text())["serving"]
        if serving.get("max_num_seqs_resolved_by_audit"):
            pytest.skip("A7 has since frozen the value; this guard no longer applies")
        with pytest.raises(SystemExit, match="max_num_seqs_resolved_by_audit"):
            sm_pilot.frozen_max_num_seqs()


class TestRunIdIsolation:
    def test_smoke_uses_a_separate_run_id(self):
        """`run_id` is JOIN_KEY[0] and the top directory of cell_path.

        A shared id would pool 10-item smoke generations into the population A11
        samples from -- blind, so uninspectable.
        """
        source = (Path(__file__).resolve().parents[1] / "sm_pilot.py").read_text()
        line = next(ln for ln in source.splitlines() if ln.startswith("RUN_ID ="))
        assert "SMOKE" in line, "RUN_ID must depend on SMOKE"
        assert "pilot-v1-smoke" in line
        assert re.search(r'else\s+"pilot-v1"', line)

    def test_run_id_matches_the_registered_value(self):
        """A8 and the artifact release (A24) both name pilot-v1 explicitly."""
        assert sm_pilot.RUN_ID in ("pilot-v1", "pilot-v1-smoke")


class TestR12Replicates:
    def test_both_awq_hashes_are_distinct_and_recorded(self):
        """R12.1 compares the two byte-different qwen AWQ checkpoints."""
        hashes = sm_pilot.AWQ_REPLICATE_HASHES
        assert len(hashes) == 2
        assert len(set(hashes.values())) == 2, "the two checkpoints must differ"
        # These are the hashes job 1 actually wrote; offload_device is in the recipe
        # hash precisely so the two cannot be confused.
        assert hashes["awq-4bit-offload"] == "e7377d975178f192"
        assert hashes["awq-4bit-nooffload"] == "184c722115d8d9bb"

    def test_replicate_cell_ids_cannot_merge(self):
        """cell_id is part of JOIN_KEY, so a shared id would pool two weight sets
        into one accuracy number -- which is the exact question being asked."""
        ids = {f"qwen2.5-3b|{label}" for label in sm_pilot.AWQ_REPLICATE_HASHES}
        assert len(ids) == 2
        # And neither may collide with the ordinary awq-4bit cell.
        assert "qwen2.5-3b|awq-4bit" not in ids


class TestCellCoverage:
    def test_five_precisions(self):
        assert set(PRECISIONS) == {
            "bf16", "gptq-8bit", "gptq-4bit", "awq-4bit", "bnb-nf4"
        }

    def test_fifteen_base_cells_plus_two(self):
        """A8's 15 cells (3 models x 5 precisions) plus R12.1's 2 = 17."""
        models = yaml.safe_load((CONFIGS / "models.yaml").read_text())["models"]
        assert len(models) * len(PRECISIONS) == 15
        assert 15 + len(sm_pilot.AWQ_REPLICATE_HASHES) == 17
