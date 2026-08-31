"""Tests for the `reference` stage — THE MAIN RUN (`run_id=main-v1`).

Every hypothesis except H8 is estimated from this run, and it is also the pre-registration's
freeze line. So the invariants pinned here are the ones whose violation would be *invisible*:
a wrong item count still produces a full-looking run, a missing arm still produces a report,
and a rehearsal sharing the run_id still prints success.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import sm_reference  # noqa: E402

SOURCE = (REPO / "sm_reference.py").read_text()
TASKS = yaml.safe_load((REPO / "configs" / "tasks.yaml").read_text())["tasks"]


class TestRunId:
    def test_run_id_is_main_v1(self):
        """The paper cites this name, and it is the freeze line."""
        assert sm_reference.RUN_ID in ("main-v1", "main-v1-smoke")

    def test_smoke_uses_a_separate_run_id(self):
        """A rehearsal sharing it would pool 4-item junk into the population every
        downstream estimate is drawn from -- the `pilot-smoke-r2` defect, but on the
        primary evidence rather than a pilot."""
        line = next(ln for ln in SOURCE.splitlines() if ln.startswith("RUN_ID ="))
        assert "SMOKE" in line
        assert "main-v1-smoke" in line


class TestFullPrecisionLadder:
    def test_all_five_arms_present(self):
        """Five, not three: gptq-8bit and bnb-nf4 carry the dose-response argument
        (override O7), which is stronger internal validity than a two-point comparison.
        They are excluded from the expensive sweep, not from the reference condition."""
        assert sm_reference.PRECISIONS == (
            "bf16", "gptq-8bit", "gptq-4bit", "awq-4bit", "bnb-nf4")

    def test_bf16_control_is_first(self):
        assert sm_reference.PRECISIONS[0] == "bf16"

    def test_dose_response_ordering_is_monotone_in_bits(self):
        """The dose-response claim reads bf16 -> 8-bit -> 4-bit; the tuple order is what
        the paper's tables inherit, so a scrambled order would misrepresent it."""
        order = sm_reference.PRECISIONS
        assert order.index("gptq-8bit") < order.index("gptq-4bit")


class TestCountsComeFromConfigNotCode:
    """🚨 The counts are pre-registered. If this file hardcoded them they could diverge
    from `configs/tasks.yaml` silently -- the failure this project has hit seven times.
    """

    def test_plan_reads_reference_counts_from_the_config(self):
        plan = sm_reference.reference_plan(TASKS)
        by_task = {(t, s): n for _, t, s, n in plan}
        assert by_task[("gsm8k", None)] == TASKS["gsm8k"]["n_items_reference"]
        assert by_task[("gsm8k_cot", None)] == TASKS["gsm8k_cot"]["n_items_reference"]
        assert by_task[("arc_challenge_chat", None)] == \
            TASKS["arc_challenge_chat"]["n_items"]

    def test_bbh_uses_per_subtask_counts_not_a_flat_default(self):
        """R1.3 raised the two single-subtask contrast levels to 250; at 150 they could
        not detect anything under ~7.2 pp, leaving two of three gradient levels unpowered."""
        plan = sm_reference.reference_plan(TASKS)
        bbh = {s: n for _, t, s, n in plan if t == "bbh"}
        assert len(bbh) == 6, "all six subtasks must run"
        assert bbh["boolean_expressions"] == 250
        assert bbh["logical_deduction_five_objects"] == 250
        assert bbh["word_sorting"] == 150
        assert sum(bbh.values()) == 1100

    def test_total_items_per_cell_matches_the_documented_figure(self):
        plan = sm_reference.reference_plan(TASKS)
        assert sum(n for *_, n in plan) == 4238

    def test_total_generations_match_the_budget_projection(self):
        plan = sm_reference.reference_plan(TASKS)
        n_cells = 3 * len(sm_reference.PRECISIONS)
        assert n_cells * sum(n for *_, n in plan) == 63_570

    def test_no_item_count_is_hardcoded_in_the_plan(self):
        """No pre-registered count may appear as a literal in the plan builder.

        Comments are stripped first: the docstrings there legitimately CITE the counts to
        explain R1.3's reasoning, and a test that cannot tell an explanation from an
        instruction would punish documenting the design.
        """
        body = SOURCE.split("def reference_plan")[1].split("def build_cells")[0]
        code = "\n".join(
            line.split("#")[0] for line in body.splitlines()
            if not line.strip().startswith(("#", '"""', "*"))
        )
        for literal in ("1319", "1100", "250", "150"):
            assert literal not in code, (
                f"{literal} is hardcoded in reference_plan; it must come from tasks.yaml"
            )


class TestTemplateLevel:
    def test_reference_runs_the_primary_template_level(self):
        """prereg §3b-bis: `chat` is primary; `raw` is the template stage's job."""
        assert sm_reference.TEMPLATE_LEVEL == "chat"

    def test_level_is_forwarded_to_the_cell_script(self):
        assert '"template_level": cell["template_level"]' in SOURCE

    def test_cell_script_is_imported_not_copied(self):
        """Copying it would let the reference run drift from the pilot in prompt
        rendering or token-id handling, which is what makes the pilot a valid rehearsal."""
        assert "from sm_pilot import CELL_SCRIPT" in SOURCE


class TestFailClosed:
    def test_missing_checkpoint_refuses_before_any_engine_loads(self):
        """Fail-fast: a missing checkpoint must cost seconds, not surface at hour six."""
        assert "REFUSING TO RUN: no checkpoint" in SOURCE

    def test_frozen_max_num_seqs_is_read_not_assumed(self):
        assert "frozen_max_num_seqs()" in SOURCE

    def test_any_failed_cell_fails_the_job(self):
        """The ladder is a dose-response argument, so a missing arm breaks the
        monotonicity claim rather than merely reducing coverage."""
        assert "load-bearing" in SOURCE
        assert "return 1" in SOURCE

    def test_report_is_written_incrementally(self):
        """`output_path` only uploads when the job ENDS, so a timeout at cell 12 of 15
        must still leave a readable record of what completed."""
        body = SOURCE.split("for index, cell in enumerate")[1]
        assert "reference_report.json" in body, (
            "the report must be written inside the cell loop, not only after it"
        )


class TestFreezeLineIsDocumented:
    def test_source_states_it_is_the_freeze_line(self):
        assert "FREEZE LINE" in SOURCE

    def test_prereg_pegs_the_freeze_to_this_run(self):
        prereg = (REPO / "analysis" / "preregistration.md").read_text()
        assert "first `run_id=main-v1` generation" in prereg, (
            "the prereg must peg append-only status to this run, not to a calendar date"
        )

    def test_loglikelihood_omission_is_flagged_not_silent(self):
        """H6's mode contrast is incomplete until the log-likelihood arms exist; that
        must be stated in the stage rather than discovered later."""
        assert "log-likelihood arms are NOT in this stage" in SOURCE
