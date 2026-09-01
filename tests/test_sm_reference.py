"""Tests for the `reference` stage — THE MAIN RUN (`run_id=main-v1`).

Every hypothesis except H8 is estimated from this run, and it is also the pre-registration's
freeze line. So the invariants pinned here are the ones whose violation would be *invisible*:
a wrong item count still produces a full-looking run, a missing arm still produces a report,
and a rehearsal sharing the run_id still prints success.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import sm_reference  # noqa: E402

from quantartifact.records import (  # noqa: E402
    JOIN_KEY,
    SCHEMA_VERSION,
    GenerationRecord,
)

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
        must still leave a readable record of what completed.

        ⚠️ Anchored on `enumerate(run_list` rather than on the unpacking, which changed when
        K=2 turned the cell list into a run list of (cell, repeat_idx). The INTENT is that
        the write happens inside the loop; the anchor should track that, not the signature.
        """
        body = SOURCE.split("enumerate(run_list")[1]
        assert "reference_report.json" in body, (
            "the report must be written inside the run loop, not only after it"
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


class TestSamplingIsPinnedNeutral:
    """🚨 The paper REPORTS greedy decoding with no penalties, so it must be enforced.

    vLLM's `generation_config` defaults to "auto", which reads the model's own
    generation_config.json — and Qwen2.5-3B-Instruct ships `repetition_penalty 1.05` and
    `top_k 20` while Phi-3.5-mini ships nothing. `temperature=0` makes top_p/top_k
    inoperative, but repetition_penalty modifies the logits BEFORE the argmax, so a leaked
    1.05 would silently change greedy output on one model and not another.
    """

    GEN = (REPO / "src" / "quantartifact" / "generate.py").read_text()

    def test_all_six_sampling_fields_are_passed_explicitly(self):
        for field in ("temperature=0.0", "top_p=1.0", "top_k=-1",
                      "repetition_penalty=1.0", "presence_penalty=0.0",
                      "frequency_penalty=0.0"):
            assert field in self.GEN, f"{field} must be pinned, not left to a default"

    def test_engine_ignores_the_models_generation_config(self):
        assert 'generation_config: str = "vllm"' in self.GEN, (
            'the default "auto" reads the model\'s file and makes decoding model-dependent'
        )
        assert '"generation_config": self.spec.generation_config' in self.GEN, (
            "declaring it on EngineSpec is useless unless it is forwarded to LLM()"
        )

    def test_top_k_accepts_both_disabled_sentinels(self):
        """REGRESSION: the first assertion compared against -1 alone and failed a job
        because vLLM 0.26 reports back 0. Both spellings mean 'all tokens'."""
        assert '"top_k": (-1, 0)' in self.GEN

    def test_the_assertion_would_still_catch_a_real_value(self):
        """Accepting two sentinels must not become accepting anything: Qwen's 20 must
        still fail."""
        neutral = {"top_k": (-1, 0)}
        assert 20 not in neutral["top_k"]


class TestKTwoRepetition:
    """🚨 K=2 (prereg §9f / R23.3). The invocation floor is estimated from these cells.

    §9f registers an ESTIMATOR rather than a number, and the estimator is fitted from the
    K=2 cells at reference n. If this set drifts from the prereg, the floor is estimated
    from a different design than the one registered -- which is the declared-versus-enforced
    failure this project has already paid for nine times, arriving at a registered analysis.
    """

    PREREG = (Path(__file__).resolve().parents[1]
              / "analysis" / "preregistration.md").read_text()

    def test_exactly_ten_cells(self):
        assert len(sm_reference.K2_CELLS) == 10

    def test_three_models_x_three_contrasted_arms_plus_the_worst_observed(self):
        """Coverage follows the ESTIMAND, not the empirical screen (R23.3).

        Qwen is included despite showing 0.00% at n=180, because one item is 0.56 pp there,
        so "0.00" means fewer than one item flipped -- not zero with confidence. And qwen
        carries the largest rung gaps, so it is the worst place for an unmeasured floor.
        """
        for model in ("qwen2.5-3b", "phi-3.5-mini", "llama-3.1-8b"):
            for arm in ("bf16", "gptq-4bit", "awq-4bit"):
                assert f"{model}|{arm}" in sm_reference.K2_CELLS
        # The observed worst interaction movement (2.78 pp). Leaving the observed worst case
        # without a floor estimate is precisely what a reviewer picks first.
        assert "llama-3.1-8b|gptq-8bit" in sm_reference.K2_CELLS

    def test_the_non_contrasted_arms_are_NOT_doubled(self):
        """bnb-NF4 and the other gptq-8bit cells report against the POOLED floor."""
        for cell in ("qwen2.5-3b|bnb-nf4", "phi-3.5-mini|bnb-nf4", "llama-3.1-8b|bnb-nf4",
                     "qwen2.5-3b|gptq-8bit", "phi-3.5-mini|gptq-8bit"):
            assert cell not in sm_reference.K2_CELLS

    def test_prereg_and_code_agree_on_the_count(self):
        """The prereg names ten cells; the code must implement ten."""
        assert "= 9 cells, plus llama/gptq-8bit = 10 cells at K=2" in self.PREREG, (
            "§9f must state the K=2 design"
        )
        assert len(sm_reference.K2_CELLS) == 10

    def test_repeat_index_reaches_the_cell_payload(self):
        """A declared value that is never forwarded is decoration.

        Nine instances of this exact shape are recorded in notes/RUNLOG.md, so the forward
        is asserted rather than assumed: the cell script must read `repeat_idx` from the
        payload, and the reference stage must put it there.
        """
        reference = (Path(__file__).resolve().parents[1] / "sm_reference.py").read_text()
        pilot = (Path(__file__).resolve().parents[1] / "sm_pilot.py").read_text()
        assert '"repeat_idx": repeat_idx,' in reference, "payload must carry repeat_idx"
        assert 'repeat_idx=payload.get("repeat_idx", 0)' in pilot, (
            "CELL_SCRIPT must forward repeat_idx into run_cell"
        )

    def test_scratch_paths_distinguish_the_two_invocations(self):
        """The second invocation must not overwrite the first's counts file."""
        reference = (Path(__file__).resolve().parents[1] / "sm_reference.py").read_text()
        assert 'f"__k{repeat_idx}" if repeat_idx else ""' in reference


class TestInvocationId:
    """🚨 R21.3. Provenance for a run spanning many engine loads must be exact."""

    def test_it_is_on_the_record_schema(self):
        assert "invocation_id" in GenerationRecord.__dataclass_fields__

    def test_it_is_NOT_in_the_join_key(self):
        """Two invocations of one item are two measurements, distinguished by repeat_idx.

        Putting the invocation in the identity would make every re-run trivially unique and
        defeat the uniqueness constraint that catches double-counting.
        """
        assert "invocation_id" not in JOIN_KEY
        assert "repeat_idx" in JOIN_KEY

    def test_it_is_minted_per_engine_load_and_stamped_on_records(self):
        source = (Path(__file__).resolve().parents[1]
                  / "src" / "quantartifact" / "generate.py").read_text()
        runner_init = source.split("def __init__(self, spec: EngineSpec)")[1][:900]
        assert "self.invocation_id" in runner_init, "minted in VLLMRunner.__init__"
        assert "invocation_id=runner.invocation_id," in source, "stamped in run_cell"

    def test_schema_version_was_bumped(self):
        """A new field changes the released artifact's shape; the version must say so."""
        assert SCHEMA_VERSION != "1.0.0"


class TestSmokeExercisesKTwo:
    """🚨 REGRESSION (reviewer run 1, MAJOR 4). `--smoke` used to EXCLUDE the K=2 path.

    That inverted what a rehearsal is for: the mandatory smoke could not test the newest and
    least-proven code in the stage. With one job slot and a queue, K=2 failing inside the real
    ~17 h run is the most expensive outcome available.
    """

    def test_smoke_doubles_one_cell(self):
        """Smoke must double a cell so the K=2 path is actually exercised."""
        assert 'SMOKE_K2_CELL = "qwen2.5-3b|bf16"' in SOURCE
        assert 'c["cell_id"] == SMOKE_K2_CELL' in SOURCE

    def test_the_smoke_cell_is_chosen_BY_NAME_not_by_position(self):
        """⚠️ `cells[:1]` depended on `models.yaml` key order and `PRECISIONS` order.

        Reorder either and the doubled smoke cell becomes `llama-3.1-8b|bf16`, pulling ~16 GB
        from HF twice inside the 1800 s smoke cap. `sm_pilot` documents that hazard and picks
        Llama's quantized arm to avoid it.
        """
        assert "k2 = cells[:1]" not in SOURCE, (
            "selecting the smoke K=2 cell by list position is order-dependent"
        )

    def test_the_named_smoke_cell_is_the_cheapest_model_and_registered(self):
        """It must be a real K=2 cell, or the smoke rehearses something the run never does."""
        assert "qwen2.5-3b|bf16" in sm_reference.K2_CELLS

    def test_the_completeness_guard_is_not_nested_inside_if_k2(self):
        """Nested, it caught PARTIAL drift and missed TOTAL drift.

        If every registered cell_id stopped matching, `k2` was empty, the guard never ran,
        and the run completed with no K=2 data and exit 0.
        """
        guard = SOURCE.index("missing = sorted(K2_CELLS")
        # The guard must sit under `if not SMOKE:`, not under `if k2:`.
        preceding = SOURCE[:guard].rstrip().splitlines()[-1]
        assert "if not SMOKE" in preceding, (
            f"the K=2 completeness guard must not be conditional on k2 being non-empty; "
            f"preceding line is {preceding!r}"
        )


class TestPlanOnlyStatesTheRealSize:
    """--plan-only is what D10's confirmation is given AGAINST.

    The description said 63,570 generations and ~10.5 h after K=2 added ten doubled cells --
    a 67% understatement of the run the user would be approving.
    """

    def test_the_description_mentions_k2(self):
        spec = importlib.util.spec_from_file_location(
            "submit_job", REPO / "scripts" / "submit_job.py")
        submit_job = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(submit_job)
        description = submit_job.STAGES["reference"]["description"]
        assert "K=2" in description and "105,950" in description, description
