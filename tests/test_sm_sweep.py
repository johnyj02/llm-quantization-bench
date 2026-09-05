"""Tests for `sm_sweep.py`, H1's registered instrument.

🚨 **WHY THIS FILE EXISTS.** It did not, and that was the CODE REVIEWER's first finding on
2026-09-05: H1 is the LEAD hypothesis, marked *"never cut"*, its registered instrument had
never existed until this week, and the instrument arrived with **zero tests** while
`sm_anchor.py` got 17 and `sm_reference.py` has 39. The one property this stage must never get
wrong was established by a single smoke observation on one cell at one level and protected by a
comment.

The defect class is what makes it urgent rather than untidy. If `budget_level` and
`max_gen_toks` disagree, the artifact is **mislabelled**: a generation capped at 256 tokens
recorded as `4x`. It scores cleanly, looks plausible in aggregate, and is **unrecoverable
afterwards**, because nothing else in the record says what cap was in force. H1's entire claim
is a curve over that variable.

None of these tests need a GPU, network, or credentials.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import sm_sweep  # noqa: E402

from quantartifact.records import SchemaError  # noqa: E402

TASKS = yaml.safe_load((REPO / "configs" / "tasks.yaml").read_text())["tasks"]


class TestRegisteredScope:
    """The counts and levels the pre-registration fixes, asserted rather than inferred."""

    def test_plan_sums_to_the_registered_item_count(self):
        """§2.2 registers 1,400 items per sweep cell: gsm8k_cot 500 + BBH 900."""
        plan = sm_sweep.sweep_plan(TASKS)
        assert sum(limit for *_, limit in plan) == 1400

    def test_bbh_is_split_across_all_six_subtasks(self):
        """The format gradient is a PREDICTION of the hypothesis, not six arbitrary tasks.

        Dropping or merging a subtask would silently change what the gradient measures.
        """
        plan = sm_sweep.sweep_plan(TASKS)
        bbh = [(sub, limit) for _, task, sub, limit in plan if task == "bbh"]
        declared = [s["name"] for s in TASKS["bbh"]["subtasks"]]
        assert sorted(sub for sub, _ in bbh) == sorted(declared)
        assert len(bbh) == 6
        assert sum(limit for _, limit in bbh) == 900

    def test_eighteen_units(self):
        """3 models x 3 arms x 2 levels = 18 units, x 1,400 items = 25,200 generations.

        ⚠️ Asserted because `submit_job.py` described this as "3 arms x {2x,4x} x 1,400", which
        reads as SIX units, and that is the surface a human confirms a submission against.
        """
        assert len(sm_sweep.SWEEP_ARMS) == 3
        assert len(sm_sweep.SWEEP_LEVELS) == 2
        models = yaml.safe_load((REPO / "configs" / "models.yaml").read_text())["models"]
        units = len(models) * len(sm_sweep.SWEEP_ARMS) * len(sm_sweep.SWEEP_LEVELS)
        assert units == 18
        assert units * 1400 == 25_200

    def test_one_x_is_excluded_because_it_comes_free_from_the_reference_run(self):
        """🚨 The nesting IS the design, and re-adding 1x would double-measure it.

        `n_items_sweep: 500` is a prefix-nested subset of the reference condition's 1,319, so
        `main-v1` already contains this sweep's 1x level. Generating it again would produce a
        second measurement of one cell, which §9f's invocation-noise accounting treats as two
        invocations rather than one, and H1's curve would compare a fresh 1x against a
        `main-v1` 2x from a different job.
        """
        assert "1x" not in sm_sweep.SWEEP_LEVELS

    def test_swept_arms_exclude_the_two_arms_the_prereg_excludes(self):
        """§2.2 excludes bnb-NF4 and GPTQ-8bit from the expensive arms, with reasons."""
        assert set(sm_sweep.SWEEP_ARMS) == {"bf16", "gptq-4bit", "awq-4bit"}
        assert "bnb-nf4" not in sm_sweep.SWEEP_ARMS
        assert "gptq-8bit" not in sm_sweep.SWEEP_ARMS


class TestBudgetLabelCannotDisagreeWithTheCap:
    """🚨 The unrecoverable defect. The guard is now structural, in `validate()`."""

    def test_every_swept_level_is_declared_for_every_swept_task(self):
        """A level the config does not declare must not be silently substitutable."""
        plan = sm_sweep.sweep_plan(TASKS)
        for _, task, _, _ in plan:
            caps = TASKS[task]["max_gen_toks"]
            for level in sm_sweep.SWEEP_LEVELS:
                assert level in caps, f"{task} declares no {level} cap"

    def test_levels_are_strictly_increasing_so_the_axis_is_ordered(self):
        """H1's claim is monotone in the budget, so the levels must actually be a gradient."""
        for task in {t for _, t, _, _ in sm_sweep.sweep_plan(TASKS)}:
            caps = TASKS[task]["max_gen_toks"]
            assert caps["1x"] < caps["2x"] < caps["4x"], task

    def test_validate_refuses_a_mislabelled_budget(self):
        """The structural guard: a 4x label against a 1x cap must not be writable.

        This is the test that would have caught the defect the build only commented about.
        """
        from test_records import make_record  # noqa: PLC0415 - shared fixture

        bad = make_record(task="gsm8k_cot", budget_level="4x",
                          max_gen_toks=TASKS["gsm8k_cot"]["max_gen_toks"]["1x"])
        with pytest.raises(SchemaError, match="disagrees with"):
            bad.validate()

    def test_validate_accepts_every_task_level_pair_the_sweep_will_write(self):
        """The guard must not reject the run it is protecting."""
        from test_records import make_record  # noqa: PLC0415 - shared fixture

        for _, task, sub, _ in sm_sweep.sweep_plan(TASKS):
            for level in sm_sweep.SWEEP_LEVELS:
                make_record(task=task, subtask=sub or None, budget_level=level,
                            max_gen_toks=TASKS[task]["max_gen_toks"][level]).validate()


class TestFourXFitsInsideTheContextWindow:
    """MODERATE 3. The 4x/BBH corner the smoke could not reach.

    Measured from `main-v1`'s stored `n_prompt_tokens` on 2026-09-05: the worst swept prompt is
    BBH `dyck_languages` at 1,083 tokens, so 1,083 + 4,096 = 5,179 against `max_model_len`
    8,192, leaving about 3,000 tokens of headroom. That is comfortable, and it is exactly why
    this is a test rather than a note: the margin holds *today*, and a config change to either
    the cap or the window could close it silently. A silent clamp would produce a cell labelled
    4x whose effective cap is smaller, which is the same unrecoverable class as a mislabel
    arriving by a different route.
    """

    #: Worst observed prompt length per swept task, from `main-v1`, 2026-09-05.
    OBSERVED_MAX_PROMPT = {
        "gsm8k_cot": 954,
        "bbh": 1083,          # dyck_languages, the longest of the six
    }

    def test_prompt_plus_four_x_cap_fits(self):
        serving = yaml.safe_load(
            (REPO / "configs" / "models.yaml").read_text())["serving"]
        window = serving["max_model_len"]
        for task, observed in self.OBSERVED_MAX_PROMPT.items():
            cap = TASKS[task]["max_gen_toks"]["4x"]
            assert observed + cap <= window, (
                f"{task}: {observed} prompt + {cap} cap = {observed + cap} exceeds "
                f"max_model_len {window}. vLLM may CLAMP rather than raise, which would "
                f"produce a cell labelled 4x at a smaller effective cap."
            )

    def test_headroom_is_not_marginal(self):
        """Fail while there is still room to react, not at the moment it overflows."""
        serving = yaml.safe_load(
            (REPO / "configs" / "models.yaml").read_text())["serving"]
        window = serving["max_model_len"]
        for task, observed in self.OBSERVED_MAX_PROMPT.items():
            cap = TASKS[task]["max_gen_toks"]["4x"]
            assert window - (observed + cap) >= 512, (
                f"{task} has under 512 tokens of headroom at 4x; re-measure prompts "
                f"before trusting the cap"
            )
