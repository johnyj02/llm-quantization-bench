"""Tests for the `determinism` entry point's decision logic (job 2, task A7).

These functions decide *what gets measured* and *how an arm is tagged*. Both are
things a paid GPU job would otherwise be the first to exercise, and both have a
trap in them worth pinning down:

* `bbh_probe_split` — BBH is three answer-format classes, not one. A split that
  silently dropped or double-counted items would change the probe size and so the
  disagreement *rate*, which is the number H8 turns on.
* `arm_engine_fields` — `bf16` is **absent** from `specs` (`load_specs` skips
  `method == "none"`), so the naive `specs[arm]` is a KeyError.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sm_determinism import AUDIT_ARMS, arm_engine_fields, bbh_probe_split, cell_tag

from quantartifact.determinism import PROBE_COMPOSITION, SWEEP_FLOOR
from quantartifact.quantize import load_specs

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

SUBTASKS = [
    "word_sorting",
    "dyck_languages",
    "multistep_arithmetic_two",
    "object_counting",
    "boolean_expressions",
    "logical_deduction_five_objects",
]


class TestBBHProbeSplit:
    def test_sums_to_total(self):
        """The probe size is the denominator of every disagreement rate."""
        split = bbh_probe_split(200, SUBTASKS)
        assert sum(split.values()) == 200

    def test_covers_every_subtask(self):
        """A subtask with zero items lets a format-dependent bug hide."""
        split = bbh_probe_split(200, SUBTASKS)
        assert set(split) == set(SUBTASKS)
        assert all(count > 0 for count in split.values())

    def test_spans_all_three_format_classes(self):
        """free_form, boolean and multiple_choice must all be represented.

        This is the actual reason the split exists -- BBH's subtasks are not one
        answer format, and the probe's whole job is to stop a format-dependent
        nondeterminism from hiding.
        """
        split = bbh_probe_split(PROBE_COMPOSITION["bbh"], SUBTASKS)
        assert split["boolean_expressions"] > 0            # boolean
        assert split["logical_deduction_five_objects"] > 0  # multiple_choice
        assert split["word_sorting"] > 0                   # free_form

    def test_remainder_goes_to_earliest_deterministically(self):
        """200 / 6 = 33 r 2 -> the first two subtasks get the extra item."""
        split = bbh_probe_split(200, SUBTASKS)
        assert [split[name] for name in SUBTASKS] == [34, 34, 33, 33, 33, 33]

    def test_is_a_pure_function_of_order(self):
        assert bbh_probe_split(200, SUBTASKS) == bbh_probe_split(200, SUBTASKS)

    def test_exact_division(self):
        assert list(bbh_probe_split(12, SUBTASKS).values()) == [2, 2, 2, 2, 2, 2]

    def test_fewer_items_than_subtasks_does_not_crash(self):
        """Only reachable in smoke mode, but it must not raise."""
        split = bbh_probe_split(2, SUBTASKS)
        assert sum(split.values()) == 2


class TestArmEngineFields:
    def test_bf16_is_not_in_specs_at_all(self):
        """The premise of the whole guard. If this ever changes, the guard is wrong."""
        specs, _ = load_specs(CONFIGS / "quant.yaml")
        assert "bf16" not in specs

    def test_bf16_maps_to_no_quant_method(self):
        specs, _ = load_specs(CONFIGS / "quant.yaml")
        assert arm_engine_fields("bf16", specs) == (None, 16)

    def test_bf16_does_not_raise_keyerror(self):
        """The trap: `specs["bf16"]` is a KeyError, not a spec with method 'none'."""
        with pytest.raises(KeyError):
            _ = {}["bf16"]
        assert arm_engine_fields("bf16", {}) == (None, 16)

    @pytest.mark.parametrize(
        ("arm", "method", "bits"),
        [
            ("gptq-4bit", "gptq", 4),
            ("gptq-8bit", "gptq", 8),
            ("awq-4bit", "awq", 4),
            ("bnb-nf4", "bitsandbytes", 4),
        ],
    )
    def test_quantized_arms_read_from_specs(self, arm, method, bits):
        specs, _ = load_specs(CONFIGS / "quant.yaml")
        assert arm_engine_fields(arm, specs) == (method, bits)

    def test_every_audit_arm_resolves(self):
        """No arm in the audit may be unmappable -- that would crash mid-job."""
        specs, _ = load_specs(CONFIGS / "quant.yaml")
        for arm in AUDIT_ARMS:
            method, bits = arm_engine_fields(arm, specs)
            assert bits in (4, 8, 16)
            assert method is None or isinstance(method, str)


class TestCellTag:
    def test_is_filesystem_safe(self):
        """It is used as a filename, so no '@' or '+' may survive."""
        tag = cell_tag("gptq-4bit", 64, None)
        assert tag == "qwen2.5-3b__gptq-4bit__at64__r3"
        assert not (set(tag) & set("@+/ "))

    def test_model_is_in_the_tag(self):
        """The Phi model-generality probe (R12.3.3) runs the SAME arms at the SAME
        level as the Qwen audit. Without the model key the cells collide and Phi
        resumes Qwen's result, reporting Qwen's numbers under Phi's name."""
        assert cell_tag("bf16", 8) != cell_tag("bf16", 8, model_key="phi-3.5-mini")

    def test_repeat_count_is_in_the_tag(self):
        """The 10-repeat confirmation (R12.3.2) runs the same arm at the same level
        as the 3-repeat sweep cell. Without the count it resumes the 3-repeat result
        and 'confirms' nothing."""
        assert cell_tag("bf16", 8, repeats=3) != cell_tag("bf16", 8, repeats=10)

    def test_arm_is_recoverable_from_the_tag(self):
        """The Marlin-probe filter parses the arm back out of the tag."""
        for arm in ("bf16", "gptq-4bit", "awq-4bit", "bnb-nf4"):
            assert cell_tag(arm, 8).split("__")[1] == arm
            assert cell_tag(arm, 8, "MarlinLinearKernel").split("__")[1] == arm

    def test_fallback_is_a_distinct_cell(self):
        """The Marlin-fallback probe must not overwrite the plain measurement.

        Both run at the floor for the same arm, so a shared tag would make the
        fallback silently replace the finding it is supposed to explain.
        """
        plain = cell_tag("gptq-4bit", SWEEP_FLOOR, None)
        fallback = cell_tag("gptq-4bit", SWEEP_FLOOR, "MarlinLinearKernel")
        assert plain != fallback

    def test_levels_are_distinct(self):
        tags = {cell_tag("bf16", level, None) for level in (8, 16, 32, 64)}
        assert len(tags) == 4

    def test_arms_are_distinct(self):
        tags = {cell_tag(arm, 8, None) for arm in ("bf16", "gptq-4bit", "awq-4bit")}
        assert len(tags) == 3
