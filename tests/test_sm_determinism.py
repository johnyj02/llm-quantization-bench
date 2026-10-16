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

import inspect
import json
from pathlib import Path

import pytest
import sm_determinism
from sm_determinism import AUDIT_ARMS, arm_engine_fields, bbh_probe_split, cell_tag

from quantartifact.determinism import (
    ORDERING_SEEDS,
    PROBE_COMPOSITION,
    SWEEP_FLOOR,
    RepeatOutputs,
    compute_disagreement,
    order_sensitivity,
)
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


class TestSinglePassCells:
    """A single-pass cell has nothing to compare against itself.

    The ordering confirmation (R13.1) deliberately runs ONE pass per ordering and
    compares the orderings afterwards. `run_probe_cell` was calling
    `compute_disagreement` on every cell regardless, so a 1-repeat cell raised
    "need at least 2 repeats" and killed the job after it had already produced good
    data — it conflated "run the passes" with "compare the passes".
    """

    def test_compute_disagreement_still_rejects_one_pass(self):
        """The guard itself is correct and must stay: one pass is not a comparison."""
        single = [RepeatOutputs(fmt="bf16", max_num_seqs=8, repeat_idx=0,
                                outputs={"a": "x"})]
        with pytest.raises(ValueError, match="at least 2 repeats"):
            compute_disagreement(single)

    def test_three_orderings_are_comparable(self):
        """The confirmation combines ORDERING_SEEDS passes, which must be >= 2."""
        assert len(ORDERING_SEEDS) >= 2, "the ordering comparison needs >=2 passes"


class TestOneEngineForAllPasses:
    """R15.1 — every pass must share ONE engine, or ordering is confounded.

    My first version gave each ordering its own subprocess. The planner's correction:
    my argument that bf16's stability ruled out process-state variation was UNSOUND,
    because it assumed both arms are equally sensitive to whatever varies across
    processes — and GPTQ's Marlin path could be sensitive to workspace addresses or
    cuBLAS handle state in ways bf16 is not. The mechanism decides the write-up, so the
    confound has to be designed out rather than argued away.
    """

    def test_probe_script_loops_passes_inside_one_runner(self):
        script = sm_determinism.PROBE_SCRIPT
        runner_init = script.index("VLLMRunner(spec)")
        pass_loop = script.index('for index, spec_pass in enumerate(payload["passes"])')
        assert runner_init < pass_loop, (
            "the engine must be constructed BEFORE the pass loop, or each ordering "
            "gets its own engine and ordering is confounded with process state"
        )
        # And only one engine may be built.
        assert script.count("VLLMRunner(") == 1

    def test_probe_script_shuffles_before_grouping(self):
        """Shuffling after grouping would change call order but not CO-RESIDENCY,
        which is the actual mechanism."""
        script = sm_determinism.PROBE_SCRIPT
        shuffle = script.index("shuffle(items)")
        grouping = script.index("groups.setdefault(")
        assert shuffle < grouping


class TestTwoFloorsFromOneCell:
    """R15.6 — three numbers, three names. Conflating them misreports either our
    noise or the field's."""

    def _passes(self, order_diverges: bool):
        base = {"a": "1", "b": "2", "c": "3"}
        out = [
            RepeatOutputs(fmt="gptq-4bit", max_num_seqs=8, repeat_idx=0,
                          outputs=dict(base), ordering_seed=0),
            RepeatOutputs(fmt="gptq-4bit", max_num_seqs=8, repeat_idx=1,
                          outputs=dict(base), ordering_seed=0),
        ]
        shuffled = dict(base)
        if order_diverges:
            shuffled["b"] = "DIFFERENT"
        out.append(RepeatOutputs(fmt="gptq-4bit", max_num_seqs=8, repeat_idx=2,
                                 outputs=shuffled, ordering_seed=20260827))
        return out

    def test_repetition_floor_is_measured_from_same_ordering_only(self):
        result = order_sensitivity(self._passes(order_diverges=True))
        assert result["repetition_passes"] == 2
        assert result["repetition_floor"] == 0.0
        assert result["repetition_stable"] is True

    def test_order_divergence_is_measured_across_orderings(self):
        result = order_sensitivity(self._passes(order_diverges=True))
        assert result["order_invariant"] is False
        assert result["order_n_disagreeing"] == 1

    def test_the_exact_state_the_smoke_found_is_flagged(self):
        """Stable under repetition, unstable under reordering.

        This is what the 20-item smoke measured for gptq-4bit (0/20 on repeats,
        13/20 on reordering) and what R8's retired criterion called "stable". If this
        flag ever stops firing, the freeze rule silently reverts to the dead one.
        """
        result = order_sensitivity(self._passes(order_diverges=True))
        assert result["repeat_stable_but_order_sensitive"] is True

    def test_order_invariant_arm_is_not_flagged(self):
        """bf16's observed behaviour: stable on both axes."""
        result = order_sensitivity(self._passes(order_diverges=False))
        assert result["order_invariant"] is True
        assert result["repeat_stable_but_order_sensitive"] is False


class TestOrderSweepLevels:
    def test_sweep_is_on_the_ordering_axis(self):
        """R15.4: R8's repeat-based sweep measured an axis with no variance."""
        assert sm_determinism.ORDER_SWEEP_LEVELS == (8, 32, 64)

    def test_smoke_includes_awq_the_prediction_test(self):
        """R15.2: AWQ HAS the guard GPTQ lacks, so it is the arm that converts H8
        from an observation into a mechanism."""

        source = inspect.getsource(sm_determinism.main)
        smoke_arms = source[source.index('arms = ["bf16"'):]
        assert "awq-4bit" in smoke_arms.split("]")[0]


class TestOrderingsMode:
    """R17.3 / prereg §9a: 10 orderings at the frozen batch size.

    The derived-value wiring is pinned because this project has now been bitten five
    times by a value set in one place and silently not honoured in another
    (`desc_act`, `async_scheduling`, `add_bos_token`, the pilot's `until`, job 1b's
    destination path). `orderings` is the sixth candidate: set in `main()`, and needed by
    `measure_cell`, by the pass-count arithmetic, by the log line and by the report.
    """

    SOURCE = (Path(__file__).resolve().parents[1] / "sm_determinism.py").read_text()

    def test_extended_seeds_are_ten_and_prefix_the_original_three(self):
        from quantartifact.determinism import (  # noqa: PLC0415
            ORDERING_SEEDS,
            ORDERING_SEEDS_EXTENDED,
        )
        assert len(ORDERING_SEEDS_EXTENDED) == 10
        assert ORDERING_SEEDS_EXTENDED[:3] == ORDERING_SEEDS, (
            "the 3-ordering result must stay recoverable as a prefix"
        )
        assert len(set(ORDERING_SEEDS_EXTENDED)) == 10, "seeds must be distinct"
        assert ORDERING_SEEDS_EXTENDED[0] == 0, "ordering 0 is the unshuffled reference"

    def test_active_orderings_reach_measure_cell(self):
        assert "orderings=orderings" in self.SOURCE

    def test_pass_count_uses_the_active_orderings(self):
        assert "len([s for s in orderings if s != 0])" in self.SOURCE, (
            "a stale count would misreport the run's size by 7 passes per cell"
        )

    def test_report_records_the_active_orderings_and_the_mode(self):
        assert '"orderings": list(orderings)' in self.SOURCE
        assert '"sd_mode": SD_MODE' in self.SOURCE

    def test_smoke_plus_orderings_keeps_the_orderings_shape(self):
        """A rehearsal must exercise the path it rehearses, not a third configuration."""
        assert "if SMOKE and SD_MODE:" in self.SOURCE
        assert "elif SMOKE:" in self.SOURCE

    def test_bf16_is_included_so_the_control_gets_the_same_orderings(self):
        assert '"bf16", "gptq-4bit", "awq-4bit", "bnb-nf4"' in self.SOURCE


class TestPreregAndCodeAgree:
    """🚨 The prereg is the one document where declared-vs-enforced drift is fatal.

    Its entire value is the timestamp: if `analysis/preregistration.md` §9a names one set
    of ordering seeds and the code runs another, the registration certifies nothing and a
    reviewer who notices caps acceptance at ~25% (`EXECUTOR-PLAN.md` §4.0). Given this
    project has been bitten six times by a value declared in one place and not honoured in
    another, the prereg gets the same treatment as the engine settings.
    """

    PREREG = (Path(__file__).resolve().parents[1]
              / "analysis" / "preregistration.md").read_text()

    def test_every_registered_ordering_seed_is_in_the_code(self):
        from quantartifact.determinism import ORDERING_SEEDS_EXTENDED  # noqa: PLC0415

        # The seed row pins them as a backtick-quoted comma list; parse rather than trust.
        row = next(line for line in self.PREREG.splitlines()
                   if "Seeds, PINNED HERE" in line)
        registered = [int(s.strip()) for s in
                      row.split("`")[1].split(",")]
        assert registered == list(ORDERING_SEEDS_EXTENDED), (
            f"prereg §9a registers {registered} but the code runs "
            f"{list(ORDERING_SEEDS_EXTENDED)}"
        )

    def test_prereg_registers_ten_orderings(self):
        assert "**10**" in self.PREREG

    def test_prereg_names_sd_as_the_estimand_not_range(self):
        """R17.3: range is unstable and grows with n, so it cannot carry a population
        claim. If the prereg ever says otherwise, the analysis is unregistered."""
        assert "Standard deviation of accuracy across orderings is the inferential" \
            in self.PREREG
        # And that the range is explicitly demoted, since that is the part a later editor
        # would be tempted to soften.
        assert "The range is NOT an estimand" in self.PREREG

    def test_prereg_records_that_the_10_ordering_run_supersedes(self):
        """The 3-ordering spreads were attenuated by inert BBH items, so pooling them
        would mix two populations. Registered before the run so the supersession cannot
        look like a post-hoc choice."""
        assert "SUPERSEDES THE 3-ORDERING FIGURES" in self.PREREG

    def test_frozen_batch_size_matches_the_audited_value(self):
        import yaml  # noqa: PLC0415

        configs = Path(__file__).resolve().parents[1] / "configs"
        serving = yaml.safe_load((configs / "models.yaml").read_text())["serving"]
        assert serving["max_num_seqs_resolved_by_audit"] is True
        frozen = serving["max_num_seqs"]
        # §9's freeze statement and §9a's design table must BOTH name it, so the value
        # cannot drift in one place while the other still certifies the old one.
        assert f"`max_num_seqs = {frozen}` is frozen" in self.PREREG, (
            f"§9 must record the frozen value {frozen}"
        )
        assert f"**{frozen} only** (the frozen serving value)" in self.PREREG, (
            f"§9a's design table must pin {frozen}"
        )


class TestCellTagDistinguishesOrderingCount:
    """🚨 REGRESSION, the R17.3 run (2026-08-30, 417 s, $0.32).

    The run announced *"4 arms x 10 orderings"* and `passes/cell : 11`, then printed
    `[skip ] qwen2.5-3b__bf16__at64__r2  already measured` for every cell, reused job 2's
    THREE-ordering results, and exited `Completed`. An SD computed from that output would
    have been a fabricated statistic in the paper.

    Cause: the redesign runs every ordering inside ONE cell, so `ordering_seed` is 0 and
    contributes nothing to the tag -- making a 3-ordering and a 10-ordering cell share a
    name. `cell_tag`'s own docstring already stated the governing principle ("anything that
    changes what a cell measures must change its name") and listed three cases; the
    ordering COUNT was not one of them.

    ⚠️ The smoke could not have caught it: `--smoke` writes to a separate prefix and so had
    no prior cells to collide with. **A rehearsal in a clean prefix cannot detect a cache
    collision in a dirty one** -- which is why this is pinned in a test instead.
    """

    def test_three_and_ten_ordering_cells_have_different_tags(self):
        three = sm_determinism.cell_tag("bf16", 64, n_orderings=3)
        ten = sm_determinism.cell_tag("bf16", 64, n_orderings=10)
        assert three != ten, (
            "a 3-ordering and a 10-ordering cell are different measurements and must not "
            "share a resume key"
        )

    def test_default_ordering_count_leaves_the_tag_unchanged(self):
        """Existing filenames must stay valid, and reanalyze's parser reads segments 0-2."""
        from quantartifact.determinism import ORDERING_SEEDS  # noqa: PLC0415

        assert sm_determinism.cell_tag("bf16", 64) == \
            sm_determinism.cell_tag("bf16", 64, n_orderings=len(ORDERING_SEEDS))

    def test_measure_cell_passes_the_ordering_count(self):
        source = (Path(__file__).resolve().parents[1] / "sm_determinism.py").read_text()
        assert "n_orderings=len(orderings)" in source, (
            "the tag is only correct if the ACTIVE ordering count reaches it"
        )

    def test_the_extended_run_gets_a_distinct_tag(self):
        from quantartifact.determinism import ORDERING_SEEDS_EXTENDED  # noqa: PLC0415

        tag = sm_determinism.cell_tag(
            "bf16", 64, n_orderings=len(ORDERING_SEEDS_EXTENDED))
        assert tag.endswith("ord10x"), tag


class TestDecodingFingerprint:
    """🚨 THE ROOT FIX for a bug family this project hit EIGHT times: a cached unit of work
    keyed on an identity that omits something determining what the work produces.

    `desc_act`, `async_scheduling`, `add_bos_token` (declared, never forwarded); the pilot's
    per-level `until` (forwarded, not in the key); job 1b's destination path (derived in the
    callee); the determinism resume key (no ordering count); the diagnose run_id (no probe
    index). Each was patched individually. The general rule: **anything that changes what a
    cell produces must change the cell's identity.**
    """

    def test_env_hash_was_never_a_config_fingerprint(self):
        """It covers {gpu, cuda} only, which is why it stayed byte-identical across BOTH
        the BBH terminator change and the sampling change. Documented so nobody reaches
        for it again."""
        gen = (Path(__file__).resolve().parents[1] / "src" / "quantartifact"
               / "generate.py").read_text()
        assert 'env_hash({"gpu": hardware["gpu"], "cuda": hardware["cuda"]})' in gen

    def test_fingerprint_changes_when_the_sampling_config_changes(self):
        from quantartifact.generate import (  # noqa: PLC0415
            EngineSpec,
            decoding_fingerprint,
        )
        base = EngineSpec(model_repo="m", model_revision="r")
        # generation_config is part of what decoding IS.
        other = EngineSpec(model_repo="m", model_revision="r", generation_config="auto")
        assert decoding_fingerprint(base) != decoding_fingerprint(other)

    def test_fingerprint_changes_with_disabled_kernels(self):
        """The Marlin-off cells are a different measurement and must not share a key."""
        from quantartifact.generate import (  # noqa: PLC0415
            EngineSpec,
            decoding_fingerprint,
        )
        a = EngineSpec(model_repo="m", model_revision="r")
        b = EngineSpec(model_repo="m", model_revision="r",
                       disabled_kernels="MarlinLinearKernel")
        assert decoding_fingerprint(a) != decoding_fingerprint(b)

    def test_fingerprint_is_stable_for_the_same_config(self):
        from quantartifact.generate import (  # noqa: PLC0415
            EngineSpec,
            decoding_fingerprint,
        )
        spec = EngineSpec(model_repo="m", model_revision="r")
        assert decoding_fingerprint(spec) == decoding_fingerprint(spec)

    def test_cell_tag_includes_it_and_so_invalidates_old_cells(self):
        from quantartifact.generate import (  # noqa: PLC0415
            EngineSpec,
            decoding_fingerprint,
        )
        fp = decoding_fingerprint(EngineSpec(model_repo="m", model_revision="r"))
        assert sm_determinism.cell_tag("bf16", 64) != \
            sm_determinism.cell_tag("bf16", 64, decoding=fp)
        assert sm_determinism.cell_tag("bf16", 64, decoding=fp).endswith(f"dec{fp}")

    def test_measure_cell_passes_a_fingerprint(self):
        source = (Path(__file__).resolve().parents[1] / "sm_determinism.py").read_text()
        assert "decoding=fingerprint" in source

    def test_the_assertion_and_the_fingerprint_share_one_source_of_truth(self):
        """If the asserted sampling values and the hashed ones could diverge, the key
        would certify a configuration that was never enforced."""
        gen = (Path(__file__).resolve().parents[1] / "src" / "quantartifact"
               / "generate.py").read_text()
        assert "for field, allowed in NEUTRAL_SAMPLING.items()" in gen
        assert '"sampling": {k: v[0] for k, v in NEUTRAL_SAMPLING.items()}' in gen


class TestOrderingsModeReadsTheFrozenValue:
    """🚨 REGRESSION (R24.1, 2026-08-31). A frozen value must be READ, never re-derived.

    The orderings mode's level was `max(ORDER_SWEEP_LEVELS)`. That equalled the frozen
    value while the freeze was 64 and diverged the instant R24.1 moved it to 32 -- the
    sweep still measures {8, 32, 64}, so it would have re-run at the value the study just
    abandoned for exceeding its own pre-registered trigger, in the table H8 LEADS with.

    Third instance of a DERIVED value standing in for a frozen one, after
    `\\hEightFrozenB` in the table generator and `max_num_seqs` missing from the decoding
    fingerprint.
    """

    SOURCE = (Path(__file__).resolve().parents[1] / "sm_determinism.py").read_text()

    def test_it_calls_the_frozen_getter(self):
        assert "levels = [frozen_max_num_seqs()]" in self.SOURCE

    def test_it_does_NOT_derive_the_level_from_the_sweep(self):
        assert "levels = [max(ORDER_SWEEP_LEVELS)]" not in self.SOURCE, (
            "the orderings level must be read from models.yaml, not derived from the "
            "sweep's levels -- those diverged when the freeze moved"
        )

    def test_the_frozen_getter_still_fails_closed(self):
        """It must refuse an unaudited value rather than print one into the paper."""
        pilot = (Path(__file__).resolve().parents[1] / "sm_pilot.py").read_text()
        assert "max_num_seqs_resolved_by_audit" in pilot
        assert "REFUSING TO RUN" in pilot


class TestSpotcheckGpuGuard:
    """The GPU is not in the cell tag, so a cell measured on the L40S has the same name as
    the spot-check's cell on another GPU. Reusing it would report L40S numbers as the
    second GPU's replication, and the job would exit `Completed`."""

    def _cell(self, tmp_path, monkeypatch, stored_gpu, box_gpu):
        monkeypatch.setattr(sm_determinism, "PROBE_DIR", tmp_path)
        monkeypatch.setattr(sm_determinism, "cell_tag", lambda *a, **k: "T")
        monkeypatch.setattr(sm_determinism, "current_gpu", lambda: box_gpu)
        (tmp_path / "T.json").write_text(json.dumps({"hardware": {"gpu": stored_gpu}, "ok": 1}))
        return sm_determinism.measure_cell("bf16", 32, [], {}, {"revision": "r"},
                                           {"bf16": ("m", None)})

    def test_refuses_a_cell_from_another_gpu(self, tmp_path, monkeypatch):
        with pytest.raises(SystemExit, match="REFUSING TO REUSE"):
            self._cell(tmp_path, monkeypatch, "NVIDIA L40S", "NVIDIA A10G")

    def test_reuses_a_cell_from_the_same_gpu(self, tmp_path, monkeypatch):
        assert self._cell(tmp_path, monkeypatch, "NVIDIA A10G", "NVIDIA A10G")["ok"] == 1
