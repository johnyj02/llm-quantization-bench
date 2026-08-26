"""Tests for the quantization recipe mapping — the `desc_act`/`actorder` trap.

Context: the first `quantize` smoke run failed with pydantic
`extra_forbidden` on `group_size` and `desc_act`, because in llmcompressor
0.12.0.1 neither is a keyword on either modifier. Fixing that surfaced a worse,
*silent* problem, which is what these tests guard.

`GPTQModifier.actorder` defaults to `Sentinel("static")`, and
`resolve_quantization_config` treats that sentinel as deferential: it injects
`ActivationOrdering.STATIC` whenever the scheme's actorder is `None`. Because
"explicitly None" and "never set" are indistinguishable there, **omitting
`actorder` silently produces static activation reordering** — neither the
`desc_act: false` that `configs/quant.yaml` requires (matching Qwen's official
checkpoint) nor the `desc_act: true` it explicitly rejects.

llmcompressor's own W4A16 example omits `actorder`, so the canonical tutorial
recipe has this property. Anchor A is a **blocking** gate against Qwen's
`desc_act: false` checkpoint, so inheriting that default would have quietly
invalidated it while every test still passed.

These run without llmcompressor installed: `gptq_actorder` only imports
`compressed_tensors` on the `desc_act=True` branch, which our config never takes.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from quantartifact.quantize import (
    QuantConfigError,
    QuantSpec,
    build_recipe,
    gptq_actorder,
    load_specs,
    validate_normalisation,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def gptq_spec(**overrides) -> QuantSpec:
    base = {
        "key": "gptq-4bit",
        "method": "gptq",
        "scheme": "W4A16",
        "bits": 4,
        "symmetric": True,
        "group_size": 128,
        "desc_act": False,
    }
    return QuantSpec(**{**base, **overrides})


class TestGPTQActorder:
    def test_desc_act_false_is_none_not_static(self):
        """The whole point. `None` means no reordering; STATIC is a real permutation.

        If this ever returns a Sentinel, the string "static", or
        `ActivationOrdering.WEIGHT`, we are silently not running the recipe the
        paper claims to run.
        """
        assert gptq_actorder(gptq_spec(desc_act=False)) is None

    def test_returns_none_without_importing_compressed_tensors(self):
        """Must stay importable in the analysis venv, which has no compressed-tensors.

        The `desc_act=False` branch returns before the lazy import; this test only
        passes because that ordering holds.
        """
        assert gptq_actorder(gptq_spec(desc_act=False)) is None

    def test_every_configured_gptq_arm_resolves_to_none(self):
        """configs/quant.yaml sets desc_act:false on every GPTQ arm. Pin that."""
        specs, _ = load_specs(CONFIGS / "quant.yaml")
        gptq_arms = [s for s in specs.values() if s.method == "gptq"]
        assert gptq_arms, "expected GPTQ arms in the config"
        for spec in gptq_arms:
            assert spec.desc_act is False, f"{spec.key} unexpectedly sets desc_act"
            assert gptq_actorder(spec) is None, f"{spec.key} would get reordering"


class TestBuildRecipe:
    def test_bitsandbytes_has_no_offline_recipe(self):
        """NF4 is quantized in-flight by vLLM; asking for a recipe is a bug."""
        spec = gptq_spec(key="bnb-nf4", method="bitsandbytes", scheme="NF4",
                         in_flight=True)
        with pytest.raises(QuantConfigError, match="IN-FLIGHT"):
            build_recipe(spec)

    def test_unknown_method_is_refused(self):
        with pytest.raises(QuantConfigError, match="unknown method"):
            build_recipe(gptq_spec(method="int4-magic"))

    def test_no_forbidden_kwargs_are_passed_to_modifiers(self):
        """Regression guard for the smoke failure, by source inspection.

        `group_size=` and `desc_act=` as modifier keywords raise pydantic
        `extra_forbidden` in llmcompressor 0.12.0.1. We cannot construct a
        modifier here (no llmcompressor in this venv), so assert on the source of
        `build_recipe` itself -- which is exactly the mistake that reached a paid
        GPU job, and is worth catching in 0.1s instead.
        """
        source = inspect.getsource(build_recipe)
        # Strip the docstring: it discusses these names deliberately.
        body = source.split('"""')[-1]
        assert "group_size=" not in body, "group_size= is not a valid modifier kwarg"
        assert "desc_act=" not in body, "desc_act= is not a valid modifier kwarg"
        assert "actorder=" in body, (
            "actorder MUST be passed explicitly -- omitting it silently yields "
            "static activation reordering via Sentinel('static')"
        )


class TestNormalisationStillHolds:
    def test_desc_act_true_is_still_refused(self):
        """The existing guard must survive the refactor."""
        specs, calibration = load_specs(CONFIGS / "quant.yaml")
        specs["gptq-4bit"] = gptq_spec(desc_act=True)
        with pytest.raises(QuantConfigError, match="desc_act"):
            validate_normalisation(specs, calibration)

    def test_config_group_size_is_128_everywhere(self):
        """assert_preset_matches_spec compares against these; pin the expectation."""
        specs, _ = load_specs(CONFIGS / "quant.yaml")
        for spec in specs.values():
            if spec.in_flight:
                continue
            assert spec.group_size == 128, f"{spec.key} group_size drifted"
            assert spec.symmetric is True, f"{spec.key} symmetry drifted"

    def test_preset_names_are_the_ones_we_assert_against(self):
        """Only presets whose contents we verified upstream may be used."""
        specs, _ = load_specs(CONFIGS / "quant.yaml")
        verified = {"W4A16", "W8A16"}
        for spec in specs.values():
            if spec.in_flight:
                continue
            assert spec.scheme in verified, (
                f"{spec.key} uses preset {spec.scheme!r}, whose resolved args were "
                f"not verified against compressed-tensors 0.17.1"
            )
