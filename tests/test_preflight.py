"""Preflight assertion tests — the determinism invariants, checked on a laptop.

These matter more than they look. A run that proceeds with a broken serving
invariant produces numbers that look publishable and are not: uncontrolled batch
nondeterminism can manufacture or erase an effect of the size this paper
measures (vLLM #9567 reported exactly that on an INT8-quantized Llama-3-8B, tied
to unstable lm-eval scores). So the failure has to be loud and it has to happen
before a single token is generated.

Also tested: the D10 confirmation gate, because a hard human gate that can be
tripped past by habit is not a gate.
"""

from __future__ import annotations

import pytest

from quantartifact.generate import (
    REQUIRED_ENV,
    EngineSpec,
    PreflightError,
    build_parser,
    main,
    preflight,
)

GOOD_SHA = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


@pytest.fixture
def clean_env(monkeypatch):
    for name, value in REQUIRED_ENV.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


def spec(**overrides) -> EngineSpec:
    defaults = dict(model_repo="Qwen/Qwen2.5-3B-Instruct", model_revision=GOOD_SHA)
    defaults.update(overrides)
    return EngineSpec(**defaults)  # type: ignore[arg-type]


def problems_for(the_spec: EngineSpec) -> str:
    return "\n".join(preflight(the_spec, strict=False))


# ---------------------------------------------------------------------------
# The single most important setting
# ---------------------------------------------------------------------------


def test_tensor_parallel_greater_than_one_is_rejected(clean_env):
    """Batch invariance is BROKEN under TP>1 (vLLM #51290).

    This is why the plan says not to rent a multi-GPU box "for headroom" -- it
    invites exactly this mistake.
    """
    found = problems_for(spec(tensor_parallel_size=2))
    assert "tensor_parallel_size" in found
    assert "#51290" in found


def test_batch_invariant_unset_is_rejected(clean_env):
    clean_env.delenv("VLLM_BATCH_INVARIANT", raising=False)
    assert "VLLM_BATCH_INVARIANT" in problems_for(spec())


def test_batch_invariant_zero_is_rejected(clean_env):
    clean_env.setenv("VLLM_BATCH_INVARIANT", "0")
    assert "VLLM_BATCH_INVARIANT" in problems_for(spec())


def test_v1_multiprocessing_must_be_off(clean_env):
    clean_env.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    assert "VLLM_ENABLE_V1_MULTIPROCESSING" in problems_for(spec())


def test_prefix_caching_is_rejected_with_the_reason(clean_env):
    """It would help throughput a LOT here, which is exactly why it is a trap.

    Few-shot CoT prompts share a long identical prefix, so prefix caching is the
    single biggest speedup available -- and it changes greedy output
    (vLLM #39389). The rejection message has to say so, or someone will
    reasonably re-enable it later.
    """
    found = problems_for(spec(enable_prefix_caching=True))
    assert "enable_prefix_caching" in found
    assert "#39389" in found


def test_enforce_eager_is_rejected_as_pure_throughput_loss(clean_env):
    """CUDA graphs became BI-compatible in PR #27660; eager buys nothing."""
    found = problems_for(spec(enforce_eager=True))
    assert "enforce_eager" in found
    assert "#27660" in found


def test_nonzero_seed_is_rejected(clean_env):
    assert "seed" in problems_for(spec(seed=1234))


def test_tag_instead_of_sha_is_rejected(clean_env):
    """Tags move. Reproducibility holds only within identical versions."""
    assert "commit SHA" in problems_for(spec(model_revision="main"))


def test_clean_spec_has_no_config_problems(clean_env):
    """Only the toolchain check may fire locally; no CONFIG problem should."""
    found = preflight(spec(), strict=False)
    config_problems = [
        problem for problem in found if "compiler" not in problem.lower()
    ]
    assert config_problems == [], config_problems


def test_strict_mode_raises_rather_than_returning(clean_env):
    with pytest.raises(PreflightError, match="preflight FAILED"):
        preflight(spec(tensor_parallel_size=8), strict=True)


def test_all_problems_are_reported_at_once(clean_env):
    """Report every violation, not just the first -- one round trip, not five."""
    found = preflight(
        spec(tensor_parallel_size=4, enable_prefix_caching=True, seed=99, enforce_eager=True),
        strict=False,
    )
    joined = "\n".join(found)
    for expected in ("tensor_parallel_size", "enable_prefix_caching", "seed", "enforce_eager"):
        assert expected in joined


# ---------------------------------------------------------------------------
# The Triton toolchain landmine
# ---------------------------------------------------------------------------


def test_missing_c_compiler_is_reported(clean_env, monkeypatch):
    """No C compiler => no batch-invariant Triton kernels => no determinism.

    VLLM_BATCH_INVARIANT works by substituting Triton kernels that are
    JIT-compiled at RUNTIME. The sibling project hit this for real after a build
    toolchain was purged post-pip-install (notes/reuse.md §1). Cheap to check,
    fatal to miss: it would silently invalidate every delta in the paper.
    """
    monkeypatch.setattr("quantartifact.generate.shutil.which", lambda _name: None)
    found = problems_for(spec())
    assert "no C compiler" in found
    assert "JIT" in found


# ---------------------------------------------------------------------------
# Decision D10 -- the human confirmation gate
# ---------------------------------------------------------------------------


def test_run_refuses_without_confirmation(capsys):
    """D10: no generation starts without explicit user confirmation."""
    code = main(["--cell", "qwen|bf16", "--task", "gsm8k_cot", "--run-id", "main-v1"])
    assert code == 2
    assert "REFUSING TO RUN" in capsys.readouterr().err


def test_preflight_only_is_allowed_without_confirmation():
    """Validating configuration costs nothing and starts nothing."""
    code = main(
        ["--cell", "qwen|bf16", "--task", "gsm8k_cot", "--run-id", "main-v1", "--preflight-only"]
    )
    assert code == 0


def test_confirmation_flag_exists_and_is_not_defaulted_on():
    parser = build_parser()
    args = parser.parse_args(["--cell", "c", "--task", "gsm8k_cot", "--run-id", "r"])
    assert args.i_have_confirmation is False
