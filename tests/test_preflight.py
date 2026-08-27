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

import dataclasses
import inspect
import os
from pathlib import Path

import pytest
import yaml

from quantartifact.generate import (
    REQUIRED_ENV,
    EngineSpec,
    PreflightError,
    VLLMRunner,
    assert_batch_invariance_engaged,
    build_parser,
    main,
    preflight,
    read_back_engine_state,
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


# ---------------------------------------------------------------------------
# DECLARED-BUT-UNENFORCED settings. This class of bug has now bitten twice:
#
#   * GPTQ's `desc_act: false` was in configs/quant.yaml but llmcompressor's
#     `actorder` default silently overrode it.
#   * `async_scheduling: false` was in configs/models.yaml but nothing forwarded it,
#     and the first real engine log read "Asynchronous scheduling is enabled".
#
# Both would have been REPORTED IN THE PAPER as though they held. So every serving
# value the config declares must be a real EngineSpec field, forwarded to the engine,
# and asserted by preflight.
# ---------------------------------------------------------------------------

def test_flashinfer_sampler_is_required_off():
    """Crashes without it (needs nvcc, which the image has no toolkit for), AND
    batch invariance only substitutes Triton kernels -- the flashinfer sampler is a
    separate CUDA path it does not cover."""
    assert REQUIRED_ENV["VLLM_USE_FLASHINFER_SAMPLER"] == "0"


def test_async_scheduling_defaults_off_and_is_asserted():
    spec = EngineSpec(model_repo="m", model_revision="a" * 40)
    assert spec.async_scheduling is False
    problems = preflight(
        EngineSpec(model_repo="m", model_revision="a" * 40, async_scheduling=True),
        strict=False,
    )
    assert any("async_scheduling" in p for p in problems)


def test_async_scheduling_is_forwarded_to_the_engine():
    """Declaring it is not enough -- it has to reach LLM(). Source-inspected because
    constructing a real engine needs a GPU, and this is exactly the gap that let
    vLLM's default win on the real box."""
    source = inspect.getsource(VLLMRunner)
    assert '"async_scheduling": self.spec.async_scheduling' in source


def test_every_required_env_var_is_asserted_by_preflight():
    """A new REQUIRED_ENV entry must actually gate the run, not just sit in a dict."""
    for name in REQUIRED_ENV:
        saved = os.environ.get(name)
        os.environ[name] = "definitely-wrong"
        try:
            problems = preflight(
                EngineSpec(model_repo="m", model_revision="a" * 40), strict=False
            )
            assert any(name in p for p in problems), f"{name} is not enforced"
        finally:
            if saved is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = saved


def test_models_yaml_serving_keys_that_are_engine_settings_exist_on_EngineSpec():
    """Anything configs/models.yaml declares under `serving:` and that names an
    engine knob must be a real EngineSpec field, or it is decoration that the paper
    would nonetheless report as frozen."""
    configs = Path(__file__).resolve().parents[1] / "configs"
    serving = yaml.safe_load((configs / "models.yaml").read_text())["serving"]
    fields = {f.name for f in dataclasses.fields(EngineSpec)}
    # Keys that are engine constructor knobs (as opposed to env vars, sampling
    # params, or audit bookkeeping).
    engine_knobs = {
        "tensor_parallel_size", "enable_prefix_caching", "max_model_len",
        "max_num_seqs", "gpu_memory_utilization", "seed", "enforce_eager",
        "async_scheduling", "dtype",
    }
    for key in engine_knobs & set(serving):
        assert key in fields, f"models.yaml declares serving.{key} but EngineSpec has no such field"


# ---------------------------------------------------------------------------
# R14.3 item 1 — "the single most important assertion in the study". Batch
# invariance being REQUESTED is not batch invariance being APPLIED, which is the
# whole async_scheduling lesson. These run without a GPU because the failure path
# (nothing patched) is exactly the state a laptop is in.
# ---------------------------------------------------------------------------

def test_batch_invariance_assertion_fails_when_nothing_was_patched():
    """On a machine where vLLM never ran, torch.bmm is unpatched -> must raise.

    That is the same observable state as a GPU job where
    enable_batch_invariant_mode() silently did not run, which is the case this
    guards: no substitution => H8 dead and every delta untrustworthy.
    """
    pytest.importorskip("torch")
    with pytest.raises(PreflightError, match="BATCH INVARIANCE DID NOT ENGAGE"):
        assert_batch_invariance_engaged()


def test_batch_invariance_assertion_names_what_it_checked():
    """The error must be diagnosable from the log alone -- there is no rerunning a
    queued multi-hour job to add print statements."""
    pytest.importorskip("torch")
    try:
        assert_batch_invariance_engaged()
    except PreflightError as exc:
        message = str(exc)
        assert "torch.bmm" in message
        assert "observed:" in message
    else:
        pytest.skip("batch invariance is engaged in this environment")


def test_read_back_survives_an_unreadable_engine():
    """An introspection failure is NOT a config disagreement.

    vLLM moves this attribute between versions and it cannot be exercised locally, so
    a missing path must report loudly and let the job continue; only a real mismatch
    is fatal. Killing a queued multi-hour job because an attribute moved is the wrong
    trade.
    """
    class Bare:
        pass

    spec = EngineSpec(model_repo="m", model_revision=GOOD_SHA)
    observed = read_back_engine_state(Bare(), spec)
    assert observed["read_back"] == "unavailable"
    assert observed["requested_max_num_seqs"] == spec.max_num_seqs


def test_read_back_is_fatal_on_a_real_mismatch():
    """A silently reduced max_num_seqs corrupts the exact quantity A7 freezes."""
    class Sched:
        max_num_seqs = 1          # engine silently reduced it
        async_scheduling = False

    class Config:
        scheduler_config = Sched()
        cache_config = None
        model_config = None

    class Engine:
        vllm_config = Config()

    class Llm:
        llm_engine = Engine()

    spec = EngineSpec(model_repo="m", model_revision=GOOD_SHA, max_num_seqs=8)
    with pytest.raises(PreflightError, match="max_num_seqs"):
        read_back_engine_state(Llm(), spec)
