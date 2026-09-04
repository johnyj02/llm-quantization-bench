"""Generation core — THE ONLY component in this project that touches a GPU.

Everything else in the study is a pure function over what this module writes.
That separation is the one architectural rule that matters: **generate once,
score many times.**

Three properties this module must guarantee, in priority order:

1. **Every record carries ``finish_reason`` and ``n_gen_tokens``.** They are
   unreconstructable afterwards and H1 is impossible without them.
   ``records.GenerationRecord.validate()`` refuses to write a record missing
   either, so a silent None cannot reach disk.
2. **The serving configuration is asserted, not assumed.** ``preflight()``
   fails loudly before a single token is generated if batch invariance is off,
   if ``tensor_parallel_size != 1``, or if Triton cannot JIT (see below). A run
   that silently drifts is worse than a run that does not happen, because the
   numbers look fine.
3. **Resumable and idempotent.** Re-running a partially-complete cell skips
   already-generated join keys and re-running a complete one does zero work.

⚠️ **Nothing here starts automatically.** Per decision D10 (``notes/decisions.md``)
no generation run begins without explicit user confirmation. ``main()`` requires
``--i-have-confirmation`` and refuses otherwise; that flag exists to make the
gate impossible to trip past by accident, including by me.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quantartifact.records import (
    GenerationRecord,
    append_records,
    assert_cell_reusable,
    build_resume_index,
    cell_path,
    env_hash,
    prompt_sha256,
)

# 🚨 DECODING FINGERPRINT — the root fix for a bug family this project has now hit EIGHT
# times: a cached or resumed unit of work keyed on an identity that does not include
# everything determining what the work produces.
#
#   desc_act, async_scheduling, add_bos_token   values declared and never forwarded
#   the pilot's per-level `until`               forwarded but not in the cache key
#   job 1b's destination path                   derived in the callee, unoverridable
#   the determinism resume key                  did not encode the ordering COUNT
#   the diagnose run_id                         did not encode the probe index
#
# Each was patched individually. The general statement is: **anything that changes what a
# cell produces must change the cell's identity.** `env_hash` does not do this — it covers
# only `{gpu, cuda}`, which is why it stayed byte-identical across the BBH terminator change
# AND the sampling-parameter change. It is a hardware fingerprint, not a configuration one.
#
# So: bump DECODING_VERSION whenever anything below changes, and include the fingerprint in
# any resume key. A stale cache then becomes impossible rather than merely unlikely.
#
# ⚠️ Deliberately NOT added to the per-record schema. The fingerprint goes in the job REPORT
# instead, where it is equally auditable. Revisit at the artifact release (A24).
# ⚠️ The original reason given here was "it would bump `schema_version`". That reason is now
# spent: `schema_version` moved 1.0.0 -> 1.1.0 anyway to carry `invocation_id`, which R21.3
# requires per RECORD because it must be joinable per item. The fingerprint is a property of
# the whole cell, so the report is still the right home — but the argument is now "the report
# is the right level", not "we cannot afford the bump".
DECODING_VERSION = "4"   # 1 -> 2: pinned top_k/penalties, generation_config="vllm"
#                        # 2 -> 3: max_num_seqs ADDED to the fingerprint (see below)
#                        # 3 -> 4: `until` moved from a generation-time `stop=` to SCORING
#                        #         time (R34). This genuinely changes what a generation IS --
#                        #         the text now runs past the terminator -- so every cached
#                        #         cell is correctly invalidated. That invalidation is the
#                        #         real price of the fix and it is the intended behaviour.

# The six sampling fields, pinned. `top_k` has two sentinel spellings for "disabled" and
# vLLM normalises between them, so both are accepted on read-back.
NEUTRAL_SAMPLING: dict[str, tuple] = {
    "temperature": (0.0,),
    "top_p": (1.0,),
    "top_k": (-1, 0),
    "repetition_penalty": (1.0,),
    "presence_penalty": (0.0,),
    "frequency_penalty": (0.0,),
}


def decoding_fingerprint(spec) -> str:
    """Short hash of everything that determines what a generation IS.

    Include this in any cache or resume key. If it changes, previously cached cells were
    produced by a different decoding configuration and must not be reused.
    """
    import hashlib  # noqa: PLC0415 - only needed here
    import json as _json  # noqa: PLC0415

    payload = _json.dumps({
        "decoding_version": DECODING_VERSION,
        "sampling": {k: v[0] for k, v in NEUTRAL_SAMPLING.items()},
        "generation_config": getattr(spec, "generation_config", None),
        "required_env": REQUIRED_ENV,
        "batch_invariant": True,
        "seed": getattr(spec, "seed", None),
        "enable_prefix_caching": getattr(spec, "enable_prefix_caching", None),
        "async_scheduling": getattr(spec, "async_scheduling", None),
        "enforce_eager": getattr(spec, "enforce_eager", None),
        "disabled_kernels": getattr(spec, "disabled_kernels", None),
        # R34: WHERE `until` is applied. In the fingerprint because it changes the stored
        # text, not merely how the text is read: under "scoring" the generation continues
        # past the terminator. Two cells with the same everything-else but different modes
        # hold genuinely different data and must never be pooled or resumed across.
        "until_applied": "scoring",
        # 🚨 ADDED 2026-08-31 (R24.1). Its absence was the NINTH instance of this
        # project's recurring bug family, and the most indefensible one: this study's
        # own H8 result is that `max_num_seqs` changes generated text by up to **59%**,
        # so it is among the strongest determinants of what a generation IS — and this
        # function is documented as hashing exactly that. Without it, moving the frozen
        # batch size from 64 to 32 would have silently REUSED generations produced at
        # 64, and the re-run would have reported success having regenerated nothing.
        #
        # `cell_tag` in sm_determinism.py encodes the level explicitly (`at8`/`at32`),
        # which is why the determinism sweep never collided and the gap stayed hidden.
        # The generation stages (`pilot`, `template`, `reference`) key on
        # `run_id/cell_id/task`, where `cell_id` is `model|precision` — no batch size
        # anywhere. Those are precisely the stages whose output feeds the paper's deltas.
        #
        # ⚠️ AND THIS FUNCTION DOES NOT PROTECT THEM. It is called from `sm_determinism.py`
        # only. The comment above described a fix that landed in one of four callers, which
        # the code reviewer caught on 2026-08-31. The generation stages are guarded instead by
        # `records.assert_cell_reusable()`, called from `run_cell` below, which compares the
        # stored records' own fields and REFUSES rather than silently partitioning. Keep both:
        # this fingerprint is the determinism stage's cell identity, that guard is the
        # generation stages' resume check.
        "max_num_seqs": getattr(spec, "max_num_seqs", None),
    }, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


# Required environment, per EXECUTOR-TASKS.md §3.3. These are the settings that
# matter; everything else is cargo cult.
REQUIRED_ENV: dict[str, str] = {
    "VLLM_BATCH_INVARIANT": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    # 🚨 Required for TWO reasons, and the second is the important one.
    #
    # 1. Without it the job CRASHES. vLLM 0.26 routes sampling through
    #    `flashinfer.sampling.top_k_top_p_sampling_from_logits`, which JIT-compiles a
    #    CUDA module at runtime and therefore needs `nvcc`. Our image has the CUDA
    #    *runtime* (from torch's cu130 wheels) but not the toolkit, so it dies with
    #    "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist".
    #    Exactly the same class of trap as Triton needing `build-essential`: vLLM
    #    compiles things at RUNTIME, and a missing compiler is invisible until then.
    #
    # 2. It is the setting this study should want anyway. vLLM's own comment on the
    #    flag says disabling it "forces the PyTorch-native (Triton for bs>=8) path" --
    #    and `VLLM_BATCH_INVARIANT=1` works precisely by substituting batch-invariant
    #    **Triton** kernels. The flashinfer sampler is a separate CUDA path that batch
    #    invariance does NOT cover, so leaving it on would put an unguarded kernel
    #    inside the pipeline whose reproducibility we are trying to establish.
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
}


class PreflightError(RuntimeError):
    """A serving-configuration invariant is violated. Always fatal."""


@dataclass
class EngineSpec:
    """Frozen serving configuration. Identical across every arm.

    Backend choice alone moves benchmark scores up to 16.6 pp
    (arXiv:2605.19537) and explains ~39% of out-of-the-box variance
    (arXiv:2608.04714), so anything that floats here is a confound with the
    precision effect we are trying to measure.
    """

    model_repo: str
    model_revision: str
    quant_method: str | None = None
    bits: int = 16
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1      # REQUIRED 1 -- BI broken under TP>1 (#51290)
    enable_prefix_caching: bool = False  # breaks greedy determinism (#39389)
    max_model_len: int = 8192
    max_num_seqs: int = 8
    gpu_memory_utilization: float = 0.90
    seed: int = 0
    enforce_eager: bool = False        # CUDA graphs are BI-compatible since PR #27660
    add_bos_token: bool = True         # quantized models are BOS-sensitive
    disabled_kernels: str | None = None
    # 🚨 "vllm" tells the engine to IGNORE the model's own generation_config.json and use
    # vLLM's neutral defaults. The default is "auto", which loads that file -- and Qwen's
    # ships do_sample/temperature 0.7/top_p 0.8/top_k 20/repetition_penalty 1.05 while
    # Phi's ships none, so "auto" makes the decoding configuration MODEL-DEPENDENT in a
    # study whose whole point is holding the apparatus fixed across arms and models.
    #
    # Belt and braces with the explicit SamplingParams below: this stops the file being
    # read at all, and those pin every field regardless. Either alone would probably do;
    # both is the right amount of paranoia for a value the paper REPORTS.
    generation_config: str = "vllm"
    # ⚠️ MUST be passed to the engine, not merely declared. `configs/models.yaml` has
    # said `async_scheduling: false` all along, but nothing forwarded it -- and the
    # first real engine log read "Asynchronous scheduling is enabled", because vLLM's
    # own default won. Same failure as GPTQ's `actorder`: a value we declare, do not
    # enforce, and would then have reported in the paper as though it held.
    #
    # It selects AsyncScheduler over Scheduler, which changes WHEN requests are
    # batched -- and batch composition is the entire mechanism behind
    # batch-invariance failures, so it is not a setting to leave floating in a study
    # about reproducibility.
    async_scheduling: bool = False
    # 🚨 OFF for the study (planner ruling R14.1), and correct on the merits rather
    # than merely convenient. Batch invariance's premise is a FIXED, KNOWN kernel set;
    # an autotuner picks kernels by STARTUP BENCHMARK, i.e. by a noisy timing
    # measurement, so the configuration is not identical across runs BY CONSTRUCTION.
    # That violates §4.6's "reproducibility holds only within an identical
    # configuration" directly.
    #
    # It is also the 4th undocumented gap in VLLM_BATCH_INVARIANT's coverage, after
    # auto_awq's guard, auto_gptq's absence of one, and the sampler path — which is
    # what makes H8 a four-gap claim rather than a two-gap one.
    #
    # Left settable so the A7 audit can measure the ON-vs-OFF contrast on GPTQ-4bit,
    # the arm H8's claim is about. Bounding note: autotune variance can only INFLATE
    # measured instability, never manufacture false stability, so any arm that comes
    # out stable is genuinely stable regardless.
    enable_flashinfer_autotune: bool = False


def preflight(spec: EngineSpec, *, strict: bool = True) -> list[str]:
    """Assert every determinism invariant BEFORE loading a model.

    Returns the list of problems found. With ``strict=True`` (the default) it
    raises instead, because a run that proceeds with a broken invariant produces
    numbers that look publishable and are not.
    """
    problems: list[str] = []

    for name, expected in REQUIRED_ENV.items():
        actual = os.environ.get(name)
        if actual != expected:
            problems.append(
                f"env {name}={actual!r}, must be {expected!r} "
                f"(uncontrolled batch nondeterminism can manufacture or erase the effect, vLLM #9567)"
            )

    if spec.tensor_parallel_size != 1:
        problems.append(
            f"tensor_parallel_size={spec.tensor_parallel_size}, must be 1 -- "
            f"batch invariance is BROKEN under TP>1 (vLLM #51290). This is the "
            f"single most important setting."
        )

    if spec.enable_prefix_caching:
        problems.append(
            "enable_prefix_caching=True -- prefix caching changes greedy output "
            "(vLLM #39389) and is explicitly NOT covered by batch invariance. "
            "It would help throughput a lot here (few-shot CoT prompts share a "
            "long prefix) and is forbidden for exactly that reason."
        )

    if spec.enforce_eager:
        problems.append(
            "enforce_eager=True is pure throughput loss -- CUDA graphs became "
            "batch-invariance-compatible in vLLM PR #27660."
        )

    if spec.seed != 0:
        problems.append(f"seed={spec.seed}, must be 0")

    if spec.async_scheduling:
        problems.append(
            "async_scheduling=True -- it selects a different scheduler and so changes "
            "WHEN requests are batched, and batch composition is the mechanism behind "
            "batch-invariance failures. configs/models.yaml declares it false; the "
            "first engine log showed vLLM's default had silently overridden that."
        )

    if len(spec.model_revision) < 7:
        problems.append(
            f"model_revision={spec.model_revision!r} is not a commit SHA -- "
            f"tags move, and reproducibility holds only within identical versions."
        )

    problems.extend(_check_triton_toolchain())

    if problems and strict:
        raise PreflightError(
            "preflight FAILED -- refusing to generate:\n  - " + "\n  - ".join(problems)
        )
    return problems


def assert_batch_invariance_engaged() -> dict[str, Any]:
    """Verify batch invariance was actually APPLIED, not merely requested.

    🚨 **Planner ruling R14.3 item 1: the single most important assertion in the
    study.** No substitution ⇒ H8 is dead and every precision delta is untrustworthy.
    `preflight()` only checks that `VLLM_BATCH_INVARIANT=1` is *set*; that is a
    request, and the whole `async_scheduling` lesson is that a request is not an
    outcome. This reads the engine's state instead.

    Must be called AFTER the engine is constructed, since vLLM applies the overrides
    during initialisation.

    **What is observable, and why these checks.** vLLM 0.26.0 exposes no
    `vllm_is_batch_invariant()` accessor and keeps no registry of the ops it patched —
    they are registered imperatively onto one `torch.library.Library("aten", "IMPL")`
    handle. But `enable_batch_invariant_mode()` also *monkey-patches* `torch.bmm`, and
    that is directly visible.

    ⚠️ **Our SM89 L40S takes the Triton path**, which is why `build-essential` is
    load-bearing. The branch is `is_device_capability_family(80)` — a FAMILY check
    covering 8.0/8.6/8.9 alike — so we get Triton overrides of `aten::mm`,
    `addmm`, `matmul` and `linear`, with vLLM's own comment that Ampere-family chips
    "cannot rely on cuBLASLt-only determinism". **SM90/SM100 take a different path
    entirely** (no matmul overrides; `CUBLAS_WORKSPACE_CONFIG=:16:8` instead), so this
    study's mechanism claim is architecture-scoped and the paper must say so.
    """
    observed: dict[str, Any] = {}
    problems: list[str] = []

    import torch  # noqa: PLC0415 - GPU-box-only dependency

    # 1. The monkey-patch. The most direct evidence the function ran at all.
    bmm_name = getattr(torch.bmm, "__name__", "?")
    observed["torch_bmm"] = bmm_name
    if bmm_name != "bmm_batch_invariant":
        problems.append(
            f"torch.bmm is {bmm_name!r}, not 'bmm_batch_invariant' -- "
            f"enable_batch_invariant_mode() did not run"
        )

    # 2. vLLM's own private flag. Private, but it is the authoritative state.
    try:
        from vllm.model_executor.layers import batch_invariant  # noqa: PLC0415

        mode = getattr(batch_invariant, "_batch_invariant_MODE", None)
        observed["_batch_invariant_MODE"] = mode
        if mode is not True:
            problems.append(f"vLLM's _batch_invariant_MODE is {mode!r}, expected True")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"could not read vLLM's batch-invariant state: {exc}")

    # 3. Reduced-precision reduction must be OFF -- enable_batch_invariant_mode sets
    # it, so a truthy value here means our settings were overwritten afterwards.
    for attr in ("allow_bf16_reduced_precision_reduction",
                 "allow_fp16_reduced_precision_reduction"):
        value = getattr(torch.backends.cuda.matmul, attr, None)
        observed[attr] = value
        if value:
            problems.append(f"torch.backends.cuda.matmul.{attr}={value!r}, expected off")

    # 4. Set by override_envs_for_invariance() before the overrides are applied.
    observed["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if not observed["CUBLAS_WORKSPACE_CONFIG"]:
        problems.append(
            "CUBLAS_WORKSPACE_CONFIG is unset -- override_envs_for_invariance() "
            "did not run, so batch invariance was never initialised"
        )

    if problems:
        raise PreflightError(
            "BATCH INVARIANCE DID NOT ENGAGE -- refusing to generate. Every delta in "
            "this study would be untrustworthy and H8 unanswerable:\n  - "
            + "\n  - ".join(problems)
            + f"\nobserved: {observed}"
        )
    return observed


def read_back_engine_state(llm, spec: EngineSpec) -> dict[str, Any]:
    """Compare what the engine REPORTS against what we asked for. R14.3.

    The rule the three declared-but-unenforced failures produced: *assert on the
    engine's reported state, not on the value we passed.* We knew what we passed
    every time; what we never did was read back what the engine did with it.
    """
    observed: dict[str, Any] = {}
    problems: list[str] = []

    # ⚠️ An introspection FAILURE is not a config disagreement. vLLM moves this
    # attribute between versions and I cannot exercise it locally (no GPU), so a
    # missing path is reported loudly and does not kill the job -- whereas an actual
    # mismatch below is fatal. Killing a queued multi-hour job because an attribute
    # moved would be the wrong trade; silently skipping a real mismatch would be worse.
    config = None
    for path in ("llm_engine.vllm_config", "llm_engine.engine_core.vllm_config"):
        probe = llm
        for attr in path.split("."):
            probe = getattr(probe, attr, None)
            if probe is None:
                break
        if probe is not None:
            config = probe
            break
    if config is None:
        print("  ⚠️ engine state UNREADABLE on this vLLM version -- the read-back "
              "assertions did not run. Values below are our REQUESTS, not the "
              "engine's report.")
        return {"read_back": "unavailable", "requested_max_num_seqs": spec.max_num_seqs}

    model_cfg = getattr(config, "model_config", None)
    cache_cfg = getattr(config, "cache_config", None)
    sched_cfg = getattr(config, "scheduler_config", None)

    # R14.3 item 5. If max_model_len outruns the KV cache, vLLM SILENTLY reduces
    # concurrency -- which corrupts the exact quantity A7 exists to freeze.
    if sched_cfg is not None:
        observed["max_num_seqs"] = getattr(sched_cfg, "max_num_seqs", None)
        if observed["max_num_seqs"] != spec.max_num_seqs:
            problems.append(
                f"engine reports max_num_seqs={observed['max_num_seqs']}, we asked for "
                f"{spec.max_num_seqs}. A7 freezes this value; a silent reduction makes "
                f"every arm incomparable."
            )
        observed["async_scheduling"] = getattr(sched_cfg, "async_scheduling", None)
        if observed["async_scheduling"]:
            problems.append("engine reports async_scheduling enabled")

    if cache_cfg is not None:
        observed["enable_prefix_caching"] = getattr(cache_cfg, "enable_prefix_caching", None)
        if observed["enable_prefix_caching"]:
            problems.append("engine reports prefix caching ENABLED (vLLM #39389)")
        observed["num_gpu_blocks"] = getattr(cache_cfg, "num_gpu_blocks", None)

    # 🚨 RECORDED AS A FIELD, NOT AS A HYPOTHESIS (planner ruling R22). The
    # cross-invocation nondeterminism of `notes/cross-invocation.md` is undiagnosed; three
    # explanations have been refuted, and `num_gpu_blocks` above was one of them (it is
    # byte-identical across invocations). `cudagraph_mode=FULL_AND_PIECEWISE` captures
    # graphs at a startup-chosen batch-size set, so if THAT set differs between invocations
    # then dispatch differs. Nobody is betting on it -- the point is that a future reader
    # or reviewer can check, instead of having to re-derive the question. It belongs in the
    # record anyway under the standing assert-what-you-declare rule.
    compilation_cfg = getattr(config, "compilation_config", None)
    if compilation_cfg is not None:
        for field in ("cudagraph_mode", "cudagraph_capture_sizes",
                      "max_cudagraph_capture_size", "compile_ranges_endpoints"):
            value = getattr(compilation_cfg, field, None)
            # Enums and lists both need to survive JSON.
            observed[field] = (
                list(value) if isinstance(value, (list, tuple)) else
                (str(value) if value is not None else None)
            )

    if model_cfg is not None:
        # R14.3 item 4: the revision as LOADED. The HF cache can serve a different
        # commit than the pin.
        observed["revision"] = getattr(model_cfg, "revision", None)
        if observed["revision"] and observed["revision"] != spec.model_revision:
            problems.append(
                f"engine loaded revision {observed['revision']}, we pinned "
                f"{spec.model_revision}"
            )
        # torch reports "torch.bfloat16"; our spec says "bfloat16". Substring, so the
        # comparison is real rather than the no-op it was on first writing.
        observed["dtype"] = str(getattr(model_cfg, "dtype", ""))
        if spec.dtype not in observed["dtype"]:
            problems.append(
                f"engine dtype {observed['dtype']!r} does not contain {spec.dtype!r}"
            )
        observed["quantization"] = getattr(model_cfg, "quantization", None)

    if problems:
        raise PreflightError(
            "ENGINE STATE DISAGREES WITH THE FROZEN CONFIG -- refusing to generate:\n  - "
            + "\n  - ".join(problems) + f"\nobserved: {observed}"
        )
    return observed


def _check_triton_toolchain() -> list[str]:
    """Verify Triton can actually JIT-compile. THE landmine.

    ``VLLM_BATCH_INVARIANT=1`` works by substituting batch-invariant **Triton**
    kernels, which Triton JIT-compiles at runtime. With no C compiler present
    that fails -- and the sibling project hit exactly this after a build
    toolchain was purged post-``pip install`` (see ``notes/reuse.md`` §1).

    No compiler ⇒ no batch-invariant kernels ⇒ no determinism ⇒ H8 is dead and
    every precision delta in the paper is untrustworthy. Cheap to check, fatal
    to miss.
    """
    problems: list[str] = []

    compiler = next(
        (path for path in (shutil.which("gcc"), shutil.which("cc"), shutil.which("clang")) if path),
        None,
    )
    if compiler is None:
        problems.append(
            "no C compiler on PATH (gcc/cc/clang). Triton JIT-compiles the "
            "batch-invariant kernels at RUNTIME, so without one "
            "VLLM_BATCH_INVARIANT silently cannot work. Install build-essential "
            "and do NOT purge it after pip install."
        )
        return problems

    try:
        subprocess.run(  # noqa: S603 - fixed argv, no shell
            [compiler, "--version"], capture_output=True, check=True, timeout=30
        )
    except (subprocess.SubprocessError, OSError) as exc:
        problems.append(f"C compiler at {compiler} is not runnable: {exc}")

    return problems


def describe_hardware() -> dict[str, Any]:
    """GPU / CUDA / driver facts for the record.

    Reproducibility holds only within identical hardware and versions, so these
    go on every record rather than into a README that can drift.
    """
    facts: dict[str, Any] = {"gpu": "unknown", "cuda": None, "driver": None}
    smi = shutil.which("nvidia-smi")
    if not smi:
        return facts
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        first = result.stdout.strip().splitlines()[0]
        name, driver = (part.strip() for part in first.split(","))
        facts["gpu"], facts["driver"] = name, driver
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        pass

    try:
        import torch  # noqa: PLC0415 - GPU-box-only dependency

        facts["cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001 - torch absent locally, by design
        pass
    return facts


class VLLMRunner:
    """One vLLM engine per (model, precision). Loaded once, reused per cell."""

    def __init__(self, spec: EngineSpec) -> None:
        preflight(spec)
        self.spec = spec
        self._llm = None
        self.hardware = describe_hardware()
        # 🚨 One id per ENGINE LOAD (R21.3), stamped on every record this runner emits.
        # Minted here rather than per cell because the boundary that matters is the engine:
        # `pilot-v1` vs `pilot-v2` diverged on 41-45% of Llama items across separate engine
        # loads of a byte-identical configuration, bf16 included.
        #
        # Carries the pid and a UTC timestamp rather than a bare uuid so a record can be
        # traced back to a specific process in a specific job's log without a lookup table
        # -- the run spans many loads and the provenance should be readable, not inferred.
        self.invocation_id = (
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
            f"-pid{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        # Populated at engine init by the read-back assertions.
        self.batch_invariance: dict[str, Any] = {}
        self.engine_state: dict[str, Any] = {}

    @property
    def llm(self):
        if self._llm is None:
            from vllm import LLM  # noqa: PLC0415 - GPU-box-only dependency

            kwargs: dict[str, Any] = {
                "model": self.spec.model_repo,
                "revision": self.spec.model_revision,
                "dtype": self.spec.dtype,
                "tensor_parallel_size": self.spec.tensor_parallel_size,
                "enable_prefix_caching": self.spec.enable_prefix_caching,
                "max_model_len": self.spec.max_model_len,
                "max_num_seqs": self.spec.max_num_seqs,
                "gpu_memory_utilization": self.spec.gpu_memory_utilization,
                "seed": self.spec.seed,
                "enforce_eager": self.spec.enforce_eager,
                # Declared in configs/models.yaml and now actually ENFORCED.
                "async_scheduling": self.spec.async_scheduling,
                # Ignore the MODEL's generation_config.json. See EngineSpec: "auto" would
                # make decoding model-dependent (Qwen ships repetition_penalty 1.05, Phi
                # ships nothing), in a study that reports greedy with no penalties.
                "generation_config": self.spec.generation_config,
                # Passed as the NESTED config, not the top-level shortcut: the
                # shortcut goes through merge logic in create_engine_config that
                # raises if both are set, and a False there risks being read as
                # "unset". A dict is coerced by EngineArgs.__post_init__ directly.
                "kernel_config": {
                    "enable_flashinfer_autotune": self.spec.enable_flashinfer_autotune,
                },
            }
            if self.spec.quant_method == "bitsandbytes":
                # NF4 is quantized IN-FLIGHT by vLLM rather than loaded from a
                # pre-quantized checkpoint. That is a backend difference layered
                # on the bit-width difference -- a confound to STATE, not hide.
                kwargs["quantization"] = "bitsandbytes"
            self._llm = LLM(**kwargs)
            # R14.3 item 1, and it must be HERE: vLLM applies the batch-invariant
            # overrides during engine init, so this is the first moment the engine's
            # actual state can be read rather than our request for it.
            self.batch_invariance = assert_batch_invariance_engaged()
            self.engine_state = read_back_engine_state(self._llm, self.spec)
        return self._llm

    def generate(
        self,
        prompts: list[str],
        max_gen_toks: int,
        until: list[str],
        token_ids: list[list[int]] | None = None,
        *,
        until_at_generation: bool,
    ) -> list[dict[str, Any]]:
        """Greedy generation. Returns one dict per prompt, input order preserved.

        vLLM's continuous batching schedules the whole list, so short sequences
        do not wait on the batch's longest -- passing the full batch is strictly
        better than hand-rolled padded batches.

        🚨 **PASS ``token_ids`` WHENEVER YOU HAVE THEM.** Handing vLLM text makes it
        re-tokenize, and it does so on the *completion* path, whose
        ``default_cmpl_tok_params`` sets **``add_special_tokens=True``** (vLLM's
        ``default_chat_tok_params`` uses ``False``, with the comment that special
        tokens are "suppressed for chat, where the template already supplies them").
        Our prompts ARE chat-templated, so text input means BOS gets added a second
        time on any model whose template already embeds one -- Llama-3.1 emits
        ``<|begin_of_text|>``.

        That is precisely the double-BOS failure ``tasks/materialize.py`` exists to
        prevent: it chooses ``add_special_tokens`` per model, records
        ``prompt_token_ids``, and asserts a single leading BOS. Passing text throws
        all of that away and feeds the model something other than the frozen G2
        artifact -- silently, and differently per model. Red Hat's finding that
        quantized models are BOS-sensitive is what makes it consequential rather
        than cosmetic.

        ⚠️ vLLM documents that ``generate()`` preserves input order, but the
        sibling project flagged that as *never directly observed*. We therefore
        assert the count and let the caller key by ``item_id`` rather than trust
        position alone.
        """
        from vllm import SamplingParams, TokensPrompt  # noqa: PLC0415 - box-only

        # 🚨 EVERY SAMPLING FIELD IS SET EXPLICITLY, including the ones that are
        # inoperative under greedy decoding. Reason: vLLM's `generation_config` defaults
        # to "auto", which loads the MODEL's own `generation_config.json` and uses it to
        # fill fields the request leaves unset -- and those files differ per model:
        #
        #   Qwen2.5-3B-Instruct : do_sample true, temperature 0.7, top_p 0.8, top_k 20,
        #                         repetition_penalty 1.05
        #   Phi-3.5-mini        : no sampling fields at all
        #
        # `temperature=0.0` makes top_p/top_k inoperative (argmax), but
        # **repetition_penalty modifies the logits BEFORE the argmax**, so a leaked 1.05
        # would silently change greedy output on Qwen and not on Phi. Pape et al.
        # (arXiv:2605.19537) traced one of their two largest cross-backend outliers to
        # exactly this -- an unnoticed repetition penalty worth 11.67 points.
        #
        # We do not rely on reasoning about which vLLM version merges what. Every field is
        # pinned and then asserted, which is this project's standing rule after
        # `desc_act`, `async_scheduling`, `add_bos_token`, the pilot's `until`, job 1b's
        # destination path and the determinism resume key.
        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,                 # -1 = disabled. Qwen's config would supply 20.
            repetition_penalty=1.0,   # 1.0 = none. Qwen's config would supply 1.05.
            presence_penalty=0.0,
            frequency_penalty=0.0,
            max_tokens=max_gen_toks,
            # 🚨 `stop` is DELIBERATELY NOT SET (planner ruling R34). `until` is applied at
            # SCORING time by `scoring.extract.apply_until` instead.
            #
            # Under greedy decoding the two are equivalent for the kept prefix, so published
            # semantics are preserved exactly and R19.1's constraint (do not replace the
            # harness's terminators with our own invention) is respected. What changes is that
            # the discarded continuation survives, and that is a correctness fix rather than a
            # convenience:
            #
            # Llama-3.1-8B emits `Q:` as its FIRST TWO TOKENS on some chat-templated BBH items,
            # continuing the few-shot transcript instead of answering. As a generation-time stop
            # that produced an EMPTY record at `n_gen_tokens=2`, indistinguishable from "the
            # model had nothing to say" -- and at an ARM-DIFFERENTIAL rate, 17.5% on bf16 against
            # 3.3% on gptq-4bit. A 14 pp gap that scores as competence and flatters this paper's
            # own hypothesis is the failure mode SS4.0 exists to prevent.
            #
            # ⚠️ CONSEQUENCE FOR THE RECORD, and it is why `until_applied` is stored: with no
            # generation-time stop, `finish_reason` and `n_gen_tokens` now describe the FULL
            # generation. `stop_reason` will be None or an EOS token, never an `until` string.
            # So `TRUNC_STOPSEQ` can no longer be read off `stop_reason` and must be DERIVED
            # from the text. `compliance.classify_termination` does that, keyed on
            # `until_applied`, so records from before this change keep their old semantics.
            #
            # 🚨 `until_at_generation` IS REQUIRED AND KEYWORD-ONLY, with NO DEFAULT, and that is
            # the point. My first version of this change hardcoded `stop=None` and left `until` as
            # a dead positional parameter -- which the CODE REVIEWER caught, because
            # `sm_determinism.py` calls this method DIRECTLY and depended on the old behaviour.
            # Its H8 probe would have started storing full-length generations, shifting the
            # measured text-divergence rates for a reason unrelated to batch invariance, in the
            # table the H8 section leads with. A parameter accepted and then ignored is the
            # charter's named pattern and is how `add_bos_token` and `async_scheduling` happened.
            #
            # No default, so a new caller cannot omit the decision and inherit someone else's.
            stop=until if until_at_generation else None,
            seed=self.spec.seed,
            n=1,
        )
        # R14.3 item 6: assert the values as they reached SamplingParams, not as we
        # believe we passed them. Same failure shape as `async_scheduling`.
        # ⚠️ `top_k` has TWO sentinel spellings for "disabled" and vLLM normalises between
        # them: we pass -1 ("consider all tokens") and 0.26.0 reports back 0. Both mean the
        # same thing. The first version of this assertion compared against -1 alone and
        # failed a job on a representation difference rather than a semantic one -- so the
        # accepted set is explicit, and the assertion still catches a REAL value like 20
        # (which is what Qwen's generation_config.json would supply).
        # Single source of truth: the same dict that feeds `decoding_fingerprint`, so the
        # asserted values and the cache key can never disagree.
        wrong = {
            field: getattr(params, field)
            for field, allowed in NEUTRAL_SAMPLING.items()
            if getattr(params, field, allowed[0]) not in allowed
        }
        if wrong or params.seed != self.spec.seed:
            raise PreflightError(
                f"SamplingParams did not take our values: {wrong}, "
                f"seed={params.seed} (wanted {self.spec.seed}). Greedy decoding with no "
                f"penalties is a REPORTED property of this study, not a default."
            )

        if token_ids is not None:
            if len(token_ids) != len(prompts):
                raise RuntimeError(
                    f"{len(token_ids)} token id lists for {len(prompts)} prompts"
                )
            request = [TokensPrompt(prompt_token_ids=ids) for ids in token_ids]
        else:
            request = prompts

        outputs = self.llm.generate(request, params, use_tqdm=False)

        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts -- "
                f"positional correspondence is broken, refusing to attribute results"
            )

        # R14.3 item 7: single-BOS on EVERY generation, from what the engine actually
        # received -- not once per model at G2. Cheap, and a known-live failure on
        # Llama. Checked here because this is the only place that sees the engine's
        # own view of the prompt.
        if token_ids is not None:
            for output, expected in zip(outputs, token_ids, strict=True):
                got = list(output.prompt_token_ids or [])
                if got != list(expected):
                    raise RuntimeError(
                        "the engine's prompt_token_ids differ from the frozen "
                        f"artifact ({len(got)} vs {len(expected)} tokens; first ids "
                        f"{got[:4]} vs {expected[:4]}). Something re-tokenized."
                    )

        results: list[dict[str, Any]] = []
        for output in outputs:
            completion = output.outputs[0]
            results.append(
                {
                    "text": completion.text,
                    # The two unreconstructable fields. Taken straight from the
                    # engine; never inferred from the text.
                    "finish_reason": completion.finish_reason,
                    "n_gen_tokens": len(completion.token_ids),
                    "stop_reason": _normalise_stop_reason(completion.stop_reason),
                    "n_prompt_tokens": len(output.prompt_token_ids or []),
                }
            )
        return results


def _normalise_stop_reason(stop_reason: Any) -> str | None:
    """vLLM returns a token id, a string, or None. Normalise to str | None."""
    if stop_reason is None:
        return None
    return str(stop_reason)


def run_cell(
    runner: VLLMRunner,
    *,
    run_id: str,
    cell_id: str,
    task: str,
    subtask: str | None,
    items: list[dict[str, Any]],
    prompt_variant: str,
    budget_level: str,
    max_gen_toks: int,
    until: list[str],
    quant_config_hash: str | None,
    # Which level of the template factor this cell belongs to. Recorded per generation so a
    # re-analyst never has to know it out of band -- see GenerationRecord.template_level.
    template_level: str | None = None,
    output_root: Path,
    repeat_idx: int = 0,
    batch_size: int = 64,
    on_progress=None,
) -> dict[str, int]:
    """Generate one cell, resumably. Returns counts.

    ``items`` is a list of ``{"item_id", "prompt", "gold"}``.

    Flushes after every batch and calls ``on_progress`` so an interruption costs
    at most one batch rather than the whole cell -- the pattern inherited from
    the sibling project (``notes/reuse.md`` §3), and the single largest cost
    saver available to us.
    """
    path = cell_path(output_root, run_id, cell_id, task, subtask, budget_level, prompt_variant)

    # 🚨 REFUSE to resume a cell produced under a different generation configuration.
    #
    # This is the protection `decoding_fingerprint()` was supposed to give the generation
    # stages and did not: it was wired into `sm_determinism.py` only, while its own comment
    # named pilot/template/reference as the vulnerable ones. Those are the stages whose output
    # feeds the paper's deltas, and they were the three left unguarded.
    #
    # Checked against the STORED RECORDS rather than a sidecar, because the record already
    # carries every field that matters -- so this works retroactively on `pilot-v2`'s existing
    # data without regenerating it, and without changing any path or join key.
    assert_cell_reusable(path, {
        "max_num_seqs": runner.spec.max_num_seqs,
        "disabled_kernels": runner.spec.disabled_kernels,
        "seed": runner.spec.seed,
        "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT") == "1",
        "engine": f"vllm=={_vllm_version()}",
        "max_gen_toks": max_gen_toks,
        "decoding": "greedy",
        "model_revision": runner.spec.model_revision,
        "quant_config_hash": quant_config_hash,
        # See RESUME_CRITICAL_FIELDS: `until` is read from tasks.yaml at run time, so a
        # mid-run edit must refuse to resume rather than mix two terminator sets in one file.
        "until": list(until) if until is not None else None,
        # R34: WHERE `until` was applied. The same terminator list means different things in the
        # two regimes, so a cell generated under one must not be resumed under the other -- the
        # stored text would mean different things in the two halves of one file.
        "until_applied": "scoring",
    })

    already = build_resume_index(path)

    pending = [
        item
        for item in items
        if (
            run_id, cell_id, task, subtask, item["item_id"],
            budget_level, prompt_variant, 0, repeat_idx,
        ) not in already
    ]

    counts = {"total": len(items), "skipped": len(items) - len(pending), "written": 0}
    if not pending:
        return counts  # idempotent: zero new work, exit clean

    hardware = runner.hardware
    environment = env_hash({"gpu": hardware["gpu"], "cuda": hardware["cuda"]})

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        started = time.monotonic()
        # Token ids when the caller has them, which is the correct path -- see
        # VLLMRunner.generate. `items` carry them straight from the frozen G2 artifact.
        ids = [item.get("prompt_token_ids") for item in batch]
        outputs = runner.generate(
            [item["prompt"] for item in batch], max_gen_toks, until,
            # R34: the SCORING regime. `until` is stamped on the record and applied by
            # `scoring.extract.scored_text_for`, so the continuation survives.
            until_at_generation=False,
            token_ids=[i for i in ids if i is not None] if all(ids) else None,
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        per_item_ms = elapsed_ms // max(len(batch), 1)

        records = [
            GenerationRecord(
                run_id=run_id,
                cell_id=cell_id,
                model_repo=runner.spec.model_repo,
                model_revision=runner.spec.model_revision,
                quant_method=runner.spec.quant_method,
                bits=runner.spec.bits,
                quant_config_hash=quant_config_hash,
                task=task,
                subtask=subtask,
                item_id=item["item_id"],
                gold=item["gold"],
                prompt_variant=prompt_variant,
                scoring_mode="generate",
                budget_level=budget_level,
                max_gen_toks=max_gen_toks,
                decoding="greedy",
                sample_idx=0,
                seed=runner.spec.seed,
                repeat_idx=repeat_idx,
                prompt_text=item["prompt"],
                prompt_sha256=prompt_sha256(item["prompt"]),
                generation_text=output["text"],
                n_prompt_tokens=output["n_prompt_tokens"],
                n_gen_tokens=output["n_gen_tokens"],
                finish_reason=output["finish_reason"],
                stop_reason=output["stop_reason"],
                option_logprobs=None,
                engine=f"vllm=={_vllm_version()}",
                batch_invariant=os.environ.get("VLLM_BATCH_INVARIANT") == "1",
                max_num_seqs=runner.spec.max_num_seqs,
                disabled_kernels=runner.spec.disabled_kernels,
                gpu=hardware["gpu"],
                cuda=hardware["cuda"],
                driver=hardware["driver"],
                env_hash=environment,
                wall_ms=per_item_ms,
                invocation_id=runner.invocation_id,
                # The list actually passed to the engine for THIS cell, so no scorer has to
                # infer it from a task-keyed table. See GenerationRecord.until.
                until=list(until) if until is not None else None,
                # R34: `until` is applied by the SCORER, not by the engine. Recorded so a
                # reader never has to infer it, and so pre-R34 records keep their own
                # semantics rather than being reinterpreted under the new ones.
                until_applied="scoring",
                template_level=template_level,
            )
            for item, output in zip(batch, outputs, strict=True)
        ]

        counts["written"] += append_records(path, records)
        if on_progress is not None:
            on_progress(path, counts)

    return counts


def _vllm_version() -> str:
    try:
        import vllm  # noqa: PLC0415 - GPU-box-only dependency

        return vllm.__version__
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qa generate",
        description="Generate one cell. Resumable and idempotent.",
    )
    parser.add_argument("--cell", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--subtask", default=None)
    parser.add_argument("--budget", default="1x", choices=["1x", "2x", "4x"])
    parser.add_argument("--variant", default="fewshot_cot", choices=["fewshot_cot", "zeroshot"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repeat-idx", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/generations"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run the determinism assertions and exit. Costs nothing, starts nothing.",
    )
    # Decision D10. Deliberately awkward: this is a human gate, and it should
    # not be possible to satisfy it by habit.
    parser.add_argument(
        "--i-have-confirmation",
        action="store_true",
        help="The user has explicitly confirmed THIS run (decision D10). Required.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.preflight_only and not args.i_have_confirmation:
        sys.stderr.write(
            "REFUSING TO RUN.\n\n"
            "Decision D10 (notes/decisions.md): no generation, quantization, or\n"
            "billable compute starts without explicit user confirmation.\n\n"
            "Use --preflight-only to validate configuration at zero cost, or pass\n"
            "--i-have-confirmation once the user has confirmed this specific run.\n"
        )
        return 2

    sys.stderr.write("generate.py is wired but the run path needs configs/ loading;\n")
    sys.stderr.write("see EXECUTOR-TASKS.md A4. Use --preflight-only today.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
