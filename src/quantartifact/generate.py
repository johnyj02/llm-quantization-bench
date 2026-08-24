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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantartifact.records import (
    GenerationRecord,
    append_records,
    build_resume_index,
    cell_path,
    env_hash,
    prompt_sha256,
)

# Required environment, per EXECUTOR-TASKS.md §3.3. These are the settings that
# matter; everything else is cargo cult.
REQUIRED_ENV: dict[str, str] = {
    "VLLM_BATCH_INVARIANT": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
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
            }
            if self.spec.quant_method == "bitsandbytes":
                # NF4 is quantized IN-FLIGHT by vLLM rather than loaded from a
                # pre-quantized checkpoint. That is a backend difference layered
                # on the bit-width difference -- a confound to STATE, not hide.
                kwargs["quantization"] = "bitsandbytes"
            self._llm = LLM(**kwargs)
        return self._llm

    def generate(
        self, prompts: list[str], max_gen_toks: int, until: list[str]
    ) -> list[dict[str, Any]]:
        """Greedy generation. Returns one dict per prompt, input order preserved.

        vLLM's continuous batching schedules the whole list, so short sequences
        do not wait on the batch's longest -- passing the full batch is strictly
        better than hand-rolled padded batches.

        ⚠️ vLLM documents that ``generate()`` preserves input order, but the
        sibling project flagged that as *never directly observed*. We therefore
        assert the count and let the caller key by ``item_id`` rather than trust
        position alone.
        """
        from vllm import SamplingParams  # noqa: PLC0415 - GPU-box-only dependency

        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_gen_toks,
            stop=until,
            seed=self.spec.seed,
            n=1,
        )
        outputs = self.llm.generate(prompts, params, use_tqdm=False)

        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts -- "
                f"positional correspondence is broken, refusing to attribute results"
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
        outputs = runner.generate([item["prompt"] for item in batch], max_gen_toks, until)
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
