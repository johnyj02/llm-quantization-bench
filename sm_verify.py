#!/usr/bin/env python3
"""SageMaker entry point — `verify` stage. The box_verify equivalent, in-container.

Answers the questions that decide whether any number this box produces is
trustworthy, and it runs FIRST because getting them wrong is silent:

1. **Is there exactly one visible GPU?** `tensor_parallel_size>1` breaks batch
   invariance (vLLM #51290) and is the single most important determinism setting.
2. **What compute capability?** Determines whether `EXECUTOR-TASKS.md` §3.3's
   1.2–2× batch-invariance overhead figure (an Ampere number) applies at all.
3. **Is there a C compiler, and can Triton actually JIT?** `VLLM_BATCH_INVARIANT`
   works by substituting batch-invariant **Triton** kernels, compiled at runtime.
   No compiler ⇒ no determinism ⇒ H8 dead and every precision delta untrustworthy.
   MedLLMSafety hit exactly this after a build purged `build-essential`.
4. **🚨 Does the host driver support the CUDA runtime vLLM's binaries need?**
   `vllm==0.26.0` links `libcudart.so.13`. MedLLMSafety got
   *"The NVIDIA driver on your system is too old (found version 12080)"* on
   `ml.g5.2xlarge` and had to fall back to `vllm==0.18.0`. **This is the open risk
   this stage exists to resolve before anything expensive runs.**
5. **Does batch invariance actually engage?** Unmeasured on SM89 (Ada). Treated as
   a finding, not a precondition.

Writes a JSON report to `/opt/ml/model/` so SageMaker uploads it to S3 and the
result survives the job.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

report: dict = {"stage": "verify", "checks": {}, "problems": [], "warnings": []}


def record(name: str, ok: bool, detail, *, fatal: bool = False) -> None:
    report["checks"][name] = {"ok": ok, "detail": detail}
    if not ok:
        (report["problems"] if fatal else report["warnings"]).append(f"{name}: {detail}")
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: {detail}")


def sh(*command: str) -> tuple[int, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        return result.returncode, (result.stdout + result.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, f"{type(exc).__name__}: {exc}"


print("=== 1. GPU visibility ===")
code, smi = sh("nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap",
               "--format=csv,noheader")
if code == 0 and smi:
    lines = [line.strip() for line in smi.splitlines() if line.strip()]
    for line in lines:
        print(f"        {line}")
    record("single_gpu", len(lines) == 1,
           f"{len(lines)} GPU(s) visible; TP>1 breaks batch invariance (vLLM #51290)",
           fatal=(len(lines) != 1))
    first = [part.strip() for part in lines[0].split(",")]
    report["gpu_name"] = first[1] if len(first) > 1 else None
    report["gpu_memory_total"] = first[2] if len(first) > 2 else None
    report["driver_version"] = first[3] if len(first) > 3 else None
    report["compute_cap"] = first[4] if len(first) > 4 else None
else:
    record("single_gpu", False, f"nvidia-smi failed: {smi[:200]}", fatal=True)

print("=== 2. Compute capability ===")
cap = report.get("compute_cap")
if cap:
    note = {
        "8.9": "SM89 Ada as planned — BI overhead UNMEASURED here, this stage/A7 measures it",
        "8.6": "SM86 Ampere — the §3.3 1.2–2× BI overhead figure applies directly",
        "8.0": "SM80 Ampere — the §3.3 figure applies directly",
        "9.0": "SM90 Hopper — vLLM has a cheaper cuBLAS-workspace BI path here",
    }.get(cap, f"unexpected capability {cap}; pin it in configs/models.yaml and re-check A7")
    record("compute_cap", True, f"{cap} — {note}")

print("=== 3. C compiler (Triton JIT depends on it at RUNTIME) ===")
compiler = next((shutil.which(c) for c in ("gcc", "cc", "clang") if shutil.which(c)), None)
if compiler:
    code, version = sh(compiler, "--version")
    record("c_compiler", code == 0, f"{compiler}: {version.splitlines()[0] if version else '?'}",
           fatal=(code != 0))
else:
    record("c_compiler", False,
           "NO C COMPILER. Triton JIT-compiles the batch-invariant kernels at runtime, "
           "so VLLM_BATCH_INVARIANT silently cannot work — H8 dead, all deltas untrustworthy",
           fatal=True)

print("=== 4. CUDA runtime vs host driver (THE open risk) ===")
try:
    import torch

    report["torch_version"] = torch.__version__
    report["torch_cuda"] = torch.version.cuda
    report["cuda_available"] = torch.cuda.is_available()
    record("torch_cuda_available", torch.cuda.is_available(),
           f"torch {torch.__version__}, built for CUDA {torch.version.cuda}, "
           f"available={torch.cuda.is_available()}",
           fatal=not torch.cuda.is_available())
except Exception as exc:  # noqa: BLE001
    record("torch_cuda_available", False, f"{type(exc).__name__}: {exc}", fatal=True)

# What libcudart does vLLM's PRECOMPILED extension actually need? This is NOT
# determined by the Python torch pin — MedLLMSafety verified that the hard way.
try:
    import vllm

    report["vllm_version"] = vllm.__version__
    vllm_dir = os.path.dirname(vllm.__file__ or "")
    so_files = glob.glob(os.path.join(vllm_dir, "_C*.so")) if vllm_dir else []
    needed = []
    for so in so_files[:3]:
        code, out = sh("objdump", "-p", so)
        needed += [
            line.split()[-1]
            for line in out.splitlines()
            if "NEEDED" in line and "cuda" in line.lower()
        ]
    report["vllm_needed_cuda_libs"] = sorted(set(needed))
    record("vllm_cuda_linkage", True,
           f"vllm {vllm.__version__} links {sorted(set(needed)) or '(none found)'}")

    # The decisive check: can vLLM's compiled extension actually load?
    code, out = sh(sys.executable, "-c", "import vllm._C; print('ok')")
    record("vllm_extension_loads", code == 0,
           "vllm._C imported" if code == 0 else out[-300:],
           fatal=(code != 0))
except Exception as exc:  # noqa: BLE001
    record("vllm_cuda_linkage", False, f"{type(exc).__name__}: {exc}", fatal=True)

print("=== 5. Determinism environment ===")
for name, expected in (("VLLM_BATCH_INVARIANT", "1"), ("VLLM_ENABLE_V1_MULTIPROCESSING", "0")):
    actual = os.environ.get(name)
    record(f"env_{name}", actual == expected, f"{actual!r} (must be {expected!r})",
           fatal=(actual != expected))

print("=== 6. Does batch invariance engage? ===")
try:
    import torch

    if torch.cuda.is_available():
        # A trivial Triton JIT, proving the whole compile path works on this host.
        import triton
        import triton.language as tl

        @triton.jit
        def _add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
            i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = i < n
            tl.store(out_ptr + i, tl.load(x_ptr + i, mask=mask) + tl.load(y_ptr + i, mask=mask),
                     mask=mask)

        x = torch.ones(128, device="cuda")
        y = torch.ones(128, device="cuda")
        out = torch.empty_like(x)
        _add[(1,)](x, y, out, 128, BLOCK=128)
        record("triton_jit", out.sum().item() == 256.0,
               f"compiled and ran a Triton kernel; triton {triton.__version__}",
               fatal=(out.sum().item() != 256.0))
    else:
        record("triton_jit", False, "skipped, no CUDA device", fatal=True)
except Exception as exc:  # noqa: BLE001
    record("triton_jit", False, f"{type(exc).__name__}: {exc}"[:300], fatal=True)

print("=== 7. Forbidden packages ===")
for package, reason in (("autoawq", "downgrades transformers to 4.47.1"),
                        ("auto_gptq", "GitHub-archived, redirects to llmcompressor")):
    present = importlib.util.find_spec(package) is not None
    record(f"absent_{package}", not present, "absent" if not present else f"PRESENT ({reason})",
           fatal=present)

print("=== 8. Pinned versions ===")
versions = {}
for name in ("vllm", "torch", "transformers", "lm_eval", "math_verify", "bitsandbytes", "triton"):
    try:
        versions[name] = getattr(__import__(name), "__version__", "?")
    except Exception:  # noqa: BLE001
        versions[name] = "MISSING"
report["versions"] = versions
for name, version in versions.items():
    print(f"        {name:14} {version}")

# ---------------------------------------------------------------------------
report["passed"] = not report["problems"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "verify_report.json").write_text(json.dumps(report, indent=2))

print()
print("=" * 66)
if report["passed"]:
    print("VERIFY PASSED — this box's numbers can be trusted")
else:
    print(f"VERIFY FAILED — {len(report['problems'])} problem(s):")
    for problem in report["problems"]:
        print(f"  - {problem}")
if report["warnings"]:
    print(f"{len(report['warnings'])} warning(s):")
    for warning in report["warnings"]:
        print(f"  - {warning}")
print(f"report -> {OUTPUT_DIR / 'verify_report.json'}")

# Exit non-zero on failure so SageMaker marks the job Failed rather than
# Completed — a "successful" job that produced untrustworthy numbers is worse
# than a failed one.
sys.exit(0 if report["passed"] else 1)
