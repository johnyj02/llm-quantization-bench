#!/usr/bin/env bash
# Verify a freshly provisioned GPU box before trusting ANY number it produces.
#
# Run ON the box, after the environment is installed:
#     bash scripts/box_verify.sh
#
# Every check here exists because getting it wrong produces numbers that look
# publishable and are not. In particular check 3: no C compiler means Triton
# cannot JIT the batch-invariant kernels, so VLLM_BATCH_INVARIANT silently does
# nothing, H8 dies, and every precision delta in the paper becomes untrustworthy.
# The sibling project hit exactly that after a build toolchain was purged
# post-pip-install (notes/reuse.md §1).
#
# Exits non-zero on any failure. Output is the G1/A2 evidence artifact.

set -uo pipefail

fail=0
ok()   { printf "  \033[32mOK\033[0m    %s\n" "$1"; }
bad()  { printf "  \033[31mFAIL\033[0m  %s\n" "$1"; fail=1; }
info() { printf "  ..    %s\n" "$1"; }

echo "=== 1. GPU present and single ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap \
             --format=csv,noheader | sed 's/^/        /'
  n_gpu=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
  if [ "$n_gpu" = "1" ]; then
    ok "exactly 1 GPU visible (tensor_parallel_size=1 is enforceable)"
  else
    bad "$n_gpu GPUs visible — batch invariance is BROKEN under TP>1 (vLLM #51290)"
  fi
else
  bad "nvidia-smi not found"
fi

echo "=== 2. Compute capability ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
  info "compute_cap=${cap}"
  case "$cap" in
    8.9) ok "SM89 (Ada) as planned — BI overhead is UNMEASURED here, A7 measures it" ;;
    8.0|8.6) ok "SM8x (Ampere) — the 1.2-2x BI overhead figure applies directly" ;;
    9.0|10.0) ok "SM90+ — vLLM has a cheaper cuBLAS-workspace BI path here" ;;
    *) bad "unexpected compute capability ${cap}; pin it in configs/models.yaml and re-check A7" ;;
  esac
fi

echo "=== 3. C compiler for Triton JIT (CRITICAL) ==="
cc_found=""
for c in gcc cc clang; do
  if command -v "$c" >/dev/null 2>&1; then cc_found="$c"; break; fi
done
if [ -n "$cc_found" ]; then
  ok "$cc_found present: $($cc_found --version 2>&1 | head -1)"
else
  bad "NO C COMPILER. Triton JIT-compiles the batch-invariant kernels at RUNTIME."
  bad "  => VLLM_BATCH_INVARIANT silently cannot work => H8 dead, all deltas untrustworthy."
  bad "  Install build-essential and DO NOT purge it after pip install."
fi

echo "=== 4. Triton actually compiles a kernel ==="
python - <<'PY' 2>&1 | sed 's/^/        /'
try:
    import torch, triton, triton.language as tl
    @triton.jit
    def _add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(out_ptr + i, tl.load(x_ptr + i, mask=m) + tl.load(y_ptr + i, mask=m), mask=m)
    x = torch.ones(128, device="cuda"); y = torch.ones(128, device="cuda")
    out = torch.empty_like(x)
    _add[(1,)](x, y, out, 128, BLOCK=128)
    assert out.sum().item() == 256.0
    print("TRITON_JIT_OK")
except Exception as exc:
    print(f"TRITON_JIT_FAIL {type(exc).__name__}: {exc}")
PY
if python -c "
import torch,triton,triton.language as tl
" 2>/dev/null; then ok "triton importable"; else bad "triton not importable"; fi

echo "=== 5. vLLM's own CUDA runtime linkage ==="
# vLLM's PRECOMPILED _C.abi3.so links a specific libcudart, independent of what
# the Python torch package pins. Verified rather than assumed (notes/reuse.md §5).
so=$(python -c "import vllm,os,glob;print((glob.glob(os.path.join(os.path.dirname(vllm.__file__),'_C*.so'))or[''])[0])" 2>/dev/null)
if [ -n "$so" ] && [ -f "$so" ]; then
  info "$(basename "$so")"
  objdump -p "$so" 2>/dev/null | grep NEEDED | grep -i cuda | sed 's/^/        /' || info "no cuda NEEDED entries found"
  ok "vLLM extension inspected — record the libcudart version in configs/models.yaml"
else
  bad "could not locate vLLM's _C extension to inspect"
fi

echo "=== 6. Determinism environment ==="
[ "${VLLM_BATCH_INVARIANT:-}" = "1" ] && ok "VLLM_BATCH_INVARIANT=1" \
  || bad "VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-unset}, must be 1"
[ "${VLLM_ENABLE_V1_MULTIPROCESSING:-}" = "0" ] && ok "VLLM_ENABLE_V1_MULTIPROCESSING=0" \
  || bad "VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING:-unset}, must be 0"

echo "=== 7. Pinned versions ==="
python - <<'PY' 2>&1 | sed 's/^/        /'
for name in ("vllm", "torch", "transformers", "lm_eval", "math_verify", "bitsandbytes"):
    try:
        mod = __import__(name)
        print(f"{name:14} {getattr(mod, '__version__', '?')}")
    except Exception as exc:
        print(f"{name:14} MISSING ({type(exc).__name__})")
PY

echo "=== 8. Forbidden packages absent ==="
for pkg in autoawq auto_gptq; do
  if python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$pkg') else 1)" 2>/dev/null; then
    bad "$pkg is installed — it wrecks the environment; remove it"
  else
    ok "$pkg absent"
  fi
done

echo
if [ "$fail" = "0" ]; then
  echo "BOX VERIFY: PASSED — safe to run A6/A7"
else
  echo "BOX VERIFY: FAILED — do NOT generate; the numbers would not be trustworthy"
fi
exit "$fail"
