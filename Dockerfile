# SageMaker training image for QuantArtifact. SMInterface `framework: generic`.
#
# Derived from MedLLMSafety/Dockerfile.vllm, whose comments record several
# failures paid for with real job time. Everything below that looks like
# over-specification is there because a vaguer version already failed once.
#
# THREE HARD-WON RULES, do not "simplify" any of them:
#
# 1. build-essential MUST STAY INSTALLED, never purged after pip.
#    vLLM JIT-compiles Triton kernels AT RUNTIME for the specific model and
#    shapes it is serving. MedLLMSafety/Dockerfile purges build-essential and its
#    vLLM sibling explicitly does not, after a real job failed with
#    "RuntimeError: Failed to find C compiler" having cleared every earlier
#    hurdle. For us it is worse than a crash: VLLM_BATCH_INVARIANT works by
#    SUBSTITUTING batch-invariant Triton kernels, so no compiler means no
#    determinism, which kills H8 and makes every precision delta untrustworthy.
#    gcc alone is NOT enough -- it runs but has no libc headers ("stdlib.h: No
#    such file or directory"); build-essential pulls in libc6-dev.
#
# 2. torch MUST come from the matching CUDA wheel index, not plain PyPI.
#    PyPI's default wheel for torch/torchvision/torchaudio is not the build we
#    need, and pip will happily install a mismatched one.
#
# 3. The CUDA runtime vLLM's OWN precompiled .so links is NOT controlled by the
#    Python torch pin. MedLLMSafety verified with
#    `objdump -p vllm/_C.abi3.so | grep NEEDED` that vllm==0.18.0 links
#    libcudart.so.12 while vllm==0.26.0 links libcudart.so.13 -- discovered via
#    "ImportError: libcudart.so.13: cannot open shared object file" from vllm's
#    _C_stable_libtorch extension, AFTER torch was already pinned correctly.
#
# ⚠️ OPEN RISK, resolved by the `verify` stage before anything expensive runs.
# vllm==0.26.0 needs a CUDA 13 runtime, so the HOST DRIVER must support it.
# MedLLMSafety hit "The NVIDIA driver on your system is too old (found version
# 12080)" on ml.g5.2xlarge -- i.e. that host capped at CUDA 12.8 and could not
# run a CUDA-13 build at all. ml.g6e.2xlarge is a newer instance type and the
# current SageMaker PyTorch DLC ships CUDA 13.0, so it is likely fine, but
# "likely" is not good enough to build a study on.
#
#   => `python scripts/submit_job.py --stage verify` prints the driver and CUDA
#      version. If it reports < 13, we CANNOT use vllm 0.26.0 here, and the
#      choice of engine version is a PLANNER decision (it is frozen and reported
#      in the paper, and batch-invariance behaviour is version-specific).
#      Fallback would be vllm==0.18.0 + torch==2.10.0 from the cu128 index.
#
# Build and push:  bash scripts/build_image.sh

FROM python:3.11-slim

# Keep the toolchain. See rule 1.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        binutils \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# torch first, from the CUDA-13 index, because vllm 0.26.0 pins these exactly
# (verified against its own requires_dist: torch==2.11.0, torchvision==0.26.0,
# torchaudio==2.11.0) and its compiled extensions expect the CUDA 13 runtime.
RUN pip install --no-cache-dir \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu130

# Serving + scoring. Versions pinned per EXECUTOR-TASKS.md §3.2.
#   `sagemaker-training` provides the `train` entrypoint SageMaker invokes.
#   math-verify needs the antlr extra, and its parse()/verify() use
#   signal.alarm() so they raise in worker threads -- handled in extract.py.
RUN pip install --no-cache-dir \
        sagemaker-training \
        vllm==0.26.0 \
        bitsandbytes==0.50.0 \
        "math-verify[antlr4_13_2]==0.9.0" \
        lm-eval==0.4.12 \
        boto3>=1.34 \
        pyyaml>=6.0 \
        numpy>=1.26 \
        pandas>=2.2 \
        scipy>=1.13 \
        statsmodels>=0.14

# NEVER install autoawq (downgrades transformers to 4.47.1 and wrecks the env) or
# auto-gptq (GitHub-archived; its own README redirects to llmcompressor).
# quantize.py asserts their absence at runtime.
#
# llmcompressor is deliberately NOT in this image: it pins
# compressed-tensors==0.17.1 exactly and transformers<=5.10.1, which conflicts
# with what vllm 0.26.0 wants. That is the two-venv split of §3.2, expressed here
# as a separate image -- see Dockerfile.quant.

# Fail the BUILD if the toolchain got dropped, rather than discovering it in a
# job that has already queued for hours.
RUN gcc --version >/dev/null 2>&1 || (echo "FATAL: no C compiler in final image" && exit 1)
RUN python -c "import vllm, lm_eval, math_verify; print('imports OK', vllm.__version__)"

ENTRYPOINT ["train"]
