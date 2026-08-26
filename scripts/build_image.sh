#!/usr/bin/env bash
# Build the training image and push it to ECR, then record the URI.
#
#   bash scripts/build_image.sh                # plan only, builds nothing
#   bash scripts/build_image.sh --apply        # build + push (needs confirmation)
#
# Decision D10: this is billable-adjacent (ECR storage, data transfer, and it
# gates a job that costs real money), so it is plan-by-default like everything
# else.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${AWS_PROFILE:=qa-gpu}"
: "${AWS_ACCOUNT:=654654594225}"
# ECR repos already exist in us-east-1 in this account; none in us-west-2.
# The image must live in the SAME region as the training job.
: "${QA_REGION:=us-east-1}"
: "${ECR_REPO:=johnyj02-qa}"
: "${IMAGE_TAG:=v0.1-$(date -u +%Y%m%d)}"

REGISTRY="${AWS_ACCOUNT}.dkr.ecr.${QA_REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

aws_() { aws --profile "${AWS_PROFILE}" --region "${QA_REGION}" "$@"; }

cat <<EOF
======================================================================
 IMAGE BUILD PLAN
======================================================================
account/region : ${AWS_ACCOUNT} / ${QA_REGION}   (profile ${AWS_PROFILE})
ecr repo       : ${ECR_REPO}          <-- generic name, no project identifier
image uri      : ${IMAGE_URI}
dockerfile     : ${REPO_ROOT}/Dockerfile
platform       : linux/amd64          <-- REQUIRED; this Mac is arm64

contents       : torch 2.11.0 (cu130), vllm 0.26.0, lm-eval 0.4.12,
                 math-verify 0.9.0, bitsandbytes 0.50.0, sagemaker-training
                 build-essential RETAINED (Triton JIT needs a compiler at RUNTIME)
EOF

if ! docker info >/dev/null 2>&1; then
  echo
  echo "Docker is not running. Start Docker Desktop (or colima) first."
  [ "${APPLY}" = "1" ] && exit 1
fi

if [ "${APPLY}" = "0" ]; then
  cat <<EOF

======================================================================
 PLAN ONLY — NOTHING BUILT OR PUSHED
======================================================================
To proceed:  bash scripts/build_image.sh --apply

Note: the build is a large download (torch + vllm are multi-GB) and this Mac is
arm64 while SageMaker needs amd64, so it cross-builds under emulation and will be
slow. Expect tens of minutes.
EOF
  exit 0
fi

echo
echo "=== ensuring ECR repo exists ==="
if ! aws_ ecr describe-repositories --repository-names "${ECR_REPO}" >/dev/null 2>&1; then
  aws_ ecr create-repository --repository-name "${ECR_REPO}" \
    --image-scanning-configuration scanOnPush=true \
    --tags Key=Owner,Value=johnyj02 Key=Purpose,Value=testing-QA \
    --query 'repository.repositoryUri' --output text
  echo "  created ${ECR_REPO}"
else
  echo "  ${ECR_REPO} already exists"
fi

echo "=== docker login ==="
aws_ ecr get-login-password | docker login --username AWS --password-stdin "${REGISTRY}"

echo "=== build (linux/amd64) ==="
# --platform is mandatory: this machine is arm64, SageMaker hosts are amd64, and
# a silently-arm64 image fails at job start with an exec format error.
docker build --platform linux/amd64 -t "${IMAGE_URI}" -f "${REPO_ROOT}/Dockerfile" "${REPO_ROOT}"

echo "=== push ==="
docker push "${IMAGE_URI}"

echo
echo "pushed: ${IMAGE_URI}"
echo
echo "Recording it in configs/sm_job.yaml..."
python3 - "$IMAGE_URI" "$REPO_ROOT" <<'PY'
import pathlib, re, sys
uri = sys.argv[1]
p = pathlib.Path(sys.argv[2]) / "configs" / "sm_job.yaml"
s = p.read_text()
s = re.sub(r"^image_uri: .*$", f"image_uri: {uri}", s, flags=re.M)
p.write_text(s)
print(f"  configs/sm_job.yaml image_uri -> {uri}")
PY

cat <<EOF

NEXT — verify the host driver supports CUDA 13 before spending on anything else:
    python scripts/submit_job.py --stage verify --plan-only
    python scripts/submit_job.py --stage verify --i-have-confirmation

vllm 0.26.0 links libcudart.so.13. MedLLMSafety hit "driver too old (found
version 12080)" on ml.g5.2xlarge and had to fall back to vllm 0.18.0. If verify
reports CUDA < 13, the engine version is a PLANNER decision, not a quick swap.
EOF
