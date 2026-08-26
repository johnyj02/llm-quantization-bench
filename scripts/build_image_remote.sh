#!/usr/bin/env bash
# Build the training image ON THE DEV DESKTOP and push it to ECR.
#
#   bash scripts/build_image_remote.sh              # plan only, builds nothing
#   bash scripts/build_image_remote.sh --apply      # rsync, build, push
#
# WHY REMOTE. The dev desktop is x86_64 Linux, so it builds linux/amd64 NATIVELY
# -- which is what SageMaker needs. This Mac is arm64, so a local build either
# cross-compiles under emulation (very slow for multi-GB torch/vllm layers) or,
# if --platform is forgotten, silently produces an arm64 image that fails at job
# start with an exec format error. The dev desktop also has ~1.5TB free vs the
# Mac's ~24GB, and a much faster uplink for the pulls.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${DEV_HOST:=dev-dsk-johnyj02-2a-53853bae.us-west-2.amazon.com}"
: "${REMOTE_DIR:=~/qa-build}"
: "${AWS_PROFILE:=qa-gpu}"
: "${AWS_ACCOUNT:=654654594225}"
: "${QA_REGION:=us-east-1}"
: "${ECR_REPO:=johnyj02-qa}"
: "${DOCKERFILE:=Dockerfile}"
# Distinct tag per image so the two cannot overwrite each other in ECR.
: "${TAG_SUFFIX:=}"
[ "${DOCKERFILE}" = "Dockerfile.quant" ] && TAG_SUFFIX="-quant"
: "${IMAGE_TAG:=v0.1-$(date -u +%Y%m%d)${TAG_SUFFIX}}"

REGISTRY="${AWS_ACCOUNT}.dkr.ecr.${QA_REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

APPLY=0
for i in "$@"; do
  case "$i" in
    --apply) APPLY=1 ;;
    --dockerfile=*) DOCKERFILE="${i#*=}" ;;
  esac
done

cat <<EOF
======================================================================
 REMOTE IMAGE BUILD PLAN
======================================================================
build host  : ${DEV_HOST}
remote dir  : ${REMOTE_DIR}
dockerfile  : ${DOCKERFILE}
image uri   : ${IMAGE_URI}
platform    : linux/amd64  (NATIVE on the dev desktop -- no emulation)
EOF

# ---- Midway cert, the usual reason this fails ------------------------------
echo
echo "=== SSH cert ==="
cert="${HOME}/.ssh/id_ecdsa-cert.pub"
if [ -f "${cert}" ]; then
  ssh-keygen -L -f "${cert}" 2>/dev/null | grep -E "^ +Valid" | sed 's/^/  /'
  # Compare the cert's "to" timestamp against now.
  to=$(ssh-keygen -L -f "${cert}" 2>/dev/null | sed -nE 's/.*to ([0-9T:-]+).*/\1/p')
  if [ -n "${to}" ]; then
    to_epoch=$(date -j -f "%Y-%m-%dT%H:%M:%S" "${to}" +%s 2>/dev/null \
               || date -d "${to//T/ }" +%s 2>/dev/null || echo 0)
    if [ "${to_epoch}" != "0" ] && [ "$(date +%s)" -gt "${to_epoch}" ]; then
      echo "  ⚠️  CERT EXPIRED at ${to}"
      echo
      echo "  SSH will fail with 'Permission denied (publickey)'. That is the cert,"
      echo "  not the key. Refresh it (interactive, so YOU must run it):"
      echo "      mwinit -s"
      echo
      echo "  Do not use the shell's mwinit alias -- it pipes a hardcoded password."
      [ "${APPLY}" = "1" ] && exit 1
    else
      echo "  cert valid"
    fi
  fi
else
  echo "  no ~/.ssh/id_ecdsa-cert.pub found; run: mwinit -s"
  [ "${APPLY}" = "1" ] && exit 1
fi

# ---- remote readiness -----------------------------------------------------
if ssh -o BatchMode=yes -o ConnectTimeout=10 "${DEV_HOST}" true 2>/dev/null; then
  echo "=== remote host ==="
  ssh -o BatchMode=yes "${DEV_HOST}" '
    echo "  arch   : $(uname -m)"
    echo "  cpus   : $(nproc)"
    echo "  disk   : $(df -h ~ | tail -1 | awk "{print \$4}") free"
    printf "  docker : "; command -v docker >/dev/null 2>&1 \
      && { docker info >/dev/null 2>&1 && echo "present, daemon RUNNING" || echo "present, daemon NOT running"; } \
      || echo "MISSING"
    printf "  ada    : "; ( command -v ada >/dev/null 2>&1 || [ -x "$HOME/.toolbox/bin/ada" ] ) \
      && echo "present (~/.toolbox/bin, not on non-interactive PATH)" || echo MISSING
  ' 2>/dev/null
else
  echo "=== remote host: UNREACHABLE (see cert note above) ==="
fi

if [ "${APPLY}" = "0" ]; then
  cat <<EOF

======================================================================
 PLAN ONLY — NOTHING BUILT OR PUSHED
======================================================================
To proceed:  bash scripts/build_image_remote.sh --apply

It will:
  1. rsync this repo to ${DEV_HOST}:${REMOTE_DIR} (excluding .venv*, .git, artifacts)
  2. refresh Conduit credentials THERE (ada --provider conduit)
  3. create the ECR repo if absent, docker login, build, push
  4. write the resulting image_uri back into configs/sm_job.yaml
EOF
  exit 0
fi

echo
echo "=== 1. rsync repo -> ${DEV_HOST}:${REMOTE_DIR} ==="
ssh -o BatchMode=yes "${DEV_HOST}" "mkdir -p ${REMOTE_DIR}"
rsync -a --delete \
  --exclude '.venv' --exclude '.venv-harness' --exclude '.git' \
  --exclude 'artifacts/generations' --exclude 'artifacts/scores' --exclude 'artifacts/prompts' \
  --exclude '__pycache__' --exclude '.pytest_cache' \
  "${REPO_ROOT}/" "${DEV_HOST}:${REMOTE_DIR}/"

echo "=== 2-4. credentials, ECR, build, push (on the dev desktop) ==="
ssh -o BatchMode=yes "${DEV_HOST}" "
set -euo pipefail
cd ${REMOTE_DIR}

# ada lives in ~/.toolbox/bin, which is NOT on a non-interactive SSH PATH.
export PATH=\"\$HOME/.toolbox/bin:\$PATH\"

echo '--- Conduit credentials'
ada credentials update --provider conduit --account ${AWS_ACCOUNT} \
    --role IibsAdminAccess-DO-NOT-DELETE --profile ${AWS_PROFILE} --once
export AWS_PROFILE=${AWS_PROFILE}
aws sts get-caller-identity --query Account --output text

echo '--- ECR repo'
aws ecr describe-repositories --repository-names ${ECR_REPO} --region ${QA_REGION} >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name ${ECR_REPO} --region ${QA_REGION} \
    --image-scanning-configuration scanOnPush=true \
    --tags Key=Owner,Value=johnyj02 Key=Purpose,Value=testing-QA >/dev/null

echo '--- docker login'
aws ecr get-login-password --region ${QA_REGION} | \
  docker login --username AWS --password-stdin ${REGISTRY}

echo '--- build (native amd64)'
docker build --platform linux/amd64 -t ${IMAGE_URI} -f ${DOCKERFILE} .

echo '--- push'
docker push ${IMAGE_URI}
"

echo
echo "pushed: ${IMAGE_URI}"
python3 - "$IMAGE_URI" "$REPO_ROOT" <<'PY'
import pathlib, re, sys
uri = sys.argv[1]
p = pathlib.Path(sys.argv[2]) / "configs" / "sm_job.yaml"
s = p.read_text()
p.write_text(re.sub(r"^image_uri: .*$", f"image_uri: {uri}", s, flags=re.M))
print(f"  configs/sm_job.yaml image_uri -> {uri}")
PY

cat <<EOF

NEXT — verify the host driver supports CUDA 13 before spending on anything else:
    python scripts/submit_job.py --stage verify --plan-only
EOF
