#!/usr/bin/env bash
# Box configuration — EVERY value here is a PROPOSAL awaiting user review.
#
# Decision D11: "before you run any job you must ask me the human. Cause I want to
# make sure the job names are generic etc.. and possibly change the account"
#
# So:
#   1. Names follow the scheme the user specified on 2026-08-26 (see below). No
#      paper, project, venue, model or benchmark identifier appears anywhere.
#      This supersedes the earlier `Project=QuantArtifact` / `quantartifact-*`
#      proposal, which is withdrawn.
#   2. Account, region and profile are CONFIG, never hard-coded elsewhere -- and
#      the account DID move once already (670060057797 -> 654654594225).
#   3. Nothing in this file is applied until the user approves it.
#
# Override any value from the environment, so moving accounts is a one-liner:
#   AWS_ACCOUNT=123456789012 AWS_PROFILE=other ./scripts/box_up.sh
#
# CREDENTIALS. This account is CONDUIT-managed, not Isengard, so ada needs the
# provider flag explicitly -- without it ada defaults to isengard and returns a
# misleading "Account not found":
#
#   ada credentials update --provider conduit --account 654654594225 \
#       --role IibsAdminAccess-DO-NOT-DELETE --profile qa-gpu --once
#
# ada no longer falls back to that role name for Conduit, so it must be passed.

set -euo pipefail

# ---- account / region (NOT SETTLED — see D11) ------------------------------
#
# ⚠️ Region is set EXPLICITLY, not defaulted with `:=`, and deliberately so.
# A `:=` default silently inherits an ambient AWS_REGION from the shell -- which
# is exactly what happened on the first run of this script: it printed us-west-2
# because AWS_REGION was left over from an earlier command. Region is part of the
# quota evidence AND part of the reproducibility contract (GPU model, driver and
# CUDA are region-independent, but capacity and instance availability are not),
# so it must be a recorded decision rather than whatever the shell happened to
# carry.
#
# To use a different region, pass QA_REGION explicitly.
#
# Quota RE-VERIFIED in account 654654594225 on 2026-08-26, because a quota grant
# is account-specific and does NOT travel (D11). It genuinely differs from the
# previous account:
#   us-east-1  On-Demand G = 640 vCPUs, g6e.2xlarge in 4 AZs
#   us-west-2  On-Demand G = 768 vCPUs, g6e.2xlarge in 4 AZs
# Both are far above the 8 vCPUs g6e.2xlarge needs. us-east-1 is chosen because
# the SageMaker default bucket already exists there.
: "${AWS_PROFILE:=qa-gpu}"
: "${AWS_ACCOUNT:=654654594225}"
: "${QA_REGION:=us-east-1}"
AWS_REGION="${QA_REGION}"
export AWS_REGION
# AWS_DEFAULT_REGION too, since some SDK paths read that one instead.
export AWS_DEFAULT_REGION="${QA_REGION}"

# ---- instance --------------------------------------------------------------
# 1 x L40S 48GB (SM89), 8 vCPU, 64 GiB RAM. 64 GiB matters: llmcompressor's
# sequential calibration pass at 512x2048 on the 8B model needs host RAM, not VRAM.
: "${INSTANCE_TYPE:=g6e.2xlarge}"
: "${VOLUME_GB:=200}"
: "${VOLUME_TYPE:=gp3}"

# ---- NAMES — per the user's spec, revised 2026-08-26 -----------------------
#
# Scheme: johnyj02-QA-v<x.x>-<date>-<uuid>
#
# "QA" is the PROJECT INITIALS only and is NEVER expanded, in any name, tag,
# path, job id or bucket key. A human scanning resources should learn nothing
# about the research; QA reads as quality-assurance.
#
# VERSION is derived from the package version so the two cannot drift.
: "${QA_VERSION:=$(
    grep -m1 '^__version__' "${HERE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/src/quantartifact/__init__.py" 2>/dev/null \
      | sed -E 's/.*"([0-9]+)\.([0-9]+)\.[0-9]+".*/v\1.\2/' \
      || echo v0.1
  )}"
: "${RUN_DATE:=$(date -u +%Y%m%d)}"

# ---- the UUID must be STABLE, and this is not cosmetic ---------------------
#
# ⚠️ A freshly generated UUID on every invocation would BREAK IDEMPOTENCY.
# box_up.sh finds an existing instance by name so a re-run reuses it rather than
# launching a second one; with a rolling name every re-run would launch another
# $2.24/hr box and the first would keep billing unnoticed.
#
# So the UUID is generated ONCE and persisted to .box-run-id (gitignored, since
# it is machine-local state and not a project artifact). Delete that file to
# start a genuinely new box.
#
# box_up.sh additionally looks for pre-existing instances by TAG as a safety net,
# so even a lost .box-run-id cannot silently produce a duplicate.
_RUN_ID_FILE="${RUN_ID_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.box-run-id}"
if [ -z "${RUN_UUID:-}" ]; then
  if [ -f "${_RUN_ID_FILE}" ]; then
    RUN_UUID="$(tr -d '[:space:]' < "${_RUN_ID_FILE}")"
  else
    RUN_UUID="$(uuidgen | tr '[:upper:]' '[:lower:]' | cut -c1-8)"
    printf '%s\n' "${RUN_UUID}" > "${_RUN_ID_FILE}"
  fi
fi

: "${NAME_BASE:=johnyj02-QA-${QA_VERSION}-${RUN_DATE}-${RUN_UUID}}"

: "${INSTANCE_NAME:=${NAME_BASE}}"
: "${SECURITY_GROUP_NAME:=${NAME_BASE}-sg}"
: "${KEY_PAIR_NAME:=${NAME_BASE}-key}"
: "${VOLUME_NAME:=${NAME_BASE}-vol}"

# SageMaker default bucket for this account. Verified to already exist in
# us-east-1 (2026-08-26), which is why that is the region.
: "${S3_BUCKET:=sagemaker-${QA_REGION}-${AWS_ACCOUNT}}"
: "${S3_PREFIX:=s3://${S3_BUCKET}/johnyj02/QA/${QA_VERSION}/${RUN_DATE}-${RUN_UUID}/}"

# ---- tags — function only, nothing describing the research -----------------
# No project, paper, venue, model, hypothesis or benchmark names. "QA" appears
# only as initials, consistent with the naming scheme above.
: "${TAG_OWNER:=johnyj02}"
: "${TAG_PURPOSE:=testing-QA}"
: "${TAG_AUTOSTOP:=true}"

# ---- safety ----------------------------------------------------------------
# Data volume survives instance termination: generations ARE the artifact.
: "${DELETE_VOLUME_ON_TERMINATION:=false}"
# Termination protection on by default; an accidental terminate must not be cheap.
: "${TERMINATION_PROTECTION:=true}"

print_config() {
  cat <<EOF
account / region  : ${AWS_ACCOUNT} / ${AWS_REGION}   (profile: ${AWS_PROFILE})
instance type     : ${INSTANCE_TYPE}   (1 x L40S 48GB, SM89, 8 vCPU, 64 GiB)
volume            : ${VOLUME_GB} GB ${VOLUME_TYPE}, delete-on-termination=${DELETE_VOLUME_ON_TERMINATION}

NAMES  johnyj02-QA-v<x.x>-<date>-<uuid>   (user-specified, rev 2026-08-26)
  instance        : ${INSTANCE_NAME}
  security group  : ${SECURITY_GROUP_NAME}
  key pair        : ${KEY_PAIR_NAME}
  volume          : ${VOLUME_NAME}
  s3 bucket       : ${S3_BUCKET}
  s3 prefix       : ${S3_PREFIX}
  version         : ${QA_VERSION}  (derived from src/quantartifact/__init__.py)
  uuid            : ${RUN_UUID}  (persisted in .box-run-id -- STABLE across
                    re-runs, so box_up.sh stays idempotent and cannot launch a
                    second billing instance)
  ("QA" is project INITIALS only and is never expanded anywhere)

TAGS:
  Owner=${TAG_OWNER}
  Purpose=${TAG_PURPOSE}
  AutoStop=${TAG_AUTOSTOP}
  (no project/paper/venue/model/benchmark names anywhere)

SAFETY:
  termination protection : ${TERMINATION_PROTECTION}
  volume persists        : $([ "${DELETE_VOLUME_ON_TERMINATION}" = "false" ] && echo yes || echo NO)
EOF
}

# Sourced for config; executed to print it.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  print_config
fi
