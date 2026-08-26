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

# ---- NAMES — per the user's spec, 2026-08-26 -------------------------------
#
# Scheme: testing-QA-johnyj02-<thing>-<date>, where "QA" is the PROJECT INITIALS
# only, never the expanded name. Combined with "testing" the whole string reads
# as quality-assurance testing, which is the intended effect: a human scanning
# resource names or S3 paths should learn nothing about the research.
#
# The same applies to S3 (johnyj02/testing/QA/...). Do NOT expand "QA" anywhere
# in a name, tag, path, job id, or bucket key.
: "${NAME_PREFIX:=testing-QA-johnyj02}"
: "${RUN_DATE:=$(date -u +%Y%m%d)}"

: "${INSTANCE_NAME:=${NAME_PREFIX}-job-${RUN_DATE}}"
: "${SECURITY_GROUP_NAME:=${NAME_PREFIX}-sg}"
: "${KEY_PAIR_NAME:=${NAME_PREFIX}-key}"
: "${VOLUME_NAME:=${NAME_PREFIX}-vol-${RUN_DATE}}"

# SageMaker default bucket for this account. Verified to already exist in
# us-east-1 (2026-08-26), which is why that is the region.
: "${S3_BUCKET:=sagemaker-${QA_REGION}-${AWS_ACCOUNT}}"
: "${S3_PREFIX:=s3://${S3_BUCKET}/johnyj02/testing/QA/}"

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

NAMES (user-specified scheme, 2026-08-26):
  instance        : ${INSTANCE_NAME}
  security group  : ${SECURITY_GROUP_NAME}
  key pair        : ${KEY_PAIR_NAME}
  volume          : ${VOLUME_NAME}
  s3 bucket       : ${S3_BUCKET}
  s3 prefix       : ${S3_PREFIX}
  ("QA" is project INITIALS only and is never expanded; with "testing" the
   whole string reads as quality-assurance testing)

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
