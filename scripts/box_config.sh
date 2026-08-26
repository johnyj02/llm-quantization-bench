#!/usr/bin/env bash
# Box configuration — EVERY value here is a PROPOSAL awaiting user review.
#
# Decision D11: "before you run any job you must ask me the human. Cause I want to
# make sure the job names are generic etc.. and possibly change the account"
#
# So:
#   1. Names below are deliberately GENERIC -- no paper, project, venue or
#      hypothesis identifiers anywhere, and no tag values that describe the
#      research. This withdraws the earlier `Project=QuantArtifact` /
#      `quantartifact-*` proposal in notes/compute.md.
#   2. Account, region and profile are CONFIG, never hard-coded elsewhere. The
#      account is not settled and may move.
#   3. Nothing in this file is applied until the user approves it.
#
# Override any value from the environment, so moving accounts is a one-liner:
#   AWS_ACCOUNT=123456789012 AWS_PROFILE=other ./scripts/box_up.sh

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
# To use a different region, pass QA_REGION explicitly. Quota checked 2026-08-26:
# both us-east-1 and us-west-2 have On-Demand G = 768 vCPUs and 4 AZs offering
# g6e.2xlarge, so either is viable -- but the choice is recorded, not inherited.
: "${AWS_PROFILE:=obtmodel}"
: "${AWS_ACCOUNT:=670060057797}"
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

# ---- PROPOSED NAMES — generic, for user review -----------------------------
# Rationale for each: identifies the owner and the machine's function, and
# nothing about the research. Swap freely; nothing depends on these strings.
: "${NAME_PREFIX:=johnyj02-ml}"
: "${INSTANCE_NAME:=${NAME_PREFIX}-gpu-01}"
: "${SECURITY_GROUP_NAME:=${NAME_PREFIX}-sg}"
: "${KEY_PAIR_NAME:=${NAME_PREFIX}-key}"
: "${VOLUME_NAME:=${NAME_PREFIX}-data-01}"
: "${S3_PREFIX:=s3://sagemaker-${AWS_REGION}-${AWS_ACCOUNT}/johnyj02-ml/}"

# ---- tags — function only, nothing describing the research -----------------
: "${TAG_OWNER:=johnyj02}"
: "${TAG_PURPOSE:=ml-experiments}"
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

PROPOSED NAMES (D11 — awaiting your review):
  instance        : ${INSTANCE_NAME}
  security group  : ${SECURITY_GROUP_NAME}
  key pair        : ${KEY_PAIR_NAME}
  volume          : ${VOLUME_NAME}
  s3 prefix       : ${S3_PREFIX}

PROPOSED TAGS:
  Owner=${TAG_OWNER}
  Purpose=${TAG_PURPOSE}
  AutoStop=${TAG_AUTOSTOP}
  (no project/paper/venue identifiers, no tag describing the research)

SAFETY:
  termination protection : ${TERMINATION_PROTECTION}
  volume persists        : $([ "${DELETE_VOLUME_ON_TERMINATION}" = "false" ] && echo yes || echo NO)
EOF
}

# Sourced for config; executed to print it.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  print_config
fi
