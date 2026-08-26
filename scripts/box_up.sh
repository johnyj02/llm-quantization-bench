#!/usr/bin/env bash
# Provision the GPU box. DRY-RUN BY DEFAULT.
#
#   bash scripts/box_up.sh                  # print the plan, create nothing
#   bash scripts/box_up.sh --apply          # actually create (needs user approval)
#
# Decision D10/D11: nothing is created until the user has approved the plan,
# including every name. So --apply is required AND the script prints the full
# name list first either way.
#
# Creates only NEW resources, all clearly owned. Never modifies, terminates, or
# deletes anything pre-existing in the account.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=box_config.sh
source "${HERE}/box_config.sh"

APPLY=0
WAIT=0
for arg in "$@"; do
  case "${arg}" in
    --apply) APPLY=1 ;;
    --wait)  WAIT=1 ;;
  esac
done
# How long to keep retrying when the only problem is capacity, and how often.
: "${WAIT_INTERVAL_SEC:=60}"
: "${WAIT_MAX_MIN:=180}"

aws_() { aws --profile "${AWS_PROFILE}" --region "${QA_REGION}" "$@"; }

echo "======================================================================"
echo " PROVISION PLAN"
echo "======================================================================"
print_config
echo

# ---- identity, so we never create in the wrong account ---------------------
echo "=== caller identity ==="
actual_account=$(aws_ sts get-caller-identity --query Account --output text)
aws_ sts get-caller-identity --query '[Account,Arn]' --output text | sed 's/^/  /'
if [ "${actual_account}" != "${AWS_ACCOUNT}" ]; then
  echo
  echo "REFUSING: credentials resolve to account ${actual_account}, but this plan"
  echo "targets ${AWS_ACCOUNT}. Set AWS_ACCOUNT explicitly if the move is intended"
  echo "(D11: the account is not settled), and re-verify quota for the new account —"
  echo "the 768-vCPU On-Demand G grant is account-specific and is the sole reason"
  echo "EC2 is viable here."
  exit 1
fi

# ---- capacity, since quota is not capacity --------------------------------
echo "=== ${INSTANCE_TYPE} availability in ${QA_REGION} ==="
aws_ ec2 describe-instance-type-offerings --location-type availability-zone \
  --filters "Name=instance-type,Values=${INSTANCE_TYPE}" \
  --query 'InstanceTypeOfferings[].Location' --output text | tr '\t' '\n' | sed 's/^/  /'

echo "=== On-Demand G quota (must exceed this instance's vCPUs) ==="
aws_ service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA \
  --query 'Quota.[QuotaName,Value]' --output text | sed 's/^/  /'

# ---- latest Deep Learning AMI ---------------------------------------------
echo "=== base image ==="
AMI_ID="${AMI_ID:-$(aws_ ec2 describe-images --owners amazon \
  --filters 'Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*' \
            'Name=state,Values=available' \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text 2>/dev/null)}"
if [ -z "${AMI_ID}" ] || [ "${AMI_ID}" = "None" ]; then
  echo "  could not resolve a Deep Learning AMI; set AMI_ID explicitly"
  exit 1
fi
aws_ ec2 describe-images --image-ids "${AMI_ID}" \
  --query 'Images[0].[ImageId,Name,CreationDate]' --output text | sed 's/^/  /'
echo "  (chosen for a preinstalled NVIDIA driver AND build toolchain — the C"
echo "   compiler is REQUIRED for Triton to JIT the batch-invariant kernels)"

if [ "${APPLY}" = "0" ]; then
  cat <<'EOF'

======================================================================
 DRY RUN — NOTHING WAS CREATED
======================================================================
Review the names above. To proceed, the user must approve, then:

    bash scripts/box_up.sh --apply

After it boots, ON THE BOX:
    bash scripts/box_verify.sh      # refuses to bless a box that cannot JIT Triton
EOF
  exit 0
fi

echo
echo "=== APPLYING ==="

# key pair -- private key stays local, never committed.
#
# The local filename carries the REGION, because EC2 key pairs are region-scoped:
# the same KEY_PAIR_NAME in two regions is two different keys with two different
# private halves. Sharing one filename silently pairs a key with the wrong region
# -- and the first attempt failed exactly here, with "Permission denied" writing
# over the mode-400 us-east-1 key.
KEY_FILE="${HOME}/.ssh/${KEY_PAIR_NAME}-${QA_REGION}.pem"

if ! aws_ ec2 describe-key-pairs --key-names "${KEY_PAIR_NAME}" >/dev/null 2>&1; then
  mkdir -p ~/.ssh
  # Mode 400 blocks overwrite, so relax before writing if a stale file is present.
  [ -f "${KEY_FILE}" ] && chmod 600 "${KEY_FILE}"
  aws_ ec2 create-key-pair --key-name "${KEY_PAIR_NAME}" \
    --query 'KeyMaterial' --output text > "${KEY_FILE}"
  chmod 400 "${KEY_FILE}"
  echo "  created key pair ${KEY_PAIR_NAME} (${QA_REGION}) -> ${KEY_FILE}"
else
  echo "  key pair ${KEY_PAIR_NAME} already exists in ${QA_REGION}"
  if [ ! -f "${KEY_FILE}" ]; then
    echo
    echo "REFUSING: the key pair exists in AWS but its private half is NOT at"
    echo "  ${KEY_FILE}"
    echo "AWS only returns the private key at creation time, so a box launched with"
    echo "it would be unreachable. Delete the AWS key pair and re-run to get a fresh"
    echo "pair, or restore the .pem:"
    echo "  aws --profile ${AWS_PROFILE} --region ${QA_REGION} ec2 delete-key-pair --key-name ${KEY_PAIR_NAME}"
    exit 1
  fi
fi

# security group -- SSH only, from this machine's IP only
if ! aws_ ec2 describe-security-groups --group-names "${SECURITY_GROUP_NAME}" >/dev/null 2>&1; then
  vpc_id=$(aws_ ec2 describe-vpcs --filters Name=isDefault,Values=true \
             --query 'Vpcs[0].VpcId' --output text)
  sg_id=$(aws_ ec2 create-security-group --group-name "${SECURITY_GROUP_NAME}" \
            --description "SSH access for ML experiments" --vpc-id "${vpc_id}" \
            --query 'GroupId' --output text)
  my_ip=$(curl -s https://checkip.amazonaws.com || echo "")
  if [ -n "${my_ip}" ]; then
    aws_ ec2 authorize-security-group-ingress --group-id "${sg_id}" \
      --protocol tcp --port 22 --cidr "${my_ip}/32" >/dev/null
    echo "  created security group ${SECURITY_GROUP_NAME} (${sg_id}), SSH from ${my_ip}/32 only"
  else
    echo "  created ${sg_id} but could NOT determine your IP — add an SSH rule manually"
  fi
else
  sg_id=$(aws_ ec2 describe-security-groups --group-names "${SECURITY_GROUP_NAME}" \
            --query 'SecurityGroups[0].GroupId' --output text)
  echo "  security group ${SECURITY_GROUP_NAME} already exists (${sg_id})"
fi

# ---- instance, with TWO duplicate guards -----------------------------------
#
# Guard 1: exact name. Normal idempotent re-run.
existing=$(aws_ ec2 describe-instances \
  --filters "Name=tag:Name,Values=${INSTANCE_NAME}" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' --output text)

# Guard 2: by TAG, ignoring the name. The name embeds a date and a uuid, so a
# lost .box-run-id (or simply running on a later date) would produce a DIFFERENT
# name for the same logical box and guard 1 would miss it -- launching a second
# $2.24/hr instance while the first kept billing unnoticed. This catches that.
if [ -z "${existing}" ]; then
  by_tag=$(aws_ ec2 describe-instances \
    --filters "Name=tag:Owner,Values=${TAG_OWNER}" \
              "Name=tag:Purpose,Values=${TAG_PURPOSE}" \
              "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].[InstanceId,Tags[?Key==`Name`].Value|[0],State.Name]' \
    --output text)
  if [ -n "${by_tag}" ]; then
    echo
    echo "REFUSING: an instance with Owner=${TAG_OWNER} Purpose=${TAG_PURPOSE} already exists,"
    echo "under a DIFFERENT name than this run would create:"
    echo "${by_tag}" | sed 's/^/    /'
    echo
    echo "This run would have created: ${INSTANCE_NAME}"
    echo
    echo "The name embeds a date and a uuid, so a new date or a lost .box-run-id"
    echo "produces a new name for the same logical box. Launching would leave TWO"
    echo "instances billing at \$2.24/hr."
    echo
    echo "Either reuse the existing one, or if you genuinely want a second box:"
    echo "    ALLOW_SECOND_BOX=1 bash scripts/box_up.sh --apply"
    [ "${ALLOW_SECOND_BOX:-0}" = "1" ] || exit 1
    echo "ALLOW_SECOND_BOX=1 set — proceeding anyway."
  fi
fi

if [ -n "${existing}" ]; then
  echo "  instance ${INSTANCE_NAME} already exists: ${existing}"
  echo "  (idempotent: not creating a second one. Use box_down.sh to stop it.)"
  instance_id="${existing}"
else
  # ---- try every AZ: QUOTA IS NOT CAPACITY ---------------------------------
  #
  # The first launch attempt failed with InsufficientInstanceCapacity despite
  # 640 vCPUs of quota. g6e is a relatively scarce instance family, and
  # run-instances without a subnet lets AWS pick one AZ and give up if it is
  # full. So iterate explicitly over the default VPC's subnets.
  vpc_id=$(aws_ ec2 describe-vpcs --filters Name=isDefault,Values=true \
             --query 'Vpcs[0].VpcId' --output text)
  # Only AZs that actually offer this instance type, so we do not waste attempts.
  offered=$(aws_ ec2 describe-instance-type-offerings --location-type availability-zone \
              --filters "Name=instance-type,Values=${INSTANCE_TYPE}" \
              --query 'InstanceTypeOfferings[].Location' --output text)

  # Order AZs by CURRENT SPOT PRICE, ascending. Spot price floats with demand, so
  # the cheapest AZ is the least contended and the likeliest on-demand success.
  # Falls back to the unordered list if spot history is unavailable.
  spot_start=$(date -u -v-3H +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -u -d '3 hours ago' +%Y-%m-%dT%H:%M:%S)
  ranked=$(aws_ ec2 describe-spot-price-history --instance-types "${INSTANCE_TYPE}" \
    --product-descriptions "Linux/UNIX" --start-time "${spot_start}" \
    --query 'SpotPriceHistory[].[SpotPrice,AvailabilityZone]' --output text 2>/dev/null \
    | sort -u -k2,2 | sort -n -k1,1 | awk '{print $2}')
  for az in ${ranked}; do
    case " ${offered} " in *" ${az} "*) ordered="${ordered:-} ${az}" ;; esac
  done
  for az in ${offered}; do
    case " ${ordered:-} " in *" ${az} "*) ;; *) ordered="${ordered:-} ${az}" ;; esac
  done
  attempt_azs="${ordered:-${offered}}"
  echo "  AZ order (least-contended first, by spot price):${attempt_azs}"

  deadline=$(( $(date +%s) + WAIT_MAX_MIN * 60 ))
  round=0
  instance_id=""

while :; do
  round=$(( round + 1 ))
  for az in ${attempt_azs}; do
    subnet=$(aws_ ec2 describe-subnets \
      --filters "Name=vpc-id,Values=${vpc_id}" "Name=availability-zone,Values=${az}" \
      --query 'Subnets[0].SubnetId' --output text 2>/dev/null)
    [ -z "${subnet}" ] || [ "${subnet}" = "None" ] && continue

    echo "  trying ${az} (${subnet})..."
    if instance_id=$(aws_ ec2 run-instances \
        --image-id "${AMI_ID}" \
        --instance-type "${INSTANCE_TYPE}" \
        --key-name "${KEY_PAIR_NAME}" \
        --security-group-ids "${sg_id}" \
        --subnet-id "${subnet}" \
        --associate-public-ip-address \
        --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${VOLUME_GB},\"VolumeType\":\"${VOLUME_TYPE}\",\"DeleteOnTermination\":${DELETE_VOLUME_ON_TERMINATION}}}]" \
        --tag-specifications \
          "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}},{Key=Owner,Value=${TAG_OWNER}},{Key=Purpose,Value=${TAG_PURPOSE}},{Key=AutoStop,Value=${TAG_AUTOSTOP}}]" \
          "ResourceType=volume,Tags=[{Key=Name,Value=${VOLUME_NAME}},{Key=Owner,Value=${TAG_OWNER}},{Key=Purpose,Value=${TAG_PURPOSE}}]" \
        --query 'Instances[0].InstanceId' --output text 2>/dev/null); then
      echo "  launched ${instance_id} in ${az}"
      break
    fi
    instance_id=""
    echo "    no capacity in ${az}"
  done

  [ -n "${instance_id}" ] && break

  # Only loop when the user asked to wait AND we still have time.
  if [ "${WAIT}" = "1" ] && [ "$(date +%s)" -lt "${deadline}" ]; then
    left=$(( (deadline - $(date +%s)) / 60 ))
    echo "  round ${round}: no capacity in any AZ. Retrying in ${WAIT_INTERVAL_SEC}s (${left} min left)."
    echo "  (nothing is billing while we wait; only free key pair + security group exist)"
    sleep "${WAIT_INTERVAL_SEC}"
    continue
  fi
  break
done

  if [ -z "${instance_id}" ]; then
    echo
    echo "NO CAPACITY for ${INSTANCE_TYPE} in any AZ of ${QA_REGION}."
    echo "Quota is fine (640 vCPUs) -- this is physical capacity, which quota does"
    echo "not guarantee. Options, cheapest first:"
    echo "  1. wait for capacity:  bash scripts/box_up.sh --apply --wait"
    echo "     (polls every ${WAIT_INTERVAL_SEC}s for up to ${WAIT_MAX_MIN} min, grabs it the moment it frees)"
    echo "  2. check pressure first: bash scripts/check_capacity.sh"
    echo "  3. a different instance type -- but that is a SUBSTRATE change and"
    echo "     4.6 pins hardware, so it needs a decision, not a fallback"
    echo
    echo "Nothing was left running. The key pair and security group persist and"
    echo "will be reused on the next attempt (both are free)."
    exit 1
  fi

  if [ "${TERMINATION_PROTECTION}" = "true" ]; then
    aws_ ec2 modify-instance-attribute --instance-id "${instance_id}" \
      --disable-api-termination >/dev/null
    echo "  termination protection enabled"
  fi
fi

echo "  waiting for running state..."
aws_ ec2 wait instance-running --instance-ids "${instance_id}"
public_dns=$(aws_ ec2 describe-instances --instance-ids "${instance_id}" \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text)

cat <<EOF

======================================================================
 UP
======================================================================
instance : ${instance_id}
dns      : ${public_dns}
ssh      : ssh -i ${KEY_FILE} ubuntu@${public_dns}

NEXT, ON THE BOX — before trusting any number it produces:
    bash scripts/box_verify.sh

⚠️  \$$(printf '%.2f' 2.24)/hr while running. Stop it when idle:
    bash scripts/box_down.sh
EOF
