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
[ "${1:-}" = "--apply" ] && APPLY=1

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

# key pair -- private key stays local, never committed
if ! aws_ ec2 describe-key-pairs --key-names "${KEY_PAIR_NAME}" >/dev/null 2>&1; then
  mkdir -p ~/.ssh
  aws_ ec2 create-key-pair --key-name "${KEY_PAIR_NAME}" \
    --query 'KeyMaterial' --output text > "${HOME}/.ssh/${KEY_PAIR_NAME}.pem"
  chmod 400 "${HOME}/.ssh/${KEY_PAIR_NAME}.pem"
  echo "  created key pair ${KEY_PAIR_NAME} -> ~/.ssh/${KEY_PAIR_NAME}.pem"
else
  echo "  key pair ${KEY_PAIR_NAME} already exists"
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

# instance
existing=$(aws_ ec2 describe-instances \
  --filters "Name=tag:Name,Values=${INSTANCE_NAME}" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' --output text)

if [ -n "${existing}" ]; then
  echo "  instance ${INSTANCE_NAME} already exists: ${existing}"
  echo "  (idempotent: not creating a second one. Use box_down.sh to stop it.)"
  instance_id="${existing}"
else
  instance_id=$(aws_ ec2 run-instances \
    --image-id "${AMI_ID}" \
    --instance-type "${INSTANCE_TYPE}" \
    --key-name "${KEY_PAIR_NAME}" \
    --security-group-ids "${sg_id}" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${VOLUME_GB},\"VolumeType\":\"${VOLUME_TYPE}\",\"DeleteOnTermination\":${DELETE_VOLUME_ON_TERMINATION}}}]" \
    --tag-specifications \
      "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}},{Key=Owner,Value=${TAG_OWNER}},{Key=Purpose,Value=${TAG_PURPOSE}},{Key=AutoStop,Value=${TAG_AUTOSTOP}}]" \
      "ResourceType=volume,Tags=[{Key=Name,Value=${VOLUME_NAME}},{Key=Owner,Value=${TAG_OWNER}},{Key=Purpose,Value=${TAG_PURPOSE}}]" \
    --query 'Instances[0].InstanceId' --output text)
  echo "  launched ${instance_id}"

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
ssh      : ssh -i ~/.ssh/${KEY_PAIR_NAME}.pem ubuntu@${public_dns}

NEXT, ON THE BOX — before trusting any number it produces:
    bash scripts/box_verify.sh

⚠️  \$$(printf '%.2f' 2.24)/hr while running. Stop it when idle:
    bash scripts/box_down.sh
EOF
