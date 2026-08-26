#!/usr/bin/env bash
# STOP the GPU box (does not terminate, does not touch the volume).
#
#   bash scripts/box_down.sh            # stop
#   bash scripts/box_down.sh --status   # just report state and cost so far
#
# Stopped EC2 bills EBS only (~$16/mo for 200 GB gp3), and unlike RunPod the
# volume price does not double while stopped -- which is the main reason the
# substrate change was affordable. At $2.24/hr an unattended weekend is ~$108,
# so this is the highest-value habit in the project.
#
# Deliberately NEVER terminates. Generations are the artifact, termination
# protection is on, and the data volume is set to survive termination anyway.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=box_config.sh
source "${HERE}/box_config.sh"

aws_() { aws --profile "${AWS_PROFILE}" --region "${QA_REGION}" "$@"; }

instance_id=$(aws_ ec2 describe-instances \
  --filters "Name=tag:Name,Values=${INSTANCE_NAME}" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' --output text)

if [ -z "${instance_id}" ]; then
  echo "no instance named ${INSTANCE_NAME} in ${QA_REGION} — nothing to stop"
  exit 0
fi

read -r state launch_time < <(aws_ ec2 describe-instances --instance-ids "${instance_id}" \
  --query 'Reservations[0].Instances[0].[State.Name,LaunchTime]' --output text)

echo "instance : ${instance_id} (${INSTANCE_NAME})"
echo "state    : ${state}"
echo "launched : ${launch_time}"

if [ "${1:-}" = "--status" ]; then
  exit 0
fi

if [ "${state}" = "stopped" ]; then
  echo "already stopped — billing EBS only"
  exit 0
fi

aws_ ec2 stop-instances --instance-ids "${instance_id}" \
  --query 'StoppingInstances[0].[InstanceId,CurrentState.Name]' --output text | sed 's/^/  /'
echo "  waiting for stopped state..."
aws_ ec2 wait instance-stopped --instance-ids "${instance_id}"
echo "STOPPED — now billing EBS only (~\$16/mo for ${VOLUME_GB}GB ${VOLUME_TYPE})"
echo "Restart with: bash scripts/box_up.sh --apply   (idempotent; will not create a second instance)"
