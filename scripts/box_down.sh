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

# Look up by TAG, not by name. The name embeds a date and a uuid, so tomorrow's
# INSTANCE_NAME would not match today's box -- and a stop script that silently
# finds nothing is how a $2.24/hr instance runs all weekend.
instance_id=$(aws_ ec2 describe-instances \
  --filters "Name=tag:Owner,Values=${TAG_OWNER}" \
            "Name=tag:Purpose,Values=${TAG_PURPOSE}" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' --output text)

if [ -z "${instance_id}" ]; then
  echo "no instance tagged Owner=${TAG_OWNER} Purpose=${TAG_PURPOSE} in ${QA_REGION}"
  echo "— nothing to stop"
  exit 0
fi

# Stop ALL matches, not just the first: if a duplicate ever slipped through, this
# is the script that has to catch it rather than leave one billing.
n_found=$(printf '%s\n' ${instance_id} | wc -w | tr -d ' ')
[ "${n_found}" -gt 1 ] && echo "NOTE: ${n_found} matching instances found; stopping all of them"


aws_ ec2 describe-instances --instance-ids ${instance_id} \
  --query 'Reservations[].Instances[].[InstanceId,Tags[?Key==`Name`].Value|[0],State.Name,LaunchTime]' \
  --output text | sed 's/^/  /'
state=$(aws_ ec2 describe-instances --instance-ids ${instance_id} \
  --query 'Reservations[0].Instances[0].State.Name' --output text)

if [ "${1:-}" = "--status" ]; then
  exit 0
fi

if [ "${state}" = "stopped" ]; then
  echo "already stopped — billing EBS only"
  exit 0
fi

aws_ ec2 stop-instances --instance-ids ${instance_id} \
  --query 'StoppingInstances[0].[InstanceId,CurrentState.Name]' --output text | sed 's/^/  /'
echo "  waiting for stopped state..."
aws_ ec2 wait instance-stopped --instance-ids ${instance_id}
echo "STOPPED — now billing EBS only (~\$16/mo for ${VOLUME_GB}GB ${VOLUME_TYPE})"
echo "Restart with: bash scripts/box_up.sh --apply   (idempotent; will not create a second instance)"
