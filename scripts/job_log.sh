#!/usr/bin/env bash
# Print a training job's CloudWatch log, with the noise stripped.
#
#   bash scripts/job_log.sh <job-name>          # last 60 useful lines
#   bash scripts/job_log.sh <job-name> 200      # last 200
#   bash scripts/job_log.sh <job-name> 200 head # FIRST 200 instead of last
#
# The raw stream is mostly unusable by hand: SageMaker dumps the whole training
# environment as JSON, tqdm emits a carriage-return-separated progress line per
# update (llmcompressor's per-layer grid search produces thousands), and the
# CLI returns records tab-separated so everything lands on one line.

set -uo pipefail

: "${AWS_PROFILE:=qa-gpu}"
: "${QA_REGION:=us-east-1}"
export AWS_PROFILE

JOB="${1:?usage: job_log.sh <job-name> [lines] [head|tail]}"
LINES="${2:-60}"
WHICH="${3:-tail}"
LOG_GROUP=/aws/sagemaker/TrainingJobs

STREAM=$(aws logs describe-log-streams --log-group-name "${LOG_GROUP}" \
  --region "${QA_REGION}" --log-stream-name-prefix "${JOB}/" \
  --query 'logStreams[0].logStreamName' --output text 2>&1)

case "${STREAM}" in
  *ExpiredToken*) echo "credentials expired; run ada then retry" >&2; exit 2 ;;
  ""|None) echo "no log stream for ${JOB} (has it started?)" >&2; exit 2 ;;
esac

echo "stream: ${STREAM}"
echo

FETCH=(aws logs get-log-events --log-group-name "${LOG_GROUP}"
       --log-stream-name "${STREAM}" --region "${QA_REGION}" --limit 500
       --query 'events[].message' --output text)
[ "${WHICH}" = "head" ] && FETCH+=(--start-from-head)

"${FETCH[@]}" 2>&1 \
  | tr '\t' '\n' \
  | grep -vE '^\{|^\}|^\s+"|^\s+\]|^\s+\}|^SM_|^PYTHONPATH' \
  | grep -vE 'it/s\]|examples/s\]|#015|#033' \
  | grep -vE '^\s*$' \
  | if [ "${WHICH}" = "head" ]; then head -n "${LINES}"; else tail -n "${LINES}"; fi
