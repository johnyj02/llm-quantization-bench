#!/usr/bin/env bash
# Watch a SageMaker training job to a terminal state, then print why it ended.
#
#   bash scripts/watch_job.sh                      # the most recent johnyj02-QA-* job
#   bash scripts/watch_job.sh <job-name>           # a specific one
#   bash scripts/watch_job.sh <job-name> 30        # poll every 30s (default 60)
#
# WHY THIS EXISTS. Three separate mistakes in ad-hoc polling loops:
#
#   1. The job name embeds a UTC date, and UTC rolls over mid-session. A loop with
#      the date hardcoded silently matched nothing, `--training-job-name` got an
#      empty argument, and the watcher died instantly while the job ran unobserved.
#   2. `aws sagemaker list-training-jobs` throttles ("Rate exceeded") under tight
#      polling, and with --name-contains it paginates the whole SHARED account
#      history without returning. `describe-training-job` is an O(1) lookup.
#   3. Credentials expire mid-run (~1 h). A loop with `2>/dev/null || continue`
#      swallows ExpiredTokenException and reports a stale status forever.
#
# So: resolve the name, never hardcode a date, describe rather than list, and
# treat an expired token as a refresh-and-retry rather than a status.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${AWS_PROFILE:=qa-gpu}"
: "${QA_REGION:=us-east-1}"
: "${AWS_ACCOUNT:=654654594225}"
export AWS_PROFILE

JOB="${1:-}"
INTERVAL="${2:-60}"

refresh() {
  ada credentials update --provider conduit --account "${AWS_ACCOUNT}" \
      --role IibsAdminAccess-DO-NOT-DELETE --profile "${AWS_PROFILE}" --once >/dev/null 2>&1
}

# Resolve the newest of our jobs if none was named. This is the ONE place a list
# call is justified -- it runs once, not in the loop.
if [ -z "${JOB}" ]; then
  JOB=$(aws sagemaker list-training-jobs --region "${QA_REGION}" \
        --sort-by CreationTime --sort-order Descending --max-results 20 \
        --query "TrainingJobSummaries[?starts_with(TrainingJobName,'johnyj02-QA-')].TrainingJobName | [0]" \
        --output text 2>/dev/null)
  if [ -z "${JOB}" ] || [ "${JOB}" = "None" ]; then
    refresh
    JOB=$(aws sagemaker list-training-jobs --region "${QA_REGION}" \
          --sort-by CreationTime --sort-order Descending --max-results 20 \
          --query "TrainingJobSummaries[?starts_with(TrainingJobName,'johnyj02-QA-')].TrainingJobName | [0]" \
          --output text 2>/dev/null)
  fi
fi

if [ -z "${JOB}" ] || [ "${JOB}" = "None" ]; then
  echo "could not resolve a job name; pass one explicitly" >&2
  exit 2
fi

echo "watching: ${JOB}"
echo "region  : ${QA_REGION}   profile: ${AWS_PROFILE}   interval: ${INTERVAL}s"
echo

describe() {
  aws sagemaker describe-training-job --training-job-name "${JOB}" \
      --region "${QA_REGION}" --query "$1" --output text 2>&1
}

last=""
while :; do
  out=$(describe '[TrainingJobStatus,SecondaryStatus]')
  case "${out}" in
    *ExpiredToken*|*InvalidClientTokenId*)
      echo "[$(date -u +%H:%M:%S)] credentials expired — refreshing"
      refresh
      continue ;;
    *"Requested resource not found"*)
      echo "no such job: ${JOB}" >&2
      exit 2 ;;
    *error*|*Error*)
      # Throttling and transient errors: report once, keep going. Do NOT treat as
      # a status, which is how a dead watcher looked alive.
      echo "[$(date -u +%H:%M:%S)] transient: $(echo "${out}" | tail -1 | cut -c1-100)"
      sleep "${INTERVAL}"; continue ;;
  esac

  # Only print on change, so a multi-hour queue wait stays readable.
  if [ "${out}" != "${last}" ]; then
    echo "[$(date -u +%H:%M:%S)] ${out}"
    last="${out}"
  fi

  case "${out}" in
    Completed*|Failed*|Stopped*) break ;;
  esac
  sleep "${INTERVAL}"
done

echo
echo "=== outcome ==="
echo "status        : $(describe 'TrainingJobStatus')"
echo "billable secs : $(describe 'BillableTimeInSeconds')"
echo "failure       : $(describe 'FailureReason')"
echo
echo "=== transitions ==="
describe 'SecondaryStatusTransitions[].[Status,StatusMessage]'
echo
echo "logs: bash ${REPO_ROOT}/scripts/job_log.sh ${JOB}"
