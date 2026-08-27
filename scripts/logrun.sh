#!/usr/bin/env bash
# Append one run to notes/RUNLOG.md, with the numbers pulled from the SageMaker API.
#
#   bash scripts/logrun.sh <job-name> "one-line outcome"
#   bash scripts/logrun.sh                       # re-print the ledger totals only
#
# WHY A SCRIPT. The runlog's value is that its numbers are trustworthy. Typing
# billable seconds from memory is exactly how a ledger stops being one -- and I have
# already mis-stated a test count in this project from recall. So status, billable
# time and cost are always read from the API here; only the prose is hand-written.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNLOG="${REPO_ROOT}/notes/RUNLOG.md"
: "${AWS_PROFILE:=qa-gpu}"
: "${QA_REGION:=us-east-1}"
: "${RATE:=2.80}"        # USE1-Train:ml.g6e.2xlarge, USD/hour
export AWS_PROFILE

describe() {
  aws sagemaker describe-training-job --training-job-name "$1" \
      --region "${QA_REGION}" --query "$2" --output text 2>&1
}

# ---- totals across every job of ours, for the running ledger line ---------
totals() {
  local names total=0
  names=$(aws sagemaker list-training-jobs --region "${QA_REGION}" --max-results 100 \
    --query "TrainingJobSummaries[?starts_with(TrainingJobName,'johnyj02-QA-')].TrainingJobName" \
    --output text 2>/dev/null | tr '\t' '\n' | grep -v '^$')
  local count=0
  for n in ${names}; do
    local secs
    secs=$(describe "${n}" 'BillableTimeInSeconds')
    case "${secs}" in ''|None|*error*|*Error*) secs=0 ;; esac
    total=$(( total + secs ))
    count=$(( count + 1 ))
  done
  python3 -c "
t=${total}; c=${count}; r=${RATE}
print(f'{c} jobs · {t:,} billable s · {t/3600:.2f} h · \${t/3600*r:.2f}')"
}

if [ $# -eq 0 ]; then
  echo "running total: $(totals)"
  exit 0
fi

JOB="$1"
OUTCOME="${2:-}"

STATUS=$(describe "${JOB}" 'TrainingJobStatus')
case "${STATUS}" in
  *ExpiredToken*)
    echo "credentials expired — run ada, then retry" >&2; exit 2 ;;
  *"Requested resource not found"*)
    echo "no such job: ${JOB}" >&2; exit 2 ;;
esac

SECS=$(describe "${JOB}" 'BillableTimeInSeconds')
case "${SECS}" in ''|None|*error*) SECS=0 ;; esac
FAILURE=$(describe "${JOB}" 'FailureReason' | head -4 | tr '\n' ' ')
COST=$(python3 -c "print(f'\${${SECS}/3600*${RATE}:.2f}')")

# Not yet finished? Say so rather than logging a half-run as if it were final.
case "${STATUS}" in
  InProgress|Stopping)
    echo "${JOB} is still ${STATUS} — not appending. Log it once it is terminal." >&2
    exit 1 ;;
esac

{
  printf '\n## %s · %s · %s s · %s\n\n' "${JOB##*-33b4267e-}" "${STATUS}" "${SECS}" "${COST}"
  printf '`%s`\n\n' "${JOB}"
  [ -n "${OUTCOME}" ] && printf -- '- %s\n' "${OUTCOME}"
  case "${STATUS}" in
    Failed) printf -- '- FailureReason: `%s`\n' "${FAILURE}" ;;
  esac
  printf -- '- <!-- TODO: what it taught us, and what changed as a result -->\n'
} >> "${RUNLOG}"

echo "appended to notes/RUNLOG.md:"
printf '  %s · %s · %s s · %s\n' "${JOB##*-33b4267e-}" "${STATUS}" "${SECS}" "${COST}"
echo "  running total: $(totals)"
echo
echo "Now edit the TODO line in notes/RUNLOG.md with what it taught us."
