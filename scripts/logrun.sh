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
export AWS_PROFILE

describe() {
  aws sagemaker describe-training-job --training-job-name "$1" \
      --region "${QA_REGION}" --query "$2" --output text 2>&1
}

# ---- USD/hour for one job, by the instance type it actually ran on --------------
# A literal $2.80 was right only while every job was a g6e. Rates are USE1 SageMaker Training,
# read from the Pricing API with regionCode=us-east-1 on 2026-10-16. A type not listed is
# REFUSED rather than priced at a guess, because one wrong row corrupts every later total.
rate_for() {
  local it
  it=$(describe "$1" 'ResourceConfig.InstanceType')
  case "${it}" in
    ml.g6e.2xlarge)  echo 2.80 ;;
    ml.g5.2xlarge)   echo 1.515 ;;
    ml.p4d.24xlarge) echo 25.2513 ;;
    *ExpiredToken*|*Throttling*|*rror*)
       echo "API call failed for $1 (${it}); refresh credentials or wait, then retry" >&2
       return 1 ;;
    *) echo "no rate for $1 (instance type '${it}'); add it from the Pricing API" >&2
       return 1 ;;
  esac
}

# ---- every job of ours, by name, oldest first ---------------------------------
# 🚨 BOUNDED BY CREATION TIME, and that is the whole fix. This used to ask for the newest
# `--max-results 100` jobs in the account and filter for ours. The account is SHARED, and on
# 2026-10-14 another team's `OBTGenericContainer-*` jobs filled that window, so 47 of our 48
# jobs fell outside it and the next append REWROTE THE LEDGER WITH ONE ROW. Same silent
# truncation as `--max-items 400` (CLAUDE.md SS7): a capped query fails by returning plausible
# emptiness, not an error. `--name-contains` alone paginates the whole account history and did
# not return in two minutes; with `--creation-time-after` it returns all of ours in ~20 s.
STUDY_START="2026-08-20T00:00:00Z"
our_jobs() {
  aws sagemaker list-training-jobs --region "${QA_REGION}" \
    --name-contains johnyj02-QA- --creation-time-after "${STUDY_START}" \
    --sort-by CreationTime --sort-order Ascending \
    --query 'TrainingJobSummaries[].TrainingJobName' --output text 2>/dev/null \
    | tr '\t' '\n' | grep -v '^$'
}

# ---- totals across every job of ours, for the running ledger line ---------
totals() {
  local names total=0
  names=$(our_jobs)
  local count=0 dollars=0
  for n in ${names}; do
    local secs r
    secs=$(describe "${n}" 'BillableTimeInSeconds')
    case "${secs}" in ''|None|*error*|*Error*) secs=0 ;; esac
    r=$(rate_for "${n}") || exit 3
    dollars=$(python3 -c "print(${dollars} + ${secs}/3600*${r})")
    total=$(( total + secs ))
    count=$(( count + 1 ))
  done
  python3 -c "
t=${total}; c=${count}; d=${dollars}
print(f'{c} jobs · {t:,} billable s · {t/3600:.2f} h · \${d:.2f}')"
}

# ---- regenerate the ledger block from the API ------------------------------
# ⚠️ The ledger used to be a hand-maintained markdown table, and it drifted exactly as
# you would expect: 10 rows and a "$14.58 total" while 16 jobs and $16.13 existed. A
# summary that CAN drift will. So it is generated between markers on every append, from
# the same API the per-run numbers come from. The prose sections below it stay
# append-only, because that is the part with judgement in it.
refresh_ledger() {
  local names rows="" total=0 count=0 dollars=0
  # ⚠️ DESCENDING, then reversed locally. Asking for Ascending with --max-results 100
  # returns the OLDEST 100 jobs in the account -- and this account is SHARED with
  # hundreds of other users' jobs, so OURS (the newest) fall outside the window and the
  # filter silently matches NOTHING. Same trap as the `--max-items 400` SageMaker quota
  # truncation in CLAUDE.md §7: a capped, wrongly-ordered query fails by returning
  # plausible emptiness rather than an error.
  names=$(our_jobs)
  for n in ${names}; do
    local st secs cost short r
    st=$(describe "${n}" 'TrainingJobStatus')
    secs=$(describe "${n}" 'BillableTimeInSeconds')
    case "${secs}" in ''|None|*error*) secs=0 ;; esac
    r=$(rate_for "${n}") || exit 3
    total=$(( total + secs )); count=$(( count + 1 ))
    dollars=$(python3 -c "print(${dollars} + ${secs}/3600*${r})")
    cost=$(python3 -c "print(f'\${${secs}/3600*${r}:.2f}')")
    short="${n##*-33b4267e-}"
    rows="${rows}| ${count} | \`${short}\` | ${st} | ${secs} s | ${cost} |"$'\n'
  done
  # ⚠️ rows travels via the ENVIRONMENT, not as an argument after the heredoc.
  # `python3 - ... <<'EOF' "$rows"` does NOT put "$rows" on stdin -- the heredoc owns
  # stdin, so the argument is silently ignored and the table renders EMPTY while the
  # totals line looks correct. Which is exactly what it did on first writing.
  LEDGER_ROWS="${rows}" python3 - "$RUNLOG" "$total" "$count" "$dollars" <<'PYEOF'
import os, sys, pathlib
runlog, total, count, dollars = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
rows = os.environ.get("LEDGER_ROWS", "")
block = (
    "<!-- LEDGER:BEGIN — generated by scripts/logrun.sh; do not hand-edit -->\n"
    f"**{count} jobs · {total:,} billable s · {total/3600:.2f} h · "
    f"${dollars:.2f}** (each job at its own instance type's USE1-Train rate)\n\n"
    "| # | job | status | billable | cost |\n|---|---|---|---|---|\n"
    + rows +
    "\n<!-- LEDGER:END -->"
)
text = runlog.read_text()
start, end = text.find("<!-- LEDGER:BEGIN"), text.find("<!-- LEDGER:END -->")
# 🚨 A ledger may only GROW. Jobs are never deleted, so a regenerated block with fewer rows
# than the one on disk means the listing was truncated, not that history shrank. That is
# exactly how 47 rows became 1 on 2026-10-14. Refuse rather than overwrite.
if start != -1 and end != -1:
    old_rows = sum(1 for ln in text[start:end].splitlines() if ln.startswith("| ") and ln[2:3].isdigit())
    if count < old_rows:
        sys.exit(f"REFUSING to rewrite the ledger: the API returned {count} jobs but the "
                 f"ledger already holds {old_rows}. The listing was truncated; nothing written.")
if start != -1 and end != -1:
    text = text[:start] + block + text[end + len("<!-- LEDGER:END -->"):]
    runlog.write_text(text)
    print("  ledger block refreshed from the API")
else:
    print("  (no LEDGER markers found in notes/RUNLOG.md — block not refreshed)", file=sys.stderr)
PYEOF
}

if [ $# -eq 0 ]; then
  echo "running total: $(totals)"
  refresh_ledger
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
JOB_RATE=$(rate_for "${JOB}") || exit 3
COST=$(python3 -c "print(f'\${${SECS}/3600*${JOB_RATE}:.2f}')")

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

refresh_ledger
echo "appended to notes/RUNLOG.md:"
printf '  %s · %s · %s s · %s\n' "${JOB##*-33b4267e-}" "${STATUS}" "${SECS}" "${COST}"
echo "  running total: $(totals)"
echo
echo "Now edit the TODO line in notes/RUNLOG.md with what it taught us."
