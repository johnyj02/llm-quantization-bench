#!/usr/bin/env bash
# Check GPU capacity signals. READ-ONLY — creates nothing, costs nothing.
#
#   bash scripts/check_capacity.sh                    # default: g6e.2xlarge, both regions
#   bash scripts/check_capacity.sh g5.2xlarge         # a different type
#   bash scripts/check_capacity.sh g6e.2xlarge us-west-2
#
# ⚠️ AWS publishes NO capacity API. "Is there capacity right now" is only truly
# answerable by attempting a launch. What this script gives you is the best
# available proxy, which is genuinely informative but not a guarantee:
#
#   1. Offering  — does the AZ carry this type at all? (hard fact)
#   2. Quota     — is your account allowed this many vCPUs? (hard fact)
#   3. Spot price— how contended is the pool? (PROXY)
#
# Spot price is the useful one. It floats with demand, so a spot price near
# on-demand means the pool is under pressure and an on-demand launch is likely to
# fail; a spot price well below on-demand means slack. Caveat worth knowing: spot
# and on-demand draw on related but not identical pools, so this is a signal, not
# a promise.
#
# To actually WAIT for capacity and grab it the moment it appears:
#   QA_REGION=us-west-2 bash scripts/box_up.sh --apply --wait

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=box_config.sh
source "${HERE}/box_config.sh"

TYPE="${1:-${INSTANCE_TYPE}}"
REGIONS="${2:-us-west-2 us-east-1}"

# On-demand EC2 price, PER REGION, so the spot ratio means something.
#
# ⚠️ The regionCode filter is mandatory. Without it the Pricing API returns some
# other region's row: this script first reported $2.8035/hr for g6e.2xlarge, which
# is the SAGEMAKER rate, against a true EC2 rate of $2.2421 -- understating
# contention by ~25% across the board.
od_price_for() {
  aws --profile "${AWS_PROFILE}" pricing get-products --region us-east-1 \
    --service-code AmazonEC2 \
    --filters "Type=TERM_MATCH,Field=instanceType,Value=${TYPE}" \
              "Type=TERM_MATCH,Field=operatingSystem,Value=Linux" \
              "Type=TERM_MATCH,Field=tenancy,Value=Shared" \
              "Type=TERM_MATCH,Field=preInstalledSw,Value=NA" \
              "Type=TERM_MATCH,Field=capacitystatus,Value=Used" \
              "Type=TERM_MATCH,Field=regionCode,Value=$1" \
    --max-results 1 --query 'PriceList[0]' --output text 2>/dev/null \
    | python3 -c "import sys,json;d=json.load(sys.stdin);t=list(d['terms']['OnDemand'].values())[0];print(list(t['priceDimensions'].values())[0]['pricePerUnit']['USD'])" 2>/dev/null || echo ""
}

echo "======================================================================"
echo " CAPACITY SIGNALS — ${TYPE}"
echo "======================================================================"
echo "account      : ${AWS_ACCOUNT} (profile ${AWS_PROFILE})"
echo "note         : prices below are EC2 on-demand, NOT SageMaker (which is ~25% more)"
echo
echo "Lower spot %% of on-demand = LESS contention = more likely to launch."
echo

best_region=""
best_score=999

for r in ${REGIONS}; do
  aws_r() { aws --profile "${AWS_PROFILE}" --region "$r" "$@"; }

  n_az=$(aws_r ec2 describe-instance-type-offerings --location-type availability-zone \
    --filters "Name=instance-type,Values=${TYPE}" \
    --query 'length(InstanceTypeOfferings)' --output text 2>/dev/null || echo 0)

  quota=$(aws_r service-quotas get-service-quota --service-code ec2 \
    --quota-code L-DB2E81BA --query 'Quota.Value' --output text 2>/dev/null || echo "?")

  od_price="$(od_price_for "$r")"
  if [ -n "${od_price}" ]; then
    printf "=== %s   offering AZs=%s   quota=%s vCPUs   on-demand=\$%.4f/hr\n" \
      "$r" "${n_az}" "${quota}" "${od_price}"
  else
    echo "=== ${r}   offering AZs=${n_az}   quota=${quota} vCPUs   on-demand=?"
  fi

  if [ "${n_az}" = "0" ]; then
    echo "    NOT OFFERED in this region — skip it"
    echo
    continue
  fi

  start=$(date -u -v-3H +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -u -d '3 hours ago' +%Y-%m-%dT%H:%M:%S)
  rows=$(aws_r ec2 describe-spot-price-history --instance-types "${TYPE}" \
    --product-descriptions "Linux/UNIX" --start-time "${start}" \
    --query 'SpotPriceHistory[].[AvailabilityZone,SpotPrice]' --output text 2>/dev/null \
    | sort -u -k1,1)

  if [ -z "${rows}" ]; then
    echo "    no spot history — cannot gauge pressure"
  else
    while read -r az price; do
      [ -z "${az}" ] && continue
      if [ -n "${od_price}" ]; then
        pct=$(python3 -c "print(f'{($price/$od_price)*100:.0f}')")
        bar=$(python3 -c "n=int($pct/10); print('#'*n)")
        printf "    %-14s spot=\$%-10s %3s%% %s\n" "${az}" "${price}" "${pct}" "${bar}"
        if [ "${pct}" -lt "${best_score}" ]; then best_score="${pct}"; best_region="${r}"; fi
      else
        printf "    %-14s spot=\$%s\n" "${az}" "${price}"
      fi
    done <<< "${rows}"
  fi
  echo
done

echo "======================================================================"
if [ -n "${best_region}" ]; then
  echo " RECOMMENDATION: try ${best_region} first (least-contended AZ at ${best_score}% of on-demand)"
else
  echo " RECOMMENDATION: no signal available; just attempt a launch"
fi
cat <<EOF

To wait for capacity and grab it as soon as it frees up:

    QA_REGION=${best_region:-us-west-2} bash scripts/box_up.sh --apply --wait

That polls every 60s across all AZs and stops the moment one succeeds.
It creates NOTHING until a launch actually works, and prints what it made.

To see whether anything of ours is currently running (and billing):

    bash scripts/box_down.sh --status
EOF
