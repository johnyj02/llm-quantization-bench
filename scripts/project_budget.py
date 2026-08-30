#!/usr/bin/env python3
"""Project the remaining GPU budget from MEASURED rates. The A6 report-back (§2.6).

    .venv/bin/python scripts/project_budget.py

WHY THIS IS A SCRIPT. `EXECUTOR-TASKS.md` §2.6 requires a tight estimate once
generations/hour is measured on the actual instance, and makes it a report-back if the
reference-run block moves past Sep 13. The original table is in **A100 @ $1.19/h**; we
run **ml.g6e.2xlarge (L40S) @ $2.80/h**, 2.35x the hourly rate, so the dollar column was
never transferable even before the rate changed.

⚠️ **Every earlier estimate in this project used a throughput number measured while BBH's
terminators were broken** and 5 of 6 subtasks emitted 1-7 tokens instead of 294-618. Those
rates (24,800 gen/h from job 2, 7,400 from job 3) are not just imprecise, they are
measurements of a different workload. Superseded here by `pilot-v2`, the first run with
fixed terminators at scale.

🚨 **TWO COST DRIVERS, NOT ONE.** In `pilot-v2` generation was only **25%** of billable
time: 15.1 min of generation against 44.7 min of engine loads and model downloads across 17
cells. So a plan's **cell count** matters as much as its generation count, and any estimate
that multiplies generations by a rate will understate a many-celled phase badly. Both terms
are carried separately below.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# ---- MEASURED on pilot-v2 (job johnyj02-QA-v0-1-20260830-...-pilot, 3589 s billable) ----
# Per-task-group generation rates, computed as items / generation-minutes summed over cells.
RATE_BBH = 7_099        # gen/h, fixed terminators, 1024 cap, median 294-618 tokens emitted
RATE_GSM8K = 5_250      # gen/h, 256 cap. gsm8k_cot assumed equal: same cap, 8 shots vs 5.
RATE_ARC = 200_000      # gen/h. Effectively free -- `until` includes "." so replies are ~10
                        # tokens. Measured "0.0 min" per 80 items; floored to stay honest.
OVERHEAD_MIN_PER_CELL = 2.6   # engine load + model download, amortised over 17 cells
RATE_USD_PER_HOUR = 2.80      # USE1-Train:ml.g6e.2xlarge

SPENT_USD = 29.77       # 22 jobs through pilot-v2; regenerate with scripts/logrun.sh


def hours(bbh: int = 0, gsm8k: int = 0, arc: int = 0, cells: int = 0) -> float:
    gen = bbh / RATE_BBH + gsm8k / RATE_GSM8K + arc / RATE_ARC
    return gen + cells * OVERHEAD_MIN_PER_CELL / 60


# ---- Remaining phases, from configs/tasks.yaml + EXECUTOR-TASKS.md §2.2 + prereg -------
# A "cell" is one engine load: (model x arm) for a given condition.
#
# Reference condition: 3 models x 5 arms = 15 cells. Per cell:
#   gsm8k 1319 (full test, O8) + gsm8k_cot 1319 + BBH 1100 (R1.3 counts) + arc_chat 500
REFERENCE = dict(cells=15, gsm8k=15 * (1319 + 1319), bbh=15 * 1100, arc=15 * 500)

# Token sweep: 3 models x 3 contrasted arms = 9 cells, at 2x and 4x only (1x is nested in
# the reference run, O8). gsm8k 500 nested + BBH 1100 + arc 500, x2 budget levels.
SWEEP = dict(cells=9 * 2, gsm8k=9 * 2 * 500, bbh=9 * 2 * 1100, arc=9 * 2 * 500)

# Zero-shot arm (H5): 3 models x 3 arms, gsm8k_cot 500 + BBH 1100.
ZEROSHOT = dict(cells=9, gsm8k=9 * 500, bbh=9 * 1100)

# Synthetic hash-marker arm: 3 models x 3 arms x gsm8k_cot_hashmarker 500.
SYNTHETIC = dict(cells=9, gsm8k=9 * 500)

# Template probe (prereg 3b-bis), FULL scope -- R20.1 selects by measured BBH rate.
TEMPLATE_FULL = dict(cells=9 * 2, gsm8k=9 * 2 * 500, bbh=9 * 2 * 1500)
TEMPLATE_DEFAULT = dict(cells=9 * 2, gsm8k=9 * 2 * 500, bbh=9 * 2 * 500)
TEMPLATE_FLOOR = dict(cells=6 * 2, gsm8k=6 * 2 * 500, bbh=6 * 2 * 500)

# Small approved jobs.
JOB_1B = dict(cells=1)                                   # AWQ re-quantize, ~30 min, quant image
TEN_ORDERING = dict(cells=4, gsm8k=4 * 200 * 7, bbh=4 * 200 * 7, arc=4 * 100 * 7)


def show(name: str, spec: dict) -> float:
    h = hours(**spec)
    gens = spec.get("bbh", 0) + spec.get("gsm8k", 0) + spec.get("arc", 0)
    print(f"  {name:26} {spec.get('cells',0):>4} cells {gens:>8,} gens "
          f"{h:>6.1f} h  ${h * RATE_USD_PER_HOUR:>7.2f}")
    return h


def main() -> int:
    print("MEASURED (pilot-v2, first run with fixed BBH terminators):")
    print(f"  BBH        {RATE_BBH:>7,} gen/h     gsm8k {RATE_GSM8K:>7,} gen/h"
          f"     arc ~{RATE_ARC:,} gen/h")
    print(f"  overhead   {OVERHEAD_MIN_PER_CELL} min/cell  <- was 75% of pilot-v2's billable time")
    print()

    print("REMAINING PHASES:")
    total = 0.0
    for name, spec in [
        ("reference runs", REFERENCE),
        ("token sweep (2x,4x)", SWEEP),
        ("zero-shot (H5)", ZEROSHOT),
        ("synthetic hash arm", SYNTHETIC),
        ("job 1b (AWQ requant)", JOB_1B),
        ("10-ordering (H8)", TEN_ORDERING),
    ]:
        total += show(name, spec)
    print(f"  {'SUBTOTAL':26} {'':>4}       {'':>8}  {total:>6.1f} h  "
          f"${total * RATE_USD_PER_HOUR:>7.2f}")
    print()

    print("TEMPLATE PROBE — R20.1 selects on the measured BBH rate:")
    for name, spec in [("full (R>=6,000)", TEMPLATE_FULL),
                       ("default (1.5-6k)", TEMPLATE_DEFAULT),
                       ("floor (<1,500)", TEMPLATE_FLOOR)]:
        show(name, spec)
    selected = (TEMPLATE_FULL if RATE_BBH >= 6000
                else TEMPLATE_DEFAULT if RATE_BBH >= 1500 else TEMPLATE_FLOOR)
    probe_h = hours(**selected)
    print(f"  -> measured BBH {RATE_BBH:,} gen/h selects "
          f"{'FULL' if selected is TEMPLATE_FULL else 'DEFAULT' if selected is TEMPLATE_DEFAULT else 'FLOOR'}")
    print()

    grand = total + probe_h
    print(f"{'TOTAL REMAINING':28} {grand:>6.1f} h  ${grand * RATE_USD_PER_HOUR:>7.2f}")
    print(f"{'spent so far':28} {'':>6}   ${SPENT_USD:>7.2f}")
    print(f"{'ALL-IN PROJECTION':28} {'':>6}   ${grand * RATE_USD_PER_HOUR + SPENT_USD:>7.2f}")
    print(f"{'§2.6 all-in budget':28} {'':>6}    $370-440")
    print()
    # R20.1's rule: the probe takes <=15% of REMAINING budget.
    share = probe_h / grand
    print(f"probe share of remaining: {share:.1%}  "
          f"({'OK, <=15%' if share <= 0.15 else 'EXCEEDS 15% -> step down one row'})")
    # §2.6's schedule trigger.
    ref_h = hours(**REFERENCE)
    print(f"reference-run block: {ref_h:.1f} h GPU. §2.6 requires a report-back if this "
          f"pushes it past Sep 13.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
