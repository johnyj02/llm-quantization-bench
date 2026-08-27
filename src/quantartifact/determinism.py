"""Determinism audit — task `A7`, hypothesis H8, and gate `G4`.

**Why this gates everything downstream.** The whole paper is about small accuracy
deltas, and vLLM says outright that it *"does not guarantee the reproducibility of
the results by default, for the sake of performance."* Issue #9567 reported
exactly our failure mode on an INT8-quantized Llama-3-8B and tied it to unstable
lm-eval scores. So uncontrolled batch nondeterminism could **manufacture or erase
our effect**, and there is no point spending compute on deltas we cannot trust.

**The specific gap this measures, which is the most important technical finding in
the plan.** `VLLM_BATCH_INVARIANT=1` overrides the relevant GEMM/softmax/reduction
ops with batch-invariant Triton kernels — but it is *not* validated against the
quantized kernels we depend on:

* ``auto_awq.py`` **has** a batch-invariance guard, with the source comment
  *"Marlin kernels are not batch invariant"* — so under BI mode AWQ silently falls
  back to slower non-Marlin kernels.
* ``auto_gptq.py`` and the ``choose_mp_linear_kernel()`` dispatcher GPTQ actually
  uses contain **no batch-invariance handling at all** — so GPTQ keeps using
  Marlin, which vLLM's own comment says is not batch-invariant.

That asymmetry is what this module measures, and it is why H8 exists. The
mechanism is in vLLM's **source**, so it is architecture-independent; only the
measured *rates* are specific to our L40S/SM89 box, and the paper says so.

**Two things this also settles, both pre-registered:**
1. Whether batch invariance works **at all** on SM89 — an `A7` finding, not a
   precondition. The 1.2–2× overhead figure in `EXECUTOR-TASKS.md` §3.3 is an
   Ampere number; Ada is below SM90, so vLLM must substitute Triton GEMMs there
   too, making it a reasonable prior rather than an inapplicable one.
2. The frozen ``max_num_seqs`` (planner ruling R8): the largest value that is
   bitwise-stable across 3 repeats for **every** format — a **minimum over
   formats**, because §3.3 requires one value identical across all arms.

The disagreement analysis is **pure and testable without a GPU**; only
``run_probe`` touches one.

⚠️ Nothing here runs without explicit user confirmation (D10).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DETERMINISM_VERSION = "1.0.0"

# R8: swept, not assumed. §3.3's value of 8 comes from vLLM #51187 measured on
# OTHER hardware, and it is our largest throughput cost since it deliberately
# starves GPU concurrency.
MAX_NUM_SEQS_SWEEP: tuple[int, ...] = (8, 16, 32, 64)

# ⚠️ SWEPT IN DESCENDING ORDER WITH EARLY EXIT, and the reason is cost.
#
# R8 called the sweep "near-zero marginal cost" because A7 already runs 3 repeats
# per format. That is true PER LEVEL but not across four levels: an exhaustive
# sweep is 5 formats x 4 levels x 3 repeats x 500 items = 30,000 generations,
# ~22% of the whole study, against the ~13,500 the §2.6 table budgeted for the
# determinism probe.
#
# Descending with early exit costs far less for the same answer. We want the
# LARGEST stable value, so testing 64 first and stopping on success gives the
# result in one level (7,500) rather than four. Worst case — nothing above the
# floor is stable — is still 4 levels, but that case is itself a headline H8
# finding, so the compute is well spent when it happens.
#
# 8 is never skipped: it is the pre-audit floor from §3.3, so it is measured even
# when a larger value passes, to keep the frontier table complete.
SWEEP_ORDER: tuple[int, ...] = tuple(sorted(MAX_NUM_SEQS_SWEEP, reverse=True))
SWEEP_FLOOR: int = min(MAX_NUM_SEQS_SWEEP)


def sweep_plan(formats: Sequence[str], n_probe: int, repeats: int) -> dict[str, Any]:
    """Cost of the descending sweep, best case vs worst case.

    Reported before the run so the launch request carries an honest range rather
    than a single optimistic or pessimistic number.
    """
    per_level = len(formats) * repeats * n_probe
    return {
        "order": list(SWEEP_ORDER),
        "floor_always_measured": SWEEP_FLOOR,
        "generations_per_level": per_level,
        # Best case is now THREE levels, not two: the stable one, the level below it
        # for the non-monotonicity check (R12.3.1), and the floor.
        "best_case_generations": per_level * 3,
        "worst_case_generations": per_level * len(SWEEP_ORDER),
        "exhaustive_would_cost": per_level * len(MAX_NUM_SEQS_SWEEP),
        # Charged separately because it happens once, at the selected value only.
        "confirmation_generations": len(formats) * CONFIRM_REPEATS * n_probe,
        "note": (
            "descending: stop at the first level stable for EVERY format, then also "
            "measure the level BELOW it (non-monotonicity, R12.3.1) and the floor"
        ),
    }


def next_sweep_level(
    tested: Sequence[DisagreementResult], formats: Sequence[str]
) -> int | None:
    """Which ``max_num_seqs`` to test next, or None when the sweep can stop.

    Stops as soon as a level is bitwise-stable for **every** format, since that is
    the largest such value by construction of the descending order. The floor is
    then measured separately if it has not been already.
    """
    formats = list(formats)
    by_level: dict[int, dict[str, bool]] = {}
    for result in tested:
        by_level.setdefault(result.max_num_seqs, {})[result.fmt] = result.bitwise_stable

    for level in SWEEP_ORDER:
        seen = by_level.get(level, {})
        if not all(fmt in seen for fmt in formats):
            return level  # incomplete at this level -> test it
        if all(seen[fmt] for fmt in formats):
            # Largest stable level found. Two more levels are still required.
            #
            # 1. THE LEVEL IMMEDIATELY BELOW IT (planner ruling R12.3.1). "Stable at
            #    64 but unstable at 32" is a non-monotonicity that would invalidate
            #    the whole frontier framing, and it is cheap to rule out. Without
            #    this the frontier is two points and the H8 contribution -- telling
            #    other people what they can safely get away with -- is thin.
            # 2. The floor, so the table is complete against §3.3's pre-audit value.
            index = SWEEP_ORDER.index(level)
            required = []
            if index + 1 < len(SWEEP_ORDER):
                required.append(SWEEP_ORDER[index + 1])
            if SWEEP_FLOOR not in required:
                required.append(SWEEP_FLOOR)

            for want in required:
                measured = by_level.get(want, {})
                if not all(fmt in measured for fmt in formats):
                    return want
            return None

    return None


def non_monotonic_levels(
    results: Sequence[DisagreementResult], formats: Sequence[str]
) -> list[dict[str, Any]]:
    """Levels where a SMALLER ``max_num_seqs`` was less stable than a larger one.

    The frontier framing assumes stability is monotone in batch size: fewer
    co-resident sequences cannot be worse. R12.3.1 requires ruling that out rather
    than assuming it, because a non-monotonicity would mean "the largest stable
    value" is not a well-defined thing to report.

    Returns one entry per violation, empty when the frontier is well behaved.
    """
    formats = list(formats)
    stable_at: dict[int, dict[str, bool]] = {}
    for result in results:
        # Only the plain probes describe the frontier; the Marlin-disabled fallback
        # is a different engine configuration and would corrupt the comparison.
        if result.disabled_kernels:
            continue
        stable_at.setdefault(result.max_num_seqs, {})[result.fmt] = result.bitwise_stable

    violations: list[dict[str, Any]] = []
    ascending = sorted(stable_at)
    for fmt in formats:
        for i, lower in enumerate(ascending):
            for higher in ascending[i + 1 :]:
                low = stable_at[lower].get(fmt)
                high = stable_at[higher].get(fmt)
                if low is False and high is True:
                    violations.append(
                        {
                            "format": fmt,
                            "unstable_at": lower,
                            "stable_at": higher,
                            "implication": (
                                "stability is NOT monotone in max_num_seqs for this "
                                "format, so 'the largest stable value' is not "
                                "well defined and the frontier must be reported as "
                                "a set rather than a threshold"
                            ),
                        }
                    )
    return violations

# Slower fallback kernels, to test whether determinism is RESTORED when Marlin is
# the culprit. Tried only if a format proves nondeterministic.
DISABLED_KERNELS_FALLBACK = "MarlinLinearKernel,MacheteLinearKernel"

REPEATS = 3

# Planner ruling R12.3.2: three repeats PICK the value, ten CONFIRM it, and
# everything downstream depends on this one number.
#
# ⚠️ This EXTENDS R8, which as recorded specifies 3 repeats and does not mention 10.
# Select-then-confirm is coherent rather than contradictory, so it is applied — but
# the difference is recorded in notes/planner-rulings.md rather than merged silently.
#
# ⚠️ The ten repeats CANNOT be stitched from the sweep's three plus seven more. What
# A7 measures is whether repeated identical requests to a *running* engine return
# identical text; combining repeats from two engine instances would measure
# cross-instance variation instead, which is a different quantity. So the
# confirmation is its own cell with its own engine, and the sweep's three repeats at
# that level are not reused.
CONFIRM_REPEATS = 10

# The A7 probe: 500 items spanning all three answer formats, so a
# format-dependent nondeterminism cannot hide.
PROBE_COMPOSITION: dict[str, int] = {
    "gsm8k_cot": 200,
    "bbh": 200,
    "arc_challenge_chat": 100,
}


@dataclasses.dataclass
class RepeatOutputs:
    """Generations from one (format, max_num_seqs, repeat) pass over the probe."""

    fmt: str
    max_num_seqs: int
    repeat_idx: int
    # item_id -> generated text. Bitwise comparison is on the text, since that is
    # what every downstream scorer consumes.
    outputs: dict[str, str]
    disabled_kernels: str | None = None


@dataclasses.dataclass
class DisagreementResult:
    """Bitwise disagreement for one (format, max_num_seqs) across repeats."""

    fmt: str
    max_num_seqs: int
    n_items: int
    n_disagreeing: int
    disagreement_rate: float
    max_distinct_outputs: int
    bitwise_stable: bool
    disabled_kernels: str | None = None
    first_divergence_examples: list[str] = dataclasses.field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        return {
            "format": self.fmt,
            "max_num_seqs": self.max_num_seqs,
            "n_items": self.n_items,
            "n_disagreeing": self.n_disagreeing,
            "disagreement_rate": round(self.disagreement_rate, 6),
            "max_distinct_outputs": self.max_distinct_outputs,
            "bitwise_stable": self.bitwise_stable,
            "disabled_kernels": self.disabled_kernels or "",
        }


def compute_disagreement(repeats: Sequence[RepeatOutputs]) -> DisagreementResult:
    """Bitwise disagreement rate across repeats of an identical probe.

    An item *disagrees* if the repeats did not all produce byte-identical text.
    ``max_distinct_outputs`` is the worst-case count for any single item, which is
    the shape vLLM #51187 reported (44 co-resident sequences → 3 distinct outputs
    in 5 repeats).

    Items missing from some repeat are counted as disagreeing rather than skipped:
    a dropped generation is a reproducibility failure too, and silently ignoring
    it would flatter the result.
    """
    if len(repeats) < 2:
        raise ValueError("need at least 2 repeats to measure disagreement")

    first = repeats[0]
    all_items: set[str] = set()
    for repeat in repeats:
        all_items.update(repeat.outputs)

    disagreeing: list[str] = []
    worst_distinct = 1

    for item_id in sorted(all_items):
        texts = [repeat.outputs.get(item_id) for repeat in repeats]
        distinct = {text for text in texts if text is not None}
        n_distinct = len(distinct) + (1 if any(t is None for t in texts) else 0)
        worst_distinct = max(worst_distinct, n_distinct)
        if n_distinct > 1:
            disagreeing.append(item_id)

    n_items = len(all_items)
    rate = len(disagreeing) / n_items if n_items else 0.0

    return DisagreementResult(
        fmt=first.fmt,
        max_num_seqs=first.max_num_seqs,
        n_items=n_items,
        n_disagreeing=len(disagreeing),
        disagreement_rate=rate,
        max_distinct_outputs=worst_distinct,
        bitwise_stable=(len(disagreeing) == 0),
        disabled_kernels=first.disabled_kernels,
        first_divergence_examples=disagreeing[:5],
    )


def select_max_num_seqs(results: Sequence[DisagreementResult]) -> dict[str, Any]:
    """R8: the largest value bitwise-stable for EVERY format — a MIN over formats.

    §3.3 requires one ``max_num_seqs`` identical across all arms, so a value that
    is stable for bf16 but not GPTQ is **not eligible**. Taking the max of
    per-format maxima would silently pick a value that is unstable for some arm.

    Returns the selection plus the full frontier, which the paper reports as free
    evidence for H8. If nothing above the floor is stable, **that is a finding,
    not a failed sweep.**
    """
    formats = sorted({result.fmt for result in results})
    per_format_max: dict[str, int | None] = {}

    for fmt in formats:
        stable = [
            result.max_num_seqs
            for result in results
            if result.fmt == fmt and result.bitwise_stable
        ]
        per_format_max[fmt] = max(stable) if stable else None

    # A format with NO stable value makes the whole sweep inconclusive for the
    # shared setting; surface it rather than dropping it from the min.
    unstable_formats = [fmt for fmt, value in per_format_max.items() if value is None]

    eligible = [value for value in per_format_max.values() if value is not None]
    selected = min(eligible) if eligible and not unstable_formats else None

    return {
        "selected_max_num_seqs": selected,
        "per_format_max_stable": per_format_max,
        "formats_with_no_stable_value": unstable_formats,
        "swept": list(MAX_NUM_SEQS_SWEEP),
        "rule": "min over formats of the largest bitwise-stable value",
        "finding_if_none": (
            "no max_num_seqs was bitwise-stable for every format; this is a "
            "reportable H8 result, not a failed sweep"
            if selected is None
            else None
        ),
    }


def differential_noise_floor(results: Sequence[DisagreementResult]) -> dict[str, Any]:
    """The floor every precision contrast must clear.

    If a quantized format is bitwise nondeterministic while bf16 is not, then
    **differential numerical noise exists between arms**, and that per-format
    disagreement rate is a floor on every contrast involving that format
    (`analysis/preregistration.md` §9). This is the quantity that makes H8
    consequential rather than merely curious.
    """
    by_format: dict[str, float] = {}
    for fmt in sorted({result.fmt for result in results}):
        rates = [
            result.disagreement_rate for result in results if result.fmt == fmt
        ]
        by_format[fmt] = max(rates) if rates else 0.0

    bf16_rate = by_format.get("bf16", 0.0)
    differential = {
        fmt: round(rate - bf16_rate, 6)
        for fmt, rate in by_format.items()
        if fmt != "bf16"
    }

    return {
        "per_format_worst_disagreement": {k: round(v, 6) for k, v in by_format.items()},
        "bf16_disagreement": round(bf16_rate, 6),
        "differential_vs_bf16": differential,
        "differential_noise_exists": any(value > 0 for value in differential.values()),
        "implication": (
            "any precision contrast involving a format with a positive "
            "differential cannot claim an effect smaller than that differential"
        ),
    }


def probe_fingerprint(outputs: dict[str, str]) -> str:
    """Stable hash over a whole probe pass, for a quick eyeball comparison."""
    rolling = hashlib.sha256()
    for item_id in sorted(outputs):
        rolling.update(item_id.encode("utf-8"))
        rolling.update(outputs[item_id].encode("utf-8"))
    return rolling.hexdigest()[:16]


def summarise(results: Sequence[DisagreementResult]) -> dict[str, Any]:
    """The G4 evidence object."""
    return {
        "gate": "G4",
        "determinism_version": DETERMINISM_VERSION,
        "repeats": REPEATS,
        "probe_composition": PROBE_COMPOSITION,
        "n_cells": len(results),
        "rows": [result.to_row() for result in results],
        "max_num_seqs_selection": select_max_num_seqs(results),
        "differential_noise_floor": differential_noise_floor(results),
        "formats_tested": sorted({result.fmt for result in results}),
        "any_nondeterminism": any(not r.bitwise_stable for r in results),
        # R12.3.1: the frontier framing assumes stability is monotone in batch size.
        # Empty here means it is; non-empty means "the largest stable value" is not
        # a well-defined threshold and must be reported as a set.
        "non_monotonic_violations": non_monotonic_levels(
            results, sorted({r.fmt for r in results})
        ),
    }


def write_json(path: Path, results: Sequence[DisagreementResult]) -> None:
    """Write the G4 evidence object.

    Paired with the CSV: the CSV is the table for the paper, the JSON carries the
    selection rule, the differential floor and the frontier — the parts a reviewer
    needs to check the reasoning rather than just read the numbers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summarise(results), indent=2))


def write_csv(path: Path, results: Sequence[DisagreementResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.to_row() for result in results]
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distinct_output_histogram(repeats: Sequence[RepeatOutputs]) -> dict[int, int]:
    """How many items had 1, 2, 3… distinct outputs. The #51187 shape."""
    all_items: set[str] = set()
    for repeat in repeats:
        all_items.update(repeat.outputs)

    counter: Counter[int] = Counter()
    for item_id in all_items:
        texts = [r.outputs.get(item_id) for r in repeats]
        distinct = {t for t in texts if t is not None}
        n = len(distinct) + (1 if any(t is None for t in texts) else 0)
        counter[n] += 1
    return dict(sorted(counter.items()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qa determinism",
        description="A7 determinism audit: per-format bitwise disagreement + max_num_seqs sweep",
    )
    parser.add_argument("--format", action="append", default=None, help="repeatable")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument(
        "--sweep-max-num-seqs",
        action="store_true",
        help="sweep max_num_seqs over {8,16,32,64} (planner ruling R8)",
    )
    parser.add_argument("--out-csv", type=Path, default=Path("artifacts/tables/determinism.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("artifacts/tables/determinism.json"))
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print the audit plan and exit. Zero cost, starts nothing.",
    )
    parser.add_argument(
        "--i-have-confirmation",
        action="store_true",
        help="the user has explicitly confirmed THIS run (decision D10). Required.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    formats = args.format or ["bf16", "gptq-4bit", "awq-4bit", "gptq-8bit", "bnb-nf4"]
    n_probe = sum(PROBE_COMPOSITION.values())

    print("A7 determinism audit plan")
    print(f"  formats       : {formats}")
    print(f"  repeats       : {args.repeats}")
    print(f"  probe         : {n_probe} items {PROBE_COMPOSITION}")

    if args.sweep_max_num_seqs:
        plan = sweep_plan(formats, n_probe, args.repeats)
        print(f"  max_num_seqs  : {plan['order']} DESCENDING, early exit "
              f"(floor {plan['floor_always_measured']} always measured)")
        print(f"  generations   : {plan['best_case_generations']:,} best case -> "
              f"{plan['worst_case_generations']:,} worst case")
        print(f"                  (exhaustive would be {plan['exhaustive_would_cost']:,})")
    else:
        total = len(formats) * args.repeats * n_probe
        print(f"  max_num_seqs  : [{SWEEP_FLOOR}] (no sweep)")
        print(f"  generations   : {total:,}")
    print()
    print("  measures      : per-format bitwise disagreement over repeats")
    print("                  differential noise floor vs bf16")
    print("                  largest max_num_seqs stable for EVERY format (min over formats)")
    print(f"  fallback      : VLLM_DISABLED_KERNELS={DISABLED_KERNELS_FALLBACK}")
    print("                  tried only where a format proves nondeterministic")

    if args.plan_only:
        print("\n--plan-only: nothing was run.")
        return 0

    if not args.i_have_confirmation:
        sys.stderr.write(
            "\nREFUSING TO RUN.\n\n"
            "Decision D10 (notes/decisions.md): no GPU job starts without explicit\n"
            "user confirmation. Use --plan-only to inspect the plan at zero cost.\n"
        )
        return 2

    sys.stderr.write(
        "run_probe needs the GPU box and the quantized weights from `qa quantize`;\n"
        "see notes/gates.md G4. The analysis functions in this module are complete\n"
        "and unit-tested.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
