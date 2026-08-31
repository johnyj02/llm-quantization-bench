#!/usr/bin/env python3
"""SageMaker entry point — `reference` stage. THE MAIN RUN. `run_id=main-v1`.

Every hypothesis except H8 is estimated from this run's generations. It is the reference
condition of `EXECUTOR-TASKS.md` §2.2: the **full precision ladder** on the **full item
counts**, at the `1x` token budget, with the `chat` template level (prereg §3b-bis's primary).

    3 models x 5 arms = 15 cells
    per cell: gsm8k 1319 + gsm8k_cot 1319 + BBH 1100 + arc_challenge_chat 500 = 4,238 items
    total: 63,570 generations, ~10.5 h projected (scripts/project_budget.py)

🚨 **THIS IS THE FREEZE LINE.** `analysis/preregistration.md` may be edited freely until the
first `main-v1` generation exists, because until then no edit can have been influenced by
confirmatory results. **From this job onward it is append-only, under a dated `## Amendments`
heading.** The date on the calendar was never the binding constraint; this job is.

🚨 **`max_num_seqs` is READ from the frozen A7 result, never assumed** — same guard as the
pilot, and it matters more here: these generations are the paper's primary evidence, so a value
that later changed would make them non-comparable to everything else with nothing downstream
to reveal it.

**Why the full 1,319-item GSM8K test set** (override O8): H7 makes a point about statistical
resolution, so running the full test set while auditing others' underpowered evaluations is
both stronger and better rhetoric. The 500-item sweep subsample is a **prefix** of this
ordering, which makes the reference run serve as the sweep's `1x` level for free.

⚠️ **The log-likelihood arms are NOT in this stage.** `run_cell` is generation-only
(`scoring_mode="generate"` is fixed and `option_logprobs` is always `None`), so H6's arm 2
(`arc_challenge_ll_matched`) and arm 3 (`arc_challenge`) need a separate implementation, which
also carries G3's outstanding ARC cross-validation. Flagged rather than silently omitted: the
H6 mode contrast is incomplete until it exists.

⚠️ Cells run through `sm_pilot.CELL_SCRIPT`, imported rather than copied, so the reference run
cannot drift from the pilot in prompt rendering, token-id handling or engine configuration --
which is what makes the pilot a usable rehearsal for it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import yaml  # noqa: E402
from sm_pilot import CELL_SCRIPT, frozen_max_num_seqs  # noqa: E402

from quantartifact.quantize import config_hash, load_specs, output_dir  # noqa: E402

# 🚨 K=2 REPETITION — prereg §9f, planner ruling R23.3. Ten cells run TWICE.
#
# WHY. `pilot-v1` and `pilot-v2` ran the same items in the same order with a byte-identical
# recorded configuration and diverged on 41-45% of items for Llama, **including bf16**
# (`notes/cross-invocation.md`). The mechanism is undiagnosed and four hypotheses are refuted.
# Because each arm is different weights, each arm needs its own engine load, so this variance
# is **inherently between-arm** and cannot be made common the way item ordering could. §9f
# therefore registers an ESTIMATOR rather than a number, and the estimator is fitted from
# these cells. Without them there is no invocation floor and no class-3 claim is admissible.
#
# WHY THESE TEN, and not the cells where the effect was already observed. Coverage follows the
# **estimand**, not the empirical screen (R23.3):
#   - At the pilot's n=180 one item is 0.56 pp, so qwen's observed "0.00%" means *fewer than
#     one item flipped*. That is not zero with confidence.
#   - Stratifying on an empirical screen biases the pooled floor UPWARD, by measuring only
#     where the effect was already seen, and leaves the signal-carrying cells unmeasured.
#   - Qwen carries the largest rung gaps (19-23 pp) and therefore the most H3 signal, making
#     it the worst possible place for an unmeasured floor.
# `llama-3.1-8b|gptq-8bit` is the eleventh-hour addition: it is the WORST observed interaction
# movement (2.78 pp), and leaving the observed worst case without a floor estimate is exactly
# what a reviewer picks first.
#
# ⚠️ Both invocations run INSIDE THIS JOB, contemporaneously. Days-apart reruns would add
# driver and library drift as a confound to the very quantity being measured.
#
# The remaining dose-response arms (bnb-NF4, the other gptq-8bit cells) report against the
# POOLED floor, stated as an assumption in the paper.
K2_CELLS: frozenset[str] = frozenset({
    f"{model}|{arm}"
    for model in ("qwen2.5-3b", "phi-3.5-mini", "llama-3.1-8b")
    for arm in ("bf16", "gptq-4bit", "awq-4bit")
} | {"llama-3.1-8b|gptq-8bit"})

CHECKPOINT_DIR = Path(os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints"))
OUTPUT_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
CONFIGS = Path(__file__).parent / "configs"

SMOKE = os.environ.get("SM_HP_SMOKE") == "1" or "--smoke" in sys.argv

# 🚨 `main-v1` is the name the paper cites and the freeze line. The smoke gets its own id for
# the reason established at `pilot-smoke-r2`: a rehearsal sharing a run_id pools 20-item junk
# into the population every downstream estimate is drawn from, and nothing reveals it.
RUN_ID = "main-v1-smoke" if SMOKE else "main-v1"
GENERATIONS_ROOT = CHECKPOINT_DIR / "generations"

# The FULL precision ladder. Five arms, not three: `gptq-8bit` and `bnb-nf4` are here for
# dose-response (bf16 -> 8-bit -> 4-bit) and method generality, which is a far stronger
# internal-validity argument than a two-point comparison (override O7). They are excluded from
# the expensive sweep arms, not from this one.
PRECISIONS = ("bf16", "gptq-8bit", "gptq-4bit", "awq-4bit", "bnb-nf4")

# The chat level is primary (prereg §3b-bis). The `raw` level is the template stage's job.
TEMPLATE_LEVEL = "chat"


def reference_plan(tasks: dict) -> list[tuple[str, str, str | None, int]]:
    """`(lm_eval_task, task, subtask, n_items)` for one cell, at FULL counts.

    Counts come from `configs/tasks.yaml` rather than being written here, so the
    pre-registered numbers and the numbers that run cannot diverge -- the failure this
    project has hit seven times.
    """
    plan: list[tuple[str, str, str | None, int]] = []
    for key in ("gsm8k", "gsm8k_cot"):
        config = tasks[key]
        plan.append((config["lm_eval_task"], key, None, config["n_items_reference"]))

    arc = tasks["arc_challenge_chat"]
    plan.append((arc["lm_eval_task"], "arc_challenge_chat", None, arc["n_items"]))

    # BBH per-subtask, NOT a flat default: R1.3 raised the two single-subtask contrast levels
    # of the format gradient to 250 because at 150 they could not detect anything under
    # ~7.2 pp, leaving two of three gradient levels unpowered.
    bbh = tasks["bbh"]
    prefix = bbh["lm_eval_task_prefix"]
    for subtask in bbh["subtasks"]:
        n = subtask.get("n_items", bbh["n_items_default"])
        plan.append((f"{prefix}{subtask['name']}", "bbh", subtask["name"], n))
    return plan


def build_cells(models: dict, specs: dict, calibration) -> list[dict]:
    """Every (model x arm) cell, weights resolved BEFORE any engine loads.

    Fail-fast is the point: a missing checkpoint must cost seconds, not surface at hour six of
    a ten-hour run. Job 1 established this the hard way.
    """
    cells: list[dict] = []
    for model_key, model in models.items():
        for precision in PRECISIONS:
            if precision == "bf16":
                # `bf16` is absent from specs (load_specs skips method=="none").
                path, digest, method, bits = model["repo"], None, None, 16
            else:
                spec = specs[precision]
                method, bits = spec.method, spec.bits
                if spec.in_flight:
                    # bnb-NF4 has no checkpoint: vLLM quantizes it at load, so the BASE repo
                    # is served with quant_method set.
                    path, digest = model["repo"], None
                else:
                    digest = config_hash(spec, calibration, model["revision"])
                    resolved = output_dir(CHECKPOINT_DIR, model_key, spec, digest)
                    if not (resolved / "config.json").exists():
                        raise SystemExit(
                            f"REFUSING TO RUN: no checkpoint for {model_key}/{precision} at "
                            f"{resolved} (recipe hash {digest}). Job 1 must have completed and "
                            f"this job's checkpoint_s3_uri must match job 1's."
                        )
                    path = str(resolved)
            cells.append({
                "cell_id": f"{model_key}|{precision}",
                "template_level": TEMPLATE_LEVEL,
                "model_key": model_key,
                "revision": model["revision"],
                "precision": precision,
                "quant_method": method,
                "bits": bits,
                "model_path": path,
                "quant_config_hash": digest,
            })
    return cells


def run_one_cell(cell: dict, plan: list, max_num_seqs: int, base_repo: str,
                 repeat_idx: int = 0) -> dict | None:
    """One cell in its own process, via the PILOT's cell script.

    Subprocess per cell because torch's caching allocator and the engine's hooks do not
    reliably release GPU memory in-process; 15 sequential engine loads in one process is the
    shape that cost job 1 four arms.
    """
    # ⚠️ The repeat index is in the scratch filenames too. Without it the second invocation
    # would overwrite the first's counts file and the two would be indistinguishable in the
    # report -- the same cache-identity mistake this project has made nine times, and the
    # K=2 measurement is precisely a measurement OF two runs that must not be conflated.
    tag = cell["cell_id"].replace("|", "__") + (f"__k{repeat_idx}" if repeat_idx else "")
    out_path = GENERATIONS_ROOT / f".{tag}.counts.json"
    payload = {
        "template_level": cell["template_level"],
        "cell": {**cell, "base_repo": base_repo},
        "plan": plan,
        "max_num_seqs": max_num_seqs,
        "run_id": RUN_ID,
        "repeat_idx": repeat_idx,
        "generations_root": str(GENERATIONS_ROOT),
        "tasks_yaml": str(CONFIGS / "tasks.yaml"),
        "models_yaml": str(CONFIGS / "models.yaml"),
        "out_path": str(out_path),
    }
    payload_path = GENERATIONS_ROOT / f".{tag}.payload.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(payload))

    # 8 h per cell: the largest is Llama-8B over 4,238 items. Generous because a cap costs
    # nothing unless it fires, and per-item resume means a kill costs only the current cell.
    result = subprocess.run(
        [sys.executable, "-c", CELL_SCRIPT, str(payload_path)], timeout=28800, check=False,
    )
    counts = json.loads(out_path.read_text()) if out_path.exists() else None
    out_path.unlink(missing_ok=True)
    payload_path.unlink(missing_ok=True)
    if result.returncode != 0 and counts is None:
        return None
    return counts


def main() -> int:
    models = yaml.safe_load((CONFIGS / "models.yaml").read_text())["models"]
    tasks = yaml.safe_load((CONFIGS / "tasks.yaml").read_text())["tasks"]
    specs, calibration = load_specs(CONFIGS / "quant.yaml")

    max_num_seqs = frozen_max_num_seqs()
    plan = reference_plan(tasks)

    if SMOKE:
        # Cover every DISTINCT task path and both tokenizer extremes, at 4 items each. The
        # arms are already proven by the pilot; what a reference-run rehearsal must prove is
        # that the FULL-COUNT plan resolves and that nothing in it explodes.
        models = {k: models[k] for k in ("qwen2.5-3b", "llama-3.1-8b") if k in models}
        plan = [(lm, t, s, 4) for lm, t, s, _ in plan]
        print(f"SMOKE MODE: {list(models)}, {len(plan)} task paths x 4 items")

    cells = build_cells(models, specs, calibration)
    if SMOKE:
        cells = [c for c in cells if c["precision"] in ("bf16", "gptq-4bit")]

    # 🚨 The run list, not the cell list: K=2 cells appear TWICE, as (cell, repeat_idx).
    #
    # The second pass runs AFTER every first pass rather than back-to-back with it. Two
    # reasons, and the first is the load-bearing one:
    #   1. If the job is killed part-way, a prefix of the run list is a COMPLETE reference
    #      condition plus some of the K=2 cells -- which is a usable study. Interleaving
    #      would leave some arms measured twice and others not at all, and the reference
    #      condition itself incomplete.
    #   2. It puts a maximum of other engine loads between the two invocations of a cell,
    #      so if the undiagnosed mechanism is sensitive to what ran before it, this is the
    #      arrangement most likely to EXPOSE that rather than hide it.
    k2 = [c for c in cells if c["cell_id"] in K2_CELLS] if not SMOKE else []
    run_list: list[tuple[dict, int]] = [(c, 0) for c in cells] + [(c, 1) for c in k2]

    per_cell = sum(n for *_, n in plan)
    print(f"run_id       : {RUN_ID}")
    print(f"max_num_seqs : {max_num_seqs}  (frozen by A7, not assumed)")
    print(f"template     : {TEMPLATE_LEVEL}  (prereg §3b-bis primary level)")
    print(f"arms         : {list(PRECISIONS)}")
    print(f"cells        : {len(cells)}   items/cell: {per_cell:,}   "
          f"total: {len(cells) * per_cell:,} generations")
    if k2:
        missing = sorted(K2_CELLS - {c["cell_id"] for c in cells})
        print(f"K=2 cells    : {len(k2)} of {len(K2_CELLS)} registered "
              f"(prereg §9f) -> +{len(k2) * per_cell:,} generations")
        for cell in k2:
            print(f"  {cell['cell_id']:40} repeat_idx 0 and 1, separate engine loads")
        if missing:
            # Fails LOUD: a registered K=2 cell that does not resolve means the floor is
            # estimated from fewer cells than the prereg says, which is a silent weakening
            # of a registered analysis.
            raise SystemExit(
                f"REFUSING TO RUN: {len(missing)} cell(s) registered for K=2 in prereg §9f "
                f"do not exist in this run: {missing}. The invocation floor would be "
                f"estimated from a different set than the one registered."
            )
    print(f"run units    : {len(run_list)} (cells + K=2 second invocations)")
    for lm_eval_task, task, subtask, n in plan:
        print(f"  {task + ('/' + subtask if subtask else ''):40} {n:>5}  ({lm_eval_task})")
    if not SMOKE:
        print("\n🚨 THIS IS main-v1. The pre-registration is APPEND-ONLY from here.")
    print()

    report: dict = {
        "stage": "reference", "run_id": RUN_ID, "smoke": SMOKE,
        "max_num_seqs": max_num_seqs, "template_level": TEMPLATE_LEVEL,
        "arms": list(PRECISIONS), "items_per_cell": per_cell,
        "k2_cells": sorted(c["cell_id"] for c in k2),
        "plan": [{"task": t, "subtask": s, "n": n} for _, t, s, n in plan],
        "cells": [],
    }
    failed: list[str] = []
    for index, (cell, repeat_idx) in enumerate(run_list, 1):
        label = cell["cell_id"] + (f"  [K=2 repeat {repeat_idx}]" if repeat_idx else "")
        print(f"=== unit {index}/{len(run_list)}: {label} ===", flush=True)
        started = time.monotonic()
        counts = run_one_cell(cell, plan, max_num_seqs, models[cell["model_key"]]["repo"],
                              repeat_idx=repeat_idx)
        elapsed = time.monotonic() - started
        if counts is None:
            failed.append(label)
            report["cells"].append({"cell_id": cell["cell_id"], "repeat_idx": repeat_idx,
                                    "error": "cell subprocess failed"})
            print(f"    FAILED after {elapsed/60:.1f} min", flush=True)
            continue
        report["cells"].append({
            "cell_id": cell["cell_id"], "precision": cell["precision"],
            "model_key": cell["model_key"], "repeat_idx": repeat_idx,
            "minutes": round(elapsed / 60, 2), **counts,
        })
        print(f"       {counts['written']} written, {counts['skipped']} skipped, "
              f"{elapsed/60:.1f} min", flush=True)
        # Written incrementally: a timeout at cell 12 of 15 must still leave a readable
        # record of what completed, since output_path only uploads when the job ENDS.
        (OUTPUT_DIR / "reference_report.json").write_text(json.dumps(report, indent=2))

    done = [c for c in report["cells"] if "error" not in c]
    written = sum(c["written"] for c in done)
    print("=" * 70)
    print(f"cells        : {len(done)}/{len(cells)}")
    print(f"generations  : {written:,} written")
    print(f"report -> {OUTPUT_DIR / 'reference_report.json'}")

    if failed:
        # Every cell is load-bearing: the precision ladder is a dose-response argument, so a
        # missing arm breaks the monotonicity claim rather than merely reducing coverage.
        print(f"\nFAILED CELLS (all load-bearing — the ladder is a dose-response): {failed}")
        print("Resubmitting resumes per item; completed cells are skipped by run_id.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
