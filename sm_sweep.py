#!/usr/bin/env python3
"""SageMaker entry point — `sweep` stage. **H1's registered instrument** (task A15).

🚨 **WHY THIS FILE EXISTS, AND WHY ITS ABSENCE WAS THE WORST GAP IN THE PROJECT.**

H1 is marked *"Lead. Never cut."* in `EXECUTOR-TASKS.md` §2.1, and until now its instrument
did not exist. The three token-budget levels were frozen as absolute integers in
`configs/tasks.yaml` (GSM8K 256/512/1024, BBH 1024/2048/4096, ARC-chat 100/200/400) and
`generate.py` accepted `--budget {1x,2x,4x}` — but **nothing drove them**. A repo-wide search
found `"2x"` and `"4x"` only inside that argparse choice list. Declared and never consumed,
which is this project's recurring shape, here applied to the lead hypothesis.

R41.1 named it directly: two never-cut hypotheses whose registered instruments do not exist is
*"the single most attackable fact about the current claim set."*

**What this measures, and why it is not the length finding.** `main-v1` already measures, at
full n, that arms differ by an order of magnitude in how many generated tokens their scored
answer discards (Llama `bnb-nf4` 63.9 per generation against `gptq-4bit` 5.9; Phi `gptq-4bit`
107.0; Qwen 1.2–5.4). That is **observational**. R39.3: *"the budget sweep is INTERVENTIONAL,
and interventional evidence is what licenses H1's causal claim."* We set the cap and observe
the response; the length finding stays alongside as the mechanism rather than as a substitute.

🚨 **THE DIRECTIONAL PREDICTION IS REGISTERED BEFORE THIS RUNS** (prereg §3, R39.3), which is
what converts a measurement into a test: **arms with a larger discard should show a larger
budget response.** Stated with the discard figures inline so it cannot be reinterpreted after
the fact.

**Scope, per §2.2's locked matrix.** 3 arms rather than 5 — `bf16`, `gptq-4bit`, `awq-4bit` —
because `bnb-NF4` and `gptq-8bit` are excluded from the expensive arms by design, and 2 levels
rather than 3 because **`1x` comes free from `main-v1` via the nested prefix subsample**. So
3 models × 3 arms × {2x, 4x} = 18 units over the 1,400-item sweep composition.

⚠️ **`1x` is NOT regenerated, and that is the point of the nesting.** GSM8K's 500-item sweep
sample is a verified prefix-nested subset of the reference 1,319, so the reference run already
holds the `1x` level for exactly these items. Regenerating it would cost a third more compute
and, worse, would produce a *second* `1x` measurement that cross-job noise guarantees will not
match the first (§9f), leaving two numbers where the design wants one.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import yaml  # noqa: E402
from sm_pilot import frozen_max_num_seqs  # noqa: E402
from sm_reference import build_cells, run_one_cell  # noqa: E402

from quantartifact.quantize import load_specs  # noqa: E402

CHECKPOINT_DIR = Path(os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints"))
OUTPUT_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
CONFIGS = Path(__file__).parent / "configs"

SMOKE = os.environ.get("SM_HP_SMOKE") == "1" or "--smoke" in sys.argv

#: ⚠️ Its OWN run_id, never `main-v1`. A sweep cell and a reference cell differ in
#: `budget_level` and `max_gen_toks`, so writing them under one id would pool two budget
#: levels into one population that every downstream `--run-id` consumer reads as homogeneous.
#:
#: ⚠️ CORRECTED 2026-09-05. This used to claim both fields are "in `decoding_fingerprint`" and
#: that writing them under one id "would either be refused by `assert_cell_reusable`" or pool
#: them. **The first clause was false and the second overstated the protection.** Neither field
#: is in the fingerprint payload, and the fingerprint does not guard this stage at all: it is
#: called from `sm_determinism.py` only. What actually separates a sweep cell from a reference
#: cell is `budget_level` in `cell_path`, plus `records.GenerationRecord.validate()` refusing a
#: label that disagrees with the config's cap. Recorded because someone trusting the old comment
#: could have removed `budget_level` from the path believing the fingerprint covered it.
RUN_ID = "sweep-v1-smoke" if SMOKE else "sweep-v1"

#: The three contrasted arms. `bnb-nf4` and `gptq-8bit` are deliberately absent: §2.2 excludes
#: them from the expensive arms, and the paper must say why rather than leave it inferred.
#: bnb-NF4 has no CUDA kernels in vLLM (its own RFC #39583 proposes moving it out-of-tree), so
#: it earns a place in the method-generality ladder and not in a budget sweep; `gptq-8bit` is
#: there for dose-response on the reference contrast.
SWEEP_ARMS = ("bf16", "gptq-4bit", "awq-4bit")

#: 🚨 `1x` is ABSENT and comes from `main-v1`. See the module docstring.
SWEEP_LEVELS = ("2x", "4x")

#: §2.2's sweep composition: 1,400 items per cell, the nested 500-item GSM8K prefix plus the
#: same 900 BBH items. ARC is excluded from the sweep by §2.2 (it is the H6 mode-pairing arm and
#: its budget is not the binding constraint on it).
SWEEP_COMPOSITION = {"gsm8k_cot": 500, "bbh": 900}


def sweep_plan(tasks: dict) -> list[tuple[str, str, str | None, int]]:
    """`(lm_eval_task, task, subtask, n_items)` per cell, at the SWEEP counts.

    ⚠️ `gsm8k_cot` rather than `gsm8k`: §9c excludes the hash-marker arm from the registered
    interaction because a metric floored near 1% strict has no variance, and H1's budget
    response is measured where the marker is one the models actually emit.
    """
    config = tasks["gsm8k_cot"]
    plan: list[tuple[str, str, str | None, int]] = [
        (config["lm_eval_task"], "gsm8k_cot", None, SWEEP_COMPOSITION["gsm8k_cot"]),
    ]
    bbh = tasks["bbh"]
    prefix = bbh["lm_eval_task_prefix"]
    subtasks = [s["name"] for s in bbh["subtasks"]]
    base, extra = divmod(SWEEP_COMPOSITION["bbh"], len(subtasks))
    for index, name in enumerate(subtasks):
        plan.append((f"{prefix}{name}", "bbh", name, base + (1 if index < extra else 0)))
    return plan


def main() -> int:
    models = yaml.safe_load((CONFIGS / "models.yaml").read_text())["models"]
    tasks = yaml.safe_load((CONFIGS / "tasks.yaml").read_text())["tasks"]
    specs, calibration = load_specs(CONFIGS / "quant.yaml")
    max_num_seqs = frozen_max_num_seqs()

    # 🚨 Assert the levels EXIST in the config before anything loads, and that they differ from
    # 1x. A level silently equal to 1x would produce a "sweep" with no independent variable and
    # a report that looks complete, which is the vacuous-truth shape this project has hit four
    # times.
    for task_key in SWEEP_COMPOSITION:
        budgets = tasks[task_key]["max_gen_toks"]
        for level in SWEEP_LEVELS:
            if level not in budgets:
                raise SystemExit(
                    f"REFUSING TO RUN: configs/tasks.yaml {task_key}.max_gen_toks has no "
                    f"{level!r}. The three levels are frozen in the pre-registration as "
                    f"absolute integers; a missing one means the config and the registration "
                    f"have diverged."
                )
            if budgets[level] == budgets["1x"]:
                raise SystemExit(
                    f"REFUSING TO RUN: {task_key} {level} == 1x ({budgets[level]}). A sweep "
                    f"level equal to the baseline has no independent variable, and the run "
                    f"would report as complete having measured nothing."
                )

    plan = sweep_plan(tasks)
    cells = [c for c in build_cells(models, specs, calibration)
             if c["precision"] in SWEEP_ARMS]
    if SMOKE:
        # One model, one arm, one level, few items — but a QUANTIZED arm at a NON-1x level, so
        # the two things that have never run (the budget parameter reaching `run_cell`, and a
        # cap other than 256) are both exercised rather than assumed.
        cells = [c for c in cells if c["cell_id"] == "qwen2.5-3b|gptq-4bit"]
        plan = [("gsm8k_cot", "gsm8k_cot", None, 8)]
        levels: tuple[str, ...] = ("2x",)
        print("SMOKE MODE: qwen2.5-3b|gptq-4bit, gsm8k_cot 8 items, 2x only")
    else:
        levels = SWEEP_LEVELS

    n_items = sum(limit for *_, limit in plan)
    print(f"run_id       : {RUN_ID}")
    print(f"max_num_seqs : {max_num_seqs}  (frozen by A7, not assumed)")
    print(f"arms         : {list(SWEEP_ARMS)}   levels: {list(levels)}")
    print(f"cells        : {len(cells)} x {len(levels)} levels = "
          f"{len(cells) * len(levels)} units x {n_items} items")
    print("🚨 1x is NOT regenerated: it comes from main-v1 via the nested subsample")
    for level in levels:
        print(f"  {level:3} caps: " + ", ".join(
            f"{k}={tasks[k]['max_gen_toks'][level]}" for k in SWEEP_COMPOSITION))

    report: dict = {"stage": "sweep", "run_id": RUN_ID, "smoke": SMOKE,
                    "max_num_seqs": max_num_seqs, "arms": list(SWEEP_ARMS),
                    "levels": list(levels), "cells": []}
    failed: list[str] = []
    for level in levels:
        print(f"\n=== budget level {level} ===")
        for index, cell in enumerate(cells, 1):
            print(f"--- unit {index}/{len(cells)} at {level}: {cell['cell_id']} ---",
                  flush=True)
            counts = run_one_cell(
                # `run_id` injected on the cell, because `run_one_cell` lives in sm_reference
                # and would otherwise write these under `main-v1`.
                {**cell, "run_id": RUN_ID}, plan, max_num_seqs,
                models[cell["model_key"]]["repo"], budget_level=level,
            )
            if counts is None:
                failed.append(f"{cell['cell_id']}@{level}")
                report["cells"].append({"cell_id": cell["cell_id"], "budget_level": level,
                                        "error": "cell failed"})
            else:
                report["cells"].append({"cell_id": cell["cell_id"], "budget_level": level,
                                        **counts})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "sweep_report.json").write_text(json.dumps(report, indent=2))
    done = [c for c in report["cells"] if "error" not in c]
    print(f"\n{'=' * 70}")
    print(f"units        : {len(done)}/{len(cells) * len(levels)}")
    print(f"generations  : {sum(c.get('written', 0) for c in done):,} written")
    print(f"report -> {OUTPUT_DIR / 'sweep_report.json'}")
    print(f"NEXT: .venv/bin/python scripts/peek_generations.py --run-id {RUN_ID}")
    print("      🚨 rule 4b: read 3 generations BEFORE scoring or quoting any number")
    if failed:
        # Every unit is load-bearing: H1's response curve needs both levels on every arm, so a
        # missing unit is a missing point on the curve rather than a corroborating detail.
        print(f"\nFAILED UNITS (all load-bearing for the response curve): {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
