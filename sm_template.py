#!/usr/bin/env python3
"""SageMaker entry point — `template` stage. The prompt-template FACTOR (prereg §3b-bis).

**This is the probe that defends the paper's headline**, promoted from a calibration
measurement by planner ruling R19.2.

The problem it answers. Our depressed strict scores are plausibly *caused* by our own
choice to chat-template: R1.2 diagnosed `--apply_chat_template` as the mechanism behind
arXiv:2601.14277's 24.64-vs-77.63 strict/flexible collapse, and we then adopted it for
every arm. A reviewer can therefore say *"this is your configuration, not a property of
quantization evaluation as practised."* The paper's own reframe is that **a level artifact
is not a delta artifact** — so the template belongs on the level axis, and the claim
belongs on the **invariance of the delta across levels**.

    level `chat` (PRIMARY)     apply_chat_template=True,  BBH until = ["</s>","Q:","<|im_end|>"]
    level `raw`  (robustness)  apply_chat_template=False, BBH until = ["</s>","Q","\\n\\n"]

🚨 **The terminator set is part of each level's DEFINITION, not a constant** (R19.1). At
`raw`, lm-eval's `_cot_fewshot` set is correct *by construction* — it was written for that
format, where blank lines occur only between Q/A pairs. At `chat` it is not: the assistant
opens a fresh turn and its own paragraphing emits `\\n\\n` mid-answer, which zeroed 5 of 6
BBH subtasks for every model including bf16 Llama-8B (median 1–7 generated tokens).

🚨 **THE DECISION RULE IS PRE-REGISTERED AND MUST NOT BE READ AFTER THE FACT**
(`analysis/preregistration.md` §3b-bis):

1. interaction holds in **sign and rough magnitude** under both levels → `chat` primary,
   `raw` reported as robustness, self-infliction **answered empirically**;
2. they **diverge** → the divergence **is the finding**: the template is a **moderator** of
   the measured quantization effect. Stronger, on-thesis, reported as the primary framing.

Both branches publishable, neither preferred, and we pre-committed to not relabelling
branch 2 as a limitation.

**Scope: DEFAULT tier.** R20.1 selects by measured BBH rate *and* caps the probe at 15% of
remaining budget. The measured 7,099 gen/h selects FULL by the rate table, but FULL is
22.4% of remaining against that cap, so it steps down: **3 models × 3 contrasted arms ×
2 levels × {gsm8k_cot 500, BBH 500} = 18 cells, 18,000 generations, ~3.8 h.** The two rules
conflicting is the fraction rule working as intended.

⚠️ Both levels run through `sm_pilot.CELL_SCRIPT`, imported rather than copied, so the
levels cannot differ in any respect other than the two the factor defines.
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

CHECKPOINT_DIR = Path(os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints"))
OUTPUT_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
CONFIGS = Path(__file__).parent / "configs"

SMOKE = os.environ.get("SM_HP_SMOKE") == "1" or "--smoke" in sys.argv

# Separate `run_id` from every other stage: these generations are a DIFFERENT population
# (two template levels), and pooling them into `pilot-v2` would silently contaminate the
# annotation pool that A11 samples blind.
RUN_ID = "template-v1-smoke" if SMOKE else "template-v1"
GENERATIONS_ROOT = CHECKPOINT_DIR / "generations"

# The two levels of the factor. Order matters only for log readability.
TEMPLATE_LEVELS = ("chat", "raw")

# ⚠️ The THREE CONTRASTED arms, matching prereg §1.1's amended stratum (R18.4.1). Not all
# five: the probe asks whether the *interaction* is template-invariant, and `gptq-8bit` /
# `bnb-nf4` exist for dose-response and method generality, neither of which this tests.
TEMPLATE_ARMS = ("bf16", "gptq-4bit", "awq-4bit")

# DEFAULT tier composition (R20.1). `gsm8k_cot` carries the primary interaction; BBH carries
# the largest template effect. Dropping either guts the claim, so both are mandatory.
TEMPLATE_COMPOSITION = {"gsm8k_cot": 500, "bbh": 500}


def template_plan(tasks: dict, n_gsm8k: int, n_bbh: int) -> list[tuple[str, str, str | None, int]]:
    """`(lm_eval_task, task, subtask, n_items)` for one cell.

    BBH is spread over ALL six subtasks rather than concentrated, because the format
    gradient (free-form → boolean → multiple-choice) is where the template effect is
    expected to vary most, and a single-subtask probe could not show that.
    """
    plan: list[tuple[str, str, str | None, int]] = [
        ("gsm8k_cot", "gsm8k_cot", None, n_gsm8k),
    ]
    subtasks = [s["name"] for s in tasks["bbh"]["subtasks"]]
    base, extra = divmod(n_bbh, len(subtasks))
    prefix = tasks["bbh"]["lm_eval_task_prefix"]
    for index, name in enumerate(subtasks):
        plan.append((f"{prefix}{name}", "bbh", name, base + (1 if index < extra else 0)))
    return plan


def build_cells(models: dict, specs: dict, calibration) -> list[dict]:
    """Every (model × arm × template level) cell, weights resolved up front.

    ⚠️ `cell_id` CARRIES THE TEMPLATE LEVEL. Without it both levels would share a
    `(run_id, cell_id, task, subtask, item_id, …)` join key, and `run_cell` would treat the
    second level's items as already written and skip all of them — reporting success while
    generating nothing, which is the exact defect logged against `pilot-smoke-r2`.
    """
    cells: list[dict] = []
    for level in TEMPLATE_LEVELS:
        for model_key, model in models.items():
            for precision in TEMPLATE_ARMS:
                if precision == "bf16":
                    # `bf16` is absent from specs (load_specs skips method=="none"), so it
                    # is handled before any attribute access.
                    path, digest = model["repo"], None
                else:
                    spec = specs[precision]
                    digest = config_hash(spec, calibration, model["revision"])
                    resolved = output_dir(CHECKPOINT_DIR, model_key, spec, digest)
                    if not (resolved / "config.json").exists():
                        raise SystemExit(
                            f"REFUSING TO RUN: no checkpoint for {model_key}/{precision} "
                            f"at {resolved} (recipe hash {digest}). Job 1 must have "
                            f"completed and checkpoint_s3_uri must match its."
                        )
                    path = str(resolved)
                cells.append({
                    "cell_id": f"{model_key}|{precision}|{level}",
                    "template_level": level,
                    "model_key": model_key,
                    "revision": model["revision"],
                    "precision": precision,
                    "quant_method": None if precision == "bf16" else specs[precision].method,
                    "bits": 16 if precision == "bf16" else specs[precision].bits,
                    "model_path": path,
                    "quant_config_hash": digest,
                })
    return cells


def run_one_cell(cell: dict, plan: list, max_num_seqs: int, base_repo: str) -> dict | None:
    """One cell in its own process, via the PILOT's cell script.

    Subprocess per cell for the reason job 1 established: torch's caching allocator and the
    engine's hooks do not reliably release GPU memory in-process, and many sequential engine
    loads in one process is the shape that cost four arms there.
    """
    tag = cell["cell_id"].replace("|", "__")
    out_path = GENERATIONS_ROOT / f".{tag}.counts.json"
    payload = {
        "template_level": cell["template_level"],
        "cell": {**cell, "base_repo": base_repo},
        "plan": plan,
        "max_num_seqs": max_num_seqs,
        "run_id": RUN_ID,
        "generations_root": str(GENERATIONS_ROOT),
        "tasks_yaml": str(CONFIGS / "tasks.yaml"),
        "models_yaml": str(CONFIGS / "models.yaml"),
        "out_path": str(out_path),
    }
    payload_path = GENERATIONS_ROOT / f".{tag}.payload.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(payload))

    result = subprocess.run(
        [sys.executable, "-c", CELL_SCRIPT, str(payload_path)], timeout=21600, check=False,
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
    n_gsm8k, n_bbh = TEMPLATE_COMPOSITION["gsm8k_cot"], TEMPLATE_COMPOSITION["bbh"]

    if SMOKE:
        # Cover BOTH LEVELS and both tasks -- the whole point of this stage is the
        # contrast, so a rehearsal of one level clears nothing. One model, one arm.
        models = {"qwen2.5-3b": models["qwen2.5-3b"]}
        n_gsm8k = n_bbh = 6
        print(f"SMOKE MODE: qwen2.5-3b, both template levels, {n_gsm8k}+{n_bbh} items")

    cells = build_cells(models, specs, calibration)
    if SMOKE:
        cells = [c for c in cells if c["precision"] in ("bf16", "gptq-4bit")]

    plan = template_plan(tasks, n_gsm8k, n_bbh)
    per_cell = sum(limit for *_, limit in plan)

    print(f"run_id       : {RUN_ID}")
    print(f"max_num_seqs : {max_num_seqs}  (frozen by A7, not assumed)")
    print(f"levels       : {list(TEMPLATE_LEVELS)}   arms: {list(TEMPLATE_ARMS)}")
    print(f"cells        : {len(cells)}   items/cell: {per_cell}   "
          f"total: {len(cells) * per_cell:,} generations")
    for task in tasks:
        if task == "bbh":
            print(f"  bbh until  : chat={tasks['bbh']['until_by_template']['chat']}  "
                  f"raw={tasks['bbh']['until_by_template']['raw']}")
            break
    print()

    report: dict = {
        "stage": "template", "run_id": RUN_ID, "smoke": SMOKE,
        "max_num_seqs": max_num_seqs, "levels": list(TEMPLATE_LEVELS),
        "arms": list(TEMPLATE_ARMS), "composition": {"gsm8k_cot": n_gsm8k, "bbh": n_bbh},
        "cells": [],
    }
    failed: list[str] = []
    for index, cell in enumerate(cells, 1):
        print(f"=== cell {index}/{len(cells)}: {cell['cell_id']} ===", flush=True)
        started = time.monotonic()
        counts = run_one_cell(cell, plan, max_num_seqs, models[cell["model_key"]]["repo"])
        elapsed = time.monotonic() - started
        if counts is None:
            failed.append(cell["cell_id"])
            report["cells"].append({"cell_id": cell["cell_id"],
                                    "template_level": cell["template_level"],
                                    "error": "cell subprocess failed"})
            print(f"    FAILED after {elapsed/60:.1f} min", flush=True)
            continue
        report["cells"].append({
            "cell_id": cell["cell_id"], "template_level": cell["template_level"],
            "precision": cell["precision"], "model_key": cell["model_key"],
            "minutes": round(elapsed / 60, 2), **counts,
        })
        print(f"       {counts['written']} written, {counts['skipped']} skipped, "
              f"{elapsed/60:.1f} min", flush=True)

    (OUTPUT_DIR / "template_report.json").write_text(json.dumps(report, indent=2))
    done = [c for c in report["cells"] if "error" not in c]
    written = sum(c["written"] for c in done)
    print("=" * 70)
    print(f"cells        : {len(done)}/{len(cells)}")
    print(f"generations  : {written:,} written")
    print(f"report -> {OUTPUT_DIR / 'template_report.json'}")

    if failed:
        # Every cell is load-bearing here: the claim is a CONTRAST across levels, so a
        # missing cell breaks the pairing rather than merely reducing coverage.
        print(f"\nFAILED CELLS (all load-bearing — the claim is a paired contrast): {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
