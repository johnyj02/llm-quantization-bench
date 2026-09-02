#!/usr/bin/env python3
"""SageMaker entry point, `anchor` stage. Task A9, Anchor A. Gate G5a. `run_id=anchor-v1`.

🚨 **MANDATORY, UNGATED AND BLOCKING**, and it has never run. `EXECUTOR-TASKS.md` A9:

> **Anchor A.** Reproduce the official Qwen-published `Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4` /
> `-GPTQ-Int8` / `-AWQ` checkpoints' behaviour and compare against our self-quantized equivalents
> at matched settings (`group_size:128, desc_act:false`). *Pass:* our self-quantized GSM8K accuracy
> is within **±2.0 pp** of the official checkpoint's at the same precision. **A larger gap means
> our calibration or serving config is wrong, debug before proceeding.**

**Why this is the cheapest insurance left in the project.** Nine checkpoints were built with a
specific `actorder`/`desc_act` configuration whose library default we already found silently wrong
once (`notes/RUNLOG.md`, run 2: `GPTQModifier.actorder` defaults to `Sentinel("static")`, so
omitting it yields static reordering, which is neither the `false` we require nor the `true` we
reject). Anchor A is the check that would catch a repeat of that before `main-v1` spends 17 hours
on it.

**The construction, and it is what makes the comparison mean anything.** Both sides are measured
through **our** pipeline, at the frozen serving configuration, on the same items, in the same
engine version. So any pipeline error is COMMON to both and cancels in the difference, and what
remains isolates the weights. Comparing our number against a number Qwen published elsewhere would
confound the weights with every protocol difference between us and them, which is the exact
confound §0.1(b) says makes published deltas incomparable.

**Six cells:** 3 arms × {official checkpoint, our self-quantized checkpoint}. One engine load per
cell, in its own subprocess, for the reason job 1 established the hard way: torch's caching
allocator and llmcompressor's hooks do not reliably release GPU memory in-process.

🚨 **WHAT ANCHOR A DOES AND DOES NOT TEST, and this framing must reach the paper** (R30). The
official checkpoints were calibrated on **Qwen's** data, not our `ultrachat_200k` 512×2048. So
Anchor A does **not** test *reproduction*; it tests that our pipeline produces a checkpoint of
**comparable quality**, and ±2.0 pp is a quality-comparability band. Stating it otherwise invites a
reviewer to read the anchor as a reproduction claim it cannot support.

⚠️ **Run `scripts/anchor_preflight.py` first, and gate this job behind it** (R30). It reads the three
official checkpoints' quantization settings for free and directly tests the most likely failure
mode, a settings mismatch, which is what A9's "our calibration or serving config is wrong" actually
means. Passed 2026-09-02: all three agree with our recipe on `bits`, `group_size` 128, `desc_act`
False, and AWQ's gemm/`zero_point`.

⚠️ **The verdict is NOT computed in this job.** Anchor A's comparison is CPU work over stored
generations, so computing it here would put a gate decision inside a billable process that cannot be
re-run without a queue slot. `scripts/score_anchor.py` computes it, which also means the verdict can
be recomputed if the criterion metric ever changes.
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

from quantartifact.quantize import config_hash, load_specs, output_dir  # noqa: E402

CHECKPOINT_DIR = Path(os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints"))
OUTPUT_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
CONFIGS = Path(__file__).parent / "configs"

SMOKE = os.environ.get("SM_HP_SMOKE") == "1" or "--smoke" in sys.argv

#: Separate `run_id` for the rehearsal, for the reason recorded in `sm_pilot.py`: a shared id would
#: pool 20-item smoke generations into the population a real comparison is computed over.
RUN_ID = "anchor-v1-smoke" if SMOKE else "anchor-v1"
GENERATIONS_ROOT = CHECKPOINT_DIR / "generations"

#: The model Anchor A is defined on. Qwen is the only one of the three with official quantized
#: checkpoints (`configs/models.yaml: has_official_quantized_checkpoints`), which is why A9's
#: ungated anchor is Qwen rather than Llama: no gated dependency, so it cannot be blocked.
ANCHOR_MODEL = "qwen2.5-3b"

#: 🚨 The pass criterion, set by planner ruling R30. TWO-TIER, and the tiers are not symmetric.
#:
#: **`gsm8k_cot` PERMISSIVE blocks.** Not strict, and not both. A9 says "GSM8K accuracy", and we
#: have two registered GSM8K configs and four rungs whose values differ by an order of magnitude,
#: so which one the band applies to decides whether the gate can fail at all.
#:
#: * On the hash arm at rung 1 the metric is floored near 1%, so a ±2.0 pp band is wider than the
#:   whole measurable range and the gate would validate nothing.
#: * Anchor A exists to detect a **calibration or serving error**, and those damage **competence**.
#:   Rung 1 conflates competence with marker compliance, so gating on it means a real compliance
#:   effect, which is a finding we want to measure, could spuriously fail a blocking gate and send
#:   us debugging a non-problem.
#: * Requiring BOTH rungs was rejected for the same reason: ±2.0 pp against a 13.5% value is a 15%
#:   relative band on a metric known to vary between checkpoints for reasons unrelated to
#:   calibration. False-failure risk on a blocking gate.
#:
#: `gsm8k_cot` over `gsm8k` is a **disambiguation, not a substitution**: both are GSM8K, and §9c
#: already made `gsm8k_cot` primary for the interaction.
CRITERION_METRIC: tuple[str, str] = ("gsm8k_cot", "permissive")
CRITERION_PP = 2.0

#: A strict discrepancy is a **documented diagnostic requiring explanation, NOT a block** (R30).
#: A large strict gap alongside a small permissive one is informative in its own right: it points at
#: chat-template or BOS handling rather than at calibration.
DIAGNOSTIC_METRICS: tuple[tuple[str, str], ...] = (
    ("gsm8k_cot", "strict"),
    ("gsm8k", "permissive"),
    ("gsm8k", "strict"),
)

#: 🚨 **1319 on the GATE task, not 500** (R30), and the reason is resolution rather than thoroughness.
#: The comparison is paired, so the standard error on the difference is driven by item-level
#: discordance rather than by the ~63% marginals. At a plausible 10-20% discordance between two
#: different quantizations:
#:
#: | n | SE on the difference | ±2.0 pp band |
#: |---|---|---|
#: | 500 | 1.4-2.0 pp | 1.0-1.4 SE |
#: | 1319 | 0.9-1.2 pp | 1.6-2.3 SE |
#:
#: **At n=500 a perfect pipeline fails this blocking gate roughly one time in six.** The anchor runs
#: once, so the extra ~1.8 h is the cheapest possible fix for a badly calibrated gate.
#:
#: `gsm8k` stays at 500 because it is DIAGNOSTIC only: it informs the explanation of a strict
#: discrepancy and never blocks, so it does not need gate-grade resolution.
ANCHOR_TASKS: tuple[tuple[str, str, int], ...] = (
    ("gsm8k_cot", "gsm8k_cot", 1319),
    ("gsm8k", "gsm8k", 500),
)

CELL_SCRIPT = r"""
import json, sys
sys.path.insert(0, "/opt/ml/code/src")
from pathlib import Path
from quantartifact.generate import EngineSpec, VLLMRunner, run_cell
from quantartifact.tasks.materialize import render_task_prompts
from transformers import AutoTokenizer
import yaml

payload = json.loads(Path(sys.argv[1]).read_text())
cell = payload["cell"]
tasks = yaml.safe_load(Path(payload["tasks_yaml"]).read_text())["tasks"]

spec = EngineSpec(
    model_repo=cell["model_path"],
    model_revision=cell["revision"],
    quant_method=cell["quant_method"],
    bits=cell["bits"],
    max_num_seqs=payload["max_num_seqs"],
)
runner = VLLMRunner(spec)

# 🚨 The tokenizer comes from the BASE repo for BOTH sides, never from the checkpoint. A quantized
# checkpoint's tokenizer files are a copy, and if the official and our checkpoint carried even
# trivially different copies the prompts would differ and the comparison would confound weights
# with prompts. This is the one thing that must be identical across the two sides.
tokenizer = AutoTokenizer.from_pretrained(cell["base_repo"], revision=cell["base_revision"])

totals = {"written": 0, "skipped": 0, "total": 0}
for lm_eval_task, task, limit in payload["plan"]:
    config = tasks[task]
    prompts = list(render_task_prompts(
        lm_eval_task, tokenizer, task=task, subtask=None,
        prompt_variant="fewshot_cot", model_key=cell["model_key"],
        model_repo=cell["base_repo"], model_revision=cell["base_revision"],
        num_fewshot=config.get("num_fewshot"), apply_chat_template=True,
        fewshot_as_multiturn=False, limit=limit,
    ))
    items = [{"item_id": p.item_id, "prompt": p.prompt_text, "gold": p.gold,
              "prompt_token_ids": p.prompt_token_ids} for p in prompts]
    counts = run_cell(
        runner,
        run_id=payload["run_id"], cell_id=cell["cell_id"], task=task, subtask=None,
        items=items, prompt_variant="fewshot_cot", budget_level="1x",
        max_gen_toks=config["max_gen_toks"]["1x"], until=config["until"],
        quant_config_hash=cell["quant_config_hash"],
        output_root=Path(payload["generations_root"]),
        template_level="chat",
        repeat_idx=0,
    )
    for key in totals:
        totals[key] += counts[key]
    print(f"    {task:16} {counts['written']:4} written, {counts['skipped']:4} skipped",
          flush=True)

# 🚨 The ENGINE STATE, and it is what turns this stage's central claim into a checked one.
# The docstring says both sides are measured "at the frozen serving configuration ... so any
# pipeline error is COMMON to both and cancels". Without recording the state, the job produces
# no evidence for that sentence.
#
# The axis that is NOT held fixed by construction is the one following from the weights: our
# GPTQ-Int4 checkpoint and Qwen's are the same bit-width but not the same on-disk size, so at a
# fixed gpu_memory_utilization they can yield a different KV-cache size and therefore a different
# num_gpu_blocks -- and this project's own H8 result is that scheduling changes generated text by
# up to 59%. vLLM also picks its GPTQ kernel from the checkpoint's config, so the two sides could
# take Marlin and non-Marlin paths, and H8 pinned Marlin specifically as the order-sensitive one.
# Neither is likely to bite, but the defect was that the job could not TELL us whether it had.
totals["engine_state"] = getattr(runner, "engine_state", None)
totals["batch_invariance"] = getattr(runner, "batch_invariance", None)
Path(payload["out_path"]).write_text(json.dumps(totals, default=str))
print(f"__CELL_OK__ {json.dumps({k: v for k, v in totals.items() if k in ('written','skipped','total')})}")
"""


def anchor_cells(models: dict, specs: dict, calibration) -> list[dict]:
    """The six cells: 3 arms x {official, ours}, resolved to concrete weights up front.

    Resolution happens BEFORE any engine loads, so a missing checkpoint costs seconds rather than
    surfacing an hour in. Our side resolves **by recipe hash**, which is what ties this job to the
    exact artifacts job 1 produced rather than to whatever happens to be on disk.
    """
    model = models[ANCHOR_MODEL]
    anchor = yaml.safe_load((CONFIGS / "models.yaml").read_text())
    anchor_a = anchor["anchor_checkpoints"]["anchor_a"]

    if not anchor_a.get("blocking"):
        raise SystemExit(
            "configs/models.yaml: anchor_a is not marked blocking. A9 defines it as mandatory, "
            "ungated and blocking, so a config that says otherwise is a drift I will not run past."
        )

    cells: list[dict] = []
    for arm, official in anchor_a["checkpoints"].items():
        if arm not in specs:
            raise SystemExit(f"anchor_a names arm {arm!r}, absent from configs/quant.yaml")
        spec = specs[arm]

        # --- our side, resolved BY HASH ---
        digest = config_hash(spec, calibration, model["revision"])
        ours = output_dir(CHECKPOINT_DIR, ANCHOR_MODEL, spec, digest)
        if not (ours / "config.json").exists():
            raise SystemExit(
                f"REFUSING TO RUN: no self-quantized checkpoint for {ANCHOR_MODEL}/{arm} at "
                f"{ours} (recipe hash {digest}). Job 1 must have completed and this job's "
                f"checkpoint_s3_uri must match job 1's."
            )
        cells.append({
            "cell_id": f"{ANCHOR_MODEL}|{arm}|ours",
            "side": "ours", "arm": arm,
            "model_key": ANCHOR_MODEL,
            "model_path": str(ours), "revision": model["revision"],
            "base_repo": model["repo"], "base_revision": model["revision"],
            "quant_method": spec.method, "bits": spec.bits,
            "quant_config_hash": digest,
        })

        # --- the official side, pinned by revision ---
        cells.append({
            "cell_id": f"{ANCHOR_MODEL}|{arm}|official",
            "side": "official", "arm": arm,
            "model_key": ANCHOR_MODEL,
            "model_path": official["repo"], "revision": official["revision"],
            "base_repo": model["repo"], "base_revision": model["revision"],
            "quant_method": spec.method, "bits": spec.bits,
            # No recipe hash: we did not build it. Recorded as null so a record from the official
            # side can never be mistaken for one of ours.
            "quant_config_hash": None,
        })
    return cells


def frozen_max_num_seqs() -> int:
    """Read the A7-frozen serving value. REFUSES if it was never audited.

    Same guard as `sm_pilot.py`, for the same reason: a value that later changes makes these
    generations non-comparable to everything else, and Anchor A's entire logic is a comparison.
    """
    serving = yaml.safe_load((CONFIGS / "models.yaml").read_text())["serving"]
    if not serving.get("max_num_seqs_resolved_by_audit"):
        raise SystemExit(
            "configs/models.yaml: max_num_seqs is not marked resolved_by_audit. Anchor A compares "
            "two checkpoints at a FIXED serving configuration; an unaudited value would make the "
            "comparison rest on a setting nothing has justified."
        )
    return int(serving["max_num_seqs"])


def run_one_cell(cell: dict, plan: list, max_num_seqs: int) -> dict | None:
    """One cell in its own process. Subprocess per cell, per job 1's lesson."""
    tag = cell["cell_id"].replace("|", "__")
    out_path = GENERATIONS_ROOT / f".{tag}.counts.json"
    payload_path = GENERATIONS_ROOT / f".{tag}.payload.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps({
        "cell": cell, "plan": plan, "max_num_seqs": max_num_seqs,
        "run_id": RUN_ID, "generations_root": str(GENERATIONS_ROOT),
        "tasks_yaml": str(CONFIGS / "tasks.yaml"), "out_path": str(out_path),
    }))
    sys.stdout.flush()
    result = subprocess.run(
        [sys.executable, "-c", CELL_SCRIPT, str(payload_path)], timeout=21600
    )
    payload_path.unlink(missing_ok=True)
    if result.returncode != 0:
        return None
    return json.loads(out_path.read_text()) if out_path.exists() else None


def main() -> int:
    models = yaml.safe_load((CONFIGS / "models.yaml").read_text())["models"]
    specs, calibration = load_specs(CONFIGS / "quant.yaml")

    max_num_seqs = frozen_max_num_seqs()
    plan = [(lm, task, n) for lm, task, n in ANCHOR_TASKS]
    cells = anchor_cells(models, specs, calibration)

    if SMOKE:
        # One ARM, both sides, few items. Both sides is the point: a rehearsal that ran only our
        # checkpoint would not exercise the official-repo download path, which is the only part of
        # this stage that has never run anywhere.
        cells = [c for c in cells if c["arm"] == "gptq-4bit"]
        plan = [("gsm8k", "gsm8k", 8), ("gsm8k_cot", "gsm8k_cot", 8)]
        print("SMOKE MODE: gptq-4bit only, both sides, 8 items per task")

    print(f"run_id       : {RUN_ID}")
    print(f"model        : {ANCHOR_MODEL}")
    print(f"max_num_seqs : {max_num_seqs}  (frozen by A7, not assumed)")
    print(f"criterion    : +/-{CRITERION_PP} pp on "
          f"{CRITERION_METRIC if CRITERION_METRIC else 'METRIC UNSET (planner R30)'}")
    print(f"cells        : {len(cells)}")
    for cell in cells:
        print(f"  {cell['cell_id']:34} {Path(cell['model_path']).name}")
    print()

    report: dict = {
        "gate": "G5a", "anchor": "A", "run_id": RUN_ID, "model": ANCHOR_MODEL,
        "max_num_seqs": max_num_seqs, "criterion_pp": CRITERION_PP,
        "criterion_metric": list(CRITERION_METRIC) if CRITERION_METRIC else None,
        "cells": [], "smoke": SMOKE,
    }
    failed: list[str] = []
    for index, cell in enumerate(cells, start=1):
        print(f"=== cell {index}/{len(cells)}: {cell['cell_id']} ===", flush=True)
        started = time.monotonic()
        counts = run_one_cell(cell, plan, max_num_seqs)
        elapsed = time.monotonic() - started
        if counts is None:
            failed.append(cell["cell_id"])
            report["cells"].append({**cell, "error": "cell failed"})
            print(f"    FAILED after {elapsed / 60:.1f} min", flush=True)
            continue
        report["cells"].append({**cell, **counts, "minutes": elapsed / 60})
        print(f"    {counts['written']} written, {counts['skipped']} skipped, "
              f"{elapsed / 60:.1f} min", flush=True)

    (OUTPUT_DIR / "anchor_report.json").write_text(json.dumps(report, indent=2))
    print("=" * 74)
    print(f"cells        : {len(cells) - len(failed)}/{len(cells)}")
    print(f"generations  : {sum(c.get('written', 0) for c in report['cells']):,} written")
    print(f"report -> {OUTPUT_DIR / 'anchor_report.json'}")

    # 🚨 Scoring and the verdict are DELIBERATELY NOT computed here. Anchor A's comparison is a
    # CPU operation over stored generations, so doing it in the job would put a gate decision
    # inside a billable process that cannot be re-run without a queue slot. `scripts/score_anchor.py`
    # computes it, which also means the verdict can be recomputed if R30 changes the metric.
    print()
    print("NEXT: .venv/bin/python scripts/peek_generations.py --run-id " + RUN_ID)
    print("      (rule 4b: read 3 raw generations BEFORE scoring or reporting any number)")
    print("      then .venv/bin/python scripts/score_anchor.py --run-id " + RUN_ID)

    if failed:
        print(f"\nFAILED CELLS: {failed}")
        print("Every cell is load-bearing: the comparison is a difference, so a missing side "
              "leaves it undefined rather than partial.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
