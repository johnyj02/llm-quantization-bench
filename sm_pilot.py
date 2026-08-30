#!/usr/bin/env python3
"""SageMaker entry point — `pilot` stage. JOB 3 of the plan. Task A8, `run_id=pilot-v1`.

The first job that produces **study data** rather than instruments. It serves four
purposes at once, which is why it is worth its own run:

1. **The annotation source pool** (override O11). Judge validation draws from a
   dedicated stratification pilot, not from the main runs — annotation cannot wait
   for them, and batch composition differs anyway so the generations would not be
   byte-identical. Judge validation is a claim about the *population* of
   (model × precision × task) generations, not about specific strings.
2. **Throughput on real cells**, replacing every estimate downstream.
3. **A noise-floor smoke test** against `TRUNC_STOPSEQ` and the `A7` frontier.
4. **Planner ruling R12.1** — does the AWQ weight difference move ACCURACY? Job 1
   left both `qwen2.5-3b/awq-4bit` checkpoints in S3 (one built without
   `offload_device`, one with), and they are byte-different. Two extra cells score
   them side by side. *"Weights differ, scores agree within the noise floor"* is
   reassuring and valuable; *"weights differ and scores differ"* is significant.
   Either beats reporting a byte diff with no accuracy consequence attached.

**17 cells × 300 items** (GSM8K 100 + BBH 120 spread over all six subtasks +
ARC-chat 80) = **5,100 generations**: A8's 15 plus R12.1's 2.

🚨 **`max_num_seqs` is READ from the frozen `A7` result, never assumed.** The pilot's
generations become the annotation population and a throughput baseline, so a value
that later changes would make them non-comparable to the main runs. This job REFUSES
to start if job 2 has not frozen it.
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

# 🚨 SMOKE WRITES UNDER A DIFFERENT run_id, and this one is not cosmetic. `run_id` is
# the first field of `records.JOIN_KEY` and the top directory of `cell_path`, so a
# rehearsal sharing it would pool 10-item smoke generations into the very population
# the judge-validation sample is drawn from — and `A11` draws that sample blind, so
# nothing downstream would notice. Same failure shape as the determinism smoke being
# readable as a real measurement.
# 🚨 BUMPED v1 -> v2 on 2026-08-29 (planner ruling R18.1). NOT cosmetic, and NOT a
# deletion.
#
# `run_cell` skips any item already written under this `run_id`, so re-running as
# `pilot-v1` after the BBH `until` fix would have skipped all 5,100 existing records --
# regenerating NOTHING while printing `17/17 cells`. That is precisely the skip-on-resume
# defect this file was filed for after `pilot-smoke-r2`.
#
# `pilot-v1` is deliberately RETAINED rather than deleted: it is the evidence for the BBH
# terminator finding (5 of 6 subtasks at 0.0% for every arm including bf16 Llama-8B), which
# the paper reports in the Introduction. Deleting it would destroy the artifact behind a
# main-text claim.
#
# ⚠️ `gsm8k` and `arc_challenge_chat` should regenerate BIT-IDENTICALLY here, since `until`
# is per-task and only BBH's changed. That is a free end-to-end determinism check, so
# compare them rather than assuming.
RUN_ID = "pilot-v2-smoke" if SMOKE else "pilot-v2"
# Generations are the durable artifact, so they go to the CONTINUOUSLY SYNCED
# checkpoint prefix rather than /opt/ml/model, which only uploads when the job ENDS.
# A timeout at cell 14 of 17 would otherwise lose everything.
GENERATIONS_ROOT = CHECKPOINT_DIR / "generations"

PRECISIONS = ("bf16", "gptq-8bit", "gptq-4bit", "awq-4bit", "bnb-nf4")

# A8: 300 items per cell. BBH's 120 is spread over all six subtasks so every answer
# FORMAT CLASS (free_form, boolean, multiple_choice) is represented in the annotation
# pool -- a pool drawn from one subtask would make the judge-validation claim narrower
# than the population it is supposed to describe.
PILOT_COMPOSITION = {"gsm8k": 100, "bbh": 120, "arc_challenge_chat": 80}

# R12.1: the two byte-different qwen AWQ checkpoints, scored side by side. Keyed by
# the recipe hash job 1 actually wrote, because that is what distinguishes them --
# `offload_device` is in the hash precisely so they cannot be confused.
AWQ_REPLICATE_HASHES = {
    "awq-4bit-offload": "e7377d975178f192",   # current study checkpoint, offload_device=cpu
    "awq-4bit-nooffload": "184c722115d8d9bb", # superseded, no offload. Retained on purpose.
}


def frozen_max_num_seqs() -> int:
    """The `A7`-frozen value, or refuse to run.

    🚨 Fails closed. The pilot's generations become the annotation population and the
    throughput baseline for everything after it, so running at a guessed
    `max_num_seqs` and freezing a different one later would silently make them
    incomparable to the main runs — and nothing downstream would reveal it.

    §3.3's literal `8` is a PRE-AUDIT default from vLLM #51187 measured on other
    hardware; `models.yaml` marks it unresolved with
    `max_num_seqs_resolved_by_audit: false`, which is exactly the flag this checks.
    """
    serving = yaml.safe_load((CONFIGS / "models.yaml").read_text())["serving"]
    if not serving.get("max_num_seqs_resolved_by_audit"):
        raise SystemExit(
            "REFUSING TO RUN: configs/models.yaml still has\n"
            "  max_num_seqs_resolved_by_audit: false\n\n"
            "Job 2 (determinism / A7) must freeze max_num_seqs first. Running the "
            "pilot at a guessed value would make its generations -- which become the "
            "annotation population and the throughput baseline -- incomparable to "
            "every main run, with nothing downstream to reveal it.\n"
            "Set the audited value and the flag together, from job 2's "
            "determinism.json."
        )
    return int(serving["max_num_seqs"])


def pilot_plan(tasks: dict) -> list[tuple[str, str, str | None, int]]:
    """`(lm_eval_task, task, subtask, n_items)` for one cell's 300 items."""
    plan: list[tuple[str, str, str | None, int]] = [
        ("gsm8k", "gsm8k", None, PILOT_COMPOSITION["gsm8k"]),
        ("arc_challenge_chat", "arc_challenge_chat", None,
         PILOT_COMPOSITION["arc_challenge_chat"]),
    ]
    subtasks = [s["name"] for s in tasks["bbh"]["subtasks"]]
    base, extra = divmod(PILOT_COMPOSITION["bbh"], len(subtasks))
    prefix = tasks["bbh"]["lm_eval_task_prefix"]
    for index, name in enumerate(subtasks):
        plan.append((f"{prefix}{name}", "bbh", name, base + (1 if index < extra else 0)))
    return plan


def build_cells(models: dict, specs: dict, calibration) -> list[dict]:
    """Every cell this job runs, resolved to concrete weights up front.

    Resolution happens BEFORE any engine loads so a missing checkpoint costs seconds
    rather than surfacing after hours -- the same fail-fast rule job 1 learned when a
    gated repo would have blocked at hour two.
    """
    cells: list[dict] = []
    for model_key, model in models.items():
        for precision in PRECISIONS:
            # ⚠️ `bf16` is ABSENT from specs -- `load_specs` skips method=="none" --
            # so it is handled before any attribute access rather than relying on a
            # short-circuit. Same trap sm_determinism.arm_engine_fields exists for.
            if precision == "bf16":
                cells.append({
                    "cell_id": f"{model_key}|{precision}",
                    "model_key": model_key, "revision": model["revision"],
                    "precision": precision, "quant_method": None, "bits": 16,
                    "model_path": model["repo"], "quant_config_hash": None,
                })
                continue

            spec = specs[precision]
            if spec.in_flight:
                # bnb-NF4 has no checkpoint: vLLM quantizes it at load. Serving the
                # BASE repo with quant_method set is what triggers that.
                path, digest = model["repo"], None
            else:
                digest = config_hash(spec, calibration, model["revision"])
                resolved = output_dir(CHECKPOINT_DIR, model_key, spec, digest)
                if not (resolved / "config.json").exists():
                    raise SystemExit(
                        f"REFUSING TO RUN: no checkpoint for {model_key}/{precision} "
                        f"at {resolved} (recipe hash {digest}). Job 1 must complete "
                        f"and this job's checkpoint_s3_uri must match job 1's."
                    )
                path = str(resolved)
            cells.append({
                "cell_id": f"{model_key}|{precision}",
                "model_key": model_key,
                "revision": model["revision"],
                "precision": precision,
                "quant_method": spec.method,
                "bits": spec.bits,
                "model_path": path,
                "quant_config_hash": digest,
            })

    # R12.1's two extra cells. Distinct `cell_id`s so their records never merge --
    # cell_id is part of JOIN_KEY, so a shared id would silently pool two different
    # weight sets into one accuracy number, which is the exact question being asked.
    spec = specs["awq-4bit"]
    for label, digest in AWQ_REPLICATE_HASHES.items():
        resolved = CHECKPOINT_DIR / f"qwen2.5-3b__awq-4bit__{digest}"
        if not (resolved / "config.json").exists():
            print(f"[warn] R12.1 replicate {label} absent at {resolved.name} — "
                  f"skipping this cell; the accuracy comparison will be incomplete.")
            continue
        cells.append({
            "cell_id": f"qwen2.5-3b|{label}",
            "model_key": "qwen2.5-3b",
            "revision": models["qwen2.5-3b"]["revision"],
            "precision": "awq-4bit",
            "quant_method": spec.method,
            "bits": spec.bits,
            "model_path": str(resolved),
            "quant_config_hash": digest,
            "r12_1_replicate": label,
        })
    return cells


CELL_SCRIPT = r"""
import json, sys, time
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
runner = VLLMRunner(spec)   # preflight() asserts every determinism invariant

# Tokenizer comes from the BASE repo, never the quantized directory: a quantized
# checkpoint's tokenizer files are a copy, and prompts must be byte-identical across
# precisions or the precision contrast is confounded with tokenization.
tokenizer = AutoTokenizer.from_pretrained(cell["base_repo"], revision=cell["revision"])

totals = {"written": 0, "skipped": 0, "total": 0}
for lm_eval_task, task, subtask, limit in payload["plan"]:
    config = tasks[task]
    prompts = list(render_task_prompts(
        lm_eval_task, tokenizer, task=task, subtask=subtask,
        prompt_variant="fewshot_cot", model_key=cell["model_key"],
        model_repo=cell["base_repo"], model_revision=cell["revision"],
        num_fewshot=config.get("num_fewshot"),
        # ⚠️ PER-TEMPLATE-LEVEL, not hardcoded (prereg 3b-bis / R19.1). The pilot passes
        # "chat", which is the primary level; sm_template.py drives both levels through
        # this same script so the two cannot diverge in any other respect.
        apply_chat_template=(payload["template_level"] == "chat"),
        fewshot_as_multiturn=False, limit=limit,
    ))
    # prompt_token_ids carried through so vLLM never re-tokenizes -- text input
    # takes the completion path (add_special_tokens=True) and double-BOSes any model
    # whose chat template already emits one.
    items = [{"item_id": p.item_id, "prompt": p.prompt_text, "gold": p.gold,
              "prompt_token_ids": p.prompt_token_ids}
             for p in prompts]
    started = time.monotonic()
    counts = run_cell(
        runner,
        run_id=payload["run_id"], cell_id=cell["cell_id"], task=task, subtask=subtask,
        items=items, prompt_variant="fewshot_cot", budget_level="1x",
        # The terminator set is part of the DEFINITION of each template level: at `raw`
        # lm-eval's few-shot set is correct by construction, at `chat` it is not. Tasks
        # without a per-level override fall back to their single `until`.
        max_gen_toks=config["max_gen_toks"]["1x"],
        until=(config.get("until_by_template") or {}).get(
            payload["template_level"], config["until"]),
        quant_config_hash=cell["quant_config_hash"],
        output_root=Path(payload["generations_root"]),
    )
    elapsed = time.monotonic() - started
    for key in totals:
        totals[key] += counts[key]
    rate = counts["written"] / elapsed * 3600 if elapsed and counts["written"] else 0
    label = f"{task}/{subtask}" if subtask else task
    print(f"    {label:44} {counts['written']:4} written, {counts['skipped']:4} "
          f"skipped, {elapsed/60:5.1f} min (~{rate:,.0f} gen/h)", flush=True)

Path(payload["out_path"]).write_text(json.dumps(totals))
print(f"__CELL_OK__ {json.dumps(totals)}")
"""


def run_one_cell(cell: dict, plan: list, max_num_seqs: int, base_repo: str) -> dict | None:
    """One cell in its own process.

    Subprocess per cell for the reason job 1 established the hard way: torch's
    caching allocator and the engine's hooks do not reliably release GPU memory
    in-process, and 17 sequential engine loads in one process is the same shape that
    cost four arms there. Process exit is the only teardown that always works.
    """
    out_path = GENERATIONS_ROOT / f".{cell['cell_id'].replace('|', '__')}.counts.json"
    payload = {
        # The pilot is the PRIMARY level only. The template factor's `raw` level is a
        # separate stage (sm_template.py); prereg 3b-bis keeps the annotation pool and
        # H1's budget sweep chat-only.
        "template_level": "chat",
        "cell": {**cell, "base_repo": base_repo},
        "plan": plan,
        "max_num_seqs": max_num_seqs,
        "run_id": RUN_ID,
        "generations_root": str(GENERATIONS_ROOT),
        "tasks_yaml": str(CONFIGS / "tasks.yaml"),
        "out_path": str(out_path),
    }
    payload_path = GENERATIONS_ROOT / f".{cell['cell_id'].replace('|', '__')}.payload.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(payload))

    # Output inherited, not captured: a 40-minute cell would otherwise emit nothing
    # to CloudWatch and look like a hang, and a killed job would lose the child's
    # output entirely.
    sys.stdout.flush()
    result = subprocess.run(
        [sys.executable, "-c", CELL_SCRIPT, str(payload_path)], timeout=21600
    )
    if result.returncode != 0:
        print(f"       FAILED exit {result.returncode} (traceback above)")
        return None
    payload_path.unlink(missing_ok=True)
    return json.loads(out_path.read_text()) if out_path.exists() else None


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GENERATIONS_ROOT.mkdir(parents=True, exist_ok=True)

    models = yaml.safe_load((CONFIGS / "models.yaml").read_text())["models"]
    tasks = yaml.safe_load((CONFIGS / "tasks.yaml").read_text())["tasks"]
    specs, calibration = load_specs(CONFIGS / "quant.yaml")

    max_num_seqs = frozen_max_num_seqs()
    plan = pilot_plan(tasks)

    if SMOKE:
        # 🚨 COVER EVERY PATH THAT CAN FAIL *SILENTLY*, not just one that can crash.
        #
        # The first version of this rehearsal ran qwen/gsm8k/{bf16,gptq-4bit} and was
        # green -- while leaving ARC, all six BBH subtasks, and the two non-Qwen
        # tokenizers uncleared. That asymmetry matters here in a way it does not for
        # `quantize`: per-cell checkpointing makes a CRASH cheap (resubmit, completed
        # cells are skipped), but a wrong `until` or `max_gen_toks` on BBH does not
        # crash -- it silently truncates and yields plausible, wrong generations. This
        # job is the ANNOTATION POOL, so that propagates into ~250 human annotations
        # and into H1's truncation numbers. Checkpointing protects against
        # interruption; only coverage protects against silent wrongness.
        #
        # One cell per distinct code path, 6 items each:
        #   gsm8k                     the answer_marker=hash level
        #   arc_challenge_chat        a different renderer and a different answer space
        #   bbh/<first subtask>       different `until` strings AND 4x generation budget
        plan = [
            ("gsm8k", "gsm8k", None, 6),
            ("arc_challenge_chat", "arc_challenge_chat", None, 6),
        ]
        bbh_subtasks = [s["name"] for s in tasks["bbh"]["subtasks"]]
        prefix = tasks["bbh"]["lm_eval_task_prefix"]
        plan.append((f"{prefix}{bbh_subtasks[0]}", "bbh", bbh_subtasks[0], 6))
        print(f"SMOKE MODE: 3 task paths (gsm8k, arc, bbh/{bbh_subtasks[0]}), "
              f"6 items each")

    cells = build_cells(models, specs, calibration)
    if SMOKE:
        # TOKENIZER coverage, which is the other thing a green Qwen-only run cannot
        # speak to. G2 measured the BOS hazard as model-specific: Qwen has no
        # bos_token, Phi has one but `add_bos_token=False` (zero BOS is CORRECT for
        # it), and Llama-3.1's chat template emits `<|begin_of_text|>` itself -- so
        # Llama is the ONLY model where the naive path double-BOSes. Generation now
        # passes frozen prompt_token_ids, and this is where that assertion gets
        # exercised against all three tokenizers rather than one.
        #
        # ARM coverage: bf16 (no quant path), gptq-4bit (compressed-tensors + Marlin)
        # and awq-4bit (a different checkpoint). `bnb-nf4` is deliberately omitted --
        # its in-flight load path is the one gap already closed elsewhere, since job 2
        # served it successfully across all three batch sizes.
        smoke_arms = ("bf16", "gptq-4bit", "awq-4bit")
        keep = [
            ("qwen2.5-3b", "bf16"), ("qwen2.5-3b", "gptq-4bit"),
            ("qwen2.5-3b", "awq-4bit"),
            ("phi-3.5-mini", "gptq-4bit"),   # zero-BOS tokenizer
            # ⚠️ Llama on its QUANTIZED arm, not bf16, and the reason is the 1800 s
            # smoke cap. The tokenizer always loads from the BASE repo (see the
            # PROBE/cell script: a quantized checkpoint's tokenizer files are a copy,
            # and prompts must be byte-identical across arms), so the BOS-emitting
            # template is exercised either way -- but bf16 would additionally pull
            # ~16 GB of weights from HF, while the quantized arm reads a checkpoint
            # already restored from `checkpoint_s3_uri`. Same coverage, ~16 GB cheaper.
            ("llama-3.1-8b", "gptq-4bit"),
        ]
        cells = [
            c for c in cells
            if c["precision"] in smoke_arms
            and "r12_1_replicate" not in c
            and (c["model_key"], c["precision"]) in keep
        ]
        print(f"SMOKE MODE: {len(cells)} cells across "
              f"{len({c['model_key'] for c in cells})} tokenizers, "
              f"{len({c['precision'] for c in cells})} arms")

    n_items = sum(limit for *_, limit in plan)
    print(f"run_id       : {RUN_ID}")
    print(f"max_num_seqs : {max_num_seqs}  (frozen by A7, not assumed)")
    print(f"cells        : {len(cells)}")
    print(f"items/cell   : {n_items} {PILOT_COMPOSITION if not SMOKE else ''}")
    print(f"generations  : {len(cells) * n_items:,}")
    print(f"output       : {GENERATIONS_ROOT}")
    print()
    for cell in cells:
        tag = f" [R12.1 {cell['r12_1_replicate']}]" if "r12_1_replicate" in cell else ""
        print(f"  {cell['cell_id']:34} {Path(cell['model_path']).name}{tag}")
    print()

    report: dict = {"stage": "pilot", "run_id": RUN_ID, "smoke": SMOKE,
                    "max_num_seqs": max_num_seqs, "cells": []}
    failed: list[str] = []

    for index, cell in enumerate(cells, start=1):
        print(f"=== cell {index}/{len(cells)}: {cell['cell_id']} ===")
        started = time.monotonic()
        counts = run_one_cell(cell, plan, max_num_seqs, models[cell["model_key"]]["repo"])
        elapsed = time.monotonic() - started
        if counts is None:
            failed.append(cell["cell_id"])
            report["cells"].append({**{k: cell[k] for k in ("cell_id", "precision")},
                                    "error": "cell subprocess failed"})
            continue
        rate = counts["written"] / elapsed * 3600 if elapsed and counts["written"] else 0
        print(f"       {counts['written']} written, {counts['skipped']} skipped, "
              f"{elapsed / 60:.1f} min (~{rate:,.0f} gen/h)")
        report["cells"].append({
            **{k: cell[k] for k in ("cell_id", "precision", "quant_config_hash")},
            **counts, "minutes": round(elapsed / 60, 1),
            "generations_per_hour": round(rate),
        })
        print()

    (OUTPUT_DIR / "pilot_report.json").write_text(json.dumps(report, indent=2))

    done = [c for c in report["cells"] if "error" not in c]
    written = sum(c.get("written", 0) for c in done)
    print("=" * 70)
    print(f"cells        : {len(done)}/{len(cells)}")
    print(f"generations  : {written:,} written")
    rates = [c["generations_per_hour"] for c in done if c.get("generations_per_hour")]
    if rates:
        # The measured number every downstream estimate has been guessing at, and
        # A6's deliverable (R13.5).
        print(f"throughput   : {min(rates):,}-{max(rates):,} gen/h across cells "
              f"(median {sorted(rates)[len(rates) // 2]:,})")
    if failed:
        print(f"FAILED       : {failed}")
    print(f"report -> {OUTPUT_DIR / 'pilot_report.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
