#!/usr/bin/env python3
"""SageMaker entry point, `h6` stage. **H6's log-likelihood arm** (prereg §3c, R39.4/R24.2).

H6 is marked *"Never cut"* because it is the study's **negative control**: it shows the
extraction effect vanishes where the answer space is structurally constrained. Arm 1
(`arc_challenge_chat`, already in `main-v1`) generates and parses a letter. Arm 2, here, scores
`" A".." D"` by log-likelihood on **the identical prompt**. Same items, prompt, engine and
weights; only the extraction step differs, so the gap between arms **is** the extraction cost.

Arm 3 (lm-eval's native `arc_challenge`) is deliberately not built: R39.4 scoped the build to
the arm 1/arm 2 pairing, and arm 3 scores full answer texts on a different prompt.

🚨 **The cell logic is a real function, `cell_main`, not a string.** The first version put it
in a `CELL_SCRIPT` string like the pilot's, which no linter or test can see, and the smoke died
on four API mistakes in it (`preflight()` with no argument, three wrong `EngineSpec` field
names, `cell_path` arguments swapped). Each subprocess now runs this same file with `--cell`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

CONFIGS = Path(os.environ.get("QA_CONFIGS", REPO / "configs"))
CHECKPOINT_DIR = Path(os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints"))
OUTPUT_DIR = Path(os.environ.get("SM_OUTPUT_DIR", "/opt/ml/model"))

SMOKE = os.environ.get("SM_HP_SMOKE") == "1" or "--smoke" in sys.argv

#: Its own `run_id`: these records are `scoring_mode="loglikelihood"`, which no `main-v1`
#: record is, so pooling them would mix two measurement modes under one id.
RUN_ID = "h6-v1-smoke" if SMOKE else "h6-v1"
GENERATIONS_ROOT = CHECKPOINT_DIR / "generations"
TASK = "arc_challenge_ll_matched"
H6_ARMS = ("bf16", "gptq-4bit", "awq-4bit")


def h6_config(tasks: dict) -> dict:
    """Arm 2's config merged with what it inherits from arm 1, invariants asserted."""
    cfg = dict(tasks[TASK])
    if cfg.get("prompt_source") != "arc_challenge_chat":
        raise SystemExit(f"{TASK}.prompt_source must be arc_challenge_chat; sharing arm 1's "
                         f"prompt is what makes the arm gap the extraction cost")
    if cfg.get("scoring_mode") != "loglikelihood" or not cfg.get("continuations"):
        raise SystemExit(f"{TASK} must declare scoring_mode=loglikelihood and continuations")
    arm1 = tasks["arc_challenge_chat"]
    # Inherited from arm 1, not restated, so the two arms cannot drift apart.
    cfg["gen_prefix"] = arm1.get("gen_prefix") or ""
    cfg["num_fewshot"] = arm1.get("num_fewshot", 0)
    if not cfg["gen_prefix"]:
        raise SystemExit("arc_challenge_chat has no gen_prefix; scoring a letter straight "
                         "after the assistant turn would measure the wrong token")
    return cfg


def build_records(payload: dict, prompts: list, scores: list, runner, spec) -> list:
    """One `GenerationRecord` per item. Pure, so it is testable without a GPU."""
    from quantartifact.records import GenerationRecord, env_hash  # noqa: PLC0415

    cell, cfg = payload["cell"], payload["cfg"]
    records = []
    for prompt, per in zip(prompts, scores, strict=True):
        if cfg.get("assert_single_token_per_option"):
            multi = {o: v["n_tokens"] for o, v in per.items() if v["n_tokens"] != 1}
            if multi:
                # A multi-token option makes summed logprobs length-biased, so §3c asserts
                # one token each. Checked in context on the real tokenizer, not standalone.
                raise SystemExit(f"assert_single_token_per_option FAILED: {multi}")
        # EXECUTOR-TASKS SS7.1: "per-option logprob + length", as a LIST in option order.
        # Length is recorded so acc_norm is computable later without a GPU.
        option_logprobs = [{"option": o.strip(), "logprob": v["logprob"],
                            "n_tokens": v["n_tokens"]} for o, v in per.items()]
        # The argmax IS the prediction. There is no extraction step, which is the point.
        predicted = max(option_logprobs, key=lambda x: x["logprob"])["option"]
        records.append(GenerationRecord(
            run_id=payload["run_id"], cell_id=cell["cell_id"],
            model_repo=cell["base_repo"], model_revision=cell["revision"],
            quant_method=cell["quant_method"], bits=cell["bits"],
            quant_config_hash=cell["quant_config_hash"],
            task=TASK, subtask=None, item_id=prompt.item_id, gold=prompt.gold,
            prompt_variant="fewshot_cot", scoring_mode="loglikelihood",
            budget_level="1x", max_gen_toks=None,
            decoding="greedy", sample_idx=0, seed=0, repeat_idx=0,
            prompt_text=prompt.prompt_text, prompt_sha256=prompt.prompt_sha256,
            generation_text=predicted, n_prompt_tokens=prompt.n_prompt_tokens,
            n_gen_tokens=None, finish_reason=None, stop_reason=None,
            option_logprobs=option_logprobs,
            engine="vllm==0.26.0",
            batch_invariant=os.environ.get("VLLM_BATCH_INVARIANT") == "1",
            max_num_seqs=spec.max_num_seqs, disabled_kernels=spec.disabled_kernels,
            # Read off the running box: these four are the reproducibility envelope.
            gpu=runner.hardware["gpu"], cuda=runner.hardware["cuda"],
            driver=runner.hardware["driver"],
            env_hash=env_hash(), invocation_id=runner.invocation_id,
            template_level="chat",
            wall_ms=0,
        ))
    return records


def cell_main(payload_path: str, runner_cls=None) -> int:
    """One cell, in its own process. `runner_cls` is injectable so a test can fake the GPU."""
    from transformers import AutoTokenizer  # noqa: PLC0415 - harness/serving only

    from quantartifact.generate import EngineSpec, VLLMRunner  # noqa: PLC0415
    from quantartifact.records import append_records, cell_path  # noqa: PLC0415
    from quantartifact.tasks.materialize import render_task_prompts  # noqa: PLC0415

    payload = json.loads(Path(payload_path).read_text())
    cell, cfg = payload["cell"], payload["cfg"]

    spec = EngineSpec(model_repo=cell["model_path"], model_revision=cell["revision"],
                      quant_method=cell["quant_method"], bits=cell["bits"],
                      max_num_seqs=payload["max_num_seqs"])
    runner = (runner_cls or VLLMRunner)(spec)   # preflight(spec) runs inside

    # Base-repo tokenizer, rendered through ARM 1's task name: that is what makes the arms
    # prompt-matched.
    tokenizer = AutoTokenizer.from_pretrained(cell["base_repo"], revision=cell["revision"])
    prompts = list(render_task_prompts(
        cfg["prompt_source"], tokenizer, task=cfg["prompt_source"], subtask=None,
        prompt_variant="fewshot_cot", model_key=cell["model_key"],
        model_repo=cell["base_repo"], model_revision=cell["revision"],
        num_fewshot=cfg["num_fewshot"], apply_chat_template=True,
        fewshot_as_multiturn=False, limit=payload["n_items"],
    ))

    started = time.monotonic()
    scores = runner.score_options([p.prompt_text for p in prompts],
                                  [p.prompt_token_ids for p in prompts],
                                  cfg["gen_prefix"], list(cfg["continuations"]))
    records = build_records(payload, prompts, scores, runner, spec)

    path = cell_path(Path(payload["generations_root"]), payload["run_id"], cell["cell_id"],
                     TASK, None, "1x", "fewshot_cot")
    written = append_records(path, records)
    counts = {"written": written, "total": len(records),
              "seconds": round(time.monotonic() - started, 1)}
    Path(payload["out_path"]).write_text(json.dumps(counts))
    print(f"__CELL_OK__ {json.dumps(counts)}", flush=True)
    return 0


def main() -> int:
    sys.path.insert(0, str(REPO))
    from sm_pilot import frozen_max_num_seqs  # noqa: PLC0415
    from sm_reference import build_cells  # noqa: PLC0415

    from quantartifact.quantize import load_specs  # noqa: PLC0415

    models = yaml.safe_load((CONFIGS / "models.yaml").read_text())["models"]
    cfg = h6_config(yaml.safe_load((CONFIGS / "tasks.yaml").read_text())["tasks"])
    specs, calibration = load_specs(CONFIGS / "quant.yaml")
    max_num_seqs = frozen_max_num_seqs()

    cells = [c for c in build_cells(models, specs, calibration) if c["precision"] in H6_ARMS]
    n_items = 20 if SMOKE else cfg["n_items"]
    if SMOKE:
        # One cell PER MODEL, not one cell: the in-context tokenisation differs per tokenizer
        # (Phi is SentencePiece, Qwen and Llama are BPE), and that is the part most likely to
        # fail. A Qwen-only rehearsal would clear nothing about the other two.
        cells = [c for c in cells if c["precision"] == "bf16"]

    print(f"run_id       : {RUN_ID}")
    print(f"max_num_seqs : {max_num_seqs}  (frozen by A7)")
    print(f"gen_prefix   : {cfg['gen_prefix']!r}  (context before the scored option)")
    print(f"cells        : {len(cells)} x {n_items} items x {len(cfg['continuations'])} options")

    report: dict = {"stage": "h6", "run_id": RUN_ID, "smoke": SMOKE, "n_items": n_items,
                    "max_num_seqs": max_num_seqs, "gen_prefix": cfg["gen_prefix"], "cells": []}
    scratch = Path(os.environ.get("SM_SCRATCH", "/tmp"))
    failed = []
    for i, cell in enumerate(cells, 1):
        print(f"\n=== cell {i}/{len(cells)}: {cell['cell_id']} ===", flush=True)
        tag = cell["cell_id"].replace("|", "_")
        payload_path, out_path = scratch / f"h6_{tag}.json", scratch / f"h6_{tag}_out.json"
        payload_path.write_text(json.dumps({
            "cell": {**cell, "base_repo": models[cell["model_key"]]["repo"]},
            "cfg": cfg, "run_id": RUN_ID, "n_items": n_items, "max_num_seqs": max_num_seqs,
            "generations_root": str(GENERATIONS_ROOT), "out_path": str(out_path),
        }))
        proc = subprocess.run([sys.executable, __file__, "--cell", str(payload_path)],
                              timeout=7200)
        if proc.returncode != 0 or not out_path.exists():
            failed.append(cell["cell_id"])
            report["cells"].append({"cell_id": cell["cell_id"], "error": "subprocess failed"})
            continue
        counts = json.loads(out_path.read_text())
        report["cells"].append({"cell_id": cell["cell_id"], **counts})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "h6_report.json").write_text(json.dumps(report, indent=2))
    done = [c for c in report["cells"] if "error" not in c]
    print(f"\ncells   : {len(done)}/{len(cells)}")
    print(f"records : {sum(c.get('written', 0) for c in done):,}")
    if failed:
        # All load-bearing: H6 is a 3x3 contrast, and a missing cell removes a model or arm.
        print(f"FAILED CELLS: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    if "--cell" in sys.argv:
        raise SystemExit(cell_main(sys.argv[sys.argv.index("--cell") + 1]))
    raise SystemExit(main())
