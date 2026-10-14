#!/usr/bin/env python3
"""SageMaker entry point, `h6` stage. **H6's log-likelihood arm** (prereg §3c, R39.4/R24.2).

🚨 **WHY THIS FILE EXISTS.** H6 is marked *"Never cut"* in `EXECUTOR-TASKS.md` §2.1 because it
is the study's **negative control**: it answers *"extraction fragility is everywhere, so what"*
by showing the effect vanishes where the answer space is structurally constrained. Without it
the paper measures an extraction effect and has nothing establishing where that effect stops.

The arm did not exist, and not for want of a flag. `run_cell` is generation-only: its
`scoring_mode` is fixed to `"generate"` and `option_logprobs` is always `None`. H6 needs the
**same prompt** scored by log-likelihood over the option letters rather than generated and
parsed, which is a different engine path. `VLLMRunner.score_options` is that path and this
stage drives it.

**The pairing, and why it is the control.** Arm 1 (`arc_challenge_chat`, already in `main-v1`)
shows the model the chat prompt and parses a letter out of free text. Arm 2, here, shows it
**the identical prompt** and instead scores `" A"`, `" B"`, `" C"`, `" D"` by log-likelihood,
taking the argmax. Same items, same prompt, same engine, same weights: the only thing that
changes is whether an extraction step exists at all. The gap between the two arms **is** the
extraction cost, measured rather than inferred.

⚠️ **Arm 3 is NOT built here, deliberately.** §3c registers three arms, and R39.4 scoped the
build to *"the 0-shot log-likelihood against 0-shot generative pairing, 9 cells"*, which is
arms 1 and 2. Arm 3 is lm-eval's native `arc_challenge`, a different prompt scoring full answer
*texts* rather than letters, so it is multi-token and not prompt-matched. It would answer a
different question and is left unbuilt rather than half-built.
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

#: Its own `run_id`. These records carry `scoring_mode="loglikelihood"` and populated
#: `option_logprobs`, which no `main-v1` record does, so pooling them under one id would mix
#: two measurement modes into a population every `--run-id` consumer reads as homogeneous.
RUN_ID = "h6-v1-smoke" if SMOKE else "h6-v1"
GENERATIONS_ROOT = CHECKPOINT_DIR / "generations"

#: §2.2's three contrasted arms. `bnb-nf4` and `gptq-8bit` are excluded from the expensive
#: arms, and H6 is a contrast rather than a dose-response.
H6_ARMS = ("bf16", "gptq-4bit", "awq-4bit")


def h6_config() -> dict:
    """`arc_challenge_ll_matched` from the config, with its registered invariants asserted."""
    tasks = yaml.safe_load((CONFIGS / "tasks.yaml").read_text())["tasks"]
    cfg = tasks["arc_challenge_ll_matched"]
    # The arm is defined by sharing arm 1's prompt. If that ever stops being true the two
    # arms are no longer prompt-matched and the gap between them stops being the extraction
    # cost, so it is asserted rather than assumed.
    if cfg.get("prompt_source") != "arc_challenge_chat":
        raise SystemExit(
            f"arc_challenge_ll_matched.prompt_source is {cfg.get('prompt_source')!r}, not "
            f"'arc_challenge_chat'. H6's whole design is that the two arms share one prompt."
        )
    if cfg.get("scoring_mode") != "loglikelihood":
        raise SystemExit("arc_challenge_ll_matched must declare scoring_mode=loglikelihood")
    if not cfg.get("continuations"):
        raise SystemExit("arc_challenge_ll_matched declares no continuations to score")
    return cfg


CELL_SCRIPT = r'''
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, "/opt/ml/code/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

payload = json.loads(Path(sys.argv[1]).read_text())
cell, cfg = payload["cell"], payload["cfg"]

from transformers import AutoTokenizer
from quantartifact.generate import EngineSpec, VLLMRunner, preflight
from quantartifact.records import GenerationRecord, append_records, cell_path, env_hash
from quantartifact.tasks.materialize import render_task_prompts

preflight()

tokenizer = AutoTokenizer.from_pretrained(cell["base_repo"], revision=cell["revision"])

# ⚠️ Rendered through arm 1's task name, which is what makes the two arms prompt-matched.
prompts = list(render_task_prompts(
    cfg["prompt_source"], tokenizer,
    task="arc_challenge_ll_matched", subtask=None,
    prompt_variant="fewshot_cot", model_key=cell["model_key"],
    model_repo=cell["base_repo"], model_revision=cell["revision"],
    num_fewshot=cfg.get("num_fewshot", 0),
    apply_chat_template=True, fewshot_as_multiturn=False,
    limit=payload["n_items"],
))

spec = EngineSpec(
    model=cell["model_path"], revision=cell["revision"],
    max_num_seqs=payload["max_num_seqs"],
    quantization=cell.get("quant_method"),
)
runner = VLLMRunner(spec)

options = list(cfg["continuations"])
scores = runner.score_options([p.prompt_text for p in prompts], options)

# §3c asserts one token per option. Checked on the real tokenizer rather than trusted,
# because a multi-token option would make the summed logprobs length-biased.
if cfg.get("assert_single_token_per_option"):
    bad = {o: len(tokenizer.encode(o, add_special_tokens=False)) for o in options}
    if any(n != 1 for n in bad.values()):
        raise SystemExit(f"assert_single_token_per_option FAILED: {bad}")

records, started = [], time.monotonic()
for prompt, score in zip(prompts, scores, strict=True):
    # argmax over the option logprobs IS the prediction. No extraction step exists, which
    # is the entire point of this arm.
    predicted = max(score, key=score.get).strip()
    records.append(GenerationRecord(
        run_id=payload["run_id"], cell_id=cell["cell_id"],
        model_repo=cell["base_repo"], model_revision=cell["revision"],
        quant_method=cell.get("quant_method"), bits=cell.get("bits"),
        quant_config_hash=cell.get("quant_config_hash"),
        task="arc_challenge_ll_matched", subtask=None, item_id=prompt.item_id,
        gold=prompt.gold, prompt_variant="fewshot_cot",
        scoring_mode="loglikelihood",
        option_logprobs={k.strip(): v for k, v in score.items()},
        generation_text=predicted,
        budget_level="1x", max_gen_toks=None,
        decoding="greedy", sample_idx=0, seed=0, repeat_idx=0,
        prompt_text=prompt.prompt_text, prompt_sha256=prompt.prompt_sha256,
        n_prompt_tokens=len(prompt.prompt_token_ids or []),
        finish_reason=None, n_gen_tokens=None, stop_reason=None,
        engine="vllm==0.26.0", batch_invariant=os.environ.get("VLLM_BATCH_INVARIANT") == "1",
        max_num_seqs=payload["max_num_seqs"],
        disabled_kernels=spec.disabled_kernels,
        # Read off the running box rather than copied from config: these four are the
        # reproducibility envelope the paper reports, and a hardcoded value would survive a
        # hardware change silently.
        gpu=runner.hardware["gpu"], cuda=runner.hardware["cuda"],
        driver=runner.hardware["driver"],
        env_hash=env_hash(), invocation_id=runner.invocation_id,
        template_level="chat", until=None, until_applied=None,
        stop_string_hit=None, n_gen_tokens_scored=None,
        wall_ms=int((time.monotonic() - started) * 1000),
    ))

path = cell_path(Path(payload["generations_root"]), payload["run_id"], cell["cell_id"],
                 "arc_challenge_ll_matched", None, "fewshot_cot", "1x")
written = append_records(path, records)
Path(payload["out_path"]).write_text(json.dumps({"written": written,
                                                 "total": len(records)}))
print(f"__CELL_OK__ {json.dumps({'written': written, 'total': len(records)})}")
'''


def main() -> int:
    sys.path.insert(0, str(REPO))
    from sm_reference import build_cells, frozen_max_num_seqs  # noqa: PLC0415

    from quantartifact.quantize import load_specs  # noqa: PLC0415

    models = yaml.safe_load((CONFIGS / "models.yaml").read_text())["models"]
    specs, calibration = load_specs(CONFIGS / "quant.yaml")
    cfg = h6_config()
    max_num_seqs = frozen_max_num_seqs()

    cells = [c for c in build_cells(models, specs, calibration)
             if c["precision"] in H6_ARMS]
    n_items = 20 if SMOKE else cfg["n_items"]
    if SMOKE:
        cells = cells[:1]

    print(f"run_id       : {RUN_ID}")
    print(f"max_num_seqs : {max_num_seqs}  (frozen by A7)")
    print(f"cells        : {len(cells)}  x {n_items} items x "
          f"{len(cfg['continuations'])} options")
    print(f"continuations: {cfg['continuations']}")
    for c in cells:
        print(f"  {c['cell_id']:30} {Path(c['model_path']).name}")

    report: dict = {"stage": "h6", "run_id": RUN_ID, "smoke": SMOKE,
                    "max_num_seqs": max_num_seqs, "n_items": n_items, "cells": []}
    scratch = Path(os.environ.get("SM_SCRATCH", "/tmp"))
    failed = []

    for i, cell in enumerate(cells, 1):
        print(f"\n=== cell {i}/{len(cells)}: {cell['cell_id']} ===", flush=True)
        payload_path = scratch / f"h6_payload_{cell['cell_id'].replace('|', '_')}.json"
        out_path = scratch / f"h6_counts_{cell['cell_id'].replace('|', '_')}.json"
        payload_path.write_text(json.dumps({
            "cell": {**cell, "base_repo": models[cell["model_key"]]["repo"]},
            "cfg": cfg, "run_id": RUN_ID, "n_items": n_items,
            "max_num_seqs": max_num_seqs,
            "generations_root": str(GENERATIONS_ROOT), "out_path": str(out_path),
        }))
        script = scratch / "h6_cell.py"
        script.write_text(CELL_SCRIPT)
        started = time.monotonic()
        proc = subprocess.run([sys.executable, str(script), str(payload_path)],
                              timeout=7200)
        if proc.returncode != 0 or not out_path.exists():
            failed.append(cell["cell_id"])
            report["cells"].append({"cell_id": cell["cell_id"], "error": "subprocess failed"})
            continue
        counts = json.loads(out_path.read_text())
        report["cells"].append({"cell_id": cell["cell_id"], **counts,
                                "minutes": (time.monotonic() - started) / 60})
        print(f"       {counts['written']} written of {counts['total']}, "
              f"{(time.monotonic() - started) / 60:.1f} min", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "h6_report.json").write_text(json.dumps(report, indent=2))
    done = [c for c in report["cells"] if "error" not in c]
    print(f"\ncells        : {len(done)}/{len(cells)}")
    print(f"records      : {sum(c.get('written', 0) for c in done):,}")
    print(f"report -> {OUTPUT_DIR / 'h6_report.json'}")
    print(f"NEXT: .venv/bin/python scripts/peek_generations.py --run-id {RUN_ID}")
    if failed:
        # Every cell is load-bearing: H6 is a 3x3 contrast and a missing cell removes a
        # model or an arm from it rather than degrading precision.
        print(f"\nFAILED CELLS (all load-bearing for a 3x3 contrast): {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
