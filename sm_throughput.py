#!/usr/bin/env python3
"""SageMaker entry point — `throughput` stage. Task `A6`, the tight estimate.

Two numbers the whole schedule rests on, neither of which can be reasoned out:

1. **Measured generations/hour on THIS hardware**, replacing the §2.6 table's
   A100-based estimate. `EXECUTOR-TASKS.md` §9 trigger 3 fires if it pushes the
   reference-run block past **Sep 13**, so this is a schedule instrument, not a
   curiosity.
2. **The honest batch-invariance overhead**, measured WITH and WITHOUT
   `VLLM_BATCH_INVARIANT=1`. §3.3 budgets 1.2–2× but that is an **Ampere** figure;
   ours is L40S/SM89 (Ada), where it is unmeasured. This is also the number the
   paper reports as the cost of determinism.

⚠️ **The BI-off run is a MEASUREMENT ONLY.** Its generations are discarded and
must never enter the study: §3.3 forbids reusing a BI-mode timing as a quantized
throughput number, and the converse holds too — a non-BI generation is not
comparable to anything else we produce. `records.GenerationRecord.validate()`
refuses `batch_invariant=False`, so this cannot leak by accident.

Runs bf16 only. Quantized throughput needs the weights from the `quantize` stage,
and mixing the two would conflate engine overhead with kernel differences.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

MODEL = os.environ.get("QA_THROUGHPUT_MODEL", "Qwen/Qwen2.5-3B-Instruct")
REVISION = os.environ.get("QA_THROUGHPUT_REVISION", "aa8e72537993ba99e69dfaafa59ed015b17504d1")
N_PROMPTS = int(os.environ.get("QA_THROUGHPUT_N", "100"))
MAX_GEN_TOKS = int(os.environ.get("QA_THROUGHPUT_MAX_TOKS", "256"))

# Frozen serving config (§3.3). Identical in both passes except VLLM_BATCH_INVARIANT,
# so the difference is attributable to BI alone.
ENGINE_KWARGS = {
    "tensor_parallel_size": 1,        # REQUIRED — TP>1 breaks BI (vLLM #51290)
    "enable_prefix_caching": False,   # breaks greedy determinism (vLLM #39389)
    "max_model_len": 8192,
    "max_num_seqs": 8,
    "gpu_memory_utilization": 0.90,
    "seed": 0,
    "enforce_eager": False,           # CUDA graphs are BI-compatible since PR #27660
    "dtype": "bfloat16",
}


def run_pass(batch_invariant: bool) -> dict:
    """One timed pass in a SUBPROCESS.

    Subprocess because `VLLM_BATCH_INVARIANT` is read at import/engine-init time;
    flipping it in-process after vLLM has loaded would measure nothing.
    """
    script = f'''
import json, os, time
os.environ["VLLM_BATCH_INVARIANT"] = {"1" if batch_invariant else "0"!r}
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm import LLM, SamplingParams

kwargs = {ENGINE_KWARGS!r}
t0 = time.monotonic()
llm = LLM(model={MODEL!r}, revision={REVISION!r}, **kwargs)
load_s = time.monotonic() - t0

# Prompts long enough to be decode-bound rather than prefill-bound, which is the
# regime the real runs sit in at max_num_seqs=8.
prompts = [
    "Q: A shop sells {{n}} apples at 3 dollars each and {{n}} pears at 2 dollars. "
    "Work through the arithmetic step by step, then state the total.\\nA:".format(n=i + 2)
    for i in range({N_PROMPTS})
]
params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens={MAX_GEN_TOKS}, seed=0, n=1)

t0 = time.monotonic()
outs = llm.generate(prompts, params, use_tqdm=False)
gen_s = time.monotonic() - t0

n_out = sum(len(o.outputs[0].token_ids) for o in outs)
finish = {{}}
for o in outs:
    r = o.outputs[0].finish_reason
    finish[r] = finish.get(r, 0) + 1

print("QA_RESULT_JSON" + json.dumps({{
    "batch_invariant": {batch_invariant!r},
    "n_prompts": len(outs),
    "load_seconds": round(load_s, 2),
    "generate_seconds": round(gen_s, 2),
    "output_tokens": n_out,
    "output_tokens_per_second": round(n_out / gen_s, 1) if gen_s else None,
    "generations_per_hour": round(len(outs) / gen_s * 3600) if gen_s else None,
    "finish_reasons": finish,
}}))
'''
    label = "ON" if batch_invariant else "OFF"
    print(f"\n=== pass: VLLM_BATCH_INVARIANT={label} ===")
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=5400
    )
    elapsed = time.monotonic() - started

    for line in (result.stdout or "").splitlines():
        if line.startswith("QA_RESULT_JSON"):
            payload = json.loads(line[len("QA_RESULT_JSON"):])
            payload["wall_seconds"] = round(elapsed, 2)
            print(f"  load {payload['load_seconds']}s, generate "
                  f"{payload['generate_seconds']}s, "
                  f"{payload['output_tokens_per_second']} out-tok/s, "
                  f"{payload['generations_per_hour']} gen/hr")
            return payload

    return {
        "batch_invariant": batch_invariant,
        "error": True,
        "returncode": result.returncode,
        "tail": (result.stdout + result.stderr)[-2000:],
        "wall_seconds": round(elapsed, 2),
    }


report: dict = {
    "stage": "throughput",
    "model": MODEL,
    "revision": REVISION,
    "n_prompts": N_PROMPTS,
    "max_gen_toks": MAX_GEN_TOKS,
    "engine_kwargs": ENGINE_KWARGS,
    "passes": [],
}

# BI ON first: it is the configuration the study actually uses, so if only one
# pass survives the time budget it should be the one that matters.
report["passes"].append(run_pass(True))
report["passes"].append(run_pass(False))

on = next((p for p in report["passes"] if p.get("batch_invariant") and not p.get("error")), None)
off = next((p for p in report["passes"] if not p.get("batch_invariant") and not p.get("error")), None)

if on and off and on.get("generate_seconds") and off.get("generate_seconds"):
    overhead = on["generate_seconds"] / off["generate_seconds"]
    report["bi_overhead_ratio"] = round(overhead, 3)
    report["bi_overhead_vs_ampere_budget"] = (
        "within the §3.3 1.2–2x Ampere budget" if 1.0 <= overhead <= 2.0
        else "OUTSIDE the §3.3 1.2–2x budget — this is an SM89 finding, report it"
    )

# Project the whole study from the measured rate. 133,500 generations is the
# revised total (§2.6 superseded by planner ruling R6 plus R9's matched arm).
if on and on.get("generations_per_hour"):
    per_hour = on["generations_per_hour"]
    total = 133_500
    hours = total / per_hour
    report["projection"] = {
        "measured_generations_per_hour_bi_on": per_hour,
        "total_generations_planned": total,
        "gpu_hours_required": round(hours, 1),
        "days_at_20h_per_day": round(hours / 20, 1),
        "usd_at_2_8035_per_hour": round(hours * 2.8035, 2),
        "caveat": (
            "extrapolated from bf16 on a 3B model with 256-token outputs. BBH at 4x "
            "budget generates up to 4096 tokens, so the real total will be higher. "
            "Treat as a lower bound."
        ),
    }

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "throughput_report.json").write_text(json.dumps(report, indent=2))

print("\n" + "=" * 66)
if "bi_overhead_ratio" in report:
    print(f"BI overhead: {report['bi_overhead_ratio']}x  "
          f"({report['bi_overhead_vs_ampere_budget']})")
if "projection" in report:
    proj = report["projection"]
    print(f"projection : {proj['gpu_hours_required']} GPU-hours for "
          f"{proj['total_generations_planned']:,} generations "
          f"(~{proj['days_at_20h_per_day']} days, ~${proj['usd_at_2_8035_per_hour']})")
    print("             LOWER BOUND — see caveat in the report")
print(f"report -> {OUTPUT_DIR / 'throughput_report.json'}")

failed = [p for p in report["passes"] if p.get("error")]
if failed:
    print(f"\n{len(failed)} pass(es) FAILED:")
    for pass_ in failed:
        print(f"  BI={pass_['batch_invariant']} rc={pass_['returncode']}")
        print("  " + pass_["tail"][-600:].replace("\n", "\n  "))

# A failed BI-ON pass is fatal: without it there is no usable estimate and no
# overhead number. A failed BI-OFF pass only costs the overhead ratio.
sys.exit(0 if on else 1)
