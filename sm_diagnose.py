#!/usr/bin/env python3
"""SageMaker entry point — `diagnose` stage. ONE cheap job, TWO open questions.

Run this before re-running anything expensive. It is not a rehearsal; it is a measurement
designed to answer two questions that are currently open, and to answer them from data
rather than from reasoning about vLLM internals.

---

## Q1. Was the OLD configuration silently applying `repetition_penalty` / `top_k`?

vLLM's `generation_config` defaults to `"auto"`, which reads the model's own
`generation_config.json`. Qwen2.5-3B-Instruct ships `repetition_penalty 1.05` and
`top_k 20`; Phi-3.5-mini ships nothing. We passed `temperature=0.0` and `top_p=1.0`
explicitly but never the other fields, so a leaked penalty would have modified the logits
**before** the argmax and changed greedy output on Qwen alone.

🎯 **Qwen is the perfect probe, and this is a natural experiment rather than a contrived
one.** Qwen showed **0/100 divergence across invocations** in every one of its seven cells,
so it has no invocation noise to confound the comparison. Therefore:

    regenerate Qwen's stored `pilot-v2` items under the NEW pinned configuration
      -> IDENTICAL  => the old config was already neutral; all prior Qwen data is fine
      -> DIFFERENT  => the old config was applying something; prior numbers are affected

No other model can answer this, because for llama and phi a difference would be
indistinguishable from their own cross-invocation churn.

## Q2. Does the cross-invocation nondeterminism SURVIVE the new configuration?

Same cell, same items, **twice in one job** — two separate engine loads, contemporaneous,
so driver and software drift cannot be the explanation. `llama-3.1-8b/bf16` is the probe
because it is the worst affected (41/100) and because it is `bf16`, which rules out the
quantized-kernel mechanism by construction. Qwen bf16 runs the same way as the negative
control.

    llama differs again => the effect is not the sampling config; it is still open, and the
                           K=2 design in prereg §9f is required
    llama identical     => the sampling config WAS the cause, and §9f can be relaxed

Either answer is worth far more than the ~$0.50 it costs, and both are needed before
committing ~$17 of re-runs and 10.5 h of `main-v1`.

⚠️ The engine read-back now records the CUDA-graph capture shape set (R22), so this job also
banks that field for both invocations whatever the outcome.
"""

from __future__ import annotations

import gzip
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
GENERATIONS_ROOT = CHECKPOINT_DIR / "generations"

# Compared AGAINST this stored run. Not written to.
BASELINE_RUN = "pilot-v2"

# Small: this is a diagnostic, and the questions are answered by identity/difference rather
# than by an accuracy estimate, so precision in n buys nothing.
N_ITEMS = 40

# (model, arm, n_invocations). Two invocations where the question is reproducibility.
PROBES: tuple[tuple[str, str, int], ...] = (
    ("qwen2.5-3b", "bf16", 1),       # Q1: the clean probe -- zero invocation noise
    ("qwen2.5-3b", "gptq-4bit", 1),  # Q1: and on a quantized arm
    ("llama-3.1-8b", "bf16", 2),     # Q2: worst affected, and bf16 rules out the kernel path
    ("qwen2.5-3b", "bf16", 2),       # Q2: negative control, same treatment
)


def resolve(model_key: str, arm: str, models: dict, specs: dict, calibration):
    model = models[model_key]
    if arm == "bf16":
        return model["repo"], None
    spec = specs[arm]
    if spec.in_flight:
        return model["repo"], None
    digest = config_hash(spec, calibration, model["revision"])
    resolved = output_dir(CHECKPOINT_DIR, model_key, spec, digest)
    if not (resolved / "config.json").exists():
        raise SystemExit(f"REFUSING TO RUN: no checkpoint at {resolved}")
    return str(resolved), digest


def baseline_items(model_key: str, arm: str) -> list[str]:
    """The first N item_ids of the stored baseline cell, so the comparison is like-for-like."""
    path = (GENERATIONS_ROOT / BASELINE_RUN / f"{model_key}__{arm}"
            / "gsm8k__fewshot_cot__1x.jsonl.gz")
    if not path.exists():
        raise SystemExit(f"no baseline at {path} — cannot answer Q1 without it")
    ids = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            ids.append(json.loads(line)["item_id"])
            if len(ids) >= N_ITEMS:
                break
    return ids


def stored_text(run: str, model_key: str, arm: str) -> dict[str, str]:
    path = (GENERATIONS_ROOT / run / f"{model_key}__{arm}"
            / "gsm8k__fewshot_cot__1x.jsonl.gz")
    if not path.exists():
        return {}
    with gzip.open(path, "rt") as handle:
        return {r["item_id"]: r["generation_text"]
                for r in (json.loads(line) for line in handle)}


def run_invocation(model_key: str, arm: str, run_id: str, path: str, digest, model: dict,
                   max_num_seqs: int, item_ids: list[str]) -> dict | None:
    tag = f"{model_key}__{arm}__{run_id}"
    out_path = GENERATIONS_ROOT / f".{tag}.counts.json"
    payload = {
        "template_level": "chat",
        "cell": {
            "cell_id": f"{model_key}|{arm}", "model_key": model_key,
            "revision": model["revision"], "precision": arm,
            "quant_method": None, "bits": 16,
            "model_path": path, "quant_config_hash": digest,
            "base_repo": model["repo"],
        },
        # Same task, capped to the baseline's first N items so the item SET is identical.
        "plan": [["gsm8k", "gsm8k", None, len(item_ids)]],
        "max_num_seqs": max_num_seqs,
        "run_id": run_id,
        "generations_root": str(GENERATIONS_ROOT),
        "tasks_yaml": str(CONFIGS / "tasks.yaml"),
        "models_yaml": str(CONFIGS / "models.yaml"),
        "out_path": str(out_path),
    }
    payload_path = GENERATIONS_ROOT / f".{tag}.payload.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(payload))
    result = subprocess.run(
        [sys.executable, "-c", CELL_SCRIPT, str(payload_path)], timeout=3600, check=False)
    counts = json.loads(out_path.read_text()) if out_path.exists() else None
    out_path.unlink(missing_ok=True)
    payload_path.unlink(missing_ok=True)
    return None if (result.returncode != 0 and counts is None) else counts


def main() -> int:
    models = yaml.safe_load((CONFIGS / "models.yaml").read_text())["models"]
    specs, calibration = load_specs(CONFIGS / "quant.yaml")
    max_num_seqs = frozen_max_num_seqs()

    print(f"baseline     : {BASELINE_RUN}   items/probe: {N_ITEMS}")
    print(f"max_num_seqs : {max_num_seqs}")
    print("probes       :")
    for model_key, arm, k in PROBES:
        print(f"  {model_key}/{arm}  x{k} invocation(s)")
    print()

    report: dict = {"stage": "diagnose", "baseline": BASELINE_RUN,
                    "n_items": N_ITEMS, "probes": []}

    for model_key, arm, k in PROBES:
        path, digest = resolve(model_key, arm, models, specs, calibration)
        item_ids = baseline_items(model_key, arm)
        runs = []
        for invocation in range(1, k + 1):
            run_id = f"diagnose-v1__{model_key}__{arm}__inv{invocation}"
            print(f"=== {model_key}/{arm} invocation {invocation}/{k} ===", flush=True)
            started = time.monotonic()
            counts = run_invocation(model_key, arm, run_id, path, digest,
                                    models[model_key], max_num_seqs, item_ids)
            if counts is None:
                print("    FAILED", flush=True)
                report["probes"].append({"model": model_key, "arm": arm,
                                         "error": "invocation failed"})
                break
            runs.append(run_id)
            print(f"    {counts['written']} written, {(time.monotonic()-started)/60:.1f} min",
                  flush=True)

        if not runs:
            continue
        entry: dict = {"model": model_key, "arm": arm, "invocations": runs}
        base = stored_text(BASELINE_RUN, model_key, arm)
        new = stored_text(runs[0], model_key, arm)
        shared = sorted(set(base) & set(new))
        if shared:
            same = sum(base[i] == new[i] for i in shared)
            entry["vs_baseline"] = {"n": len(shared), "identical": same,
                                   "differ": len(shared) - same}
            verdict = ("OLD CONFIG WAS NEUTRAL" if same == len(shared)
                       else "OLD CONFIG DIFFERED -> prior data affected")
            print(f"  Q1  vs {BASELINE_RUN}: {same}/{len(shared)} identical  <- {verdict}")
        if len(runs) == 2:
            a, b = stored_text(runs[0], model_key, arm), stored_text(runs[1], model_key, arm)
            sh = sorted(set(a) & set(b))
            same = sum(a[i] == b[i] for i in sh)
            entry["across_invocations"] = {"n": len(sh), "identical": same,
                                          "differ": len(sh) - same}
            verdict = ("REPRODUCIBLE -> sampling config WAS the cause"
                       if same == len(sh) else
                       "STILL DIVERGES -> cause remains open, §9f's K=2 required")
            print(f"  Q2  invocation 1 vs 2: {same}/{len(sh)} identical  <- {verdict}")
        report["probes"].append(entry)
        print()

    (OUTPUT_DIR / "diagnose_report.json").write_text(json.dumps(report, indent=2))
    print("=" * 70)
    print(f"report -> {OUTPUT_DIR / 'diagnose_report.json'}")
    # Exit 0 whatever the answers: a diagnostic that reports "still broken" has SUCCEEDED.
    # Only a failure to measure is a failure, and that is caught above.
    return 1 if any("error" in probe for probe in report["probes"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
