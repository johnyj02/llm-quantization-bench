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
#
# 🚨 `main-v1`, not `pilot-v2` (planner ruling R36.Q1b). The registered invocation floor must be
# measured at REFERENCE n, and against the run the paper's numbers come from. Comparing to the
# pilot would reintroduce the 15x extrapolation (n=180 -> 2,719) that R36 chose (b) over (a)
# specifically to remove.
BASELINE_RUN = "main-v1"

#: 🚨 FULL reference n. Prereg amendment A1 makes this job the SOURCE of the class-2 and class-3
#: floors and of R28's destruction-rate floor, so a short cell would not meet the registered
#: resolution -- the same reasoning R30.2 applied to Anchor A.
N_ITEMS_FLOOR = 4238

#: Enough to DETECT divergence, not to estimate an accuracy. The cross-job rate on llama is
#: 41-45%, so 500 items would show ~200 differences; identity versus difference is the question.
N_ITEMS_CACHE = 500

#: ⚠️ TWO GROUPS, and they must not be conflated: group A must hold `main-v1`'s configuration
#: EXACTLY so the only difference is the job boundary, while group B deliberately CHANGES the
#: compile-cache path. Running them in one job is fine and saves a queue cycle, but the per-unit
#: configuration is recorded so a reader can tell which is which.
#:
#: `(model, arm, n_invocations, n_items, cache_mode, group, why)`
#:   cache_mode "shared"   -- vLLM's default cache root, exactly as `main-v1` ran
#:              "isolated" -- a FRESH cache root per invocation, so each compiles cold
PROBES: tuple[tuple, ...] = (
    # ---- GROUP A: the cross-job invocation floor at reference n (A1, R36.Q1b) ----------
    # One invocation each, compared against main-v1's stored output. The job boundary IS the
    # treatment, so nothing else may differ.
    ("llama-3.1-8b", "bf16", 1, N_ITEMS_FLOOR, "shared", "A",
     "primary contrast, and the model with the largest cross-job effect (41-45%)"),
    ("llama-3.1-8b", "gptq-4bit", 1, N_ITEMS_FLOOR, "shared", "A",
     "the other half of the primary contrast"),
    ("qwen2.5-3b", "bf16", 1, N_ITEMS_FLOOR, "shared", "A",
     "the control: 0% cross-job in the pilot, so it should stay 0%"),

    # ---- GROUP B: is cold torch.compile the mechanism? (R36.Q3) ------------------------
    # 🔬 THE HYPOTHESIS. The compile cache is shared WITHIN a job and cold for each NEW job.
    # main-v1's K=2 invocations shared a warm cache and agreed on 4,238 of 4,238 items; every
    # cross-JOB comparison recompiled from cold and diverged. If that is the mechanism, then two
    # invocations with ISOLATED caches, inside one job, should DIVERGE -- which is the one
    # prediction that separates it from every refuted hypothesis in notes/cross-invocation.md.
    ("llama-3.1-8b", "bf16", 2, N_ITEMS_CACHE, "isolated", "B",
     "cold-compile test: isolated caches should diverge if compilation is the cause"),
    ("llama-3.1-8b", "bf16", 2, N_ITEMS_CACHE, "shared", "B",
     "NEGATIVE CONTROL, same cell and n with a SHARED cache: must stay identical"),
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


def baseline_items(model_key: str, arm: str, n_items: int) -> list[str]:
    """The first `n_items` item_ids of the stored baseline cell, so it is like-for-like.

    ⚠️ Taken FROM the baseline rather than re-derived, because the item SET must be identical:
    `run_cell` submits the whole list in one `generate()` call, so a different item count is a
    different batch composition, and H8 shows batch-sensitive arms diverge up to 71.6% under
    recomposition. A diagnostic that changed n would measure H8 instead of what it asked.
    """
    path = (GENERATIONS_ROOT / BASELINE_RUN / f"{model_key}__{arm}"
            / "gsm8k__fewshot_cot__1x.jsonl.gz")
    if not path.exists():
        raise SystemExit(f"no baseline at {path} — cannot answer Q1 without it")
    ids = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            ids.append(json.loads(line)["item_id"])
            if len(ids) >= n_items:
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
                   max_num_seqs: int, item_ids: list[str],
                   cache_mode: str = "shared") -> dict | None:
    """One engine load. `cache_mode` controls the torch.compile cache root (R36.Q3).

    🔬 `"isolated"` points `VLLM_CACHE_ROOT` at a fresh directory per invocation, so the engine
    compiles its kernels COLD. That is the one manipulation which separates the cold-compile
    hypothesis from every refuted candidate in `notes/cross-invocation.md`: the cache is shared
    within a job and cold for each new job, which is precisely the pattern the divergence follows.
    `"shared"` leaves the default, i.e. exactly what `main-v1` did.
    """
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
        "cache_mode": cache_mode,
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
    env = dict(os.environ)
    if cache_mode == "isolated":
        # A fresh root per invocation. Under the run_id so it is traceable in the container, and
        # NOT cleaned up, so a failure can be inspected rather than guessed at.
        cache_root = Path("/tmp") / f"vllm-cache-{run_id}"    # noqa: S108 - container-local
        cache_root.mkdir(parents=True, exist_ok=True)
        env["VLLM_CACHE_ROOT"] = str(cache_root)
    elif cache_mode != "shared":
        raise SystemExit(f"unknown cache_mode {cache_mode!r}; expected 'shared' or 'isolated'")
    # ⚠️ timeout raised: these are full reference-n cells (4,238 items), not 40-item probes.
    result = subprocess.run(
        [sys.executable, "-c", CELL_SCRIPT, str(payload_path)], timeout=14400, check=False,
        env=env)
    counts = json.loads(out_path.read_text()) if out_path.exists() else None
    out_path.unlink(missing_ok=True)
    payload_path.unlink(missing_ok=True)
    return None if (result.returncode != 0 and counts is None) else counts


def main() -> int:
    models = yaml.safe_load((CONFIGS / "models.yaml").read_text())["models"]
    specs, calibration = load_specs(CONFIGS / "quant.yaml")
    max_num_seqs = frozen_max_num_seqs()

    print(f"baseline     : {BASELINE_RUN}   floor n: {N_ITEMS_FLOOR}   cache-test n: {N_ITEMS_CACHE}")
    print(f"max_num_seqs : {max_num_seqs}")
    print("probes       :")
    for model_key, arm, k, n_items, cache_mode, group, why in PROBES:
        print(f"  [{group}] {model_key}/{arm}  x{k} inv  n={n_items}  cache={cache_mode}")
        print(f"        {why}")
    print()

    report: dict = {"stage": "diagnose", "baseline": BASELINE_RUN,
                    "purpose": "prereg amendment A1 (R36): the CROSS-JOB invocation floor at "
                               "reference n, plus the cold-compile test",
                    "probes": []}

    for model_key, arm, k, n_items, cache_mode, group, why in PROBES:
        path, digest = resolve(model_key, arm, models, specs, calibration)
        item_ids = baseline_items(model_key, arm, n_items)
        runs = []
        for invocation in range(1, k + 1):
            # 🚨 The run_id carries the GROUP, the cache mode and the item count, because two
            # probes in this job use the same (model, arm) and differ only in those. Omitting any
            # of them is the resume-key family this project has hit eleven times.
            run_id = (f"diagnose-v2__{model_key}__{arm}__g{group}"
                      f"__{cache_mode}__n{n_items}__inv{invocation}")
            print(f"=== [{group}] {model_key}/{arm} inv {invocation}/{k} "
                  f"cache={cache_mode} n={n_items} ===", flush=True)
            started = time.monotonic()
            counts = run_invocation(model_key, arm, run_id, path, digest,
                                    models[model_key], max_num_seqs, item_ids,
                                    cache_mode=cache_mode)
            if counts is None:
                print("    FAILED", flush=True)
                report["probes"].append({"model": model_key, "arm": arm,
                                         "group": group, "cache_mode": cache_mode,
                                         "n_items": n_items,
                                         "error": "invocation failed"})
                break
            runs.append(run_id)
            print(f"    {counts['written']} written, {(time.monotonic()-started)/60:.1f} min",
                  flush=True)

        if not runs:
            continue
        entry: dict = {"model": model_key, "arm": arm, "invocations": runs,
                       "group": group, "cache_mode": cache_mode, "n_items": n_items,
                       "why": why}
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
