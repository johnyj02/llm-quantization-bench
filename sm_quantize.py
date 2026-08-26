#!/usr/bin/env python3
"""SageMaker entry point — `quantize` stage. JOB 1 of the plan.

Produces the 9 quantized checkpoints every later job depends on:
3 models × {GPTQ-4bit, AWQ-4bit, GPTQ-8bit}. `bnb-NF4` needs none — vLLM
quantizes it in-flight at load.

**Zero generations.** This job only writes weights.

🚨 **The whole point is the normalised calibration budget.** `llmcompressor`'s own
examples differ by **8× in token budget** between the two methods (GPTQ 512×2048
symmetric; AWQ 256×512 asymmetric). Accepting those defaults would make the
GPTQ-vs-AWQ delta uninterpretable — the difference would be method *plus* an 8×
calibration difference *plus* a symmetry difference.
`quantize.validate_normalisation()` refuses to proceed if that contract is broken.

**Resumable.** Each checkpoint is written to `/opt/ml/checkpoints`, which SageMaker
syncs to S3 continuously and restores on restart — so a timeout after 6 of 9 arms
costs 3 arms, not 9. Arms already present with their exact `quant_config_hash` are
skipped.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from quantartifact.quantize import (  # noqa: E402
    config_hash,
    load_specs,
    output_dir,
    quantize_model,
)

CHECKPOINT_DIR = Path(os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints"))
OUTPUT_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
CONFIGS = Path(__file__).parent / "configs"

SMOKE = os.environ.get("SM_HP_SMOKE") == "1" or "--smoke" in sys.argv

import yaml  # noqa: E402

models = yaml.safe_load((CONFIGS / "models.yaml").read_text())["models"]
specs, calibration = load_specs(CONFIGS / "quant.yaml")

# Only offline-quantizable arms. bnb-NF4 is in-flight, so it has no checkpoint.
arms = [key for key, spec in specs.items() if not spec.in_flight]

if SMOKE:
    # One model, a tiny calibration set, but ONE ARM PER METHOD — proves the whole
    # code path (dataset fetch, llmcompressor recipe, weight write, checkpoint
    # sync) in minutes rather than hours.
    #
    # ⚠️ BOTH gptq AND awq, and that is not padding. The first smoke ran gptq only
    # and caught a real recipe bug; had it passed, AWQ would still have failed nine
    # arms into the real job, because in llmcompressor 0.12.0.1 the two methods use
    # genuinely different APIs — `AWQModifier` is not even a class, but a function
    # returning two modifiers. A smoke that exercises one method does not clear the
    # other. Cover every distinct code path or the rehearsal is theatre.
    import dataclasses

    models = {"qwen2.5-3b": models["qwen2.5-3b"]}
    seen_methods: set[str] = set()
    smoke_arms = []
    for key in arms:
        if specs[key].method not in seen_methods:
            seen_methods.add(specs[key].method)
            smoke_arms.append(key)
    arms = smoke_arms
    calibration = dataclasses.replace(calibration, num_samples=8, max_seq_length=256)
    print(f"SMOKE MODE: 1 model x {len(arms)} arm(s) covering methods "
          f"{sorted(seen_methods)}, calibration 8 x 256")
    print("  ⚠️ smoke weights are NOT usable for the study — the calibration budget")
    print("     is deliberately wrong. They exist only to prove the path runs.")

print(f"models       : {list(models)}")
print(f"arms         : {arms}")
print(f"calibration  : {calibration.num_samples} x {calibration.max_seq_length} "
      f"= {calibration.total_tokens:,} tokens, seed {calibration.seed}")
print(f"dataset      : {calibration.dataset} [{calibration.dataset_split}]")
print(f"checkpoints  : {CHECKPOINT_DIR}")
print()

report: dict = {
    "stage": "quantize",
    "smoke": SMOKE,
    "calibration": calibration.to_dict(),
    "arms": [],
}

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

for model_key, model in models.items():
    for arm in arms:
        spec = specs[arm]
        digest = config_hash(spec, calibration, model["revision"])
        destination = output_dir(CHECKPOINT_DIR, model_key, spec, digest)

        label = f"{model_key}/{arm}"
        # Resume: an arm produced by THIS exact recipe is already done. The hash
        # covers spec + calibration + model revision, so a changed recipe yields a
        # different path and cannot be mistaken for a completed one.
        if (destination / "config.json").exists():
            print(f"[skip] {label}  already present ({digest})")
            report["arms"].append(
                {"model": model_key, "arm": arm, "hash": digest,
                 "path": str(destination), "skipped": True}
            )
            continue

        print(f"[run ] {label}  hash={digest}")
        started = time.monotonic()
        try:
            path = quantize_model(
                model["repo"], model["revision"], model_key, spec, calibration,
                CHECKPOINT_DIR,
            )
            elapsed = time.monotonic() - started
            size_gb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
            print(f"       done in {elapsed / 60:.1f} min, {size_gb:.2f} GB")
            report["arms"].append(
                {"model": model_key, "arm": arm, "hash": digest, "path": str(path),
                 "minutes": round(elapsed / 60, 1), "size_gb": round(size_gb, 2),
                 "skipped": False}
            )
        except Exception as exc:  # noqa: BLE001
            # Record and continue: one failed arm should not cost the arms that
            # would have followed it, since the job has to queue again to retry.
            print(f"       FAILED {type(exc).__name__}: {exc}")
            report["arms"].append(
                {"model": model_key, "arm": arm, "hash": digest,
                 "error": f"{type(exc).__name__}: {exc}"[:500], "skipped": False}
            )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "quantize_report.json").write_text(json.dumps(report, indent=2))

done = [a for a in report["arms"] if "error" not in a]
failed = [a for a in report["arms"] if "error" in a]

print()
print("=" * 66)
print(f"produced/present : {len(done)}/{len(report['arms'])} arms")
for arm in done:
    print(f"  {arm['model']}/{arm['arm']:10} {arm['hash']}"
          f"{'  (skipped)' if arm.get('skipped') else ''}")
if failed:
    print(f"FAILED: {len(failed)}")
    for arm in failed:
        print(f"  {arm['model']}/{arm['arm']}: {arm['error'][:150]}")
print(f"report -> {OUTPUT_DIR / 'quantize_report.json'}")

sys.exit(1 if failed else 0)
