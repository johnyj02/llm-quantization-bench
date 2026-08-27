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
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from quantartifact.quantize import (  # noqa: E402
    config_hash,
    load_specs,
    output_dir,
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
      f"= {calibration.total_tokens:,} tokens (budget ceiling), seed {calibration.seed}")
print(f"dataset      : {calibration.dataset} [{calibration.dataset_split}]")
print(f"checkpoints  : {CHECKPOINT_DIR}")
print()

# Verify EVERY model is reachable at its pinned revision before quantizing any of
# them. `meta-llama/Llama-3.1-8B-Instruct` is gated, and it is LAST in the config
# order — so without this an expired or unset HF_TOKEN surfaces roughly two hours
# into the job, after the slot is spent and the two ungated models are done.
# Seconds to check, hours to discover otherwise.
print("=== access preflight ===")
# Blocks ONLY on a gated repo, and that asymmetry is deliberate. A gated failure
# is a credential problem: deterministic, and certain to recur two hours from now.
# An ungated failure is far more likely to be a transient network error, and
# killing a job that would have succeeded wastes the one queue slot we waited for
# — so those are reported and left to the per-arm error handling below.
_blocked = []
try:
    from huggingface_hub import HfApi

    _api = HfApi(token=os.environ.get("HF_TOKEN"))
    for _key, _model in models.items():
        _gated = bool(_model.get("gated"))
        try:
            _info = _api.model_info(_model["repo"], revision=_model["revision"])
            # Confirm the pin resolved to exactly the commit we asked for. A tag
            # that moved would otherwise be quantized silently under the old name.
            _sha = getattr(_info, "sha", None)
            _match = "" if _sha in (None, _model["revision"]) else f"  ⚠️ resolved {_sha[:12]}"
            print(f"  [OK ] {_key:14} @ {_model['revision'][:12]}"
                  f"{' (gated)' if _gated else ''}{_match}")
        except Exception as _exc:  # noqa: BLE001
            print(f"  [{'FAIL' if _gated else 'WARN'}] {_key:14} "
                  f"{type(_exc).__name__}: {_exc}"[:220])
            if _gated:
                _blocked.append(_key)
except ImportError:
    print("  huggingface_hub unavailable; skipping (transformers will fail later)")

if _blocked:
    print()
    print(f"BLOCKED: gated repo(s) {_blocked} unreachable at the pinned revision.")
    print("  This is a credential problem and will not fix itself mid-job. A gated")
    print("  repo needs a valid HF_TOKEN in the job environment — check that")
    print("  submit_job.py reported 'HF_TOKEN: present' at submit time.")
    sys.exit(1)
print()

# 🚨 ONE SUBPROCESS PER ARM. This is why the first real run only got 5 of 9.
#
# Running all nine arms in one process leaked GPU memory monotonically: the arms
# succeeded in order, then phi's AWQ (the most memory-hungry) died, then Llama's
# 8-bit got partway, then the last two failed within seconds — with the log
# reporting `Accelerator 0 | usage: 95.70%`. A *fresh* Llama-8B GPTQ pass needs
# roughly 10 GB of the 47.7 GB card (≈436 MB of layer weights, ≈1.2 GB of
# Hessians, ≈8.6 GB of cached calibration activations), so 96% was accumulated
# state from previous arms, not a genuine requirement.
#
# Python cannot reliably give it back: `del model` leaves torch's caching
# allocator holding blocks, llmcompressor installs hooks on the model, and one arm
# had already logged "Exception during finalizing modifier: Failed to compress 7
# modules" — so the next arm starts from a poisoned lifecycle. Process exit is the
# only teardown that always works.
#
# This is the pattern `sm_determinism.py` and `sm_throughput.py` already use for
# exactly this reason. Applying it to vLLM but not to llmcompressor was an
# inconsistency, and it cost 76 minutes and four arms.
#
# ⚠️ Deliberately NOT changing `sequential_targets` to "Linear", which the error
# message suggests. That alters the granularity at which GPTQ accumulates Hessians
# within a decoder layer, and therefore potentially the weights themselves. Process
# isolation is semantically neutral; a granularity change is not, and the recipe is
# frozen and reported in the paper.
ARM_SCRIPT = r"""
import json, sys
sys.path.insert(0, "/opt/ml/code/src")
from pathlib import Path
from quantartifact.quantize import load_specs, quantize_model
import dataclasses

payload = json.loads(Path(sys.argv[1]).read_text())
specs, calibration = load_specs(Path(payload["configs"]) / "quant.yaml")
if payload["calibration_override"]:
    calibration = dataclasses.replace(calibration, **payload["calibration_override"])

path = quantize_model(
    payload["model_repo"], payload["model_revision"], payload["model_key"],
    specs[payload["arm"]], calibration, Path(payload["checkpoint_dir"]),
)
print(f"__ARM_OK__ {path}")
"""


def quantize_arm_subprocess(
    model_repo: str, model_revision: str, model_key: str, arm: str,
    digest: str, destination: Path,
) -> Path:
    """Quantize ONE arm in a child process, so its GPU memory dies with it.

    Raises on failure, so the caller's existing per-arm error handling is
    unchanged and one bad arm still does not cost the arms after it.
    """
    payload = {
        "model_repo": model_repo,
        "model_revision": model_revision,
        "model_key": model_key,
        "arm": arm,
        "configs": str(CONFIGS),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        # The parent may have shrunk the calibration budget for --smoke; the child
        # re-reads quant.yaml, so that override has to travel explicitly or the
        # smoke would silently run the FULL 512x2048 budget.
        "calibration_override": (
            {"num_samples": calibration.num_samples,
             "max_seq_length": calibration.max_seq_length}
            if SMOKE else None
        ),
    }
    payload_path = CHECKPOINT_DIR / f"payload_{model_key}__{arm}__{digest}.json"
    payload_path.write_text(json.dumps(payload))

    # ⚠️ Output is INHERITED, not captured. `capture_output=True` buffers until the
    # child exits, so a 30-minute arm would emit nothing to CloudWatch and look like
    # a hang -- and if the job were killed we would lose the child's output entirely,
    # which is exactly the diagnostic that found this bug. Inheriting streams
    # llmcompressor's per-layer progress live. The traceback therefore lands in
    # CloudWatch rather than in this exception, which is where we read it anyway.
    sys.stdout.flush()
    result = subprocess.run(
        [sys.executable, "-c", ARM_SCRIPT, str(payload_path)],
        timeout=10800,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"arm subprocess exited {result.returncode} "
            f"(traceback is above, in this job's CloudWatch log)"
        )
    if not (destination / "config.json").exists():
        raise RuntimeError(
            f"arm subprocess reported success but {destination} has no config.json"
        )
    payload_path.unlink(missing_ok=True)
    return destination


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

        # ⚠️ SMOKE NEVER SKIPS. A rehearsal exists to exercise the code path, and a
        # previous smoke has already written these exact hashes into the smoke
        # prefix -- so with normal resume semantics a re-run would skip every arm,
        # report success, and prove nothing. That would be worse than not running
        # it, because it looks like a pass. Discard the prior smoke artifact (it is
        # deliberately unusable junk, in the smoke-only prefix) and redo the work.
        if SMOKE and destination.exists():
            assert CHECKPOINT_DIR in destination.parents, destination
            shutil.rmtree(destination)
            print(f"[smoke] discarded previous rehearsal artifact at {destination.name}")

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
            path = quantize_arm_subprocess(
                model["repo"], model["revision"], model_key, arm, digest, destination,
            )
            elapsed = time.monotonic() - started
            size_gb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
            # Read back the provenance the writer just emitted, so the job report
            # carries the MEASURED calibration token count rather than the budget
            # ceiling. `total_tokens` is num_samples * max_seq_length, which is an
            # upper bound: with padding disabled, a short sample contributes less.
            provenance = {}
            provenance_path = path / "quantartifact_provenance.json"
            if provenance_path.exists():
                provenance = json.loads(provenance_path.read_text())
            tokens = provenance.get("calibration_tokens_actual")
            budget = provenance.get("calibration_tokens_budget")
            if tokens and budget:
                print(f"       done in {elapsed / 60:.1f} min, {size_gb:.2f} GB, "
                      f"calibration {tokens:,} tokens ({tokens / budget:.1%} of the "
                      f"{budget:,} ceiling)")
            else:
                print(f"       done in {elapsed / 60:.1f} min, {size_gb:.2f} GB")
            report["arms"].append(
                {"model": model_key, "arm": arm, "hash": digest, "path": str(path),
                 "minutes": round(elapsed / 60, 1), "size_gb": round(size_gb, 2),
                 "calibration_tokens_actual": tokens,
                 "calibration_tokens_budget": budget,
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
