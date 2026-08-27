#!/usr/bin/env python3
"""SageMaker entry point — `determinism` stage. JOB 2 of the plan. Task A7, gate G4.

**This job decides whether the rest of the study is worth running**, so it runs
before any reference generation. The paper is about small accuracy deltas, and
vLLM states outright that it *"does not guarantee the reproducibility of the
results by default, for the sake of performance"* — issue #9567 reported exactly
our failure mode on an INT8 Llama-3-8B and tied it to unstable lm-eval scores.
Uncontrolled batch nondeterminism could **manufacture or erase our effect**.

Three things it produces, all pre-registered:

1. **Per-format bitwise disagreement** across 3 repeats of an identical probe.
   H8 predicts an *asymmetry*: `auto_awq.py` carries a batch-invariance guard
   ("Marlin kernels are not batch invariant"), while `auto_gptq.py` and the
   `choose_mp_linear_kernel()` dispatcher GPTQ actually uses have **none** — so
   GPTQ should keep using Marlin and be the less stable arm.
2. **The frozen `max_num_seqs`** (planner ruling R8): the largest value bitwise
   stable for *every* format — a **minimum over formats**, because §3.3 requires
   one value identical across all arms. Overwrites `models.yaml`.
3. **The differential noise floor**: if a quantized format is nondeterministic
   while bf16 is not, every contrast involving it inherits that rate as a floor
   (`analysis/preregistration.md` §9). This is what makes H8 consequential
   rather than merely curious.

⚠️ **A null result here is a FINDING, not a failure.** If nothing above the floor
is stable for every format, that is a headline H8 result and the study proceeds
at `max_num_seqs=8`. The job exits 0 either way; it exits non-zero only when it
could not *measure*.

**Cost control.** The sweep is DESCENDING with early exit (`SWEEP_ORDER`): we want
the largest stable value, so testing 64 first answers it in one level instead of
four. Every cell is checkpointed, so a timeout costs one cell rather than the job.

**Three additions from planner rulings R12.3 and R13**, each closing a specific hole:

* **R12.3.1 — the level BELOW the stable one is always measured.** "Stable at 64
  but unstable at 32" would be a non-monotonicity, and the frontier framing assumes
  stability is monotone in batch size. Without it the frontier is two points and
  the H8 contribution — telling others what they can safely get away with — is thin.
* **R13.1 — the confirmation varies item ORDERING, not repeat count.** My first
  version ran 10 repeats at the selected value; that measures the wrong axis.
  Batch-invariance failures happen because batch *composition* changes float
  reduction order, and repeating the same items in the same order re-presents the
  scheduler with nearly the same batches. A different order means different
  co-residency, which is what vLLM #51187 reported. Ordering 0 is the sweep's own
  pass, reused from disk, so this costs 5,000 generations rather than 25,000.
* **R12.3.3 — a model-generality probe on Phi-3.5-mini.** One model establishes the
  *mechanism* (it lives in vLLM's kernel-dispatch source, so it is
  architecture-general) but not the *rates*. vLLM #48271's leak depends on token
  count and block-size selection, both model-dependent, and Phi's fused
  3072→16384 gate+up projection — the same quirk that broke job 1's AWQ memory
  budget — could take a different kernel path. Phi over Llama deliberately: the
  architectural outlier is the higher-information second point.

**R13.2 makes the level count conditional on the finding.** The floor existed to
guarantee that *some* value is stable; once a level is stable that purpose is served,
so the floor is skipped. 64-stable costs 2 levels; 64-unstable walks down and costs
up to 4 — cheap when the answer is good news, expensive only when the frontier is
genuinely interesting. **23,000 generations best case, 45,500 worst.**

The three write SEPARATE artifacts. Only the sweep defines the frontier, and
`select_max_num_seqs` must see only its own single-ordering 3-repeat cells — folding
in a different measurement would let the confirmation veto the value it exists to
confirm.

**A6's deliverable falls out of this job** (R13.5): per-pass wall time and
generations/hour are logged for every cell, so measured throughput ± the SM89
batch-invariance overhead does not need a separate job.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import yaml  # noqa: E402

from quantartifact.determinism import (  # noqa: E402
    DISABLED_KERNELS_FALLBACK,
    ORDERING_SEEDS,
    PROBE_COMPOSITION,
    REPEATS,
    SWEEP_FLOOR,
    DisagreementResult,
    RepeatOutputs,
    compute_disagreement,
    distinct_output_histogram,
    next_sweep_level,
    probe_fingerprint,
    select_max_num_seqs,
    summarise,
    sweep_plan,
    write_csv,
    write_json,
)
from quantartifact.quantize import config_hash, load_specs, output_dir  # noqa: E402

CHECKPOINT_DIR = Path(os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints"))
OUTPUT_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
CONFIGS = Path(__file__).parent / "configs"

SMOKE = os.environ.get("SM_HP_SMOKE") == "1" or "--smoke" in sys.argv

# 🚨 SMOKE RESULTS GO SOMEWHERE ELSE ENTIRELY, and this is a correctness issue,
# not hygiene. A probe cell is keyed by (arm, max_num_seqs) — `bf16_at_8` — with no
# reference to the probe SIZE. So with a shared directory the resume check would
# let a smoke's 20-item result be read by a real run as its 500-item measurement,
# and a disagreement RATE computed over 20 items would be reported as if it came
# from 500. That is a wrong number that looks entirely plausible, which is the
# worst kind. Separated by directory so the two can never meet.
PROBE_DIR = CHECKPOINT_DIR / ("determinism-smoke" if SMOKE else "determinism")

# ONE model for the audit, and that is a deliberate scope decision, not a
# shortcut. H8's mechanism is in vLLM's *kernel dispatch* source, so it is
# model-independent; only the measured rates are box-specific. Auditing all three
# models would trible the cost of a gate whose job is to decide whether to spend
# on the real matrix at all. The smallest model also loads fastest, and it is the
# one with official vendor checkpoints (anchor A) if we ever need to cross-check.
AUDIT_MODEL = os.environ.get("QA_AUDIT_MODEL", "qwen2.5-3b")

# The five precision arms of the reference condition.
AUDIT_ARMS = ("bf16", "gptq-8bit", "gptq-4bit", "awq-4bit", "bnb-nf4")


def arm_engine_fields(arm: str, specs: dict) -> tuple[str | None, int]:
    """`(quant_method, bits)` for an arm's `EngineSpec`.

    ⚠️ **`bf16` is deliberately absent from `specs`**: `load_specs` skips
    `method == "none"`, so `specs["bf16"]` is a KeyError, not a spec with
    `method="none"`. Centralised here because getting it wrong in one of three
    call sites would mis-tag an arm's records rather than crash.
    """
    if arm == "bf16":
        return None, 16
    return specs[arm].method, specs[arm].bits


def cell_tag(
    arm: str,
    max_num_seqs: int,
    disabled_kernels: str | None = None,
    *,
    model_key: str = AUDIT_MODEL,
    repeats: int = REPEATS,
    ordering_seed: int = 0,
) -> str:
    """Filesystem-safe identity for one probe cell. Also its resume key.

    ⚠️ **MODEL, REPEATS and ORDERING all belong in the tag.** The tag is the resume
    key, so anything that changes what a cell measures must change its name, or a
    later cell silently inherits an earlier cell's answer:

    * The Phi-3.5-mini model-generality probe (R12.3.3) runs the same arms at the
      same level as the Qwen audit, so without the model key ``bf16_at_8`` would
      collide and Phi would report Qwen's numbers.
    * The ordering confirmation (R13.1) runs the same arm at the same level as the
      sweep. Without the ordering seed it would resume the sweep's result and
      "confirm" nothing at all.
    * The repeat count is in there for the same reason — a 1-pass ordering cell and
      a 3-repeat sweep cell are different measurements.
    """
    parts = [model_key, arm, f"at{max_num_seqs}", f"r{repeats}"]
    if ordering_seed:
        parts.append(f"ord{ordering_seed}")
    if disabled_kernels:
        parts.append("fallback")
    return "__".join(parts)


def bbh_probe_split(total: int, subtasks: list[str]) -> dict[str, int]:
    """Spread BBH's share of the probe across subtasks, deterministically.

    BBH is not one answer format: `configs/tasks.yaml` classes its subtasks as
    free_form, boolean and multiple_choice. Taking all 200 from one subtask would
    let a format-dependent nondeterminism hide, which is the exact thing the probe
    exists to prevent. Remainder goes to the earliest subtasks so the split is a
    pure function of the config order.
    """
    base, extra = divmod(total, len(subtasks))
    return {name: base + (1 if i < extra else 0) for i, name in enumerate(subtasks)}


def build_probe(model: dict, model_key: str) -> list[dict]:
    """Materialize the A7 probe with lm-eval, the same code path as G2.

    Rendered on the box rather than shipped, because the committed prompt artifact
    is the 25-item G2 parity sample and the probe needs 500. Same renderer, same
    pinned `FEWSHOT_SEED`, so these prompts are byte-comparable with the study's.
    """
    # Lazy, and deliberately so: neither is importable in the analysis venv, which
    # is what keeps every non-GPU part of this repo runnable on a laptop.
    from transformers import AutoTokenizer  # noqa: PLC0415 - harness/box-only dependency

    from quantartifact.tasks.materialize import (  # noqa: PLC0415 - pulls in transformers
        manifest_for,
        render_task_prompts,
    )

    tasks = yaml.safe_load((CONFIGS / "tasks.yaml").read_text())["tasks"]
    tokenizer = AutoTokenizer.from_pretrained(model["repo"], revision=model["revision"])

    bbh_subtasks = [s["name"] for s in tasks["bbh"]["subtasks"]]
    bbh_split = bbh_probe_split(PROBE_COMPOSITION["bbh"], bbh_subtasks)

    plan: list[tuple[str, str, str | None, int]] = [
        ("gsm8k_cot", "gsm8k_cot", None, PROBE_COMPOSITION["gsm8k_cot"]),
        ("arc_challenge_chat", "arc_challenge_chat", None, PROBE_COMPOSITION["arc_challenge_chat"]),
    ]
    for name, count in bbh_split.items():
        plan.append((f"{tasks['bbh']['lm_eval_task_prefix']}{name}", "bbh", name, count))

    items: list[dict] = []
    manifests: list[dict] = []
    for lm_eval_task, task, subtask, limit in plan:
        config = tasks[task]
        rendered = list(
            render_task_prompts(
                lm_eval_task,
                tokenizer,
                task=task,
                subtask=subtask,
                prompt_variant="fewshot_cot",
                model_key=model_key,
                model_repo=model["repo"],
                model_revision=model["revision"],
                num_fewshot=config.get("num_fewshot"),
                apply_chat_template=True,
                fewshot_as_multiturn=False,
                limit=limit,
            )
        )
        manifests.append(manifest_for(rendered))
        # `until` and `max_gen_toks` are per-task and must match the study's 1x
        # budget: the probe measures the engine under the conditions we will
        # actually run, not under a convenient uniform setting.
        for prompt in rendered:
            items.append(
                {
                    # Namespaced so gsm8k item "3" and a bbh item "3" cannot collide
                    # into a false agreement when the probe is pooled.
                    "item_id": f"{task}:{subtask or '-'}:{prompt.item_id}",
                    "prompt": prompt.prompt_text,
                    "max_gen_toks": config["max_gen_toks"]["1x"],
                    "until": config["until"],
                }
            )
        print(f"  {lm_eval_task:44} {len(rendered):4} items")

    (OUTPUT_DIR / "probe_manifest.json").write_text(
        json.dumps({"n_items": len(items), "manifests": manifests}, indent=2)
    )
    return items


def resolve_model_path(
    arm: str, model: dict, specs: dict, calibration, model_key: str = AUDIT_MODEL
) -> tuple[str, str | None]:
    """Where the weights for one arm live, and the recipe hash that produced them.

    bf16 and bnb-nf4 serve from the base repo (NF4 is quantized in-flight by vLLM
    at load). The other three must find the checkpoint job 1 wrote, resolved BY
    RECIPE HASH — so a checkpoint built with a different calibration budget cannot
    be silently substituted for the one this study registered.
    """
    if arm in ("bf16", "bnb-nf4"):
        return model["repo"], None

    spec = specs[arm]
    digest = config_hash(spec, calibration, model["revision"])
    path = output_dir(CHECKPOINT_DIR, model_key, spec, digest)
    if not (path / "config.json").exists():
        raise FileNotFoundError(
            f"no checkpoint for {model_key}/{arm} at {path}. Job 1 (quantize) "
            f"must complete first, and its checkpoint_s3_uri must match this "
            f"job's so SageMaker restores it. Expected recipe hash {digest}."
        )
    return str(path), digest


PROBE_SCRIPT = r"""
import json, os, random, sys, time
sys.path.insert(0, "/opt/ml/code/src")
from quantartifact.generate import EngineSpec, VLLMRunner

payload = json.loads(open(sys.argv[1]).read())
spec = EngineSpec(**payload["spec"])
started = time.monotonic()
runner = VLLMRunner(spec)          # preflight() asserts every BI invariant here
load_s = time.monotonic() - started

items = list(payload["items"])
# ORDERING (R13.1). Seed 0 is the natural probe order used by the sweep. A non-zero
# seed shuffles the items, which changes which sequences are CO-RESIDENT in each
# batch -- the actual mechanism behind batch-invariance failures, and the thing
# vLLM #51187 reported. Shuffled BEFORE grouping so the order changes inside each
# vLLM call, not merely between calls.
if payload.get("ordering_seed"):
    random.Random(payload["ordering_seed"]).shuffle(items)

out = {"repeats": [], "hardware": runner.hardware, "engine_load_s": round(load_s, 1)}
for repeat_idx in range(payload["repeats"]):
    texts = {}
    # Group by (max_gen_toks, until) so each vLLM call is a single homogeneous
    # batch, exactly as run_cell does it. Sorted for a deterministic call order:
    # the probe must differ across repeats ONLY by engine nondeterminism.
    groups = {}
    for item in items:
        groups.setdefault((item["max_gen_toks"], tuple(item["until"])), []).append(item)
    pass_started = time.monotonic()
    for (max_toks, until), group in sorted(groups.items()):
        results = runner.generate([g["prompt"] for g in group], max_toks, list(until))
        for g, r in zip(group, results):
            texts[g["item_id"]] = r["text"]
    elapsed = time.monotonic() - pass_started
    out["repeats"].append(texts)
    # Per-pass wall time is how A6's deliverable (measured generations/hour on SM89)
    # falls out of this job instead of needing its own (R13.5).
    out.setdefault("pass_seconds", []).append(round(elapsed, 1))
    rate = len(texts) / elapsed * 3600 if elapsed else 0
    print(f"  pass {repeat_idx}: {len(texts)} outputs in {elapsed / 60:.1f} min "
          f"(~{rate:,.0f} gen/h)", flush=True)

open(payload["out_path"], "w").write(json.dumps(out))
"""


def run_probe_cell(
    arm: str,
    model_path: str,
    revision: str,
    quant_method: str | None,
    bits: int,
    max_num_seqs: int,
    items: list[dict],
    disabled_kernels: str | None,
    model_key: str = AUDIT_MODEL,
    repeats: int = REPEATS,
    ordering_seed: int = 0,
) -> DisagreementResult | None:
    """Run one (arm, max_num_seqs) cell: `REPEATS` passes over the probe.

    **In a subprocess, and that is required, not tidy.** `max_num_seqs` and
    `VLLM_DISABLED_KERNELS` are read at engine init, and vLLM's CUDA teardown is
    not reliable enough to build five engines in one process. One process per
    cell also means a crash costs one cell.

    The repeats share ONE engine deliberately: the question A7 asks is whether
    repeated identical requests to a *running* engine return identical text, which
    is the shape vLLM #51187 reported. `enable_prefix_caching=False` keeps the
    repeats from contaminating each other.
    """
    tag = cell_tag(arm, max_num_seqs, disabled_kernels,
                   model_key=model_key, repeats=repeats,
                   ordering_seed=ordering_seed)
    result_path = PROBE_DIR / f"{tag}.json"

    # A rehearsal must actually rehearse. Resuming from a previous smoke's cached
    # cell would report success while running no vLLM at all.
    if SMOKE and result_path.exists():
        result_path.unlink()
        print(f"[smoke] discarded previous rehearsal result for {tag}")

    if result_path.exists():
        print(f"[skip] {tag}  already measured")
        cached = json.loads(result_path.read_text())
        return DisagreementResult(**cached["result"])

    raw_path = PROBE_DIR / f"raw_{tag}.json"
    payload = {
        "spec": {
            "model_repo": model_path,
            "model_revision": revision,
            "quant_method": quant_method,
            "bits": bits,
            "max_num_seqs": max_num_seqs,
            "disabled_kernels": disabled_kernels,
        },
        "items": items,
        "repeats": repeats,
        "ordering_seed": ordering_seed,
        "out_path": str(raw_path),
    }
    payload_path = PROBE_DIR / f"payload_{tag}.json"
    payload_path.write_text(json.dumps(payload))

    env = dict(os.environ)
    # Never inherited: BI must be ON for every study measurement, and
    # VLLM_DISABLED_KERNELS must be absent unless this cell is the fallback probe.
    env["VLLM_BATCH_INVARIANT"] = "1"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.pop("VLLM_DISABLED_KERNELS", None)
    if disabled_kernels:
        env["VLLM_DISABLED_KERNELS"] = disabled_kernels

    print(f"[run ] {tag}  {repeats} repeats x {len(items)} items")
    # Output INHERITED, not captured. `capture_output=True` buffers until exit, so a
    # multi-hour cell would emit nothing to CloudWatch and look like a hang -- and
    # vLLM's engine-startup logs, the ones worth watching during a long load, are
    # exactly what would be withheld. The traceback lands in CloudWatch instead.
    sys.stdout.flush()
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", PROBE_SCRIPT, str(payload_path)],
        env=env, timeout=14400,
    )
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        print(f"       FAILED after {elapsed / 60:.1f} min, exit {proc.returncode} "
              f"(traceback above)")
        return None

    raw = json.loads(raw_path.read_text())
    # NOT named `repeats` -- that is the int parameter, and shadowing it here made
    # the count unavailable to anything after this point.
    repeat_outputs = [
        RepeatOutputs(
            fmt=arm, max_num_seqs=max_num_seqs, repeat_idx=i,
            outputs=texts, disabled_kernels=disabled_kernels,
        )
        for i, texts in enumerate(raw["repeats"])
    ]
    result = compute_disagreement(repeat_outputs)

    result_path.write_text(json.dumps({
        "result": dataclasses.asdict(result),
        "repeats": repeats,
        "fingerprints": [probe_fingerprint(r.outputs) for r in repeat_outputs],
        "distinct_output_histogram": {
            str(k): v for k, v in distinct_output_histogram(repeat_outputs).items()
        },
        "minutes": round(elapsed / 60, 1),
        "hardware": raw.get("hardware"),
    }, indent=2))

    verdict = "STABLE" if result.bitwise_stable else (
        f"{result.n_disagreeing}/{result.n_items} disagree "
        f"({result.disagreement_rate:.3%}), worst {result.max_distinct_outputs} distinct"
    )
    print(f"       {elapsed / 60:.1f} min — {verdict}")
    return result


def confirm_ordering(
    arm: str,
    level: int,
    items: list[dict],
    specs: dict,
    model: dict,
    resolved: dict[str, tuple[str, str | None]],
    model_key: str = AUDIT_MODEL,
) -> DisagreementResult | None:
    """Does this arm's output depend on the ORDER the probe was submitted in?

    Three passes at the same `max_num_seqs`, differing only in item order. Ordering 0
    is read back from the sweep's own run at this level rather than regenerated —
    that reuse is what makes this cost 5,000 generations instead of 25,000.

    Returns None if the sweep's pass is missing, since without it there is nothing to
    compare the shuffled orderings against.

    ⚠️ The comparison is order-INSENSITIVE by construction: outputs are dicts keyed by
    `item_id`, so shuffling the submission order cannot by itself register as
    disagreement. Only a genuinely different generation for the same item does.
    """
    baseline_tag = cell_tag(arm, level, model_key=model_key, repeats=REPEATS)
    baseline_raw = PROBE_DIR / f"raw_{baseline_tag}.json"
    if not baseline_raw.exists():
        print(f"[skip] ordering {arm}@{level}: no sweep pass at this level to compare "
              f"against ({baseline_raw.name})")
        return None

    passes = [
        RepeatOutputs(
            fmt=arm, max_num_seqs=level, repeat_idx=0,
            outputs=json.loads(baseline_raw.read_text())["repeats"][0],
        )
    ]

    path, _ = resolved[arm]
    method, bits = arm_engine_fields(arm, specs)
    for index, seed in enumerate(ORDERING_SEEDS[1:], start=1):
        result = run_probe_cell(
            arm, path, model["revision"], quant_method=method, bits=bits,
            max_num_seqs=level, items=items, disabled_kernels=None,
            model_key=model_key, repeats=1, ordering_seed=seed,
        )
        if result is None:
            return None
        raw = PROBE_DIR / (
            "raw_"
            + cell_tag(arm, level, model_key=model_key, repeats=1, ordering_seed=seed)
            + ".json"
        )
        passes.append(
            RepeatOutputs(
                fmt=arm, max_num_seqs=level, repeat_idx=index,
                outputs=json.loads(raw.read_text())["repeats"][0],
            )
        )

    combined = compute_disagreement(passes)
    verdict = "order-INVARIANT" if combined.bitwise_stable else (
        f"{combined.n_disagreeing}/{combined.n_items} items changed with order "
        f"({combined.disagreement_rate:.3%})"
    )
    print(f"       {arm}: {verdict}")
    return combined


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROBE_DIR.mkdir(parents=True, exist_ok=True)

    models = yaml.safe_load((CONFIGS / "models.yaml").read_text())["models"]
    specs, calibration = load_specs(CONFIGS / "quant.yaml")
    model = models[AUDIT_MODEL]

    arms = list(AUDIT_ARMS)
    if SMOKE:
        arms = ["bf16"]
        print("SMOKE MODE: bf16 only, floor level only, 20 items")
        print("  ⚠️ proves the probe path runs. It CANNOT answer H8 or select")
        print("     max_num_seqs — one arm measures no asymmetry and one level no frontier.")

    print(f"audit model  : {AUDIT_MODEL} ({model['repo']})")
    print(f"arms         : {arms}")
    print(f"repeats      : {REPEATS}")
    plan = sweep_plan(arms, sum(PROBE_COMPOSITION.values()), REPEATS)
    print(f"sweep        : {plan['order']} descending, early exit; floor "
          f"{plan['floor_always_measured']} always measured")
    print(f"generations  : {plan['best_case_generations']:,} best -> "
          f"{plan['worst_case_generations']:,} worst "
          f"(exhaustive would be {plan['exhaustive_would_cost']:,})")
    print()

    print("=== materializing the probe ===")
    items = build_probe(model, AUDIT_MODEL)
    if SMOKE:
        items = items[:20]
    print(f"probe        : {len(items)} items")
    print()

    # Resolve every arm's weights BEFORE loading anything. A missing checkpoint
    # should fail in seconds, not after an hour of measuring the arms that do exist.
    resolved: dict[str, tuple[str, str | None]] = {}
    for arm in arms:
        try:
            resolved[arm] = resolve_model_path(arm, model, specs, calibration)
        except FileNotFoundError as exc:
            print(f"BLOCKED: {exc}")
            return 1
        print(f"  {arm:10} -> {resolved[arm][0]}")
    print()

    results: list[DisagreementResult] = []
    unmeasured: set[str] = set()

    def measure(
        arm: str,
        level: int,
        fallback: str | None = None,
        *,
        repeats: int = REPEATS,
        collect: list[DisagreementResult] | None = None,
        model_key: str = AUDIT_MODEL,
        model_cfg: dict | None = None,
        paths: dict[str, tuple[str, str | None]] | None = None,
    ) -> DisagreementResult:
        """Run one cell and record it, treating an unmeasured cell as UNSTABLE.

        That default matters: if a cell crashes and we simply skip it,
        `next_sweep_level` sees the level as incomplete and retries forever, and
        `select_max_num_seqs` could pick a level on the strength of a measurement
        that never happened. Recording it as maximally unstable makes a missing
        measurement fail closed.

        ``collect`` decides which list the result joins. The frontier list must
        contain ONLY the Qwen sweep cells: the 10-repeat confirmation and the Phi
        probe measure different things, and folding them into the frontier would let
        `select_max_num_seqs` mix probes of different repeat counts and models.
        """
        cfg = model_cfg if model_cfg is not None else model
        lookup = paths if paths is not None else resolved
        path, _ = lookup[arm]
        method, bits = arm_engine_fields(arm, specs)
        result = run_probe_cell(
            arm, path, cfg["revision"], quant_method=method, bits=bits,
            max_num_seqs=level, items=items, disabled_kernels=fallback,
            model_key=model_key, repeats=repeats,
        )
        if result is None:
            unmeasured.add(
                cell_tag(arm, level, fallback, model_key=model_key, repeats=repeats)
            )
            result = DisagreementResult(
                fmt=arm, max_num_seqs=level, n_items=len(items),
                n_disagreeing=len(items), disagreement_rate=1.0,
                max_distinct_outputs=repeats, bitwise_stable=False,
                disabled_kernels=fallback,
            )
        (results if collect is None else collect).append(result)
        return result

    # SMOKE pins the floor only. Otherwise `next_sweep_level` drives the descending
    # sweep and decides when the frontier is settled.
    pending_levels = [SWEEP_FLOOR] if SMOKE else None
    while True:
        if pending_levels is not None:
            if not pending_levels:
                break
            level = pending_levels.pop(0)
        else:
            level = next_sweep_level(results, arms)
            if level is None:
                break
        print(f"=== max_num_seqs = {level} ===")
        for arm in arms:
            measure(arm, level)
        print()

    # H8's mechanism test: where a format was nondeterministic, does disabling
    # Marlin RESTORE determinism? That converts "GPTQ is less stable" from a
    # correlation into a claim about the kernel, which is the whole point.
    #
    # Arms whose cells could not be measured are excluded: their instability is an
    # artefact of the crash, so probing the kernel would answer nothing.
    # cell_tag is "<model>__<arm>__at<N>__r<K>", so the arm is field 1. Parsing the
    # old "<arm>_at_<N>" shape here silently matched nothing after the tag changed,
    # which would have let a crashed arm into the Marlin probe.
    crashed_arms = {tag.split("__")[1] for tag in unmeasured if "__" in tag}
    unstable = sorted(
        {r.fmt for r in results if not r.bitwise_stable and r.fmt not in crashed_arms}
    )
    if unstable and not SMOKE:
        print(f"=== Marlin fallback probe: {unstable} ===")
        print(f"    VLLM_DISABLED_KERNELS={DISABLED_KERNELS_FALLBACK}")
        for arm in unstable:
            measure(arm, SWEEP_FLOOR, fallback=DISABLED_KERNELS_FALLBACK)
        print()

    # ---------------------------------------------------------------------
    # R13.1 — CONFIRM the selected value by varying ITEM ORDERING, not repeats.
    #
    # Batch-invariance failures happen because batch COMPOSITION changes float
    # reduction order. Repeating the same items in the same order re-presents the
    # scheduler with nearly the same batches, so repeat count is a weak way to
    # explore the failure surface. A different ORDER means different co-residency,
    # which is what vLLM #51187 actually reported.
    #
    # Ordering 0 is the natural order the sweep already ran at this level, so it is
    # REUSED from disk. Only the extra orderings cost generations.
    #
    # Kept OUT of `results`: `select_max_num_seqs` is defined over the 3-repeat
    # single-ordering sweep cells, and folding in a different measurement would let
    # the confirmation veto the value it exists to confirm.
    # ---------------------------------------------------------------------
    confirmation: list[DisagreementResult] = []
    selected = select_max_num_seqs(results)["selected_max_num_seqs"]
    confirm_level = selected if selected is not None else SWEEP_FLOOR
    if not SMOKE:
        print(f"=== ordering confirmation: max_num_seqs={confirm_level}, "
              f"orderings {list(ORDERING_SEEDS)} (R13.1) ===")
        if selected is None:
            print("    ⚠️ no value was stable for every arm — confirming at the floor,")
            print("       which is itself the reportable H8 result.")
        for arm in arms:
            result = confirm_ordering(
                arm, confirm_level, items, specs, model, resolved,
            )
            if result is None:
                unmeasured.add(f"ordering:{arm}@{confirm_level}")
            else:
                confirmation.append(result)
        print()

    # ---------------------------------------------------------------------
    # R12.3.3 — MODEL-GENERALITY probe on Phi-3.5-mini. One model establishes the
    # MECHANISM (it is in vLLM's kernel-dispatch source, so architecture-general);
    # it does not establish the RATES. vLLM #48271's leak depends on token count and
    # block-size selection, which are model-dependent, and Phi's fused 3072->16384
    # gate+up projection -- the same quirk that broke the AWQ memory budget in job 1
    # -- could take a different kernel path. Phi over Llama on purpose: the
    # architectural outlier is the higher-information second point.
    # ---------------------------------------------------------------------
    phi_results: list[DisagreementResult] = []
    phi_key = os.environ.get("QA_CONFIRM_MODEL", "phi-3.5-mini")
    phi_arms = ["bf16", "gptq-4bit"]
    if not SMOKE and phi_key in models:
        phi_cfg = models[phi_key]
        print(f"=== model-generality probe: {phi_key} {phi_arms} "
              f"at max_num_seqs={confirm_level} (R12.3.3) ===")
        phi_paths: dict[str, tuple[str, str | None]] = {}
        try:
            for arm in phi_arms:
                phi_paths[arm] = resolve_model_path(
                    arm, phi_cfg, specs, calibration, model_key=phi_key
                )
        except FileNotFoundError as exc:
            print(f"    SKIPPED — {exc}")
            phi_paths = {}

        if phi_paths:
            # Re-materialised for Phi: prompts are tokenizer-specific, so reusing
            # Qwen's would silently probe the wrong strings.
            items = build_probe(phi_cfg, phi_key)
            print(f"    probe re-materialised for {phi_key}: {len(items)} items")
            for arm in phi_arms:
                measure(arm, confirm_level, repeats=REPEATS, collect=phi_results,
                        model_key=phi_key, model_cfg=phi_cfg, paths=phi_paths)
        print()

    # The frontier CSV/JSON carry the sweep only, so the selection rule is applied
    # to exactly the cells it is defined over. The confirmation and model-generality
    # probes are separate artifacts -- they answer different questions and must not
    # be averaged into the frontier.
    write_json(OUTPUT_DIR / "determinism.json", results)
    write_csv(OUTPUT_DIR / "determinism.csv", results)
    report = summarise(results)

    if confirmation:
        write_json(OUTPUT_DIR / "determinism_confirmation.json", confirmation)
        write_csv(OUTPUT_DIR / "determinism_confirmation.csv", confirmation)
    if phi_results:
        write_json(OUTPUT_DIR / "determinism_model_generality.json", phi_results)
        write_csv(OUTPUT_DIR / "determinism_model_generality.csv", phi_results)

    print("=" * 70)
    print(f"{'format':12} {'max_seqs':>9} {'disagree':>10} {'distinct':>9}  kernels")
    for row in report["rows"]:
        print(f"{row['format']:12} {row['max_num_seqs']:>9} "
              f"{row['n_disagreeing']:>4}/{row['n_items']:<5} "
              f"{row['max_distinct_outputs']:>9}  {row['disabled_kernels'] or '-'}")
    print()

    selection = report["max_num_seqs_selection"]
    print(f"max_num_seqs : {selection['selected_max_num_seqs']} "
          f"({selection['rule']})")
    print(f"  per format : {selection['per_format_max_stable']}")
    if selection["finding_if_none"]:
        print(f"  ⚠️ {selection['finding_if_none']}")

    violations = report["non_monotonic_violations"]
    if violations:
        print("  🚨 NON-MONOTONIC FRONTIER — a smaller max_num_seqs was LESS stable "
              "than a larger one:")
        for violation in violations:
            print(f"     {violation['format']}: unstable at "
                  f"{violation['unstable_at']} but stable at {violation['stable_at']}")
        print("     'the largest stable value' is not well defined; report the "
              "frontier as a SET, not a threshold.")
    else:
        print("  frontier   : monotone (no smaller level was less stable) — R12.3.1 OK")

    floor = report["differential_noise_floor"]
    print(f"noise floor  : bf16 {floor['bf16_disagreement']:.4%}")
    print(f"  vs bf16    : {floor['differential_vs_bf16']}")
    if floor["differential_noise_exists"]:
        print(f"  ⚠️ DIFFERENTIAL NOISE EXISTS — {floor['implication']}")

    if confirmation:
        print()
        print(f"=== ordering confirmation at max_num_seqs={confirm_level} "
              f"(orderings {list(ORDERING_SEEDS)}) ===")
        held = [r for r in confirmation if r.bitwise_stable]
        for result in confirmation:
            verdict = "order-invariant" if result.bitwise_stable else (
                f"{result.n_disagreeing}/{result.n_items} changed "
                f"({result.disagreement_rate:.3%})")
            print(f"  {result.fmt:12} {verdict}")
        if len(held) == len(confirmation):
            print(f"  ✅ {confirm_level} survives ITEM REORDERING for every arm — "
                  f"freeze it")
        else:
            broke = [r.fmt for r in confirmation if not r.bitwise_stable]
            print(f"  🚨 {confirm_level} is NOT order-invariant for {broke}.")
            print("     Repeating the identical order agreed while REORDERING did "
                  "not, which is the")
            print("     mechanism itself: batch composition, not repetition, is what "
                  "moves the output.")
            print("     The single-ordering frontier OVERSTATES reproducibility. Do "
                  "not freeze without a planner ruling.")

    if phi_results:
        print()
        print(f"=== model generality: {phi_key} at max_num_seqs={confirm_level} ===")
        for result in phi_results:
            verdict = "stable" if result.bitwise_stable else (
                f"{result.n_disagreeing}/{result.n_items} disagree "
                f"({result.disagreement_rate:.3%})")
            print(f"  {result.fmt:12} {verdict}")
        print("  Mechanism is architecture-general (vLLM source); these RATES are "
              "what is model-scoped.")

    print()
    print(f"report -> {OUTPUT_DIR / 'determinism.json'}")

    if unmeasured:
        # Exit non-zero only when we could not MEASURE. Nondeterminism itself is
        # a finding and must not be reported as a job failure -- a null result
        # here is a headline H8 result and the study proceeds at the floor.
        print(f"\nUNMEASURED CELLS: {sorted(unmeasured)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
