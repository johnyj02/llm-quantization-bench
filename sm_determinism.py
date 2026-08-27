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
the largest stable value, so testing 64 first and stopping on success answers it
in one level (7,500 generations) instead of four (30,000). Every (arm, level)
result is checkpointed, so a timeout costs one cell rather than the job.
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
    PROBE_COMPOSITION,
    REPEATS,
    SWEEP_FLOOR,
    DisagreementResult,
    RepeatOutputs,
    compute_disagreement,
    distinct_output_histogram,
    next_sweep_level,
    probe_fingerprint,
    summarise,
    sweep_plan,
    write_csv,
    write_json,
)
from quantartifact.quantize import config_hash, load_specs, output_dir  # noqa: E402

CHECKPOINT_DIR = Path(os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints"))
OUTPUT_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
CONFIGS = Path(__file__).parent / "configs"
PROBE_DIR = CHECKPOINT_DIR / "determinism"

SMOKE = os.environ.get("SM_HP_SMOKE") == "1" or "--smoke" in sys.argv

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


def cell_tag(arm: str, max_num_seqs: int, disabled_kernels: str | None) -> str:
    """Filesystem-safe identity for one probe cell. Also its checkpoint key."""
    return f"{arm}_at_{max_num_seqs}" + ("_fallback" if disabled_kernels else "")


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


def resolve_model_path(arm: str, model: dict, specs: dict, calibration) -> tuple[str, str | None]:
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
    path = output_dir(CHECKPOINT_DIR, AUDIT_MODEL, spec, digest)
    if not (path / "config.json").exists():
        raise FileNotFoundError(
            f"no checkpoint for {AUDIT_MODEL}/{arm} at {path}. Job 1 (quantize) "
            f"must complete first, and its checkpoint_s3_uri must match this "
            f"job's so SageMaker restores it. Expected recipe hash {digest}."
        )
    return str(path), digest


PROBE_SCRIPT = r"""
import json, os, sys
sys.path.insert(0, "/opt/ml/code/src")
from quantartifact.generate import EngineSpec, VLLMRunner

payload = json.loads(open(sys.argv[1]).read())
spec = EngineSpec(**payload["spec"])
runner = VLLMRunner(spec)          # preflight() asserts every BI invariant here

items = payload["items"]
out = {"repeats": [], "hardware": runner.hardware}
for repeat_idx in range(payload["repeats"]):
    texts = {}
    # Group by (max_gen_toks, until) so each vLLM call is a single homogeneous
    # batch, exactly as run_cell does it. Sorted for a deterministic call order:
    # the probe must differ across repeats ONLY by engine nondeterminism.
    groups = {}
    for item in items:
        groups.setdefault((item["max_gen_toks"], tuple(item["until"])), []).append(item)
    for (max_toks, until), group in sorted(groups.items()):
        results = runner.generate([g["prompt"] for g in group], max_toks, list(until))
        for g, r in zip(group, results):
            texts[g["item_id"]] = r["text"]
    out["repeats"].append(texts)
    print(f"  repeat {repeat_idx}: {len(texts)} outputs", flush=True)

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
    tag = cell_tag(arm, max_num_seqs, disabled_kernels)
    result_path = PROBE_DIR / f"{tag}.json"

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
        "repeats": REPEATS,
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

    print(f"[run ] {tag}  {REPEATS} repeats x {len(items)} items")
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
    repeats = [
        RepeatOutputs(
            fmt=arm, max_num_seqs=max_num_seqs, repeat_idx=i,
            outputs=texts, disabled_kernels=disabled_kernels,
        )
        for i, texts in enumerate(raw["repeats"])
    ]
    result = compute_disagreement(repeats)

    result_path.write_text(json.dumps({
        "result": dataclasses.asdict(result),
        "fingerprints": [probe_fingerprint(r.outputs) for r in repeats],
        "distinct_output_histogram": {
            str(k): v for k, v in distinct_output_histogram(repeats).items()
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

    def measure(arm: str, level: int, fallback: str | None = None) -> None:
        """Run one cell and record it, treating an unmeasured cell as UNSTABLE.

        That default matters: if a cell crashes and we simply skip it,
        `next_sweep_level` sees the level as incomplete and retries forever, and
        `select_max_num_seqs` could pick a level on the strength of a measurement
        that never happened. Recording it as maximally unstable makes a missing
        measurement fail closed.
        """
        path, _ = resolved[arm]
        method, bits = arm_engine_fields(arm, specs)
        result = run_probe_cell(
            arm, path, model["revision"], quant_method=method, bits=bits,
            max_num_seqs=level, items=items, disabled_kernels=fallback,
        )
        if result is None:
            unmeasured.add(cell_tag(arm, level, fallback))
            result = DisagreementResult(
                fmt=arm, max_num_seqs=level, n_items=len(items),
                n_disagreeing=len(items), disagreement_rate=1.0,
                max_distinct_outputs=REPEATS, bitwise_stable=False,
                disabled_kernels=fallback,
            )
        results.append(result)

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
    crashed_arms = {tag.split("_at_")[0] for tag in unmeasured}
    unstable = sorted(
        {r.fmt for r in results if not r.bitwise_stable and r.fmt not in crashed_arms}
    )
    if unstable and not SMOKE:
        print(f"=== Marlin fallback probe: {unstable} ===")
        print(f"    VLLM_DISABLED_KERNELS={DISABLED_KERNELS_FALLBACK}")
        for arm in unstable:
            measure(arm, SWEEP_FLOOR, fallback=DISABLED_KERNELS_FALLBACK)
        print()

    write_json(OUTPUT_DIR / "determinism.json", results)
    write_csv(OUTPUT_DIR / "determinism.csv", results)
    report = summarise(results)

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

    floor = report["differential_noise_floor"]
    print(f"noise floor  : bf16 {floor['bf16_disagreement']:.4%}")
    print(f"  vs bf16    : {floor['differential_vs_bf16']}")
    if floor["differential_noise_exists"]:
        print(f"  ⚠️ DIFFERENTIAL NOISE EXISTS — {floor['implication']}")
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
