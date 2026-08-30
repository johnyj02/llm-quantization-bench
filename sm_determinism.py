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

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import yaml  # noqa: E402

from quantartifact.determinism import (  # noqa: E402
    DETERMINISM_VERSION,
    DISABLED_KERNELS_FALLBACK,
    ORDERING_SEEDS,
    ORDERING_SEEDS_EXTENDED,
    PROBE_COMPOSITION,
    REPEATS,
    SWEEP_FLOOR,
    RepeatOutputs,
    divergence_by_level,
    order_sensitivity,
    probe_fingerprint,
)
from quantartifact.quantize import config_hash, load_specs, output_dir  # noqa: E402

CHECKPOINT_DIR = Path(os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints"))
OUTPUT_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
CONFIGS = Path(__file__).parent / "configs"

SMOKE = os.environ.get("SM_HP_SMOKE") == "1" or "--smoke" in sys.argv

# Recompute the ANALYSIS from raw passes already on disk, generating nothing.
#
# Cells are cached by tag, so adding a new analysis field (here: per-ordering accuracy
# and the signed spread) would otherwise require re-running 38,000 generations to
# surface it. The raw outputs persist, so the analysis is a pure function over them --
# this needs the probe re-materialised only for GOLD, and no engine at all.
REANALYZE = os.environ.get("SM_HP_REANALYZE") == "1" or "--reanalyze" in sys.argv

# R17.3's confirmatory extension, pre-registered in `analysis/preregistration.md` §9a.
# Ten orderings at the FROZEN batch size only, so the ordering spread becomes an SD rather
# than a range over three draws.
#
# ⚠️ Registered explicitly as a **pre-specified confirmatory extension of an exploratory
# finding**, NOT as pre-data: the 3-ordering result was already seen when §9a was written,
# and the git history shows it. Stating the sequence is what makes it read as integrity.
#
# ⚠️ This runs AFTER the BBH terminator fix (R19.1), so its numbers **supersede** the
# 3-ordering figures rather than extending them -- job 2's probe had ~2/5 of its items inert
# and its spreads are therefore attenuated. The two must not be pooled.
SD_MODE = os.environ.get("SM_HP_ORDERINGS") == "1" or "--orderings" in sys.argv

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
    autotune: bool = False,
    n_orderings: int | None = None,
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
    * 🚨 **The NUMBER OF ORDERINGS**, added 2026-08-30 after it cost a real run. The
      redesign runs every ordering inside ONE cell, so ``ordering_seed`` is 0 and
      contributes nothing — which made a 3-ordering cell and a 10-ordering cell share the
      tag ``qwen2.5-3b__bf16__at64__r2``. The R17.3 run therefore announced
      *"4 arms x 10 orderings"*, skipped all four cells as ``already measured``, reused job
      2's THREE-ordering results, and exited ``Completed`` in 417 s. An SD computed from
      that would have been a fabricated statistic in the paper.

      ⚠️ Appended only when it differs from the default, so every filename already written
      stays valid and ``reanalyze``'s parser (which reads segments 0–2) is unaffected.

      ⚠️ **The smoke could not have caught this**: ``--smoke`` writes to a separate
      ``determinism-smoke/`` prefix, so it had no prior cells to collide with. A rehearsal
      in a clean prefix cannot detect a cache collision in a dirty one.
    """
    parts = [model_key, arm, f"at{max_num_seqs}", f"r{repeats}"]
    if ordering_seed:
        parts.append(f"ord{ordering_seed}")
    if autotune:
        parts.append("autotune")
    if n_orderings is not None and n_orderings != len(ORDERING_SEEDS):
        parts.append(f"ord{n_orderings}x")
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
                    # The FROZEN token ids, so vLLM never re-tokenizes. Handing it
                    # text takes the completion path (add_special_tokens=True) and
                    # double-BOSes any model whose chat template already emits one.
                    "prompt_token_ids": prompt.prompt_token_ids,
                    # R15.3 needs divergence at THREE levels, and correctness needs
                    # gold. Carried here so the job computes it rather than leaving it
                    # to a manual pass -- "65% text / 3% correctness and 65% text /
                    # 20% correctness are different papers".
                    "gold": prompt.gold,
                    "task": task,
                    "subtask": subtask,
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
_ = runner.llm                     # force init so the ENGINE read-back runs now
load_s = time.monotonic() - started
print(f"  batch invariance: {runner.batch_invariance}", flush=True)
print(f"  engine state    : {runner.engine_state}", flush=True)

base = list(payload["items"])
out = {"passes": [], "hardware": runner.hardware, "engine_load_s": round(load_s, 1),
       "batch_invariance": runner.batch_invariance, "engine_state": runner.engine_state}

# 🚨 EVERY PASS RUNS IN THIS ONE ENGINE. That is the whole point of R15.1: my first
# version gave each ordering its own subprocess, which confounded item ordering with
# process state -- and the argument that bf16's stability ruled that out was unsound,
# because it assumed both arms are equally sensitive to whatever varies across
# processes. Holding the engine constant makes ordering the only thing that changes.
for index, spec_pass in enumerate(payload["passes"]):
    seed = spec_pass["ordering_seed"]
    items = list(base)
    # Seed 0 is the natural probe order. A non-zero seed shuffles, changing which
    # sequences are CO-RESIDENT in each batch -- the actual mechanism behind
    # batch-invariance failures. Shuffled BEFORE grouping so the order changes inside
    # each vLLM call, not merely between calls.
    if seed:
        random.Random(seed).shuffle(items)

    texts = {}
    groups = {}
    for item in items:
        groups.setdefault((item["max_gen_toks"], tuple(item["until"])), []).append(item)
    pass_started = time.monotonic()
    for (max_toks, until), group in sorted(groups.items()):
        results = runner.generate(
            [g["prompt"] for g in group], max_toks, list(until),
            token_ids=[g["prompt_token_ids"] for g in group],
        )
        for g, r in zip(group, results):
            texts[g["item_id"]] = r["text"]
    elapsed = time.monotonic() - pass_started
    out["passes"].append({"ordering_seed": seed, "outputs": texts,
                          "seconds": round(elapsed, 1)})
    rate = len(texts) / elapsed * 3600 if elapsed else 0
    print(f"  pass {index} (ordering {seed}): {len(texts)} outputs in "
          f"{elapsed / 60:.1f} min (~{rate:,.0f} gen/h)", flush=True)

open(payload["out_path"], "w").write(json.dumps(out))
"""


def measure_cell(
    arm: str,
    level: int,
    items: list[dict],
    specs: dict,
    model: dict,
    resolved: dict[str, tuple[str, str | None]],
    *,
    model_key: str = AUDIT_MODEL,
    orderings: tuple[int, ...] = ORDERING_SEEDS,
    repetition_passes: int = 2,
    disabled_kernels: str | None = None,
    autotune: bool = False,
) -> dict | None:
    """One (arm, max_num_seqs) cell: BOTH floors, from ONE engine load.

    Passes are `repetition_passes` at ordering 0 followed by one pass per further
    ordering. Comparing the ordering-0 passes gives the **repetition floor** (what §9
    binds to, R15.6); comparing across distinct orderings gives **order sensitivity**
    (the H8 headline, and a third-party reproducibility bound).

    🚨 **One engine for all of them (R15.1).** My first version ran each ordering in
    its own subprocess, so ordering was confounded with process state — and my
    argument that bf16's stability ruled that out was *unsound*: it assumed both arms
    are equally sensitive to whatever varies across processes, and GPTQ's Marlin path
    could be sensitive to workspace addresses or cuBLAS handle state in ways bf16 is
    not. Holding the engine fixed is what makes ordering the only variable, and the
    mechanism decides the write-up: reordering-changes-outputs is the
    batch-composition thesis, restarting-changes-outputs is a different claim.

    Returns the analysis dict, or None if the cell could not be measured.
    """
    # ⚠️ `n_orderings` MUST be passed. It is what distinguishes a 3-ordering cell from a
    # 10-ordering one at the same arm and level; omitting it is what made the R17.3 run
    # silently reuse job 2's results.
    tag = cell_tag(arm, level, disabled_kernels, model_key=model_key,
                   repeats=repetition_passes, autotune=autotune,
                   n_orderings=len(orderings))
    result_path = PROBE_DIR / f"{tag}.json"

    if SMOKE and result_path.exists():
        result_path.unlink()
        print(f"[smoke] discarded previous rehearsal result for {tag}")
    if result_path.exists():
        print(f"[skip ] {tag}  already measured")
        return json.loads(result_path.read_text())

    passes = [{"ordering_seed": 0} for _ in range(repetition_passes)]
    passes += [{"ordering_seed": seed} for seed in orderings if seed != 0]

    raw_path = PROBE_DIR / f"raw_{tag}.json"
    path, _ = resolved[arm]
    method, bits = arm_engine_fields(arm, specs)
    payload = {
        "spec": {
            "model_repo": path,
            "model_revision": model["revision"],
            "quant_method": method,
            "bits": bits,
            "max_num_seqs": level,
            "disabled_kernels": disabled_kernels,
            "enable_flashinfer_autotune": autotune,
        },
        "items": items,
        "passes": passes,
        "out_path": str(raw_path),
    }
    payload_path = PROBE_DIR / f"payload_{tag}.json"
    payload_path.write_text(json.dumps(payload))

    env = dict(os.environ)
    # Never inherited: BI must be ON for every study measurement, and
    # VLLM_DISABLED_KERNELS must be absent unless this cell is the Marlin probe.
    env["VLLM_BATCH_INVARIANT"] = "1"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env.pop("VLLM_DISABLED_KERNELS", None)
    if disabled_kernels:
        env["VLLM_DISABLED_KERNELS"] = disabled_kernels

    print(f"[run  ] {tag}  {len(passes)} passes "
          f"({repetition_passes} at ordering 0 + {len(passes) - repetition_passes} "
          f"reordered) x {len(items)} items")
    sys.stdout.flush()
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", PROBE_SCRIPT, str(payload_path)],
        env=env, timeout=21600,
    )
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        print(f"        FAILED after {elapsed / 60:.1f} min, exit {proc.returncode}")
        return None

    raw = json.loads(raw_path.read_text())
    outputs = [
        RepeatOutputs(
            fmt=arm, max_num_seqs=level, repeat_idx=index,
            outputs=entry["outputs"], disabled_kernels=disabled_kernels,
            ordering_seed=entry["ordering_seed"],
        )
        for index, entry in enumerate(raw["passes"])
    ]
    analysis = order_sensitivity(outputs)
    # R15.3: text alone is the least consequential of the three levels. Computed here
    # so the job reports it rather than leaving it to a manual pass afterwards.
    analysis["by_level"] = divergence_by_level(
        outputs, {i["item_id"]: i for i in items}
    )
    analysis.update({
        "arm": arm, "model_key": model_key, "max_num_seqs": level,
        "disabled_kernels": disabled_kernels, "autotune": autotune,
        "minutes": round(elapsed / 60, 1),
        "seconds_per_pass": [entry["seconds"] for entry in raw["passes"]],
        "fingerprints": [probe_fingerprint(o.outputs) for o in outputs],
        "hardware": raw.get("hardware"),
        "batch_invariance": raw.get("batch_invariance"),
        "engine_state": raw.get("engine_state"),
    })
    result_path.write_text(json.dumps(analysis, indent=2))

    rep = analysis.get("repetition_floor")
    order = analysis.get("order_divergence")
    print(f"        repetition floor {rep if rep is None else f'{rep:.3%}'}  |  "
          f"order divergence {order if order is None else f'{order:.3%}'}"
          f"{'  ← STABLE UNDER REPEATS BUT NOT REORDERING' if analysis['repeat_stable_but_order_sensitive'] else ''}")
    levels = analysis.get("by_level", {})
    if levels.get("text_divergence") is not None:
        ans = levels["answer_divergence"]["strict"]
        cor = levels["correctness_divergence"]["strict"]
        perm = levels["answer_divergence"]["permissive"]
        fmt = lambda v: "-" if v is None else f"{v:.1%}"  # noqa: E731
        print(f"        by level (strict): text {fmt(levels['text_divergence'])} -> "
              f"answer {fmt(ans)} -> correctness {fmt(cor)}   "
              f"[permissive answer {fmt(perm)}]")
        if levels.get("n_items_unscored_no_gold"):
            print(f"        ⚠️ {levels['n_items_unscored_no_gold']} items had no gold "
                  f"and were excluded from the scored levels")
    payload_path.unlink(missing_ok=True)
    return analysis


ORDER_SWEEP_LEVELS: tuple[int, ...] = (8, 32, 64)


def reanalyze() -> int:
    """Recompute every cell's analysis from its stored raw passes. No generation."""
    models = yaml.safe_load((CONFIGS / "models.yaml").read_text())["models"]
    raws = sorted(PROBE_DIR.glob("raw_*.json"))
    if not raws:
        print(f"nothing to re-analyse in {PROBE_DIR}")
        return 1
    print(f"re-analysing {len(raws)} cells from stored passes — NO generation")

    # Gold, per model. The probe is tokenizer-specific, so it is re-materialised per
    # model; item_ids are stable, which is what makes the join possible at all.
    gold: dict[str, dict[str, dict]] = {}
    for model_key in {r.stem.split("__")[0].removeprefix("raw_") for r in raws}:
        if model_key not in models:
            continue
        print(f"  materialising probe for gold: {model_key}")
        gold[model_key] = {i["item_id"]: i for i in build_probe(models[model_key], model_key)}

    out = []
    for raw_path in raws:
        raw = json.loads(raw_path.read_text())
        stem = raw_path.stem.removeprefix("raw_")
        model_key, arm = stem.split("__")[0], stem.split("__")[1]
        level = int(stem.split("__")[2].removeprefix("at"))
        outputs = [
            RepeatOutputs(fmt=arm, max_num_seqs=level, repeat_idx=i,
                          outputs=e["outputs"], ordering_seed=e["ordering_seed"])
            for i, e in enumerate(raw["passes"])
        ]
        analysis = order_sensitivity(outputs)
        analysis["by_level"] = divergence_by_level(outputs, gold.get(model_key, {}))
        analysis.update({"arm": arm, "model_key": model_key, "max_num_seqs": level,
                         "cell": stem})
        out.append(analysis)
        acc = (analysis["by_level"].get("accuracy_by_ordering") or {}).get("strict")
        spread = (analysis["by_level"].get("accuracy_spread") or {}).get("strict")
        print(f"  {stem:44} accuracy/ordering {acc}  signed spread "
              f"{'-' if spread is None else f'{spread:.2%}'}")

    (OUTPUT_DIR / "determinism_reanalysis.json").write_text(json.dumps(out, indent=2))
    print(f"report -> {OUTPUT_DIR / 'determinism_reanalysis.json'}")
    return 0


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    if REANALYZE:
        return reanalyze()

    models = yaml.safe_load((CONFIGS / "models.yaml").read_text())["models"]
    specs, calibration = load_specs(CONFIGS / "quant.yaml")
    model = models[AUDIT_MODEL]

    arms = list(AUDIT_ARMS)
    levels = list(ORDER_SWEEP_LEVELS)
    orderings = ORDERING_SEEDS

    if SD_MODE:
        # R17.3's scope, frozen: 4 arms at the FROZEN batch size ONLY, 500 items, 10
        # orderings. `gptq-8bit` is dropped because it is the dose-response arm and the
        # SD claim is about the 4-bit methods plus the control.
        #
        # ⚠️ **bf16 gets the same 10 orderings.** Exactly-zero across ten is a materially
        # stronger control than exactly-zero across three, and the control is what makes
        # the effect attributable to quantization rather than to the engine.
        #
        # ⚠️ Items are NOT traded for orderings (R17.3). Spread is PAIRED across orderings
        # so sampling variance largely cancels, but resolution does not: at 250 items one
        # item is 0.4 pp while the whole effect is ~8 items.
        arms = ["bf16", "gptq-4bit", "awq-4bit", "bnb-nf4"]
        levels = [max(ORDER_SWEEP_LEVELS)]
        orderings = ORDERING_SEEDS_EXTENDED
        print(f"ORDERINGS MODE (R17.3): {len(arms)} arms x {len(orderings)} orderings "
              f"at max_num_seqs={levels[0]} only")
    if SMOKE and SD_MODE:
        # ⚠️ ORDER MATTERS AND THIS BLOCK EXISTS BECAUSE OF IT. Written as a plain
        # `if SMOKE:` after the SD_MODE block, the smoke would have overridden arms and
        # levels while LEAVING `orderings` extended -- rehearsing 3 arms x 3 levels x 11
        # passes, which is neither the smoke's scope nor the real run's. A rehearsal must
        # exercise the path it is rehearsing, at a smaller size, not a third configuration
        # that exists nowhere.
        #
        # So: keep SD_MODE's SHAPE (frozen level, all 10 orderings -- the new code path)
        # and shrink only the arms, which the item cap in the probe already handles.
        arms = ["bf16", "gptq-4bit"]
        print(f"SMOKE + ORDERINGS: {arms} at max_num_seqs={levels[0]}, "
              f"{len(orderings)} orderings, 20 items")
    elif SMOKE:
        # bf16 + gptq-4bit + awq-4bit: the three that carry the prediction. AWQ is
        # the one that matters most (R15.2) -- it HAS the guard GPTQ lacks, so AWQ
        # coming out order-invariant shows the source-level asymmetry CAUSES the
        # behavioural one, which is the strongest form H8 can take.
        arms = ["bf16", "gptq-4bit", "awq-4bit"]
        levels = [8]
        print("SMOKE MODE: bf16 + gptq-4bit + awq-4bit at max_num_seqs=8, 20 items")

    n_probe = len(PROBE_COMPOSITION) and sum(PROBE_COMPOSITION.values())
    # ⚠️ Derived from the ACTIVE orderings, not the module default. With --orderings
    # these differ by 7, and a stale count would misreport the run's size in the log
    # and in the report -- the same declared-vs-enforced gap that has bitten five
    # times already.
    passes_per_cell = 2 + len([s for s in orderings if s != 0])

    print(f"audit model  : {AUDIT_MODEL} ({model['repo']})")
    print(f"arms         : {arms}")
    print(f"levels       : {levels}   (ORDER-induced divergence, per R15.4 --")
    print("               R8's repeat-based criterion measured an axis with no")
    print("               variance and called a 65%-unstable arm 'stable')")
    print(f"passes/cell  : {passes_per_cell}  (2 at ordering 0 for the repetition")
    print(f"               floor, + {len(orderings) - 1} reordered), orderings "
          f"{list(orderings)}")
    print(f"generations  : {len(arms) * len(levels) * passes_per_cell * n_probe:,} "
          f"for the sweep")
    print()

    print("=== materializing the probe ===")
    items = build_probe(model, AUDIT_MODEL)
    if SMOKE:
        items = items[:20]
    print(f"probe        : {len(items)} items")
    print()

    resolved: dict[str, tuple[str, str | None]] = {}
    for arm in arms:
        try:
            resolved[arm] = resolve_model_path(arm, model, specs, calibration)
        except FileNotFoundError as exc:
            print(f"BLOCKED: {exc}")
            return 1
        print(f"  {arm:10} -> {resolved[arm][0]}")
    print()

    # ⚠️ TWO lists, not one. The sweep is load-bearing — the frontier IS the
    # contribution, so a missing sweep cell must fail the job. The corroborating
    # probes are not, and conflating them cost real confusion: job 2 exited 1 and
    # therefore read `Failed` in the console on the strength of ONE optional cell
    # (`gptq-8bit@8+marlin-off`, which needs a CUDA library the image lacks) while
    # 15/15 sweep cells, the autotune contrast and the Phi probe had all completed.
    # Three separate times a complete run has had to be talked back out of looking
    # like a lost one.
    #
    # Never silently: both lists go in the report AND are printed, so a tolerated
    # absence is visible rather than swallowed. Failing closed is right; reporting a
    # finished study as a failure is not the way to do it.
    cells: list[dict] = []
    unmeasured: list[str] = []            # load-bearing: exit non-zero
    unmeasured_optional: list[str] = []   # corroborating: report, do not fail

    for level in levels:
        print(f"=== max_num_seqs = {level} ===")
        for arm in arms:
            analysis = measure_cell(arm, level, items, specs, model, resolved,
                                    orderings=orderings)
            if analysis is None:
                unmeasured.append(f"{arm}@{level}")
            else:
                cells.append(analysis)
        print()

    # ------------------------------------------------------------------
    # R15.7 — the Marlin probe must itself be an ORDERING test, and PAIRED.
    # A repeat-based fallback would print "stable" for the same reason the old sweep
    # did and prove nothing. Order-divergence dropping to 0 with Marlin disabled pins
    # the mechanism to a NAMED kernel rather than leaving it inferred.
    # ------------------------------------------------------------------
    order_sensitive = sorted({
        c["arm"] for c in cells
        if c.get("order_invariant") is False and not c["disabled_kernels"]
    })
    marlin: list[dict] = []
    marlin_pinned: list[str] = []
    marlin_missing: list[str] = []
    if order_sensitive and not SMOKE:
        print(f"=== Marlin attribution probe (PAIRED ordering test): "
              f"{order_sensitive} ===")
        print(f"    VLLM_DISABLED_KERNELS={DISABLED_KERNELS_FALLBACK}")
        print("    The unpaired half is already measured above; this is the other half.")
        for arm in order_sensitive:
            analysis = measure_cell(
                arm, SWEEP_FLOOR, items, specs, model, resolved,
                disabled_kernels=DISABLED_KERNELS_FALLBACK,
            )
            if analysis is None:
                marlin_missing.append(f"{arm}@{SWEEP_FLOOR}+marlin-off")
            else:
                marlin.append(analysis)
                baseline = next(
                    (c for c in cells if c["arm"] == arm
                     and c["max_num_seqs"] == SWEEP_FLOOR
                     and not c["disabled_kernels"]), None)
                if baseline and analysis.get("order_invariant") and \
                        baseline.get("order_invariant") is False:
                    marlin_pinned.append(arm)
                    print(f"    🔬 {arm}: order divergence "
                          f"{baseline['order_divergence']:.3%} -> 0 with Marlin "
                          f"disabled. The mechanism is a NAMED KERNEL.")
        # R16.3: attribution is an EXISTENCE claim about a named kernel, so one clean
        # arm establishes it and further arms corroborate. If NO arm pinned, the
        # mechanism is only inferred and a missing cell is load-bearing.
        if marlin_missing:
            (unmeasured_optional if marlin_pinned else unmeasured).extend(marlin_missing)
        print()

    # ------------------------------------------------------------------
    # R14.1 — autotune contrast on gptq-4bit, the arm H8's claim is about.
    # ------------------------------------------------------------------
    autotune_contrast: list[dict] = []
    if not SMOKE and "gptq-4bit" in resolved:
        print("=== autotune contrast: gptq-4bit, flashinfer autotune ON (R14.1) ===")
        analysis = measure_cell(
            "gptq-4bit", SWEEP_FLOOR, items, specs, model, resolved, autotune=True,
        )
        if analysis is None:
            unmeasured.append(f"gptq-4bit@{SWEEP_FLOOR}+autotune")
        else:
            autotune_contrast.append(analysis)
        print()

    # ------------------------------------------------------------------
    # R12.3.3 — model generality on Phi-3.5-mini, the architectural outlier.
    # ------------------------------------------------------------------
    phi: list[dict] = []
    phi_key = os.environ.get("QA_CONFIRM_MODEL", "phi-3.5-mini")
    if not SMOKE and phi_key in models:
        phi_cfg = models[phi_key]
        phi_arms = ["bf16", "gptq-4bit"]
        print(f"=== model generality: {phi_key} {phi_arms} (R12.3.3) ===")
        phi_paths: dict[str, tuple[str, str | None]] = {}
        try:
            for arm in phi_arms:
                phi_paths[arm] = resolve_model_path(
                    arm, phi_cfg, specs, calibration, model_key=phi_key)
        except FileNotFoundError as exc:
            print(f"    SKIPPED — {exc}")
            phi_paths = {}
        if phi_paths:
            # Re-materialised: prompts are tokenizer-specific, so reusing Qwen's
            # would silently probe the wrong strings.
            phi_items = build_probe(phi_cfg, phi_key)
            for arm in phi_arms:
                analysis = measure_cell(
                    arm, SWEEP_FLOOR, phi_items, specs, phi_cfg, phi_paths,
                    model_key=phi_key)
                if analysis is None:
                    # Generality on a second architecture STRENGTHENS the claim; its
                    # absence does not invalidate the Qwen frontier, which is what the
                    # gate is about. Reported, not fatal.
                    unmeasured_optional.append(f"{phi_key}/{arm}@{SWEEP_FLOOR}")
                else:
                    phi.append(analysis)
        print()

    report = {
        "gate": "G4",
        "determinism_version": DETERMINISM_VERSION,
        "audit_model": AUDIT_MODEL,
        "orderings": list(orderings),
        "sd_mode": SD_MODE,
        "order_sweep_levels": levels,
        "probe_composition": PROBE_COMPOSITION,
        "n_items": len(items),
        "sweep": cells,
        "marlin_attribution": marlin,
        "autotune_contrast": autotune_contrast,
        "model_generality": phi,
        "marlin_mechanism_pinned_on": marlin_pinned,
        "unmeasured": unmeasured,
        "unmeasured_optional": unmeasured_optional,
    }
    (OUTPUT_DIR / "determinism.json").write_text(json.dumps(report, indent=2))

    print("=" * 78)
    print(f"{'arm':12} {'max_seqs':>9} {'repetition':>12} {'ordering':>12}   verdict")
    for cell in cells:
        rep = cell.get("repetition_floor")
        order = cell.get("order_divergence")
        verdict = ""
        if cell["repeat_stable_but_order_sensitive"]:
            verdict = "🚨 stable on repeats, NOT on reordering"
        elif cell.get("order_invariant"):
            verdict = "order-invariant"
        print(f"{cell['arm']:12} {cell['max_num_seqs']:>9} "
              f"{'-' if rep is None else f'{rep:>11.3%}'} "
              f"{'-' if order is None else f'{order:>11.3%}'}   {verdict}")
    print()

    # R15.5: freezing must be EARNED on the axis that varies, not defaulted to.
    invariant = {}
    for level in levels:
        at_level = [c for c in cells if c["max_num_seqs"] == level
                    and not c["disabled_kernels"]]
        invariant[level] = bool(at_level) and all(
            c.get("order_invariant") for c in at_level)
    admissible = [level for level, ok in invariant.items() if ok]
    print(f"order-invariant at : {invariant}")
    if admissible:
        print(f"✅ freeze max_num_seqs = {max(admissible)} — the LARGEST level "
              f"order-invariant for every arm")
    else:
        print("🚨 NO level is order-invariant for every arm.")
        print("   Per R15.5 this does NOT halt the study: order-invariance is needed "
              "for third parties")
        print("   to reproduce our strings, not for our comparisons to be valid, and "
              "we fix ordering.")
        print("   But the freeze must be EARNED on this axis, so report the frontier "
              "and pick on it.")
        print("   §9's noise-floor pre-commitment binds to the REPETITION floor "
              "(R15.6), which is")
        print("   separate from this and reported separately.")
    print()
    print(f"report -> {OUTPUT_DIR / 'determinism.json'}")

    # Tolerated absences are printed whether or not they change the exit code, so
    # "it exited 0" can never be read as "everything ran".
    if unmeasured_optional:
        print(f"\nUNMEASURED, TOLERATED (corroborating, not load-bearing): "
              f"{unmeasured_optional}")
        if marlin_pinned:
            print(f"    Marlin attribution already pinned on: {marlin_pinned} — "
                  f"a further arm corroborates an existence claim (R16.3).")
        print("    Recorded in determinism.json under `unmeasured_optional`. State "
              "the absence in the paper; do not imply full coverage.")
    if unmeasured:
        print(f"\nUNMEASURED, LOAD-BEARING: {unmeasured}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
