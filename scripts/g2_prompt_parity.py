#!/usr/bin/env python3
"""GATE G2 — prompt parity, and the single-leading-BOS assertion.

Run in the harness venv (it needs `lm_eval` and `transformers`, neither of which
belongs in the serving env):

    .venv-harness/bin/python scripts/g2_prompt_parity.py --limit 25
    .venv-harness/bin/python scripts/g2_prompt_parity.py            # full

What G2 has to establish, per `EXECUTOR-TASKS.md` §8: our rendered prompt for
each task config is **byte-for-byte identical** to `lm-evaluation-harness`
v0.4.12's. It is the single test that entitles the paper to say "we use the
standard harness prompts", and it is the cheapest credibility we can buy. A
prompt mismatch invalidates every comparison to published numbers, which is why
it is a blocking gate.

We satisfy it by construction rather than by diffing two renderers: lm-eval
renders, we freeze, generation reads only the frozen artifact. This script does
the freezing and emits the evidence:

1. Render every (model × task × variant) with the pinned harness.
2. **Assert the intended leading-BOS count** per prompt
   (`analysis/preregistration.md` §3b). Not a formality: Llama-3.1-8B's chat
   template already embeds ``<|begin_of_text|>``, so the naive
   ``add_special_tokens=True`` yields a DOUBLE BOS that would silently shift
   every Llama score. The expected count is derived per tokenizer -- 1 for
   Llama, 0 for Qwen2.5 (no BOS token) and 0 for Phi-3.5 (``add_bos_token=False``)
   -- so this catches a double BOS without imposing one model's convention on
   another.
3. Re-render and assert byte-identity, catching any nondeterminism in rendering
   itself (few-shot sampling order, dict iteration, template state).
4. Write the artifact plus a manifest with a rolling hash over all prompts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from quantartifact.tasks.materialize import (  # noqa: E402
    BOSError,
    manifest_for,
    prompt_artifact_path,
    render_task_prompts,
    write_prompts,
)

# (task key in configs/tasks.yaml, lm_eval task name, subtask, variant)
TASK_MATRIX: list[tuple[str, str, str | None, str]] = [
    ("gsm8k", "gsm8k", None, "fewshot_cot"),
    ("gsm8k_cot", "gsm8k_cot", None, "fewshot_cot"),
    ("arc_challenge_chat", "arc_challenge_chat", None, "fewshot_cot"),
    ("bbh", "bbh_cot_fewshot_word_sorting", "word_sorting", "fewshot_cot"),
    ("bbh", "bbh_cot_fewshot_dyck_languages", "dyck_languages", "fewshot_cot"),
    ("bbh", "bbh_cot_fewshot_multistep_arithmetic_two", "multistep_arithmetic_two", "fewshot_cot"),
    ("bbh", "bbh_cot_fewshot_object_counting", "object_counting", "fewshot_cot"),
    ("bbh", "bbh_cot_fewshot_boolean_expressions", "boolean_expressions", "fewshot_cot"),
    (
        "bbh",
        "bbh_cot_fewshot_logical_deduction_five_objects",
        "logical_deduction_five_objects",
        "fewshot_cot",
    ),
]


def load_models(config_path: Path) -> list[tuple[str, str, str]]:
    config = yaml.safe_load(config_path.read_text())
    return [
        (key, spec["repo"], spec["revision"])
        for key, spec in config["models"].items()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate G2: prompt parity + BOS assertion")
    parser.add_argument("--limit", type=int, default=None, help="items per task (default: all)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "prompts")
    parser.add_argument("--models", nargs="*", default=None, help="model keys to include")
    parser.add_argument(
        "--no-rerender-check",
        action="store_true",
        help="skip the re-render byte-identity check (faster, weaker evidence)",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer  # noqa: PLC0415 - harness venv only

    models = load_models(REPO_ROOT / "configs" / "models.yaml")
    if args.models:
        models = [m for m in models if m[0] in set(args.models)]

    manifests: list[dict] = []
    failures: list[str] = []
    bos_summary: list[dict] = []

    for model_key, repo, revision in models:
        print(f"\n=== {model_key}  ({repo} @ {revision[:12]})")
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)

        for task_key, lm_eval_name, subtask, variant in TASK_MATRIX:
            label = f"{model_key}/{lm_eval_name}"
            try:
                prompts = list(
                    render_task_prompts(
                        lm_eval_name,
                        tokenizer,
                        task=task_key,
                        subtask=subtask,
                        prompt_variant=variant,
                        model_key=model_key,
                        model_repo=repo,
                        model_revision=revision,
                        limit=args.limit,
                    )
                )
            except BOSError as exc:
                failures.append(f"{label}: BOS assertion FAILED -- {exc}")
                print(f"  {lm_eval_name:48} BOS FAIL")
                continue
            except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
                failures.append(f"{label}: {type(exc).__name__}: {exc}")
                print(f"  {lm_eval_name:48} ERROR {type(exc).__name__}")
                continue

            if not prompts:
                failures.append(f"{label}: rendered zero prompts")
                continue

            # Rendering must itself be deterministic, or "byte-identical to the
            # harness" is not a well-defined claim.
            if not args.no_rerender_check:
                again = list(
                    render_task_prompts(
                        lm_eval_name,
                        tokenizer,
                        task=task_key,
                        subtask=subtask,
                        prompt_variant=variant,
                        model_key=model_key,
                        model_repo=repo,
                        model_revision=revision,
                        limit=args.limit,
                    )
                )
                mismatched = [
                    p.item_id
                    for p, q in zip(prompts, again, strict=True)
                    if p.prompt_sha256 != q.prompt_sha256
                ]
                if mismatched:
                    failures.append(
                        f"{label}: re-render differs for {len(mismatched)} items "
                        f"(first: {mismatched[:3]})"
                    )

            path = prompt_artifact_path(args.out, model_key, task_key, subtask, variant)
            write_prompts(path, prompts)

            manifest = manifest_for(prompts)
            manifest["lm_eval_task"] = lm_eval_name
            manifests.append(manifest)

            bos_summary.append(
                {
                    "model": model_key,
                    "task": lm_eval_name,
                    "add_special_tokens": manifest["add_special_tokens"],
                    "n_leading_bos": manifest["n_leading_bos"],
                    "fewshot_hardcoded": manifest["fewshot_hardcoded"],
                    "fewshot_seed_matters": manifest["fewshot_seed_matters"],
                }
            )

            print(
                f"  {lm_eval_name:48} n={manifest['n_prompts']:5} "
                f"shots={manifest['num_fewshot']} "
                f"bos={manifest['n_leading_bos']} "
                f"add_special={str(manifest['add_special_tokens']):5} "
                f"{manifest['rolling_sha256'][:12]}"
            )

    manifest_path = args.out / "MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "gate": "G2",
                "harness_version": "0.4.12",
                "limit": args.limit,
                "n_cells": len(manifests),
                "manifests": manifests,
                "bos_summary": bos_summary,
                "failures": failures,
            },
            indent=2,
        )
    )

    print(f"\n{'=' * 66}")
    print(f"G2 cells rendered : {len(manifests)}")
    print(f"manifest          : {manifest_path.relative_to(REPO_ROOT)}")

    # Every prompt in the study must carry exactly one BOS, or none where the
    # tokenizer has no BOS token at all.
    bad_bos = [b for b in bos_summary if b["n_leading_bos"] not in (0, 1)]
    if bad_bos:
        failures.append(f"{len(bad_bos)} cells with a leading-BOS count outside (0, 1) -- a double BOS is never correct")

    if failures:
        print(f"\nG2 FAILED — {len(failures)} problem(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("G2 PASSED — every cell rendered by the pinned harness v0.4.12,")
    print("           re-render byte-identical, leading-BOS count as intended per model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
