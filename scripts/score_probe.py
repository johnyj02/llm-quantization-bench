#!/usr/bin/env python3
"""Score job 2's determinism probe BY TASK. Run with .venv-harness (needs lm-eval).

    .venv-harness/bin/python scripts/score_probe.py

WHY THIS EXISTS. The determinism job reported accuracy per *cell*, blended over
gsm8k_cot + arc_challenge_chat + six BBH subtasks. Planner ruling R18.2 asks for
`gsm8k_cot`'s own strict-vs-permissive gap, because that is the **other end** of the
marker-naturalness gradient from `gsm8k`'s hash marker and the direct test of the
monotonicity prediction. A blended number cannot show it.

The raw passes store `outputs` keyed `task:subtask:item_id`, so the task split is free;
only **gold** has to be recovered, which means re-materialising the probe through
lm-eval. That is the same renderer and the same pinned `FEWSHOT_SEED` as G2 and as the
job itself, and it needs no GPU and no torch --- which is the whole reason materialization
was factored out this way.

⚠️ Uses `.venv-harness`, NOT `.venv`: lm-eval lives only in the former.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PROBE_DIR = REPO / "artifacts" / "determinism-probe"
OUT = REPO / "artifacts" / "scores" / "scores_determinism-probe.json"

# Mirrors sm_determinism.PROBE_COMPOSITION. Duplicated rather than imported because
# importing sm_determinism pulls in the serving stack, which the harness venv lacks.
PROBE_COMPOSITION = {"gsm8k_cot": 200, "arc_challenge_chat": 100, "bbh": 200}
AUDIT_MODEL = "qwen2.5-3b"
RUNGS = ("strict", "strict_last", "permissive")


def bbh_probe_split(total: int, subtasks: list[str]) -> dict[str, int]:
    base, extra = divmod(total, len(subtasks))
    return {name: base + (1 if i < extra else 0) for i, name in enumerate(subtasks)}


def build_gold(model_key: str) -> dict[str, dict]:
    """item_id -> {gold, task, subtask}, keyed exactly as the raw passes are."""
    import yaml  # noqa: PLC0415 - harness venv only
    from transformers import AutoTokenizer  # noqa: PLC0415 - harness venv only

    from quantartifact.tasks.materialize import (  # noqa: PLC0415 - pulls in transformers
        render_task_prompts,
    )

    configs = REPO / "configs"
    models = yaml.safe_load((configs / "models.yaml").read_text())["models"]
    tasks = yaml.safe_load((configs / "tasks.yaml").read_text())["tasks"]
    model = models[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model["repo"], revision=model["revision"])

    plan: list[tuple[str, str, str | None, int]] = [
        ("gsm8k_cot", "gsm8k_cot", None, PROBE_COMPOSITION["gsm8k_cot"]),
        ("arc_challenge_chat", "arc_challenge_chat", None,
         PROBE_COMPOSITION["arc_challenge_chat"]),
    ]
    subtasks = [s["name"] for s in tasks["bbh"]["subtasks"]]
    prefix = tasks["bbh"]["lm_eval_task_prefix"]
    for name, count in bbh_probe_split(PROBE_COMPOSITION["bbh"], subtasks).items():
        plan.append((f"{prefix}{name}", "bbh", name, count))

    gold: dict[str, dict] = {}
    for lm_eval_task, task, subtask, limit in plan:
        config = tasks[task]
        # ⚠️ Must match sm_determinism.build_probe EXACTLY, or the item_ids differ and the
        # join silently drops everything. Same renderer, same pinned FEWSHOT_SEED, same
        # `num_fewshot` from the task config, same chat-template settings.
        prompts = render_task_prompts(
            lm_eval_task, tokenizer,
            task=task, subtask=subtask,
            prompt_variant="fewshot_cot",
            model_key=model_key,
            model_repo=model["repo"], model_revision=model["revision"],
            num_fewshot=config.get("num_fewshot"),
            apply_chat_template=True, fewshot_as_multiturn=False,
            limit=limit,
        )
        for prompt in prompts:
            key = f"{task}:{subtask or '-'}:{prompt.item_id}"
            gold[key] = {"gold": prompt.gold, "task": task, "subtask": subtask}
    return gold


def main() -> int:
    from quantartifact.scoring import extract  # noqa: PLC0415 - harness venv only

    raws = sorted(PROBE_DIR.glob("raw_*.json"))
    if not raws:
        raise SystemExit(f"no raw_*.json under {PROBE_DIR} — sync from S3 first")

    gold = build_gold(AUDIT_MODEL)
    print(f"gold          : {len(gold)} items materialised for {AUDIT_MODEL}")

    report: dict = {"audit_model": AUDIT_MODEL, "cells": {}}
    for path in raws:
        stem = path.stem.removeprefix("raw_")
        if not stem.startswith(AUDIT_MODEL):
            continue
        raw = json.loads(path.read_text())
        # Ordering 0 only: this is a per-task accuracy question, not an ordering one,
        # and mixing orderings would average over the very effect H8 measures.
        first = next((p for p in raw["passes"] if p.get("ordering_seed") == 0), None)
        if first is None:
            continue

        by_task: dict[str, list[dict]] = defaultdict(list)
        for key, text in first["outputs"].items():
            meta = gold.get(key)
            if meta is None:
                continue
            scored = extract.score_generation(
                text, meta["gold"], meta["task"], meta["subtask"],
                rungs=RUNGS, math_verify_available=False,
            )
            label = meta["task"] if not meta["subtask"] else \
                f"{meta['task']}/{meta['subtask']}"
            by_task[label].append(
                {rung: bool(scored[rung]["correct"]) for rung in RUNGS})

        entry: dict = {}
        for label, rows in sorted(by_task.items()):
            entry[label] = {
                "n_items": len(rows),
                **{f"acc_{rung}": statistics.mean(r[rung] for r in rows)
                   for rung in RUNGS},
            }
            entry[label]["rung_gap"] = (
                entry[label]["acc_permissive"] - entry[label]["acc_strict"])
        report["cells"][stem] = entry

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True))

    print()
    print(f"{'cell':46} {'task':22} {'n':>4} {'strict':>8} {'perm':>8} {'gap':>8}")
    print("-" * 100)
    for cell, tasks_ in report["cells"].items():
        for label, e in tasks_.items():
            print(f"{cell:46} {label:22} {e['n_items']:>4} "
                  f"{e['acc_strict']:>7.1%} {e['acc_permissive']:>7.1%} "
                  f"{e['rung_gap']:>+7.1%}")
    print(f"\nreport -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
