#!/usr/bin/env python3
"""Print a sample of generations in full, so a human can see whether they are sane.

    .venv/bin/python scripts/peek_generations.py --run-id pilot-v2
    .venv/bin/python scripts/peek_generations.py --run-id main-v1 --n 3 --task bbh
    .venv/bin/python scripts/peek_generations.py --run-id main-v1 --cell 'llama-3.1-8b|bf16'

🚨 WHY THIS EXISTS, and why it is a standing obligation rather than a debugging tool.

Three defects in this project were plainly visible in the raw generation text and were found
weeks later by a scorer instead, because nobody looked at the text:

* **The BBH terminator zeroing.** `generation_text` was the empty string with
  `n_gen_tokens=1` and `stop_reason="Q"` on five of six subtasks, for every arm including
  bf16 Llama-8B. Reading three records would have taken ten seconds and shown it immediately.
  Instead it survived a full pilot run and was caught only when accuracy came out at 0.0%.
* **The double BOS on Llama.** `prompt_text` began with two `<|begin_of_text|>` tokens.
  Visible in the stored prompt.
* **GSM8K's unreduced gold.** `gold` was the entire worked solution ending `#### 18` while
  the extractor produced `18`, so every item scored wrong. Visible by reading one record.

The common shape is that each was a WRONG NUMBER that looked plausible in aggregate, and
obvious in a single record. Aggregates hide exactly what a sample reveals, which is why this
check is not redundant with the scorers.

⚠️ This prints the FULL text, deliberately truncating nothing. Truncated output is how you
miss a generation that is degenerate after the first 200 characters.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# The fields worth reading before trusting a run. Each one has caught something.
HEADER_FIELDS = (
    "run_id", "cell_id", "task", "subtask", "item_id",
    "finish_reason", "stop_reason", "n_gen_tokens", "max_gen_toks",
    "until", "template_level", "max_num_seqs", "repeat_idx", "invocation_id",
    "model_revision", "quant_config_hash", "schema_version",
)


def load(root: Path, task: str | None, cell: str | None) -> list[dict]:
    records: list[dict] = []
    for path in sorted(root.rglob("*.jsonl.gz")):
        if cell and cell.replace("|", "__") not in path.parent.name:
            continue
        with gzip.open(path, "rt") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if task and record.get("task") != task:
                    continue
                records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read a sample of generations to check they are sane")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n", type=int, default=3,
                        help="how many to print (default 3, the standing obligation)")
    parser.add_argument("--task", default=None, help="restrict to one task")
    parser.add_argument("--cell", default=None, help="restrict to one cell_id")
    parser.add_argument("--seed", type=int, default=None,
                        help="fix the sample for a reproducible check; default is random, "
                             "because a fixed sample stops being a sample")
    args = parser.parse_args()

    root = REPO / "artifacts" / "generations" / args.run_id
    if not root.is_dir():
        raise SystemExit(f"no generations at {root}; sync them from S3 first")

    records = load(root, args.task, args.cell)
    if not records:
        raise SystemExit(
            f"no records matched under {root} "
            f"(task={args.task!r}, cell={args.cell!r})"
        )

    rng = random.Random(args.seed)
    sample = rng.sample(records, min(args.n, len(records)))
    print(f"{len(records):,} records available; showing {len(sample)}\n")

    for index, record in enumerate(sample, 1):
        print("=" * 78)
        print(f"SAMPLE {index}/{len(sample)}")
        print("=" * 78)
        for field in HEADER_FIELDS:
            if field in record:
                print(f"  {field:20} {record[field]!r}")
        print("\n  --- prompt_text (last 300 chars; check for a doubled BOS at the start) ---")
        prompt = record.get("prompt_text") or ""
        print(f"  starts: {prompt[:80]!r}")
        print(f"  ends:   {prompt[-300:]!r}")
        print("\n  --- gold (is it REDUCED to the answer, or the whole solution?) ---")
        print(f"  {record.get('gold')!r}")
        print("\n  --- generation_text, IN FULL ---")
        print(record.get("generation_text"))
        print()

    print("=" * 78)
    print("""Check each of these before trusting the run:
  1. Is `generation_text` non-empty and a plausible length for the task?
  2. Does it actually ANSWER, or stop mid-reasoning? Cross-check `finish_reason`
     and `stop_reason`: a `stop_reason` from `until` with no answer is a truncation.
  3. Is `n_gen_tokens` far below `max_gen_toks`? On a CoT task that is suspicious.
  4. Does `prompt_text` start with exactly ONE BOS token for this model?
  5. Is `gold` the bare answer, or an unreduced solution the extractor cannot match?
  6. Is `until` the set you intended for this `template_level`?
  7. Is the text degenerate (repeating) after the first few lines?""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
