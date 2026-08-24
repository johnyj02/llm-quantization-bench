"""Generation and score record schemas, JSONL I/O, and the resume index.

Deliberately separate from ``generate.py`` so that the record schema, the join
key and the resume logic are testable on a laptop with no vLLM and no GPU --
which is also the property that makes the released artifact re-analysable by
anyone.

Two fields are load-bearing and unreconstructable after the fact:

* ``finish_reason`` -- H1 (truncation, the paper's lead hypothesis) is
  impossible without it.
* ``n_gen_tokens``  -- same, and it settles the arXiv:2606.25519 vs
  arXiv:2504.04823 contradiction about whether quantization lengthens CoT.

``GenerationRecord.validate()`` therefore REFUSES to emit a record missing
either. There is no recovering them from a re-read of the text, so a silent
None would poison the whole run.

Schema: ``EXECUTOR-TASKS.md`` §7.1 (generations) and §7.2 (scores).
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"

# The join key across generations and EVERY scorer. Enforced as a uniqueness
# constraint by the loader, which fails loudly on a duplicate -- a silently
# duplicated item would double-weight it in the GLMM.
JOIN_KEY: tuple[str, ...] = (
    "run_id",
    "cell_id",
    "task",
    "subtask",
    "item_id",
    "budget_level",
    "prompt_variant",
    "sample_idx",
    "repeat_idx",
)

VALID_FINISH_REASONS = frozenset({"stop", "length", "abort"})


class SchemaError(ValueError):
    """A record violates the interface contract. Always fatal, never warned."""


@dataclasses.dataclass
class GenerationRecord:
    """One generation. Complete enough that ANY future scorer can run over it.

    Field order and names follow `EXECUTOR-TASKS.md` §7.1 so a reviewer can
    diff the released JSONL against the spec.
    """

    run_id: str                     # pilot-v1 | main-v1 | determinism-v1 | anchor-v1
    cell_id: str                    # e.g. "qwen2.5-3b|gptq-4bit"
    model_repo: str
    model_revision: str             # COMMIT SHA, never a tag
    quant_method: str | None
    bits: int
    quant_config_hash: str | None
    task: str
    subtask: str | None             # REQUIRED for bbh
    item_id: str
    gold: str
    prompt_variant: str             # fewshot_cot | zeroshot
    scoring_mode: str               # generate | loglikelihood
    budget_level: str               # 1x | 2x | 4x
    max_gen_toks: int | None
    decoding: str
    sample_idx: int
    seed: int
    repeat_idx: int                 # for the determinism probe's 3 repeats
    prompt_text: str
    prompt_sha256: str
    generation_text: str            # FULL text, NEVER truncated for storage
    n_prompt_tokens: int
    n_gen_tokens: int | None        # REQUIRED for scoring_mode=generate
    finish_reason: str | None       # REQUIRED for scoring_mode=generate
    stop_reason: str | None
    option_logprobs: list[dict[str, Any]] | None  # loglikelihood mode only
    engine: str
    batch_invariant: bool
    max_num_seqs: int
    disabled_kernels: str | None
    gpu: str
    cuda: str | None
    driver: str | None
    env_hash: str
    wall_ms: int
    schema_version: str = SCHEMA_VERSION

    def key(self) -> tuple:
        return tuple(getattr(self, field) for field in JOIN_KEY)

    def validate(self) -> None:
        """Fail loudly on anything unreconstructable or contract-violating."""
        if self.scoring_mode == "generate":
            # These two are the whole reason this method exists.
            if self.finish_reason is None:
                raise SchemaError(
                    f"finish_reason is REQUIRED for scoring_mode=generate "
                    f"({self.cell_id}/{self.task}/{self.item_id}). H1 is "
                    f"impossible without it and it cannot be recovered later."
                )
            if self.finish_reason not in VALID_FINISH_REASONS:
                raise SchemaError(
                    f"finish_reason {self.finish_reason!r} not in {sorted(VALID_FINISH_REASONS)}"
                )
            if self.n_gen_tokens is None:
                raise SchemaError(
                    f"n_gen_tokens is REQUIRED for scoring_mode=generate "
                    f"({self.cell_id}/{self.task}/{self.item_id})."
                )
        elif self.scoring_mode == "loglikelihood":
            if self.option_logprobs is None:
                raise SchemaError(
                    "option_logprobs is REQUIRED for scoring_mode=loglikelihood"
                )
        else:
            raise SchemaError(f"unknown scoring_mode {self.scoring_mode!r}")

        if self.task == "bbh" and not self.subtask:
            raise SchemaError("subtask is REQUIRED for task=bbh")

        # Determinism contract. A record produced with these wrong is not
        # comparable to the rest of the study, so it must never reach disk.
        if self.batch_invariant is not True:
            raise SchemaError(
                "batch_invariant must be True -- uncontrolled batch "
                "nondeterminism can manufacture or erase the effect (vLLM #9567)"
            )
        if len(self.model_revision) < 7 or self.model_revision != self.model_revision.strip():
            raise SchemaError(f"model_revision must be a commit SHA, got {self.model_revision!r}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def append_records(path: Path, records: Sequence[GenerationRecord]) -> int:
    """Append-only, gzipped, one file per cell. Validates before writing.

    Append-only and flushed per call so an interrupted run loses at most the
    current batch. Nothing is ever rewritten in place.
    """
    if not records:
        return 0
    for record in records:
        record.validate()

    path.parent.mkdir(parents=True, exist_ok=True)
    # gzip append is legal: concatenated gzip members decode as one stream.
    with gzip.open(path, "at", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    return len(records)


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream records back. Tolerates a truncated final line.

    A half-written final line is the expected signature of a killed process, so
    it is skipped rather than treated as corruption -- the resume path then
    simply regenerates that one item.
    """
    if not path.exists():
        return
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Truncated tail from an interrupted write. Stop here; anything
                # after it is unreadable anyway.
                return


def build_resume_index(path: Path) -> set[tuple]:
    """Keys already generated in this cell file, for idempotent re-runs.

    `EXECUTOR-TASKS.md` §7.3: re-running a command with identical arguments
    must produce zero new work and exit 0.
    """
    return {
        tuple(record.get(field) for field in JOIN_KEY) for record in read_records(path)
    }


def assert_join_key_unique(records: Sequence[dict[str, Any]]) -> None:
    """Fail loudly on a duplicated join key.

    Loudly, because a duplicate silently double-weights an item in the GLMM and
    in every rate we report.
    """
    seen: dict[tuple, int] = {}
    for index, record in enumerate(records):
        key = tuple(record.get(field) for field in JOIN_KEY)
        if key in seen:
            described = {JOIN_KEY[i]: key[i] for i in range(len(JOIN_KEY))}
            raise SchemaError(
                f"duplicate join key at records[{index}] (first seen at "
                f"records[{seen[key]}]): {described}"
            )
        seen[key] = index


def cell_path(root: Path, run_id: str, cell_id: str, task: str, subtask: str | None,
              budget_level: str, prompt_variant: str) -> Path:
    """One JSONL per (cell x task x subtask x budget x variant).

    Finer-grained than "one file per cell" so that parallel jobs never contend
    on a single handle, and so a failed BBH subtask can be re-run without
    touching the rest -- the same reason the sibling project split its output
    per model (see notes/reuse.md §3).
    """
    safe_cell = cell_id.replace("|", "__").replace("/", "_")
    parts = [task] + ([subtask] if subtask else []) + [prompt_variant, budget_level]
    name = "__".join(parts)
    return root / run_id / safe_cell / f"{name}.jsonl.gz"


# ---------------------------------------------------------------------------
# Score records (§7.2)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ScoreRecord:
    """One (generation x scorer) result. Keyed by the same JOIN_KEY."""

    run_id: str
    cell_id: str
    task: str
    subtask: str | None
    item_id: str
    budget_level: str
    prompt_variant: str
    sample_idx: int
    repeat_idx: int
    scorer_name: str
    scorer_version: str
    extracted: str | None
    correct: bool | None
    compliance: str              # COMPLIANT | RECOVERABLE | ABSENT
    truncated: bool              # orthogonal to compliance
    termination: str             # TRUNC_BUDGET | TRUNC_STOPSEQ | COMPLETE
    # `found` vs `committed`: extraction succeeding is not the model
    # committing. The §4.0 always-compliant stratum is built from `committed`.
    extraction_found: bool = False
    extraction_committed: bool = False
    extraction_rule: str | None = None
    capitalization_failure: bool = False
    gold_in_trace: bool = False               # H2 permissive
    gold_in_trace_conservative: bool = False  # H2 conservative
    small_gold: bool = False
    judge_verdict: str | None = None
    judge_parse_ok: bool | None = None
    judge_model: str | None = None
    notes: str | None = None
    schema_version: str = SCHEMA_VERSION

    def key(self) -> tuple:
        return tuple(getattr(self, field) for field in JOIN_KEY)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def append_scores(path: Path, records: Sequence[ScoreRecord]) -> int:
    """Scores go to files keyed by (cell, scorer_name, scorer_version).

    NEVER overwritten: a version bump writes a new file, so every number in the
    paper can be traced to the exact scorer that produced it.
    """
    if not records:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return len(records)


def score_path(root: Path, run_id: str, cell_id: str, scorer_name: str,
               scorer_version: str) -> Path:
    safe_cell = cell_id.replace("|", "__").replace("/", "_")
    return root / run_id / safe_cell / f"{scorer_name}__v{scorer_version}.jsonl.gz"


def env_hash(extra: dict[str, Any] | None = None) -> str:
    """Stable hash over the environment facts that can move a score.

    Included because reproducibility holds only within identical hardware and
    versions, so the record must carry enough to prove two runs were comparable
    rather than merely asserting it.
    """
    facts = {
        "schema_version": SCHEMA_VERSION,
        "VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT"),
        "VLLM_ENABLE_V1_MULTIPROCESSING": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
        "VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND"),
        "VLLM_DISABLED_KERNELS": os.environ.get("VLLM_DISABLED_KERNELS"),
    }
    if extra:
        facts.update(extra)
    payload = json.dumps(facts, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
