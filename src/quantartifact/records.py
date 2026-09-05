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

SCHEMA_VERSION = "1.5.0"   # 1.0.0 -> 1.1.0: added invocation_id (R21.3)
#                          # 1.1.0 -> 1.2.0: added until (self-classifying artifact)
#                          # 1.2.0 -> 1.3.0: added template_level (ditto)
#                          # 1.3.0 -> 1.4.0: added until_applied (R34; see the field)
#                          # 1.4.0 -> 1.5.0: added stop_string_hit + n_gen_tokens_scored (R35.Q4)

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


def _expected_max_gen_toks(task: str, budget_level: str) -> int | None:
    """`configs/tasks.yaml`'s declared cap for a task at a budget level, or None.

    Returns None rather than raising when the answer is genuinely unknown: an unrecognised
    task, a level the task does not declare, or a config that cannot be read. `validate()`
    then skips the comparison instead of rejecting a record it cannot judge.

    ⚠️ **None means "I cannot know", never "it is fine", and the distinction is load-bearing
    here.** `compliance.classify_termination` used to swallow a config read failure and return
    an empty table, which made every stop-sequence kill look like a natural stop and silently
    restored the bug the read was added to fix. So the failure is bounded to a *skipped* check
    on a *single* comparison rather than a green light, and a task genuinely absent from the
    config is caught by the config audit instead.
    """
    global _TASK_CAPS
    if _TASK_CAPS is None:
        try:
            import yaml  # noqa: PLC0415 - only needed on this path

            configs = Path(__file__).resolve().parents[2] / "configs" / "tasks.yaml"
            tasks = yaml.safe_load(configs.read_text())["tasks"]
            _TASK_CAPS = {
                name: dict(cfg["max_gen_toks"])
                for name, cfg in tasks.items()
                if isinstance(cfg, dict) and isinstance(cfg.get("max_gen_toks"), dict)
            }
        except (OSError, KeyError, TypeError, ValueError):
            _TASK_CAPS = {}
    return (_TASK_CAPS.get(task) or {}).get(budget_level)


#: Cache for the above. Populated once; `{}` means the config was unreadable.
_TASK_CAPS: dict[str, dict[str, int]] | None = None


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
    # 🚨 WHICH ENGINE LOAD PRODUCED THIS RECORD (planner ruling R21.3).
    #
    # Minted once per `VLLMRunner`, i.e. once per engine load, and stamped on every record
    # that runner emits. It exists because `pilot-v1` and `pilot-v2` ran the same items in
    # the same order with a byte-identical recorded configuration and **diverged on 41-45%
    # of items for Llama, including bf16** (`notes/cross-invocation.md`). The cause is
    # undiagnosed; four hypotheses are refuted.
    #
    # ⚠️ Deliberately NOT part of `JOIN_KEY`. Two invocations of the same item are two
    # measurements of one cell, distinguished by `repeat_idx` (the K=2 design of prereg
    # §9f), and adding the invocation to the identity would make every re-run trivially
    # "unique" and defeat the uniqueness constraint that catches double-counting.
    #
    # Why it is not optional: per-arm invocation is unavoidable (each arm is different
    # weights, so each needs its own engine load), which makes this variance inherently
    # BETWEEN-ARM. §9f's estimator is fitted per arm from the K=2 cells, and without this
    # field a resumed run's records cannot be attributed to the process that produced them
    # -- the noise could not be estimated at all, only assumed.
    invocation_id: str = ""
    # 🚨 THE STOP SEQUENCES THAT PRODUCED THIS GENERATION.
    #
    # Recorded because a scorer that has to INFER them will get it wrong. It already did:
    # `compliance.HARNESS_UNTIL` hardcoded BBH's pre-correction terminators, so every BBH
    # generation cut by `"Q:"` was classified COMPLETE and `trunc_stopseq_rate` read 0.0% --
    # 1,106 records misclassified, in §4's pre-registered TRUNC_STOPSEQ test.
    #
    # ⚠️ And a task-keyed table cannot be right in principle. `tasks.yaml` gives BBH TWO
    # terminator sets via `until_by_template`, one per template level, because the set is part
    # of each level's definition (prereg §3b-bis) -- and `template-v1` runs both levels in one
    # run. Only the record itself knows which one applied to it.
    #
    # This is also what makes the released artifact SELF-CLASSIFYING (A24): a third party can
    # recompute the termination classes from the JSONL alone, without reconstructing which
    # config produced which cell. `records.py`'s own docstring already claims that property;
    # this field is what actually delivers it.
    until: list[str] | None = None
    # Which level of the template FACTOR produced this (prereg §3b-bis): "chat" or "raw".
    # `sm_template` encodes it in `cell_id`; `sm_pilot` and `sm_reference` do not, so for
    # `main-v1` a third party would otherwise have to KNOW it is chat-only. With `until`
    # recorded they could infer it -- BBH's two sets differ -- but inference is exactly what
    # recording `until` removed from the termination path, and the same argument applies here.
    template_level: str | None = None
    #: WHERE the `until` strings were applied: `"generation"` or `"scoring"`.
    #
    # 🚨 This field is what keeps records from before and after planner ruling R34 from being
    # silently mixed, and that is not a hypothetical: `pilot-v1`, `pilot-v2`, `template-v1` and
    # `template-v2` all carry `"generation"`, and every record after R34 carries `"scoring"`.
    #
    # `"generation"` — `until` was passed to vLLM as `stop=`, so the engine halted there and the
    #   continuation is GONE. `stop_reason` names the string that fired, and `n_gen_tokens`
    #   counts only the kept prefix.
    # `"scoring"`   — nothing was passed as `stop=`. The generation ran to EOS or the budget, so
    #   `generation_text` holds the FULL output, `finish_reason`/`n_gen_tokens` describe all of
    #   it, and `stop_reason` is None or an EOS token but never an `until` string. Scoring calls
    #   `scoring.extract.apply_until` to recover the identical prefix.
    #
    # ⚠️ So `TRUNC_STOPSEQ` is read off `stop_reason` under `"generation"` and DERIVED FROM THE
    # TEXT under `"scoring"`. `compliance.classify_termination` branches on this field, which is
    # why it has to be recorded rather than inferred from `schema_version`: a reader should not
    # have to know which release changed the generation path.
    #
    # None means a record predating the field, which is treated as `"generation"` because that
    # is what every such record actually did.
    until_applied: str | None = None
    #: 🚨 THE SCORED PREFIX, recorded separately from the full generation (planner R35.Q4).
    #
    # R34 moved `until` to scoring time, which made `finish_reason` and `n_gen_tokens` describe
    # the FULL generation. That silently redefined §4's two truncation classes, and the planner
    # caught it before the freeze:
    #
    # * `TRUNC_BUDGET` began OVER-COUNTING: a generation that ran past its stop string to the
    #   cap reads as budget-truncated even when its scored prefix ended complete and early.
    # * H1's estimand became unanswerable. H1 asks whether the **scored answer** was cut off by
    #   the budget, and a full-generation `finish_reason` cannot say.
    #
    # So the classes are defined on the SCORED PREFIX, which needs these recorded at generation
    # time: the scorer has no tokenizer, so `n_gen_tokens_scored` is unrecoverable afterwards.
    # Exactly the rule that already governs `finish_reason` (README §1, task A4).
    #
    # `stop_string_hit`      which `until` string terminated the prefix, or None if none did.
    # `n_gen_tokens_scored`  tokens up to that string; equals the full count when None.
    #
    # ⚠️ `finish_reason` and `n_gen_tokens` retain their existing names and now describe the
    # FULL generation under `until_applied == "scoring"`. They are NOT renamed, because
    # `validate()` requires them and README §1 and A4 name them as the two unreconstructable
    # fields; renaming would break every existing reader for a cosmetic gain. `until_applied`
    # is what tells a reader which they are.
    stop_string_hit: str | None = None
    n_gen_tokens_scored: int | None = None
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

        # 🚨 THE BUDGET LABEL MUST AGREE WITH THE CAP IT WAS GENERATED UNDER.
        #
        # This is H1's whole instrument. `budget_level` is the independent variable of the
        # registered token-budget sweep (prereg §3, A15), and `max_gen_toks` is the cap that
        # actually bound the generation. If they disagree, the artifact is MISLABELLED: a
        # generation capped at 256 tokens recorded as `4x`. It scores cleanly, it looks
        # plausible in aggregate, and it is unrecoverable afterwards, because nothing else in
        # the record says what cap was in force.
        #
        # Checked HERE rather than only at the call site because `validate()` is the fail-closed
        # point every stage passes through, and the mislabel can arrive from any of them. The
        # CODE REVIEWER's point was that the call-site guard was a comment instructing a future
        # editor, i.e. incidental rather than structural; this is the structural half.
        #
        # ⚠️ Deliberately compares against `configs/tasks.yaml` rather than a literal, because a
        # hardcoded table is exactly how `compliance.HARNESS_UNTIL` drifted from the config and
        # reported a stale terminator set as a measurement.
        if self.scoring_mode == "generate" and self.budget_level and self.max_gen_toks:
            expected = _expected_max_gen_toks(self.task, self.budget_level)
            if expected is not None and expected != self.max_gen_toks:
                raise SchemaError(
                    f"budget_level={self.budget_level!r} disagrees with "
                    f"max_gen_toks={self.max_gen_toks} for task={self.task!r}: "
                    f"configs/tasks.yaml declares {expected} at that level "
                    f"({self.cell_id}/{self.item_id}). A mislabelled budget is "
                    f"unrecoverable after the fact, so this fails closed."
                )

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


class TruncatedCellError(RuntimeError):
    """The gzip stream ends before its end-of-stream marker: a killed write.

    Raised by ``read_records`` unless the caller explicitly opts into recovery. The
    distinction is the point -- see that function.
    """


def read_records(
    path: Path, *, tolerate_truncation: bool = False
) -> Iterator[dict[str, Any]]:
    """Stream records back.

    Two DIFFERENT kinds of truncation, and they need opposite treatment:

    * **A half-written final LINE** -- valid gzip, invalid JSON on the last line. Always
      tolerated: it is the ordinary signature of a killed process and the resume path simply
      regenerates that one item.
    * **A truncated gzip MEMBER** -- the stream ends before its end-of-stream marker, so
      `gzip` raises ``EOFError`` partway through. This is the failure that actually happens
      when a container is killed mid-flush, and it was **unhandled**: verified by truncating
      a real 139 KB cell file to half its length, both this function and
      ``scripts/score_pilot.read_records`` raised
      ``EOFError: Compressed file ended before the end-of-stream marker was read``.

    🚨 **The default is to REFUSE, and the asymmetry is deliberate.**

    A truncated member means an unknown number of records are missing from the tail. For the
    **resume** path that is recoverable and even routine -- take what decompressed and
    regenerate the rest -- so `run_cell` passes ``tolerate_truncation=True``. For the
    **scoring** path it is not: silently returning a short cell gives every rate in that cell
    a wrong denominator, and a wrong denominator is a plausible-looking wrong number, which is
    the one failure this project cannot afford. So scoring gets the exception.

    A single flag with opposite defaults per caller is the whole fix; making both callers
    tolerant would have been the bug, and making both strict would have made a killed job
    unresumable.
    """
    if not path.exists():
        return
    recovered = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    # 🚨 SAME EVENT CLASS as a truncated member, from the consumer's point of
                    # view: an unknown number of records is missing from the tail, so every
                    # rate in this cell has a wrong denominator. It used to `return` bare
                    # regardless of the flag -- guarding one of two routes to the same wrong
                    # number. Both routes now obey `tolerate_truncation`.
                    if not tolerate_truncation:
                        raise TruncatedCellError(
                            f"{path} has a TRUNCATED final line ({recovered} records "
                            f"recovered, unknown number lost).\n"
                            f"  Scoring a short cell silently would give every rate in it a "
                            f"wrong denominator. Resume the stage to regenerate the tail."
                        ) from exc
                    return
                recovered += 1
                yield record
    except EOFError as exc:
        if not tolerate_truncation:
            raise TruncatedCellError(
                f"{path} is a TRUNCATED gzip stream ({recovered} records recovered before "
                f"the break, unknown number lost).\n"
                f"  Scoring a short cell silently would give every rate in it a wrong "
                f"denominator.\n"
                f"  The generation path can recover this by resuming -- re-run the stage and "
                f"the missing tail regenerates. Do not score it until then."
            ) from exc
        # Resume path: the recovered prefix is exactly what we want to skip re-generating.
        return


def build_resume_index(path: Path) -> set[tuple]:
    """Keys already generated in this cell file, for idempotent re-runs.

    `EXECUTOR-TASKS.md` §7.3: re-running a command with identical arguments
    must produce zero new work and exit 0.
    """
    # Tolerant: this IS the recovery path. A truncated tail means those records are missing
    # from the index, so `run_cell` regenerates them -- which is correct.
    return {
        tuple(record.get(field) for field in JOIN_KEY)
        for record in read_records(path, tolerate_truncation=True)
    }


# The record fields that must match for a cached cell to be REUSABLE. Every one of these
# changes what a generation IS, and every one is already stored on the record -- so a stale
# cache is detectable from data we already have, with no path change and no new sidecar.
#
# ⚠️ `decoding_fingerprint()` was the declared root fix for this, and it was wired into
# `sm_determinism.py` ONLY -- while its own comment named `pilot`, `template` and `reference`
# as the vulnerable stages, "precisely the stages whose output feeds the paper's deltas".
# So the fix did not land where the comment said it did. Found by the code reviewer,
# 2026-08-31.
#
# The alternative fixes -- appending the fingerprint to `run_id` or to `cell_path` -- were
# REJECTED: both silently partition, so an unchanged config would keep resuming while a
# changed one would quietly start a parallel file set and regenerate everything. That
# regenerates `pilot-v2`, the annotation pool the planner ruled must NOT be re-run (R11/O11),
# and it turns a configuration mistake into an expensive silent success. Guards in this
# project refuse rather than warn, so this one refuses.
RESUME_CRITICAL_FIELDS: tuple[str, ...] = (
    "max_num_seqs",       # our own H8 result: changes generated text by up to 59%
    "disabled_kernels",   # the Marlin intervention; changes the kernel doing the matmul
    "seed",
    "batch_invariant",
    "engine",             # vllm==X.Y.Z
    "max_gen_toks",       # the budget level's resolved absolute cap
    "decoding",           # "greedy"
    "model_revision",     # commit SHA, never a tag
    "quant_config_hash",  # which checkpoint's weights
    # 🚨 The field with the PROVEN track record of causing exactly this failure. `pilot-v1`
    # ran BBH with the raw-level terminators, five of six subtasks came out inert (§9b), and
    # the only thing that stopped a resume from mixing two terminator sets inside one cell
    # file was a manual `run_id` bump. `CELL_SCRIPT` reads `until` from `tasks.yaml` at run
    # time, so editing it mid-run is a live path. This guard exists for this family; `until`
    # is the member it should most obviously cover.
    "until",
    # 🚨 R34, and it needs the ABSENT_DEFAULTS entry below to work at all. `until` alone is not
    # enough: the same terminator list means two different things depending on WHERE it was
    # applied, and a cell mixing the two holds text whose meaning changes halfway through the
    # file. Found by checking before a resubmission rather than by a failure: yesterday's
    # `main-v1-smoke2` cells predate R34, every other resume-critical field matched, and the
    # resume would have appended scoring-regime records into them and exited 0.
    "until_applied",
)

#: What a MISSING field means, for fields whose absence is itself information.
#:
#: 🚨 Without this the guard cannot see the R34 transition at all. `assert_cell_reusable` skips
#: any field absent from the stored record, which is right for a field that is merely newer and
#: catastrophically wrong for one whose absence carries meaning: every pre-R34 record lacks
#: `until_applied` precisely BECAUSE it used generation-time truncation. Treating that as "no
#: information" is how the two regimes would have been mixed silently.
ABSENT_DEFAULTS: dict[str, object] = {
    "until_applied": "generation",
}


class StaleCacheError(RuntimeError):
    """An existing cell was produced under a different generation configuration.

    Always fatal. Resuming across it would mix two configurations inside one cell file and
    report success, which is the failure mode this project has hit ten times.
    """


def assert_cell_reusable(path: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    """Refuse to resume a cell that was generated under a different configuration.

    Compares the FIRST stored record's decoding-critical fields against ``expected``.
    Returns the stored values (for logging), or None when the file does not exist yet.

    Only the first record is read: every record in one cell comes from one `run_cell` call
    with one spec, so the first is representative, and reading one line keeps this cheap
    enough to run before every cell.

    ⚠️ A field absent from the stored record is SKIPPED, not treated as a mismatch. Records
    written before a field existed (`invocation_id`, and any future addition) must stay
    resumable -- otherwise adding a field to the schema silently invalidates every cell ever
    written, which is the same class of expensive surprise this guard exists to prevent.
    """
    if not path.exists():
        return None
    # Only the FIRST record is needed, and one line always decompresses even in a truncated
    # member -- so this is unaffected by truncation. ⚠️ The flag is therefore INERT here: the
    # generator never reaches its `raise`. Passed explicitly anyway to state the intent, but do
    # not read it as protection that is being exercised.
    stored = next(iter(read_records(path, tolerate_truncation=True)), None)
    if stored is None:
        return None

    mismatches = []
    for field in RESUME_CRITICAL_FIELDS:
        # ⚠️ A field absent from BOTH sides is genuinely uncomparable and is skipped. A field
        # absent from ONE side is only skipped when `ABSENT_DEFAULTS` has nothing to say about
        # it: where absence carries meaning, substitute that meaning and COMPARE. See R34.
        if field in ABSENT_DEFAULTS:
            was = stored.get(field, ABSENT_DEFAULTS[field])
            now = expected.get(field, ABSENT_DEFAULTS[field])
        elif field not in stored or field not in expected:
            continue
        else:
            was, now = stored[field], expected[field]
        if was != now:
            mismatches.append((field, was, now))
    if mismatches:
        lines = "\n".join(
            f"    {field}: stored {was!r}  ->  now {now!r}" for field, was, now in mismatches
        )
        raise StaleCacheError(
            f"REFUSING TO RESUME {path}\n"
            f"  It was generated under a DIFFERENT configuration:\n{lines}\n\n"
            f"  Resuming would mix two configurations inside one cell file and exit 0.\n"
            f"  Either use a new run_id, or delete this cell deliberately -- but do not\n"
            f"  resume across a configuration change."
        )
    return stored


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
