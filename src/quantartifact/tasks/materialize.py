"""Materialize prompts using `lm-evaluation-harness` itself, then freeze them.

**Why this exists instead of a parallel re-implementation.**

`EXECUTOR-TASKS.md` O9 requires that we inherit the harness's prompts *exactly*,
enforced by a byte-for-byte parity test (gate G2). The obvious way to do that is
to write our own renderer and diff it against lm-eval's. This module takes the
stronger route: **lm-eval renders the prompts, we freeze the result as a
versioned artifact, and generation reads only from the frozen artifact.** Parity
then holds by construction rather than by test, and the exact prompt strings and
token ids become part of the released artifact so a reviewer can verify them
without running anything.

lm-eval remains a **validation instrument only** (O9) — it never generates. It is
importable here without torch, which is why this can run on a laptop.

**Prompts are MODEL-SPECIFIC.** Chat templates differ per model, and R1.2 fixes
the chat template ON, so the same task yields a different prompt per model. The
artifact is keyed accordingly.

**Prompts are frozen as TOKEN IDS, not just text.** Text alone leaves the
tokenization ambiguous, and that ambiguity is not theoretical — see
``assert_single_leading_bos``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

PROMPT_ARTIFACT_VERSION = "1.0.0"
HARNESS_VERSION = "0.4.12"

# lm-eval's own default `fewshot_random_seed`. Pinned rather than left implicit
# because it is LOAD-BEARING for one of our arms.
#
# ⚠️ Discovered by gate G2 on 2026-08-26: the `gsm8k` task has
# `fewshot_split: train` and NO `fewshot_config`, so lm-eval SAMPLES its 5
# few-shot exemplars at random. Two renders of the same item produced different
# prompts. `gsm8k_cot` and every BBH subtask hardcode their exemplars
# (`sampler: first_n`) and are unaffected.
#
# Why it matters beyond tidiness: `gsm8k` is the `answer_marker=hash` level, a
# confirmatory arm. Unpinned, its prompts are irreproducible; and if exemplars
# differed across precision arms the contrast would be confounded by the
# exemplars rather than the precision. Freezing prompts once (this module's whole
# design) already prevents the cross-arm confound; the seed makes the artifact
# itself regenerable.
FEWSHOT_SEED = 1234


class BOSError(RuntimeError):
    """A rendered prompt has the wrong number of leading BOS tokens."""


class PromptTransformError(RuntimeError):
    """A synthetic-arm prompt transform did not apply as specified."""


# ---------------------------------------------------------------------------
# Synthetic matched arm (planner ruling R9)
# ---------------------------------------------------------------------------

# `gsm8k` and `gsm8k_cot` differ in FOUR ways, not the one we want to manipulate:
#   (a) marker            "#### N"  vs  "The answer is N."   <- the intended contrast
#   (b) shots             5         vs  8
#   (c) exemplar content  SAMPLED from train  vs  8 hardcoded
#   (d) until             ["Question:",…]     vs  ["Q:",…]
#
# (b) and (c) run TOWARD our own hypothesis: 8 hand-written clean exemplars
# should teach format compliance better than 5 sampled ones carrying
# `<<48/2=24>>` calculator noise, predicting higher `gsm8k_cot` compliance for
# reasons unrelated to marker naturalness -- the same direction the marker
# prediction points. A caveat cannot fix a bias that runs our way.
#
# So the identified marker contrast comes from a SYNTHETIC arm: `gsm8k_cot`'s
# prompt verbatim with ONLY the exemplar terminator swapped. That inherits
# (b), (c) and (d) from `gsm8k_cot` and controls all three at once.
#
# This arm is NOT harness-verbatim and must be labelled as such everywhere.
_ANSWER_PHRASE_RE = re.compile(r"The answer is (-?[0-9][0-9.,]*)\.")


def to_hash_marker(prompt_text: str) -> tuple[str, int]:
    """Rewrite `gsm8k_cot` exemplar terminators to the `####` marker.

    ``"… = 6. The answer is 6."`` → ``"… = 6. #### 6"``

    Safe to apply to the whole prompt: only *exemplars* carry an answer, and the
    trailing question ends at ``"A:"`` with nothing after it, so there is no
    terminator to rewrite there.

    Returns ``(rewritten, n_substitutions)``. The caller asserts the count equals
    the shot count — a silent zero-substitution would produce an arm identical to
    `gsm8k_cot`, i.e. a null contrast that looks like a real one.
    """
    rewritten, n = _ANSWER_PHRASE_RE.subn(r"#### \1", prompt_text)
    return rewritten, n


# Registry of synthetic arms: task id -> (base lm-eval task, transform, label).
SYNTHETIC_ARMS: dict[str, dict[str, Any]] = {
    "gsm8k_cot_hashmarker": {
        "base_task": "gsm8k_cot",
        "transform": to_hash_marker,
        "answer_marker": "hash",
        "harness_verbatim": False,
        "strict_regex": r"#### (\-?[0-9\.\,]+)",
        "note": (
            "SYNTHETIC, NOT harness-verbatim. gsm8k_cot's prompt with only the "
            "exemplar terminator changed to ####. Controls shots, exemplar "
            "content and until simultaneously, so the marker contrast is identified."
        ),
    },
}


@dataclasses.dataclass
class MaterializedPrompt:
    """One frozen prompt. This is what generation consumes."""

    task: str
    subtask: str | None
    prompt_variant: str
    model_key: str
    model_repo: str
    model_revision: str
    item_id: str
    gold: str
    prompt_text: str
    prompt_sha256: str
    prompt_token_ids: list[int]
    n_prompt_tokens: int
    num_fewshot: int
    apply_chat_template: bool
    fewshot_as_multiturn: bool
    add_special_tokens: bool
    n_leading_bos: int
    fewshot_seed: int = FEWSHOT_SEED
    # True when the task hardcodes its exemplars in the YAML, so the seed cannot
    # affect the prompt.
    fewshot_hardcoded: bool = False
    # True only when exemplars are actually SAMPLED, i.e. not hardcoded AND
    # num_fewshot > 0. This is the flag that matters: a 0-shot task draws no
    # exemplars, so its seed is irrelevant even though nothing is hardcoded.
    # Of our nine cells, `gsm8k` is the ONLY seed-sensitive one.
    fewshot_seed_matters: bool = False
    # False for synthetic arms (R9). Recorded per prompt so the distinction
    # cannot be lost between the artifact and the paper.
    harness_verbatim: bool = True
    prompt_transform: str | None = None
    n_transform_substitutions: int | None = None
    harness_version: str = HARNESS_VERSION
    artifact_version: str = PROMPT_ARTIFACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# ARC arm 2 — log-likelihood continuations (planner ruling R11)
# ---------------------------------------------------------------------------

# The generative arm's prompt ends `…The best answer is` with NO trailing space,
# so the continuation it must emit begins with one. The log-likelihood arm is a
# valid counterfactual only if it scores exactly that.
ARC_CONTINUATIONS: tuple[str, ...] = (" A", " B", " C", " D")


def continuation_token_ids(tokenizer, prefix: str, continuation: str) -> list[int]:
    """Token ids for ``continuation`` **as it appears after** ``prefix``.

    ⚠️ Tokenized IN CONTEXT and differenced, rather than encoded standalone,
    because encoding the continuation on its own is wrong for some tokenizer
    families. Measured 2026-08-26:

    ========================= =================== ==================
    model                     ``encode(" A")``    tokens
    ========================= =================== ==================
    Qwen2.5-3B (byte BPE)     ``[362]``           ``['ĠA']``     ✅ 1
    Llama-3.1-8B (byte BPE)   ``[362]``           ``['ĠA']``     ✅ 1
    **Phi-3.5-mini (SP)**     ``[29871, 319]``    ``['▁','▁A']`` ❌ 2
    ========================= =================== ==================

    Phi-3.5 uses SentencePiece, where ``▁A`` *already encodes the leading space*,
    so the literal ``" A"`` emits a redundant standalone ``▁`` first. "Always
    prefix a space" is therefore the wrong rule — the same shape as the double-BOS
    trap, right for two models and wrong for a third.

    Differencing in context sidesteps the whole question: it yields precisely the
    ids the model would have to produce, whatever the tokenizer convention.
    """
    base = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full = tokenizer(prefix + continuation, add_special_tokens=False)["input_ids"]

    if full[: len(base)] != base:
        # The continuation re-tokenized the boundary (it merged with the prefix's
        # final token). Fall back to the longest common prefix so we still return
        # only the tokens attributable to the continuation, and let the caller see
        # the length.
        common = 0
        for left, right in zip(base, full, strict=False):
            if left != right:
                break
            common += 1
        return full[common:]

    return full[len(base) :]


def arc_continuation_ids(tokenizer, prefix: str) -> dict[str, list[int]]:
    """Per-option continuation ids, recorded in the manifest as R11 requires.

    A model whose option continuation is not a single token needs explicit length
    normalization, and R11 requires that discrepancy be **reported, not absorbed**
    — so the ids are returned for recording rather than silently collapsed.
    """
    return {
        continuation: continuation_token_ids(tokenizer, prefix, continuation)
        for continuation in ARC_CONTINUATIONS
    }


# ---------------------------------------------------------------------------
# The BOS check — a verified hazard, not a hypothetical one
# ---------------------------------------------------------------------------

def count_leading_bos(token_ids: list[int], bos_token_id: int | None) -> int:
    if bos_token_id is None:
        return 0
    count = 0
    for token in token_ids:
        if token != bos_token_id:
            break
        count += 1
    return count


def expected_leading_bos(tokenizer) -> int:
    """How many leading BOS tokens THIS tokenizer intends on a plain string.

    Derived rather than hardcoded, because the right answer is model-specific and
    all three of ours differ:

    * **Qwen2.5-3B** — no BOS token at all → 0.
    * **Phi-3.5-mini** — defines ``bos_token='<s>'`` but sets
      ``add_bos_token=False``, so it is *intended* to run with no leading BOS → 0.
      Demanding 1 here would be wrong, and would "fix" a non-problem by
      corrupting the prompt.
    * **Llama-3.1-8B** — adds ``<|begin_of_text|>`` → 1.

    The invariant that actually matters is therefore *"the templated prompt has
    exactly as many leading BOS as this tokenizer produces for a bare string"* —
    which still catches the double-BOS, without imposing one model's convention
    on another.
    """
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None:
        return 0
    baseline = tokenizer("x", add_special_tokens=True)["input_ids"]
    return count_leading_bos(baseline, bos_id)


def choose_add_special_tokens(tokenizer, rendered_text: str) -> bool:
    """Decide ``add_special_tokens`` so the prompt gets EXACTLY one BOS.

    ⚠️ **Verified on 2026-08-25, and it differs per model:**

    ======================== ===================== ==========================
    model                    template embeds BOS?  ids with add_special=True
    ======================== ===================== ==========================
    Qwen2.5-3B-Instruct      n/a (no BOS token)    ``[151644, …]``
    Phi-3.5-mini-instruct    no                    ``[32010, …]``
    **Llama-3.1-8B-Instruct**  **yes**             **``[128000, 128000, …]``**
    ======================== ===================== ==========================

    Llama's chat template already inserts ``<|begin_of_text|>``, so tokenizing
    with ``add_special_tokens=True`` yields a **double BOS**. Red Hat's
    ``add_bos_token=True`` advice is for *non-template* use; applied blanket
    alongside a chat template it silently shifts every Llama score and would
    confound every arm at once.

    So this cannot be a single global flag. We decide per (model, rendered text)
    and record the decision on every prompt record.
    """
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None:
        # No BOS concept for this tokenizer (e.g. Qwen2.5). Either value is
        # equivalent w.r.t. BOS; keep the harness default.
        return True

    want = expected_leading_bos(tokenizer)

    with_special = tokenizer(rendered_text, add_special_tokens=True)["input_ids"]
    if count_leading_bos(with_special, bos_id) == want:
        return True

    without_special = tokenizer(rendered_text, add_special_tokens=False)["input_ids"]
    if count_leading_bos(without_special, bos_id) == want:
        return False

    raise BOSError(
        f"cannot obtain the expected {want} leading BOS: "
        f"add_special_tokens=True gives {count_leading_bos(with_special, bos_id)}, "
        f"add_special_tokens=False gives {count_leading_bos(without_special, bos_id)}"
    )


def assert_single_leading_bos(token_ids: list[int], tokenizer, context: str) -> int:
    """Fail loudly unless the prompt's leading-BOS count is the intended one.

    Required by `analysis/preregistration.md` §3b and checked in gate G2. The
    expected count comes from the tokenizer itself (see ``expected_leading_bos``),
    so this catches a **double BOS** without forcing one model's convention onto
    another. A count of 2 is always wrong; a count of 0 is correct for Qwen2.5 and
    Phi-3.5 and wrong for Llama-3.1.
    """
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None:
        return 0
    n = count_leading_bos(token_ids, bos_id)
    want = expected_leading_bos(tokenizer)
    if n != want:
        raise BOSError(
            f"{context}: expected {want} leading BOS (id={bos_id}), found {n}. "
            f"first ids={token_ids[:5]}"
        )
    if n > 1:
        raise BOSError(f"{context}: {n} leading BOS tokens -- a double BOS is never correct")
    return n


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_task_prompts(
    lm_eval_task_name: str,
    tokenizer,
    *,
    task: str,
    subtask: str | None,
    prompt_variant: str,
    model_key: str,
    model_repo: str,
    model_revision: str,
    num_fewshot: int | None = None,
    apply_chat_template: bool = True,
    fewshot_as_multiturn: bool = False,
    limit: int | None = None,
    item_ids: list[str] | None = None,
    fewshot_seed: int = FEWSHOT_SEED,
    transform: Callable[[str], tuple[str, int]] | None = None,
    transform_name: str | None = None,
    expected_substitutions: int | None = None,
) -> Iterator[MaterializedPrompt]:
    """Render prompts for one (task, model) with lm-eval, yielding frozen records.

    ``apply_chat_template=True, fewshot_as_multiturn=False`` is R1.2's *chat
    template ON, single-turn* — the harness collapses the whole few-shot block
    into one user message, which is the configuration that makes
    instruction-tuned models revert to their own prose format and is the likely
    cause of arXiv:2601.14277's collapsed strict scores.
    """
    from lm_eval.tasks import TaskManager  # noqa: PLC0415 - harness venv only

    manager = TaskManager()
    task_dict = manager.load_task_or_group([lm_eval_task_name])
    harness_task = task_dict[lm_eval_task_name]

    if num_fewshot is None:
        num_fewshot = harness_task.config.num_fewshot or 0

    # PIN THE FEW-SHOT SAMPLING SEED before rendering anything. Required for
    # `gsm8k`, whose exemplars are sampled rather than hardcoded -- gate G2
    # caught two renders of the same item disagreeing. Seeded ONCE and then
    # rendered in index order, which is exactly how the harness behaves in a real
    # eval run, so parity is preserved.
    harness_task.set_fewshot_seed(fewshot_seed)

    # `fewshot_config` is a FewshotConfig dataclass in v0.4.12, not a dict.
    fewshot_config = getattr(harness_task.config, "fewshot_config", None)
    fewshot_hardcoded = bool(getattr(fewshot_config, "samples", None))

    chat_template_fn = None
    if apply_chat_template:
        chat_template_fn = _build_chat_template_fn(tokenizer)

    docs = list(harness_task.test_docs())
    selected = _select_docs(docs, harness_task, limit=limit, item_ids=item_ids)

    add_special: bool | None = None

    for index, doc in selected:
        context = harness_task.fewshot_context(
            doc,
            num_fewshot=num_fewshot,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            chat_template=chat_template_fn,
        )
        if isinstance(context, list):
            raise NotImplementedError(
                f"{lm_eval_task_name} yields multiple contexts per doc; not in scope"
            )

        n_subs: int | None = None
        if transform is not None:
            context, n_subs = transform(context)
            # A silent zero-substitution would yield an arm byte-identical to the
            # base task -- a null contrast that looks like a real one. Assert.
            if expected_substitutions is not None and n_subs != expected_substitutions:
                raise PromptTransformError(
                    f"{transform_name} on {lm_eval_task_name} item {index}: "
                    f"expected {expected_substitutions} substitutions, made {n_subs}"
                )
            if n_subs == 0:
                raise PromptTransformError(
                    f"{transform_name} on {lm_eval_task_name} item {index}: "
                    f"transform made NO substitutions -- the arm would be identical "
                    f"to its base task"
                )

        # Decided once per (model, task) from the first prompt, then held fixed
        # so every prompt in a cell is tokenized identically.
        if add_special is None:
            add_special = choose_add_special_tokens(tokenizer, context)

        token_ids = tokenizer(context, add_special_tokens=add_special)["input_ids"]
        n_bos = assert_single_leading_bos(
            token_ids, tokenizer, f"{model_key}/{lm_eval_task_name}/{index}"
        )

        yield MaterializedPrompt(
            task=task,
            subtask=subtask,
            prompt_variant=prompt_variant,
            model_key=model_key,
            model_repo=model_repo,
            model_revision=model_revision,
            item_id=_doc_id(doc, index),
            gold=str(harness_task.doc_to_target(doc)),
            prompt_text=context,
            prompt_sha256=sha256_text(context),
            prompt_token_ids=token_ids,
            n_prompt_tokens=len(token_ids),
            num_fewshot=num_fewshot,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            add_special_tokens=bool(add_special),
            n_leading_bos=n_bos,
            fewshot_seed=fewshot_seed,
            fewshot_hardcoded=fewshot_hardcoded,
            fewshot_seed_matters=(not fewshot_hardcoded and num_fewshot > 0),
            harness_verbatim=(transform is None),
            prompt_transform=transform_name,
            n_transform_substitutions=n_subs,
        )


def _build_chat_template_fn(tokenizer):
    """Adapt a tokenizer to the callable lm-eval's ``chat_template`` expects.

    lm-eval wraps this with ``partial(chat_template, add_generation_prompt=…)``,
    so the callable must accept that keyword.
    """

    def chat_template(messages, add_generation_prompt: bool = True) -> str:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    return chat_template


def _doc_id(doc: dict, index: int) -> str:
    """Stable per-item id.

    Prefers a natural key so ids survive a dataset reorder; falls back to the
    index. Recorded in the artifact either way, so the join key is stable even if
    this logic changes later.
    """
    for field in ("id", "idx", "example_id", "question_id"):
        if field in doc and doc[field] is not None:
            return str(doc[field])
    return f"idx-{index}"


def _select_docs(docs, harness_task, *, limit: int | None, item_ids: list[str] | None):
    """Deterministic doc selection.

    ``item_ids`` takes precedence so ARC arms 1/2/3 and the nested GSM8K
    subsample can be pinned to exactly the same items.
    """
    indexed = list(enumerate(docs))
    if item_ids is not None:
        wanted = set(item_ids)
        indexed = [(i, d) for i, d in indexed if _doc_id(d, i) in wanted]
    if limit is not None:
        indexed = indexed[:limit]
    return indexed


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------

def prompt_artifact_path(
    root: Path, model_key: str, task: str, subtask: str | None, prompt_variant: str
) -> Path:
    parts = [task] + ([subtask] if subtask else []) + [prompt_variant]
    return root / model_key / f"{'__'.join(parts)}.jsonl"


def write_prompts(path: Path, prompts: list[MaterializedPrompt]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(json.dumps(prompt.to_dict(), ensure_ascii=False) + "\n")
    return len(prompts)


def read_prompts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def manifest_for(prompts: list[MaterializedPrompt]) -> dict[str, Any]:
    """Summary used by gate G2 to prove the artifact came from the pinned harness.

    The rolling hash covers every prompt in order, so any change to any prompt --
    or to their order -- changes it.
    """
    rolling = hashlib.sha256()
    for prompt in prompts:
        rolling.update(prompt.prompt_sha256.encode("ascii"))

    first = prompts[0] if prompts else None
    return {
        "artifact_version": PROMPT_ARTIFACT_VERSION,
        "harness_version": HARNESS_VERSION,
        "n_prompts": len(prompts),
        "rolling_sha256": rolling.hexdigest(),
        "model_key": first.model_key if first else None,
        "model_revision": first.model_revision if first else None,
        "task": first.task if first else None,
        "subtask": first.subtask if first else None,
        "prompt_variant": first.prompt_variant if first else None,
        "num_fewshot": first.num_fewshot if first else None,
        "apply_chat_template": first.apply_chat_template if first else None,
        "fewshot_as_multiturn": first.fewshot_as_multiturn if first else None,
        "add_special_tokens": first.add_special_tokens if first else None,
        "n_leading_bos": first.n_leading_bos if first else None,
        "fewshot_seed": first.fewshot_seed if first else None,
        "fewshot_hardcoded": first.fewshot_hardcoded if first else None,
        "fewshot_seed_matters": first.fewshot_seed_matters if first else None,
        "harness_verbatim": first.harness_verbatim if first else None,
        "prompt_transform": first.prompt_transform if first else None,
    }
