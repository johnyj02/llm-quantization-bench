"""Synthetic-arm transform and continuation-tokenization tests.

Covers planner rulings R9 (the `gsm8k_cot_hashmarker` matched arm) and R11 (ARC
log-likelihood continuations). Fakes stand in for the real tokenizers so these
run with no network and no lm-eval.
"""

from __future__ import annotations

import pytest

from quantartifact.tasks.materialize import (
    ARC_CONTINUATIONS,
    SYNTHETIC_ARMS,
    arc_continuation_ids,
    continuation_token_ids,
    to_hash_marker,
)

# A faithful excerpt of a rendered gsm8k_cot prompt: exemplars ending
# "The answer is N.", then the unanswered final question ending "A:".
GSM8K_COT_PROMPT = (
    "Q: There are 15 trees. How many were planted?\n"
    "A: There were 21 after planting. So 21 - 15 = 6. The answer is 6.\n\n"
    "Q: Leah had 32 chocolates and her sister had 42. They ate 35. How many left?\n"
    "A: 32 + 42 = 74. After eating 35, 74 - 35 = 39. The answer is 39.\n\n"
    "Q: Olivia has $23 and buys five bagels for $3 each. How much is left?\n"
    "A:"
)


# ---------------------------------------------------------------------------
# R9 — the synthetic matched arm
# ---------------------------------------------------------------------------


def test_hash_marker_transform_rewrites_every_exemplar_terminator():
    rewritten, n = to_hash_marker(GSM8K_COT_PROMPT)
    assert n == 2, "one substitution per exemplar carrying an answer"
    assert "The answer is" not in rewritten
    assert rewritten.count("#### ") == 2
    assert "#### 6" in rewritten
    assert "#### 39" in rewritten


def test_hash_marker_transform_drops_the_sentence_period():
    """`#### N` is the real gsm8k terminator -- no trailing period."""
    rewritten, _ = to_hash_marker("A: reasoning. The answer is 6.\n")
    assert "#### 6\n" in rewritten
    assert "#### 6." not in rewritten


def test_hash_marker_transform_leaves_the_unanswered_question_alone():
    """The final question has no terminator to rewrite; it must survive intact."""
    rewritten, _ = to_hash_marker(GSM8K_COT_PROMPT)
    assert rewritten.endswith("Olivia has $23 and buys five bagels for $3 each. How much is left?\nA:")


def test_hash_marker_transform_preserves_reasoning_text():
    """ONLY the terminator changes -- shots, exemplars and scaffold are inherited.

    That is the whole point of the matched arm: it controls shot count, exemplar
    content and `until` simultaneously by taking all three from gsm8k_cot.
    """
    rewritten, _ = to_hash_marker(GSM8K_COT_PROMPT)
    for fragment in ("So 21 - 15 = 6", "32 + 42 = 74", "After eating 35, 74 - 35 = 39"):
        assert fragment in rewritten


def test_hash_marker_transform_does_not_touch_intermediate_numbers():
    """"= 6." mid-reasoning must not become a marker."""
    rewritten, n = to_hash_marker("A: we get 6. then 21 - 15 = 6. The answer is 6.\n")
    assert n == 1, "only the terminator phrase matches"
    assert rewritten.count("####") == 1


def test_hash_marker_transform_reports_zero_when_nothing_matches():
    """A zero count is what the caller asserts on -- it means a null contrast."""
    _, n = to_hash_marker("A: no terminator here at all\n")
    assert n == 0


def test_synthetic_arm_is_registered_as_not_harness_verbatim():
    spec = SYNTHETIC_ARMS["gsm8k_cot_hashmarker"]
    assert spec["harness_verbatim"] is False, "must never be presented as harness-verbatim"
    assert spec["base_task"] == "gsm8k_cot", "inherits shots/exemplars/until from gsm8k_cot"
    assert spec["answer_marker"] == "hash"
    assert "####" in spec["strict_regex"]


# ---------------------------------------------------------------------------
# R11 — continuation tokenization
# ---------------------------------------------------------------------------


class ByteBPETokenizer:
    """Qwen/Llama-style: a space-prefixed letter is its own single token."""

    VOCAB = {" A": 362, " B": 425, " C": 356, " D": 422, "A": 32, "B": 33}

    def __call__(self, text: str, add_special_tokens: bool = True):
        ids, index = [], 0
        while index < len(text):
            for length in (2, 1):
                chunk = text[index : index + length]
                if chunk in self.VOCAB:
                    ids.append(self.VOCAB[chunk])
                    index += length
                    break
            else:
                ids.append(900 + (ord(text[index]) % 50))
                index += 1
        return {"input_ids": ids}


class SentencePieceTokenizer:
    """Phi-style: ``▁A`` already encodes the leading space.

    Encoding ``" A"`` standalone therefore emits a redundant standalone ``▁``
    (29871) first -- the measured Phi-3.5 behaviour. In context the space is
    already consumed by the prefix, so differencing yields the single token.
    """

    SPACE = 29871
    LETTER = {"A": 319, "B": 350, "C": 315, "D": 360}

    def __call__(self, text: str, add_special_tokens: bool = True):
        ids: list[int] = []
        index = 0
        leading = True
        while index < len(text):
            char = text[index]
            if char == " ":
                nxt = text[index + 1] if index + 1 < len(text) else ""
                if nxt in self.LETTER:
                    if leading:
                        ids.append(self.SPACE)  # the redundant standalone marker
                    ids.append(self.LETTER[nxt])
                    index += 2
                    leading = False
                    continue
                index += 1
                continue
            ids.append(700 + (ord(char) % 50))
            index += 1
            leading = False
        return {"input_ids": ids}


def test_byte_bpe_continuation_is_single_token():
    tokenizer = ByteBPETokenizer()
    assert continuation_token_ids(tokenizer, "The best answer is", " A") == [362]


def test_sentencepiece_standalone_encoding_is_the_hazard():
    """Reproduce the measured Phi behaviour so the fix is demonstrably a fix."""
    tokenizer = SentencePieceTokenizer()
    standalone = tokenizer(" A", add_special_tokens=False)["input_ids"]
    assert len(standalone) == 2, "standalone ' A' splits into ['_', '_A'] on SentencePiece"


def test_sentencepiece_continuation_in_context_is_single_token():
    """THE FIX. Differencing in context drops the redundant space marker."""
    tokenizer = SentencePieceTokenizer()
    ids = continuation_token_ids(tokenizer, "The best answer is", " A")
    assert ids == [319], "in context, ' A' is the single token _A"


def test_all_arc_continuations_resolve_for_both_tokenizer_families():
    """R11 guard #1: assert single-token-ness per tokenizer, both families."""
    for tokenizer in (ByteBPETokenizer(), SentencePieceTokenizer()):
        ids = arc_continuation_ids(tokenizer, "The best answer is")
        assert set(ids) == set(ARC_CONTINUATIONS)
        assert all(len(v) == 1 for v in ids.values()), ids


def test_arc_continuations_are_distinct():
    """Four options must map to four different tokens, or the arm cannot rank them."""
    for tokenizer in (ByteBPETokenizer(), SentencePieceTokenizer()):
        ids = arc_continuation_ids(tokenizer, "The best answer is")
        flat = [v[0] for v in ids.values()]
        assert len(set(flat)) == 4, ids


def test_continuations_carry_the_leading_space():
    """The generative arm must emit a space, since gen_prefix has no trailing one."""
    assert ARC_CONTINUATIONS == (" A", " B", " C", " D")
    for continuation in ARC_CONTINUATIONS:
        assert continuation.startswith(" ")


def test_boundary_retokenization_falls_back_to_common_prefix():
    """If the continuation merges with the prefix's last token, still return only its ids."""

    class MergingTokenizer:
        def __call__(self, text: str, add_special_tokens: bool = True):
            if text.endswith(" A"):
                return {"input_ids": [1, 2, 99]}  # last prefix token changed
            return {"input_ids": [1, 2, 3]}

    ids = continuation_token_ids(MergingTokenizer(), "prefix", " A")
    assert ids == [99]


@pytest.mark.parametrize("continuation", [" A", " B", " C", " D"])
def test_each_continuation_is_nonempty(continuation):
    ids = continuation_token_ids(ByteBPETokenizer(), "The best answer is", continuation)
    assert ids, f"{continuation!r} produced no tokens"
