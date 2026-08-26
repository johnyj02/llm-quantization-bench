"""Leading-BOS logic tests. Part of GATE G2.

Uses fake tokenizers modelled on the three real ones, so these run anywhere with
no network, no torch and no lm-eval. The real behaviour was measured on
2026-08-25 and is reproduced in the fakes:

======================== ============ ================== ===========================
model                    bos_token_id add_bos on "x"     template output starts with
======================== ============ ================== ===========================
Qwen2.5-3B-Instruct      None         n/a                ``<|im_start|>``
Phi-3.5-mini-instruct    1            NOT added          ``<|user|>``
**Llama-3.1-8B-Instruct**  128000     added              ``<|begin_of_text|>``
======================== ============ ================== ===========================

Llama is the dangerous one: its chat template already embeds BOS, so the naive
``add_special_tokens=True`` produces ``[128000, 128000, …]``. A silent double BOS
shifts scores and would confound **every arm at once**, which is why this is an
assertion rather than a comment.
"""

from __future__ import annotations

import pytest

from quantartifact.tasks.materialize import (
    BOSError,
    assert_single_leading_bos,
    choose_add_special_tokens,
    count_leading_bos,
    expected_leading_bos,
)


class FakeTokenizer:
    """Minimal stand-in: `bos_token_id` plus template-embedded-BOS behaviour."""

    def __init__(self, bos_token_id: int | None, adds_bos: bool, template_embeds_bos: bool):
        self.bos_token_id = bos_token_id
        self._adds_bos = adds_bos
        self._template_embeds_bos = template_embeds_bos

    def __call__(self, text: str, add_special_tokens: bool = True):
        ids: list[int] = []
        if self._template_embeds_bos and text.startswith("<BOS>"):
            ids.append(self.bos_token_id)  # type: ignore[arg-type]
        if add_special_tokens and self._adds_bos and self.bos_token_id is not None:
            ids.insert(0, self.bos_token_id)
        ids.extend([100, 101, 102])
        return {"input_ids": ids}


QWEN = FakeTokenizer(bos_token_id=None, adds_bos=False, template_embeds_bos=False)
PHI = FakeTokenizer(bos_token_id=1, adds_bos=False, template_embeds_bos=False)
LLAMA = FakeTokenizer(bos_token_id=128000, adds_bos=True, template_embeds_bos=True)


# ---------------------------------------------------------------------------
# count_leading_bos
# ---------------------------------------------------------------------------


def test_count_leading_bos_counts_only_the_prefix():
    assert count_leading_bos([5, 5, 9, 5], 5) == 2
    assert count_leading_bos([9, 5, 5], 5) == 0
    assert count_leading_bos([], 5) == 0


def test_count_leading_bos_is_zero_when_no_bos_token():
    assert count_leading_bos([1, 2, 3], None) == 0


# ---------------------------------------------------------------------------
# expected_leading_bos — derived per tokenizer, not hardcoded
# ---------------------------------------------------------------------------


def test_expected_bos_qwen_has_no_bos_token():
    assert expected_leading_bos(QWEN) == 0


def test_expected_bos_phi_defines_bos_but_does_not_add_it():
    """Phi sets add_bos_token=False -- 0 is CORRECT, not a defect.

    Demanding 1 here would "fix" a non-problem by corrupting the prompt.
    """
    assert expected_leading_bos(PHI) == 0


def test_expected_bos_llama_adds_one():
    assert expected_leading_bos(LLAMA) == 1


# ---------------------------------------------------------------------------
# choose_add_special_tokens — the actual double-BOS fix
# ---------------------------------------------------------------------------


def test_llama_template_forces_add_special_false():
    """THE fix. Template already embeds BOS, so adding another is a double BOS."""
    templated = "<BOS><|start_header_id|>system<|end_header_id|>"

    naive = LLAMA(templated, add_special_tokens=True)["input_ids"]
    assert count_leading_bos(naive, 128000) == 2, "the hazard must be present in the fake"

    assert choose_add_special_tokens(LLAMA, templated) is False

    fixed = LLAMA(templated, add_special_tokens=False)["input_ids"]
    assert count_leading_bos(fixed, 128000) == 1


def test_llama_untemplated_text_keeps_add_special_true():
    """Without the template there is no embedded BOS, so the flag must flip back."""
    assert choose_add_special_tokens(LLAMA, "Q: plain text") is True


def test_qwen_and_phi_keep_the_harness_default():
    for tokenizer in (QWEN, PHI):
        assert choose_add_special_tokens(tokenizer, "<|im_start|>system") is True


def test_unfixable_bos_raises_rather_than_guessing():
    """Neither flag yields the intended count => fail loudly, never guess.

    The baseline ("x") must give 1 so the expected count is 1, while the templated
    text gives 2 regardless of the flag. Then no choice is correct and the caller
    has to be told rather than silently handed a double-BOS prompt.
    """

    class StubbornlyDouble:
        bos_token_id = 7

        def __call__(self, text: str, add_special_tokens: bool = True):
            if text == "x":
                return {"input_ids": [7, 100]}          # baseline -> expected 1
            return {"input_ids": [7, 7, 100]}           # templated -> always 2

    tokenizer = StubbornlyDouble()
    assert expected_leading_bos(tokenizer) == 1

    with pytest.raises(BOSError, match="cannot obtain the expected 1 leading BOS"):
        choose_add_special_tokens(tokenizer, "<BOS>anything")


# ---------------------------------------------------------------------------
# assert_single_leading_bos
# ---------------------------------------------------------------------------


def test_double_bos_is_always_rejected():
    with pytest.raises(BOSError):
        assert_single_leading_bos([128000, 128000, 9], LLAMA, "llama/gsm8k/0")


def test_missing_bos_rejected_for_llama():
    with pytest.raises(BOSError, match="expected 1 leading BOS"):
        assert_single_leading_bos([128006, 9125], LLAMA, "llama/gsm8k/0")


def test_zero_bos_accepted_for_phi_and_qwen():
    assert assert_single_leading_bos([32010, 660], PHI, "phi/gsm8k/0") == 0
    assert assert_single_leading_bos([151644, 8948], QWEN, "qwen/gsm8k/0") == 0


def test_single_bos_accepted_for_llama():
    assert assert_single_leading_bos([128000, 128006, 9125], LLAMA, "llama/gsm8k/0") == 1
