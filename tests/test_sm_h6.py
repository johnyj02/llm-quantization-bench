"""Tests for sm_h6.py, H6's log-likelihood arm. No GPU, network or credentials.

The first H6 smoke died on four API mistakes inside an untyped CELL_SCRIPT string that no
linter or test could see. The logic is now a real function, and these tests pin the parts a
GPU run would otherwise be the first to exercise.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import sm_h6  # noqa: E402

TASKS = yaml.safe_load((REPO / "configs" / "tasks.yaml").read_text())["tasks"]


def test_gen_prefix_is_inherited_from_arm_1():
    """🚨 Scoring a letter straight after the assistant turn measures the wrong token.

    Arm 1's model writes "The best answer is C" itself, so the scored context must include
    that prefix or every letter looks unlikely and the arm gap is fabricated.
    """
    cfg = sm_h6.h6_config(TASKS)
    assert cfg["gen_prefix"] == TASKS["arc_challenge_chat"]["gen_prefix"]
    assert cfg["gen_prefix"].strip(), "an empty gen_prefix would score the wrong position"
    assert cfg["prompt_source"] == "arc_challenge_chat"


def test_h6_config_refuses_a_prompt_that_is_not_arm_1s():
    bad = {**TASKS, sm_h6.TASK: {**TASKS[sm_h6.TASK], "prompt_source": "arc_challenge"}}
    with pytest.raises(SystemExit, match="prompt_source"):
        sm_h6.h6_config(bad)


def _payload(cfg):
    return {"run_id": "h6-v1-test", "cfg": cfg, "cell": {
        "cell_id": "qwen2.5-3b|bf16", "base_repo": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "quant_method": None, "bits": 16, "quant_config_hash": None}}


def _fakes():
    prompt = SimpleNamespace(item_id="Mercury_1", gold="C", prompt_text="Q: ...",
                             prompt_sha256="0" * 64, n_prompt_tokens=42)
    runner = SimpleNamespace(invocation_id="inv-test", hardware={
        "gpu": "NVIDIA L40S", "cuda": "13.0", "driver": "595.91.07"})
    spec = SimpleNamespace(max_num_seqs=32, disabled_kernels=None)
    return prompt, runner, spec


def test_records_are_valid_and_argmax_is_the_prediction(monkeypatch):
    """The record must survive validate(), and the prediction is the highest logprob."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    prompt, runner, spec = _fakes()
    scores = [{" A": {"logprob": -3.0, "n_tokens": 1}, " B": {"logprob": -2.5, "n_tokens": 1},
               " C": {"logprob": -0.1, "n_tokens": 1}, " D": {"logprob": -4.0, "n_tokens": 1}}]
    recs = sm_h6.build_records(_payload(sm_h6.h6_config(TASKS)), [prompt], scores, runner, spec)
    assert len(recs) == 1
    r = recs[0]
    r.validate()
    assert r.scoring_mode == "loglikelihood"
    assert r.generation_text == "C"
    # SS7.1: a LIST of per-option logprob + length, in option order.
    assert [o["option"] for o in r.option_logprobs] == ["A", "B", "C", "D"]
    assert all(o["n_tokens"] == 1 for o in r.option_logprobs)


def test_a_multi_token_option_fails_loud(monkeypatch):
    """A multi-token option makes summed logprobs length-biased; SS3c forbids it."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    prompt, runner, spec = _fakes()
    scores = [{" A": {"logprob": -1.0, "n_tokens": 2}, " B": {"logprob": -2.0, "n_tokens": 1},
               " C": {"logprob": -3.0, "n_tokens": 1}, " D": {"logprob": -4.0, "n_tokens": 1}}]
    with pytest.raises(SystemExit, match="assert_single_token_per_option"):
        sm_h6.build_records(_payload(sm_h6.h6_config(TASKS)), [prompt], scores, runner, spec)
