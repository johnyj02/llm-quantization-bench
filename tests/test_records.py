"""Record schema, JSONL round-trip, and resume-index tests.

The most important tests here are the ones asserting that a record **cannot be
written** without ``finish_reason`` and ``n_gen_tokens``. Those two fields are
unreconstructable after the fact and H1 -- the paper's lead hypothesis --
is impossible without them. A run that silently wrote None for 40,000
generations would not be recoverable at any price, so the schema refuses.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import tempfile
from pathlib import Path

import pytest

from quantartifact.records import (
    JOIN_KEY,
    RESUME_CRITICAL_FIELDS,
    GenerationRecord,
    SchemaError,
    StaleCacheError,
    TruncatedCellError,
    _expected_max_gen_toks,
    append_records,
    assert_cell_reusable,
    assert_join_key_unique,
    build_resume_index,
    cell_path,
    env_hash,
    prompt_sha256,
    read_records,
)


def make_record(**overrides) -> GenerationRecord:
    """A valid record, with `max_gen_toks` DERIVED from the task and budget level.

    ⚠️ It is derived rather than defaulted for the same reason production derives it. The old
    fixture hardcoded `max_gen_toks=256`, correct for its default `gsm8k_cot` at 1x and wrong for
    any test that overrode `task=`. `validate()` now refuses a `budget_level` that disagrees with
    the cap `configs/tasks.yaml` declares, and it caught this fixture immediately: a BBH record at
    1x carrying 256 where BBH declares 1024.

    A fixture that cannot express the inconsistency is worth more than one that is merely correct
    today, because the whole point of the new guard is that a mislabelled budget is unrecoverable.
    An explicit `max_gen_toks=` override still wins, so a test can still construct the violation
    on purpose.
    """
    task = overrides.get("task", "gsm8k_cot")
    level = overrides.get("budget_level", "1x")
    defaults = dict(
        run_id="main-v1",
        cell_id="qwen2.5-3b|gptq-4bit",
        model_repo="Qwen/Qwen2.5-3B-Instruct",
        model_revision="aa8e72537993ba99e69dfaafa59ed015b17504d1",
        quant_method="gptq",
        bits=4,
        quant_config_hash="deadbeef",
        task="gsm8k_cot",
        subtask=None,
        item_id="test-0",
        gold="18",
        prompt_variant="fewshot_cot",
        scoring_mode="generate",
        budget_level=level,
        max_gen_toks=_expected_max_gen_toks(task, level) or 256,
        decoding="greedy",
        sample_idx=0,
        seed=0,
        repeat_idx=0,
        prompt_text="Q: ...\nA:",
        prompt_sha256=prompt_sha256("Q: ...\nA:"),
        generation_text="The answer is 18.",
        n_prompt_tokens=812,
        n_gen_tokens=42,
        finish_reason="stop",
        stop_reason=None,
        option_logprobs=None,
        engine="vllm==0.26.0",
        batch_invariant=True,
        max_num_seqs=8,
        disabled_kernels=None,
        gpu="NVIDIA L40S",
        cuda="12.8",
        driver="560.35.03",
        env_hash="abc123",
        wall_ms=1830,
    )
    defaults.update(overrides)
    return GenerationRecord(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The two unreconstructable fields
# ---------------------------------------------------------------------------


def test_missing_finish_reason_is_refused():
    """H1 is impossible without it and it cannot be recovered from the text."""
    with pytest.raises(SchemaError, match="finish_reason is REQUIRED"):
        make_record(finish_reason=None).validate()


def test_missing_n_gen_tokens_is_refused():
    with pytest.raises(SchemaError, match="n_gen_tokens is REQUIRED"):
        make_record(n_gen_tokens=None).validate()


def test_bogus_finish_reason_is_refused():
    with pytest.raises(SchemaError, match="not in"):
        make_record(finish_reason="finished").validate()


@pytest.mark.parametrize("reason", ["stop", "length", "abort"])
def test_valid_finish_reasons_accepted(reason):
    make_record(finish_reason=reason).validate()


def test_append_refuses_to_write_an_invalid_record(tmp_path):
    """Validation happens BEFORE the write, so bad data never reaches disk."""
    path = tmp_path / "cell.jsonl.gz"
    with pytest.raises(SchemaError):
        append_records(path, [make_record(finish_reason=None)])
    assert not path.exists()


# ---------------------------------------------------------------------------
# Determinism contract
# ---------------------------------------------------------------------------


def test_batch_invariant_false_is_refused():
    """A non-BI record is not comparable to the rest of the study."""
    with pytest.raises(SchemaError, match="batch_invariant"):
        make_record(batch_invariant=False).validate()


def test_revision_must_be_a_sha_not_a_tag():
    with pytest.raises(SchemaError, match="commit SHA"):
        make_record(model_revision="main").validate()


def test_bbh_requires_a_subtask():
    """§7.1 marks subtask REQUIRED for bbh; without it records collide."""
    with pytest.raises(SchemaError, match="subtask is REQUIRED"):
        make_record(task="bbh", subtask=None).validate()
    make_record(task="bbh", subtask="word_sorting").validate()


# ---------------------------------------------------------------------------
# Log-likelihood mode
# ---------------------------------------------------------------------------


def test_loglikelihood_mode_requires_option_logprobs():
    with pytest.raises(SchemaError, match="option_logprobs is REQUIRED"):
        make_record(
            task="arc_challenge",
            scoring_mode="loglikelihood",
            finish_reason=None,
            n_gen_tokens=None,
            option_logprobs=None,
        ).validate()


def test_loglikelihood_mode_does_not_require_finish_reason():
    """Not a generative call, so the generative fields do not apply."""
    make_record(
        task="arc_challenge",
        scoring_mode="loglikelihood",
        finish_reason=None,
        n_gen_tokens=None,
        option_logprobs=[{"option": "A", "logprob": -1.2, "n_tokens": 3}],
    ).validate()


def test_unknown_scoring_mode_is_refused():
    with pytest.raises(SchemaError, match="unknown scoring_mode"):
        make_record(scoring_mode="vibes").validate()


# ---------------------------------------------------------------------------
# JSONL round-trip and resume
# ---------------------------------------------------------------------------


def test_round_trip(tmp_path):
    path = tmp_path / "cell.jsonl.gz"
    written = append_records(path, [make_record(item_id=f"item-{i}") for i in range(5)])
    assert written == 5

    back = list(read_records(path))
    assert len(back) == 5
    assert back[0]["generation_text"] == "The answer is 18."
    assert back[0]["finish_reason"] == "stop"


def test_append_is_additive_across_calls(tmp_path):
    """Concatenated gzip members must decode as one stream."""
    path = tmp_path / "cell.jsonl.gz"
    append_records(path, [make_record(item_id="a")])
    append_records(path, [make_record(item_id="b")])
    assert len(list(read_records(path))) == 2


def test_full_generation_text_is_never_truncated(tmp_path):
    """The stored text IS the artifact -- anyone must be able to re-score it."""
    long_text = "reasoning step. " * 5000
    path = tmp_path / "cell.jsonl.gz"
    append_records(path, [make_record(generation_text=long_text)])
    assert list(read_records(path))[0]["generation_text"] == long_text


def test_resume_index_matches_join_key(tmp_path):
    path = tmp_path / "cell.jsonl.gz"
    append_records(path, [make_record(item_id=f"item-{i}") for i in range(3)])

    index = build_resume_index(path)
    assert len(index) == 3
    expected = tuple(
        getattr(make_record(item_id="item-1"), field) for field in JOIN_KEY
    )
    assert expected in index


def test_resume_index_of_missing_file_is_empty(tmp_path):
    assert build_resume_index(tmp_path / "nope.jsonl.gz") == set()


def _cell_with_truncated_final_line(tmp_path):
    path = tmp_path / "cell.jsonl.gz"
    append_records(path, [make_record(item_id="good")])
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write('{"run_id": "main-v1", "cell_id": "trunc')  # no newline, no close
    return path


def test_truncated_final_line_is_tolerated_on_the_RESUME_path(tmp_path):
    """A half-written line is the signature of a killed process, not corruption.

    On the resume path it must be skipped so that one item regenerates, rather than the whole
    cell being treated as unreadable.
    """
    path = _cell_with_truncated_final_line(tmp_path)
    back = list(read_records(path, tolerate_truncation=True))
    assert len(back) == 1
    assert back[0]["item_id"] == "good"


def test_truncated_final_line_REFUSES_on_the_scoring_path(tmp_path):
    """🚨 The default changed, and deliberately.

    A short cell silently scored gives every rate in it a wrong denominator, which is a
    plausible-looking wrong number. Tolerance is now something a caller opts INTO -- the
    resume path does, scoring does not. Previously this route returned bare regardless, so
    the guard covered only one of two ways to reach the same wrong denominator.
    """
    path = _cell_with_truncated_final_line(tmp_path)
    with pytest.raises(TruncatedCellError, match="TRUNCATED final line"):
        list(read_records(path))


def test_the_resume_index_is_tolerant_by_construction(tmp_path):
    """`build_resume_index` must never raise on a killed write -- that IS the recovery path."""
    path = _cell_with_truncated_final_line(tmp_path)
    assert len(build_resume_index(path)) == 1


def test_repeat_idx_distinguishes_determinism_probe_repeats(tmp_path):
    """A7 reruns an identical probe 3x; the records must not collide."""
    path = tmp_path / "cell.jsonl.gz"
    append_records(path, [make_record(repeat_idx=i) for i in range(3)])
    assert len(build_resume_index(path)) == 3


# ---------------------------------------------------------------------------
# Join-key uniqueness
# ---------------------------------------------------------------------------


def test_duplicate_join_key_fails_loudly():
    """A duplicate silently double-weights an item in the GLMM and every rate."""
    records = [make_record(item_id="dup").to_dict(), make_record(item_id="dup").to_dict()]
    with pytest.raises(SchemaError, match="duplicate join key"):
        assert_join_key_unique(records)


def test_distinct_join_keys_pass():
    records = [make_record(item_id=f"item-{i}").to_dict() for i in range(10)]
    assert_join_key_unique(records)


def test_budget_level_is_part_of_the_join_key():
    """Same item at 1x and 4x are DIFFERENT observations -- H1 depends on it."""
    records = [
        make_record(item_id="x", budget_level="1x").to_dict(),
        make_record(item_id="x", budget_level="4x").to_dict(),
    ]
    assert_join_key_unique(records)


def test_join_key_covers_every_experimental_axis():
    """Guard against an axis being added without extending the key."""
    for axis in (
        "run_id", "cell_id", "task", "subtask", "item_id",
        "budget_level", "prompt_variant", "sample_idx", "repeat_idx",
    ):
        assert axis in JOIN_KEY
    assert {field.name for field in dataclasses.fields(GenerationRecord)} >= set(JOIN_KEY)


# ---------------------------------------------------------------------------
# Paths and hashing
# ---------------------------------------------------------------------------


def test_cell_path_separates_subtasks_and_budgets(tmp_path):
    """Finer than one-file-per-cell so a failed BBH subtask re-runs alone."""
    a = cell_path(tmp_path, "main-v1", "qwen|gptq-4bit", "bbh", "word_sorting", "1x", "fewshot_cot")
    b = cell_path(tmp_path, "main-v1", "qwen|gptq-4bit", "bbh", "dyck_languages", "1x", "fewshot_cot")
    c = cell_path(tmp_path, "main-v1", "qwen|gptq-4bit", "bbh", "word_sorting", "4x", "fewshot_cot")
    assert a != b != c
    assert "|" not in str(a), "pipe in cell_id must not leak into a filename"


def test_prompt_sha256_is_stable():
    assert prompt_sha256("abc") == prompt_sha256("abc")
    assert prompt_sha256("abc") != prompt_sha256("abd")


def test_env_hash_tracks_batch_invariance(monkeypatch):
    """The env hash must change when a determinism-relevant setting changes."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with_bi = env_hash()
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    without_bi = env_hash()
    assert with_bi != without_bi


class TestUntilAppliedIsResumeCritical:
    """🚨 R34. Caught by checking before a resubmission, not by a failure.

    `main-v1-smoke2` held cells generated before R34. Every other resume-critical field
    matched, so the resume would have appended scoring-regime records into a
    generation-regime file and exited 0 -- one cell whose text means different things in
    its two halves.
    """

    def _cell(self, tmp_path, **overrides):
        base = {
            "run_id": "r", "cell_id": "qwen|bf16", "task": "bbh", "subtask": "word_sorting",
            "item_id": "idx-0", "budget_level": "1x", "prompt_variant": "fewshot_cot",
            "sample_idx": 0, "repeat_idx": 0,
            "max_num_seqs": 32, "disabled_kernels": None, "seed": 0,
            "batch_invariant": True, "engine": "vllm==0.26.0", "max_gen_toks": 1024,
            "decoding": "greedy", "model_revision": "abc", "quant_config_hash": None,
            "until": ["</s>", "Q:"],
        }
        base.update(overrides)
        path = tmp_path / "cell.jsonl.gz"
        with gzip.open(path, "wt") as handle:
            handle.write(json.dumps(base) + "\n")
        return path

    EXPECTED = {
        "max_num_seqs": 32, "disabled_kernels": None, "seed": 0, "batch_invariant": True,
        "engine": "vllm==0.26.0", "max_gen_toks": 1024, "decoding": "greedy",
        "model_revision": "abc", "quant_config_hash": None, "until": ["</s>", "Q:"],
        "until_applied": "scoring",
    }

    def test_until_applied_is_in_the_critical_set(self):
        assert "until_applied" in RESUME_CRITICAL_FIELDS

    def test_a_pre_r34_cell_is_REFUSED(self):
        """The live case: the stored record has NO `until_applied` at all.

        This is the one a plain `field not in stored: continue` would wave through, and it is
        the only one that actually existed on disk.
        """
        path = self._cell(Path(tempfile.mkdtemp()))
        with pytest.raises(StaleCacheError) as excinfo:
            assert_cell_reusable(path, self.EXPECTED)
        assert "until_applied" in str(excinfo.value)
        # The message must name what it saw, not merely that something differed.
        assert "generation" in str(excinfo.value)
        assert "scoring" in str(excinfo.value)

    def test_an_explicit_generation_regime_cell_is_REFUSED(self):
        path = self._cell(Path(tempfile.mkdtemp()), until_applied="generation")
        with pytest.raises(StaleCacheError):
            assert_cell_reusable(path, self.EXPECTED)

    def test_a_matching_scoring_regime_cell_RESUMES(self):
        path = self._cell(Path(tempfile.mkdtemp()), until_applied="scoring")
        assert_cell_reusable(path, self.EXPECTED)   # must not raise

    def test_absence_on_BOTH_sides_still_resumes(self):
        """Absent from both is genuinely uncomparable, and pre-R34 tooling must keep working."""
        path = self._cell(Path(tempfile.mkdtemp()))
        expected = {k: v for k, v in self.EXPECTED.items() if k != "until_applied"}
        assert_cell_reusable(path, expected)   # both default to "generation"
