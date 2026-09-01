"""Tests for `scripts/score_pilot.py` — the full-ladder scorer.

🚨 These are BEHAVIOURAL, not source-text assertions. The bug they exist for (K=2 pooling)
was invisible to every source-grep style test in this repo, because the code was perfectly
well-formed: `by_cell[row["cell_id"]]` reads correctly and does the wrong thing. Only calling
`aggregate()` with two invocations of one cell and looking at the output catches it.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# `scripts/` is not a package, so load the module by path.
_spec = importlib.util.spec_from_file_location(
    "score_pilot", REPO / "scripts" / "score_pilot.py")
score_pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score_pilot)

from quantartifact.records import (  # noqa: E402
    StaleCacheError,
    assert_cell_reusable,
)


def _row(cell_id: str, item_id: str, *, repeat_idx: int, correct: bool,
         invocation_id: str = "inv") -> dict:
    """One scored row, in the shape `aggregate()` consumes."""
    row = {
        "cell_id": cell_id,
        "repeat_idx": repeat_idx,
        "invocation_id": invocation_id,
        "task": "gsm8k",
        "subtask": None,
        "item_id": item_id,
        "n_gen_tokens": 100,
        "max_gen_toks": 256,
        "finish_reason": "stop",
        "compliance": "COMPLIANT",
        "termination": "COMPLETE",
    }
    for rung in score_pilot.RUNGS:
        row[f"correct_{rung}"] = correct
        row[f"found_{rung}"] = True
    return row


class TestKTwoIsNotPooled:
    """🚨 REGRESSION. `aggregate()` grouped by `cell_id` alone.

    Ten cells of the reference run are generated TWICE, in separate engine loads, to
    estimate the invocation-noise floor (prereg §9f). Pooling them gave those ten cells a
    two-invocation mean while the other five kept one-invocation means -- not comparable,
    and the table looks entirely normal.

    ⚠️ The join-key uniqueness check does NOT catch this, which is the instructive part:
    `repeat_idx` IS in `records.JOIN_KEY`, so the two invocations are legitimately distinct
    records and the duplicate check passes. A uniqueness constraint on the records says
    nothing about the grouping later applied to them.
    """

    CELL = "qwen2.5-3b|bf16"

    def _two_invocations(self) -> list[dict]:
        # k0: 2 of 4 correct (50%). k1: 4 of 4 correct (100%). Pooled would be 75%,
        # which is a value NEITHER invocation produced.
        rows = [_row(self.CELL, f"i{n}", repeat_idx=0, correct=n < 2,
                     invocation_id="inv-A") for n in range(4)]
        rows += [_row(self.CELL, f"i{n}", repeat_idx=1, correct=True,
                      invocation_id="inv-B") for n in range(4)]
        return rows

    def test_the_two_invocations_stay_separate(self):
        cells = score_pilot.aggregate(self._two_invocations())["cells"]
        assert len(cells) == 2, f"expected two entries, got {sorted(cells)}"
        assert self.CELL in cells, "repeat_idx 0 keeps the bare cell_id"
        assert f"{self.CELL}#k1" in cells, "repeat_idx 1 is suffixed"

    def test_neither_entry_is_the_pooled_mean(self):
        """The pooled value (75%) must appear nowhere. This is the actual bug."""
        cells = score_pilot.aggregate(self._two_invocations())["cells"]
        accuracies = {round(e["acc_strict"], 6) for e in cells.values()}
        assert accuracies == {0.5, 1.0}, accuracies
        assert 0.75 not in accuracies, (
            "0.75 is the two-invocation pooled mean -- aggregate() is averaging across "
            "invocations, which is exactly the K=2 defect"
        )

    def test_each_entry_counts_only_its_own_items(self):
        cells = score_pilot.aggregate(self._two_invocations())["cells"]
        assert all(e["n_items"] == 4 for e in cells.values()), (
            "n_items of 8 means both invocations landed in one group"
        )

    def test_repeat_idx_is_a_FIELD_not_only_a_key_suffix(self):
        """A value recoverable only by parsing a string is the shape of bug we keep finding."""
        cells = score_pilot.aggregate(self._two_invocations())["cells"]
        assert cells[self.CELL]["repeat_idx"] == 0
        assert cells[f"{self.CELL}#k1"]["repeat_idx"] == 1
        assert cells[f"{self.CELL}#k1"]["cell_id"] == self.CELL

    def test_invocation_ids_are_carried_for_provenance(self):
        """§9f's estimator needs to know the two means came from two different loads."""
        cells = score_pilot.aggregate(self._two_invocations())["cells"]
        assert cells[self.CELL]["invocation_ids"] == ["inv-A"]
        assert cells[f"{self.CELL}#k1"]["invocation_ids"] == ["inv-B"]

    def test_by_task_is_split_the_same_way(self):
        """A per-task breakdown pooled across invocations has the same defect."""
        by_task = score_pilot.aggregate(self._two_invocations())["by_task"]
        assert set(by_task) == {self.CELL, f"{self.CELL}#k1"}
        assert by_task[self.CELL]["gsm8k"]["n_items"] == 4


class TestSingleInvocationRunsAreUnchanged:
    """Every run before K=2 is repeat_idx 0 throughout; their keys must not move.

    The suffix is applied only for repeat_idx != 0 precisely so that saved scores from
    pilot-v2 and earlier stay readable and comparable.
    """

    def test_keys_are_bare_cell_ids(self):
        rows = [_row("qwen2.5-3b|bf16", f"i{n}", repeat_idx=0, correct=n == 0)
                for n in range(4)]
        cells = score_pilot.aggregate(rows)["cells"]
        assert list(cells) == ["qwen2.5-3b|bf16"]
        assert cells["qwen2.5-3b|bf16"]["acc_strict"] == 0.25

    def test_rows_missing_repeat_idx_default_to_zero(self):
        """Older records predate the field; they must not crash or become a second group."""
        rows = [_row("m|a", f"i{n}", repeat_idx=0, correct=True) for n in range(2)]
        for row in rows:
            del row["repeat_idx"]
        cells = score_pilot.aggregate(rows)["cells"]
        assert list(cells) == ["m|a"]
        assert cells["m|a"]["repeat_idx"] == 0


class TestAwqReplicateAssumesOneInvocation:
    """R12.1's contrast compares two CHECKPOINTS, not two invocations."""

    def test_it_refuses_if_an_arm_became_a_k2_cell(self):
        cells = {
            "qwen2.5-3b|awq-4bit-offload": {"repeat_idx": 1, "n_items": 4,
                                            **{f"acc_{r}": 0.5 for r in score_pilot.RUNGS}},
            "qwen2.5-3b|awq-4bit-nooffload": {"repeat_idx": 0, "n_items": 4,
                                              **{f"acc_{r}": 0.5 for r in score_pilot.RUNGS}},
        }
        with pytest.raises(SystemExit, match="repeat_idx"):
            score_pilot.awq_replicate(cells)

    def test_it_returns_none_when_the_replicate_cells_are_absent(self):
        assert score_pilot.awq_replicate({"m|a": {"repeat_idx": 0}}) is None


class TestResumeRefusesAcrossAConfigChange:
    """🚨 `records.assert_cell_reusable` — the guard the generation stages lacked.

    `decoding_fingerprint()` was the declared root fix and was wired into
    `sm_determinism.py` only, while its own comment named pilot/template/reference as the
    vulnerable stages. Those three were unguarded. Found by the code reviewer 2026-08-31.

    Checked against the STORED RECORDS rather than a sidecar or a path suffix, so it works
    retroactively on existing data and does not silently regenerate `pilot-v2` -- the
    annotation pool the planner ruled must not be re-run (R11/O11).
    """

    def _write(self, tmp_path, **fields):
        path = tmp_path / "cell.jsonl.gz"
        record = {"max_num_seqs": 64, "seed": 0, "decoding": "greedy",
                  "engine": "vllm==0.26.0", "disabled_kernels": None, **fields}
        with gzip.open(path, "wt") as handle:
            handle.write(json.dumps(record) + "\n")
        return path

    def test_identical_config_is_reusable(self, tmp_path):
        path = self._write(tmp_path)
        stored = assert_cell_reusable(path, {"max_num_seqs": 64, "seed": 0})
        assert stored["max_num_seqs"] == 64

    def test_a_changed_batch_size_REFUSES(self, tmp_path):
        """The live risk: main-v1 at 32, the freeze moves, someone resubmits."""
        path = self._write(tmp_path)
        with pytest.raises(StaleCacheError, match="max_num_seqs"):
            assert_cell_reusable(path, {"max_num_seqs": 32})

    def test_a_changed_kernel_set_REFUSES(self, tmp_path):
        path = self._write(tmp_path)
        with pytest.raises(StaleCacheError, match="disabled_kernels"):
            assert_cell_reusable(path, {"disabled_kernels": "MarlinLinearKernel"})

    def test_the_message_names_the_field_and_both_values(self, tmp_path):
        """A guard that refuses without saying why costs a debugging cycle."""
        path = self._write(tmp_path)
        try:
            assert_cell_reusable(path, {"max_num_seqs": 32})
        except StaleCacheError as exc:
            text = str(exc)
        assert "max_num_seqs" in text and "64" in text and "32" in text
        assert "REFUSING TO RESUME" in text

    def test_a_field_absent_from_old_records_does_NOT_refuse(self, tmp_path):
        """Adding a schema field must not invalidate every cell ever written.

        `invocation_id` was added on 2026-08-31; records written before it must stay
        resumable, or a schema addition becomes a silent full regeneration.
        """
        path = self._write(tmp_path)
        assert assert_cell_reusable(path, {"invocation_id": "brand-new"}) is not None

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        """Nothing to compare against is the normal first-run case."""
        assert assert_cell_reusable(tmp_path / "absent.jsonl.gz", {"seed": 0}) is None

    def test_run_cell_actually_calls_it(self):
        """A guard that exists but is never called is the bug it guards against."""
        source = (REPO / "src" / "quantartifact" / "generate.py").read_text()
        assert "assert_cell_reusable(path, {" in source, (
            "run_cell must call the guard before build_resume_index"
        )
        assert source.index("assert_cell_reusable(path") < source.index("already = build_resume_index"), (
            "the guard must run BEFORE the resume index is built"
        )
