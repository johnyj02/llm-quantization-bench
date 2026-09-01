"""Tests for `scripts/score_template.py` — the probe that decides §3b-bis's branch.

🚨 There were NO tests for this file, and it could declare the favourable branch of a
pre-registered two-branch decision rule from an empty dataset, exit 0.

`if n_ok == len(verdicts)` is TRUE for an empty list (`0 == 0`), so any upstream condition
that emptied `rows` printed *"BRANCH 1: the interaction is template-invariant … ANSWERED
EMPIRICALLY rather than argued"* — the half of the rule that lets the paper keep `chat` as
primary — having measured nothing. The trigger was mundane: `cell.split("__")` expects
exactly `model__arm__level`, and any change to `cell_id`'s arity, or an added arm, skipped
every record via a bare `continue`.
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

_spec = importlib.util.spec_from_file_location(
    "score_template", REPO / "scripts" / "score_template.py")
score_template = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score_template)


def _cell(root: Path, name: str, n: int = 2) -> None:
    """One cell directory with `n` trivially-scoreable gsm8k records."""
    directory = root / name
    directory.mkdir(parents=True)
    with gzip.open(directory / "gsm8k__fewshot_cot__1x.jsonl.gz", "wt") as handle:
        for index in range(n):
            handle.write(json.dumps({
                "task": "gsm8k",
                "subtask": None,
                "gold": f"Some working here.\n#### {index}",
                "generation_text": f"The answer is {index}.",
            }) + "\n")


class TestItRefusesToReportFromNothing:
    """The vacuous-truth path: no data must never satisfy a pre-registered criterion."""

    def test_all_cells_unmatched_raises(self, tmp_path):
        """Arms/levels that match nothing used to empty `rows` and report BRANCH 1."""
        _cell(tmp_path, "qwen2.5-3b__not-an-arm__chat")
        _cell(tmp_path, "qwen2.5-3b__also-not__raw")
        with pytest.raises(SystemExit, match="matched NONE"):
            score_template.score_run(tmp_path)

    def test_the_refusal_names_the_favourable_branch_risk(self, tmp_path):
        """The message must say WHY an empty result is dangerous, not just that it is."""
        _cell(tmp_path, "qwen2.5-3b__not-an-arm__chat")
        with pytest.raises(SystemExit) as excinfo:
            score_template.score_run(tmp_path)
        assert "favourable" in str(excinfo.value)

    def test_an_unparseable_cell_name_raises_instead_of_skipping(self, tmp_path):
        """`model__arm__level` is a contract. A 2-part or 4-part name is a bug, not a filter."""
        _cell(tmp_path, "qwen2.5-3b__bf16")           # only two parts
        with pytest.raises(SystemExit, match="model__arm__level"):
            score_template.score_run(tmp_path)

    def test_an_empty_root_is_not_an_error(self, tmp_path):
        """Nothing found at all is a path problem, reported by the caller, not a branch."""
        assert score_template.score_run(tmp_path) == {}


class TestPartialSkipsAreVisible:
    """Skipping SOME cells is legitimate; it must not look like a complete run."""

    def test_recognised_cells_are_scored_and_others_reported(self, tmp_path, capsys):
        _cell(tmp_path, "qwen2.5-3b__bf16__chat")
        _cell(tmp_path, "qwen2.5-3b__bf16__raw")
        _cell(tmp_path, "qwen2.5-3b__unknown-arm__chat")
        rows = score_template.score_run(tmp_path)
        assert set(rows) == {("chat", "qwen2.5-3b", "bf16"), ("raw", "qwen2.5-3b", "bf16")}
        assert "skipped 1 cell" in capsys.readouterr().out


class TestTheVacuousTruthBackstop:
    """Even if `score_run` returned data, an empty verdict list must not decide a branch."""

    def test_empty_verdicts_would_satisfy_the_criterion(self):
        """Documents the arithmetic the guard exists for, so nobody 'simplifies' it away."""
        verdicts: list[bool] = []
        assert sum(verdicts) == len(verdicts), (
            "0 == 0 is why an empty run declared BRANCH 1; the guard in main() is the backstop"
        )

    def test_main_guards_against_it(self):
        source = (REPO / "scripts" / "score_template.py").read_text()
        assert "if not verdicts:" in source, "main() must refuse before comparing counts"
        assert source.index("if not verdicts:") < source.index("if n_ok == len(verdicts):"), (
            "the guard must come BEFORE the branch comparison"
        )
