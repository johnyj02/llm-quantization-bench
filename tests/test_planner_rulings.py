"""The rulings ledger must not drift from the raw answers on disk.

🚨 **This exists because the drift happened TWICE.** `notes/planner-rulings.md` is named by
`README.md` §0 as the authoritative record of planner rulings, and twice it has been missing
rulings whose raw answers were sitting in `notes/planner-answers/`:

* **R21, R22, R23** — backfilled 2026-08-31. R23 *reversed* part of R22, so a reader of the
  ledger alone would have acted on R22's classification of H3's interaction as carrying zero
  invocation noise, which is exactly the error R23 exists to correct.
* **R34, R36, R37** — backfilled 2026-09-05. R36 moved a registered floor **up from zero**,
  so a reader of the ledger alone would have used a floor of 0 for every class-3 claim.

Both times nothing was lost, because the raw answers were on disk and other documents cited
the rulings by number. That is precisely what made it easy to miss: the project kept working
while its authoritative index was incomplete.

⚠️ **The second occurrence of a defect is the point at which noticing it stops being
sufficient.** Every other instance of this project's recurring bug family was closed by a
test rather than by intending to remember, and this is the same shape one level up: a value
recorded in two places, where one is derived from the other by hand.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ANSWERS = REPO / "notes" / "planner-answers"
LEDGER = REPO / "notes" / "planner-rulings.md"

#: `R25-degenerate-interval.txt` -> 25. Only the leading R-number is significant; the rest of
#: the filename is a human-readable slug and is deliberately not constrained.
_ANSWER_NUMBER = re.compile(r"^R(\d+)[-.]")


#: Below this many bytes an answer file is a question IN FLIGHT, not a ruling. `ask-planner`
#: redirects into the file, so it exists and is empty from the moment the call starts, and a
#: planner call at `--effort max` runs for well over ten minutes. Without this floor the suite
#: goes red for the entire window between asking a question and recording its answer, which
#: would train exactly the wrong reflex: the fix would look like "do not ask", or worse,
#: "record the ruling before reading it".
_MIN_ANSWER_BYTES = 400


def ruling_numbers_with_raw_answers() -> set[int]:
    return {
        int(match.group(1))
        for path in ANSWERS.glob("R*.txt")
        if (match := _ANSWER_NUMBER.match(path.name))
        and path.stat().st_size >= _MIN_ANSWER_BYTES
    }


def ruling_numbers_in_ledger() -> set[int]:
    return {
        int(n)
        for n in re.findall(r"^## R(\d+)\b", LEDGER.read_text(), re.M)
    }


def test_every_raw_answer_has_a_ledger_entry():
    """The direction that has actually failed, twice.

    A raw answer with no ledger entry means a ruling exists that the authoritative record
    does not mention, so a reader of the record alone can act on a superseded position.
    """
    missing = sorted(ruling_numbers_with_raw_answers() - ruling_numbers_in_ledger())
    assert not missing, (
        f"rulings with a raw answer but NO entry in notes/planner-rulings.md: "
        f"{[f'R{n}' for n in missing]}. README §0 names that file as the authoritative "
        f"record, so an absent entry means a reader of it alone can act on a superseded "
        f"position. This has happened twice: R21-R23 and R34/R36/R37."
    )


def test_ledger_entries_are_not_invented():
    """The opposite direction, which has not failed but would be worse if it did.

    A ledger entry with no raw answer would mean a ruling recorded without the planner's
    text behind it, which is unfalsifiable. Rulings raised and answered inline in a prior
    session predate the `notes/planner-answers/` convention, so the check starts at the
    first ruling that convention covers rather than at R1.
    """
    first_with_convention = min(ruling_numbers_with_raw_answers())
    unbacked = sorted(
        n for n in ruling_numbers_in_ledger()
        if n >= first_with_convention and n not in ruling_numbers_with_raw_answers()
    )
    assert not unbacked, (
        f"ledger entries at or after R{first_with_convention} with no raw answer in "
        f"notes/planner-answers/: {[f'R{n}' for n in unbacked]}. A distilled ruling with "
        f"no source text cannot be checked against what the planner actually said."
    )


def test_the_two_known_drift_episodes_stay_closed():
    """A named regression test, because these are the six that were actually missing."""
    in_ledger = ruling_numbers_in_ledger()
    for number in (21, 22, 23, 34, 36, 37):
        assert number in in_ledger, (
            f"R{number} was missing from the ledger once already and has been backfilled; "
            f"its absence now is a regression, not a new omission"
        )
