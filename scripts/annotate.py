#!/usr/bin/env python3
"""The annotation tool. One item per screen, one keystroke, append-only. Task `A11`.

    .venv/bin/python scripts/annotate.py --who A2 --practice   # 10-item dry run FIRST
    .venv/bin/python scripts/annotate.py --who A2              # the real thing

`A11` says a roughly fifty-line local script is genuinely the right answer for 250 binary
labels, and names the alternatives it rejects: Label Studio Community cannot do agreement
matrices or per-task assignment (Enterprise only), and Prodigy costs $390 to $1,950, computes no
agreement statistics, and shows annotator names.

**Resumable, and that matters more than it sounds.** Labels append to
`annotation/labels_<who>.jsonl` as you go, so closing the terminal loses nothing and you can do
250 items over several sittings. Re-running skips what you have already labelled.

🚨 **This tool CANNOT show you the precision condition, the parser's verdict or the judge's
verdict, because they are not in the file it reads** (prereg §10). Blinding is by omission
rather than by hiding, since a hidden field is not hidden. It also never pre-fills a label:
Fort and Sagot (LAW IV 2010) measured the better annotator dropping 98.1 to 95.8 under
low-quality pre-annotation, and Mikulová et al. (LREC 2022) found pre-annotation *raised*
agreement, which is shared anchoring rather than validity and is worse for looking like success.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ANN = REPO / "annotation"

LABELS = {
    "c": "CORRECT",
    "i": "INCORRECT",
    "n": "NO_ANSWER",
    "g": "RUBRIC_GAP",
}
HELP = """
  c = CORRECT     the model committed to an answer and it matches
  i = INCORRECT   it committed to an answer and it does not match
  n = NO ANSWER   it never committed to a final answer at all
  g = RUBRIC GAP  the rubric genuinely does not decide this (you will be asked why)

  f = flag this item (e.g. the reference answer looks wrong) and then label it anyway
  ? = show the rubric's decision procedure
  s = skip for now (it stays in your queue)
  q = save and quit
"""

PROCEDURE = """
  RUBRIC, short form. Full text: annotation/rubric.md

  1. Find the answer the model PRESENTS AS FINAL. No commitment anywhere -> n
  2. Several commitments? Take the LAST one. (Deliberately unlike one of the scorers.)
  3. Compare VALUE, not formatting. $64 / 64.00 / 64 dollars all match 64.
  4. Edge cases:
     4.1 cut off AFTER committing -> judge that answer.  Cut off BEFORE -> n
     4.2 right number in the reasoning but a different final answer -> i
     4.3 "either 6 or 8" -> i  (a hedge is not a commitment)
     4.4 refusal -> n
     4.5 degenerate repetition -> n unless a value is clearly committed
     4.6 answered a different question -> i
     4.7 wrong units, right value -> c
     4.8 reference looks wrong -> judge against it anyway AND press f
     4.9 genuinely undecidable -> g with a note
"""


def load_queue(who: str, practice: bool) -> list[dict]:
    path = ANN / f"to_annotate_{who}.json"
    if not path.is_file():
        raise SystemExit(
            f"no queue at {path.relative_to(REPO)}. Run:\n"
            f"  .venv/bin/python scripts/draw_annotation_sample.py --run-id pilot-v2"
        )
    items = json.loads(path.read_text())
    # The practice round is the FIRST 10 in this annotator's own shuffled order, so the two
    # annotators practise on different items, which is what makes the dry run a calibration
    # of the rubric rather than a rehearsal of one shared sequence.
    return items[:10] if practice else items


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate model answers (A11)")
    parser.add_argument("--who", required=True, choices=["A1", "A2"],
                        help="A1 = Joseph Sankoorikal Johny, A2 = Syed Ahmed Danish")
    parser.add_argument("--practice", action="store_true",
                        help="the 10-item blind dry run required by B5 before real labelling")
    args = parser.parse_args()

    queue = load_queue(args.who, args.practice)
    suffix = "_practice" if args.practice else ""
    out = ANN / f"labels_{args.who}{suffix}.jsonl"

    done: set[str] = set()
    if out.is_file():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["annotation_id"])

    todo = [x for x in queue if x["annotation_id"] not in done]
    print(f"\n{'PRACTICE ROUND' if args.practice else 'ANNOTATION'} — annotator {args.who}")
    print(f"{len(done)} done, {len(todo)} remaining of {len(queue)}")
    if args.practice:
        print("\nThis is the B5 dry run. Label all 10 blind, then compare with the other")
        print("annotator and talk through EVERY disagreement before real labelling starts.")
    print(HELP)

    for n, item in enumerate(todo, 1):
        while True:
            print("\n" + "=" * 78)
            print(f"ITEM {n}/{len(todo)}   id {item['annotation_id']}")
            print("=" * 78)
            print("\n--- QUESTION " + "-" * 63)
            print(item["question"])
            print("\n--- THE MODEL'S ANSWER " + "-" * 54)
            print(item["model_answer"] or "(empty)")
            print("\n--- REFERENCE ANSWER " + "-" * 56)
            print(item["reference_answer"])
            print("-" * 78)

            choice = input("\n  [c]orrect  [i]ncorrect  [n]o answer  [g]ap  |  f ? s q  > ")
            choice = choice.strip().lower()

            if choice == "q":
                print(f"\nsaved {len(done)} labels to {out.relative_to(REPO)}")
                return 0
            if choice == "?":
                print(PROCEDURE)
                continue
            if choice == "s":
                break
            flag = ""
            if choice == "f":
                flag = input("  flag note: ").strip()
                choice = input("  now label it  [c/i/n/g] > ").strip().lower()
            if choice not in LABELS:
                print("  not a label. Press ? for the rubric.")
                continue
            note = ""
            if choice == "g":
                note = input("  what does the rubric not decide? ").strip()
                if not note:
                    print("  a RUBRIC_GAP needs a reason; it is excluded from the statistic.")
                    continue

            record = {
                "annotation_id": item["annotation_id"],
                "annotator": args.who,
                "label": LABELS[choice],
                "note": note,
                "flag": flag,
                "practice": args.practice,
                "labelled_at": datetime.now(UTC).isoformat(),
            }
            with out.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            done.add(item["annotation_id"])
            break

    print(f"\nDONE. {len(done)} labels in {out.relative_to(REPO)}")
    if args.practice:
        print("\nNEXT: both annotators compare practice labels and adjudicate BY RULE LOOKUP,")
        print("then sign the table at the end of annotation/rubric.md.")
    else:
        print("\nNEXT: .venv/bin/python scripts/score_annotation.py")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\ninterrupted; labels already entered are saved")
        sys.exit(130)
