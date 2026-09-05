#!/usr/bin/env python3
"""Census EVERY rung-monotonicity violation in a run and classify its mechanism.

    .venv/bin/python scripts/census_monotonicity.py --run-id main-v1

A monotonicity violation is an item where **rung 1 (strict) is CORRECT and rung 2
(permissive) is WRONG**. Prereg §5a registers those as *destruction*: evidence that
flexible extraction is not a monotone relaxation of strict extraction but a different
selection rule with two-sided error.

🚨 **WHY THIS SCRIPT EXISTS, AND WHY IT IS A CENSUS RATHER THAN A SAMPLE.** §5a's
taxonomy (classes A, B, C) was derived from **six** cases on `pilot-v2`, a corpus that did
not contain `gsm8k_cot` at all, because the pilot ran the `gsm8k` hash arm instead. On
`main-v1` there are **2,218** violations and **95.8% of them are on `gsm8k_cot`**, which is
the task §9c makes PRIMARY for H3. So the registered taxonomy was built without the task
that dominates the phenomenon.

Planner ruling R37.Q1 therefore required a **full census with a hand-audited random
subsample**, explicitly not a sample: sampling a population whose taxonomy you are
revising cannot show you the class you do not yet know about, and the *unclassified
residual* is the number that matters.

🚨 **WHAT THE CENSUS FOUND, AND IT REVERSES THE REGISTERED READING.** Two of the four
classes are **defects in our own rung-2 parser**, not properties of published practice:

| class | what | count | share | whose |
|---|---|---|---|---|
| `D1_unit_attached` | permissive returned the RIGHT answer with a unit or trailing text (`'5 days'` for gold `'5'`) and failed the match | 2,068 | 93.2% | **ours** |
| `D3_prose_captured` | `commitment_phrase` matched *"The answer is derived by following the order of operations…"* and captured the sentence | 36 | 1.6% | **ours** |
| `B_number` | the model committed several times; permissive took a later, wrong NUMBER | 35 | 1.6% | genuine |
| `B_token` | same, for a short non-numeric answer (`'True'` for gold `'False'`) | 79 | 3.6% | genuine |

⇒ **Genuine destruction is 114, not 2,218.** Against ~15,873 recovered that is 0.72%, and
`pilot-v2` measured 6 against 914, or 0.66%. **The pilot's RATE was correct; the apparent
20-fold inflation at reference n was entirely our own parser.**

⚠️ **The two defect classes cannot come from lm-eval's flexible-extract.** Its regex is
`(-?[$0-9.,]{2,})|(-?[0-9]+)`, which admits no letter, so it cannot return `'5 days'`.
Both come from `commitment_phrase`, which is **our addition**, ordered ahead of the harness
rule, and which therefore returns a strictly worse answer than published practice would on
this population. That is why repairing it moves rung 2 **toward** the harness rather than
away from it, and so does not run into R27.1's prohibition on changing rung 2's
last-candidate semantics.

⚠️ **Run this again after any change to `extract.py`.** The counts above are what a
registered number was restated on, so a silent drift here is a silent drift in §5a.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from quantartifact.analysis import study_wide_magnitude  # noqa: E402
from quantartifact.scoring import extract  # noqa: E402

#: A value that is only digits, separators and an optional sign. Anything else on a numeric
#: task means the extractor kept text it should have stripped.
_NUMERIC_ONLY = re.compile(r"^\s*-?[\d,.]+\s*$")

#: Above this many whitespace-separated words, a permissive value is prose rather than an
#: answer. Deliberately generous: `word_sorting` golds are genuine multi-word lists, so a
#: tight threshold would misclassify a real answer as captured prose. Every case in the
#: `D3` bucket at 5+ words was hand-checked to be a sentence.
_PROSE_WORDS = 4


def _load_analyze_run():
    """`scripts/analyze_run.py` is a script, not a module, so import it by path.

    Its `read_records` and `to_analysis_row` are the ones the registered analysis uses, and
    reimplementing either here would let this census disagree with the numbers it exists to
    explain.
    """
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("_ar", REPO / "scripts" / "analyze_run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ar"] = module
    spec.loader.exec_module(module)
    return module


def record_key(record: dict) -> tuple:
    """The identity `study_wide_magnitude` reports violations under."""
    cell = record.get("cell_id", "|")
    model = record.get("model_key") or cell.split("|")[0]
    arm = cell.split("|")[1] if "|" in cell else None
    item = f"{model}|{record['task']}|{record.get('subtask')}|{record['item_id']}"
    return (item, arm, record.get("repeat_idx", 0))


def classify(strict: str, permissive: str) -> str:
    """The mechanism behind one violation. See the module docstring for the taxonomy.

    ⚠️ Order matters. `D1` is tested before the numeric check because a unit-suffixed value
    is not numeric-only, and before the prose check because `'230 miles for each train'` is
    five words yet is a `D1` rather than a captured sentence: what distinguishes them is
    whether the value STARTS with what strict found.
    """
    if not permissive:
        return "E_permissive_empty"
    if strict and permissive.strip().startswith(strict.strip()) and not _NUMERIC_ONLY.match(
        permissive
    ):
        return "D1_unit_attached"
    if _NUMERIC_ONLY.match(permissive):
        return "B_number"
    if len(permissive.split()) > _PROSE_WORDS:
        return "D3_prose_captured"
    return "B_token"


#: Which classes are defects in our own parser rather than properties of published practice.
OURS = {"D1_unit_attached", "D3_prose_captured"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default="main-v1")
    parser.add_argument("--audit-n", type=int, default=100,
                        help="how many to write out for the hand audit R37.Q1 requires")
    parser.add_argument("--audit-seed", type=int, default=20260905,
                        help="published so the audit subsample is reproducible, not chosen")
    args = parser.parse_args()

    ar = _load_analyze_run()
    root = REPO / "artifacts" / "generations" / args.run_id
    if not root.is_dir():
        raise SystemExit(f"no generations at {root}")

    records = ar.read_records(root)
    rows = [ar.to_analysis_row(r) for r in records]
    _, n_violations, violations = study_wide_magnitude(rows)
    keys = {(v["item_key"], v["arm"], v["repeat_idx"]) for v in violations}
    print(f"run_id      : {args.run_id}")
    print(f"records     : {len(records):,}")
    print(f"violations  : {n_violations:,}  ({n_violations / len(records):.3%})")

    by_class: collections.Counter = collections.Counter()
    by_class_task: collections.Counter = collections.Counter()
    by_class_arm: collections.Counter = collections.Counter()
    detail: list[dict] = []

    for record in records:
        if record_key(record) not in keys:
            continue
        text = extract.scored_text_for(record)
        strict = extract.extract_strict(text, record["task"], record.get("subtask")).value or ""
        perm = extract.extract_permissive(text, record["task"], record.get("subtask")).value or ""
        klass = classify(strict, perm)
        by_class[klass] += 1
        by_class_task[(record["task"], record.get("subtask"), klass)] += 1
        by_class_arm[(record["cell_id"], klass)] += 1
        detail.append({
            "cell_id": record["cell_id"], "task": record["task"],
            "subtask": record.get("subtask"), "item_id": record["item_id"],
            "repeat_idx": record.get("repeat_idx", 0),
            "strict": strict, "permissive": perm, "gold": str(record.get("gold") or "")[:120],
            "class": klass,
        })

    total = sum(by_class.values())
    ours = sum(n for k, n in by_class.items() if k in OURS)
    genuine = total - ours

    print(f"\n{'class':30} {'n':>6} {'share':>8}  owner")
    print("-" * 60)
    for klass, n in by_class.most_common():
        print(f"  {klass:28} {n:>6} {n / total:>7.1%}  "
              f"{'OURS (parser defect)' if klass in OURS else 'genuine'}")
    print(f"\n  OUR PARSER DEFECTS : {ours:>6} {ours / total:>7.1%}")
    print(f"  GENUINE destruction: {genuine:>6} {genuine / total:>7.1%}")

    # 🚨 The audit subsample R37.Q1 requires, so classifier error is BOUNDED rather than
    # assumed. The seed is a published default, not chosen after seeing the result.
    import random  # noqa: PLC0415

    rng = random.Random(args.audit_seed)
    audit = rng.sample(detail, min(args.audit_n, len(detail)))

    out = REPO / "artifacts" / "analysis" / f"census_monotonicity_{args.run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "run_id": args.run_id,
        "n_records": len(records),
        "n_violations": total,
        "by_class": dict(by_class),
        "n_ours": ours,
        "n_genuine": genuine,
        "our_classes": sorted(OURS),
        "by_class_task": {f"{t}|{s}|{c}": n for (t, s, c), n in sorted(by_class_task.items())},
        "by_class_cell": {f"{c}|{k}": n for (c, k), n in sorted(by_class_arm.items())},
        "audit_seed": args.audit_seed,
        "audit_subsample": audit,
        "violations": detail,
    }, indent=2))
    print(f"\nreport -> {out.relative_to(REPO)}")
    print(f"  audit subsample of {len(audit)} written under `audit_subsample` "
          f"(seed {args.audit_seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
