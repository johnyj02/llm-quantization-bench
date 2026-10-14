#!/usr/bin/env python3
"""Score the human annotation: IAA, then validate the automated rungs against it. `A11`/`A12`.

    .venv/bin/python scripts/score_annotation.py

Two outputs, and the second is the one that reaches the paper:

1. **IAA on the double-annotated overlap.** `A12` requires raw agreement with a **Wilson 95%
   CI**, Cohen's kappa, **Gwet's AC1**, and the confusion matrix. Not kappa alone: these labels
   are heavily skewed, so a scorer at 95% raw agreement can post kappa ~0.3 purely from
   prevalence (the kappa paradox, Feinstein & Cicchetti 1990; Byrt, Bishop & Carlin 1993).

2. **Rung 1 and rung 2 validated against the human labels.** `A10`'s LLM judge does not exist,
   so rung 4 cannot be validated. That is survivable by design: §2.2 makes the headline the
   strict-versus-permissive contrast, which is exactly what these labels can validate.

🚨 **The check a good reviewer will ask for** (`A12`): scorer error rates reported **separately
for bf16 and 4-bit**. A scorer lenient toward quantized output specifically would manufacture
this paper's result, so it is computed per arm rather than pooled.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantartifact.scoring import extract  # noqa: E402

ANN = REPO / "annotation"
RUNGS = ("strict", "permissive")


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval. Correct at the boundaries, unlike the normal approximation."""
    if n == 0:
        return (0.0, 1.0)
    z, p = 1.959964, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def agreement_stats(a: list[str], b: list[str]) -> dict:
    """Raw agreement, Cohen's kappa and Gwet's AC1 on paired categorical labels.

    AC1 is the primary here. Cohen's kappa divides by a chance term that *grows* with
    prevalence imbalance, so on skewed labels it punishes exactly the data we have. AC1's
    chance term does not, which is why `A12` names it primary and forbids reporting kappa alone.
    """
    n = len(a)
    cats = sorted(set(a) | set(b))
    obs = sum(x == y for x, y in zip(a, b, strict=True)) / n

    # Cohen: expected agreement from each rater's own marginals.
    ca, cb = Counter(a), Counter(b)
    exp_cohen = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    kappa = (obs - exp_cohen) / (1 - exp_cohen) if exp_cohen < 1 else float("nan")

    # Gwet AC1: chance term from the POOLED marginal, which is what makes it prevalence-robust.
    q = len(cats)
    pi = {c: (ca[c] + cb[c]) / (2 * n) for c in cats}
    exp_ac1 = sum(pi[c] * (1 - pi[c]) for c in cats) / (q - 1) if q > 1 else 0.0
    ac1 = (obs - exp_ac1) / (1 - exp_ac1) if exp_ac1 < 1 else float("nan")

    lo, hi = wilson(sum(x == y for x, y in zip(a, b, strict=True)), n)
    return {
        "n": n, "raw_agreement": obs, "raw_agreement_ci95": [lo, hi],
        "cohen_kappa": kappa, "gwet_ac1": ac1,
        "confusion": {f"{x}|{y}": c for (x, y), c in
                      Counter(zip(a, b, strict=True)).items()},
    }


def main() -> int:
    labels: dict[str, dict[str, str]] = {}
    for who in ("A1", "A2"):
        path = ANN / f"labels_{who}.jsonl"
        if not path.is_file():
            raise SystemExit(f"missing {path.relative_to(REPO)}")
        labels[who] = {}
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                labels[who][r["annotation_id"]] = r["label"]
        print(f"{who}: {len(labels[who])} labels")

    key = {k["annotation_id"]: k for k in
           json.loads((ANN / "sample_key.json").read_text())}

    # ---- 1. IAA on the overlap --------------------------------------------------------
    overlap = sorted(set(labels["A1"]) & set(labels["A2"]))
    # §10: RUBRIC_GAP items are excluded from the agreement statistic and the count reported.
    gaps = [i for i in overlap if "RUBRIC_GAP" in (labels["A1"][i], labels["A2"][i])]
    scored = [i for i in overlap if i not in set(gaps)]
    iaa = agreement_stats([labels["A1"][i] for i in scored],
                          [labels["A2"][i] for i in scored])
    iaa["n_rubric_gap_excluded"] = len(gaps)

    print(f"\n§10 IAA on {iaa['n']} of {len(overlap)} overlap items "
          f"({len(gaps)} RUBRIC_GAP excluded)")
    print(f"  raw agreement : {iaa['raw_agreement']:.3f} "
          f"[{iaa['raw_agreement_ci95'][0]:.3f}, {iaa['raw_agreement_ci95'][1]:.3f}]")
    print(f"  Cohen kappa   : {iaa['cohen_kappa']:.3f}   (NOT reported alone, see A12)")
    print(f"  Gwet AC1      : {iaa['gwet_ac1']:.3f}   <- primary")
    if iaa["raw_agreement"] < 0.85:
        print("  🚨 below §10's ~0.85 threshold: that is a RUBRIC problem, not a judge "
              "problem. Revise the rubric, redo the dry run, re-annotate.")

    # ---- 2. the human verdict per item, for scorer validation -------------------------
    # Where both labelled and agreed, that is the verdict. Where they disagree, §10 requires
    # joint adjudication by rule lookup, which has not happened, so those items are held out
    # rather than resolved by a tie-break this script invents.
    human: dict[str, str] = {}
    unadjudicated = []
    for i in sorted(set(labels["A1"]) | set(labels["A2"])):
        got = [labels[w][i] for w in ("A1", "A2") if i in labels[w]]
        if "RUBRIC_GAP" in got:
            continue
        if len(set(got)) == 1:
            human[i] = got[0]
        else:
            unadjudicated.append(i)
    print(f"\nhuman verdicts  : {len(human)} ({len(unadjudicated)} disagreements await "
          f"§10 joint adjudication and are HELD OUT, not tie-broken)")

    # ---- 3. validate rungs 1 and 2 against the human verdict -------------------------
    rows = {}
    for p in (REPO / "artifacts" / "generations" / "pilot-v2").rglob("*.jsonl.gz"):
        import gzip  # noqa: PLC0415
        with gzip.open(p, "rt") as h:
            for line in h:
                if line.strip():
                    r = json.loads(line)
                    rows[(r["cell_id"], r["task"], r.get("subtask"), r["item_id"])] = r

    report: dict = {"iaa": iaa, "n_human_verdicts": len(human),
                    "n_unadjudicated_held_out": len(unadjudicated), "rungs": {}}

    for rung in RUNGS:
        per_arm: dict[str, list[tuple[bool, bool]]] = {"bf16": [], "4bit": []}
        for aid, verdict in human.items():
            k = key[aid]
            rec = rows.get((k["cell_id"], k["task"], k["subtask"], k["item_id"]))
            if rec is None:
                continue
            scored_text = extract.scored_text_for(rec)
            s = extract.score_generation(scored_text, rec.get("gold") or "",
                                         rec["task"], rec.get("subtask"),
                                         rungs=RUNGS, math_verify_available=False)
            bucket = "bf16" if k["arm"] == "bf16" else "4bit"
            per_arm[bucket].append((bool(s[rung]["correct"]), verdict == "CORRECT"))

        entry = {}
        for bucket, pairs in per_arm.items():
            if not pairs:
                continue
            n = len(pairs)
            acc = sum(a == b for a, b in pairs)
            fp = sum(1 for a, b in pairs if a and not b)   # scorer says right, human says not
            fn = sum(1 for a, b in pairs if not a and b)   # scorer says wrong, human says right
            lo, hi = wilson(acc, n)
            entry[bucket] = {"n": n, "agreement": acc / n, "agreement_ci95": [lo, hi],
                             "false_positive_rate": fp / n, "false_negative_rate": fn / n}
        report["rungs"][rung] = entry

    print("\nA12: scorer vs human, SEPARATELY by arm (a scorer lenient toward quantized "
          "output specifically would manufacture our result)")
    print(f"  {'rung':12} {'arm':6} {'n':>4} {'agree':>7} {'95% CI':>16} {'FP':>6} {'FN':>6}")
    for rung, entry in report["rungs"].items():
        for bucket, s in entry.items():
            print(f"  {rung:12} {bucket:6} {s['n']:>4} {s['agreement']:>6.1%} "
                  f"[{s['agreement_ci95'][0]:.3f},{s['agreement_ci95'][1]:.3f}] "
                  f"{s['false_positive_rate']:>5.1%} {s['false_negative_rate']:>5.1%}")

    # ⚠️ A10's LLM judge does not exist, so rung 4 is unvalidated and the paper says so.
    report["judge_validated"] = False
    report["judge_note"] = ("A10's judge harness does not exist, so rung 4 is unvalidated and "
                            "no claim rests on it. §2.2's headline is rung 1 vs rung 2, which "
                            "these labels validate directly.")

    out = REPO / "artifacts" / "annotation_validation.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
