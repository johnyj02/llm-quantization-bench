#!/usr/bin/env python3
"""Draw the blinded annotation sample. Task `A8`'s second half and `A11`'s input.

    .venv/bin/python scripts/draw_annotation_sample.py --run-id pilot-v2

Writes, per annotator, a file containing ONLY what they are permitted to see, shuffled with
that annotator's own published seed. Also writes a key file that maps item back to condition,
which annotators never open.

🚨 **BLINDING IS BY OMISSION, NOT BY HIDING** (prereg §10). The precision condition, the
parser's verdict and the judge's verdict are **absent from the annotator file**, because a
hidden column can be un-hidden and a hidden field in a JSON file is not hidden at all. The
join back to condition happens at scoring time through `item_id` and the key file.

**Why the blinding is load-bearing rather than hygiene.** Hróbjartsson et al. (*BMJ*
2012;344:e1119) found non-blinded assessors exaggerated odds ratios by 36% across 21 trials
with binary subjective outcomes, and the entire bias came from misclassifying a median of 3% of
items while agreeing on 78%. Our effects are small enough that a bias hiding behind high
agreement could manufacture the whole finding.

⚠️ **No pre-annotation, ever** (prereg §10). This script never writes a suggested label. Fort
and Sagot (LAW IV 2010) measured the better annotator dropping 98.1 to 95.8 under low-quality
pre-annotation, and Mikulová et al. (LREC 2022) found pre-annotation *raised* agreement, which
is shared anchoring rather than higher validity and is worse because it looks like success.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantartifact.scoring import extract  # noqa: E402

#: Published seeds, one per annotator. §10 requires the two orderings to differ and the seeds
#: to be published, so they live here in version control rather than in a shell history.
ANNOTATOR_SEEDS = {
    "A1": 20260907,   # Joseph Sankoorikal Johny
    "A2": 20260908,   # Syed Ahmed Danish
}

#: §10: at least 250 items, at least 120 of them double-annotated for IAA.
TARGET_TOTAL = 250
TARGET_OVERLAP = 120

#: §10 oversamples items where the automated rungs DISAGREE, because the agreement statistic
#: only carries information where the scorers differ. The natural distribution is reported too,
#: so the oversample is a stratum rather than a substitution.
#:
#: ⚠️ This is a CEILING, not a target, and the distinction was forced by the data. §10 registers
#: "oversampling rung-1/2/3 disagreement cases" without fixing a fraction, so the fraction is an
#: unregistered implementation parameter. `pilot-v2` turns out to contain only **59** disagreeing
#: items across 3,600 eligible records (1.6%), so a 0.6 target is unsatisfiable. Taking ALL of
#: them is the maximal oversample available and therefore satisfies §10 as written; asking for
#: more would mean inventing disagreements. The realised enrichment is reported in the manifest
#: rather than assumed, because a reader needs the achieved figure and not the requested one.
OVERSAMPLE_FRACTION = 0.6

RUNGS = ("strict", "permissive")


def load(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*.jsonl.gz")):
        with gzip.open(path, "rt") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def arm_of(cell_id: str) -> str:
    return cell_id.split("|", 1)[1] if "|" in cell_id else cell_id


def model_of(cell_id: str) -> str:
    return cell_id.split("|", 1)[0]


def stratum_of(record: dict) -> tuple:
    """§10: stratified across 3 models x {bf16, 4-bit} x all tasks."""
    arm = arm_of(record["cell_id"])
    bucket = "bf16" if arm == "bf16" else "4bit"
    task = record["task"] if not record.get("subtask") else \
        f"{record['task']}/{record['subtask']}"
    return (model_of(record["cell_id"]), bucket, task)


def rungs_disagree(record: dict) -> bool:
    """Do the automated rungs disagree on this item? Drives the §10 oversample."""
    text = extract.scored_text_for(record)
    gold = record.get("gold") or ""
    scored = extract.score_generation(text, gold, record["task"], record.get("subtask"),
                                      rungs=RUNGS, math_verify_available=False)
    return bool(scored["strict"]["correct"]) != bool(scored["permissive"]["correct"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw the blinded annotation sample")
    parser.add_argument("--run-id", default="pilot-v2",
                        help="§10 fixes this to pilot-v2 (O11); pilot-v1 is NOT usable")
    parser.add_argument("--out", default="annotation")
    args = parser.parse_args()

    root = REPO / "artifacts" / "generations" / args.run_id
    if not root.is_dir():
        raise SystemExit(f"no generations at {root}; sync them from S3 first")

    rows = load(root)
    print(f"pool            : {len(rows):,} records from {args.run_id}")

    # ⚠️ Only the 4-bit arms and bf16 are in scope per §10's stratification. gptq-8bit is a
    # dose-response arm and is not part of the {bf16, 4-bit} contrast the judge validates.
    eligible = [r for r in rows if arm_of(r["cell_id"]) in
                ("bf16", "gptq-4bit", "awq-4bit", "bnb-nf4")]
    print(f"eligible arms   : {len(eligible):,} records "
          f"({sorted({arm_of(r['cell_id']) for r in eligible})})")

    disagree = [r for r in eligible if rungs_disagree(r)]
    agree = [r for r in eligible if r not in disagree]
    print(f"rungs disagree  : {len(disagree):,}  (the §10 oversample stratum)")
    print(f"rungs agree     : {len(agree):,}  (the natural distribution)")

    n_over = min(int(TARGET_TOTAL * OVERSAMPLE_FRACTION), len(disagree))
    n_nat = TARGET_TOTAL - n_over
    if n_over < int(TARGET_TOTAL * OVERSAMPLE_FRACTION):
        print(f"  ⚠️ only {len(disagree)} disagreeing items exist, so the oversample is "
              f"CAPPED at all of them rather than at {OVERSAMPLE_FRACTION:.0%} of the draw")

    # Sample WITHIN strata so no (model, bucket, task) cell is starved. Deterministic seed so
    # the drawn set is reproducible from the repo alone.
    rng = random.Random(20260907)

    def stratified(pool: list[dict], want: int) -> list[dict]:
        by_stratum: dict[tuple, list[dict]] = defaultdict(list)
        for r in pool:
            by_stratum[stratum_of(r)].append(r)
        for group in by_stratum.values():
            rng.shuffle(group)
        picked, keys = [], sorted(by_stratum)
        while len(picked) < want and any(by_stratum[k] for k in keys):
            for k in keys:
                if by_stratum[k] and len(picked) < want:
                    picked.append(by_stratum[k].pop())
        return picked

    sample = stratified(disagree, n_over) + stratified(agree, n_nat)
    if len(sample) < TARGET_TOTAL:
        raise SystemExit(
            f"drew only {len(sample)} of {TARGET_TOTAL}; the pool cannot satisfy §10. "
            f"Refusing to under-draw silently."
        )
    print(f"drawn           : {len(sample)} items "
          f"({n_over} oversampled + {n_nat} natural)")

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    # ---- the KEY, which annotators never open ------------------------------------------
    key = [{
        "annotation_id": f"ann-{i:04d}",
        "run_id": r["run_id"], "cell_id": r["cell_id"], "task": r["task"],
        "subtask": r.get("subtask"), "item_id": r["item_id"],
        "arm": arm_of(r["cell_id"]), "model_key": model_of(r["cell_id"]),
        "stratum": list(stratum_of(r)),
        "rungs_disagree": r in disagree,
    } for i, r in enumerate(sample)]
    (out / "sample_key.json").write_text(json.dumps(key, indent=2))

    # ---- the ANNOTATOR files: only what they may see -----------------------------------
    # 🚨 This dict is the blinding. Adding `cell_id`, `arm` or any verdict here breaks §10.
    items = [{
        "annotation_id": k["annotation_id"],
        "question": r["prompt_text"].rsplit("\n", 1)[0][-1200:],
        "model_answer": extract.scored_text_for(r),
        "reference_answer": r.get("gold") or "",
    } for k, r in zip(key, sample, strict=True)]

    overlap = items[:TARGET_OVERLAP]
    rest = items[TARGET_OVERLAP:]
    half = len(rest) // 2
    assigned = {"A1": overlap + rest[:half], "A2": overlap + rest[half:]}

    orders = {}
    for who, seed in ANNOTATOR_SEEDS.items():
        block = list(assigned[who])
        random.Random(seed).shuffle(block)
        orders[who] = [x["annotation_id"] for x in block]
        (out / f"to_annotate_{who}.json").write_text(json.dumps(block, indent=2))
        print(f"  {who}: {len(block)} items, seed {seed} -> annotation/to_annotate_{who}.json")

    # §10 REQUIRES asserting the two orderings actually differ. A silent collision would mean
    # the two annotators saw the same sequence, and shared order is a shared anchor.
    common = [i for i in orders["A1"] if i in set(orders["A2"])]
    a1 = [i for i in orders["A1"] if i in set(common)]
    a2 = [i for i in orders["A2"] if i in set(common)]
    if a1 == a2:
        raise SystemExit("BLINDING FAILURE: both annotators would see the same order")
    print(f"  orderings differ on the {len(common)}-item overlap: OK")

    manifest = {
        "run_id": args.run_id, "n_items": len(sample),
        "n_overlap_double_annotated": TARGET_OVERLAP,
        "seeds": ANNOTATOR_SEEDS,
        "oversample_fraction_requested": OVERSAMPLE_FRACTION,
        "oversample_fraction_realised": round(n_over / TARGET_TOTAL, 4),
        "n_disagreeing_in_pool": len(disagree),
        "natural_disagreement_rate": round(len(disagree) / len(eligible), 5),
        "enrichment_factor": round(
            (n_over / TARGET_TOTAL) / (len(disagree) / len(eligible)), 1),
        "strata_covered": len({tuple(k["stratum"]) for k in key}),
        "per_annotator_counts": {w: len(v) for w, v in assigned.items()},
        "arms_in_sample": dict(Counter(k["arm"] for k in key)),
        "blinding": "condition, parser verdict and judge verdict are ABSENT from annotator "
                    "files, not hidden; no pre-annotation of any kind",
    }
    (out / "sample_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nmanifest        : annotation/sample_manifest.json")
    print(f"strata covered  : {manifest['strata_covered']}")
    print(f"arms            : {manifest['arms_in_sample']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
