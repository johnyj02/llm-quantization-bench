#!/usr/bin/env python3
"""H6: does the quantization delta on ARC depend on whether an extraction step exists?

    .venv/bin/python scripts/analyze_h6.py

Arm 1 is `arc_challenge_chat` in `main-v1`: generate, then parse a letter. Arm 2 is `h6-v1`:
the identical prompt, scored by log-likelihood over the four letters, so no extraction step
exists. Same items, prompt, engine, weights and hardware (verified on the records).

The registered quantity is the **mode x precision interaction**: the quantization delta
(bf16 minus the quantized arm) under generation, minus the same delta under log-likelihood,
paired by item. H6 predicts it is ZERO, because ARC's answer space is a single letter, so
extraction has nothing to distort. A non-zero value would mean extraction moves the reported
quantization delta even where the answer space is maximally constrained.

Paired cluster bootstrap over items, 10,000 draws, fixed seed, two-sided bootstrap p-value,
Holm over the family. Class 3 (compounding) under SS9f: it contrasts arms.
"""

from __future__ import annotations

import gzip
import json
import random
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantartifact.analysis.results import Effect, NoiseClass, holm_bonferroni  # noqa: E402
from quantartifact.scoring import extract  # noqa: E402

MODELS = ("qwen2.5-3b", "phi-3.5-mini", "llama-3.1-8b")
QUANT = ("gptq-4bit", "awq-4bit")
GEN_RUNG = "permissive"   # §2.2: rung 2 primary. On ARC strict and permissive agree (H3).


def load(run: str, task: str) -> dict:
    out = {}
    for path in (REPO / "artifacts" / "generations" / run).rglob(f"{task}*.jsonl.gz"):
        with gzip.open(path, "rt") as handle:
            for line in handle:
                if line.strip():
                    r = json.loads(line)
                    if r.get("repeat_idx", 0) == 0:   # §9f: one measurement per item
                        out[(r["cell_id"], r["item_id"])] = r
    return out


def gen_correct(r: dict) -> int:
    s = extract.score_generation(extract.scored_text_for(r), r.get("gold") or "", r["task"],
                                 None, rungs=(GEN_RUNG,), math_verify_available=False)
    return int(bool(s[GEN_RUNG]["correct"]))


def paired_effect(contrib: list[int], name: str) -> Effect:
    """Mean of per-item contributions, in accuracy points, with a paired bootstrap."""
    rng = random.Random(20261016)
    n = len(contrib)
    point = 100 * sum(contrib) / n
    draws = sorted(100 * sum(contrib[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(10_000))
    wrong = min(sum(d >= 0 for d in draws), sum(d <= 0 for d in draws))
    return Effect(
        name=name, estimate=point, unit="pp", n=n,
        ci_low=draws[250], ci_high=draws[9749],
        p_value=max(2 * wrong / len(draws), 1 / len(draws)),
        test="paired bootstrap two-sided (proportion of draws past the null)",
        interval_method="paired cluster bootstrap over items, 10000 percentile draws",
        noise_class=NoiseClass.COMPOUNDING,
        context={"seed": 20261016, "gen_rung": GEN_RUNG},
    )


def main() -> int:
    ll = load("h6-v1", "arc_challenge_ll_matched")
    gen = load("main-v1", "arc_challenge_chat")
    print(f"arm 2 (log-likelihood): {len(ll):,} records | arm 1 (generation): {len(gen):,}")

    report: dict = {"accuracy": {}, "extraction_cost": {}, "effects": []}
    effects = []
    print(f"\n{'cell':24} {'LL acc':>7} {'gen acc':>8} {'LL - gen':>9}")
    for model in MODELS:
        arms = ("bf16", *QUANT)
        items = sorted({iid for (c, iid) in ll if c.startswith(model)}
                       & {iid for (c, iid) in gen if c.startswith(model)})
        # Every arm must cover the same items, or the paired contrast is not paired.
        items = [i for i in items if all((f"{model}|{a}", i) in ll and
                                         (f"{model}|{a}", i) in gen for a in arms)]
        c = {a: ([int(ll[(f"{model}|{a}", i)]["generation_text"] ==
                      ll[(f"{model}|{a}", i)]["gold"]) for i in items],
                 [gen_correct(gen[(f"{model}|{a}", i)]) for i in items]) for a in arms}
        for a in arms:
            acc_ll, acc_gen = statistics.mean(c[a][0]), statistics.mean(c[a][1])
            report["accuracy"][f"{model}|{a}"] = {"ll": acc_ll, "gen": acc_gen,
                                                  "n": len(items)}
            report["extraction_cost"][f"{model}|{a}"] = 100 * (acc_ll - acc_gen)
            print(f"  {model}|{a:10} {acc_ll:>6.1%} {acc_gen:>7.1%} "
                  f"{100 * (acc_ll - acc_gen):>+8.2f}")
        for q in QUANT:
            # d_i = (bf16 - q) under generation  minus  (bf16 - q) under log-likelihood
            d = [(c["bf16"][1][k] - c[q][1][k]) - (c["bf16"][0][k] - c[q][0][k])
                 for k in range(len(items))]
            effects.append(paired_effect(d, f"H6_mode_x_precision[{model}|{q}]"))

    report["effects"] = [e.as_dict() for e in effects]
    holm = holm_bonferroni(effects, family="H6")
    report["holm"] = holm

    print("\nH6, mode x precision interaction (gen delta minus LL delta), pp:")
    print(f"  {'contrast':30} {'est':>7} {'95% CI':>18} {'p':>7} {'Holm':>7}")
    for e, h in zip(effects, holm, strict=True):
        hd = h if isinstance(h, dict) else h.as_dict()
        print(f"  {e.name[len('H6_mode_x_precision['):-1]:30} {e.estimate:>+6.2f} [{e.ci_low:>+6.2f},{e.ci_high:>+6.2f}]"
              f" {e.p_value:>7.4f} {'REJECT' if hd.get('rejected') else 'keep':>7}")

    out = REPO / "artifacts" / "analysis" / "analysis_h6.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True,
                              default=lambda o: o.as_dict() if hasattr(o, "as_dict") else str(o)))
    print(f"\nreport -> {out.relative_to(REPO)}")

    # Prose macros for the H6 subsection, so the section inlines no number (R17.6.2).
    E = {e.name[len('H6_mode_x_precision['):-1]: e for e in effects}
    acc = report["accuracy"]
    rej = sum(1 for h in holm if (h if isinstance(h, dict) else h.as_dict()).get("rejected"))
    m = {
        "hSixQwenGptq": f"{E['qwen2.5-3b|gptq-4bit'].estimate:+.1f}",
        "hSixQwenGptqCI": f"{E['qwen2.5-3b|gptq-4bit'].ci_low:+.1f} to "
                          f"{E['qwen2.5-3b|gptq-4bit'].ci_high:+.1f}",
        "hSixQwenAwq": f"{E['qwen2.5-3b|awq-4bit'].estimate:+.1f}",
        "hSixMaxOther": f"{max(abs(E[k].estimate) for k in E if not k.startswith('qwen')):.1f}",
        "hSixRejected": str(rej), "hSixM": str(len(effects)),
        "hSixQwenBfGen": f"{100 * acc['qwen2.5-3b|bf16']['gen']:.1f}",
        "hSixQwenBfLL": f"{100 * acc['qwen2.5-3b|bf16']['ll']:.1f}",
        "hSixQwenGptqGen": f"{100 * acc['qwen2.5-3b|gptq-4bit']['gen']:.1f}",
        "hSixQwenGptqLL": f"{100 * acc['qwen2.5-3b|gptq-4bit']['ll']:.1f}",
        "hSixItems": str(acc["qwen2.5-3b|bf16"]["n"]),
    }
    tex = REPO / "paper" / "tables" / "h6_macros.tex"
    tex.write_text("% GENERATED by scripts/analyze_h6.py -- do not hand-edit\n" + "".join(
        rf"\newcommand{{\{k}}}{{{v}\xspace}}" + "\n" for k, v in m.items()))
    print(f"wrote {tex.relative_to(REPO)} ({len(m)} macros)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
