#!/usr/bin/env python3
"""Generate the paper's figures from the analysis reports. No number is typed by hand.

    .venv/bin/python scripts/make_figures.py

**One figure, deliberately.** The body is 10 pages against a hard 8, so a figure has to
displace prose to earn its place. Of the three candidates, this is the one that does:

- ❌ H1's degradation-versus-budget curve. Four lines, none of which excludes zero after Holm
  over m=4. Plotting four overlapping nulls invites the reader to see a trend the statistics
  do not support, which is the opposite of what this paper argues for.
- ✅ **Cap-hit rate versus budget, by arm.** The mechanism rather than the effect: a clean
  monotone separation where the quantized arms start more truncated and converge downward.
  This is the strongest evidence in H1 and it needs a figure, because six monotone series
  across two tasks is exactly what a table reads badly.
- ❌ H8's `bf16` at exactly zero. One sentence carries it ("identical under all ten orderings
  at both rungs"), and a bar at zero next to three small bars is a weak picture of a strong
  fact.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # no display on a build box
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "figures"

LEVELS = ("1x", "2x", "4x")
ARMS = ("bf16", "gptq-4bit", "awq-4bit")
LABEL = {"bf16": "bf16", "gptq-4bit": "GPTQ W4A16", "awq-4bit": "AWQ W4A16"}
# Colour-blind safe, and distinguishable in greyscale print by marker as well as hue.
STYLE = {"bf16": ("#0072B2", "o", "-"),
         "gptq-4bit": ("#D55E00", "s", "--"),
         "awq-4bit": ("#009E73", "^", ":")}
TASK_TITLE = {"gsm8k_cot": "GSM8K (CoT prompt)", "bbh": "BIG-Bench Hard"}


def main() -> int:
    data = json.loads((REPO / "artifacts" / "analysis" / "analysis_h1.json").read_text())
    length = data["length"]
    OUT.mkdir(parents=True, exist_ok=True)

    tasks = [t for t in ("gsm8k_cot", "bbh")
             if any(k.startswith(f"{t}|") for k in length)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(7.0, 2.9), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks, strict=True):
        for arm in ARMS:
            xs, ys = [], []
            for i, lv in enumerate(LEVELS):
                entry = length.get(f"{task}|{arm}|{lv}")
                if entry is None:
                    continue
                xs.append(i)
                ys.append(100 * entry["hit_cap_rate"])
            if not xs:
                continue
            colour, marker, dash = STYLE[arm]
            ax.plot(xs, ys, marker=marker, linestyle=dash, color=colour,
                    label=LABEL[arm], linewidth=1.6, markersize=5)
        ax.set_xticks(range(len(LEVELS)))
        ax.set_xticklabels([r"$1\times$", r"$2\times$", r"$4\times$"])
        ax.set_xlabel("generation budget")
        ax.set_title(TASK_TITLE.get(task, task), fontsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel("generations hitting\nthe cap (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(pad=0.4)

    path = OUT / "h1_cap_hit_rate.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # H1's prose macros, generated here so the section inlines no number (R17.6.2).
    eff = {e["name"]: e for e in data["effects"]}
    holm = sorted(data["holm"], key=lambda r: r["p_value"])[0]
    best = min(data["effects"], key=lambda e: e["p_value"])
    L = lambda k: length[k]          # noqa: E731
    macros = {
        "hOneItems": f"{data['n_items_joined']:,}",
        "hOneCapBfOne": f"{100 * L('gsm8k_cot|bf16|1x')['hit_cap_rate']:.1f}",
        "hOneCapBfFour": f"{100 * L('gsm8k_cot|bf16|4x')['hit_cap_rate']:.1f}",
        "hOneCapGptqOne": f"{100 * L('gsm8k_cot|gptq-4bit|1x')['hit_cap_rate']:.1f}",
        "hOneCapGptqFour": f"{100 * L('gsm8k_cot|gptq-4bit|4x')['hit_cap_rate']:.1f}",
        "hOneCapAwqOne": f"{100 * L('gsm8k_cot|awq-4bit|1x')['hit_cap_rate']:.1f}",
        "hOneCapAwqFour": f"{100 * L('gsm8k_cot|awq-4bit|4x')['hit_cap_rate']:.1f}",
        "hOneDiscardGptq": f"{L('bbh|gptq-4bit|4x')['full_mean'] - L('bbh|gptq-4bit|4x')['scored_mean']:.0f}",
        "hOneDiscardBf": f"{L('bbh|bf16|4x')['full_mean'] - L('bbh|bf16|4x')['scored_mean']:.0f}",
        "hOnePMin": f"{holm['p_value']:.3f}",
        "hOneAlphaMin": f"{holm['adjusted_alpha']:.4f}",
        "hOneBestEst": f"{best['estimate']:+.2f}",
        "hOneBestCI": f"{best['ci_low']:+.2f} to {best['ci_high']:+.2f}",
        "hOneLenBf": f"{L('gsm8k_cot|bf16|1x')['scored_mean']:.0f}",
        "hOneLenGptq": f"{L('gsm8k_cot|gptq-4bit|1x')['scored_mean']:.0f}",
        "hOneLenAwq": f"{L('gsm8k_cot|awq-4bit|1x')['scored_mean']:.0f}",
    }
    tex = REPO / "paper" / "tables" / "h1_macros.tex"
    tex.write_text("% GENERATED by scripts/make_figures.py -- do not hand-edit\n" +
                   "".join(rf"\newcommand{{\{k}}}{{{v}\xspace}}" + "\n"
                           for k, v in macros.items()))
    print(f"wrote {tex.relative_to(REPO)} ({len(macros)} macros)")
    assert eff, "no effects in the report"

    # Print the plotted numbers so the figure can be checked against the report by eye.
    print(f"wrote {path.relative_to(REPO)} (+ .png)")
    print(f"\n{'cell':26} {'cap-hit %':>10}")
    for task in tasks:
        for arm in ARMS:
            for lv in LEVELS:
                e = length.get(f"{task}|{arm}|{lv}")
                if e:
                    print(f"  {task}|{arm}|{lv:3} {100 * e['hit_cap_rate']:>9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
