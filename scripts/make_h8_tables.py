#!/usr/bin/env python3
r"""Generate H8's LaTeX tables from the job reports. No number is typed by hand.

    .venv/bin/python scripts/make_h8_tables.py

Reads  `artifacts/determinism_reanalysis.json`  -- every cell, every metric level
       `artifacts/determinism.json`             -- run metadata and the Marlin pairing
Writes `paper/tables/h8_*.tex` and `paper/tables/h8_macros.tex`

WHY THIS EXISTS (planner ruling R17.6.2). Every statistic in the H8 section must be
sourced from a generated file rather than inlined, because the 10-ordering job (R17.3)
will replace all of them and a hand-edited section would then be silently stale --- in a
paper whose whole thesis is that unreported measurement choices move published numbers.
An inlined number nobody re-derives is precisely the failure mode we are writing about.

⚠️ **The estimand is the SD across orderings, not the range** (R17.3). Range is unstable
and *grows with sample size*, so it cannot carry a population claim; it is emitted as
descriptive only. With the orderings currently in hand the SD is itself weakly estimated,
which is why the ordering COUNT is a macro rather than the word "three" --- the prose has
to stay hedged until the 10-ordering run lands, and the macro makes that visible.

⚠️ The two reports have DIFFERENT shapes for the same field: `text_divergence` is a bare
float, while `answer_divergence` and `correctness_divergence` are keyed by scoring rung.
Everything is therefore read from the reanalysis, which carries all metrics for all 21
cells; `determinism.json` is consulted only for metadata and the Marlin pairing.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
TABLES = REPO / "paper" / "tables"

# Display order, coarsest precision last. Matches the arm ordering used elsewhere.
ARM_ORDER = ["bf16", "gptq-8bit", "gptq-4bit", "awq-4bit", "bnb-nf4"]
ARM_LABEL = {
    "bf16": r"\textsc{bf16}",
    "gptq-8bit": "GPTQ W8A16",
    "gptq-4bit": "GPTQ W4A16",
    "awq-4bit": "AWQ W4A16",
    "bnb-nf4": "bnb NF4",
}
AUDIT_MODEL = "qwen2.5-3b"
# R17.4: rung 2 is primary, rung 1 reported alongside, per SS2.2's headline contrast.
PRIMARY_RUNG = "permissive"


def frozen_max_num_seqs() -> int:
    """The frozen serving value, read from `configs/models.yaml` — the one source of truth.

    ⚠️ Deliberately NOT derived from the measured levels. R24.1 moved the freeze from 64 to
    32 while the sweep continues to measure {8, 32, 64}, so `max(levels)` would report a
    value the study no longer runs at.
    """
    import yaml  # noqa: PLC0415 - only needed here

    serving = yaml.safe_load(
        (REPO / "configs" / "models.yaml").read_text())["serving"]
    if not serving.get("max_num_seqs_resolved_by_audit"):
        raise SystemExit(
            "configs/models.yaml: max_num_seqs is not marked resolved_by_audit; refusing "
            "to print an unaudited value into the paper"
        )
    return int(serving["max_num_seqs"])



def engine_version() -> str:
    """vLLM version as recorded on the generations themselves, e.g. "0.26.0"."""
    import gzip  # noqa: PLC0415

    for path in sorted((REPO / "artifacts" / "generations" / "main-v1").rglob("*.jsonl.gz")):
        with gzip.open(path, "rt") as handle:
            line = handle.readline()
        if line.strip():
            engine = json.loads(line)["engine"]
            if not engine.startswith("vllm=="):
                raise SystemExit(f"unexpected engine field {engine!r}")
            return engine.removeprefix("vllm==")
    raise SystemExit("no main-v1 records to read the engine version from")

def pct(value: float | None, places: int = 2) -> str:
    return "--" if value is None else f"{value * 100:.{places}f}"


def load(name: str):
    path = ARTIFACTS / name
    if not path.is_file():
        raise SystemExit(f"missing {path} — download it from the job's ModelArtifacts")
    return json.loads(path.read_text())


def variant(cell: dict) -> str:
    """`…__at8__r2__fallback` -> "fallback"; a plain sweep cell -> ""."""
    parts = cell["cell"].split("__")
    return parts[4] if len(parts) > 4 else ""


def sweep_cells(reanalysis: list, model: str = AUDIT_MODEL) -> dict:
    """(arm, level) -> cell, for the plain sweep only (no fallback/autotune)."""
    return {
        (c["arm"], c["max_num_seqs"]): c
        for c in reanalysis
        if c["model_key"] == model and not variant(c)
    }


def levels_of(reanalysis: list, model: str = AUDIT_MODEL) -> list[int]:
    return sorted({c["max_num_seqs"] for c in reanalysis if c["model_key"] == model})


def sd_and_range(cell: dict, rung: str) -> tuple[float | None, float | None]:
    values = ((cell.get("by_level") or {}).get("accuracy_by_ordering") or {}).get(rung)
    if not values or len(values) < 2:
        return None, None
    return statistics.stdev(values), max(values) - min(values)


def _tabular(colspec: str, lines: list[str]) -> str:
    return "\n".join([
        r"% GENERATED by scripts/make_h8_tables.py -- do not hand-edit",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        *lines,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def frontier_table(reanalysis: list) -> str:
    """Divergence at three metric levels x each batch size.

    The frontier IS the contribution (R16.2) -- it tells other people what they can get
    away with -- so all three levels are shown rather than only the frozen value. Text
    divergence overstates the correctness impact by roughly an order of magnitude, which
    is exactly why R15.3 required reporting all three.
    """
    levels = levels_of(reanalysis)
    cells = sweep_cells(reanalysis)
    rows = []
    for arm in ARM_ORDER:
        parts = []
        for level in levels:
            by = (cells.get((arm, level)) or {}).get("by_level") or {}
            parts.append(" & ".join([
                pct(by.get("text_divergence"), 1),
                pct((by.get("correctness_divergence") or {}).get("strict"), 1),
                pct((by.get("correctness_divergence") or {}).get("permissive"), 1),
            ]))
        rows.append(f"{ARM_LABEL[arm]} & " + " & ".join(parts) + r" \\")
    return _tabular(
        "l" + "ccc" * len(levels),
        [
            "& " + " & ".join(rf"\multicolumn{{3}}{{c}}{{$B={lv}$}}" for lv in levels)
            + r" \\",
            r"\cmidrule(lr){2-" + str(1 + 3 * len(levels)) + "}",
            "arm & " + " & ".join(["text & str. & perm."] * len(levels)) + r" \\",
            r"\midrule",
            *rows,
        ],
    )


def spread_table(reanalysis: list) -> str:
    """Accuracy across orderings: SD is inferential, range descriptive (R17.3)."""
    levels = levels_of(reanalysis)
    cells = sweep_cells(reanalysis)
    rows = []
    for arm in ARM_ORDER:
        parts = []
        for level in levels:
            cell = cells.get((arm, level))
            sd, rng = sd_and_range(cell, PRIMARY_RUNG) if cell else (None, None)
            parts.append(f"{pct(sd)} & {pct(rng)}")
        rows.append(f"{ARM_LABEL[arm]} & " + " & ".join(parts) + r" \\")
    return _tabular(
        "l" + "cc" * len(levels),
        [
            "& " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$B={lv}$}}" for lv in levels)
            + r" \\",
            r"\cmidrule(lr){2-" + str(1 + 2 * len(levels)) + "}",
            "arm & " + " & ".join(["SD & range"] * len(levels)) + r" \\",
            r"\midrule",
            *rows,
        ],
    )


def mechanism_table(reanalysis: list) -> str:
    """The intervention, paired +/- VLLM_DISABLED_KERNELS.

    Reported on BOTH axes: text divergence (job 2) and accuracy SD (the reanalysis).
    The second is what matters -- it pins the mechanism to the quantity the paper
    actually reports rather than to strings. bnb NF4 is unmoved by the intervention,
    which is a second, distinct, Marlin-independent mechanism.
    """
    floor = min(levels_of(reanalysis))
    rows = []
    for arm in ARM_ORDER:
        on = next((c for c in reanalysis
                   if c["arm"] == arm and c["model_key"] == AUDIT_MODEL
                   and c["max_num_seqs"] == floor and not variant(c)), None)
        off = next((c for c in reanalysis
                    if c["arm"] == arm and c["model_key"] == AUDIT_MODEL
                    and variant(c) == "fallback"), None)
        if on is None or off is None:
            continue
        on_sd, _ = sd_and_range(on, PRIMARY_RUNG)
        off_sd, _ = sd_and_range(off, PRIMARY_RUNG)
        rows.append(
            f"{ARM_LABEL[arm]} & "
            f"{pct((on.get('by_level') or {}).get('text_divergence'), 1)} & "
            f"{pct((off.get('by_level') or {}).get('text_divergence'), 1)} & "
            f"{pct(on_sd)} & {pct(off_sd)}" + r" \\"
        )
    return _tabular("lcccc", [
        r"& \multicolumn{2}{c}{text divergence} & \multicolumn{2}{c}{accuracy SD} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"arm & Marlin on & off & Marlin on & off \\",
        r"\midrule",
        *rows,
    ])


def macros(report: dict, reanalysis: list, report10: dict) -> str:
    """Single-number facts the prose needs, as macros so the prose cannot go stale."""
    sd_level = max(c["max_num_seqs"] for c in report10["sweep"])
    # Worst text and worst correctness divergence across the plain sweep cells, for the
    # body-prose anchor described below. bf16 is excluded: it is exactly zero at every
    # level, so including it would make the ratio undefined rather than conservative.
    _sweep = [c for c in reanalysis
              if c["model_key"] == AUDIT_MODEL and not variant(c) and c["arm"] != "bf16"]
    worst_text = max((c["by_level"] or {}).get("text_divergence") or 0.0 for c in _sweep)
    worst_corr = max(
        max(((c["by_level"] or {}).get("correctness_divergence") or {}).get(r) or 0.0
            for r in ("strict", "permissive"))
        for c in _sweep
    )
    quant = [c for c in reanalysis
             if c["model_key"] == AUDIT_MODEL and c["arm"] != "bf16" and not variant(c)]
    worst_range = worst_sd = 0.0
    for cell in quant:
        for rung in ("strict", PRIMARY_RUNG):
            sd, rng = sd_and_range(cell, rung)
            worst_sd = max(worst_sd, sd or 0.0)
            worst_range = max(worst_range, rng or 0.0)

    # The control, COUNTED rather than asserted from memory: every bf16 cell, both
    # rungs, must be exactly zero for the attribution to hold.
    bf16_cells = bf16_zero = 0
    for cell in (c for c in reanalysis if c["arm"] == "bf16"):
        for rung in ("strict", PRIMARY_RUNG):
            _, rng = sd_and_range(cell, rung)
            if rng is not None:
                bf16_cells += 1
                bf16_zero += int(rng == 0)

    # 10-ordering aggregates (R17.3). bf16's zero count is COUNTED, not asserted: it is
    # the strongest single number in H8 and must not rest on recall.
    worst_sd10 = worst_range10 = 0.0
    bf16_ten_zero = bf16_ten_cells = 0
    for cell in report10["sweep"]:
        for rung in ("strict", PRIMARY_RUNG):
            sd, rng = sd_and_range(cell, rung)
            if sd is None:
                continue
            if cell["arm"] == "bf16":
                bf16_ten_cells += 1
                bf16_ten_zero += int(rng == 0)
            else:
                worst_sd10 = max(worst_sd10, sd)
                worst_range10 = max(worst_range10, rng)

    pinned = [c["arm"] for c in reanalysis
              if variant(c) == "fallback"
              and (sd_and_range(c, PRIMARY_RUNG)[1] == 0)
              and c["arm"] != "bnb-nf4"]
    floors = [c["repetition_floor"] for c in report["sweep"]
              if c.get("repetition_floor") is not None]

    return "\n".join([
        r"% GENERATED by scripts/make_h8_tables.py -- do not hand-edit",
        rf"\newcommand{{\hEightOrderings}}{{{len(report['orderings'])}\xspace}}",
        rf"\newcommand{{\hEightItems}}{{{report['n_items']}\xspace}}",
        rf"\newcommand{{\hEightWorstRange}}{{{worst_range * 100:.1f}\xspace}}",
        rf"\newcommand{{\hEightWorstSD}}{{{worst_sd * 100:.2f}\xspace}}",
        rf"\newcommand{{\hEightBFCells}}{{{bf16_cells}\xspace}}",
        rf"\newcommand{{\hEightBFZero}}{{{bf16_zero}\xspace}}",
        rf"\newcommand{{\hEightRepFloor}}{{{max(floors) * 100:.3f}\xspace}}",
        rf"\newcommand{{\hEightNCells}}{{{len(report['sweep'])}\xspace}}",
        rf"\newcommand{{\hEightPinned}}{{{len(pinned)}\xspace}}",
        # 🚨 From configs/models.yaml, NOT from the largest measured level. Those
        # coincided while the freeze was 64 and DIVERGED the moment R24.1 moved it to
        # 32 -- the sweep still measures {8,32,64}, so the old expression silently kept
        # printing 64 into a paper that now runs at 32. A derived value that happens to
        # agree with the real one is not a source of truth.
        rf"\newcommand{{\hEightFrozenB}}{{{frozen_max_num_seqs()}\xspace}}",
        # 🚨 The vLLM version the RECORDS report. This used to print
        # report['determinism_version'], our own probe script's "1.0.0", and the paper then
        # said "vLLM 1.0.0" in Method and in the frozen-configuration appendix. We ran 0.26.0.
        rf"\newcommand{{\hEightEngine}}{{{engine_version()}\xspace}}",
        # 10-ordering facts. DELIBERATELY separate macros from the 3-ordering ones: the two
        # measure different populations (see sd_table) and must never be interchanged.
        rf"\newcommand{{\hEightSDOrderings}}{{{len(report10['orderings'])}\xspace}}",
        rf"\newcommand{{\hEightWorstSDTen}}{{{worst_sd10 * 100:.2f}\xspace}}",
        rf"\newcommand{{\hEightRangeTen}}{{{worst_range10 * 100:.1f}\xspace}}",
        rf"\newcommand{{\hEightBFTenZero}}{{{bf16_ten_zero}\xspace}}",
        rf"\newcommand{{\hEightBFTenCells}}{{{bf16_ten_cells}\xspace}}",
        # 🚨 The level the 10-ordering run ACTUALLY measured, read from its own report.
        # NOT `\hEightFrozenB`: that run measured B=64 only, and R24.1 then moved the
        # freeze to 32 — so using the frozen-value macro in its caption would have
        # labelled B=64 data as B=32, in the table the H8 section LEADS with.
        rf"\newcommand{{\hEightSDLevel}}{{{sd_level}\xspace}}",
        # 🚨 The three-level comparison IN BODY PROSE, not only in the appendix table.
        # R42/§7: an appendix must not carry load-bearing content, because reviewers are
        # not expected to read it and hiding substantive material there is itself a
        # desk-reject ground. When the frontier table moved to the appendix, the body's
        # only quantitative claim became "roughly an order of magnitude", which depends on
        # a table the reviewer will not see. These anchor it: worst text divergence against
        # the worst correctness divergence at the same measurement, so the ratio the claim
        # rests on is checkable from the prose alone.
        rf"\newcommand{{\hEightWorstText}}{{{worst_text * 100:.1f}\xspace}}",
        rf"\newcommand{{\hEightWorstCorr}}{{{worst_corr * 100:.1f}\xspace}}",
        rf"\newcommand{{\hEightTextCorrRatio}}{{{worst_text / worst_corr:.0f}\xspace}}",
    ])


def sd_table(report10: dict) -> str:
    """SD and range of accuracy across the TEN orderings (R17.3, prereg SS9a).

    ⚠️ **SUPERSEDES the 3-ordering spread table rather than extending it.** The earlier probe
    predated the BBH terminator fix, so ~2/5 of its items were inert and its spreads were
    attenuated. The two are never pooled.

    ⚠️ SD is the inferential quantity; the range is descriptive ONLY -- and our own data now
    demonstrates why rather than merely asserting it: going from 3 to 10 orderings grew the
    observed range from 1.6 pp to 3.8 pp with NO change to the underlying effect, because the
    range is monotone in the number of draws.
    """
    cells = {c["arm"]: c for c in report10["sweep"]}
    rows = []
    for arm in ARM_ORDER:
        cell = cells.get(arm)
        if cell is None:
            continue
        parts = []
        for rung in ("strict", PRIMARY_RUNG):
            sd, rng = sd_and_range(cell, rung)
            parts.append(f"{pct(sd)} & {pct(rng)}")
        rows.append(f"{ARM_LABEL[arm]} & " + " & ".join(parts) + r" \\")
    return _tabular("lcccc", [
        r"& \multicolumn{2}{c}{strict} & \multicolumn{2}{c}{permissive} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"arm & SD & range & SD & range \\",
        r"\midrule",
        *rows,
    ])


def as_cells(report: dict) -> list:
    """`determinism.json`'s cell lists in the flat shape the table builders expect.

    🚨 WHY THIS EXISTS. `frontier_table` and `mechanism_table` used to read
    `determinism_reanalysis.json`, on the reasoning that the reanalysis carried all three
    metric levels. It does — but it is the output of a SEPARATE, EARLIER job, so once the
    determinism sweep was re-run with corrected BBH terminators (job 33) the frontier table
    silently kept printing the SUPERSEDED numbers: gptq-4bit 4.8% correctness at B=64 where
    the re-run measured 11.2%. That is the number R24.1 moved the freeze over, so the paper
    would have contradicted its own pre-registration.

    `determinism.json` carries `by_level` with all three levels too, so the fresh report is
    both sufficient and authoritative. The reanalysis is now used ONLY for the accuracy
    spread, which is the one quantity it alone computes.
    """
    out = []
    for key, suffix in (("sweep", ""), ("marlin_attribution", "fallback"),
                        ("autotune_contrast", "autotune"),
                        ("model_generality", "")):
        for c in report.get(key) or []:
            cell = dict(c)
            tag = c.get("cell")
            if not tag:
                tag = (f"{c['model_key']}__{c['arm']}__at{c['max_num_seqs']}__r2"
                       + (f"__{suffix}" if suffix else ""))
            cell["cell"] = tag
            out.append(cell)
    return out


def spotcheck_macros(spot: dict, report10: dict) -> str:
    """The second-GPU replication (§4.1), from its own job report.

    🚨 Refuses unless the report says it is a spot-check on a GPU other than the one
    `report10` was measured on, because the paper's sentence is a claim of replication.
    """
    l40 = {c["arm"]: c for c in report10["sweep"]}
    gpu10 = (l40["gptq-4bit"].get("hardware") or {}).get("gpu")
    if not spot.get("spotcheck") or spot["gpu"] in (gpu10, None, "unknown"):
        raise SystemExit(f"not a second-GPU spot-check: gpu={spot.get('gpu')!r}, "
                         f"Table 1 gpu={gpu10!r}")
    # The paper prints the two GPUs side by side, so they must be the same measurement.
    # Divergence grows with the number of orderings compared, so a mismatch is not noise.
    shape = lambda r: (r["order_sweep_levels"], len(r["orderings"]), r["n_items"])  # noqa: E731
    if shape(spot) != shape(report10):
        raise SystemExit(f"spot-check {shape(spot)} != Table 1 {shape(report10)} "
                         f"(levels, orderings, items); not comparable")
    # Read from the report, never from the freeze (see `\hEightSDLevel` above).
    (level,) = spot["order_sweep_levels"]
    ten = {c["arm"]: c for c in spot["sweep"] if c["max_num_seqs"] == level}
    off = spot["marlin_attribution"][0]
    # The Marlin-ON half is the sweep cell at the SAME level as the Marlin-off cell.
    on = next(c for c in spot["sweep"] if c["max_num_seqs"] == off["max_num_seqs"]
              and not c["disabled_kernels"])
    bf_sd = max(sd_and_range(ten["bf16"], r)[1] for r in ("strict", PRIMARY_RUNG))
    if bf_sd != 0 or ten["bf16"]["order_divergence"] != 0:
        raise SystemExit(f"bf16 accuracy moved on the spot-check (range {bf_sd}); "
                         f"or its text differed; the prose says exactly unchanged")
    m = {
        "hEightSpotGPU": spot["gpu"].removeprefix("NVIDIA "),
        "hEightSpotItems": str(spot["n_items"]),
        "hEightSpotOrderings": str(len(spot["orderings"])),
        "hEightSpotLevel": str(level),
        "hEightSpotBFText": f"{ten['bf16']['order_divergence'] * 100:.1f}",
        "hEightSpotText": f"{ten['gptq-4bit']['order_divergence'] * 100:.1f}",
        "hEightTenText": f"{l40['gptq-4bit']['order_divergence'] * 100:.1f}",
        "hEightSpotSD": f"{sd_and_range(ten['gptq-4bit'], 'strict')[0] * 100:.2f}",
        "hEightTenSD": f"{sd_and_range(l40['gptq-4bit'], 'strict')[0] * 100:.2f}",
        "hEightSpotFloor": str(off["max_num_seqs"]),
        "hEightSpotMarlinOn": f"{on['order_divergence'] * 100:.1f}",
        "hEightSpotMarlinOff": f"{off['order_divergence'] * 100:.1f}",
    }
    return "% GENERATED by scripts/make_h8_tables.py -- do not hand-edit\n" + "\n".join(
        rf"\newcommand{{\{k}}}{{{v}\xspace}}" for k, v in m.items())


def main() -> int:
    report = load("determinism.json")
    report10 = load("determinism_ord10.json")
    # ⚠️ EVERY 3-ordering table reads the SAME fresh report. `determinism.json` now carries
    # `accuracy_by_ordering` natively (the R16.5 work folded it into `divergence_by_level`),
    # so `determinism_reanalysis.json` -- a separate, earlier job's output -- is redundant and
    # is deliberately NOT loaded. Loading it was how the frontier table came to print
    # superseded numbers after the sweep was re-run; a second source for a quantity the
    # primary source already has is a stale-data trap, not redundancy.
    fresh = as_cells(report)
    TABLES.mkdir(parents=True, exist_ok=True)

    for name, body in {
        "h8_frontier.tex": frontier_table(fresh),
        "h8_spread.tex": spread_table(fresh),
        "h8_sd10.tex": sd_table(report10),
        "h8_mechanism.tex": mechanism_table(fresh),
        "h8_macros.tex": macros(report, fresh, report10),
        "h8_spot_macros.tex": spotcheck_macros(load("determinism_spotcheck_a10g.json"),
                                               report10),
    }.items():
        (TABLES / name).write_text(body + "\n")
        print(f"  wrote paper/tables/{name}  ({len(body.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
