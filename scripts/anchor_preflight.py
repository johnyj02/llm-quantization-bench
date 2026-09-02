#!/usr/bin/env python3
"""Anchor A's FREE preflight: do the official checkpoints' quantization settings match ours?

    export HF_TOKEN=$(grep '^HF_TOKEN=' ~/workspaces/MedLLMSafety/.env | cut -d= -f2-)
    .venv-harness/bin/python scripts/anchor_preflight.py

⚠️ **`.venv-harness`, not `.venv`.** `huggingface_hub` lives only in the harness environment, the
same reason `scripts/score_probe.py` runs there. Nothing here needs torch or vLLM.

🚨 **Run this BEFORE the generation job, and gate the job behind it** (planner ruling R30). It
downloads three small JSON config files and no weights, so it costs nothing, and it directly tests
the most likely failure mode: **a settings mismatch**. A9's pass criterion exists to catch "our
calibration or serving config is wrong", and a `group_size` or `desc_act` disagreement is exactly
that, detectable here for free rather than after three GPU-hours.

⚠️ **This is not a substitute for the accuracy comparison.** Matching settings do not imply matching
quality: the official checkpoints were calibrated on **Qwen's** data, not our `ultrachat_200k`
512×2048, so equal settings and unequal accuracy is a perfectly possible and informative outcome.
This preflight only rules out the cheap explanation first.

**What a mismatch means, and why it is not automatically our bug.** `EXECUTOR-TASKS.md` §2.5 records
that `hugging-quants/...-GPTQ-INT4` uses `desc_act: true` where Qwen's official uses `false`, and
that mixing them is a silent confound. If an official checkpoint here disagrees with our recipe, the
question is which of the two is the intended comparison target, and that is a registration question
rather than something to patch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

from quantartifact.quantize import load_specs  # noqa: E402

CONFIGS = REPO / "configs"
#: Files that can carry quantization settings, in the order to try. Older GPTQ exports use
#: `quantize_config.json`; newer ones and AWQ put a `quantization_config` block inside `config.json`.
CANDIDATE_FILES = ("quantize_config.json", "quant_config.json", "config.json")

#: The settings the comparison depends on. `EXECUTOR-TASKS.md` §2.5 is explicit that `desc_act`
#: differing between sources is a silent confound, and `group_size` changes the quantization
#: granularity outright, so neither can be left unchecked.
CHECKED_KEYS = ("bits", "group_size", "desc_act", "sym", "quant_method", "version",
                "zero_point")


def _load_any(repo: str, revision: str) -> tuple[dict, str]:
    """First config file that exists, with the quantization block extracted if nested."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from huggingface_hub.errors import EntryNotFoundError  # noqa: PLC0415

    last_error: Exception | None = None
    for name in CANDIDATE_FILES:
        try:
            path = hf_hub_download(repo_id=repo, filename=name, revision=revision)
        except EntryNotFoundError as exc:  # noqa: PERF203 - the loop IS the fallback
            last_error = exc
            continue
        data = json.loads(Path(path).read_text())
        # `config.json` nests it; the dedicated files are the block itself.
        if name == "config.json":
            nested = data.get("quantization_config")
            if nested is None:
                last_error = KeyError(
                    f"{repo}@{revision}: config.json has no `quantization_config` block")
                continue
            return nested, name
        return data, name
    raise SystemExit(
        f"{repo}@{revision}: none of {CANDIDATE_FILES} carried quantization settings. "
        f"Last error: {last_error}. Refusing to guess: an unread setting is an unchecked one."
    )


def main() -> int:
    models_cfg = yaml.safe_load((CONFIGS / "models.yaml").read_text())
    anchor_a = models_cfg["anchor_checkpoints"]["anchor_a"]
    specs, _ = load_specs(CONFIGS / "quant.yaml")
    matched = anchor_a.get("matched_settings", {})

    print("Anchor A preflight: official checkpoint settings vs our recipe")
    print(f"A9's matched_settings: {matched}")
    print()

    disagreements: list[str] = []
    report: dict = {"matched_settings": matched, "checkpoints": {}}

    for arm, official in anchor_a["checkpoints"].items():
        spec = specs[arm]
        block, source = _load_any(official["repo"], official["revision"])
        ours = {
            "bits": spec.bits,
            "group_size": getattr(spec, "group_size", matched.get("group_size")),
            "desc_act": getattr(spec, "desc_act", matched.get("desc_act")),
            "method": spec.method,
        }
        print(f"=== {arm}  ({official['repo']} @ {official['revision'][:8]}, {source})")
        row: dict = {"repo": official["repo"], "revision": official["revision"],
                     "source_file": source, "official": {}, "ours": ours, "mismatches": []}
        for key in CHECKED_KEYS:
            if key not in block:
                continue
            row["official"][key] = block[key]
            print(f"    official {key:12} = {block[key]!r}")

        # The two settings A9 names explicitly. Compared only where the official file states them,
        # because absent is not the same as disagreeing and must not be reported as one.
        for key, expected in (("group_size", matched.get("group_size")),
                              ("desc_act", matched.get("desc_act"))):
            if expected is None or key not in block:
                continue
            if block[key] != expected:
                message = (f"{arm}: official {key}={block[key]!r} but A9's matched_settings "
                           f"require {expected!r}")
                row["mismatches"].append(message)
                disagreements.append(message)
        if block.get("bits") is not None and int(block["bits"]) != int(spec.bits):
            message = (f"{arm}: official bits={block['bits']} but our recipe is {spec.bits}")
            row["mismatches"].append(message)
            disagreements.append(message)

        print(f"    ours          bits={ours['bits']} group_size={ours['group_size']} "
              f"desc_act={ours['desc_act']} method={ours['method']}")
        print(f"    verdict       {'MISMATCH' if row['mismatches'] else 'settings agree'}")
        print()
        report["checkpoints"][arm] = row

    out = REPO / "artifacts" / "anchor_preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"report -> {out.relative_to(REPO)}")

    if disagreements:
        print()
        print("🚨 SETTINGS MISMATCH. Do NOT spend the GPU job yet.")
        for message in disagreements:
            print(f"  - {message}")
        print()
        print("A9's criterion exists to catch 'our calibration or serving config is wrong', and a")
        print("settings disagreement is that, found for free. Which checkpoint is the intended")
        print("comparison target is a REGISTRATION question (see EXECUTOR-TASKS.md §2.5 on")
        print("hugging-quants' desc_act:true vs Qwen official's false), so take it to the planner")
        print("rather than patching the recipe to match.")
        return 1

    print()
    print("✅ Settings agree on every checked key. The cheap explanation for an accuracy gap is")
    print("   ruled out, so the generation job is worth its 3 hours.")
    print("⚠️ Matching settings do NOT imply matching quality: the official checkpoints were")
    print("   calibrated on Qwen's data, not our ultrachat_200k 512x2048, so Anchor A tests")
    print("   COMPARABLE QUALITY rather than reproduction. That framing goes in the paper.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
