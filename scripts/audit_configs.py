#!/usr/bin/env python3
"""Find config keys that no code path consumes. Planner ruling R14.3.

    python scripts/audit_configs.py            # report
    python scripts/audit_configs.py --strict    # exit 1 if any unexplained orphan

**Why this exists.** Three times now a value declared in `configs/` was silently
overridden by a library default or never forwarded at all — `desc_act`,
`async_scheduling` — and each would have been *reported in the paper as though it
held*. R14.3's instruction: *"for every key in models.yaml, quant.yaml, tasks.yaml,
scoring.yaml, confirm a code path consumes it. Any orphan is decoration."*

This is a **heuristic**, and deliberately a loud one. It greps for the key name
across `src/`, `sm_*.py`, `scripts/` and `analysis/`. A key can be consumed
indirectly (spread into a dataclass, iterated over) and still be flagged, so
`DOCUMENTED_ORPHANS` records the ones that are intentionally descriptive — each with
a reason. That list is the point: an orphan is fine if someone decided it is, and a
problem if nobody did.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG_FILES = ("models.yaml", "quant.yaml", "tasks.yaml", "scoring.yaml")
SEARCH_ROOTS = ("src", "scripts", "analysis")
SEARCH_GLOBS = ("sm_*.py",)

# Keys that are documentation rather than inputs. Each needs a REASON, because
# "nothing reads it" is only acceptable as a decision, never as an accident.
DOCUMENTED_ORPHANS: dict[str, str] = {
    # Provenance and prose, read by humans and by the paper, not by code.
    "note": "free-text annotation",
    "notes": "free-text annotation",
    "license_note": "reported in the artifact release (A24), not consumed",
    "normalisation_note": "states the AWQ symmetry deviation in the paper",
    "exclusion_reason": "the paper must give the reason, not just the exclusion",
    "backend_confound": "confound to STATE; no code branches on it",
    "determinism_audit_supersedes": "records which estimate this replaces",
    "supersedes": "records which estimate this replaces",
    "access_verified": "date access was confirmed; notes/access.md is the record",
    "params_b": "reported in the paper's model table",
    "bf16_size_gb": "capacity planning by a human",
    "has_official_quantized_checkpoints": "explains why an anchor exists",
    "published": "external published numbers, compared by hand at analysis time",
    "published_accuracies": "external published numbers for the anchor comparison",
    "source": "citation for an anchor",
    "reported_f1": "cited judge accuracy, reported not consumed",
    "family_clean": "states the judge is not from a model family under test",
    "redistributable": "licence constraint enforced by a human (A24), not code",
    "gated_on": "records the gate a stretch arm waits on",
    "expected_minutes_3b": "planning estimate",
    "expected_minutes_8b": "planning estimate",
    "vram_gb_required": "planning estimate",
    "host_ram_gb_required": "planning estimate why the instance has 64 GiB",
    "bi_guard_in_vllm": "the H8 mechanism, asserted against vLLM source by hand",
    "cuda13_risk": "records that the open risk closed",
    "forbidden": "enforced by name in quantize.FORBIDDEN_PACKAGES, not read from yaml",
    "pins": "documents the exact pins; the Dockerfile enforces them",
    "package": "names the toolchain; the image provides it",
    "git_tag": "records the harness tag; the image pins it",
    "strict_regex_case_sensitive": "documents harness behaviour we reproduce",
    "strict_regex_drops_final_char": "documents the BBH regex bug we must reproduce",
    "fewshot_source": "documents whether exemplars are sampled or hardcoded",
    "doc_to_text": "documents the harness template; lm-eval owns it",
    "doc_to_target": "documents the harness template; lm-eval owns it",
    "gradient_role": "records why a BBH subtask is in the design",
    "format_class": "records the answer-format class for stratification by hand",
    "n_items_reference": "planning counts; the entry points set their own limits",
    "n_items_sweep": "planning counts",
    "n_items_default": "planning counts",
    "revised_total_estimate": "study-wide budget for the schedule, not an input",
    "determinism_audit_best": "registered budget (R13.7)",
    "determinism_audit_worst": "registered budget (R13.7)",
    "reference_generative_per_cell": "planning counts",
    "sweep_per_cell": "planning counts",
    "zeroshot_per_cell": "planning counts",
    "task_variant": "records which GSM8K variant an anchor used",
    "arc_arm": "records which of ARC's three arms this is",
    "substitution_armed": "records that the Llama substitution is VOID",
    "matched_settings": "records the settings an anchor is matched on",
    "pass_criterion": "the gate's threshold, checked at analysis time",
    "blocking": "records whether an anchor gates progress",
    "total_calibration_tokens": "the CEILING; the realised count is measured instead",
    "compute_capability": "asserted by sm_verify against nvidia-smi, not read here",
    "gpu": "asserted by sm_verify against nvidia-smi",
    "gpu_count": "asserted by sm_verify",
    "vcpu": "hardware fact, reported",
    "host_ram_gib": "hardware fact, reported",
    "instance_type": "set in configs/sm_job.yaml, which IS consumed",
    "gpu_memory_mib": "measured hardware fact, reported in the paper",
    "cuda": "measured, reported in the paper",
    "driver": "measured, reported in the paper",
    "torch": "measured, reported in the paper",
    "bi_supported_on_sm89": "an A7 FINDING slot, written by job 2",
    "max_num_seqs_resolved_by_audit": "consumed by sm_pilot.frozen_max_num_seqs",
}


def leaf_keys(node, path: tuple[str, ...] = ()) -> list[tuple[str, tuple[str, ...]]]:
    """Every mapping key in the tree, with the path that reached it."""
    found: list[tuple[str, tuple[str, ...]]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str):
                found.append((key, path))
            found.extend(leaf_keys(value, path + (str(key),)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(leaf_keys(value, path + (str(index),)))
    return found


def code_corpus() -> str:
    parts: list[str] = []
    for root in SEARCH_ROOTS:
        base = REPO / root
        if base.is_dir():
            for path in base.rglob("*.py"):
                parts.append(path.read_text(errors="ignore"))
            for path in base.rglob("*.md"):
                parts.append(path.read_text(errors="ignore"))
    for pattern in SEARCH_GLOBS:
        for path in REPO.glob(pattern):
            parts.append(path.read_text(errors="ignore"))
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 if any orphan is undocumented")
    args = parser.parse_args()

    corpus = code_corpus()
    undocumented: list[tuple[str, str, tuple[str, ...]]] = []
    documented = 0
    consumed = 0

    for filename in CONFIG_FILES:
        path = REPO / "configs" / filename
        if not path.is_file():
            print(f"  (missing: {filename})")
            continue
        data = yaml.safe_load(path.read_text())
        seen: set[str] = set()
        for key, where in leaf_keys(data):
            if key in seen:
                continue
            seen.add(key)
            # A key counts as consumed if its name appears in code as a string
            # literal or attribute. Deliberately generous: the goal is to find
            # keys NOTHING mentions.
            if re.search(rf'["\']{re.escape(key)}["\']|\.{re.escape(key)}\b', corpus):
                consumed += 1
            elif key in DOCUMENTED_ORPHANS:
                documented += 1
            else:
                undocumented.append((filename, key, where))

    print(f"consumed by code      : {consumed}")
    print(f"documented as prose   : {documented}")
    print(f"UNDOCUMENTED ORPHANS  : {len(undocumented)}")
    if undocumented:
        print()
        print("Each of these is either (a) a value we rely on that nothing enforces —")
        print("the async_scheduling failure — or (b) prose that belongs in")
        print("DOCUMENTED_ORPHANS with a reason. Decide which; do not leave it blank.")
        print()
        for filename, key, where in undocumented:
            location = ".".join(where) or "(top level)"
            print(f"  {filename:14} {key:38} at {location}")
    return 1 if (undocumented and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
