#!/usr/bin/env python3
"""Submit a job via SMInterface. THE compute path (decision D14).

    python scripts/submit_job.py --stage verify --plan-only
    python scripts/submit_job.py --stage verify --i-have-confirmation

D14 overrides `EXECUTOR-TASKS.md` O4: SMInterface is the only substrate, EC2 is
withdrawn. The operative reason is that **SageMaker training jobs queue for
capacity** while EC2 `RunInstances` fails immediately — and `g6e.2xlarge` was
unavailable in 10 AZs across three regions.

This wrapper exists rather than calling SMInterface directly because three things
have to happen that SMInterface does not do for us:

1. **Conduit credentials.** `session.refresh_credentials()` hardcodes
   ``ada credentials update --profile=<p> --once`` with no ``--provider conduit``,
   so its auto-refresh cannot work on this account. We refresh correctly first and
   set ``auto_refresh_credentials: false``.
2. **The D10 confirmation gate.** No billable job starts without explicit user
   confirmation.
3. **Job naming** per D11: ``johnyj02-QA-v<x.x>-<date>-<uuid>-<stage>``, with
   "QA" as project initials that are never expanded.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SMINTERFACE_SRC = Path.home() / "workspaces" / "SMInterface" / "src" / "SMInterface" / "src"

# Stages, each a separate queued job. Kept explicit rather than free-form so a
# typo cannot silently submit something expensive.
STAGES: dict[str, dict] = {
    "verify": {
        "entry_point": "sm_verify.py",
        "description": "box_verify equivalent: GPU, compute cap, C compiler, Triton JIT, CUDA linkage",
        "max_run": 1800,
    },
    "throughput": {
        "entry_point": "sm_throughput.py",
        "description": "A6: measured generations/hour WITH and WITHOUT VLLM_BATCH_INVARIANT",
        "max_run": 7200,
    },
    "quantize": {
        "entry_point": "sm_quantize.py",
        "description": "GPTQ + AWQ at the normalised 512x2048 calibration budget",
        "max_run": 21600,
    },
    "determinism": {
        "entry_point": "sm_determinism.py",
        "description": "A7/H8: per-format bitwise disagreement + max_num_seqs frontier",
        "max_run": 86400,
    },
}


def qa_version() -> str:
    text = (REPO / "src" / "quantartifact" / "__init__.py").read_text()
    for line in text.splitlines():
        if line.startswith("__version__"):
            parts = line.split('"')[1].split(".")
            return f"v{parts[0]}.{parts[1]}"
    return "v0.1"


def run_uuid() -> str:
    """Stable per-box identity, shared with the EC2 scripts' .box-run-id.

    Reused rather than regenerated so job names stay correlatable across stages.
    """
    path = REPO / ".box-run-id"
    if path.exists():
        return path.read_text().strip()
    value = str(uuid.uuid4())[:8]
    path.write_text(value + "\n")
    return value


def job_name(stage: str) -> str:
    """D11 naming. "QA" is initials only and is never expanded."""
    date = datetime.now(UTC).strftime("%Y%m%d")
    return f"johnyj02-QA-{qa_version()}-{date}-{run_uuid()}-{stage}"


def refresh_conduit_credentials(profile: str) -> None:
    """Refresh Conduit credentials, which SMInterface cannot do itself.

    Its ``refresh_credentials`` omits ``--provider conduit``, and ada no longer
    falls back to the role name for Conduit, so both flags are required or the
    call fails with a misleading "Account not found".
    """
    command = [
        "ada", "credentials", "update",
        "--provider", "conduit",
        "--account", "654654594225",
        "--role", "IibsAdminAccess-DO-NOT-DELETE",
        "--profile", profile,
        "--once",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise SystemExit(
            "credential refresh FAILED:\n"
            + (result.stderr or result.stdout)
            + "\nrun manually:\n  " + " ".join(command)
        )


def build_config(stage: str, overrides: dict) -> dict:
    config = yaml.safe_load((REPO / "configs" / "sm_job.yaml").read_text())
    spec = STAGES[stage]

    config["entry_point"] = spec["entry_point"]
    config["source_dir"] = str(REPO)
    config["job_name"] = job_name(stage)
    # Per-stage max_run, but never below the file's floor: a stage that needs
    # longer must say so, and a stage that needs less should not hold a slot.
    config["max_run"] = spec["max_run"]
    if config.get("use_spot_instances"):
        config["max_wait"] = config["max_run"]

    # HF_TOKEN travels in the environment, never in a committed file.
    token = os.environ.get("HF_TOKEN")
    if token:
        config.setdefault("environment", {})["HF_TOKEN"] = token

    config["hyperparameters"] = {"stage": stage, **overrides}
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit a QuantArtifact job via SMInterface")
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--plan-only", action="store_true", help="print the job spec, submit nothing")
    parser.add_argument("--wait", action="store_true", help="block and stream logs")
    parser.add_argument(
        "--i-have-confirmation",
        action="store_true",
        help="the user has explicitly confirmed THIS submission (D10). Required.",
    )
    args = parser.parse_args()

    config = build_config(args.stage, {})
    spec = STAGES[args.stage]

    print(f"stage        : {args.stage} — {spec['description']}")
    print(f"job name     : {config['job_name']}")
    print(f"instance     : {config['instance_type']} x{config['instance_count']}  "
          f"(1 x L40S 48GB, SM89)")
    print(f"region       : {config['region']}   profile: {config['profile']}")
    print(f"role         : {config['role'].split('/')[-1]}")
    print(f"max_run      : {config['max_run']}s   volume: {config['volume_size']}GB   "
          f"spot: {config['use_spot_instances']}")
    print(f"output       : {config['output_path']}")
    print(f"image_uri    : {config.get('image_uri') or 'NOT SET'}")
    print(f"env          : {sorted(k for k in config.get('environment', {}) if k != 'HF_TOKEN')}")
    print(f"HF_TOKEN     : {'present' if config.get('environment', {}).get('HF_TOKEN') else 'ABSENT'}")

    problems = []
    if not config.get("image_uri"):
        problems.append(
            "image_uri is not set. vLLM pins torch exactly and its precompiled "
            "_C.abi3.so links a specific libcudart, so the stock PyTorch DLC will "
            "not work (notes/reuse.md §5). Build and push the image first."
        )
    if config.get("distributed"):
        problems.append("distributed=true — TP>1 breaks batch invariance (vLLM #51290)")
    if not (REPO / config["entry_point"]).is_file():
        problems.append(f"entry_point {config['entry_point']} does not exist yet")

    if problems:
        print("\nBLOCKED:")
        for problem in problems:
            print(f"  - {problem}")

    if args.plan_only:
        print("\n--plan-only: nothing submitted.")
        return 0

    if problems:
        return 1

    if not args.i_have_confirmation:
        sys.stderr.write(
            "\nREFUSING TO SUBMIT.\n\n"
            "Decision D10: no billable job starts without explicit user confirmation.\n"
            "Use --plan-only to inspect at zero cost.\n"
        )
        return 2

    if str(SMINTERFACE_SRC) not in sys.path:
        sys.path.insert(0, str(SMINTERFACE_SRC))

    refresh_conduit_credentials(config["profile"])

    # Lazy: SMInterface is only importable once its src is on sys.path above.
    from sm_interface.config import JobConfig  # noqa: PLC0415
    from sm_interface.runner import submit  # noqa: PLC0415

    handle = submit(JobConfig.from_dict(config), wait=args.wait)
    print(json.dumps({"job_name": config["job_name"], "submitted": True}, indent=2))
    if not args.wait:
        print(f"\nSageMaker QUEUES for capacity — this will start when a "
              f"{config['instance_type']} frees up.\nCheck with: "
              f"sm-interface status {config['job_name']}")
    return 0 if handle else 1


if __name__ == "__main__":
    raise SystemExit(main())
