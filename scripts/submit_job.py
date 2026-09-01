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
import shutil
import subprocess
import sys
import tempfile
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
        # DIFFERENT IMAGE. llmcompressor pins compressed-tensors==0.17.1 exactly
        # and transformers<=5.10.1, which conflicts with vllm 0.26.0 -- one image
        # cannot host both (EXECUTOR-TASKS.md 3.2's two-venv split, as two images).
        "image_key": "quant",
        # This stage WRITES weights into the checkpoint tree, so a rehearsal must
        # not share a prefix with the study.
        "smoke_isolates_checkpoints": True,
    },
    "pilot": {
        "entry_point": "sm_pilot.py",
        "description": "A8 pilot-v1: 17 cells x 300 items = 5,100 stored generations",
        # ~2-3 h expected (5,100 generations plus 17 engine loads). Generous is free
        # (R13.4) but a cap bounds a hang: 24 h is ~$67.
        "max_run": 86400,
        # READS job 1's checkpoints, so the smoke must see the study prefix.
        # Isolation of its own writes is by run_id (`pilot-v1-smoke`), not by prefix.
        "smoke_isolates_checkpoints": False,
    },
    "diagnose": {
        "entry_point": "sm_diagnose.py",
        "description": "ONE cheap job, TWO open questions: was the old sampling config "
                       "neutral, and does cross-invocation divergence survive the fix?",
        # 4 probes, 6 invocations, 40 items each. Minutes. 1 h caps a hang.
        "max_run": 3600,
        # READS job 1's checkpoints AND the stored pilot-v2 baseline it compares against,
        # so it must see the study prefix. It writes only under `diagnose-v1__*` run_ids.
        "smoke_isolates_checkpoints": False,
    },
    "reference": {
        "entry_point": "sm_reference.py",
        # ⚠️ Keep this in step with sm_reference.K2_CELLS. --plan-only is what D10's
        # confirmation is given AGAINST, so an understated figure here means the user approves
        # a run that is not the run. This said "63,570 gens / ~10.5 h" after K=2 added ten
        # doubled cells -- a 67% understatement.
        "description": ("THE MAIN RUN, run_id=main-v1: 15 cells x 4,238 items = 63,570 gens, "
                        "PLUS 10 K=2 cells run twice (prereg §9f) = ~105,950 total"),
        # ~17.5 h projected (scripts/project_budget.py, from pilot-v2's measured rates).
        # 48 h because a cap costs nothing unless it fires and a timeout costs a queue cycle
        # on a 1-slot shared account; per-item resume means a restart loses no work.
        "max_run": 172800,
        # READS job 1's checkpoints, so the smoke must see the study prefix. Its own outputs
        # are isolated by run_id (`main-v1-smoke`), never by prefix.
        "smoke_isolates_checkpoints": False,
    },
    "template": {
        "entry_point": "sm_template.py",
        "description": "prereg 3b-bis: the prompt-template FACTOR, both levels, 18 cells",
        # ~3.8 h projected (scripts/project_budget.py). 12 h caps a hang at ~$34 while
        # leaving generous room -- a cap costs nothing unless it fires (R13.4).
        "max_run": 43200,
        # READS job 1's checkpoints, so the smoke must see the study prefix. Its own
        # outputs are isolated by run_id (`template-v1-smoke`), not by prefix.
        "smoke_isolates_checkpoints": False,
    },
    "determinism": {
        "entry_point": "sm_determinism.py",
        "description": "A7/H8: per-format bitwise disagreement + max_num_seqs frontier",
        # 48 h, per planner ruling R13.4. ⚠️ A GENEROUS CAP IS FREE — SageMaker bills
        # time USED, not the cap, so there is no reason to run this close to the wire.
        # 48 h is generous enough that a timeout is unlikely at any plausible
        # generation rate, and short enough that a HANG caps exposure at ~$134 rather
        # than the ~$336 a 5-day cap (SageMaker's maximum) would allow. Per-cell
        # checkpointing already makes a timeout cost a queue wait rather than work.
        "max_run": 172800,
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
    """D11 naming: ``johnyj02-QA-v<x-x>-<date>-<uuid>-<stage>``.

    ⚠️ **Dots become dashes.** SageMaker training-job names must match
    ``[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}`` — no dots allowed — so ``v0.1`` is
    rendered ``v0-1``. Submitting with the dot fails with a ValidationException at
    ``CreateTrainingJob``. S3 prefixes keep the dotted form, which is legal there.

    Also bounded at 63 characters by that pattern; the current form is ~40.
    """
    date = datetime.now(UTC).strftime("%Y%m%d")
    version = qa_version().replace(".", "-")
    name = f"johnyj02-QA-{version}-{date}-{run_uuid()}-{stage}"
    if len(name) > 63:
        raise SystemExit(f"job name too long for SageMaker ({len(name)} > 63): {name}")
    return name


def unique_job_name(base: str, region: str, profile: str) -> str:
    """Append ``-r<N>`` if ``base`` was already used. Requires live credentials.

    ⚠️ **SageMaker training-job names must be unique for the lifetime of the
    account**, and ours are deterministic: the uuid comes from ``.box-run-id`` and
    is reused ON PURPOSE so a stage's jobs correlate across the study. So the
    FIRST retry of any stage on the same day collides and is rejected at
    ``CreateTrainingJob`` — which, with one queue slot, is a slot we waited for
    and then wasted.

    Regenerating the uuid instead would fix uniqueness and lose correlatability,
    so we keep the base name and count attempts. ``-r2`` also reads as what it is
    in the console, which a random suffix would not (D11: a human scanning
    resources should be able to review these names).
    """
    import boto3  # noqa: PLC0415 - only needed on the submit path

    client = boto3.Session(profile_name=profile, region_name=region).client("sagemaker")

    from botocore.exceptions import ClientError  # noqa: PLC0415 - submit path only

    def exists(name: str) -> bool:
        # ⚠️ `describe_training_job` (an O(1) direct lookup), NOT
        # `list_training_jobs(NameContains=…)`. The latter paginates the whole
        # account history -- and this account is SHARED, so that history is long
        # enough that the call did not return within two minutes.
        #
        # ⚠️ A missing job raises **ValidationException**, not ResourceNotFound.
        # Catching `client.exceptions.ResourceNotFound` looks right and never
        # fires. Match on the message, and re-raise anything else rather than
        # reporting a permissions or throttling error as "name is free".
        try:
            client.describe_training_job(TrainingJobName=name)
            return True
        except ClientError as exc:
            if "not found" in str(exc).lower():
                return False
            raise

    if not exists(base):
        return base
    for attempt in range(2, 100):
        candidate = f"{base}-r{attempt}"
        # The 63-char bound is re-checked here because the suffix is appended
        # AFTER job_name()'s check, so that guard cannot see it.
        if len(candidate) > 63:
            raise SystemExit(
                f"retry suffix pushes the job name past SageMaker's 63-char limit "
                f"({len(candidate)}): {candidate}"
            )
        if not exists(candidate):
            print(f"job name     : {base} already used -> {candidate}")
            return candidate
    raise SystemExit(f"{base}: 99 attempts already used; something is wrong")


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


# Only these land in the uploaded tarball. Everything else is excluded.
#
# ⚠️ The SageMaker SDK tars **all** of `source_dir` and honours no ignore file.
# Pointed at the repo root that is 575 MB, of which 568 MB is `.venv-harness` +
# `.venv` — uploaded on EVERY submission, and useless in the container, which has
# its own Python. Worse than wasteful: those are macOS **arm64** wheels, so they
# are actively wrong for a linux/amd64 container.
STAGE_INCLUDE_DIRS = ("src", "configs", "scripts", "vendor")
STAGE_INCLUDE_GLOBS = ("sm_*.py", "pyproject.toml")


def stage_source(stage: str) -> Path:
    """Copy only what the job needs into a temp dir, and return that as source_dir.

    Deliberately an allowlist, not a denylist: a new large directory in the repo
    should not silently start being uploaded.
    """
    staged = Path(tempfile.mkdtemp(prefix=f"qa-src-{stage}-"))

    for name in STAGE_INCLUDE_DIRS:
        source = REPO / name
        if source.is_dir():
            shutil.copytree(
                source, staged / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
            )
    for pattern in STAGE_INCLUDE_GLOBS:
        for path in REPO.glob(pattern):
            if path.is_file():
                shutil.copy2(path, staged / path.name)

    size_mb = sum(f.stat().st_size for f in staged.rglob("*") if f.is_file()) / 1e6
    print(f"staged       : {staged}  ({size_mb:.1f} MB, "
          f"{sum(1 for _ in staged.rglob('*') if _.is_file())} files)")
    return staged


def build_config(stage: str, overrides: dict) -> dict:
    config = yaml.safe_load((REPO / "configs" / "sm_job.yaml").read_text())
    spec = STAGES[stage]

    config["entry_point"] = spec["entry_point"]
    config["source_dir"] = str(REPO)  # replaced by stage_source() at submit time
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

    # Per-stage image. The quantize stage needs llmcompressor, which cannot
    # coexist with vllm in one image; everything else needs the serving image.
    image_key = spec.get("image_key", "serve")
    images = config.pop("images", {}) or {}
    if image_key in images:
        config["image_uri"] = images[image_key]

    config["hyperparameters"] = {"stage": stage, **overrides}
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit a QuantArtifact job via SMInterface")
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--plan-only", action="store_true", help="print the job spec, submit nothing")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="20 items, short max_run, same code path. MANDATORY rehearsal before "
             "any real submission: with one job slot and a queue, a broken 20-hour "
             "job is the most expensive mistake available.",
    )
    parser.add_argument("--wait", action="store_true", help="block and stream logs")
    parser.add_argument(
        "--orderings",
        action="store_true",
        help="R17.3 / prereg 9a: 10 orderings at the frozen batch size, 4 arms, so the "
             "ordering spread becomes an SD rather than a range over three draws. Range "
             "is unstable and grows with n, so it cannot carry a population claim.",
    )
    parser.add_argument(
        "--replicate",
        action="store_true",
        help="JOB 1b (R18.3): re-quantize ONE arm at a bit-identical recipe into a "
             "'-rep2' path, so the two can be byte-compared. Resolves whether AWQ is "
             "inherently nondeterministic or whether `offload_device` changed the weights "
             "-- two different main-text claims whose ambiguity cannot be published.",
    )
    parser.add_argument(
        "--reanalyze",
        action="store_true",
        help="recompute a stage's ANALYSIS from stored raw outputs, generating nothing. "
             "Cells are cached by tag, so a new analysis field would otherwise need the "
             "whole run repeated to surface it.",
    )
    parser.add_argument(
        "--i-have-confirmation",
        action="store_true",
        help="the user has explicitly confirmed THIS submission (D10). Required.",
    )
    args = parser.parse_args()

    overrides: dict = {}
    if args.smoke:
        overrides["smoke"] = 1
    if args.reanalyze:
        overrides["reanalyze"] = 1
    if args.replicate:
        overrides["replicate"] = 1
    if args.orderings:
        overrides["orderings"] = 1
    config = build_config(args.stage, overrides)
    spec = STAGES[args.stage]
    if args.reanalyze:
        # No generation and no engine, so it is minutes. A short leash bounds a hang.
        config["max_run"] = min(config["max_run"], 3600)
        config["job_name"] = config["job_name"] + "-reanalyze"

    if args.orderings:
        if args.stage != "determinism":
            raise SystemExit("--orderings applies to --stage determinism only")
        # 4 arms x 500 items x 11 passes = 22,000 generations. At the measured mixed rate
        # (~6,000 gen/h on this probe's composition) that is ~4 h; 12 h caps a hang.
        config["max_run"] = min(config["max_run"], 43200)
        config["job_name"] = config["job_name"] + "-orderings"

    if args.replicate:
        if args.stage != "quantize":
            raise SystemExit("--replicate applies to --stage quantize only")
        # One arm, ~30 min observed for qwen AWQ with offload. 2 h caps a hang.
        config["max_run"] = min(config["max_run"], 7200)
        config["job_name"] = config["job_name"] + "-replicate"
        # ⚠️ The study prefix, NOT the smoke prefix: the replicate must sit beside the
        # checkpoint it is compared against, and it is a real artifact rather than junk.
        # Its distinct `-rep2` directory is what keeps it from being mistaken for the study
        # checkpoint, and the recipe digest stays in the name as proof of identity.

    if args.smoke:
        # Short leash and a distinct name, so a smoke run can never be
        # mistaken for the real thing in the job list or in S3.
        config["max_run"] = min(config["max_run"], 1800)
        config["job_name"] = config["job_name"] + "-smoke"
        if config.get("use_spot_instances"):
            config["max_wait"] = config["max_run"]
        # SEPARATE checkpoint prefix, but ONLY for stages that WRITE weights there.
        #
        # Smoke artefacts cannot be *used* by a real job (the recipe hash differs, so
        # they land on a different path), but SageMaker restores the whole
        # `checkpoint_s3_uri` prefix into the container at every job start -- so junk
        # left here would be re-downloaded by every subsequent job forever. And in a
        # study whose central claim is a normalised calibration budget, a
        # deliberately-wrong-budget checkpoint sitting in the study's own tree is
        # exactly the artefact not to leave for a reviewer to trip over.
        #
        # ⚠️ BUT for a stage that READS those weights -- `determinism` serves the
        # quantized checkpoints job 1 produced -- redirecting the prefix means the
        # rehearsal cannot see them at all and dies on a missing checkpoint. Such a
        # stage isolates its own outputs internally instead (`sm_determinism.py`
        # writes to a `determinism-smoke/` subdirectory), which is the isolation that
        # actually matters: a smoke result must never be readable as a real one.
        if spec.get("smoke_isolates_checkpoints") and config.get("checkpoint_s3_uri"):
            config["checkpoint_s3_uri"] = (
                config["checkpoint_s3_uri"].rstrip("/") + "-smoke/"
            )

    print(f"stage        : {args.stage} — {spec['description']}")
    print(f"job name     : {config['job_name']}"
          f"  (+ '-r<N>' if already used — names must be unique per account)")
    print(f"instance     : {config['instance_type']} x{config['instance_count']}  "
          f"(1 x L40S 48GB, SM89)")
    print(f"region       : {config['region']}   profile: {config['profile']}")
    print(f"role         : {config['role'].split('/')[-1]}")
    print(f"max_run      : {config['max_run']}s   volume: {config['volume_size']}GB   "
          f"spot: {config['use_spot_instances']}")
    print(f"checkpoints  : {config.get('checkpoint_s3_uri') or 'NONE — a long job would lose everything on timeout'}")
    if args.smoke:
        print("mode         : SMOKE (20 items, short max_run, name suffixed -smoke)")
    print(f"output       : {config['output_path']}")
    print(f"code upload  : {config['output_path'].rsplit('/johnyj02',1)[0]}"
          f"/{config['job_name']}/source/sourcedir.tar.gz")
    print("               -> extracted to /opt/ml/code in the container")
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

    # Stage a minimal source tree so the upload is small and linux-correct.
    config["source_dir"] = str(stage_source(args.stage))

    if str(SMINTERFACE_SRC) not in sys.path:
        sys.path.insert(0, str(SMINTERFACE_SRC))

    refresh_conduit_credentials(config["profile"])

    # After the refresh, because it needs live credentials. Before submit, because
    # a duplicate name is rejected at CreateTrainingJob.
    config["job_name"] = unique_job_name(
        config["job_name"], config["region"], config["profile"]
    )

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
