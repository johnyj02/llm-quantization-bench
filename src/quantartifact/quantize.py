"""Quantization — GPTQ and AWQ through ONE tool, at ONE normalised calibration budget.

🚨 **The single most likely reviewer objection, so it is the thing this module
exists to get right.**

`llmcompressor`'s own official examples differ by **8× in calibration token
budget** between the two methods:

    GPTQ: 512 samples × 2048 seqlen, symmetric  W4A16
    AWQ:  256 samples ×  512 seqlen, asymmetric W4A16_ASYM

Accepting those defaults would make the GPTQ-vs-AWQ delta **uninterpretable** —
the difference would be method *plus* an 8× calibration difference *plus* a
symmetry difference. We normalise both arms to 512 × 2048, symmetric W4A16, one
pinned calibration set, one pinned seed, and state it in the paper
(`EXECUTOR-TASKS.md` §2.5, `EXECUTOR-PLAN.md` §5).

Every parameter here is hashed into ``quant_config_hash`` and written onto every
generation record, so a reviewer can verify that no arm drifted.

**One tool for both methods** (`llmcompressor`) means one calibration code path,
which removes a confound rather than merely documenting one. Never install
``autoawq`` (downgrades transformers to 4.47.1 and wrecks the env) or
``auto-gptq`` (GitHub-archived; its own README redirects here).

⚠️ **Nothing here runs without explicit user confirmation** (decision D10).
Quantization is billable GPU work. ``main()`` requires ``--i-have-confirmation``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import yaml

QUANTIZE_VERSION = "1.0.0"

# Methods we refuse to import, with the reason, so a future contributor does not
# "helpfully" add them back.
FORBIDDEN_PACKAGES = {
    "autoawq": "downgrades transformers to 4.47.1 and wrecks the environment",
    "auto_gptq": "GitHub-archived; its own README redirects to llmcompressor",
}


class QuantConfigError(ValueError):
    """A quantization configuration violates the normalisation contract."""


@dataclasses.dataclass(frozen=True)
class CalibrationSpec:
    """The calibration budget. IDENTICAL for every method and every bit width."""

    dataset: str = "HuggingFaceH4/ultrachat_200k"
    dataset_split: str = "train_sft"
    num_samples: int = 512          # NOT AWQ's example default of 256
    max_seq_length: int = 2048      # NOT AWQ's example default of 512
    seed: int = 20260823
    shuffle: bool = True

    @property
    def total_tokens(self) -> int:
        """Stated in the paper as a token budget, not a sample count."""
        return self.num_samples * self.max_seq_length

    def to_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "total_calibration_tokens": self.total_tokens}


@dataclasses.dataclass(frozen=True)
class QuantSpec:
    """One precision arm."""

    key: str                        # e.g. "gptq-4bit"
    method: str                     # gptq | awq | bitsandbytes | none
    scheme: str                     # W4A16 | W8A16 | NF4 | none
    bits: int
    symmetric: bool = True
    group_size: int = 128
    desc_act: bool = False          # matches Qwen official; NOT hugging-quants' true
    in_flight: bool = False         # bnb only: quantized by vLLM at load

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def config_hash(spec: QuantSpec, calibration: CalibrationSpec, model_revision: str) -> str:
    """Stable hash over everything that could change the quantized weights.

    Includes ``model_revision`` because the same recipe over a different base
    checkpoint is a different artifact, and a reviewer must be able to tell.
    """
    payload = json.dumps(
        {
            "quantize_version": QUANTIZE_VERSION,
            "spec": spec.to_dict(),
            "calibration": calibration.to_dict(),
            "model_revision": model_revision,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def load_specs(config_path: Path) -> tuple[dict[str, QuantSpec], CalibrationSpec]:
    """Read `configs/quant.yaml` and validate the normalisation contract."""
    config = yaml.safe_load(config_path.read_text())

    calibration_raw = config["calibration"]
    calibration = CalibrationSpec(
        dataset=calibration_raw["dataset"],
        dataset_split=calibration_raw["dataset_split"],
        num_samples=calibration_raw["num_samples"],
        max_seq_length=calibration_raw["max_seq_length"],
        seed=calibration_raw["seed"],
        shuffle=calibration_raw.get("shuffle", True),
    )

    specs: dict[str, QuantSpec] = {}
    for key, raw in config["precisions"].items():
        if raw["method"] == "none":
            continue
        specs[key] = QuantSpec(
            key=key,
            method=raw["method"],
            scheme=raw.get("scheme", "none"),
            bits=raw["bits"],
            symmetric=raw.get("symmetric", True),
            group_size=raw.get("group_size", 128),
            desc_act=raw.get("desc_act", False),
            in_flight=raw.get("in_flight", False),
        )

    validate_normalisation(specs, calibration)
    return specs, calibration


def validate_normalisation(
    specs: dict[str, QuantSpec], calibration: CalibrationSpec
) -> None:
    """Fail loudly if the GPTQ/AWQ contrast would be uninterpretable.

    Checked here rather than trusted, because accepting `llmcompressor`'s own
    example defaults is the *easy* mistake — they are what every tutorial shows —
    and it silently destroys the comparison the paper is built on.
    """
    problems: list[str] = []

    if calibration.num_samples != 512:
        problems.append(
            f"num_samples={calibration.num_samples}, must be 512. AWQ's example "
            f"default is 256, which would make the GPTQ-vs-AWQ delta uninterpretable."
        )
    if calibration.max_seq_length != 2048:
        problems.append(
            f"max_seq_length={calibration.max_seq_length}, must be 2048. AWQ's "
            f"example default is 512 — an 8x token-budget difference."
        )

    # GPTQ and AWQ must agree on symmetry and group size, or the method contrast
    # carries a second difference on top of the algorithm.
    contrast = [spec for spec in specs.values() if spec.method in {"gptq", "awq"} and spec.bits == 4]
    if len(contrast) >= 2:
        symmetries = {spec.symmetric for spec in contrast}
        if len(symmetries) > 1:
            problems.append(
                f"4-bit GPTQ and AWQ disagree on symmetry ({symmetries}). "
                f"llmcompressor's AWQ example is W4A16_ASYM; normalising to "
                f"symmetric W4A16 is a deliberate deviation and must be uniform."
            )
        group_sizes = {spec.group_size for spec in contrast}
        if len(group_sizes) > 1:
            problems.append(f"4-bit arms disagree on group_size ({group_sizes})")

    for spec in specs.values():
        if spec.method == "gptq" and spec.desc_act:
            problems.append(
                f"{spec.key}: desc_act=True conflicts with Qwen's official "
                f"desc_act=False and would be a silent confound against Anchor A"
            )

    if problems:
        raise QuantConfigError(
            "calibration normalisation FAILED:\n  - " + "\n  - ".join(problems)
        )


def assert_no_forbidden_packages() -> None:
    """Refuse to run in an environment containing the archived toolchains."""
    found = [
        f"{name} ({reason})"
        for name, reason in FORBIDDEN_PACKAGES.items()
        if importlib.util.find_spec(name) is not None
    ]
    if found:
        raise QuantConfigError(
            "forbidden package(s) present in this environment: " + "; ".join(found)
        )


# ---------------------------------------------------------------------------
# The recipe
# ---------------------------------------------------------------------------

def build_recipe(spec: QuantSpec):
    """Map a QuantSpec to an `llmcompressor` modifier.

    Imported lazily and only here, so the module is importable (and testable) in
    an environment with no `llmcompressor` — which is every environment except
    ``venv-quant`` on the GPU box.
    """
    if spec.method == "gptq":
        from llmcompressor.modifiers.quantization import (  # noqa: PLC0415
            GPTQModifier,
        )

        return GPTQModifier(
            targets="Linear",
            scheme=spec.scheme,
            ignore=["lm_head"],
            group_size=spec.group_size,
            desc_act=spec.desc_act,
        )

    if spec.method == "awq":
        from llmcompressor.modifiers.awq import AWQModifier  # noqa: PLC0415

        return AWQModifier(
            targets="Linear",
            scheme=spec.scheme,   # normalised to W4A16, NOT the example's W4A16_ASYM
            ignore=["lm_head"],
            group_size=spec.group_size,
        )

    if spec.method == "bitsandbytes":
        raise QuantConfigError(
            "bnb-NF4 is quantized IN-FLIGHT by vLLM at load time; there is no "
            "offline recipe. Set quantization='bitsandbytes' on the engine instead."
        )

    raise QuantConfigError(f"unknown method {spec.method!r}")


def output_dir(root: Path, model_key: str, spec: QuantSpec, digest: str) -> Path:
    """Quantized weights go in a hash-stamped directory.

    The hash is in the path so two different calibration recipes cannot silently
    overwrite each other — which, given the whole point of this module, would be
    the worst possible failure.
    """
    return root / f"{model_key}__{spec.key}__{digest}"


def quantize_model(
    model_repo: str,
    model_revision: str,
    model_key: str,
    spec: QuantSpec,
    calibration: CalibrationSpec,
    out_root: Path,
    *,
    dry_run: bool = False,
) -> Path:
    """Quantize one (model, precision) and write it to disk.

    Cheap: ~10-15 min for a 3B, ~30 min for an 8B, and only ~1 GB VRAM, because
    sequential layer onloading is the default. The ~16 GB requirement is **host**
    RAM, which is why the instance has 64 GiB.
    """
    digest = config_hash(spec, calibration, model_revision)
    destination = output_dir(out_root, model_key, spec, digest)

    if destination.exists() and (destination / "config.json").exists():
        return destination  # idempotent: already produced by this exact recipe

    if dry_run:
        return destination

    assert_no_forbidden_packages()

    from datasets import load_dataset  # noqa: PLC0415
    from llmcompressor import oneshot  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model_repo, revision=model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_repo, revision=model_revision, torch_dtype="auto", device_map="auto"
    )

    dataset = load_dataset(calibration.dataset, split=calibration.dataset_split)
    if calibration.shuffle:
        dataset = dataset.shuffle(seed=calibration.seed)
    dataset = dataset.select(range(calibration.num_samples))

    def preprocess(example: dict) -> dict:
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False
            )
        }

    dataset = dataset.map(preprocess)

    oneshot(
        model=model,
        dataset=dataset,
        recipe=build_recipe(spec),
        max_seq_length=calibration.max_seq_length,
        num_calibration_samples=calibration.num_samples,
        output_dir=str(destination),
    )
    tokenizer.save_pretrained(destination)

    # Provenance travels with the weights, not just in the repo.
    (destination / "quantartifact_provenance.json").write_text(
        json.dumps(
            {
                "quant_config_hash": digest,
                "model_repo": model_repo,
                "model_revision": model_revision,
                "spec": spec.to_dict(),
                "calibration": calibration.to_dict(),
                "quantize_version": QUANTIZE_VERSION,
            },
            indent=2,
        )
    )
    return destination


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qa quantize", description="Quantize one or more (model, precision) arms."
    )
    parser.add_argument("--model", required=True, help="model key from configs/models.yaml")
    parser.add_argument("--precision", action="append", default=None, help="repeatable")
    parser.add_argument("--out-root", type=Path, default=Path("weights"))
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="check the normalisation contract and print the plan. Zero cost.",
    )
    parser.add_argument(
        "--i-have-confirmation",
        action="store_true",
        help="the user has explicitly confirmed THIS run (decision D10). Required.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    specs, calibration = load_specs(args.configs / "quant.yaml")
    models = yaml.safe_load((args.configs / "models.yaml").read_text())["models"]

    if args.model not in models:
        sys.stderr.write(f"unknown model {args.model!r}; have {sorted(models)}\n")
        return 2

    model = models[args.model]
    selected = args.precision or [
        key for key, spec in specs.items() if not spec.in_flight
    ]

    print(f"model       : {model['repo']} @ {model['revision'][:12]}")
    print(f"calibration : {calibration.num_samples} x {calibration.max_seq_length} "
          f"= {calibration.total_tokens:,} tokens, seed {calibration.seed}")
    print(f"dataset     : {calibration.dataset} [{calibration.dataset_split}]")
    print("normalisation contract: PASSED")
    print()
    for key in selected:
        if key not in specs:
            sys.stderr.write(f"unknown precision {key!r}; have {sorted(specs)}\n")
            return 2
        spec = specs[key]
        digest = config_hash(spec, calibration, model["revision"])
        destination = output_dir(args.out_root, args.model, spec, digest)
        print(f"  {key:14} {spec.method:5} {spec.scheme:8} sym={spec.symmetric} "
              f"g={spec.group_size} hash={digest} -> {destination}")

    if args.validate_only:
        print("\n--validate-only: nothing was run.")
        return 0

    if not args.i_have_confirmation:
        sys.stderr.write(
            "\nREFUSING TO RUN.\n\n"
            "Decision D10 (notes/decisions.md): quantization is billable GPU work and\n"
            "does not start without explicit user confirmation.\n\n"
            "Use --validate-only to check the configuration at zero cost.\n"
        )
        return 2

    for key in selected:
        spec = specs[key]
        destination = quantize_model(
            model["repo"], model["revision"], args.model, spec, calibration, args.out_root
        )
        print(f"done: {key} -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
