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
    # AWQ only. `llmcompressor`'s documented remedy for OOM during the scale
    # search: hold the cached forward args on this device instead of the GPU.
    #
    # ⚠️ IN THE HASH ON PURPOSE, even though it is a resource setting rather than an
    # algorithm one. It moves a cache; it does not change the math, the dtype, or
    # the grid, so the weights *should* be bitwise identical either way. But
    # "should" is not good enough for the one claim this paper rests on — that every
    # arm was produced identically — and the cost of being conservative is one
    # re-run of two cheap arms. Hashing it also makes an accidental mix of
    # offloaded and non-offloaded arms impossible rather than merely unlikely,
    # because they land on different paths.
    #
    # Contrast with `n_grid`, which llmcompressor documents as trading scale
    # QUALITY for time. That one would be a genuine recipe change and is left alone.
    offload_device: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def config_hash(spec: QuantSpec, calibration: CalibrationSpec, model_revision: str) -> str:
    """Stable hash over everything that could change the quantized weights.

    Includes ``model_revision`` because the same recipe over a different base
    checkpoint is a different artifact, and a reviewer must be able to tell.

    ⚠️ **Unset (``None``) spec fields are EXCLUDED**, and that is what makes this
    hash extensible. Adding an optional field otherwise changes the digest of every
    arm that does not use it, purely because ``to_dict()`` gained a key: adding
    AWQ's ``offload_device`` invalidated all six GPTQ checkpoints, which an AWQ-only
    setting cannot possibly affect. That is a re-run of hours of GPU time for a
    field whose value is "not in use".

    So an option in use changes the hash (correct -- it may change the weights) and
    an option left unset does not (correct -- it changed nothing). No pre-existing
    field defaults to ``None``, so this preserves every digest already on disk.
    """
    payload = json.dumps(
        {
            "quantize_version": QUANTIZE_VERSION,
            "spec": {k: v for k, v in spec.to_dict().items() if v is not None},
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
            offload_device=raw.get("offload_device"),
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

def assert_preset_matches_spec(spec: QuantSpec) -> dict[str, Any]:
    """Resolve the named preset scheme and assert it equals what `quant.yaml` says.

    🚨 **This function is why `group_size` is trustworthy.** In
    `llmcompressor` 0.12.0.1 neither modifier accepts `group_size` as a keyword;
    it lives inside the preset scheme (`W4A16` → `num_bits=4, type=int,
    strategy=group, group_size=128, symmetric=True`). So the only honest options
    are to hand-build `config_groups` or to use the preset **and check it**.

    We check it. The study's central claim is a normalised quantization
    configuration, and `quant_config_hash` hashes the values in `quant.yaml` — so
    if a future `compressed-tensors` changed what `W4A16` means, the weights would
    change while the hash did not. That is the one failure this project cannot
    tolerate, and it is silent by construction. Hence an assertion rather than a
    comment.

    Returns the resolved weight args for the record.
    """
    from compressed_tensors.quantization import (  # noqa: PLC0415 - box-only dependency
        preset_name_to_scheme,
    )

    scheme = preset_name_to_scheme(spec.scheme, ["Linear"])
    weights = scheme.weights
    if weights is None:
        raise QuantConfigError(f"preset {spec.scheme!r} declares no weight quantization")

    # `use_enum_values=True` on QuantizationScheme means these may be plain
    # strings or enum members depending on construction path; compare on value.
    strategy = getattr(weights.strategy, "value", weights.strategy)
    resolved = {
        "num_bits": weights.num_bits,
        "group_size": weights.group_size,
        "symmetric": bool(weights.symmetric),
        "strategy": strategy,
        "type": getattr(weights.type, "value", weights.type),
    }

    problems: list[str] = []
    if weights.num_bits != spec.bits:
        problems.append(f"num_bits {weights.num_bits} != configured bits {spec.bits}")
    if weights.group_size != spec.group_size:
        problems.append(
            f"group_size {weights.group_size} != configured {spec.group_size}. "
            f"The preset no longer means what configs/quant.yaml assumes, and "
            f"quant_config_hash would NOT change to reflect it."
        )
    if bool(weights.symmetric) != spec.symmetric:
        problems.append(
            f"symmetric {weights.symmetric} != configured {spec.symmetric}. "
            f"llmcompressor's AWQ example uses W4A16_ASYM; normalising both arms "
            f"to symmetric is deliberate and must hold."
        )
    if strategy != "group":
        problems.append(
            f"strategy {strategy!r} is not 'group' — group_size would be ignored"
        )

    if problems:
        raise QuantConfigError(
            f"preset {spec.scheme!r} does not match configs/quant.yaml for "
            f"{spec.key}:\n  - " + "\n  - ".join(problems)
        )
    return resolved


def gptq_actorder(spec: QuantSpec):
    """Translate `desc_act` into `llmcompressor`'s `actorder`, explicitly.

    🚨 **`actorder` MUST be passed even when it is None.** `GPTQModifier.actorder`
    defaults to `Sentinel("static")`, and `resolve_quantization_config` treats
    that sentinel as *deferential*: it injects `ActivationOrdering.STATIC`
    whenever the scheme's own actorder is `None`. Since "explicitly None" and
    "never set" are indistinguishable there, **omitting this argument silently
    yields static activation reordering** — which is neither `desc_act: false`
    (what `quant.yaml` requires, matching Qwen's official checkpoint) nor the
    `desc_act: true` we explicitly reject.

    `llmcompressor`'s own W4A16 example omits `actorder`, so the canonical
    tutorial recipe produces static-actorder weights. Anchor A is a **blocking**
    gate comparing our self-quantized accuracy against Qwen's official
    `desc_act: false` checkpoint, so inheriting that default would have quietly
    invalidated it.

    `ActivationOrdering.GROUP` is the classic `desc_act=True` (runtime group
    index); `STATIC` is an alias for `WEIGHT`, a statically applied permutation.
    Neither is what `desc_act: false` means, and only `None` is.
    """
    if not spec.desc_act:
        return None

    from compressed_tensors.quantization import (  # noqa: PLC0415 - box-only dependency
        ActivationOrdering,
    )

    return ActivationOrdering.GROUP


def build_recipe(spec: QuantSpec) -> list:
    """Map a QuantSpec to a list of `llmcompressor` modifiers.

    Imported lazily and only here, so the module is importable (and testable) in
    an environment with no `llmcompressor` — which is every environment except
    ``venv-quant`` on the GPU box.

    ⚠️ **Always a LIST, even for GPTQ, and not merely for tidiness.** In 0.12.0.1
    `llmcompressor.modifiers.awq.AWQModifier` is not a class at all: it is a
    compatibility function that splits its keyword arguments and returns
    ``[AWQTransformModifier(...), QuantizationModifier(...)]``. Returning a
    uniform list means the caller never has to know which arm produced how many
    modifiers, and `oneshot(recipe=...)` accepts a list either way.

    ⚠️ **No `group_size=` or `desc_act=` keyword on either modifier.** Passing
    them raises pydantic `extra_forbidden` — which is how the smoke run caught
    this before the real job. See `assert_preset_matches_spec` for where
    `group_size` actually lives and `gptq_actorder` for `desc_act`.
    """
    if spec.method == "gptq":
        # Non-deprecated path. `llmcompressor.modifiers.quantization.gptq` is a
        # shim that warns and re-exports from here.
        from llmcompressor.modifiers.gptq import GPTQModifier  # noqa: PLC0415

        assert_preset_matches_spec(spec)
        return [
            GPTQModifier(
                targets="Linear",
                scheme=spec.scheme,
                ignore=["lm_head"],
                actorder=gptq_actorder(spec),
            )
        ]

    if spec.method == "awq":
        from llmcompressor.modifiers.awq import AWQModifier  # noqa: PLC0415

        assert_preset_matches_spec(spec)
        # `scheme` is routed to the QuantizationModifier half; the AWQ transform
        # half infers its layer mappings from the model. Normalised to symmetric
        # W4A16, NOT the example's W4A16_ASYM -- asserted just above.
        kwargs: dict[str, Any] = {
            "targets": "Linear",
            "scheme": spec.scheme,
            "ignore": ["lm_head"],
        }
        if spec.offload_device:
            # llmcompressor's documented OOM remedy: keep the cached forward args
            # off the GPU. Needed because Phi-3.5-mini is the WORST case here
            # despite being smaller than Llama-8B -- it fuses gate and up into one
            # 3072->16384 projection, so AWQ's grid search must materialise a
            # single wide output tensor over every calibration token instead of two
            # narrower ones. It OOM'd with 42.64 GiB allocated of a 44.39 GiB card.
            import torch  # noqa: PLC0415 - box-only dependency

            kwargs["offload_device"] = torch.device(spec.offload_device)

        result = AWQModifier(**kwargs)
        return list(result) if isinstance(result, (list, tuple)) else [result]

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
    dest_override: Path | None = None,
) -> Path:
    """Quantize one (model, precision) and write it to disk.

    Cheap: ~10-15 min for a 3B, ~30 min for an 8B, and only ~1 GB VRAM, because
    sequential layer onloading is the default. The ~16 GB requirement is **host**
    RAM, which is why the instance has 64 GiB.

    ⚠️ `dest_override` exists ONLY for job 1b's replicate (R18.3), which must rebuild a
    bit-identical recipe to a DIFFERENT path so the two can be byte-compared. The digest is
    still computed from the recipe and is unaffected -- that is the point, since a changed
    digest would make the comparison meaningless.

    🚨 It was added because the first attempt suffixed the destination in the PARENT process
    while this function recomputed its own path from `out_root`. The child therefore found
    the original study checkpoint already present, returned it, and the parent then failed
    looking for a suffixed directory that was never written. Nothing was damaged -- the
    idempotence guard below is what protected job 1's artifact -- but the run was wasted.
    A derived path cannot be overridden from the outside unless it is passed in.
    """
    digest = config_hash(spec, calibration, model_revision)
    destination = dest_override or output_dir(out_root, model_key, spec, digest)

    if destination.exists() and (destination / "config.json").exists():
        return destination  # idempotent: already produced by this exact recipe

    if dry_run:
        return destination

    assert_no_forbidden_packages()

    from datasets import load_dataset  # noqa: PLC0415
    from llmcompressor import oneshot  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model_repo, revision=model_revision)
    # ⚠️ NO `device_map="auto"`, deliberately, matching llmcompressor's own W4A16
    # example. `device_map` installs accelerate hooks and places the whole model on
    # the GPU, which defeats llmcompressor's sequential layer onloading -- the very
    # thing `configs/quant.yaml` relies on when it budgets ~1 GB VRAM and ~16 GB
    # HOST RAM. Preloading an 8B onto the card and then accumulating GPTQ Hessians
    # (which are large: in_features^2 floats per layer) is how this OOMs.
    # The example calls `dispatch_model` only AFTER oneshot, for a sanity
    # generation we do not need.
    model = AutoModelForCausalLM.from_pretrained(
        model_repo, revision=model_revision, torch_dtype="auto"
    )

    dataset = load_dataset(calibration.dataset, split=calibration.dataset_split)
    if calibration.shuffle:
        # Shuffle the FULL split, then take n -- not llmcompressor's
        # `split="train_sft[:512]"` then shuffle, which only permutes a fixed
        # prefix. A real random sample is what `subsample_seed` is for, and the
        # cost is one pass over 207,865 rows (~2 s, measured).
        dataset = dataset.shuffle(seed=calibration.seed)
    dataset = dataset.select(range(calibration.num_samples))

    def preprocess(example: dict) -> dict:
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False
            )
        }

    dataset = dataset.map(preprocess)

    # 🚨 TOKENIZE EXPLICITLY. Two reasons, both load-bearing.
    #
    # 1. `add_special_tokens=False` — the chat template has ALREADY inserted BOS.
    #    Letting anything downstream tokenize with the default True gives every
    #    calibration sample a DOUBLE BOS. That is the same class of bug gate G2
    #    caught in the prompt path, and here it would be a systematic difference
    #    from the reference recipe that no accuracy number would reveal.
    # 2. `truncation=True, max_length=...` is what makes the declared token budget
    #    actually BIND. Without it, `max_seq_length` is a hope rather than a bound,
    #    and the budget is the paper's central normalisation claim.
    def tokenize(sample: dict) -> dict:
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=calibration.max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )

    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    # The ACTUAL token count, not the budget ceiling. With `padding=False` a
    # sample shorter than max_seq_length contributes its own length, so the real
    # total is <= num_samples * max_seq_length. The paper must report this number,
    # not the product -- `calibration.total_tokens` is an upper bound by
    # construction and stating it as the realised budget would be wrong.
    actual_tokens = sum(len(ids) for ids in dataset["input_ids"])

    oneshot(
        model=model,
        dataset=dataset,
        recipe=build_recipe(spec),
        max_seq_length=calibration.max_seq_length,
        num_calibration_samples=calibration.num_samples,
    )

    # Saved explicitly rather than via `oneshot(output_dir=...)`, because
    # `save_compressed=True` is what actually writes a compressed checkpoint --
    # this is the example's pattern, and the difference is a full-size
    # unquantized-on-disk artifact that vLLM would not load as a quantized model.
    model.save_pretrained(str(destination), save_compressed=True)
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
                "calibration_tokens_actual": actual_tokens,
                "calibration_tokens_budget": calibration.total_tokens,
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
