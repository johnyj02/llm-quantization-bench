"""The `qa` CLI — one job, one command, resumable and idempotent.

Contract from `EXECUTOR-TASKS.md` §7.3: **re-running any command with identical
arguments produces zero new work and exits 0.**

```
qa prompts      --model qwen2.5-3b --task gsm8k_cot          # materialize + freeze (G2)
qa quantize     --model qwen2.5-3b --precision gptq-4bit
qa generate     --cell <cell_id> --task <task> --budget 1x --variant fewshot_cot --run-id main-v1
qa score        --run-id main-v1 --scorer permissive
qa judge        --run-id main-v1 --judge xverify --cascade none|rung23
qa determinism  --format gptq-4bit --repeats 3
qa analyze      --hypothesis h3 --task gsm8k_cot
qa report       --table decomposition
```

**Every command that costs money is gated** (decisions D10/D11): it refuses
without `--i-have-confirmation`, and each offers a zero-cost inspection mode
(`--plan-only` / `--validate-only` / `--preflight-only`). Commands that only read
stored artifacts are ungated, because scoring and analysis are pure functions over
what generation already wrote — which is the same property that lets anyone
re-analyse the released artifact with no GPU.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

# Commands that can spend money. Listed explicitly so the gate is auditable
# rather than implied by whichever module happens to import torch.
BILLABLE = frozenset({"quantize", "generate", "judge", "determinism", "prompts"})

# `prompts` is billable-adjacent rather than billable: it needs no GPU, but it
# does download datasets and tokenizers. Kept in the list so a launch request
# accounts for it, and it has its own cheap path (`--limit`).


def _dispatch_generate(argv: list[str]) -> int:
    from quantartifact.generate import (
        main as generate_main,  # noqa: PLC0415 - lazy: keeps `qa --help` from importing torch/vllm
    )

    return generate_main(argv)


def _dispatch_quantize(argv: list[str]) -> int:
    from quantartifact.quantize import (
        main as quantize_main,  # noqa: PLC0415 - lazy: keeps `qa --help` from importing torch/vllm
    )

    return quantize_main(argv)


def _dispatch_determinism(argv: list[str]) -> int:
    from quantartifact.determinism import (
        main as determinism_main,  # noqa: PLC0415 - lazy: keeps `qa --help` from importing torch/vllm
    )

    return determinism_main(argv)


def _not_yet(name: str) -> Callable[[list[str]], int]:
    def run(argv: list[str]) -> int:
        sys.stderr.write(
            f"`qa {name}` is not implemented yet.\n"
            f"See notes/gates.md for what is built and what is outstanding.\n"
        )
        return 3

    return run


COMMANDS: dict[str, Callable[[list[str]], int]] = {
    "generate": _dispatch_generate,
    "quantize": _dispatch_quantize,
    "determinism": _dispatch_determinism,
    "prompts": _not_yet("prompts"),
    "score": _not_yet("score"),
    "judge": _not_yet("judge"),
    "analyze": _not_yet("analyze"),
    "report": _not_yet("report"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qa",
        description="QuantArtifact pipeline. One job = one command, resumable and idempotent.",
        epilog=(
            f"Billable commands ({', '.join(sorted(BILLABLE))}) refuse to run "
            f"without --i-have-confirmation, per decision D10. "
            f"Each has a zero-cost inspection mode."
        ),
    )
    parser.add_argument("command", choices=sorted(COMMANDS), help="subcommand")
    parser.add_argument(
        "args", nargs=argparse.REMAINDER, help="arguments passed to the subcommand"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        build_parser().print_help()
        return 0

    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        sys.stderr.write(f"unknown command {command!r}; have {sorted(COMMANDS)}\n")
        return 2

    return COMMANDS[command](rest)


if __name__ == "__main__":
    raise SystemExit(main())
