#!/usr/bin/env python3
"""Order-invariance probe: does item ORDER change what your engine generates?

Runs the same prompts in several orderings inside ONE vLLM engine and checks whether
every item's output is byte-identical across orderings. Ordering 0 is run twice first,
so a non-zero result can be told apart from plain run-to-run nondeterminism.

    VLLM_BATCH_INVARIANT=1 python order_invariance_probe.py \\
        --model Qwen/Qwen2.5-3B-Instruct --prompts prompts.jsonl --max-num-seqs 32

`prompts.jsonl` holds one object per line, either {"id": ..., "prompt": ...} or a record
from the released corpus ({"item_id": ..., "prompt_text": ...}). Prompts must be FULLY
RENDERED (chat template already applied); they are tokenized without adding special
tokens, so a template that already emits BOS is not given a second one.

Exit code 0 means order-invariant, 1 means at least one item changed with the ordering.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

# The ten orderings used in the paper. Seed 0 is the file's own order.
PAPER_SEEDS = (0, 20260827, 20260828, 20260829, 20260830, 20260831,
               20260901, 20260902, 20260903, 20260904)


# Every sampling field pinned, as in the paper, including ones inert under greedy decoding.
DECODING = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0}
ENGINE_ENV = {"VLLM_BATCH_INVARIANT": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
              "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_ATTENTION_BACKEND": "FLASH_ATTN"}


def load_prompts(path: str) -> list[tuple[str, str]]:
    items = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                items.append((str(r.get("id", r.get("item_id"))),
                              r.get("prompt", r.get("prompt_text"))))
    if len({i for i, _ in items}) != len(items):
        sys.exit("item ids must be unique")
    return items


def differing(passes: list[dict[str, str]]) -> list[str]:
    """Items whose text is not byte-identical in every pass (a missing output counts)."""
    ids = set().union(*passes)
    return sorted(i for i in ids if len({p.get(i) for p in passes}) > 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Does item order change what vLLM generates?")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--quantization", help="passed to vLLM, e.g. bitsandbytes")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--stop", action="append", default=None, help="repeatable")
    ap.add_argument("--orderings", type=int, default=len(PAPER_SEEDS),
                    help=f"how many of the paper's {len(PAPER_SEEDS)} seeds to use")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--allow-without-batch-invariance", action="store_true",
                    help="run even if VLLM_BATCH_INVARIANT is not 1 (recorded in the report)")
    ap.add_argument("--out", help="write the JSON report here as well")
    args = ap.parse_args()

    # The paper's serving environment. Set before vLLM is imported, because it reads them then.
    # The FlashInfer sampler is a CUDA path that batch-invariant mode does not cover.
    for key, value in ENGINE_ENV.items():
        os.environ.setdefault(key, value)
    if os.environ.get("VLLM_BATCH_INVARIANT") != "1" and not args.allow_without_batch_invariance:
        sys.exit("VLLM_BATCH_INVARIANT is not 1. The paper measures with it on; pass "
                 "--allow-without-batch-invariance to measure without it.")

    from vllm import LLM, SamplingParams, TokensPrompt, __version__  # noqa: PLC0415

    items = load_prompts(args.prompts)
    # Tensor parallelism above one breaks batch invariance, and prefix caching makes an
    # item's computation depend on what ran before it, so both are pinned here.
    # `generation_config="vllm"` stops vLLM filling unset sampling fields from the model's own
    # generation_config.json: Qwen2.5-3B-Instruct ships repetition_penalty 1.05, which changes
    # greedy output because it rescales logits before the argmax.
    llm = LLM(model=args.model, revision=args.revision, quantization=args.quantization,
              dtype=args.dtype, tensor_parallel_size=1, enable_prefix_caching=False,
              max_model_len=8192, max_num_seqs=args.max_num_seqs, seed=0,
              async_scheduling=False, generation_config="vllm",
              kernel_config={"enable_flashinfer_autotune": False})
    tok = llm.get_tokenizer()
    ids = {i: tok(text, add_special_tokens=False)["input_ids"] for i, text in items}
    params = SamplingParams(**DECODING, max_tokens=args.max_tokens, stop=args.stop, seed=0)

    def run(seed: int) -> dict[str, str]:
        order = [i for i, _ in items]
        if seed:
            random.Random(seed).shuffle(order)
        outs = llm.generate([TokensPrompt(prompt_token_ids=ids[i]) for i in order], params,
                            use_tqdm=False)
        return {i: o.outputs[0].text for i, o in zip(order, outs, strict=True)}

    seeds = PAPER_SEEDS[: max(2, args.orderings)]
    repeat = [run(0), run(0)]
    by_order = [repeat[0]] + [run(s) for s in seeds[1:]]

    rep, order = differing(repeat), differing(by_order)
    report = {
        "model": args.model, "revision": args.revision, "quantization": args.quantization,
        "vllm": __version__,
        "engine_env": {k: os.environ.get(k) for k in ENGINE_ENV}, "dtype": args.dtype,
        "decoding": DECODING, "stop": args.stop,
        "max_num_seqs": args.max_num_seqs, "max_tokens": args.max_tokens,
        "orderings": list(seeds), "n_items": len(items),
        "repeat_differing": len(rep), "order_differing": len(order),
        "order_divergence": len(order) / len(items) if items else 0.0,
        "examples": order[:5],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    if rep:
        print(f"\nNOTE: {len(rep)} items differ between two runs in the SAME order, so the "
              "engine is not deterministic even before ordering is varied.", file=sys.stderr)
    print("\nORDER-INVARIANT" if not order else
          f"\nNOT ORDER-INVARIANT: {len(order)} of {len(items)} items changed with ordering",
          file=sys.stderr)
    return 0 if not order else 1


if __name__ == "__main__":
    raise SystemExit(main())
