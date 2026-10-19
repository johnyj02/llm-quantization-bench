# Supplementary data: item ordering and scoring in quantized model evaluation

This archive accompanies the paper *Reproducibility Failures in Quantized Model Evaluation:
Item Ordering Moves 4-Bit Accuracy and Leaves bf16 Exactly Unchanged*. It holds every model
generation the paper's numbers are computed from, the order-invariance probe, and the human
annotation record. Every record keeps the full generated text and the fields that cannot be
reconstructed later, so any answer extractor can be re-run on it without a GPU.

In this repository, the extracted data is in
[`quantized-eval-ordering-supplement/`](quantized-eval-ordering-supplement/).
The layout below describes the contents of that directory and the archive.
The probe script is also available alongside this README.

```
corpus/<run_id>/<model>__<precision>/<task>[__<subtask>]__<prompt>__<budget>.jsonl
determinism/l40s/        probe passes and per-cell analyses, NVIDIA L40S (SM 8.9)
determinism/a10g/        second-GPU replication, NVIDIA A10G (SM 8.6)
determinism/reports/     the three job reports the paper's ordering tables are built from
annotation/              rubric, blinded sample, both annotators' labels, adjudication
order_invariance_probe.py
MANIFEST.json            record counts and SHA-256 of every file
```

Unpack with `tar -xJf <archive>.tar.xz`.

## Runs

| run_id | what it is | used in |
|---|---|---|
| `main-v1` | The reference condition: 3 models x 5 precisions x 4,238 items, plus a second repeat (`repeat_idx=1`) of 10 cells. **105,950 records.** | §4.2, §4.3, §4.5 |
| `sweep-v1` | A 1,400-item subset of `main-v1` (500 GSM8K-CoT, 900 BBH: 150 per subtask) at 2x and 4x the generation budget, 3 models x 3 precisions. Its 1x level is the same items in `main-v1`. | §4.4 |
| `h6-v1` | ARC-Challenge on `main-v1`'s prompt, scored by log-likelihood over the four letters (`option_logprobs`). | §4.6 |
| `template-v2` | The prompt-template factor, chat and raw. Supersedes `template-v1`, which ran at `max_num_seqs=64` before the frozen value moved to 32. | supplementary |
| `pilot-v2` | The pilot. Source of the 250-item annotation sample. Supersedes `pilot-v1`, which predates a fix to BBH's stop sequences. | annotation |
| `anchor-v1` | Our GPTQ checkpoints against the vendor's official Qwen checkpoints. | Method, Limitations |
| `pilot-v1`, `template-v1` | Superseded, kept so that nothing is selectively omitted. | none |

Rehearsal ("smoke") runs are not included.

## Record fields (schema 1.5.0)

One JSON object per line. The join key is
`(run_id, cell_id, task, subtask, item_id, budget_level, prompt_variant, sample_idx, repeat_idx)`.

| field | meaning |
|---|---|
| `cell_id`, `model_repo`, `model_revision` | model and precision; revisions are commit SHAs, not tags. For GPTQ and AWQ, `model_repo` is the path of our quantized checkpoint inside the serving container; the base model is the first part of `cell_id` and the recipe is `quant_config_hash` |
| `quant_method`, `bits`, `quant_config_hash` | quantization recipe; the hash identifies the exact calibration recipe |
| `task`, `subtask`, `item_id`, `gold` | benchmark item and its reference answer |
| `prompt_text`, `prompt_sha256`, `n_prompt_tokens` | the fully rendered prompt as served |
| `prompt_variant`, `template_level`, `scoring_mode` | few-shot CoT or zero-shot; chat or raw template; `generate` or `loglikelihood` |
| `budget_level`, `max_gen_toks` | generation budget (1x, 2x, 4x) and its token cap |
| `generation_text` | the FULL output, never truncated for storage |
| `finish_reason`, `n_gen_tokens` | why generation stopped (`stop` or `length`) and how many tokens it produced |
| `until`, `until_applied` | the stop strings, and whether they were applied at `generation` (the engine halted there) or at `scoring` (the full output is kept and the scorer cuts it) |
| `stop_reason`, `stop_string_hit`, `n_gen_tokens_scored` | which string ended the scored prefix, and that prefix's length in tokens |
| `option_logprobs` | log-likelihood mode only: per-option log-probability and token count |
| `decoding`, `seed`, `sample_idx`, `repeat_idx` | greedy throughout; `repeat_idx=1` is a second generation in the same job |
| `engine`, `batch_invariant`, `max_num_seqs`, `disabled_kernels` | serving configuration (vLLM 0.26.0, batch-invariant mode on, `max_num_seqs=32`) |
| `gpu`, `cuda`, `driver`, `env_hash`, `invocation_id`, `wall_ms` | hardware, and which engine load produced the record |

Under `until_applied == "scoring"`, `finish_reason` and `n_gen_tokens` describe the full
output; the scored answer is the prefix up to `stop_string_hit`.

## Determinism probe data

Each `raw_*.json` holds the probe passes for one cell: `passes[k]` has `ordering_seed`,
`outputs` (item id to generated text) and `seconds`. The probe stores text only, not
`finish_reason` or token counts. The file name encodes model, precision, served batch size
(`at32`), repeats, the ordering count (`ord10x` for the ten-ordering runs), a decoding
fingerprint (`dec…`), and `fallback` when the Marlin and Machete kernels were disabled.
A cell file without the `raw_` prefix is that cell's analysis. `determinism/index.json`
maps every raw file to the report cells that use it. Each report cell is identified by its
analysis file (same per-pass timings and output hashes, exactly one match required), and the
hashes of the raw file's passes are then checked against the report. The output hash alone
cannot identify a cell, because bf16 produces the same text at every batch size. Raw files no
report uses were measured under an earlier configuration and are superseded.

## The probe

`order_invariance_probe.py` runs the same prompts in several orderings inside one vLLM
engine and checks whether each output is byte-identical across orderings. It runs ordering
0 twice first, so ordering effects are separated from run-to-run nondeterminism. It accepts
corpus records directly:

```
VLLM_BATCH_INVARIANT=1 python order_invariance_probe.py \
    --model Qwen/Qwen2.5-3B-Instruct --prompts prompts.jsonl --max-num-seqs 32
```

It exits 0 if the outputs are order-invariant and 1 if not. It needs a GPU and vLLM.

## Annotation

`rubric.md` is the decision procedure both annotators applied. `sample_key.json` maps each
blinded annotation id to its corpus record (`run_id`, `cell_id`, `item_id`, ...).
`to_annotate_A*.json` are exactly what each annotator saw, in their order;
`labels_A*.jsonl` are their labels; `adjudicated.json` records how the disagreements were
resolved and the rubric clause that decided each.

## Licenses

The generations are model outputs on public benchmark items, so the terms of both apply.

| component | license |
|---|---|
| GSM8K (`openai/gsm8k`) | MIT |
| BIG-Bench Hard (`suzgunmirac/BIG-Bench-Hard`) | MIT |
| ARC (`allenai/ai2_arc`) | CC BY-SA 4.0: records from ARC tasks are redistributed under CC BY-SA 4.0 |
| Qwen2.5-3B-Instruct outputs | Qwen Research License (non-commercial) |
| Phi-3.5-mini-instruct outputs | MIT |
| Llama-3.1-8B-Instruct outputs | Llama 3.1 Community License |
| `order_invariance_probe.py` | MIT |
| annotation record | CC BY-SA 4.0, because 24 of its 250 items embed ARC question text |
| this README | CC BY 4.0 |

No weights of any model are included, and nothing derived from the xVerify judge
(CC BY-NC-ND 4.0) is redistributed.
