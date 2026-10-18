# The paper's claims, as numbers

Snapshot of commit `70c923e` (2026-10-18). Every number is copied from the generated files in
`paper/tables/`, the same source the paper reads. If a table is regenerated, re-check this file.
"pp" means percentage points of accuracy.

## Headline 1: item order moves 4-bit accuracy

L40S, Qwen2.5-3B, 500 items, 10 orderings, batch size 32.

| arm | accuracy SD, strict | accuracy SD, permissive | range, strict / permissive |
|---|---|---|---|
| bf16 | **0.00** | **0.00** | 0.00 / 0.00 |
| GPTQ W4A16 | 0.87 | 0.75 | 2.80 / 2.40 |
| AWQ W4A16 | 0.52 | **1.09** | 2.00 / 4.20 |
| bnb NF4 | 0.11 | 0.10 | 0.40 / 0.20 |
| same order, run again (all 15 cells) | 0 outputs differ | | |

## Proof by intervention: Marlin and Machete kernels switched off

Batch size 8.

| arm | text changed, kernels on | text changed, kernels off | accuracy SD, on | accuracy SD, off |
|---|---|---|---|---|
| GPTQ W4A16 | 53.2% | **0.0%** | 0.42 | **0.00** |
| AWQ W4A16 | 53.6% | **0.0%** | 0.35 | **0.00** |
| bnb NF4 | 49.4% | 49.4% (no change) | 0.31 | 0.31 |

## Replication on a second GPU

A10G (Ampere, SM 8.6) against the L40S (Ada, SM 8.9). Same image, engine, weights and probe.

| measure | L40S | A10G |
|---|---|---|
| bf16 text changed across orderings | 0.0% | 0.0% |
| GPTQ text changed across orderings | 71.2% | 69.8% |
| GPTQ accuracy SD, strict | 0.87 | 0.70 |
| GPTQ, kernels on → off (batch 8) | 53.2% → 0.0% | 53.4% → 0.0% |

## Headline 2: the scoring rule changes the quantization gap

In pp. A plus sign means lenient scoring helps the 4-bit model more than bf16. Bold means the
95% interval excludes zero.

| arm | ARC-C | BBH | GSM8K CoT |
|---|---|---|---|
| GPTQ W8A16 | 0.00 | +0.06 | +0.53 |
| GPTQ W4A16 | 0.00 | **+1.73** | **−5.21** |
| AWQ W4A16 | 0.00 | **+1.55** | −0.91 |
| bnb NF4 | 0.00 | **−0.97** | **−8.44** |

| pre-registered contrast | estimate | 95% CI | survives Holm? |
|---|---|---|---|
| GPTQ W4A16: BBH vs GSM8K | +6.93 | [+5.09, +8.76] | yes |
| GSM8K: GPTQ vs AWQ | −4.30 | [−5.84, −2.75] | yes |

## Supporting results

| result | key numbers | status |
|---|---|---|
| Same process vs separate processes | 0 of 42,380 differ in one process; up to 45% differ across processes | cause not found (4 candidates ruled out) |
| Budget truncation, GSM8K (default → 4× budget) | stopped by the cap: bf16 15.2% → 0.5%, GPTQ 22.2% → 2.0%, AWQ 29.5% → 5.7% | real in tokens |
| Budget effect on accuracy | best p = 0.048, Holm threshold 0.0125; 0 of 4 pass | null result (bounded) |
| Answering at all (commitment) | 48 of 108 cells change; GPTQ on BBH: 423 lost, 421 gained, net −0.06 | supports Headline 2 |
| ARC log-likelihood control, Qwen | GPTQ effect −13.2 pp [−16.6, −10.0]; AWQ −8.4; other models ≤ 1.8 | 2 of 6 pass Holm |

## Remedies

| remedy | removes | cost |
|---|---|---|
| Generate all precisions in one process | cross-process variation | none |
| Disable Marlin and Machete | ordering variation (GPTQ, AWQ) | 3.7× wall clock |
| Either one | NF4 ordering variation | not fixed |

## Released data

| item | size |
|---|---|
| Main-run generations | 105,950 |
| All released generations (8 runs) | 190,945 |
| Archive | 12.0 MB (940 MB unpacked) |
