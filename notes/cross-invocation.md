# Cross-invocation nondeterminism — found in a pre-flight check, cause UNDIAGNOSED

**How it surfaced.** The user asked for a sanity check on earlier results before committing to
the 10.5 h `main-v1` run. `pilot-v1` and `pilot-v2` ran the same items in the same order and
only BBH's `until` differed (R19.1), so `gsm8k` and `arc_challenge_chat` should have been
bit-identical. They were not.

⚠️ **This is not H8.** It affects `bf16`, which H8 established as *exactly* order-invariant
within a job (0.000% across ten orderings, 500 items). Different phenomenon, different cause,
and it lands on a different quantity.

---

## 1. What was verified clean first

- **All 9 quantized checkpoints present**, and the recipe hashes `sm_reference.py` computes
  match job 1's recorded digests exactly. That was the failure that would have wasted the slot.
- **Prompts are 100% identical** across runs, every cell, every task. The renderer is
  deterministic; nothing upstream of generation drifted.
- Records carry `finish_reason` and `n_gen_tokens` on every row, so H1 is not at risk.

## 2. Text divergence — the shape of it

`pilot-v1` vs `pilot-v2`, identical config, gsm8k (100 items) / arc (80):

| model | arm | gsm8k | arc | concurrency¹ |
|---|---|---|---|---|
| qwen2.5-3b | **all 7 cells** | **0/100** | 0/80 | 118.80–132.71× |
| phi-3.5-mini | bf16 | **45/100** | 0/80 | **10.70×** |
| phi-3.5-mini | bnb-nf4 | **55/100** | 2/80 | — |
| phi-3.5-mini | gptq-8/gptq-4/awq | **0/100** | 0/80 | **12.36×** (gptq-4) |
| llama-3.1-8b | **all 5 arms** | **41–45/100** | 1–2/80 | 23.76–33.66× |

¹ engine-reported *"Maximum concurrency for 8,192 tokens per request"*.

Every recorded field on a differing pair is byte-identical: `env_hash a216dd7e4174271d`,
`max_num_seqs 64`, `vllm==0.26.0`, `model_revision`, `quant_config_hash 89ac5631ebff47d9`,
`decoding greedy`, `seed 0`, `max_gen_toks 256`, `batch_invariant True`, `cuda 13.0`,
`driver 595.91.07`, `NVIDIA L40S`, `n_prompt_tokens 954`. Divergence begins mid-trace
(char 233 of ~494) and cascades — one flipped token.

**It scales with generation length:** arc (~10 tokens, `until` contains `.`) is 0–2%; gsm8k
(~150 tokens) is 0–55%.

## 3. 🚨 The accuracy consequence — MATERIAL, and it does NOT cancel across arms

Signed delta, pp, gsm8k+arc, n=180 per cell:

| cell | strict | permissive |
|---|---|---|
| llama bf16 | **+1.11** | +2.22 |
| llama gptq-8bit | +1.67 | 0.00 |
| llama gptq-4bit | **−1.67** | −1.11 |
| llama awq-4bit | −1.11 | −2.22 |
| llama bnb-nf4 | +0.56 | 0.00 |
| phi bf16 | −0.56 | −1.11 |
| phi bnb-nf4 | **+4.44** | +2.78 |
| phi gptq-8/4, awq · qwen all | 0.00 | 0.00 |

**worst |Δ| 4.44 pp · mean |Δ| 0.60 pp · SD of deltas 1.20 pp**

🚨 **The crux.** Ordering divergence was *common across arms*, so it contributed zero
**differential** noise. This does not: on Llama, `bf16` moved **+1.11** while `gptq-4bit` moved
**−1.67**, so the **bf16-vs-gptq-4bit delta swung 2.78 pp from nothing but re-running.** That is
§4.6's differential-noise concern landing for real, at the same order as the quantization
effects H3/H4 estimate.

## 4. Two hypotheses, both REFUTED cheaply

| hypothesis | test | verdict |
|---|---|---|
| vLLM profiles free GPU memory per invocation → different `num_gpu_blocks` → different batch scheduling (R21's) | grep KV cache size from both job logs | ❌ **byte-identical** cell for cell: llama gptq-4bit 273,616 both, qwen 1,087,168 both. Block count is deterministic. |
| `max_num_seqs=64` exceeds KV-derived concurrency, so batching is memory-scheduled not count-scheduled (mine) | compare concurrency against differ/stable | ❌ phi gptq-4bit **stable at 12.36×**, phi bf16 **differs at 10.70×**. Not separable. |
| base repo vs local checkpoint | cross-tabulate | ❌ explains phi exactly, but llama's checkpoint arms differ and qwen's base-repo bf16 does not. |

**What predicts it:** llama always (5/5), qwen never (0/7), phi only its two base-repo arms.
Not separable by concurrency, weight source, arm, or bit width.

⚠️ **THE MECHANISM IS UNDIAGNOSED.** Two cheap refutations are real progress and are recorded
as such, but no third guess is offered here. Remaining untested candidates: CUDA graph capture
order, Triton/inductor JIT cache state (the container's `~/.cache/vllm/torch_compile_cache` is
not persisted, so kernels are recompiled per job), and cuBLAS workspace/heuristic selection.

## 5. Why it blocks `main-v1`, and why the obvious fixes do not work

The reference stage resumes per item, so an interruption regenerates the remainder in a new
process. But the deeper problem is structural and applies even to an uninterrupted run:

🚨 **Each arm needs its own engine load, because each arm is different weights. Per-arm
invocation is therefore unavoidable, so this variance is INHERENTLY BETWEEN-ARM and cannot be
made common the way ordering could.**

- **Pinning the block count is out** — they are already identical.
- **"All arms of a model in one invocation" is impossible** — different weights, different
  engine.

So on affected models the between-arm deltas carry ~1–3 pp of irreducible invocation noise.
Held pending R22.

## 6. Committed regardless of the ruling

`invocation_id` on every generation record (R21). Provenance for a 15-cell run spanning many
engine loads should be **exact, not inferred** — and without it we cannot even tell which items
in a resumed run came from which process.


---

## 7. The gap analysis (R22) — mechanism confirmed, but H3 is NOT protected

The planner spotted what neither of us had: **rungs 1 and 2 score the same stored
generations**, so re-scoring carries zero invocation noise. It predicted the strict−permissive
*gap* would therefore be more stable than either raw accuracy. Computed on data in hand,
`pilot-v1` → `pilot-v2`, gsm8k+arc, n=180/cell, 17 cells:

| quantity | mean \|Δ\| | max \|Δ\| | SD across cells |
|---|---|---|---|
| strict accuracy | 0.65 pp | **4.44 pp** | 1.15 pp |
| permissive accuracy | 0.56 pp | 2.78 pp | 0.96 pp |
| **gap (perm − strict)** | **0.42 pp** | **1.67 pp** | **0.61 pp** |

✅ **Confirmed.** Cleanest case: phi bnb-nf4's strict moved **4.44 pp** while its gap moved
**1.67 pp** — the noise flipped items under both scorers and cancelled in the difference. The
gap is ~2.7× more stable by max, ~1.9× by SD.

🚨 **BUT THE INTERACTION IS THE WORST QUANTITY OF THE THREE, NOT THE BEST.**
`interaction(arm) = gap(arm) − gap(bf16)`, same model:

| cell | \|Δ interaction\| |
|---|---|
| llama gptq-8bit | **2.78 pp** |
| llama awq-4bit | 2.22 pp |
| llama bnb-nf4 | 1.67 pp |
| phi bnb-nf4 | 1.11 pp |
| llama gptq-4bit, phi awq/gptq-4/gptq-8 | 0.56 pp |
| qwen (all) | 0.00 pp |

mean 0.71 pp, **max 2.78 pp**.

R22 classified *"H3's interaction"* as binding to the **repetition floor (0.000%)** on the
grounds that it re-scores identical text. **That does not hold.** H3's interaction is
precision × scoring, so it contrasts **two arms**, hence **two invocations** — bf16's noise and
the arm's noise both enter and *compound* rather than cancel. Hence 2.78 pp, worse than either
raw accuracy delta on that cell.

### The taxonomy this implies — THREE classes, not two

1. **Zero invocation noise** — computed by re-scoring **one fixed set** of generations: the rung
   ladder's monotonicity, compliance labels, H2's trace scan, the found/committed split,
   per-item rung agreement. These never cross an invocation boundary.
2. **Partially cancelling** — a **within-arm** difference of two rungs (the gap): **max 1.67 pp**.
   It crosses an invocation boundary, since the generations differ between runs, but the two
   rungs share the same text so the noise is common and largely cancels.
3. **Compounding** — anything **contrasting arms**: raw between-arm deltas (max **4.44 pp**),
   **H3's interaction** (max **2.78 pp**), H4's decomposition components. Two invocations, noise
   adds.

⚠️ **H3 lands in class 3.** §2 calls it a headline and rests "a null is publishable" on it, so a
2.78 pp floor under an interaction estimate is a materially worse position than R22's ruling
implies. Raised with the planner as R23 rather than implemented, because it changes which arms
need K=2: if H3 is class 3 then qwen needs repetition too, despite showing 0.00 at n=180 — and
qwen carries the largest rung gaps (19–23 pp), i.e. the most H3 signal.
