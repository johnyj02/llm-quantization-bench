# RUNLOG — append-only record of every billable run and what it taught us

**Newest entries at the BOTTOM.** Append, never rewrite: the point is a faithful trail, including
the attempts that failed and the reasons they failed.

**How to append.** `bash scripts/logrun.sh <job-name> "<one-line outcome>"` pulls status, billable
seconds and cost straight from the SageMaker API, so the numbers in this file are never from
memory. Then hand-edit the "what it taught us" bullet underneath.

Related files, kept separate on purpose:
- `notes/decisions.md` — durable decisions (D-numbers)
- `notes/planner-rulings.md` — planner's authoritative rulings (R-numbers)
- `notes/compute.md` — technical write-ups of substrate findings
- **this file** — chronological, per-run, with money attached

---

## Ledger

| # | job | status | billable | cost | outcome |
|---|---|---|---|---|---|
| 1 | `…20260826…-verify` | Failed | 130 s | $0.10 | 13/14 checks. CUDA-13 risk CLOSED. The 1 failure was my own wrong check. |
| 2 | `…20260826…-quantize-smoke` | Failed | 196 s | $0.15 | `group_size`/`desc_act` are not llmcompressor kwargs. |
| 3 | `…20260826…-quantize-smoke-r2` | Stopped | 0 s | $0.00 | Stopped while Pending — superseded by fixes. Never ran. |
| 4 | `…20260826…-quantize-smoke-r3` | Completed | 612 s | $0.48 | 2/2 arms, both methods. Recipe fix validated. |
| 5 | `…20260826…-quantize` | Failed | 4546 s | $3.54 | **5/9 arms.** GPU memory leak across arms in one process. |
| 6 | `…20260827…-quantize-smoke` | Completed | 587 s | $0.46 | Subprocess-per-arm validated. |
| 7 | `…20260827…-quantize` | Failed | 4052 s | $3.15 | **8/9 arms.** Leak fixed; phi/awq OOM'd for real. |
| 8 | `…20260827…-quantize-smoke-r2` | Completed | 586 s | $0.46 | `offload_device` path validated. |
| 9 | `…20260827…-quantize-r2` | Completed | 7555 s | $5.88 | ✅ **9/9. JOB 1 COMPLETE.** |
| 10 | `…20260827…-determinism-smoke` | Failed | 482 s | $0.37 | `nvcc` missing → flashinfer sampler. Also caught `async_scheduling` unenforced. |

**Total to date: 18,746 s = 5.21 h = $14.58** at $2.80/h (`USE1-Train:ml.g6e.2xlarge`).

Spend by purpose: real quantization $12.56 · smokes $1.54 · verify $0.10 · determinism $0.37.
The smokes are **11% of spend** and caught four bugs that would each have cost a multi-hour run.

---

## 1 — `verify` · Failed · 130 s · $0.10

First job on the substrate. **13 of 14 checks passed.** SageMaker says Failed because
`sm_verify.py` exits non-zero on any failed check; the report was written in full.

- 🎉 **The CUDA-13 risk is closed.** driver 595.91.07, torch 2.11.0+cu130, vLLM links
  `libcudart.so.13`. Engine stays frozen at 0.26.0 — no planner escalation, and H8's
  version-specific kernel guards stay valid.
- L40S reports **46068 MiB**, not the nominal 48 GB. Tightens the 8B arms' headroom.
- The one failure was **my check, not the box**: `import vllm._C` does not exist in 0.26.0 (ops
  register through `torch.ops`), while `objdump` had already read the `.so` fine.
- `bi_supported_on_sm89` left `null`. Triton compiling a kernel is necessary but not sufficient.

## 2 — `quantize-smoke` · Failed · 196 s · $0.15

Everything up to quantization worked (model fetch, 434 weights, `ultrachat_200k`, calibration map).

```
ValidationError: 2 validation errors for GPTQModifier
group_size   Extra inputs are not permitted
desc_act     Extra inputs are not permitted
```

- Neither is a modifier kwarg in llmcompressor 0.12.0.1. `group_size` lives in the preset scheme.
- 🚨 **`desc_act` does not exist and its replacement fails SILENTLY.** `GPTQModifier.actorder`
  defaults to `Sentinel("static")`, which injects `STATIC` whenever the scheme's actorder is
  `None` — so omitting it yields static activation reordering, which is neither `desc_act: false`
  (what we require, matching Qwen official) nor the `true` we reject. **llmcompressor's own W4A16
  example omits it**, and Anchor A is a *blocking* gate against Qwen's `desc_act: false`
  checkpoint. Pydantic erroring instead of ignoring is what saved this.
- Process lesson: the smoke ran **GPTQ only**, so it said nothing about AWQ — which uses a
  genuinely different API (`AWQModifier` is a *function* returning two modifiers). `--smoke` now
  runs one arm per method.

## 3 — `quantize-smoke-r2` · Stopped · 0 s · $0.00

Stopped while still `Pending`, because it was testing superseded code. Zero cost.

- ⚠️ **Learned here: quota is enforced at submit time, capacity is not.** A `Pending` job holds
  the quota slot, so with a limit of 1 a second submission is refused outright
  (`ResourceLimitExceeded`). And `stop-training-job` does not release the slot until the job
  reaches `Stopped` (~45 s observed).

## 4 — `quantize-smoke-r3` · Completed · 612 s · $0.48

2/2 arms, both methods. `Compression lifecycle finalized for 2 modifiers` confirmed that the AWQ
shim really does return two modifiers.

## 5 — `quantize` (job 1, attempt 1) · Failed · 4546 s · $3.54

**5 of 9 arms.** Not a timeout — `max_run` was 21600 s and it used 4546.
`OutOfMemoryError: Sequential pipeline ran out of memory` on phi/awq and all three Llama arms.

- **A leak, not a capacity limit.** Log showed `Accelerator 0 | usage: 95.70% | total 47.7 Gb`,
  where a *fresh* Llama-8B GPTQ pass needs ~10 GB (~436 MB layer weights + ~1.2 GB Hessians +
  ~8.6 GB cached activations). The failure *order* confirmed it: all fine → phi's AWQ → Llama 8-bit
  partway → last two within seconds.
- Fix: **one subprocess per arm.** `del model` leaves torch's allocator holding blocks,
  llmcompressor installs hooks, and one arm had logged `Failed to compress 7 modules` — so the
  next arm inherited a poisoned lifecycle. Process exit is the only reliable teardown. This was
  already the pattern in `sm_determinism`/`sm_throughput`; applying it to vLLM but not
  llmcompressor was an inconsistency that cost 76 minutes.
- Rejected the error message's own suggestion (`sequential_targets="Linear"`): it changes GPTQ's
  Hessian granularity and therefore potentially the weights. Process isolation is neutral.
- ✅ **Checkpointing earned its keep**: all 5 arms persisted and were skipped by hash on retry.

## 6 — `quantize-smoke` (Aug 27) · Completed · 587 s · $0.46

Subprocess path validated — `__ARM_OK__` from the child, 2/2 arms, output streaming live.

- Also validated `--smoke` never skipping: `[smoke] discarded previous rehearsal artifact`. A
  rehearsal that resumes reports success while running nothing.

## 7 — `quantize` (job 1, attempt 2) · Failed · 4052 s · $3.15

**8 of 9.** All three Llama arms now succeeded — the leak was fixed. One genuine failure.

- `phi-3.5-mini/awq-4bit`, in a **fresh** subprocess: OOM with 42.64 GiB allocated of a 44.39 GiB
  card, failing on 34 MiB while holding 1.21 GiB reserved-but-unallocated.
- **Phi is the worst case for AWQ despite being smaller than Llama-8B**: it fuses gate and up into
  one 3072→16384 projection, so the grid search materialises one wide output tensor over every
  calibration token instead of two narrower ones.
- Measured calibration: **579,588 tokens = 55.3% of the 1,048,576 ceiling.** Reporting the product
  would have overstated it ~2×.

## 8 — `quantize-smoke-r2` (Aug 27) · Completed · 586 s · $0.46

`offload_device=cpu` reaches AWQModifier; AWQ hash moved `a0d9ab1c…` → `dfa10bfa…` as intended.

## 9 — `quantize-r2` (job 1, final) · Completed · 7555 s · $5.88 · ✅ JOB 1 COMPLETE

**9/9 arms.** 6 GPTQ skipped by preserved hash, 3 AWQ rebuilt with offload. 45.86 GB in S3.

- Offload costs ~2× on the slowest arm: `llama/awq-4bit` 53.7 min with, 27.1 min without.
- Near-miss: adding `offload_device` to `QuantSpec` changed the digest of **all nine** arms —
  including six GPTQ ones an AWQ-only setting cannot affect — purely because `to_dict()` gained a
  key whose value was `None`. Would have invalidated 8 finished checkpoints. `config_hash` now
  ignores unset fields; the six known digests are pinned in tests.
- 🔬 **Unresolved finding, now a main-text contribution (R12.1): AWQ weights are not reproducible.**
  Keeping the superseded checkpoints let me compare via S3 without downloading — identical size
  (2,675,871,568 B) and identical safetensors header, but different ETag and **6 of 13 sampled
  64 KB windows differ**. Not yet attributable to `offload_device` vs run-to-run nondeterminism;
  one more arm settles it.

## 10 — `determinism-smoke` · Failed · 482 s · $0.37

Got capacity, loaded the GPTQ-4bit checkpoint, died on the first `generate()`.

```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
  flashinfer/sampling.py:68  get_sampling_module   <- JIT-compiles at RUNTIME
```

- **Same class as Triton needing `build-essential`**, different compiler. Our image has the CUDA
  runtime (torch cu130 wheels) but not the toolkit. Fix: `VLLM_USE_FLASHINFER_SAMPLER=0`, now in
  `REQUIRED_ENV` so `preflight()` asserts it.
- ⚠️ Right on the merits too: vLLM's comment says disabling it *"forces the PyTorch-native
  (Triton for bs>=8) path"* — which is exactly what `VLLM_BATCH_INVARIANT` substitutes into. The
  flashinfer sampler is a separate CUDA path batch invariance does **not** cover.
- 🚨 **The bigger finding: `async_scheduling` was declared but not enforced.** The engine log read
  `Asynchronous scheduling is enabled` while `models.yaml` has said `false` throughout. Nothing
  forwarded it, so vLLM's default won — **third instance of this shape** after GPTQ's
  `desc_act`/`actorder`. A value we declare but do not enforce is decoration, *and the paper
  reports it as though it held*. Now an `EngineSpec` field, forwarded, and asserted.
- Reporting integrity: the run printed `noise floor: bf16 100.0000%` for cells that never ran.
  Failing closed is right; presenting it as data is not. `DisagreementResult.measured` now marks
  them, excludes them from the noise floor, and prints `NOT MEASURED`.
- Open for the planner, not silently changed: `enable_flashinfer_autotune=True` could vary kernel
  choice run to run, which would register in A7 as arm nondeterminism whose real cause is
  autotuning rather than the kernel-guard asymmetry H8 is about — a confound for H8.

---

## Running themes worth not relearning

1. **vLLM and llmcompressor compile at RUNTIME.** Two separate compilers have now been missing:
   `gcc`/`build-essential` for Triton, `nvcc` for flashinfer. Neither is visible until generation.
2. **Declared ≠ enforced.** Three times now a value in `configs/` was silently overridden by a
   library default (`desc_act`, `async_scheduling`) or was simply never forwarded. Every declared
   setting needs a real field, a forward, and an assertion — otherwise the paper reports fiction.
3. **A smoke that exercises one path clears only that path.** GPTQ-only missed AWQ's entirely
   different API; `bf16`-only would have missed quantized checkpoint loading.
4. **Fail closed, but never print a failure as a measurement.**
5. **Wall clock, not GPU time, is the binding constraint.** Queue waits have run 20 min – 2.5 h
   with quota 1 on a shared account. Smokes are cheap; queue cycles are not.

## determinism-smoke-r2 · Stopped · 0 s · $0.00

`johnyj02-QA-v0-1-20260827-33b4267e-determinism-smoke-r2`

- Stopped while Pending ($0) — superseded by the token-ids fix, which replaced the text path it would have validated.
- Not a failure: stopped deliberately while still `Pending`, so zero cost.
- ⚠️ Worth recording because of WHAT superseded it. R14.3's config audit asked why
  `models.yaml: serving.add_bos_token` was consumed by nothing, and the answer was a
  real bug: `run_cell` passed prompts as **text**, so vLLM re-tokenized them on the
  *completion* path (`add_special_tokens=True`) rather than the chat path (`False`,
  "suppressed for chat, where the template already supplies them"). Our prompts are
  chat-templated, so any model whose template emits BOS — Llama-3.1 emits
  `<|begin_of_text|>` — got a **second** one. `materialize.py` exists to prevent
  exactly that and its work was being discarded at generation time.
- Generation now passes the frozen `prompt_token_ids` via `TokensPrompt`, and asserts
  the engine's own reported ids match the artifact. A config audit found a
  correctness bug in every generation, which is the argument for doing such audits
  before generations exist rather than after.
