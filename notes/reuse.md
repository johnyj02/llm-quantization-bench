# Reuse from sibling projects

Investigated 2026-08-23 at the user's direction: `~/workspaces/MedLLMSafety` (vLLM
batch generation, statistics) and `~/workspaces/insertion-stability` (LaTeX).

Verdict up front: **MedLLMSafety is worth a lot, but not the part that looks most
reusable.** Its vLLM backend cannot be used as-is — it discards the two fields H1
depends on. What it *does* give us is hard-won operational knowledge that would
otherwise cost days of GPU debugging, including one landmine that would have broken
the determinism audit outright.

---

## 1. 🚨 The landmine: Triton needs a C compiler **at runtime**

From `MedLLMSafety/Dockerfile.vllm`, learned the expensive way (a real job failed
after getting past every other error):

> *"no C compiler" from Triton's JIT kernel build (torch.compile path)* — the build
> toolchain must **not** be purged after `pip install`.

**Why this is critical for us and not merely for them.**
`VLLM_BATCH_INVARIANT=1` works by *substituting batch-invariant Triton kernels* for
the default GEMM/softmax/reduction ops. Triton **JIT-compiles those at runtime**. So:

> **No C compiler on the box ⇒ no batch-invariant kernels ⇒ no determinism ⇒ H8 is
> dead and every precision delta in the paper is untrustworthy.**

`A1` must therefore install and **keep** `gcc`/`build-essential` on the instance, and
`A4`'s startup assertions must verify a working compiler is reachable, alongside the
existing checks that `VLLM_BATCH_INVARIANT` is set and `tensor_parallel_size == 1`.
Do not discover this on the first determinism run.

## 2. `max_model_len` must be capped, and the reason is measured not theoretical

Their docstring records a real job failure on a 24 GB A10G:

> Llama-3.1-8B-Instruct's **full trained context is 131072**, which needs ~16 GiB of
> KV cache at `gpu_memory_utilization=0.90`, but only **3.89 GiB** was available after
> model weights: `ValueError: ... 16.0 GiB KV cache is needed, which is larger than
> the available KV cache memory (3.89 GiB)`.

They cap at `max_model_len=4096`. **We must cap too, and higher than they do**, because
our budget sweep goes to 4× on BBH:

| Arm | prompt | + `max_gen_toks` | need |
|---|---|---|---|
| GSM8K 8-shot CoT @ 4× | ~1.2k | 1024 | ~2.2k |
| BBH 3-shot CoT @ 4× | ~1.5k | **4096** | **~5.6k** |
| ARC-chat 0-shot @ 4× | ~0.3k | 400 | ~0.7k |

**Set `max_model_len = 8192`** — covers the worst arm with headroom, and frees a large
amount of KV cache versus the 131072 default. On a 48 GB L40S with an 8 B model
(~16 GB weights) that leaves ample cache. Record the value in `configs/models.yaml`
and **hold it fixed across every arm** — it is part of the frozen serving config, and
changing it between arms would be exactly the kind of floating configuration that
makes the effects unidentifiable.

## 3. Resumability: adopt their shape wholesale

`A4` requires resumable, idempotent cells. Their `run_batch_generation.py` already
solves this and is battle-tested against spot reclaims:

1. **`download_output_from_s3_if_present()` before starting** — pulls back any partial
   output a previous attempt mirrored, so an interrupted job resumes instead of
   regenerating from scratch. No-op on a fresh run.
2. **`on_progress` fires after *every batch flush*, not at the end** — because a spot
   reclaim kills the container immediately with no chance to run a final upload.
3. **One output file per model** (`generation_records__<model>.jsonl`) so parallel jobs
   against the same output dir do not race on one shared file, plus a
   `merge_batch_outputs` step.

We adapt (1) and (2) directly; for (3) our unit is the **cell**, not the model, and our
resume key is the §7.2 join tuple `(run_id, cell_id, task, subtask, item_id,
budget_level, prompt_variant, sample_idx, repeat_idx)`.

This is the single largest cost saver available to us: a crash 80% through a 2,719-item
cell must not cost that 80%.

## 4. Batching: pass the whole batch to one `generate()` call

Their `pipeline.py` `_flush()` collects `prompts = [item[4] for item in pending_batch]`
and makes **one** `LLM.generate(prompts, ...)` call, and their condition iterator is
ordered so all cells for one model come before the next — so a batch never spans a
model boundary and no flush is forced by a model change.

The reason this matters, from their own docstring: vLLM's **continuous batching** means
*"shorter sequences don't wait on the batch's longest one, unlike a padded batch."*
So hand-rolling padded batches (as their `local_transformers.py` backend must) is
strictly worse. Do not do it.

⚠️ One thing they flag as **unverified** and we must verify: `generate()` is documented
to return outputs in input order, *"not re-sorted or reordered internally"*, but their
comment says this *"hasn't been directly observed here."* We must not rely on
positional correspondence — key every record by `item_id` explicitly, and assert the
returned `RequestOutput` count matches the input count.

## 5. Version pinning is brittle — inherit the method, re-derive the versions

Their Dockerfile records that vLLM's **own precompiled** `_C.abi3.so` links against a
specific CUDA runtime, independent of what the Python `torch` package pins:

> *"Pinning the Python torch package doesn't change what CUDA runtime vllm's own
> precompiled binaries need"* — verified via
> `objdump -p vllm/_C.abi3.so | grep NEEDED` in a real container, finding
> `libcudart.so.12`, not `.so.13`.

Also recorded: vLLM 0.26.0 → `torch==2.11.0`/`torchvision==0.26.0`; vLLM 0.18.0 →
`torch==2.10.0`/`torchvision==0.25.0`/`torchaudio==2.10.0`; they used the `cu128`
wheel index. **vLLM pins torch to exact versions, not ranges.**

`A1` should run that same `objdump` check on our pinned `vllm==0.26.0` rather than
assuming the CUDA runtime, and record the result. Their two-image split for exactly
this reason independently corroborates our two-venv requirement (§3.2).

## 6. What we CANNOT reuse: their vLLM backend

`MedLLMSafety/src/medllmsafety/models/vllm_backend.py` returns
`[output.outputs[0].text for output in outputs]` — **text only**.

It therefore drops:

- **`finish_reason`** — H1 is impossible without it, and it is unreconstructable.
- **`n_gen_tokens`** — same.
- `stop_reason`, prompt token counts, per-option logprobs.

and its defaults are wrong for us: `temperature=1.0`, no `seed`, no
`enable_prefix_caching=False`, no `max_num_seqs`, no `VLLM_BATCH_INVARIANT`.

So `src/quantartifact/generate.py` is written fresh against the §7.1 record schema.
We take the *scaffolding* (lazy import, resume, flush-to-S3, one engine per cell) and
none of the generation call itself.

## 7. Statistics modules — partial reuse, and a warning they already paid for

`MedLLMSafety/src/medllmsafety/analysis/` has `mixed_effects.py` (162 lines),
`power.py` (278), `multiple_comparison.py` (57), plus `reporting/effect_sizes.py`.

Their `mixed_effects.py` docstring records the exact wall we will hit on H3:

> *"a true mixed-effects **logistic** model (GLMM) would be more principled for a
> binary outcome but is substantially slower to fit at this design's scale and **not
> natively supported by statsmodels** for crossed random effects with Wald p-values —
> documented here as a known simplification to revisit before the real run."*

They settled for a **linear probability model** via `MixedLM` + `vc_formula`.

**We are better placed than they were, and must not inherit the compromise.** Our H3
spec is `correct ~ precision * scoring + model + (1 | item)` with **model as a fixed
effect** (`EXECUTOR-PLAN.md` §4.1: 3 levels cannot support a variance estimate). So we
need **one random intercept, not crossed random effects** — which is materially easier.
Options for `A19`, to be settled then:

- `statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM` — variational Bayes
  logistic GLMM, single random intercept, native.
- GLM + cluster-robust (by item) SEs as an assumption-light companion.
- `pymer4`/`rpy2` → R `lme4::glmer` if we want the reference implementation.

The prereg already requires **paired McNemar alongside the GLMM**, which is exactly the
assumption-light check that protects us if the GLMM fit is fragile.

`power.py` and `multiple_comparison.py` (Holm) look directly reusable for `A20`.

---

## 8. LaTeX — the good news is bigger than the reuse

`insertion-stability/preprint/main.tex` is **not** reusable as a template: it is a
single-column `\documentclass[11pt]{article}` theory preprint with `amsthm` theorem
environments, targeted at arXiv. ACL requires the official two-column `acl.sty`
template, and a Word submission is a desk reject. Different document entirely.

**But it revealed the thing that actually matters: `tectonic` is installed**
(`/opt/homebrew/bin/tectonic`, v0.16.9, 44 MB cache). `insertion-stability/preprint/SUBMIT.md`
shows the compile line: `tectonic -X compile main.tex --outdir .`

This **dissolves two constraints**:

- `EXECUTOR-TASKS.md` §3.5 assumed no local LaTeX was possible because MacTeX is ~7 GB
  and would not fit. `tectonic` is a single self-contained binary that downloads TeX
  packages on demand — no MacTeX needed.
- §11.4's open assumption *"both authors can use Overleaf"* is **no longer
  schedule-critical**. Overleaf remains the right collaborative surface for a two-author
  paper, but **gate G6 no longer depends on it**.

Verified today: the official ACL template compiles locally under `tectonic` in one
pass, and `paper/main.pdf` builds at 2 pages from our stub. `make -C paper g6` now
checks all four G6 criteria mechanically — compiles, ≤8 pages, Limitations present,
and `\todo{}` confined to Results.

One gotcha found and fixed: a stub with `\bibliography{}` but no `\cite` produces
`main.bbl:4: LaTeX Error: Something's wrong--perhaps a missing \item.` Fixed with
`\nocite{*}` until Related Work cites anything.

Also worth carrying over from their `SUBMIT.md`, for `A24`: build the submission
tarball with **`COPYFILE_DISABLE=1`** on macOS, or AppleDouble `._` files ship as junk.
