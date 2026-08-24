# QuantArtifact — Execution Plan (task-level)

**Written by:** planner agent, 2026-08-23.
**Companion to:** `EXECUTOR-PLAN.md`.
**Target:** ARR submission **2026-10-12, 23:59 UTC−12 (AoE)** → NAACL 2027 primary, Findings automatic.

---

## 0. Authority, read order, and how to use these two documents

There are now two documents and they do different jobs. Read this section before anything else.

| Document | Authority over | Do not override |
|---|---|---|
| `EXECUTOR-PLAN.md` | **Scientific design.** Hypotheses, factor definitions, scoring ladder semantics, identification strategy (§4.0), annotation protocol, statistics, related-work positioning, verified facts (Appendix A), errors not to repeat (Appendix B). | Anything in §1–§4, §7, Appendix A/B. |
| **This document** | **Execution.** Locked scope, substrate, schedule, task order, acceptance criteria, interface contracts, gates. | Anything in §2, §3, §5, §6, §7. |

**Conflict rule:** where the two disagree, *this* document wins on **scope, schedule, and substrate**; `EXECUTOR-PLAN.md` wins on **scientific design**. Every intentional override is listed in §0.2 — there are eleven. If you find a conflict *not* listed there, that is a planning bug: stop and report it (§9).

**Read order for the executor, first hour:**
1. This §0, §1, §2, §3 — what you are building and where.
2. `EXECUTOR-PLAN.md` §0.1, §1.1, §2, §4.0, §4.6 — the science that constrains it.
3. This §4, §5 — the three parallel tracks and the dated schedule.
4. This §6 — start executing task `A1` and `B1` **in the same session**.

Do not read `EXECUTOR-PLAN.md` linearly front-to-back before starting. It is a 100 KB design document; the four sections above are the load-bearing ones and the rest is reference you should consult by section when a task points you at it.

### 0.1 The standing priority test (unchanged, restated because it decides everything)

> **Does this get us closer to a submittable paper by Oct 12?** Not "is this experiment interesting."

Two corollaries specific to a 7-week runway that I am adding as binding rules:

- **Rule of the always-submittable artifact.** From Sep 20 onward there must exist a compiling `paper/main.pdf` at ≤8 pages with every section written, with `\todo{}` markers permitted **only inside Results**. If the project is hit by any disaster after Sep 20, we submit something rather than nothing. This is gate **G6** and it is the single highest-value insurance policy in the plan.
- **Rule of no new science after the freeze.** After **Sep 25** (results freeze, gate G7) no new generation runs, no new analyses, no new factors. Only writing, figure polish, and compliance. Violating this is the most common way a paper misses a deadline.

### 0.2 Intentional overrides of `EXECUTOR-PLAN.md` — the complete list

Each is a decision I am making as planner, with the reason. Do not re-litigate these; report back if evidence contradicts one.

| # | `EXECUTOR-PLAN.md` says | This plan says | Why |
|---|---|---|---|
| **O1** | §9 schedule runs Aug 6 → Oct 12 (9.5 wks), "re-derive from Aug 18" | Schedule runs **Aug 24 → Oct 12 (7.0 wks)**, §5 | 50 days remain and zero code exists. Verified today. |
| **O2** | §10 cut order defers cuts to a report-back trigger | **Cuts are applied now**, §2.3 | A cut decided in week 4 has already cost the engineering that built it. Deciding now is worth more than deciding correctly later. |
| **O3** | §1.2 makes Kurtic et al. the **primary blocking replication gate** | Kurtic is a **stretch anchor**; the blocking gate is the ungated Qwen-official-checkpoint anchor (§6, task `A9`) | The Kurtic gate needs gated Llama access and can absorb a week. Its *function* is pipeline validation; Anchors A and B do that with no gated dependency and more specific per-task numbers. |
| **O4** | §6.1 leaves SMInterface-vs-rented-GPU as a day-1 decision rule | **Decided: rented persistent GPU box.** SMInterface is a documented non-goal. §3.1 | The determinism audit (§4.6) is inherently interactive — running vLLM's own test suite, trying `VLLM_DISABLED_KERNELS`, measuring per-format disagreement. A batch-only substrate with a 3600 s default kill, 30 GB default volume, torch-2.3 base image and no fan-out driver is the worst possible fit for that, and AWS GPU quota latency is days-to-weeks against a 7-week runway. |
| **O5** | §2.1 lists self-consistency k=8 as a factor, "first to cut" | **Cut. Not implemented at all.** §2.3 | ~57–60% of compute *and* the majority-vote/tie-breaking/`repeats`-footgun engineering, in support of the least novel arm, on a task (GSM8K) that is saturated as a voting target. The Appendix-A observation about `[invalid]` votes survives as a **discussion paragraph with citations and zero compute**. |
| **O6** | §2.1/§2.5 include TriviaQA (300 items, three bespoke scorers) | **Cut from scope; stretch only** (§2.4) | §2.5 already concedes it is a weak strict-extraction instrument. It is the only arm requiring bespoke scorer code, and the load-bearing negative control is the ARC log-likelihood/generative pairing, which is stronger. |
| **O7** | §2.1 marks GPTQ-8bit "secondary" and cut #2 | **Promoted to scheduled scope** for the reference condition (§2.2) | Marginal engineering cost is zero (same `llmcompressor` path) and it converts a two-level contrast into a **dose-response** (bf16 → 8-bit → 4-bit). Monotone dose-response is a far stronger internal-validity argument than a two-point comparison, for ~20% more compute on a budget where compute is not the constraint. |
| **O8** | §2.4 GSM8K = 500 items, "full 1319 if throughput allows" | **Full 1319 in the reference condition**; nested 500-item subsample in the sweep arms (§2.2) | Compute is measured-cheap and H7 makes a point about statistical resolution — running the full test set while auditing others' underpowered evals is both stronger and better rhetoric. Nesting (500 ⊂ 1319) makes the reference run reusable as the sweep's 1× level. |
| **O9** | §5 implies `lm-eval` is part of the toolchain | `lm-eval` is a **validation instrument only**, never the experimental pipeline (§3.4) | We need per-generation `finish_reason`, full text and our own schema. But we must inherit the harness's *prompts* exactly — hence the byte-for-byte **prompt parity test**, gate G2. |
| **O10** | §6 estimates "$50–100 all-in" | **$300–500 all-in** (§2.6) | The $50–100 figure counts GPU-hours-in-use only. A persistent box you SSH into bills while you edit code, and the 200 GB volume bills for two months (doubling to $0.20/GB-mo while stopped). Same decision, honest number. |
| **O11** | §3 annotation draws from "the same stratified sample" of the main runs | Annotation draws from a **dedicated stratification pilot run** (`run_id=pilot-v1`), §6 task `A8` | Annotation cannot wait for main runs, and batch composition differs between pilot and main so the generations would not be byte-identical anyway. Judge validation is a claim about the *population* of (model × precision × task) generations, not about specific strings. State this in the paper; release `pilot-v1` in the artifact. |

---

## 1. Situation as of 2026-08-23 (verified today, not assumed)

**Runway: 50 days / 7.14 weeks.** Nothing has been built.

```
$ git log --oneline          36f8090 Plan: quantization degradation artifact study   ← only commit
$ git ls-files               .gitignore, EXECUTOR-PLAN.md                            ← no code
```

Environment audit, run today:

| Check | Result | Consequence |
|---|---|---|
| Free disk (`/System/Volumes/Data`) | **7.6 GB free, 99% full** — *worse* than the 14 GB `EXECUTOR-PLAN.md` §6 measured | **New blocker.** No local model weights (already known), and now **no local MacTeX** either (~7 GB). See `A2`. |
| `uv` | not installed | task `A1` |
| Default python | **3.8.20** (Anaconda) — unusable | task `A1` |
| `hf` / `huggingface-cli` | not installed | task `B1` |
| HF token (`~/.cache/huggingface/token`, `~/.config/huggingface/`, `$HF_TOKEN`) | **absent, all three** | task `B1` — confirms §5's finding still holds |
| `pdflatex` / `latexmk` | not installed | task `A2` — Overleaf decision |
| `aws` CLI | present at `/usr/local/bin/aws` | task `B2` quota check (2 min, do it anyway) |
| `SMInterface` workspace | present; defaults re-verified: `max_run=3600`, `volume_size=30`, `framework_version="2.3"`, `instance_type="ml.g5.xlarge"`, `max_wait←max_run` under spot | Confirms §6.1's warnings are accurate. Non-goal per **O4**. |

**The single most important structural fact:** the original 9.5-week schedule assumed a human engineering track. The executor is an agent that works continuously, so *engineering* wall-clock compresses well. What does **not** compress is anything gated on another person or on external latency: the co-author's annotation hours, the Llama license grant, the external reader's read, and the ARR profile mechanics. That asymmetry is the entire reason §4 splits the work into three tracks that start on the same day.

---

## 2. Scope — LOCKED

### 2.1 Hypotheses in scope, in priority order (unchanged from `EXECUTOR-PLAN.md` §1.1)

| ID | Claim | Instrument | Priority |
|---|---|---|---|
| **H1** | Measured 4-bit degradation shrinks as `max_new_tokens` rises, because quantized models emit longer CoT and are truncated before stating an answer. Also resolves 2606.25519 vs 2504.04823 on whether quantization lengthens CoT at all. | Token-budget sweep (§2.2) + `finish_reason` + `n_gen_tokens` | **Lead. Never cut.** |
| **H2** | Fraction of quantized failures whose correct answer is present in the trace but not in final-answer position, and how much a permissive parser / judge recovers. Targets the 52% of arXiv:2606.00206. | Trace-scan scorer, **zero extra generation** (§7.4) | High — free |
| **H3** | `precision × scoring` interaction on item-level correctness. **A null is publishable and is the paper's service to the literature.** | Re-score the reference run; GLMM + McNemar | High |
| **H4** | Strict delta decomposes into competence + compliance + truncation, with CIs, under the §4.0 identification fix. | Reference run + compliance labels | High — the centrepiece table |
| **H5** | Same interaction for few-shot CoT vs zero-shot. | Zero-shot arm (9 cells) | Medium |
| **H6** | Where extraction is structurally constrained the interaction vanishes. | ARC-C same 500 items under log-likelihood **and** `_chat` generative | **Never cut** — this is the negative control |
| **H7** | Many published deltas are reported without a paired test and some sit inside their own eval's sampling noise. Framed as "no paired test," **never** as "the effects are noise." Adopt Kübler et al.'s McNemar. | Arithmetic over published numbers + our §4.5 noise floor | Medium — nearly free |
| **H8** | *(new, promoted from `EXECUTOR-PLAN.md` §4.6)* Which quantized serving configurations are bitwise reproducible under a batch-invariant engine, and what that implies for published deltas. | Determinism audit, task `A7` | Medium — **never cut**; may become a headline |

### 2.2 The locked experimental matrix

**Models (3).** `Qwen/Qwen2.5-3B-Instruct`, `microsoft/Phi-3.5-mini-instruct` (3.8B), `meta-llama/Llama-3.1-8B-Instruct`.
→ **Pre-decided substitution (do not ask, just execute):** if Llama access is not confirmed by **Sep 1**, drop Llama and use **`Qwen/Qwen2.5-7B-Instruct`** as model 3. It is ungated, **Apache-2.0** (better artifact story than Qwen-3B's `license:other`), has official Qwen GPTQ-Int4/Int8/AWQ checkpoints for the same cross-check as model 1, and preserves a 3B → 3.8B → 7B size gradient. Record the substitution in the paper's setup section either way.

**Precision (5 in reference, 3 in expensive arms).**

| Level | Reference condition | Token sweep (H1) | Zero-shot (H5) |
|---|---|---|---|
| bf16 | ✓ | ✓ | ✓ |
| GPTQ-4bit (W4A16) | ✓ | ✓ | ✓ |
| AWQ-4bit (W4A16_ASYM→normalised, §2.5) | ✓ | ✓ | ✓ |
| bnb-NF4 (in-flight) | ✓ | ✗ | ✗ |
| GPTQ-8bit (W8A16) | ✓ | ✗ | ✗ |

bnb-NF4 and GPTQ-8bit are excluded from the expensive arms deliberately, and the paper must say so with the reason: NF4 has no CUDA kernels in vLLM (its own RFC #39583 proposes moving it out-of-tree), so it earns a place in the method-generality ladder but not in a token-budget sweep; 8-bit is there for **dose-response** on the reference contrast.

**Tasks and item counts (per cell).**

| Task | Reference | Sweep / zero-shot | Notes |
|---|---|---|---|
| GSM8K | **1319** (full test) | 500, **nested prefix ⊂ 1319**, seed 20260823 | Nesting makes the reference run the sweep's 1× level for free |
| BBH × 6 | 150 each = **900** | same 900 | Subtasks: `word_sorting`, `dyck_languages`, `multistep_arithmetic_two`, `object_counting` (free-form) + `boolean_expressions` (boolean) + `logical_deduction_five_objects` (MC). **This is the format gradient — free-form → boolean → MC — and it is a prediction of the hypothesis, not six arbitrary tasks.** |
| ARC-Challenge | **500** items, run under **both** scoring modes | ✗ | Same 500 item IDs under log-likelihood *and* `arc_challenge_chat` generative. The gap between the modes **is** the extraction cost, measured directly. H6. |
| **Total generative** | **2,719 / cell** | 1,400 / cell | + 500 log-likelihood scorings (near-free) |

**Token budget (H1) — freeze these three levels before any results exist and never change them.** `1×` = the harness default per task (GSM8K 256, BBH 1024, ARC-chat 100 — verify against the pinned harness configs in `A5`, do not trust these numbers from memory); `2×`; `4×`. Record the resolved absolute value per task in `configs/tasks.yaml` and in the paper.

**Prompt format.** few-shot CoT (reference, harness-verbatim) and zero-shot.

**Decoding.** greedy only. `temperature=0`, `seed=0`.

**Scoring (4 rungs, all applied to every stored generation, zero extra generation).** `strict` → `permissive` → `math-verify` → `judge`. **The headline number is the H3 interaction at rung 1 vs rung 2** — deterministic, auditable, and independent of the judge. Rungs 3–4 are robustness. This is what makes judge validation non-blocking.

### 2.3 CUT — do not build these

| Cut | Reason | Residue to keep |
|---|---|---|
| **Self-consistency arm (k=8)** | O5 | One discussion paragraph: `lm-eval`/`lighteval` let `[invalid]` compete in the plurality vote while `bigcode-evaluation-harness` explicitly drops failures first; nine major papers specify neither rule. Cite it as an open reporting gap. **Zero compute.** |
| **TriviaQA** | O6 | Nothing. One sentence in Limitations: no factual-recall task; the structural control is the ARC mode pairing. |
| **Li et al. (2501.03035) replication** | `EXECUTOR-PLAN.md` §1.2 — a trap | Cite as a motivating example of the claim genre. The arithmetic inconsistency may be noted **only** after independently re-deriving it from the paper's own tables, phrased neutrally ("the reported delta appears inconsistent with the accuracies given in the same row"). Build no argument on it. |
| **SMInterface / SageMaker path** | O4 | Nothing. |
| **Local model inference of any kind** | GPTQ/AWQ/bnb do not run usefully on Apple Silicon; vLLM CPU backend is x86-only; 7.6 GB free disk | The Mac is a dev, analysis and writing box. |

### 2.4 STRETCH — build only if the named gate has already passed

Touch none of these before its gate. Each is genuinely additive but none is load-bearing.

| Stretch | Unlocked by | Value if we get it |
|---|---|---|
| **W8A8-INT8 arm** | G5 passed (main reference runs complete) by Sep 10 | Adds the *measured throughput winner* (1.38× per Kurtic Table 6) and a fourth dose-response point |
| TriviaQA (200 items, 3 scorers) | G4 passed and scorers done by Aug 30 | A factual-recall control |
| Kurtic et al. full replication | Llama access + G5 | Strongest citable anchor |
| Intra-annotator test–retest (20 seeded duplicates) | Annotation running ahead of schedule | Secondary reliability number |
| `Olmo-3.1-32B-Instruct` as third judge | Everything else done | Best fully-open-data reproducibility story |

### 2.5 Two design details that decide whether the quantization result replicates

Both are in `EXECUTOR-PLAN.md` but are the most likely reviewer attack surface, so they are restated here as hard requirements.

1. **Normalise the calibration budget across GPTQ and AWQ.** `llmcompressor`'s own examples differ by **8×** in token budget between the two methods (GPTQ 512×2048 symmetric W4A16; AWQ 256×512 asymmetric W4A16_ASYM). Accepting those defaults makes the GPTQ-vs-AWQ delta uninterpretable. **Use 512 samples @ 2048 seqlen, one pinned calibration set, one pinned seed, for both arms**, and state it in the paper. Record every calibration parameter in `configs/quant.yaml` and hash it into `quant_config_hash`.
2. **Self-quantize all three models with identical settings** as the primary arms; report the official Qwen and RedHatAI checkpoints as a documented cross-check (task `A9`). Do **not** mix vendor checkpoints into the main matrix — `hugging-quants/…-GPTQ-INT4` uses `desc_act:true` against Qwen's `desc_act:false`, which is a silent confound.

### 2.6 Volume, cost and time — loose estimate (satisfies `EXECUTOR-PLAN.md` §0.3 pass 1)

**Generations under locked scope:**

| Block | Cells | Items | Levels | Generations |
|---|---|---|---|---|
| Reference | 15 | 2,719 | 1 | 40,785 |
| Noise floor (bf16, 2nd seed, §4.5) | 3 | 2,719 | 1 | 8,157 |
| Determinism probe (500-item probe × 3 repeats) | 5 fmt (Qwen) + 2 fmt × 2 models | 500 | 3 | ~12,000 |
| Token sweep (2×, 4× only — 1× is the reference) | 9 | 1,400 | 2 | 25,200 |
| Zero-shot | 9 | 1,400 | 1 | 12,600 |
| **Total** | | | | **≈ 99,000** |

Plus ~107,000 judge calls (judge 1 over reference + cascaded sweep; judge 2 over reference + the human-validation sample only — see `A12`).

**Cost and calendar, stated separately as §0.3 requires:**

| Phase | GPU-hours (A100-80G) | $ @ $1.19/hr | Calendar |
|---|---|---|---|
| Setup, quantization, smoke | 8 | 10 | W1 |
| Determinism audit (H8) | 6 | 7 | W1–W2 |
| Anchors + pilot | 6 | 7 | W1–W2 |
| Reference runs (BI mode, 1.2–2× overhead) | 22–35 | 26–42 | W2–W3 |
| Token sweep | 12–20 | 14–24 | W3 |
| Zero-shot | 6–10 | 7–12 | W4 |
| Judging | 8–14 | 10–17 | W3–W4 |
| Reruns / debugging slack (40%) | 28 | 33 | — |
| **GPU compute total** | **96–127** | **$115–150** | |
| **Idle billing** on a persistent box (~180 h up) | — | **$210** | |
| 200 GB volume × 2 months (×2 while stopped) | — | **$40–80** | |
| **All-in** | | **≈ $370–440** | 7 weeks |

**Feasibility verdict, delivered now rather than on day 2:** the compute fits with an order of magnitude of headroom; the money is trivial; **the engineering fits only under the locked scope in §2.2 and the parallel-track structure in §4.** The three things that can actually break the deadline are, in order: (1) the co-author's annotation hours, (2) the Llama license grant, (3) the related-work section. All three are mitigated by starting them on day 1 — which is what Track B and Track C exist for. **Tight estimate is due at the end of `A6`** (measured generations/hour on the actual instance), and if it moves the reference-run block past **Sep 13**, that is a report-back (§9).

---

## 3. Substrate — LOCKED

### 3.1 Compute: one rented persistent GPU box

**Decided per O4. Do not evaluate SageMaker/SMInterface.** Still run the 2-minute AWS quota check (`B2`) purely so the paper's compute section is accurate and so we know what we declined.

**Spec:**
- **1 × A100-80GB**, `tensor_parallel_size=1`. Single GPU is *required* for determinism — batch invariance is broken under TP>1 (vLLM #51290). Do not rent a multi-GPU box "for headroom"; it invites a TP>1 mistake.
- Provider order: **RunPod Community ($1.19/hr)** → **Verda spot A100-40G ($0.645/hr, flat 50% spot discount, most predictable interruptible pricing)** → L40S (~$0.79/hr) if A100 supply is tight. Re-verify pricing on day 1; these are Aug-6 figures.
  - RunPod: spot offers **no discount** over on-demand right now — use on-demand. Volume storage **doubles to $0.20/GB-mo while the pod is stopped**.
  - Vast.ai only if the above fail, and then filter `verified` + `reliability2>0.98`: headline rates are cheapest but egress is per-host and invisible in the quote (median $4/TB, p90 $21/TB).
- **200 GB network volume.** Measured requirement is ~136 GB of weights (§6): Qwen2.5-3B bf16 6.17 / GPTQ-4 2.07 / GPTQ-8 3.47 / AWQ 2.69; Llama-3.1-8B bf16 15.07 / GPTQ-4 5.74 / AWQ-4 5.73 / w8a16 9.08; Phi-3.5-mini bf16 7.64 + ~9 self-quantized; judges ~34.
- ⚠️ **Llama-3.1-8B repo trap:** the repo totals **32 GB**, not 15, because it ships a duplicate `original/consolidated.00.pth`. Always pass `ignore_patterns=["original/*"]`.
- **Stop the pod when idle.** At $1.19/hr an unattended weekend is $57. Add a `scripts/box_down.sh`. (Weigh against the stopped-volume price doubling; stopping still wins.)

### 3.2 Two virtual environments — non-negotiable

Dependency conflict is hard, verified, and will silently corrupt a run:

- `venv-quant`: `llmcompressor==0.12.0.1` (pins `compressed-tensors==0.17.1` exactly, `transformers<=5.10.1`). Used for **GPTQ and AWQ both** — one tool, one calibration code path, removes a confound.
- `venv-serve`: `vllm==0.26.0`, `bitsandbytes==0.50.0`, `math-verify==0.9.0` (`pip install "math-verify[antlr4_13_2]"`), `lm-eval==0.4.12`.
- Never install `autoawq` (downgrades transformers to 4.47.1 and wrecks the env) or `auto-gptq` (both are GitHub-archived and their own READMEs redirect to `llmcompressor`/`gptqmodel`).
- Pin **everything**; commit `uv.lock`. Reproducibility here is load-bearing because of §4.6.
- `huggingface_hub` ≥ 1.0.0: the CLI is **`hf`**, not `huggingface-cli`; `use_auth_token` → `token`; `resume_download` is gone.

### 3.3 Determinism configuration — apply to every generation call, no exceptions

From `EXECUTOR-PLAN.md` §4.6. These are the settings that matter; everything else is cargo cult.

```
VLLM_BATCH_INVARIANT=1
VLLM_ENABLE_V1_MULTIPROCESSING=0        # offline
tensor_parallel_size=1                   # MOST IMPORTANT — BI broken under TP>1 (#51290)
enable_prefix_caching=False               # prefix caching breaks greedy determinism (#39389)
attention backend = FLASH_ATTN (pinned)
seed=0, temperature=0, async scheduling off
max_num_seqs=8                            # low, and REPORTED IN THE PAPER — the only knob that
                                          # resolved #51187 (44 co-resident seqs → 3 distinct
                                          # outputs in 5 repeats; max_num_seqs=8 → 5/5 identical)
```
**Do NOT set `enforce_eager=True`** — CUDA graphs became BI-compatible in vLLM PR #27660; it is pure throughput loss.

Known residual leaks to *measure* rather than assume: RMSNorm block size varies with token count, defeating BI for sequences ≥256 tokens (#48271) — squarely inside our 200–600 token output range. And the gap that lands directly on us: `auto_awq.py` **has** a batch-invariance guard (source comment: *"Marlin kernels are not batch invariant"*, so AWQ silently falls back to slower non-Marlin kernels under BI), while `auto_gptq.py` and `choose_mp_linear_kernel()` have **no BI handling at all** — GPTQ keeps using Marlin. That asymmetry is exactly what task `A7` measures and it is why H8 exists.

Budget **1.2–2× wall-clock** for BI on Ampere (SM80 must swap in Triton GEMMs; Hopper is cheaper). **Never reuse a BI-mode timing as a quantized-throughput number** — the kernel changed underneath you.

### 3.4 `lm-eval`'s role — validation instrument only (O9)

`lm-eval==0.4.12` is installed and used for exactly three things:
1. The **prompt parity test** (gate G2): assert our rendered prompt for each task is **byte-for-byte identical** to `lm-eval`'s. This is the single test that entitles us to say "we use the standard harness prompts," and it is the cheapest credibility we can buy.
2. Extracting the harness's regexes and generation configs verbatim into `configs/` — including the GSM8K `strict-match` `#### (\-?[0-9\.\,]+)`, the `flexible-extract` `(-?[$0-9.,]{2,})|(-?[0-9]+)` with `group_select: -1`, and the BBH CoT filter `(?<=the answer is )(.*)(?=.)` (**case-sensitive lookbehind** — log every capitalization-variant failure; it is a concrete, citable brittleness and a likely source of the effect).
3. Cross-validating our own log-likelihood ARC scorer against `lm-eval`'s `arc_challenge` acc/acc_norm (gate G3).

All experimental generation goes through **our** `generate.py`. Never mix vLLM and HF `transformers` generation anywhere in the study — arXiv:2605.19537 measures **16.6 pp** of benchmark movement from backend choice alone, and arXiv:2608.04714 attributes ~39% of out-of-the-box variance to the backend. One engine, one version, stated explicitly. Pass **`add_bos_token=True`** (Red Hat's tip — quantized models are BOS-sensitive).

### 3.5 Paper toolchain: Overleaf primary

No LaTeX locally and only 7.6 GB free, so MacTeX (~7 GB) does not fit. **Use Overleaf** with the current ACL template — which is also the right answer for a two-author paper — and mirror the source into `paper/` via Overleaf's git integration so it is version-controlled with the code. Install **BasicTeX** (~100 MB) + `tlmgr` only if an offline compile check is needed. **Word submissions have been desk-rejected since March 2026**; LaTeX ACL template only.

---

## 4. Three tracks, started on the same day

The schedule only works if these run in parallel. Track A is the executor's continuous work; Tracks B and C are things that will silently slip if they are treated as later phases.

```
        Aug 24 ─────────────────────────────────────────────────── Oct 12
TRACK A  ├─ setup ─ pipeline ─ scorers ─ determinism ─ pilot ─┬─ main runs ─ sweep ─ analysis ──┐
(eng.)   │                                                    └─ anchors ──┘                    │
                                                                                                │
TRACK B  ├─ HF license ─┐                                                                       │
(people, │  OpenReview  ├─ rubric + dry run ─ ANNOTATION (co-author) ─┐                          │
latency) │  GPU account │                                             └─ judge validation ──────┤
         │  ext. reader ┘                          ext. reader briefed ─────────── ext. read ────┤
                                                                                                │
TRACK C  ├─ related work ─ method ─ intro ─ limitations ─ checklist ─ prereg ── SUBMITTABLE ─────┴─ results ─ polish
(writing)                                                              (G6, Sep 20)
```

**Track B — start day 1, all four items, because each has multi-day external latency:**
- `B1` HF token + Llama license grant (manual approval; can take days; sits on the anchor path)
- `B2` OpenReview profiles for **both** authors (incomplete profile blocks reviewer registration; registration within 48 h of submission is what prevents automatic desk rejection) + hard **Oct 14** calendar reminder for both people
- `B3` GPU provider account with payment method working
- `B4` Recruit the external reader (§6 `B4`) — a third person's calendar in late September

**Track C — the counter-intuitive one.** `EXECUTOR-PLAN.md` §11 names the related-work squeeze as *the* elevated rejection risk, and calls it "a *writing* problem, which is the most controllable kind." Related Work, Introduction framing, Method, Limitations and the Responsible NLP Checklist depend on **zero results**. Drafting them in weeks 1–2 costs almost nothing and removes the largest risk in the project. Do not defer writing to week 5.

---

## 5. Schedule — 7 weeks, Aug 24 → Oct 12

Gates are hard: **do not start the next phase's expensive work until the gate's acceptance criteria in §6 are met and its evidence artifact exists.**

### W1 — Mon Aug 24 → Sun Aug 30 · foundation, determinism, pilot
`A1` env · `A2` repo+paper skeleton · `A3` configs · `A4` generation core · `A5` scorer rungs 1–3 · `A6` smoke + **tight estimate** · `A7` **determinism audit** · `A8` **stratification pilot**
Parallel: `B1`–`B4` on day 1 · `C1` related work · `C2` prereg draft
**Gates: G1** (env+access, Aug 25) · **G2** (prompt parity, Aug 27) · **G3** (scorer unit tests + ARC ll cross-val, Aug 28) · **G4** (determinism audit reported, Aug 30)

### W2 — Mon Aug 31 → Sun Sep 6 · anchors, judge, pre-registration, main runs begin
`A9` **anchor validation** · `A10` judge harness · `A11` **human annotation window opens** · `A12` judge validation · `A13` main reference runs (start) · `A14` noise floor
Parallel: `C2` **prereg COMMITTED before any main-run generation** · `C3` method section
**Gates: G5a** (anchors pass, Sep 2) · **PREREG COMMITTED, Sep 2 — hard, blocks `A13`**

### W3 — Mon Sep 7 → Sun Sep 13 · main runs complete, token sweep
`A13` reference runs complete (15 cells) · `A15` **token-budget sweep (H1)** · `A16` scoring all rungs · `A17` compliance + trace-scan labelling
**Gate: G5** (all reference cells generated and scored, Sep 13)

### W4 — Mon Sep 14 → Sun Sep 20 · zero-shot, analysis, SUBMITTABLE DRAFT
`A18` zero-shot arm · `A19` GLMMs + decomposition + Manski bounds · `A20` powered-ness audit · `A21` figures and tables
Parallel: `C4` intro · `C5` limitations + Responsible NLP checklist · **`C6` assemble submittable draft**
**Gate: G6** (**Sep 20 — `paper/main.pdf` compiles, ≤8 pages, every section written, `\todo{}` only in Results**)
Second dated **novelty scan** (Sep 14, per `EXECUTOR-PLAN.md` §10.4 — re-run before drafting)

### W5 — Mon Sep 21 → Sun Sep 27 · results freeze, results+discussion
`A22` final analysis pass, all numbers locked · `C7` Results + Discussion + Abstract · `C8` related-work revision against the Sep 14 scan
**Gate: G7** (**Fri Sep 25 — RESULTS FROZEN.** No new runs, no new analyses after this.)
Complete draft to the external reader **Wed Sep 30**.

### W6 — Mon Sep 28 → Sun Oct 4 · external review
Draft out Sep 30 · reader attacks **related-work differentiation** and **judge validation** specifically (that targeted ask is realistic for a 4-day window; a full read is not) · begin revision
**Gate: G8** (external feedback received, Oct 4)

### W7 — Mon Oct 5 → Sun Oct 11 · revision, compliance, submit
Revise Oct 5–8 · **freeze Thu Oct 8** · `A23` compliance sweep against every desk-reject ground · `A24` artifact release prep · **submit Fri Oct 9 or Sat Oct 10**
Third dated **novelty scan** (Oct 5)
**Gate: G9** (submitted, Oct 10). Oct 11–12 is buffer for platform failure only — **not for science**.

### Post-submission — non-optional
- **Wed Oct 14:** both authors register as ARR reviewers/ACs. Missing this is an **automatic desk reject**; the obligation is per-author, not per-paper.
- Metadata stays editable for 48 h. Withdrawing >48 h after the deadline blocks resubmission until the *second* following cycle.
- **Dec 20, 2026:** commitment deadline (ARR lists Dec 20, NAACL's own site Dec 23 — plan to Dec 20, re-verify in November).
- If not accepted: the **Jan 2027 cycle feeds ACL 2027** (Kyoto, Aug 17–22). A meta-review score ≥2.0 permits next-cycle resubmission. Oct 12 is **not** one-shot.

---

## 6. Tasks — specifications and acceptance criteria

Format: **`ID` — what · *done when* · → artifact.** A task is not done until its artifact exists in the repo.

### Track B — start all four on day 1

**`B1` HF access.** `uv tool install "huggingface_hub[cli]"` → `hf auth login` (token, `read` scope) → `hf auth whoami` → `hf download meta-llama/Llama-3.1-8B-Instruct --include config.json`.
*Done when:* `config.json` downloads (confirms token **and** license grant), token is exported as an env var on the GPU box (**never** committed to the repo), and the result is recorded in `notes/access.md`.
**On 401/403: request the license in a browser the same hour**, note the request timestamp, and proceed with Qwen + Phi. If not granted by **Sep 1**, execute the O-2.2 substitution to `Qwen2.5-7B-Instruct` without asking.
→ `notes/access.md`

**`B2` ARR + AWS.** Verify **both** authors' OpenReview profiles are complete: affiliation, Semantic Scholar / DBLP / ACL Anthology links, working email. Set an **Oct 14** reminder for both people. Separately run `aws sts get-caller-identity` and one GPU vCPU quota query — read-only, 2 minutes, purely to document what we declined per O4.
*Done when:* both profiles verified complete, both reminders set, quota result recorded.
→ `notes/arr-obligations.md`

**`B3` GPU account.** Provider account + working payment + one successful 5-minute test pod that runs `nvidia-smi`.
*Done when:* a pod has been launched and destroyed successfully.
→ `notes/compute.md`

**`B4` External reader.** Identify and ask **two** candidates (one will decline) who know quantization *or* evaluation methodology and are not the co-author. Tell them now: **draft arrives Sep 30, feedback needed by Oct 4**, and the ask is a targeted attack on related-work differentiation and judge validation, not a full read.
*Done when:* one has confirmed in writing.
→ `notes/review-plan.md`

**`B5` Annotation rubric (day 2–4, needs the co-author).** Draft the rubric; dry-run ~10 items **together**; freeze it. Must give an explicit decision rule for every edge case: truncated output, multiple candidate answers, correct-but-never-stated, refusals, degenerate repetition. Rubric drift between raters is the main threat to a two-rater design. **Pre-specify the adjudication rule before seeing any disagreements** — never resolve by deferring to the senior author (Belz et al., Insights 2023, list biased aggregation as a documented reproducibility failure).
*Done when:* frozen, committed verbatim, and both annotators have signed off on the 10-item dry run.
→ `annotation/rubric.md` (published verbatim in the appendix and artifact)

### Track A — W1

**`A1` Environment.** Install `uv`; create `venv-quant` and `venv-serve` per §3.2 with everything pinned; commit `uv.lock`. Also: report the top local disk consumers to the user and **let them decide** what to remove — 7.6 GB is tight for even a minimal analysis env. Do not delete anything on the user's machine.
*Done when:* both venvs resolve, `uv.lock` committed, `python -c "import vllm, math_verify, lm_eval"` passes in `venv-serve`.

**`A2` Repo + paper skeleton.** Package layout per `EXECUTOR-PLAN.md` §5. Create the Overleaf project from the current ACL template, wire its git remote into `paper/`, and commit a compiling stub with every section header from §7 present and empty.
*Done when:* `paper/main.pdf` compiles from the stub.

**`A3` Configs.** `models.yaml` (repo IDs + **commit SHAs, not tags**), `quant.yaml` (method/bits/calibration set/size/seqlen/seed per cell, normalised per §2.5), `tasks.yaml` (item counts, subsample seeds, prompt templates, the three resolved token-budget levels), `scoring.yaml` (the four rungs declaratively).
*Done when:* all four exist, every model pinned by SHA, the GSM8K 500-item subsample is verified to be a **prefix-nested subset** of the 1319, and the three budget levels are recorded as absolute integers.

**`A4` Generation core.** `generate.py` — the only component that touches a GPU for generation. Emits the §7.1 record for every generation. One vLLM engine per `(model, precision)`. All §3.3 settings applied and asserted at startup (fail loudly if `VLLM_BATCH_INVARIANT` is unset or `tensor_parallel_size != 1`). Append-only gzipped JSONL, one file per cell. Resumable: re-running a partially-complete cell must skip already-generated `(item_id, sample_idx, budget_level)` keys.
*Done when:* a 20-item run produces a valid JSONL where **every** record has non-null `finish_reason` and `n_gen_tokens`, and a re-run of the same command generates zero new records.

**`A5` Scorers, rungs 1–3.** `scoring/extract.py` — pure functions, no I/O. Rung 1 = the harness regexes **verbatim**. Rung 2 = the documented pre-registered permissive parser (last-number-in-string, `\boxed{}`, "answer is X", comma/currency/unit stripping, Minerva `normalize_final_answer`). Rung 3 = `math-verify` 0.9.0 — ⚠️ `parse()`/`verify()` use `signal.alarm()` and **raise in worker threads**: pass `parsing_timeout=None, timeout_seconds=None` and impose your own timeout; it is intentionally **asymmetric** so gold/target order matters.
Also `scoring/compliance.py` for the §2.3 2×2 (`COMPLIANT` / `RECOVERABLE` / `ABSENT` / `TRUNCATED`, where `TRUNCATED` is recorded *separately* because a generation can be both truncated and absent) and the H2 trace-scan of §7.4.
*Done when:* **G3** — unit tests cover, at minimum: `####`-present, `####`-absent, boxed answer, comma-thousands, currency, negative number, trailing period, "The answer is" **capitalized** (must fail rung 1 for BBH — assert this explicitly, it is a finding), truncated mid-answer, degenerate repetition, empty generation, multiple candidate answers, and a `math-verify` timeout path. Rungs are asserted **monotone** on a fixture set (anything rung *n* marks correct, rung *n+1* also marks correct) — any violation is a bug or a genuine finding and must be triaged, not ignored.

**`A6` Smoke + tight estimate.** One cell end-to-end (Qwen2.5-3B, bf16, GSM8K 100 items). Then `vllm bench throughput` **on the actual instance** — published estimates are ±40% and no public measured figure exists for a 3B model.
*Done when:* measured generations/hour recorded **with and without** `VLLM_BATCH_INVARIANT=1` (this is also the honest BI-overhead number for the paper), and the §2.6 table is re-derived from measurements and committed.
→ `notes/estimates.md`. **Report back immediately if the reference-run block now lands past Sep 13.**

**`A7` Determinism audit (H8) — before any main run.** Quantize Qwen2.5-3B into all four quantized formats. Run vLLM's own `tests/v1/determinism/test_batch_invariance.py` per format. Then rerun an identical 500-item probe (GSM8K 200 + BBH 200 + ARC-chat 100) **3×** per format and measure the bitwise disagreement rate. If GPTQ is nondeterministic while bf16 is not, **differential numerical noise exists between arms and is a floor on every claim in the paper.** Workaround to test if so: `VLLM_DISABLED_KERNELS=MarlinLinearKernel,MacheteLinearKernel` (slower fallback kernels — does it restore determinism?). Extend to Llama/Phi × {bf16, GPTQ-4} in W2 as confirmation; do not do all 3 models in W1.
*Done when:* **G4** — a committed table of per-format bitwise disagreement rate over 3 repeats, the vLLM test-suite pass/fail per format, and the `VLLM_DISABLED_KERNELS` result where relevant.
→ `artifacts/tables/determinism.csv` + `notes/determinism.md`. **This gates everything downstream** — do not spend compute on runs whose deltas you cannot trust. It is also a self-contained publishable result no one has reported, so write it up as such.

**`A8` Stratification pilot (`run_id=pilot-v1`).** All 15 cells × 300 items (GSM8K 100 + BBH 120 + ARC-chat 80), using the **final frozen** prompt templates and budgets — if templates are not frozen, the annotation population is wrong. This one run serves three purposes: the annotation source pool (O11), a throughput measurement on real cells, and a noise-floor smoke test.
*Done when:* 4,500 generations stored and scored at rungs 1–3, and the stratified 250-item annotation sample is drawn and written **blinded** per `A11`.

### Track A — W2

**`A9` Anchor validation (the retargeted replication gate, O3).**
- **Anchor A — mandatory, ungated, blocking.** Reproduce the official Qwen-published `Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4` / `-GPTQ-Int8` / `-AWQ` checkpoints' behaviour and compare against our self-quantized equivalents at matched settings (`group_size:128, desc_act:false`). *Pass:* our self-quantized GSM8K accuracy is within **±2.0 pp** of the official checkpoint's at the same precision. A larger gap means our calibration or serving config is wrong — debug before proceeding.
- **Anchor B — mandatory if Llama access exists, non-blocking.** `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16` publishes measured accuracies (GSM8K 82.8→82.9, ARC-C 81.4→80.2, BBH 30.1→28.9). *Pass:* our deltas land near theirs; a wild miss means the harness config is wrong.
- **Anchor C — secondary, strategically valuable.** Reproduce the strict-vs-flexible GSM8K gap from arXiv:2601.14277's Appendix B on Llama-3.1-8B-Instruct (they report F16: flexible 77.63 vs strict 24.64; gaps of 41–64 points across 14 llama.cpp quantizations). If we reproduce their gap, **our pipeline is validated on exactly the phenomenon the paper is about** — this is the most rhetorically useful anchor we have.
- **Anchor D — stretch only.** Kurtic et al. (arXiv:2411.02355).
*Done when:* **G5a** — a committed table of anchor / published value / our value / delta / verdict.
→ `artifacts/tables/anchors.csv`

**`A10` Judge harness.** Primary `IAAR-Shanghai/xVerify-9B-I` (purpose-built for `{question, llm_output, correct_answer}` → binary; beats GPT-4o at 97.19 vs 96.25 F1; family-clean w.r.t. Qwen/Phi/Llama). ⚠️ **CC-BY-NC-ND-4.0 — run it and report numbers, but never redistribute the weights or any derivative in our artifact.** Second judge `ibm-granite/granite-4.1-8b` (Apache-2.0, IFEval 87.06) — cleanest license for the artifact release.
Use CompassVerifier's **3-way label space** (`A`/`B`/`C` with an explicit `INVALID` class) rather than a pure binary — our 3B models under 4-bit *will* emit truncated and repetitive output, and `INVALID` maps directly onto `ABSENT`/`TRUNCATED`. Borrow its "compare **ONLY** the final answer" instruction. Always supply the reference answer (reference-guided judging cuts GPT-4's math-grading failure rate from 70% to 15%). Greedy, seed fixed, single-token verdict.
*Done when:* the exact prompt template is validated on ~50 items **before scaling** and the judge's own **output-parse rate** is measured — VerifyBench found xVerify failing to emit its required format under a foreign template, so this check is not optional.

**`A11` Human annotation window (co-author dependency — the schedule's tightest constraint).** ≥250 items stratified across 3 models × {bf16, 4-bit} × all tasks, **oversampling rung-1/2/3 disagreement cases** (the agreement statistic only matters where scorers disagree; also report agreement on the natural distribution). ≥120 double-annotated for IAA; split the remainder to reach 250; report per-annotator counts.
**Blinding is load-bearing, not hygiene.** **Omit the condition column from the annotation file entirely** rather than hiding it — a hidden column can be un-hidden. Annotators must never see the judge's verdict, the parser's verdict, or the precision condition. Pre-shuffle **independently per annotator**, assert the two orderings actually differ, publish the seeds. One binary question per screen. A ~10-item blind practice round instead of attention checks (excluding check-failures yields a biased subsample — van der Lee et al., INLG 2019).
Justification to cite: Hróbjartsson et al. (*BMJ* 2012;344:e1119) — 21 trials with binary subjective outcomes, non-blinded assessors exaggerated odds ratios by **36%** (pooled ROR 0.64, 95% CI 0.43–0.96), and the entire bias came from misclassifying a median of **3%** of items while agreeing on 78%. Our effects are small enough that a bias hiding behind high agreement could manufacture the whole finding.
**Never pre-annotate with the judge and have humans correct it.** Anchoring is measured (Fort & Sagot, LAW IV 2010: the better annotator dropped 98.1→95.8 under low-quality pre-annotation). Note the trap: Mikulová et al. (LREC 2022) found pre-annotation *raised* IAA — **higher agreement under pre-annotation is shared anchoring, not higher validity**, and IAA is one of our headline numbers.
Tooling: a ~50-line local script (render question + generation, collect one keystroke, append JSONL) is genuinely the right answer for 250 binary labels. If a platform is wanted, **Potato** (`judge_calibration` mode, `/adjudicate` with `show_annotator_names: false`; cite Pei et al., EMNLP 2022 Demos) — avoid its `pre_annotation` block, and pre-shuffle per annotator yourself. **Label Studio Community will not work** (agreement matrices, review workflow and per-task assignment are Enterprise-only). Prodigy costs $390–1,950, computes no agreement statistics, and shows annotator names.
*Done when:* 250 adjudicated labels, IAA reported on the overlap subset, the number of adjudicated items reported, and the annotation design disclosed in Responsible NLP §D (instructions, recruitment, that annotators were unpaid co-authors rather than crowdworkers).
→ `annotation/labels.jsonl`, `annotation/iaa.md`

**`A12` Judge validation (gate).** Report **all** of: raw agreement with a **Wilson 95% CI**, Cohen's κ, **Gwet's AC1** (the right primary statistic here), and the confusion matrix with FP and FN **broken out separately**. Do **not** report κ alone — these labels are heavily skewed and a judge at 95% raw agreement can post κ≈0.3 purely from prevalence (the kappa paradox: Feinstein & Cicchetti 1990; Byrt, Bishop & Carlin 1993).
**The check a good reviewer will ask for:** report judge error rates **separately for bf16 vs 4-bit generations**. A judge that is lenient toward *quantized* outputs specifically would manufacture our result.
*Gate:* proceed with the judge as a primary rung if raw agreement ≥0.90 **and** AC1 ≥0.7 **and** the bf16-vs-4-bit error rates are not significantly different. Also report IAA alongside, because a judge cannot be held to better agreement than the humans achieve with each other.
*If it fails:* report rungs 1–3 as primary and the judge as a caveated robustness check. **This keeps the paper intact and is therefore NOT a blocking report-back** — the headline is already rung 1 vs rung 2 (§2.2).
Judge coverage: judge 1 over the full reference condition + a **documented cascade** on the token sweep (judge only where rung2 ≠ rung3, or rung2 = incorrect); judge 2 over the reference condition + the human-validation sample only. Report the cascade rule in the paper.
→ `artifacts/tables/judge_validation.csv`

**`A13` Main reference runs.** 15 cells × 2,719 items. **Blocked until the pre-registration is committed** (`C2`, Sep 2). Score continuously as generations land — do not batch scoring to the end.

**`A14` Noise floor (§4.5, mandatory).** Run the bf16 baseline **twice** end-to-end with different seeds on every model. The spread between the two runs is the pipeline's own noise floor, and **any effect smaller than that floor cannot be claimed by us either.** Reporting that is exactly the self-application of the paper's own standard that earns reviewer trust.
→ `artifacts/tables/noise_floor.csv`

### Track A — W3–W4

**`A15` Token-budget sweep (H1, the lead).** 9 cells × 1,400 items × {2×, 4×} (1× comes free from the reference run via the nested subsample). Follow arXiv:2608.04160's design: **freeze the levels before looking at results**, hard-cap decodes at each level, treat the cap as an **independent variable** rather than an implementation detail.
*Done when:* every generation has `finish_reason` and `n_gen_tokens`, and truncation rate × budget × precision is tabulated. Also settle the 2606.25519-vs-2504.04823 contradiction directly: report mean and median `n_gen_tokens` by precision, with a CI. That is a citable contribution on its own.

**`A16`/`A17` Score everything at all four rungs; apply compliance and trace-scan labels.** Pure functions over stored artifacts, no GPU. Scorer outputs go to separate files keyed by `(cell, scorer_name, scorer_version)`; **bump the version on any change, never overwrite.**

**`A18` Zero-shot arm (H5).** 9 cells × 1,400 items.

**`A19` Analysis.** Per `EXECUTOR-PLAN.md` §4, and **§4.0's identification fix is mandatory and must already be pre-registered**:
1. Primary competence contrast on the **always-compliant stratum only** (items where *every* arm produced a committed answer) — a pre-treatment-defined subgroup, so the contrast is identified. **Report the stratum's size as a first-class number.**
2. The **2×2 compliance-transition matrix** per precision pair (committed→committed, committed→abstained, abstained→committed, abstained→abstained). This *is* a finding, not a nuisance.
3. **Manski bounds** on the full-sample competence effect. Wide bounds are an honest result.
H3 model: `correct ~ precision * scoring + model + (1 | item)`, fit per task. **Model is a FIXED effect — 3 levels cannot support a variance estimate.** Report the interaction as an odds ratio *and* as a marginal effect in accuracy points with a 95% CI. Report **paired McNemar** on discordant pairs alongside the GLMM as an assumption-light companion (and because Kübler et al., ICLR 2026, show item-level paired tests can confidently attribute degradations as small as 0.3%). Holm–Bonferroni within each hypothesis family; one pre-specified omnibus contrast per hypothesis, not exhaustive pairwise slicing; effect sizes with CIs for everything and **never a bare significance flag**.

**`A20` Powered-ness audit (H7).** For each published degradation figure we cite, compute the 95% CI implied by that paper's own eval size and state whether the reported delta clears it. Also compute our own minimum detectable effect at 80% power. **Scrupulously neutral wording — this is an observation about statistical resolution, not an accusation.** Frame as "many papers report aggregate deltas without a paired test," never as "the deltas are noise."

**`A21` Figures and tables.** The §4.2 decomposition table is the paper's centrepiece — design it **first** and let it drive what gets logged. Near-final before drafting starts.

### Track A — W7

**`A23` Compliance sweep.** Check against **every** desk-reject ground: LaTeX ACL template, ≤8 pages, **Limitations section present** (absence = automatic desk reject, costs zero page budget), complete and **correct** Responsible NLP Checklist (incorrect or incomplete filings have themselves been a desk-reject ground since Dec 2024 — §C compute/hyperparameters/error bars and §E AI assistance both apply to us), anonymized submission, no anonymous repo on Dropbox or Google Drive (download tracking), appendices in double-column format carrying **no load-bearing content**.
**arXiv preprinting is permitted** (no ARR anonymity period since Feb 2024). Do **not** elect the optional binding no-preprint option unless the user asks — if elected, preprinting before meta-reviews is a desk reject.

**`A24` Artifact release.** Pipeline, quantization configs with full calibration details, all four scorers, analysis scripts, **the full generation JSONL** (the genuinely reusable artifact — it lets anyone re-score with a new extractor without a GPU), `pilot-v1` included, and the human-annotated judge-validation set with its agreement statistics (which stands alone as a citable judge-reliability reference for quantization eval). Pin model commit SHAs, every library version, calibration dataset and counts, all decoding hyperparameters, GPU model, CUDA and driver versions. **Do not redistribute xVerify weights or derivatives** (CC-BY-NC-ND). Note Qwen2.5-3B is `license:other`/qwen-research, **not** Apache-2.0.

### Track C — writing

**`C1` Related work (W1–W2, highest-value writing task in the project).** Differentiate precisely against every paper in `EXECUTOR-PLAN.md` §7 and §0.1. Non-negotiable positioning:
- Open with the **live contradiction** — arXiv:2606.25519 (4-bit preserves accuracy while lengthening CoT) vs Liu et al., COLM 2025, arXiv:2504.04823 (quantized models do *not* produce longer outputs). Neither swept the generation cap. That is the puzzle H1 resolves.
- **arXiv:2606.00206** — the 52%-recoverable-failure finding. Credit clearly; our addition is the measurement question they did not ask.
- **Hua et al., EMNLP 2025 (arXiv:2509.01790)** — we are structurally the sequel; say so. **Engage their MATH null result honestly** (0.9593 vs 0.9647, because MATH extraction already uses sympy symbolic equivalence): well-engineered verifiers are already robust, so a blanket "extraction is fragile" claim is vulnerable and our task choice must be justified on that basis.
- **Kurtic et al., ACL 2025** — already published "degradation was overstated," via untuned quantization hyperparameters. **Cite prominently and early; do not let a reviewer discover it.** We supply an orthogonal mechanism.
- **"Accuracy is Not All You Need" (NeurIPS 2024, arXiv:2407.09141)** and **Marchisio et al. (Findings EMNLP 2024, arXiv:2407.03211)** — the field's centre of gravity runs *against* us: the dominant finding is that benchmarks **understate** quantization damage (a 1.7% automatic drop in Japanese was **16.0%** to human raters). **This belongs in the introduction, not the limitations.** Both directions are true simultaneously: accuracy deltas can over-penalize formatting *and* under-detect behavioural change. Saying that plainly is a strength.
- **Frame the contribution as *calibrating* extraction, not loosening it.** Ren et al. (arXiv:2608.02442) find 8.2–44.1% of correct-graded answers are shortcut-hacked — lenient extraction *over*-credits too. A paper that only argues "be more lenient" walks into this literature.
- **Jin/Lee et al., IJCAI 2025 (arXiv:2409.11055)** — quantization disrupts instruction-following. Strongest counterargument; our compliance/competence split is the direct answer.
- **arXiv:2605.19537 and arXiv:2608.04714** — backend choice alone moves scores up to 16.6 pp / explains ~39% of variance. Cite when stating we hold the backend fixed. **The serving axis is a confound to pin, not a novelty axis** (§0.1a).
- Plus: arXiv:2601.14277 (published the strict/flexible numbers without interpreting them — be generous and factual, not dismissive), Jo et al. (arXiv:2510.14773), arXiv:2608.03854, arXiv:2607.09999 (unrefereed, 0 citations, tests **only** FP32/FP16/NF4 — supports no GPTQ/AWQ claim; full title ends *"and Failure Mode Shifts"*), UniComp (arXiv:2602.09130, **six** techniques per v5), SLMQuant (arXiv:2511.13023, peer-reviewed), "Quantized but Deceptive?" (2025.emnlp-main.1548 — **truthfulness only, not safety**), Schaeffer et al. "Mirage" (NeurIPS 2023, methodological template — cite the citation count with the index named or not at all), FormatSpread (ICLR 2024), xFinder (ICLR 2025), EleutherAI "Lessons from the Trenches" (arXiv:2405.14782), and the HF Open LLM Leaderboard v2 blog (models scoring **0 on GSM8K purely from output-format mismatch** — a perfect real-world instance of our mechanism).
- **Three claims we must NOT make:** first to combine LLM-judge with quantization eval (≥4 papers predate us, earliest 2024); first factorial quantization evaluation (arXiv:2606.29581 has *Factorial* in its title, 144 configurations, ~2M responses); first to show accuracy hides behavioural divergence in compressed models (Dutta et al., NeurIPS 2024, owns "flips").
- **Pre-commit:** "small models are genuinely more fragile to quantization" is replicated by ≥4 independent groups. We report the **residual effect after protocol correction**, not merely its attenuation — "protocol accounts for X of the gap, and Y remains real" is more credible *and* more useful than a debunking.
- Monitor **SLiM-Eval** (OpenReview `ssrs45kran`) — nearest neighbour on model band.

**`C2` Pre-registration — COMMIT BEFORE ANY MAIN-RUN GENERATION (hard, Sep 2).** `analysis/preregistration.md` must contain, at minimum: the §4.0 identification fix in all three parts; the frozen token-budget levels as absolute integers; the rung-2 permissive parser rules **enumerated**; the H3 model specification; the omnibus contrast per hypothesis and the Holm–Bonferroni families; the judge gate thresholds and the pre-specified fallback; the annotation adjudication rule; the noise-floor rule (no effect smaller than the floor will be claimed); and the H2 trace-scan definition including its false-positive guard. **Retro-fitting this after results is worth far less** — the value is entirely in the timestamp, and §4.0 in particular caps acceptance at ~25% if a reviewer sees it was patched rather than pre-specified.

**`C3`–`C8`** Method (W2) · Introduction (W4) · Limitations + checklist (W4) · **submittable draft assembly (W4, gate G6)** · Results + Discussion + Abstract (W5) · related-work revision against the Sep 14 novelty scan (W5).

**Working title direction** — frame as decomposition/attribution, never debunking: *"How Much of Reported Quantization Degradation Is Measurement? Decomposing Accuracy Loss into Competence, Compliance, and Truncation."*

**The precise claim, and the only version still open after Kurtic et al.:**
> *The residual reported degradation, after quantization-side hyperparameters are properly tuned, is further attributable to evaluation-protocol factors — and protocol variance exceeds bit-width variance on reasoning tasks at 1–8B.*

**Terminology to define explicitly** (real hazards): `pass@k` = solved if *any* of k is correct (oracle ceiling); `maj@k`/`cons@k` = single plurality-voted answer; `avg@k` = mean per-sample correctness, i.e. a variance-reduced pass@1 estimate and **not** a voting metric; `rm@k`/best-of-n = reward-model reranking. DeepSeek-R1 **redefines "pass@1" as a mean over k samples**, and its own numbers differ between arXiv versions — cite the version you mean.

---

## 7. Interface contracts

Fix these on day 1. They are what makes the analysis re-runnable without a GPU, what makes the artifact release valuable, and what lets me verify results without reading your code.

### 7.1 Generation record — extends `EXECUTOR-PLAN.md` §5

Fields marked **★** are additions I am requiring; each is unreconstructable after the fact.

```jsonc
{
  "run_id": "main-v1",                     // ★ pilot-v1 | main-v1 | determinism-v1 | anchor-v1
  "cell_id": "qwen2.5-3b|gptq-4bit",
  "model_repo": "...", "model_revision": "<commit sha>",
  "quant_method": "gptq", "bits": 4, "quant_config_hash": "...",
  "task": "gsm8k", "subtask": null,        // ★ subtask REQUIRED for bbh
  "item_id": "...", "gold": "18",
  "prompt_variant": "fewshot_cot",         // fewshot_cot | zeroshot
  "scoring_mode": "generate",              // ★ generate | loglikelihood  (ARC needs both)
  "budget_level": "1x", "max_gen_toks": 256, // ★ both, explicitly
  "decoding": "greedy", "sample_idx": 0, "seed": 0,
  "repeat_idx": 0,                         // ★ for the determinism probe's 3 repeats
  "prompt_text": "...", "prompt_sha256": "...",
  "generation_text": "<FULL text, NEVER truncated for storage>",
  "n_prompt_tokens": 812, "n_gen_tokens": 256,
  "finish_reason": "length",               // REQUIRED — truncation analysis is impossible without it
  "stop_reason": null,
  "option_logprobs": null,                 // ★ for scoring_mode=loglikelihood: per-option logprob + length
  "engine": "vllm==0.26.0",
  "batch_invariant": true,                 // ★ was VLLM_BATCH_INVARIANT set
  "max_num_seqs": 8,                       // ★ reported in the paper; #51187
  "disabled_kernels": null,                // ★ VLLM_DISABLED_KERNELS if set
  "gpu": "A100-80GB-PCIe", "cuda": "12.8", "driver": "...", // ★ reproducibility holds only within identical hardware
  "env_hash": "...", "wall_ms": 1830
}
```

### 7.2 Score record
```jsonc
{ "run_id": "...", "cell_id": "...", "task": "...", "subtask": null, "item_id": "...",
  "budget_level": "1x", "prompt_variant": "...", "sample_idx": 0, "repeat_idx": 0,
  "scorer_name": "permissive", "scorer_version": "1.0.0",
  "extracted": "18", "correct": true,
  "compliance": "COMPLIANT",               // COMPLIANT | RECOVERABLE | ABSENT
  "truncated": false,                      // orthogonal to compliance
  "gold_in_trace": false, "gold_in_trace_conservative": false,  // H2
  "judge_verdict": null, "judge_parse_ok": null, "judge_model": null,
  "notes": null }
```
The join key across generations and every scorer is
`(run_id, cell_id, task, subtask, item_id, budget_level, prompt_variant, sample_idx, repeat_idx)`.
Enforce it as a uniqueness constraint in the loader and fail loudly on a duplicate.

### 7.3 CLI contract — one job = one command, resumable, idempotent
```
qa generate --cell <cell_id> --task <task> --budget 1x --variant fewshot_cot --run-id main-v1
qa score    --run-id main-v1 --scorer permissive
qa judge    --run-id main-v1 --judge xverify --cascade none|rung23
qa determinism --format gptq-4bit --repeats 3
qa analyze  --hypothesis h3 --task gsm8k
qa report   --table decomposition
```
Every command: re-running with identical arguments produces zero new work and exits 0.

### 7.4 H2 trace-scan — get the false-positive guard right

Searching a trace for the gold string is naively broken: gold `18` matches a trace about "18 apples" that concluded `24`. Implement **both** variants and report both:
- **permissive:** normalised gold appears anywhere in the generation.
- **conservative:** normalised gold appears in the final 20% of the generation **or** after a commitment phrase ("the answer is", "therefore", "so", `\boxed{`), **and** the rung-2 extraction ≠ gold.

**Required calibration, or the number is meaningless:** report the base rate of incidental gold appearance in traces where extraction *succeeded and was wrong* — that is the false-positive floor for this measurement. Without it a reviewer discounts the whole H2 result, and it is the specific measurement question we are claiming arXiv:2606.00206's authors did not ask, so it has to be airtight.

---

## 8. Verification gates — what I check, and the evidence you must produce

This is my job per the handoff. **Report each gate to me with its artifact the day it closes.** I will check the numbers, not the code.

| Gate | Due | Evidence artifact | What I check | If it fails |
|---|---|---|---|---|
| **G1** env + access | Aug 25 | `notes/access.md`, `uv.lock`, `notes/compute.md` | Llama `config.json` downloaded or license requested with a timestamp; both venvs resolve; a pod launched | Llama pending → Qwen+Phi first, substitution armed for Sep 1 |
| **G2** prompt parity | Aug 27 | test output | **Byte-for-byte** equality vs `lm-eval` for all 8 task configs | Blocking — a prompt mismatch invalidates every comparison to published numbers |
| **G3** scorers | Aug 28 | test report + ARC ll cross-val | All §`A5` fixtures pass; rung monotonicity holds; our ARC log-likelihood acc/acc_norm within **0.5 pp** of `lm-eval`'s on one bf16 model | Blocking — scorers are the instrument |
| **G4** determinism (H8) | Aug 30 | `artifacts/tables/determinism.csv` | Per-format bitwise disagreement over 3 repeats; vLLM test-suite result per format; `VLLM_DISABLED_KERNELS` outcome | **Not necessarily bad news** — may become a headline result. But it changes what the paper can claim, so report the numbers and let me decide the framing |
| **PREREG** | **Sep 2** | `analysis/preregistration.md` committed | Every §`C2` item present, especially all three parts of the §4.0 identification fix | **Hard block on `A13`.** Its entire value is the timestamp |
| **G5a** anchors | Sep 2 | `artifacts/tables/anchors.csv` | Anchor A within ±2.0 pp | Blocking — debug calibration and serving config before spending on main runs |
| **G5** reference runs | Sep 13 | JSONL + score files, 15 cells | Every record has `finish_reason` and `n_gen_tokens`; join-key uniqueness holds; noise floor reported | Cut the stretch list, then the zero-shot arm. **Never** the token sweep, H6 controls, noise floor, or determinism audit |
| **G6** submittable draft | **Sep 20** | `paper/main.pdf` | Compiles, ≤8 pages, every section written, `\todo{}` **only** in Results, Limitations present | Highest-priority recovery in the whole plan. Stop analysis and write |
| **G7** results frozen | **Sep 25** | `artifacts/tables/*` final | All numbers locked; no run after this date | — |
| **G8** external review | Oct 4 | reader's notes | Related-work differentiation attacked; judge validation attacked | Reader silent → the co-author does a hostile read against §7's list |
| **G9** submitted | Oct 10 | OpenReview confirmation | Submitted with 2 days of buffer remaining | — |

**Standing weekly report to me, every Sunday.** Five lines, no prose:
```
Week ending <date>
GATES: <closed> / <open> / <slipped>
GENERATIONS: <n> stored, <n> scored, <cells complete>/15
BLOCKERS: <what, and what you need from a human>
NUMBERS THAT MOVED: <any result that changes the paper's framing>
NEXT WEEK: <the 3 things>
```

---

## 9. Report-back triggers — refined from `EXECUTOR-PLAN.md` §10

**Proceed autonomously on everything else. The decisions in these two documents are made.**

**Report immediately (stop and wait):**
1. **G2 or G3 fails** and is not fixable within a day — the instrument is broken and everything downstream is worthless.
2. **PREREG would slip past Sep 2** — this is a one-way door; its value is the timestamp.
3. **Tight estimate (`A6`) pushes reference runs past Sep 13** — with proposed cuts. Cut order: stretch list → zero-shot arm → BBH MC contrast task → GSM8K back to 500. **Never cut** the token-budget arm (H1, the lead), the H6 negative controls, the bf16 double-run noise floor, or the determinism audit.
4. **G6 will be missed** — the always-submittable rule is being broken.
5. **A HIGH-overlap paper appears.** Novelty here decays in weeks — roughly one new quantization-reasoning paper every 1–3 weeks, and title-level counts went from 6 in all of 2025 to 19 in the first seven months of 2026, with two directly relevant papers in the first five days of August. Scans due **Aug 24, Sep 14, Oct 5**. Watch specifically for anyone crossing extraction sensitivity with quantization.
6. **Quantized generation is badly nondeterministic** and neither BI mode nor `VLLM_DISABLED_KERNELS` fixes it (§`A7`). Report the measured numbers. This may be a headline rather than a blocker, but it changes what the paper claims.
7. **Genuinely new forks not covered here** — minimal recon, characterize the options, then report.

**Report but keep working (do not wait for me):**
- **Judge fails the §`A12` gate** — the pre-specified fallback (rungs 1–3 primary, judge caveated) keeps the paper intact because the headline is already rung 1 vs rung 2. Note it and continue.
- **IAA on the overlap subset comes in below ~0.85** — that is a **rubric problem, not a judge problem.** Revise the rubric, re-run the dry run together, re-annotate, and draw no conclusion about the judge until then. Report if it persists.
- **The interaction is null across all tasks and axes.** Not a blocker — this is a publishable confirmatory result that *strengthens* the existing degradation literature by ruling out its most obvious alternative explanation. H1, H4, H7 and H8 all still yield content. Report it so framing adjusts early, and **do not let it read as failure.**
- **Anchor B or C misses.** Anchor A is the blocking one.
- **Llama access never arrives.** Execute the pre-decided Qwen2.5-7B substitution and tell me it happened.

---

## 10. Risk register

| Risk | Likelihood | Pre-decided response |
|---|---|---|
| Co-author's annotation hours slip | **High** | Judge is non-blocking by construction (headline = rung 1 vs 2). Reduce to 150 items + 80 overlap and report the reduced n honestly, rather than delaying anything. |
| Llama license not granted | Medium | Qwen2.5-7B-Instruct substitution, armed Sep 1, no permission needed. |
| GPTQ nondeterministic under BI | **Medium-high** — vLLM's own source says Marlin is not batch-invariant and GPTQ has no BI guard | Becomes H8, a headline. Report the differential-noise floor on every claim. `VLLM_DISABLED_KERNELS` fallback tested. |
| Related-work squeeze → reject | Medium | The one risk that is pure writing. `C1` in week 1, external reader tasked specifically at it, third novelty scan Oct 5. |
| A scooping paper lands | Medium | Three dated scans. H1 + H8 are the least-occupied ground; pivot emphasis rather than scope. |
| Local disk (7.6 GB) blocks the analysis env | Medium | Report top consumers to the user; Overleaf removes the LaTeX requirement; keep only JSONL and figures locally. |
| Vendor checkpoint confound | Low, but fatal to the result if it happens | Self-quantize everything at normalised calibration (§2.5); vendor checkpoints are a documented cross-check only. |
| Effects smaller than the noise floor | Medium | §4.5 rule: we do not claim them either. Reporting that is a credibility asset, not a loss. |

**On the odds, so this is calibrated rather than anxious:** plan against **~37% combined** main+Findings, not ~19%. Findings is **automatic** (ACL 2026 CFP: papers not selected for the main conference "will also **automatically** be considered"), so the two shots add; conditional-on-commitment main+Findings has run 56–76% across eight venue-years. Measured against *committed* papers, ACL 2026 was 35.0% main / **67.9% combined**. And Oct 12 is not one-shot: the **Jan 2027 cycle feeds ACL 2027**, and a meta-review ≥2.0 permits next-cycle resubmission. Do not bank on resubmission as a strategy — the one published analysis found resubmission status not significantly predictive (p=0.112) — but it is a real fallback.

**What actually reduces risk, in order:** (1) nail the §`C1` differentiation, (2) lead with truncation (H1) since it is the least-occupied ground, (3) report the decomposition honestly including nulls, (4) ship the artifact — a released generation corpus others can re-score is a durable contribution regardless of the accept/reject outcome.

---

## 11. Open questions for the user — assumptions I am proceeding on

Each has a working assumption so nothing is blocked. Correct any of them and I will amend the plan.

1. **Co-author annotation window.** Assuming they can commit **~6 hours between Aug 26 and Sep 5** (rubric dry run, 250 labels, 120 overlap, adjudication). If not, the fallback is 150 items + 80 overlap, reported honestly.
2. **GPU budget.** Assuming **~$400** is approved and a RunPod or Verda account with a working payment method can exist by Aug 25.
3. **NC-licensed judge weights.** Assuming `xVerify-9B-I` (CC-BY-NC-ND-4.0) may be *run* for internal research with numbers reported and weights never redistributed. If the institution forbids NC tooling outright, `ibm-granite/granite-4.1-8b` (Apache-2.0) becomes primary.
4. **Overleaf.** Assuming both authors can use it. Local MacTeX does not fit in 7.6 GB.
5. **Local disk.** Assuming the executor may report the top consumers but **must not delete anything** without the user's explicit say-so.
6. **Author configuration.** Assuming still two authors, co-author registering as an ARR reviewer, author list frozen at submission (ARR requires justification to add afterward and removal is near-impossible). Settle this **before** the annotation work makes the co-author a substantive contributor.
