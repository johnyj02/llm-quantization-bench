# Protocol survey — does the cited work report its evaluation protocol?

**Why this file exists.** Planner ruling R19.3. H7 claims that published quantization deltas
are not mutually comparable, and one leg of that is that the *protocol* is often unreported.
Asserting "most papers don't report it" from impression would be the exact failure §0.1(b)
diagnoses — so it is a **counted survey**: N reported of M checked, each verified against the
primary source and quoted.

**Method.** For each source, read the actual paper text or model card and record whether the
protocol detail is stated. "Not reported" means *checked and absent*, never *not looked for*.
Every row cites what was read. Nothing here is inferred from a repository we did not open.

⚠️ **This table becomes a column in the H7 audit table.** It must therefore stay honest about
its own coverage: M is small and is stated, not hidden.

---

## Checked

### Kurtic et al., ACL 2025 — *"Give Me BF16 or Give Me Death"?* (arXiv:2411.02355v4)

Read: arXiv HTML rendering of v4, evaluation-setup and appendix sections.

| detail | reported? | evidence |
|---|---|---|
| harness named | ⚠️ partially | lm-evaluation-harness is **never named**; cited only as *"A framework for few-shot language model evaluation"* (Gao et al. 2021). `vLLM (version 0.6.4.post1)`, `LightEval` and `the EvalPlus library` **are** named. |
| harness version | ❌ **not reported** | no version string for the accuracy harness |
| **chat template** | ❌ **NOT REPORTED** | no mention of chat templates, `apply_chat_template`, or chat-formatted prompts anywhere in the text. Closest: Leaderboard V1 *"follows Meta's prompt guidelines for Llama-3.1 models to maintain alignment with baseline scores."* |
| few-shot counts | ✅ reported | but only as table column labels — `MMLU 5-shot`, `GSM8k CoT 8-shot`, `BBH 3-shot`, `ARC-C 0-shot`, … |
| **extraction mode** | ❌ **NOT REPORTED** | neither `strict-match` nor `flexible-extract` appears; no extraction or parsing procedure is described |
| eval code released | ❌ **not found** | no GitHub or HuggingFace link to their own evaluation code in the text |
| generation settings | ✅ partially | for reasoning tasks: *"a temperature of 0.6 and top-p of 0.95, generating 20 responses per query to estimate the pass@1 score"* |

🔬 **Relevant to H6, and worth citing there:** they note MMLU and ARC-C were run as
*"text-generation tasks rather than log-likelihood-based evaluations."* That is exactly the
mode contrast H6 measures, chosen by a major quantization study without comment on its effect.

### Red Hat AI model cards — e.g. `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16`

Read: the model card's Evaluation section, which publishes complete `lm_eval` invocations.

| detail | reported? | evidence |
|---|---|---|
| harness named | ✅ | Neural Magic's **fork** of lm-evaluation-harness, branch `llama_3.1_instruct`; vLLM for generation; an EvalPlus fork for HumanEval |
| **chat template** | ✅ **FULLY REPORTED, per task** | see split below |
| few-shot counts | ✅ | explicit `--num_fewshot` per task |
| extraction mode | ✅ | metric named in the results table, e.g. *"GSM-8K (CoT, 8-shot, strict-match)"* |
| eval commands | ✅ | verbatim, with all flags and `--model_args` |

**The template split is itself informative:**

| tasks | template flags |
|---|---|
| MMLU, MMLU-CoT, ARC-C, GSM8K, OpenLLM v2, multilingual MMLU ×7 | `--apply_chat_template` (MMLU, GSM8K and OpenLLM v2 also `--fewshot_as_multiturn`) |
| Hellaswag, Winogrande, TruthfulQA | **no template flags at all**, plus `add_bos_token=True` |

✅ **This CONFIRMS our prereg §3b item 3 from their own practice.** We wrote that *"Red Hat's
`add_bos_token` recommendation is for non-template use"* and asserted it as our reading. Their
commands use `add_bos_token=True` on exactly the three tasks that carry no chat template, and
never alongside one. Our inference is now their documented behaviour.

🔬 **AND THE STRONGEST FIND OF THE SURVEY.** The card records that its results were revised
**"after Meta modified the chat template."** A vendor's own *published accuracy numbers changed
because the chat template changed underneath them.* That is third-party, non-self-generated
evidence that the prompt template is a measurement-moving parameter — precisely what §3b-bis
makes a factor — and it is far stronger for the paper than our own configuration study, because
it cannot be dismissed as an artifact of our setup.

⚠️ **Comparability caveat, and it cuts against naive reuse of their numbers.** The tasks are a
**fork's** custom variants — `gsm8k_cot_llama_3.1_instruct`, `arc_challenge_llama_3.1_instruct`,
`mmlu_llama_3.1_instruct` — not stock lm-eval tasks. Their GSM8K is therefore **not our
`gsm8k_cot`**, and any comparison must say so.

**Their w4a16 GSM8K delta, verified at source:** 82.8 → 82.9, *"recovery 100.2%"* — i.e.
**+0.1 points**, which is the figure R16.5 quoted. Now cited from the primary source rather
than from recall.

### Kurt, *Which Quantization Should I Use?* (arXiv:2601.14277) — **Anchor C**

Read: arXiv HTML v1, including Appendix B.

| detail | reported? | evidence |
|---|---|---|
| harness + version | ✅ | *"the EleutherAI evaluation harness lm_eval (v0.4.9.2)"*, with *"the default evaluation harness and llama.cpp configurations"* |
| **chat template** | ✅ **REPORTED** | *"prompts were formatted using the default Llama-3.1-8B-Instruct chat template"*, applied *"For all evaluations"* |
| few-shot counts | ✅ | *"GSM8K v3 (5-shot), HellaSwag v1 (0-shot), IFEval v4 (0-shot), MMLU v2 (0-shot), and TruthfulQA v3 (0-shot)"* |
| extraction mode | ✅ | both filters given in Appendix B; *"FE is lenient in extracting the final numeric answer from text, while SM is stricter about the expected final-answer format"* |

✅ **R1.2's diagnosis is CONFIRMED as to fact.** R1.2 hypothesised the 24.64-vs-77.63 collapse was
*"almost certainly `--apply_chat_template`"*. The paper states it applied one. The hypothesis
stands on the evidence rather than on plausibility.

🔬 **And the paper is a stronger citation than we assumed, because ITS OWN CONCLUSION depends on
the filter.** F16 baseline: *"GSM8K (FE) EM 77.63"* vs *"GSM8K (SM) EM 24.64"* — one model, two
filters. Across formats, strict-match swings **9.86 (Q3_K_M) → 37.68 (Q5_0)** while flexible sits
in a **73–79** band. So the headline ranking — including the claim that Q5_0 beats F16 — rests on
the lenient filter. That is this paper's thesis, published, in a study that did not set out to
make it.

⚠️ **SUBSTRATE CAVEAT, and it bounds what "Anchor C is reproducible" can mean.** This is
**llama.cpp GGUF K-quants on CPU** — *"llama.cpp (commit b7600)"*, *"a dual-socket CPU server with
two Intel Xeon Platinum 8488C processors"*, formats `Q3_K_S…Q8_0` against an FP16 GGUF reference —
**not vLLM with GPTQ/AWQ on a GPU**. Consequences:

* The **F16 strict-vs-flexible anchor is still targetable**, because it is a claim about
  *extraction* on a task config we also run (`gsm8k`, 5-shot, hash marker, chat template on), not
  about any quantizer.
* Their **quantized** deltas are **not comparable to ours** in kind: different quantization
  family, different kernels, CPU, and FP32 accumulation.
* The prereg's *"`gsm8k` additionally makes Anchor C reproducible"* must therefore be read as
  reproducing the **extraction gap**, not their quantization results. State the substrate
  difference wherever the anchor is invoked.

### Pape et al., *The Silent Hyperparameter* (arXiv:2605.19537) — the NEAREST PRIOR WORK

Read: arXiv HTML v2. This is the source of the "16.6 pp" figure already used in this repo.

| detail | value |
|---|---|
| corpus | *"analyze 35,000 ML publications"*, NeurIPS/ICML/ICLR/ACL/EMNLP, 2023–2025 |
| isolated as running local inference | *"strictly isolated 9,018 papers"* |
| engines catalogued | *"we identified a total of 200 inference engines"*; 44 (22%) inactive |
| **backend reporting rate** | of 2,257 papers with no code, *"820 (36 %) explicitly named the backend"*; for empty/deleted repos *"only 180 (39.1 %) reported the engine textually"* |
| dependency specs | *"only 3,860 of the accessible GitHub repositories (approx. 63 %) contained explicit dependency specifications"* |
| the 16.6 pp | DeepSeek R1 Distill Qwen 7B on GSM8K, *"a 16.60 percentage-point spread"*: 78.47 (transformers) vs 61.87 (Ollama) |

🎯 **THIS GIVES H7 A COUNTED CLAIM ABOUT THE FIELD THAT WE CANNOT MATCH AND SHOULD NOT TRY TO.**
36% of no-code papers naming their backend, over 9,018 isolated papers, is a survey at a scale our
own N-of-M cannot approach. **Cite it for the general claim** and keep our survey for the specific
sources whose deltas we audit. That is the honest division of labour and it removes any temptation
to over-generalise from M=4.

🚨 **DOES IT SCOOP H8? NO — and the differentiation must be stated precisely, not loosely,
because this is the closest work to ours.**

| | Pape et al. | H8 (ours) |
|---|---|---|
| what varies | the **engine**, across five backends | **item ordering** within ONE fixed engine |
| batching | **batch size 1**, deliberately: *"Evaluations use a batch size of one to prevent batching-induced instability"* | batch **composition** is the independent variable |
| precision | held at *"FP16 (GGUF-FP16 for llama.cpp/Ollama)"*; quantized paths **not ablated** | quantized kernels ARE the mechanism |
| intra-engine variation | **seeds only** — *"averages across twelve unique seeds"* | ordering seeds, with `bf16` as an exact-zero control |
| batch invariance | term **never appears**; He (2025) cited only to justify batch size 1 | the feature under test |
| Marlin / quantized kernels | **no mention**; not ablated | pinned by paired ±`VLLM_DISABLED_KERNELS` |

* Their one batching ablation changes batch **size** (4), not composition — and two engines could
  not participate at all, *"Evaluated at batch size=1 due to lack of native batched generation
  support"*. So *"batching influences generation"* rests on aggregate score shifts, **not** on
  holding an engine fixed and permuting which requests share a batch. **That is exactly the gap H8
  fills.**
* ⚠️ **They set batch size 1 to AVOID the effect we measure.** Framing to use: the nearest prior
  work treated batch-induced instability as a nuisance to be designed out; we make it the object of
  study and attribute it to a named kernel.

🔬 **Their root-cause analysis independently corroborates two of our own findings**, which is worth
citing rather than merely noting: the largest outliers traced to correctable defaults — **BOS
injection** and a hidden **1.1 repetition penalty** — recovering **8.34** and **11.67** points.
Our double-BOS discovery (prereg §3b item 3, found via a config audit) is the same class of defect
found independently, and it supports the paper's *declared ≠ enforced* theme with external evidence.

---

## Running count

**Chat-template setting reported: 3 of 4 checked.** (Red Hat ✅, Kurt ✅, Pape n/a-but-thorough,
Kurtic ❌.)

⚠️ **M = 4 is still too small to support "most papers do not report it", and the paper must not
say that on our own count.** What our survey supports is narrower and sharper: *of the sources
whose deltas this paper audits most directly, the one whose numbers are most often quoted
(Kurtic et al., ACL 2025) reports neither its chat-template setting nor its extraction mode.*

✅ **And the general claim no longer needs our count.** Pape et al. surveyed **35,000
publications** and found only **36%** of no-code papers named even their inference backend. Cite
them for the field-level claim; use our table for the specific sources we audit. That division
removes the incentive to over-generalise from a small M, which is the whole risk R19.3 flagged.

## Not yet checked

Remaining `custom.bib` entries reporting a quantization or degradation delta. Each needs the
same treatment: open the source, quote or record absence, never infer.
