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

---

## Running count

**Chat-template setting reported: 1 of 2 checked.**

⚠️ **M = 2 is too small to support "most papers do not report it", and the paper must not say
that yet.** What it currently supports is narrower and still useful: *of the two sources whose
deltas this paper audits most directly, one documents its protocol completely and one does not
report the template setting or the extraction mode at all.* Extending the survey is cheap
(reading only) and should cover the remaining `paper/custom.bib` entries that report a
quantization delta before any counted claim goes in the paper.

## Not yet checked

Remaining `custom.bib` entries reporting a quantization or degradation delta. Each needs the
same treatment: open the source, quote or record absence, never infer.
