# Responsible NLP Research Checklist

**Why this file exists as an artifact rather than only as the submission form.** `A23` records
that an *incorrect or incomplete* filing has itself been a desk-reject ground since December 2024,
and gate G6 had no instrument for it until `scripts/check_submittable.py`. Keeping the answers in
the repository means they are versioned alongside the facts they assert, and
`check_checklist()` fails if any item is left blank or carries an unfilled placeholder.

**Every number here was read from the repository, not recalled.** Sources are named per item so a
reviewer, or a later session, can re-derive rather than trust.

⚠️ **The honest answers to section D are absences.** The human annotation never ran, so the judge
rung is unvalidated. Stating that plainly here is the point: filling this section is what surfaced
a false past-tense claim in the Ethics Statement, which asserted that authors had produced
validation labels when no annotation had taken place.

---

## A. For every submission

### A1. Did you discuss the limitations of your work?
**Yes.** A dedicated, unnumbered Limitations section (49 prose lines), which the ACL page limit
excludes and which is mandatory. It states, among others: the judge rung is unvalidated and no
agreement statistic is reported; H1's and H6's instruments had not run at the time of writing; the
cross-process mechanism is localised but not explained; the protocol survey covers 4 of roughly 14
cited papers reporting a delta, so no counted field-level claim is made; and gate G5a failed on
one of three arms with the decision discharged on stated grounds rather than on a pass.

### A2. Did you discuss any potential risks of your work?
**Yes**, in the Ethics Statement. The principal risk is misreading: the paper reports that a
reported quantization delta is sensitive to evaluation-protocol choices, which could be misquoted
as "quantization is harmless" or as "published quantization results are wrong". Neither follows,
and the paper says so. A second risk is that the ordering result could be read as a reason to
distrust all quantized deployment, when what it establishes is a reproducibility requirement on
*evaluation*.

### A3. Do the abstract and introduction summarize the paper's main claims?
**Yes**, and both were rewritten so that no term is used before it is defined. The abstract states
the three contributions and gives two remedies with measured costs rather than presenting the
finding as unfixable.

### A4. Have you used AI assistants in this research?
**Yes, and substantially.** An AI coding assistant was used for implementation, analysis code,
and drafting, under human direction, with the human author retaining responsibility for every
scientific decision. Two advisory agents were used and their transcripts are retained in the
repository: a planning agent for scientific decisions (`notes/planner-rulings.md`, **42** recorded
rulings numbered to R43) and a code-review agent (`notes/reviews/`, **13** rounds), with the raw
transcripts of both released rather than summaries. Every
substantive ruling and review finding is recorded with what was accepted, what was declined, and
why. See also section E.

---

## B. Did you use or create scientific artifacts?

**Yes, both.**

### B1. Did you cite the creators of artifacts you used?
**Yes.** The three evaluated models, the four quantization methods, the serving engine, the
evaluation harness, the calibration dataset and the three benchmarks are all cited. Every
citation in `paper/custom.bib` was fetched from the arXiv API rather than recalled, and the
provenance is recorded in that file's header.

### B2. Did you discuss the licenses of artifacts?
**Yes**, and one correction matters. Licenses were resolved from the Hugging Face API on
2026-08-23 and recorded in `configs/models.yaml`:

| artifact | license | note |
|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct` | `other` (qwen-research) | ⚠️ **NOT Apache-2.0**, unlike Qwen2.5-7B/14B |
| `microsoft/Phi-3.5-mini-instruct` | MIT | |
| `meta-llama/Llama-3.1-8B-Instruct` | llama3.1 community licence | gated; access granted |
| `HuggingFaceH4/ultrachat_200k` (calibration) | MIT | |
| `IAAR-Shanghai/xVerify-9B-I` (judge, unvalidated) | **CC-BY-NC-ND-4.0** | 🚨 run and reported only; weights and derivatives are **never redistributed** |

### B3. Was the use consistent with intended use, and are created artifacts' terms specified?
**Yes.** All models are used for inference-time evaluation, which is their intended use. The
non-commercial, no-derivatives judge licence is respected by never redistributing its weights or
any derivative of them; the artifact release contains numbers only for that component.

The created artifact is the full generation record set: 105,950 generations from the reference
condition, plus the pilot, template and determinism runs, released as gzipped JSONL with a
documented schema (version 1.5.0). This is deliberately the reusable part, because it lets anyone
re-score with a different extractor without a GPU, which is the paper's own point.

### B4. Did you discuss steps taken to check for personally identifying information or offensive content?
**Yes.** No new text was collected. All prompts derive from three public benchmarks (GSM8K,
BIG-Bench Hard, ARC-Challenge) and calibration data from `ultrachat_200k`, all already public and
widely used. Model outputs are mathematical and logical reasoning traces on those benchmarks. We
did not perform an additional automated PII scan, and we say so rather than implying one: the
released generations are model completions of public benchmark items, so the residual risk is
memorised training content surfacing in a reasoning trace. No such instance was observed in the
samples read under the project's standing three-generation inspection rule, which was applied
after every generating job.

### B5. Did you provide documentation of the artifacts?
**Yes.** The schema is documented field by field in `src/quantartifact/records.py`, including
which fields are unreconstructable after the fact. Domains, languages and linguistic phenomena
are those of the source benchmarks; all data is English.

### B6. Did you report relevant statistics about the data?
**Yes.** Per-cell item counts, the exact subsample seeds, and the prefix-nesting relationship
between the reference and sweep item sets are all recorded in `configs/tasks.yaml` and reported in
the paper's setup. Reference condition: 4,238 items per cell across 25 cells.

---

## C. Did you run computational experiments?

**Yes.**

### C1. Did you report the number of parameters, total computational budget, and computing infrastructure?
**Yes**, and these are measured rather than estimated.

- **Models:** 3.1B (Qwen2.5-3B), 3.8B (Phi-3.5-mini), 8.0B (Llama-3.1-8B) parameters.
- **Infrastructure:** a single `ml.g6e.2xlarge` instance, one NVIDIA L40S (46,068 MiB usable,
  compute capability 8.9), 8 vCPU, 64 GiB host RAM, via Amazon SageMaker training jobs.
- **Total budget:** **45 jobs, 125,675 billable seconds, 34.91 GPU-hours, $97.75** at the
  measured on-demand rate. The full per-job ledger, including failed and abandoned attempts, is
  released in `notes/RUNLOG.md`. Failed runs are included deliberately, because reporting only
  successful compute understates the true cost of a result.
- **Software:** vLLM 0.26.0, torch 2.11.0+cu130, CUDA 13.0, driver 595.91.07, llmcompressor
  0.12.0.1 for quantization, lm-evaluation-harness 0.4.12 as a prompt-rendering and validation
  instrument only.

### C2. Did you discuss the experimental setup, including hyperparameter search and best-found values?
**Yes**, and there was **no hyperparameter search on the evaluation side by design**, which is
itself a result. Decoding is greedy throughout: `temperature=0`, `top_p=1`, `top_k` disabled, all
penalties at their neutral values, `seed=0`, every field pinned explicitly and asserted at engine
start rather than left to the model's own `generation_config.json`. The one serving parameter that
*was* selected by measurement, `max_num_seqs`, was chosen by a pre-registered audit and then
**moved from 64 to 32** when a pre-registered escalation trigger fired; both the original value,
the trigger, and the reason for the move are reported.

Quantization calibration is normalised across methods: `ultrachat_200k` `train_sft`, 512 samples
at 2,048 tokens, seed 20260823, shuffled, giving a 1,048,576-token ceiling of which the measured
usage was 579,588 tokens (55.3%). The measured figure is reported rather than the product, because
the product overstates it roughly twofold.

### C3. Did you report descriptive statistics about your results, including error bars?
**Yes**, and error bars are on every estimate. Intervals are cluster bootstrap percentile
intervals over items, 10,000 draws, with fixed seeds recorded in code. Multiplicity is controlled
by Holm–Bonferroni within each hypothesis family, with the family size reported. Where the
observed variance is exactly zero, a Wilson zero-event bound is reported instead of a degenerate
interval, and the interval *type* is recorded on every estimate so a bound is never read as a
confidence interval. No bare significance flag appears anywhere.

### C4. Did you use existing packages, and did you report the implementation details?
**Yes.** `math-verify` 0.9.0 for the symbolic-equivalence scoring rung, `statsmodels` for the
population-averaged GEE secondary estimator, `lm-evaluation-harness` 0.4.12 for prompt rendering.
Two harness defects are pinned as regression tests rather than patched, and both are reported:
BIG-Bench Hard's answer filter drops the final character of a generation that ends exactly at its
answer, and `gsm8k_cot`'s strict regex captures the sentence period.

---

## D. Did you use human annotators or research with human participants?

🚨 **No. The planned annotation did not take place, and this is stated as an absence rather than
omitted.**

### D1. Were instructions and payment shown to participants?
**Not applicable, because no annotation ran.** An annotation protocol was pre-registered in full,
including the stratified sample design, blinding by omitting the condition column entirely rather
than hiding it, independent per-annotator shuffling with published seeds, a blind practice round
in place of attention checks, and an adjudication rule fixed before any disagreement was seen. It
was **not executed**: the rubric required a second author's availability, which did not
materialise before the submission deadline. The pre-registered protocol is described in the
appendix **as registered, not as performed**.

### D2. Did you report information about how you recruited and paid participants?
**Not applicable.** No participants were recruited. Had the protocol run, the annotators would
have been the paper's authors rather than crowdworkers, and unpaid.

### D3. Did you discuss whether and how consent was obtained?
**Not applicable.** No human subjects data was collected.

### D4. Was the data collection protocol approved by an ethics review board?
**Not applicable.** No human subjects data was collected, so no review was sought.

### D5. Did you report the basic demographic and geographic characteristics of the annotator population?
**Not applicable.** No annotator population exists.

🚨 **The consequence, stated in the paper rather than buried here:** the LLM-judge rung is
**unvalidated**. No agreement statistic, no Cohen's κ, no Gwet's AC1, and no bf16-versus-4-bit
judge error comparison are reported, because none were computed. **No claim in the paper rests on
the judge.** The headline contrast is strict against permissive extraction, both deterministic and
auditable, which is why the judge's validation gate was designed to be non-blocking.

---

## E. Did you use AI assistants?

### E1. Did you include information about your use of AI assistants?
**Yes**, in the Ethics Statement and in A4 above. An AI coding assistant performed
implementation, analysis and drafting under human direction across the project. The human author
made and owns every scientific decision, and the record of those decisions is released rather than
summarised: 42 planning rulings with their raw text, 13 code-review rounds with an outcome block
recording what was acted on and what was declined with reasons, and a decision log.

Three categories of assistant error were caught and are recorded in the repository because they
bear on the reliability of assisted work: a registered finding inflated roughly twentyfold by a
defect in our own extraction code, later corrected by a full census rather than a sample; several
comments and docstrings that asserted resolved questions as open or advertised an analysis a
ruling had withdrawn; and one instance of reporting a test suite as green while it was red. Each
is documented with the mechanism, because the transferable lesson is that assisted work needs
verification instruments rather than assurances.
