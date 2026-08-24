# QuantArtifact — Executor Plan

**Written by:** planner agent, 2026-08-06. **Revised 2026-08-18** (see §0.1).
**Status:** this is the **single authoritative plan**. The originating `quantization-degradation-artifact-plan.md` has been **retired and deleted** (2026-08-18); everything still load-bearing from it is absorbed here (§0.2 goal, §0.3 estimation protocol). Ignore any external reference to it.
**Deliverable:** an 8-page ACL long paper submitted to ARR by **2026-10-12, 23:59 UTC−12 (AoE)**, plus a released reproducibility artifact.

> ## 0.1 Corrections since 2026-08-06 — read before §1
>
> Twelve days of adversarial re-verification changed four things. **11 of the 66 planned days were spent before any code existed**, so the §9 schedule is compressed accordingly — re-derive it from 2026-08-18, not Aug 6.
>
> **(a) The serving/inference-stack axis is NOT a novelty axis — it is a confound to pin.** A reviewer had claimed it unscooped; verified false. `arXiv:2605.19537` *"The Silent Hyperparameter"* (2026-05-19) holds weights/decoding/hardware constant, swaps five engines, and finds **backend choice alone moves benchmark scores up to 16.6 pp**. `arXiv:2608.04714` (2026-08-05) runs 3 models × 5 frameworks × 6 benchmarks × 4 generation modes and attributes **~39% of out-of-the-box variance to the backend**, significant even under greedy decoding. Batch size is also taken (`2605.27763`, `2604.13413`), kernel/kv-cache too (`2608.13756` *"The Integer Alibi"*), and the clinical framing too (`2606.21023` HEAL, `2608.03854` biomedical). §5's existing rule to freeze the backend is therefore **mandatory for identifiability**, not hygiene: if engine + version + generation config are left free, the H1–H3 effects are not identifiable and a reviewer will say so. **Add `2608.04714` to related work** (`2605.19537` is already cited).
>
> **(b) Do NOT reframe onto "reconcile the 32.39% vs 3.1 pp discrepancy."** A novelty-check agent recommended making this the spine. It is a trap, for exactly the reasons Appendix B already documents: the figure is *relative* percent not pp, on Llama-3.2-1B not 3.1-8B, on MATH-500 not GSM8K, attributed to GPTQ in Table 1 and AWQ in Table 2, and **Table 1's arithmetic does not close** (18.6 − 14.94 = 3.66 ≈ 19.7%, yet it prints ↓6.8 (32.39%)). A spine built on reconciling a units error plus a cross-table copy error in one unrefereed preprint reads as a rebuttal of that preprint and **lowers P(accept) by ~15 points**. The defensible residue is a narrow *reporting-standards* claim: published quantization deltas are mutually incomparable because relative-vs-absolute, task, model, extraction, token budget and engine are unstandardised and mostly unreported.
>
> **(c) Commit to NAACL 2027 as primary, not COLING 2027.** Verified: **COLING has never had a Findings track** (zero COLING volumes in the ACL Anthology Findings index, 2020–2026; COLING 2027's CFP does not mention it). Both take the Oct 12 2026 ARR cycle, but naming COLING primary **forfeits the Findings fallback entirely** — ~20–28% vs ~43%. NAACL 2027: San Francisco, Jun 1–5 2027; commitment Dec 23 2026 per the NAACL CFP, Dec 20 per the ARR dates page — **plan to Dec 20** and re-verify in Nov.
>
> **(d) Findings is automatic, so the two shots ADD.** Verbatim, ACL 2026 CFP: papers "not selected for the main conference will also **automatically** be considered for publication in Findings." It is one commitment, one committee pass, two possible outcomes — not an author-selected track. Conditional-on-commitment Main+Findings has run **56–76%** across eight venue-years (NAACL most generous at 72.6% / 76.2%). Calibrated estimate for this paper, conditional on a *good* submission existing: **~43%** (Main ~13%, Findings ~30%).

> ## 0.2 Goal
>
> **The end goal is a submitted, accepted, citable paper** — not a folder of experiment outputs. Done means: H1–H4 supported *or honestly refuted*; a related-work section that correctly differentiates from every paper named in §1 and §0.1; reproducibility artifacts released per §8; and a full draft finished with enough runway before Oct 12 for internal review and revision. The standing priority test in §0 applies to every trade-off.

> ## 0.3 Time and cost estimation protocol
>
> Produce the estimate in **two passes**, covering the whole path to a submittable paper (not just experiments):
> 1. **Loose** — back-of-envelope, within the first day, so there is an early feasibility signal against Oct 12 before committing to detail.
> 2. **Tight** — after GPU access is confirmed and after the day-3 calibration run gives *measured* generations/hour on the actual hardware.
>
> Report both, broken out by phase (pipeline → main runs → judge validation → analysis → writing/revision), with **dollar cost and calendar time stated separately** (renting more GPUs shortens wall-clock but raises cost). **If the tight estimate does not fit the runway, say so immediately** — that is a report-back trigger in its own right (§10), and it may force the §10 cut order.

---

## 0. How to use this document

The originating plan was a good idea with several factual problems that would have cost weeks if discovered mid-execution. I ran a verification pass — citations, venue logistics, novelty, toolchain, harness configs — before writing this. §1 states what changed and why; §2–§8 are the design you should actually build; §9 is the schedule; §10 is what to report back and when. Appendix A holds verified facts worth not re-deriving; Appendix B holds errors not to propagate into the paper.

**The single most important change:** the originating plan's primary hypothesis is largely already published, and the axis it omitted entirely (generation truncation) is now the strongest remaining gap. Read §1.1 before doing anything else — it reorders both the experiments and the paper.

**Standing priority test:** does this get us closer to a submittable paper by Oct 12? Not "is this experiment interesting."

**Author configuration — CONFIRMED 2026-08-06: two authors (the user plus one co-author), and the co-author will register as an ARR reviewer.** Consequences:
1. **Both authors must register as ARR reviewers/ACs within 48h of submission (~Oct 14)** or the paper is **automatically desk-rejected** — the obligation is per-author, not per-paper. Both need a **complete OpenReview profile** (affiliation, Semantic Scholar / DBLP / ACL Anthology links, working email); an incomplete profile blocks registration. Registration is mandatory regardless of whether either is *qualified* (≥2 papers in main ACL venues/Findings plus ≥1 more in the Anthology or a major ML/AI venue) — ARR uses qualification to decide assignments, not to waive the requirement. Note ARR's May 2026 policy scales service load with submission count.
2. **Two annotators are available, so §3 uses proper inter-annotator agreement.** This is the stronger design and is what the plan now specifies — the single-rater test–retest fallback is dropped.
3. **Limited division of labor.** Two people still means one engineering track and a 9.5-week runway, so the §10 cut order stays live: treat the self-consistency arm as the designated cut, not baseline scope.

**Also non-negotiable:** a "Limitations" section is mandatory and its absence is an automatic desk reject. It costs zero page budget. Write it early.

---

## 1. What changed from the originating plan, and why

Five corrections. §1.1 (hypothesis reframe + truncation promoted to lead) and §1.2 (replication target retargeted) are the load-bearing ones.

### 1.1 The primary hypothesis as written is largely already published — but a sharper version is not

The originating plan's H1 is: *reported degradation shrinks when lenient LLM-judge extraction replaces strict regex extraction.*

The problem: **that lenient extraction yields higher absolute scores than strict extraction is already a standard, documented, widely-reported fact.** Concretely:

- The standard `lm-evaluation-harness` GSM8K task **already reports both** `strict-match` and `flexible-extract` as parallel metrics. I read the config (`lm_eval/tasks/gsm8k/gsm8k.yaml`, v3.0). The strict regex is `#### (\-?[0-9\.\,]+)` — it requires the literal `####` marker, which instruction-tuned models essentially never emit. So strict-match collapses toward zero for chat models as a matter of course.
- The gaps are enormous and already in print. **"Which Quantization Should I Use?" (arXiv:2601.14277, Jan 2026)** ran GSM8K strict-vs-flexible across 14 llama.cpp quantizations of Llama-3.1-8B-Instruct and reports gaps of **41–64 accuracy points** (e.g. F16: flexible 77.63 vs strict 24.64).
- **"Finding Answers in Thought Matters" (Jo et al., arXiv:2510.14773, Oct 2025)** already owns the general extraction-sensitivity claim across 5 extraction methods × 5 reasoning models, with GSM8K strict-vs-flexible gaps up to +85 points. It does not touch quantization.
- **"Give Me BF16 or Give Me Death?" (Kurtic et al., ACL 2025, arXiv:2411.02355)** has already published the conclusion *"prior reports of quantization degradation were inflated"* — via a different mechanism (untuned quantization hyperparameters, absmax-vs-MSE clipping, weak C4 calibration). Over 500,000 evaluations; it turns a prior-reported 10-point 405B drop into 0.7 points. It never discusses answer extraction.
- **"Flaw or Artifact? Rethinking Prompt Sensitivity" (Hua et al., EMNLP 2025 main, arXiv:2509.01790)** already executed *our exact methodological move* one domain over: it asks whether reported prompt sensitivity is "an artifact of evaluation processes," and shows an LLM judge collapses a 65-point accuracy range to 0.17 and lifts ranking stability from ρ̄=0.30 to 0.95. **We are structurally the sequel to this paper and must say so.** One finding of theirs cuts against us and must be engaged: **on MATH they found essentially no gap (0.9593 vs 0.9647) because its extraction already uses sympy symbolic equivalence.** Well-engineered verifiers are already robust, so a blanket "extraction is fragile" claim on math is vulnerable — pick tasks where extraction is genuinely brittle and justify the choice.
- **"Quantization Effects on Biomedical LLM Reliability" (arXiv:2608.03854, Aug 4 2026 — two days ago)** crosses *two of our four factors* with quantization at 7B and concludes **"the probability extraction protocol dominates apparent calibration"**: prompt template produces 7–24pp accuracy swings, "comparable to or larger than model-level effects," while INT8 moves accuracy 1–2pp. It is classification/calibration, not generative reasoning, and has no decoding, truncation, or judge arm — but it means **the mechanism is no longer ours to discover, only to instantiate in the reasoning domain.**

If we submit "lenient extraction raises quantized-model scores," a reviewer who knows the harness will say: *the harness has shipped both metrics since 2023, and the numbers you're reporting are in someone's appendix already.*

**The reframe — and this is the intellectual core of the paper.** A measurement artifact in a **level** is not the same as a measurement artifact in a **difference**. Everyone knows lenient extraction raises absolute scores. Nobody has established whether it changes the **quantization delta**. If strict extraction penalizes bf16 and 4-bit equally, the degradation estimate is unaffected and there is *no artifact in the quantity papers actually report*. The artifact exists only if quantized models are **differentially** penalized — i.e. if there is a **precision × scoring interaction**.

That interaction is genuinely open, it is the quantity that matters, and it is cheap to test. Make it the headline.

The mechanism is concrete and separately measurable: quantization may degrade a model's ability to *emit an answer in the expected slot* (format compliance) independently of its ability to *get the answer right* (competence). Strict-match multiplies these two together and reports the product as "reasoning accuracy." So:

```
measured_strict_delta  =  competence_delta  +  compliance_delta  +  truncation_delta
```

The paper's contribution is **decomposing a published effect into those components**, for the first time, on mainstream models and tasks.

**Why this survives the obvious counterattack.** A reviewer will say: *if quantization really does damage format-following, then strict-match is measuring a real capability and "artifact" is the wrong word.* There is published support for that objection (Jin/Lee et al., IJCAI 2025, arXiv:2409.11055 — quantization disrupts strict instruction-following; IFEval 405B-W4 72.90 vs 70B-FP16 75.48). **The reframe answers it instead of dodging it:** if compliance genuinely degrades, that is a real finding, but it is a *different* finding than "reasoning breaks," and papers reporting it as reasoning degradation are mis-attributing the cause. Either way we have a result. That is what makes this design safe to commit nine weeks to.

**Lead the paper with truncation, not extraction.** A second novelty scan (Aug 6) found the truncation axis — which the originating plan omitted entirely and which I initially treated as a minor addition — is now the **strongest remaining gap in the literature**, for a specific reason: there is a live, unresolved empirical contradiction sitting on top of it.

- **arXiv:2606.25519 (Jun 2026)** reports 4-bit quantization *preserves accuracy while lengthening chain-of-thought*, and introduces a "CoT Token Inflation Ratio."
- **Liu et al. (COLM 2025, arXiv:2504.04823)** explicitly reports that quantized models do **not** produce longer outputs.

Both cannot be right, and **neither paper swept the generation cap.** Longer CoT under a fixed `max_new_tokens` mechanically manufactures degradation: the model is cut off before it emits an answer and scored wrong. That is an unexamined confound at the center of an active dispute, and it is the cleanest, most legible artifact in the whole design.

Reinforcing this: **arXiv:2606.00206 (May 2026)** found that in **up to 52% of quantized-model failures, the correct answer appears in the intermediate reasoning but is never emitted as the final answer.** They attributed it to overthinking and fixed it with a logit penalty. **Nobody has asked how much of that 52% a lenient extractor or a judge simply recovers.** That is a single sharp falsifiable experiment and it is ours to run — but note it is also our biggest scoop risk, because a reviewer will ask why we didn't do exactly that.

So the priority ordering for both experiments and narrative is: **truncation → the 52% extraction test → prompt-format × bit-width → extraction strictness generally.** That is the reverse of the originating plan's emphasis. Note this also changes the cut order in §10: the token-budget arm is now load-bearing and must not be cut.

**Revised hypotheses** (replace §3 of the originating plan wholesale). Ordered by novelty strength, which is also the order to run and to write them:

- **H1 — truncation (primary, and the lead of the paper).** A non-trivial share of measured 4-bit degradation on generative reasoning tasks disappears when the generation cap is raised, because quantized models emit longer chains of thought and are cut off before stating an answer. Test by sweeping `max_new_tokens` as an explicit independent variable and reporting the degradation estimate at each budget. Also resolves the 2606.25519-vs-2504.04823 contradiction on whether quantization lengthens CoT at all, which is a citable contribution by itself.
- **H2 — the recoverable-failure test.** Among quantized-model failures, measure what fraction contain the correct answer somewhere in the reasoning trace but not in the final-answer position, and how much of that a permissive parser or judge recovers. Directly targets the reported 52% (arXiv:2606.00206) with the measurement question its authors did not ask.
- **H3 — interaction.** The measured degradation depends on scoring strictness: a `precision × scoring` interaction on item-level correctness. *Null result is publishable and is a real service:* it shows extraction leniency is a level shift, not a delta artifact, which **strengthens** the existing degradation literature by ruling out the most obvious alternative explanation.
- **H4 — decomposition.** The measured strict-match degradation decomposes into competence, compliance, and truncation components. Report the fraction attributable to each, with CIs. This is the quantitative payload and it yields a number regardless of how H3 falls.
- **H5 — prompt format × bit-width.** The same interaction test for few-shot CoT vs zero-shot. The scan found **zero** papers on quantization × prompt-format sensitivity for reasoning — a genuinely empty cell, though a smaller one.
- **H6 — structural negative control.** Where extraction is structurally constrained, the interaction vanishes. This is what ARC-Challenge is for (§1.3).
- **H7 — powered-ness audit (cheap, analytical).** Many published quantization deltas are 1–3 points on evals of 250–1319 items; at n=500 the 95% CI on a single accuracy is roughly ±4.4 points. **A meaningful fraction of published claims may sit inside the sampling noise of their own evaluation.** Nearly free — arithmetic over published numbers plus our measured run-to-run variance (§4.6).
  ⚠️ **Do not frame this as "the effects are just noise."** Kübler et al. (ICLR 2026, arXiv:2602.10144) ship a paired McNemar test on lm-evaluation-harness showing **"even empirical accuracy degradations of 0.3% can be confidently attributed to actual degradations rather than noise"** when tested per-item rather than on aggregate accuracy. **Adopt their test rather than arguing around it** — our design is naturally paired at the item level, so this is easy, and effects that survive McNemar are far stronger claims. Frame H7 as "many papers report aggregate deltas without a paired test," not as "the deltas are noise."

### 1.2 The replication target is not replicable as specified — retarget it

The originating plan makes replicating "Quantization Meets Reasoning" (Li et al., arXiv:2501.03035) a required gate, described as *"Llama-3.1-8B-Instruct, GSM8K, their reported ~32% degradation figure."* Every element of that description is wrong:

| Plan says | Actually |
|---|---|
| ~32% degradation | **32.39% is *relative* percent, not percentage points.** The absolute drop is ~6.8 points. |
| Llama-3.1-8B | The 32.39% is **Llama-3.2-1B-Instruct**, the smallest model. The 8B drops ~11–12% relative. |
| GSM8K | The figure is on **MATH / MATH-500**. GSM8K is not where that number comes from. |
| AWQ/GPTQ | The paper attributes the *same* 6.8/32.39% to **GPTQ on MATH** (Table 1) and **AWQ on MATH-500** (Table 2). Both cannot be true. |

Worse, **Table 1's arithmetic does not close**: it lists vanilla BF16 = 18.6 and GPTQ = 14.94, then prints the delta as `↓6.8(32.39%)`. But 18.6 − 14.94 = 3.66 (≈19.7%), not 6.8. The 6.8/32.39% appears to belong to the Table 2 aligned-1B/AWQ/MATH-500 row and to have been copied into Table 1, the abstract, and the intro. The paper also **released no code** and reports no decoding temperature, sample count, or seeds.

Replicating this is a trap: it would burn a week and the gate could never cleanly pass.

**Retarget the replication gate:**
1. **Primary:** Kurtic et al., ACL 2025 (arXiv:2411.02355) — peer-reviewed, code released, uses `lm-evaluation-harness`, and reports both log-likelihood and generative evaluations. This is a real, checkable target.
2. **Secondary, and strategically valuable:** reproduce the strict-vs-flexible GSM8K gap from arXiv:2601.14277's Appendix B. Same model (Llama-3.1-8B-Instruct), same task, published numbers. If we reproduce their gap, our pipeline is validated *on exactly the phenomenon our paper is about*.
3. **Do not gate on Li et al.** Cite it as a motivating example of the claim genre. You may note the internal inconsistency — but **only after independently re-deriving it from the paper's own tables yourself**, and phrase it neutrally and collegially (e.g. "the reported delta appears inconsistent with the accuracies given in the same row"). Do not build any argument that depends on it.

### 1.3 ARC-Challenge as specified cannot exhibit the effect — use the generative variant

I read the harness config. Standard `arc_challenge` inherits `arc_easy.yaml`, which is `output_type: multiple_choice` — **scored by comparing per-option log-likelihoods, with no text generation and therefore no answer-extraction step whatsoever.** As specified it cannot show an extraction artifact, so it is useless as the H6 negative control in the way the plan intends.

Two fixes, and you want both because they do different jobs:
- **`arc_challenge` (log-likelihood, unchanged)** — keep it as the *structural floor*: a scoring mode with zero extraction surface. Any degradation here is by construction not an extraction artifact. This is the real negative control.
- **`arc_challenge_chat`** — this exists in the harness (`lm_eval/tasks/arc/arc_challenge_chat.yaml`), is `generate_until`, and prompts *"Your response should end with \"The best answer is [the_answer_letter]\""* with `max_gen_toks: 100`. Running **the same items under both scoring modes** gives the cleanest possible compliance measurement in the paper: identical model, identical items, one mode immune to extraction and one not. The gap between them *is* the extraction cost, measured directly.

That pairing is a better experiment than the originating plan's ARC design and costs almost nothing extra.

### 1.4 BBH is mostly multiple-choice — pick the subset on evidence

The plan calls for "a curated 6-task subset of BBH" without specifying it. I pulled the actual targets for all 27 subtasks. **Only 4 are genuinely free-form**; the large majority are `(A)`-style multiple choice, which have little extraction surface:

- **Free-form (extraction-sensitive — these are the ones that matter):** `word_sorting` (long word sequences), `dyck_languages` (bracket strings like `] ] >`), `multistep_arithmetic_two` (signed integers), `object_counting` (integers).
- **Boolean/word:** `boolean_expressions` (True/False), `navigate`, `causal_judgement`, `web_of_lies` (Yes/No), `sports_understanding` (yes/no), `formal_fallacies` (valid/invalid).
- **Parenthesized multiple choice `(A)`–`(R)`:** the remaining 17, including `date_understanding`, `logical_deduction_*`, `tracking_shuffled_objects_*`, `penguins_in_a_table`, `reasoning_about_colored_objects`, `snarks`, `ruin_names`, `geometric_shapes`, `hyperbaton`, `temporal_sequences`, `disambiguation_qa`, `movie_recommendation`, `salient_translation_error_detection`.

**Use all 4 free-form + 2 contrast tasks** (`boolean_expressions` for the boolean format, `logical_deduction_five_objects` for the MC format). That gives a within-BBH format gradient — free-form → boolean → MC — which is itself a prediction of the artifact hypothesis: the interaction should shrink monotonically as the answer space becomes more constrained. That gradient is a much better use of six tasks than six arbitrary ones.

Also note the harness's BBH CoT extraction filter is `(?<=the answer is )(.*)(?=.)` — a **case-sensitive lookbehind requiring the exact lowercase string "the answer is."** Any capitalization variant ("The answer is", "So the answer is:") fails extraction entirely. Log this; it is a concrete, citable brittleness and a likely source of the effect.

### 1.5 One more axis the originating plan omits entirely: truncation

Add **generation-length budget** as a fourth measurement axis. If quantized models are more verbose or more repetitive, they hit `max_gen_toks` and get cut off *before emitting the answer* — scored as wrong, but the cause is a token budget, not reasoning. This is the cleanest, most legible artifact in the whole design and it is nearly free to test (raise the cap and rescore). Prior work reports 52% of failures in some settings have the correct answer present in the trace but not emitted as final (arXiv:2606.00206). **Log `finish_reason` and token counts for every single generation** — without this you cannot separate truncation from failure, and you cannot add it later without regenerating.

---

## 2. Final experimental design

### 2.1 Factors

| Factor | Levels | Cost |
|---|---|---|
| **Model** | Qwen2.5-3B-Instruct, Phi-3.5-mini-instruct (3.8B), Llama-3.1-8B-Instruct | — |
| **Precision** | bf16 baseline, GPTQ-4bit, AWQ-4bit, bnb-NF4-4bit, GPTQ-8bit (secondary) | 5 per model |
| **Task** | GSM8K, BBH×6, ARC-Challenge (both scoring modes), TriviaQA | — |
| **Token budget** | standard (reference), 2×, 4× — **frozen before results** | ×2 extra; **H1, do not cut** |
| **Prompt format** | few-shot CoT (reference), zero-shot | ×2 generations |
| **Decoding** | greedy (reference), self-consistency **k=8** @ T=0.6 | ×8; first to cut |
| **Scoring** | strict / permissive-parser / math-verify / LLM-judge | **free — rescoring only** |

Keep the originating plan's one-factor-at-a-time structure from a fixed reference condition (few-shot CoT, greedy, strict extraction, standard budget). It is the right call and the reasoning in §4.4 of that document stands.

**Two parameter corrections from the originating plan.** It specified self-consistency at **k=5, T=0.7**. Use **k=8 at T=0.6**: k=5 matches no convention in the literature (the sweep values are 1/5/10/20/40 in Wang et al., and modern work uses 8 or 64), while k=8 is the standard cost-constrained choice and is what the existing harness task supports via `take_first_k`. T=0.6 is the modern convention (Minerva, DeepSeek-R1). See Appendix A — and note the harness's own self-consistency task uses T=0.2, so whatever you pick, state it and justify it.

The **token-budget row is the paper's lead hypothesis** (§1.1) and must be treated as a first-class factor with pre-frozen levels, not a robustness check.

**The critical architectural consequence:** the scoring factor costs *zero* generation. Persist every generation in full and apply all four scorers as pure functions over stored artifacts. This means the scoring-strictness hypothesis (H3) is answered by re-scoring the reference run — cheap, and immune to generation nondeterminism because the *same* text is scored every way. Design the package around this (§5).

### 2.2 The scoring ladder (replaces the single LLM judge)

The originating plan makes the lenient arm an LLM judge. That is the wrong primary instrument for these tasks, and the evidence is strong:

- GSM8K answers are **integers**. There is no symbolic-equivalence problem for a judge to solve. Minerva's own ablation shows full sympy equivalence buys only **~1 accuracy point** on MATH over good normalization, and ~0 on GSM8K-style integer answers.
- **LLM judges are actively unreliable on math.** Zheng et al. (MT-Bench, NeurIPS 2023) measure GPT-4-as-judge failing on **14/20 (70%)** of elementary math gradings by default, and still **3/20 (15%)** when given the reference answer. Position bias gives GPT-4 only 65% swap-consistency.
- Rule-based verifiers are themselves weak in independent evaluation (xVerify measures Math-Verify at 45.6% F1, lm-eval-harness at 47.7%), so no single scorer should be trusted alone.

**Use a four-rung ladder, reported as a monotone sequence.** This is strictly better than a strict/lenient binary: it shows *where* on the leniency spectrum the degradation estimate moves, and it does not stake the paper on one instrument.

1. **`strict`** — the published harness regex, verbatim (`#### (...)` for GSM8K, the `the answer is` lookbehind for BBH). This is what prior work reports.
2. **`permissive`** — deterministic, documented, pre-registered parser: last-number-in-string, `\boxed{}`, "answer is X", comma/currency/unit stripping, Minerva `normalize_final_answer` rules. Deterministic and auditable — this is the workhorse.
3. **`math-verify`** — `math-verify` 0.9.0 (`pip install math-verify[antlr4_13_2]`) for symbolic/numeric equivalence. **Gotcha:** `parse()`/`verify()` use `signal.alarm()` and will raise in worker threads — pass `parsing_timeout=None, timeout_seconds=None` and impose your own timeout. Note it is intentionally *asymmetric* (gold/target order matters).
4. **`judge`** — open-weight LLM judge, given the question, the full generation, and the reference answer. Reported as the top rung and **validated against humans (§3)**, not assumed. Note math-verify alone is *not* a safe substitute: independent evaluation puts it at 16.1 F1 on out-of-distribution answer formats, so use it as a fast path and let the judge handle what it can't.

Report all four for every cell. The headline number is the interaction under rung 1 vs rung 2 (deterministic, hardest to attack); rungs 3–4 are robustness.

### 2.3 The compliance/competence 2×2 — the mechanism measurement

For every generation, assign **two orthogonal labels**. This is what breaks the circularity objection in §1.1, so implement it carefully:

**Compliance** (independent of correctness — is the answer *in the expected place*?)
- `COMPLIANT` — answer present in canonical expected format
- `RECOVERABLE` — no canonical format, but the permissive parser finds a committed final answer
- `ABSENT` — no identifiable answer (degeneration, refusal, or truncation)
- `TRUNCATED` — `finish_reason == "length"` (record separately; a generation can be both truncated and absent)

**Competence** (given the answer the model actually committed to, is it right?)

Then `strict_acc = P(COMPLIANT ∧ correct)` and `permissive_acc = P(answer recoverable ∧ correct)`, and the decomposition in §1.1 follows directly. Compliance rate is a **dependent variable in its own right** — if it drops under quantization, that is a real, reportable, novel effect, and it is the thing prior work has been folding into "reasoning accuracy."

### 2.4 Item counts

Per (model × precision) cell — 15 cells:

| Task | Items | Note |
|---|---|---|
| GSM8K | 500 (fixed random subsample of the 1319 test items, seed pinned) | full 1319 if throughput allows; report which |
| BBH × 6 | 150 each = 900 | subtasks have 250 (some 146–187); subsample to 150 |
| ARC-Challenge | 500 | run **both** log-likelihood and `_chat` generative on the same 500 |
| TriviaQA | 300 | see §2.5 |
| **Total** | **~2,200 items/cell** | |

Rough generation volume: reference 15 × 2200 ≈ 33k; token-budget sweep (2×, 4×) on the extraction-sensitive tasks (GSM8K + BBH ≈ 1400 items) ≈ 42k; zero-shot variant ≈ 33k; self-consistency k=8 on the same subset ≈ 168k. **Order 275k generations**, but note the self-consistency arm alone is ~60% of that and is the designated cut — without it the study is ~110k generations. Output length rises with the budget sweep, so reckon on ~100M output tokens at full scope.

That is more than the originating plan's 60–90k estimate, but still small: these are 3–8B models under vLLM with batching, and measured throughput puts even the full scope in the low tens of GPU-hours (§6). **The binding constraint is engineering wall-clock and the Oct 12 deadline, not compute cost** — the GPU bill is on the order of $100. Verify against measured throughput in the §9.1 pilot, and cut self-consistency first if it disappoints.

### 2.5 TriviaQA is a weak "strict" control — fix or replace

The harness `triviaqa` task scores `exact_match` against `doc_to_target: "{{answer.aliases}}"` — i.e. it already matches against a large human-curated alias list, with `ignore_case` and `ignore_punctuation` on. **It is already a lenient matcher**, so it is a poor instrument for a strict-extraction contrast.

Keep TriviaQA (it is the cleanest *factual-recall* control, which is what H6 needs), but score it three ways: alias-match (standard), first-span-only (artificially strict), and judge. If that proves awkward in the pilot, drop to 200 items and treat it as secondary — do **not** let it consume time. The load-bearing negative control is the ARC log-likelihood/generative pairing from §1.3, which is stronger.

---

## 3. Judge validation (gate)

Keep the originating plan's gate, with these corrections.

**Judge model — must be a different family from all three models under test** (self-preference bias is real and measured: Panickssery et al., arXiv:2404.13076, find LLM evaluators recognize and favor their own generations, with self-recognition capability linearly correlated to self-preference strength; the xVerify authors independently note that "models from the same family tend to receive higher ratings"). All repo IDs below were verified against the Hub API.

Use **two judges** and report their agreement — a single judge is a single point of failure for H3/H4, and the second one costs little:

- **Primary: `IAAR-Shanghai/xVerify-9B-I`** (9.24B, Gemma-2-9B base). This is **purpose-built for exactly our task** — binary answer-equivalence from `{question, llm_output, correct_answer}` → `Correct`/`Incorrect` — and it outperforms GPT-4o-as-judge on answer verification (97.19 vs 96.25 F1 test; 96.23 vs 96.03 generalization). Family-clean w.r.t. Qwen/Phi/Llama.
  **License blocker: CC-BY-NC-ND-4.0** (non-commercial *and* no-derivatives). Fine for running a research evaluation and reporting numbers; **do not redistribute the weights or any derivative** in our artifact. Cite it and let users download it themselves. If the user's institution forbids NC-licensed tooling even for internal research use, fall back to the backup as primary.
- **Backup / second judge: `ibm-granite/granite-4.1-8b`** — Apache-2.0, ungated, published IFEval **87.06**. Cleanest license for an artifact release and strong instruction-following. Alternative if a fully-open-data story is preferred: `allenai/Olmo-3.1-32B-Instruct` (Apache-2.0, IFEval 88.8, open data/code — the best reproducibility narrative, at 4× the size).
- **Do not use** `mistralai/Mistral-Small-24B-Instruct-2501` as originally sketched (superseded; the current in-range Apache-2.0 Mistral options are `Mistral-Small-3.2-24B-Instruct-2506` or `Ministral-3-{8,14}B-Instruct-2512`), `google/gemma-3-*` (gated + `license:gemma`), or any Qwen-/Llama-based verifier including **CompassVerifier** (Qwen2.5-based — disqualified on family grounds despite leading VerifierBench at 87.7). **Prometheus is unusable**: its parser hard-codes a 1–5 Likert range and cannot emit a binary verdict.
- Note `google/gemma-4-*` is now Apache-2.0 and ungated if a larger family-clean instruct model is wanted.

**Borrow CompassVerifier's 3-way label space** (`A`/`B`/`C` with an explicit `INVALID` class) rather than a pure binary. Our 3B models under 4-bit quantization *will* produce truncated and repetitive output, and an `INVALID` verdict maps directly onto the `ABSENT`/`TRUNCATED` compliance labels in §2.3. Also borrow its "compare ONLY the final answer" instruction.

**Judge disagreement is a finding, not noise.** xVerify and CompassVerifier are competing author groups publishing wildly inconsistent numbers for each other (95.75 vs 42.6 F1 on math), and VerifyBench found xVerify failing to emit its required output format under a foreign prompt template — so validate our exact prompt template on a small sample before scaling, and check the judge's own output-parse rate.

**Protocol:** give the judge the question, the full generation, and the reference answer (reference-guided judging cuts GPT-4's math-grading failure rate from 70% to 15% — never judge without the reference). Constrain output to a single token verdict. Fix the seed, greedy decode.

**Validation:** ≥250 items (raise from the plan's 150 — cheap, and it tightens the CI meaningfully), stratified across all 3 models × bf16/4-bit × all tasks, **oversampling disagreement cases** between rungs 1–3, since the agreement statistic only matters where scorers disagree. Report agreement on both the stratified sample and the natural distribution.

**Statistics — do not report Cohen's kappa alone.** These labels will be heavily skewed (most items agree), and kappa is well known to collapse under skewed marginals even at very high raw agreement — the "kappa paradox" (Feinstein & Cicchetti 1990; Byrt, Bishop & Carlin 1993). A judge at 95% raw agreement can post a kappa near 0.3 purely from prevalence. Report **all** of: raw agreement with a Wilson 95% CI, Cohen's kappa, **Gwet's AC1** (prevalence-robust, the right primary statistic here), and the confusion matrix with false-positive and false-negative rates broken out separately. FP and FN matter asymmetrically for our claim: a judge that is lenient toward *quantized* outputs specifically would manufacture our result, so **report judge error rates separately for bf16 vs 4-bit generations.** That check is the one a good reviewer will ask for.

**Gate:** proceed if raw agreement ≥ 0.90 **and** AC1 ≥ 0.7 **and** the bf16-vs-4-bit judge error rates are not significantly different. If the third condition fails, the judge cannot support H3/H4 — fall back to reporting rungs 1–3 as primary and the judge as a caveated robustness check. That fallback keeps the paper alive, so **this is not a blocking report-back** unless all rungs disagree with humans.

**Annotators — two-annotator protocol (confirmed available 2026-08-06).** This is the design a reviewer expects, so execute it properly:

1. **Double-annotate an overlap subset of ≥120 items**, drawn from the same stratified sample (all 3 models × bf16/4-bit × all tasks, oversampling scorer-disagreement cases). Both annotators label these **independently**, with no discussion until both passes are complete. Report **inter-annotator agreement on this subset**: raw agreement with a Wilson 95% CI, Cohen's kappa, and **Gwet's AC1** — the same bundle as the judge-vs-human statistics, and for the same prevalence reason (§ below).
2. **Split the remaining items** between the two annotators to reach the ≥250 total. Report per-annotator counts.
3. **Write the rubric before annotating and publish it verbatim** in the appendix and the artifact. Include the decision rule for every edge case: truncated output, multiple candidate answers, correct-but-never-stated, refusals, degenerate repetition. Draft it, dry-run it on ~10 items *together*, freeze it, then annotate independently. Rubric drift between annotators is the main threat to a two-rater design.
4. **Adjudicate disagreements explicitly and pre-specify the rule** — a third pass by joint discussion is fine, but **decide and record the rule before seeing the disagreements**, and report how many items were adjudicated. Belz et al. (Insights 2023) list "biased methods of aggregating judgments (choosing a preferred participant rather than using some form of average)" as a documented reproducibility failure. Never resolve by deferring to whichever annotator is the senior author.
5. **Report the judge-vs-human numbers against the adjudicated labels**, and state the IAA figure alongside them so a reader can see the human ceiling. A judge cannot meaningfully be held to better agreement than the humans achieve with each other — if IAA is 0.88 and judge agreement is 0.86, the judge is at the noise ceiling and the plan's §3 gate should be read in that light.
6. **Add an intra-annotator check only if it is free** — e.g. seed ~20 duplicate items into one annotator's queue and report test–retest agreement as a secondary number. Nice to have, not required now that IAA exists.

Blinding (below) applies to **both** annotators independently. Disclose the annotation design in the Responsible NLP Checklist section D (instructions, recruitment, whether annotators were authors, and that they were unpaid co-authors rather than crowdworkers) and describe the labels as adjudicated human reference labels with the IAA figure attached.

**Blind the annotation, and treat this as load-bearing rather than hygiene.** Annotators must **never** see (a) the judge's verdict, (b) the parser's verdict, or (c) which precision condition produced a generation. The strongest available evidence for why: Hróbjartsson et al. (*BMJ* 2012;344:e1119) studied 21 trials with **binary, subjective outcome assessments** — structurally our task — and found non-blinded assessors **exaggerated odds ratios by 36%** (pooled ratio of ORs 0.64, 95% CI 0.43–0.96). The mechanism is the alarming part: blinded and non-blinded assessors *agreed on 78% of assessments*, and the entire bias came from **misclassifying a median of 3% of items**. Our effects are small, so a bias hiding behind high agreement and a handful of flipped items could manufacture or erase the whole finding. Cite this paper when justifying the protocol.

Concretely: **omit the condition column from the annotation file entirely** rather than hiding it (a hidden column can be un-hidden), pre-shuffle items independently per annotator, verify the two shuffles actually differ, and publish the seeds — Belz et al. (Insights 2023) document "wrongly applied randomisation" as a real, recurring reproducibility failure, and found only 13% of human evaluations were reproducible at all. Ask the single binary question on its own screen (asking multiple criteria together inflates their correlations). Run a ~10-item blind practice round first; **prefer that to attention checks**, since excluding check-failures yields a biased subsample (van der Lee et al., INLG 2019).

**Do not pre-annotate with the judge and have humans "correct" it.** Anchoring is measured, not hypothetical: with low-quality pre-annotation the better annotator in Fort & Sagot (LAW IV 2010) dropped from 98.1 to 95.8, and annotators self-reported concentrating less. Note the trap this creates for us specifically — Mikulová et al. (LREC 2022) found pre-annotation *raised* IAA. **Higher agreement under pre-annotation is not evidence of higher validity; it is shared anchoring.** Since IAA is one of our headline numbers, showing the judge verdict would inflate the very statistic that is supposed to validate the judge.

**Tooling:** write a ~50-line local script (render generation + question, collect one keystroke, append JSONL). That is genuinely the right answer for 300 binary labels. If a platform is preferred, use **Potato** — it has a `judge_calibration` mode built for exactly this study (LLM labels held in a separate store and never injected into the UI, yielding accuracy, human↔LLM IAA, Cohen's κ, ECE, and confusion matrices) plus `/adjudicate` with `show_annotator_names: false`; cite Pei et al., EMNLP 2022 Demos. Avoid its `pre_annotation` block per the anchoring point above, and pre-shuffle per annotator yourself since its per-annotator ordering is unconfirmed. **Label Studio Community will not work** — agreement matrices, the review workflow, and per-task assignment are all Enterprise-only. Prodigy costs $390–1,950, computes no agreement statistics, and shows annotator names.

---

## 4. Analysis plan (pre-register this before looking at results)

Write this into `analysis/preregistration.md` and commit it **before** the main runs, so the paper can state the analysis was pre-specified.

> ### 4.0 ⚠ IDENTIFICATION FIX — must be pre-registered, cannot be patched later
>
> **The flaw.** §2.3 defines *Competence* as `P(correct | committed)` — given the answer the model actually committed to, is it right? But **whether the model commits is itself affected by precision.** Comparing competence across precision arms therefore **conditions on a post-treatment variable**: this is collider / selection bias, not a causal contrast. It is not a technicality — **the bias points toward this paper's own hypothesis**, so the more weight the centrepiece table carries, the more it looks engineered to confirm itself. An ARR reviewer who knows causal inference finds this on the first read. Unfixed, it **caps acceptance at ~25%**.
>
> **The fix — all three, pre-registered:**
> 1. **Primary competence contrast = the always-compliant stratum only.** Restrict to items where *every* arm (all precisions × all scorers) produced a committed answer. This is a pre-treatment-defined subgroup, so the contrast is identified. Report the stratum's size as a first-class number.
> 2. **Report the 2×2 compliance-transition matrix** per precision pair (committed→committed, committed→abstained, abstained→committed, abstained→abstained). This *is* a finding: compliance shift is a real quantization effect and belongs in the paper rather than being conditioned away.
> 3. **Manski bounds** on the full-sample competence effect, to show what the data can and cannot rule out without the stratum restriction. Wide bounds are an honest result.
>
> **Sequencing:** write this into `analysis/preregistration.md` **before any generations exist**, so the paper states it was pre-specified rather than patched after review. Retro-fitting it is worth far less.
>
> Note this interacts with §4.1: the interaction model is fit on *correctness*, which is well-defined for all items, so H3 is unaffected. Only the competence decomposition (H4) needs the stratum restriction.

### 4.1 Interaction test (H3)
Item-level logistic mixed-effects model, fit per task:

```
correct ~ precision * scoring + model + (1 | item)
```

The `precision × scoring` interaction coefficient **is** the artifact estimate. Report it as an odds ratio and as a marginal effect in accuracy points with a 95% CI.

Note on random effects: with only 3 models, do **not** put model family in as a random effect — 3 levels cannot support a variance estimate. Treat model as a fixed effect and item as the random effect. (The originating plan's §5 suggested crossed random effects; this is the honest version.)

Because the extraction axis re-scores *identical generations*, the strict-vs-permissive comparison is perfectly paired at the generation level. Also report the simple paired test (McNemar on discordant pairs) as a transparent, assumption-light companion to the GLMM.

### 4.2 Decomposition (H4)
Report, per cell, with CIs: compliance rate, truncation rate, competence-given-committed-answer, and the resulting additive decomposition of the strict delta into competence / compliance / truncation components. This table is the paper's centerpiece — design it early and let it drive what you log.

### 4.3 Multiple comparisons
Pre-specify one omnibus contrast per hypothesis rather than exhaustive pairwise slicing. Holm–Bonferroni within each hypothesis family. Report effect sizes with CIs for everything; never report a bare significance flag.

### 4.4 Powered-ness audit (H7)
For each published degradation figure we cite, compute the 95% CI implied by that paper's own eval size and state whether the reported delta clears it. Also compute, for our own item counts, the minimum detectable effect at 80% power. Be scrupulous and neutral in wording — this is a methodological observation about statistical resolution, not an accusation.

### 4.5 Noise floor (mandatory, do not skip)
Run the **bf16 baseline twice** end-to-end with different seeds on every model. The spread between those two runs is the pipeline's own noise floor. **Any effect smaller than that floor cannot be claimed by us either** — and reporting it is exactly the kind of self-application of the paper's own standard that earns reviewer trust.

### 4.6 Determinism — read this before generating anything

vLLM greedy decoding is **not** reproducible by default, and vLLM says so outright: *"vLLM does not guarantee the reproducibility of the results by default, for the sake of performance."* Its FAQ confirms the exact failure mode that would sink this paper — identical prompts produce different output depending on **how the scheduler batched them**, because batching changes float reduction order, and once one token differs the trajectories diverge. Issue #9567 reported precisely this on an INT8-quantized Llama-3-8B and tied it to unstable lm-eval scores across KV-cache sizes. **Our entire paper is about small accuracy deltas, so uncontrolled batch nondeterminism could manufacture or erase our effect.**

Since v0.11.1 vLLM ships `VLLM_BATCH_INVARIANT=1` (beta), which overrides the relevant GEMM/softmax/reduction ops with batch-invariant Triton kernels. **All three models under test and both judges are on its explicitly validated list.**

**But there is a gap that lands directly on us, and it is the single most important technical finding in this plan.** Batch invariance is *not* validated against the quantized kernels we depend on:
- `auto_awq.py` **has** a batch-invariance guard, with the source comment *"Marlin kernels are not batch invariant"* — under BI mode, AWQ silently falls back to slower non-Marlin kernels.
- `auto_gptq.py` and the `choose_mp_linear_kernel()` dispatcher GPTQ actually uses contain **no batch-invariance handling at all** — so GPTQ keeps using Marlin, which vLLM's own comment says is not batch-invariant.

Two consequences, both mandatory:
1. **Measure bitwise reproducibility per quantization format before trusting any delta.** Rerun one identical cell per format and report the disagreement rate. If GPTQ is nondeterministic while bf16 is not, then *differential* numerical noise exists between arms and must be reported as a floor on every claim.
2. **Never reuse BI-mode timings as quantized throughput numbers** — the kernel changed underneath you.

**Turn this into a contribution.** "Which quantized serving configurations are bitwise reproducible under a batch-invariant engine, and what does that imply for published accuracy deltas?" is a genuine, publishable, self-contained result that no one has reported, and it is directly on-thesis: it is *another* way the measurement apparatus, not the model, moves the number. Run vLLM's own `tests/v1/determinism/test_batch_invariance.py` per format and report the table. Workaround if GPTQ proves nondeterministic: `VLLM_DISABLED_KERNELS=MarlinLinearKernel,MacheteLinearKernel` forces a fallback kernel — slower, but tests whether determinism is restored.

**Settings that actually matter** (the rest is cargo cult):
- `tensor_parallel_size=1` — **most important**; BI is broken under TP>1 (open issue #51290). Our models all fit on one GPU anyway.
- `enable_prefix_caching=False` — prefix caching changes greedy output (open issue #39389) and is explicitly *not* covered by BI.
- `VLLM_BATCH_INVARIANT=1`, `VLLM_ENABLE_V1_MULTIPROCESSING=0` (offline), FLASH_ATTN backend pinned, `seed=0`, `temperature=0`, async scheduling off.
- `max_num_seqs` fixed **and reported in the paper** — it is the only knob that resolved open issue #51187, where 44 co-resident sequences produced 3 distinct outputs across 5 repeats while `max_num_seqs=8` gave 5/5 identical. Prefer a low value and accept the throughput cost.
- **Do NOT set `enforce_eager=True`** — CUDA graphs became BI-compatible in PR #27660; it is pure throughput loss. (This corrects my own earlier draft.)
- Residual known leak: RMSNorm block size varies with token count, defeating BI for sequences ≥256 tokens (open issue #48271) — squarely in our 200–600 token output range. Another reason to *measure* rather than assume.

Budget **1.2–2× wall-clock** for BI mode on Ampere (it must swap in Triton GEMMs on SM80; Hopper is cheaper). Pin vLLM, torch, CUDA, and GPU model — reproducibility holds only within identical hardware and version.

---

## 5. Package architecture

Python 3.11+, `uv` for env management (not installed on this machine — install it; the system Python is 3.8 via Anaconda and is unsuitable). Everything heavy runs on the rented GPU box (§6), not locally.

```
quantartifact/
  pyproject.toml              # every dep version-pinned; uv.lock committed
  configs/
    models.yaml               # repo IDs, revision SHAs (pin commit SHAs, not tags)
    quant.yaml                # method, bits, calibration set/size/seqlen per cell
    tasks.yaml                # item counts, subsample seeds, prompt templates
    scoring.yaml              # the 4 rungs, declaratively
  src/quantartifact/
    models.py                 # load/serve; one vLLM engine per (model, precision)
    quantize.py               # GPTQModel / llmcompressor wrappers
    tasks/                    # dataset loading, subsampling, prompt rendering
    generate.py               # the ONLY component that touches a GPU for generation
    scoring/
      extract.py              # rungs 1-3, pure functions, no I/O
      judge.py                # rung 4
      compliance.py           # the 2x2 labeling of §2.3
    analysis/                 # GLMM, decomposition, plots, tables
  scripts/                    # thin CLI entry points; one job = one command
  artifacts/
    generations/              # JSONL, one file per cell, append-only, gzipped
    scores/                   # JSONL, one file per (cell x scorer_version)
    figures/ tables/
  tests/                      # scorer unit tests are the highest-value tests here
  paper/                      # LaTeX, ACL template
```

**The one architectural rule that matters: generate once, score many times.** Generation is the expensive, nondeterministic, GPU-bound step. Scoring is cheap, deterministic, and will be revised repeatedly. Never couple them. Every generation record must be complete enough that *any* future scorer can run over it without regenerating:

```jsonc
{
  "run_id": "...", "cell_id": "qwen2.5-3b|gptq-4bit",
  "model_repo": "...", "model_revision": "<commit sha>",
  "quant_method": "gptq", "bits": 4, "quant_config_hash": "...",
  "task": "gsm8k", "item_id": "...", "gold": "18",
  "prompt_variant": "fewshot_cot", "decoding": "greedy",
  "sample_idx": 0, "seed": 1234,
  "prompt_text": "...", "prompt_sha256": "...",
  "generation_text": "<FULL text, never truncated for storage>",
  "n_prompt_tokens": 812, "n_gen_tokens": 256,
  "finish_reason": "length",          // REQUIRED - truncation analysis depends on it
  "stop_reason": null,
  "engine": "vllm==0.26.0", "env_hash": "...", "wall_ms": 1830
}
```

Scorer outputs are written to separate files keyed by `(cell, scorer_name, scorer_version)`. Bump the version on any change; never overwrite. This makes the whole analysis re-runnable from stored artifacts with no GPU — which is also exactly what the reproducibility release needs (§8).

**Toolchain, verified against PyPI and GitHub today:**
- **`auto-gptq` and `autoawq` are BOTH GitHub-archived.** AutoGPTQ's own README: *"AutoGPTQ is unmaintained - we suggest using GPTQModel."* AutoAWQ's: *"officially deprecated… adopted by the vLLM Project: llm-compressor."* The originating plan implicitly assumed these.
- **Use `llmcompressor` 0.12.0.1 for both GPTQ and AWQ.** One tool for both arms means one calibration code path, which removes a confound. `gptqmodel` 7.3.2 is the alternative for GPTQ specifically.
- ⚠️ **`autoawq` downgrades transformers to 4.47.1** — it will wreck a modern environment. And `llmcompressor` pins `compressed-tensors==0.17.1` exactly with `transformers<=5.10.1`, while `gptqmodel` tracks 5.14. **Use two separate venvs** (one for quantization, one for serving/eval).
- `vllm` 0.26.0, `bitsandbytes` 0.50.0, `math-verify` 0.9.0, `lm-eval` 0.4.12. **Pin everything.**
- ⚠️ **`huggingface-cli` was removed in huggingface_hub 1.0.0** — the command is now `hf`. `use_auth_token` → `token`; `resume_download` removed.
- **None of GPTQ / AWQ / bitsandbytes runs usefully on Apple Silicon.** vLLM's CPU backend supports AWQ and GPTQ on **x86 only**, and macOS is FP32/FP16 only — so there is no local fallback, not even a bf16 reference. The M3 Pro is a dev/analysis box.

**Calibration — the most likely reviewer objection, so get it right.** `llmcompressor`'s own official examples differ by **8× in token budget** between methods (GPTQ: 512 samples × 2048 seqlen, symmetric W4A16; AWQ: 256 × 512, asymmetric W4A16_ASYM). **If you accept those defaults, your GPTQ-vs-AWQ delta is uninterpretable.** Normalize both arms to one budget — 512 samples @ 2048, seed pinned — and state it in the paper.

Quantization is cheap: ~30 min per arm for an 8B on an A100, ~10–15 min for a 3B, and it needs only **~1 GB VRAM** (sequential layer onloading is default; the 16 GB requirement is CPU RAM). Quantize on the same rented box.

**Checkpoints.** Verified on the Hub:
- Qwen: `Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4`, `-GPTQ-Int8`, `-AWQ` — all **official Qwen-published** with consistent settings (`group_size:128, desc_act:false`). Use these to **validate our self-quantization pipeline** by reproducing them.
- Llama-3.1-8B: prefer `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16` and `.w8a16` — they **publish measured accuracy** (GSM8K 82.8→82.9, ARC-C 81.4→80.2, BBH 30.1→28.9), which is a free sanity anchor: our deltas should land near these or the harness is wrong. ⚠️ Avoid `hugging-quants/…-GPTQ-INT4` as primary — it uses `desc_act:true`, inconsistent with Qwen's `desc_act:false`, which is a silent confound.
- **Phi-3.5-mini has no credible pre-quantized checkpoints** (best GPTQ candidate has 83 downloads; two repos returned 401). Self-quantize all Phi-3.5 arms.
- **Recommendation: self-quantize all three models with identical settings** for internal consistency, and report the official Qwen/RedHat checkpoints as a documented cross-check. This is the detail that determines whether a quantization result replicates.
- ⚠️ **Qwen2.5-3B is `license:other` / qwen-research, NOT Apache-2.0** (unlike the 7B/14B). Note it in the artifact release.
- ⚠️ **`meta-llama/Llama-3.1-8B-Instruct` is `gated: manual`.** The user reports having access (2026-08-06), which is likely correct — but **it is unverified on this machine and must be confirmed before it becomes a blocker.** Checked 2026-08-06: there is **no HuggingFace token** anywhere on this box (`~/.cache/huggingface/token`, `~/.config/huggingface/`, `HF_TOKEN` env — all absent), `huggingface_hub` is not importable from the default Python, and the hub cache contains no Llama model (only BioMistral, BERT variants, ModernBERT). An unauthenticated fetch of the config returns **HTTP 401**. So access may exist on the *account* while this environment cannot yet use it.
  **Day-1 verification (2 minutes, do it before anything else depends on it):**
  ```
  uv tool install "huggingface_hub[cli]"     # note: the command is `hf`, not `huggingface-cli`
  hf auth login                              # paste a token with `read` scope
  hf auth whoami
  hf download meta-llama/Llama-3.1-8B-Instruct --include config.json
  ```
  A successful `config.json` download confirms both the token and the license grant. **A 401/403 means the license request is still pending or was never granted** — in which case request it immediately in a browser, since manual approval can take days and it sits on the critical path to the replication gate (§1.2). The token will also be needed on the rented GPU box, so store it where the provisioning script can read it (an env var, not a committed file).
  **If access turns out to be unavailable or slow:** proceed with Qwen2.5-3B and Phi-3.5-mini first — neither is gated, and the pipeline, scorers, and determinism audit can all be built and validated without Llama. Only the Kurtic et al. replication gate specifically needs Llama-3.1-8B. Do not let this block week 1.
- Red Hat's tip, worth heeding: pass **`add_bos_token=True`** to lm-eval — quantized models are BOS-sensitive.

**One more confound a reviewer will raise: the serving backend itself.** arXiv:2605.19537 (May 2026) finds **benchmark scores shift up to 16.6 percentage points from inference-backend choice alone**, with weights, decoding, and hardware fixed. Since bitsandbytes NF4 is quantized in-flight by vLLM while GPTQ/AWQ load pre-quantized checkpoints, a backend difference could confound the bit-width comparison. **Hold the backend fixed (vLLM only, one version) across every arm, and state it explicitly.** Do not mix vLLM and HF `transformers` generation anywhere in the study.

---

## 6. Compute

**Local machine is not viable for experiments:** Apple M3 Pro, 36 GB unified memory, no CUDA, and **only ~14 GB free disk** (97% full). Model weights alone need ~120–150 GB.

Rent **one A100-80GB, `tensor_parallel_size=1`** (single GPU is also required for determinism — §4.6). Verified pricing as of today: RunPod Community **$1.19/hr**, Verda spot A100-40G **$0.645/hr**, L40S ~$0.79/hr. Provision a **200 GB network volume**.

**Compute is not the constraint.** Measured-throughput estimates put the full matrix at roughly **25 A100-hours**, or **~40–50 hours with batch invariance enabled** — on the order of **$50–100 all-in**. Spend freely on rerunning cells and on BI mode; the scarce resource is the nine weeks, not the money.

**Avoid AWS** unless quota already exists: it is 2–7× the specialist providers ($3.43/GPU-hr for p4de A100-80G), charges egress, and — critically — **every GPU vCPU quota defaults to zero**, with days-to-weeks of support lead time. That is a schedule risk, not a line item. Note p5/H100 spot currently prices *above* on-demand.

Provider gotchas worth knowing: RunPod spot currently offers **no discount** over on-demand, and its volume storage **doubles to $0.20/GB-mo while a pod is stopped**. Vast.ai's headline rates are cheapest but bandwidth is per-host and invisible in the quoted rate (median $4/TB egress, p90 $21/TB) — filter on `verified` + `reliability2>0.98`. Verda offers a clean flat 50% spot discount, the most predictable interruptible pricing found.

**Do not expect 4-bit to run faster.** This is counterintuitive and worth internalizing: at the large batch sizes we'll use, W4A16 is memory-bound-optimized but decode is compute-bound, so measured speedup is **~1.08× and is actually *slower* than bf16 on some tasks** (Kurtic et al., ACL 2025, Table 6 — our own replication target). Choose GPUs on VRAM fit and price, not hoped-for quantization speedups. If throughput becomes a problem, the measured winner on Ampere is **W8A8-INT8 at 1.38×** — and adding that arm would strengthen the paper anyway.

**Disk (measured from the HF tree API):** Qwen2.5-3B bf16 6.17 / GPTQ-4 2.07 / GPTQ-8 3.47 / AWQ 2.69 GB; Llama-3.1-8B bf16 15.07 / GPTQ-4 5.74 / AWQ-4 5.73 / w8a16 9.08 GB; Phi-3.5-mini bf16 7.64 GB (+~9 GB self-quantized); judges ~34 GB. With cache overhead and outputs: **~136 GB → provision 200 GB.**

⚠️ **Llama-3.1-8B repo trap:** the repo totals **32 GB**, not 15, because it ships a duplicate `original/consolidated.00.pth`. Pass `ignore_patterns=["original/*"]` or you download twice the bytes.

**bnb NF4 needs no separate checkpoint** — vLLM quantizes in-flight from bf16 via `quantization="bitsandbytes"`. But **do not use bitsandbytes for any throughput claim**: vLLM's own open RFC #39583 proposes moving it out-of-tree, noting zero CUDA kernels and that "performance is not great."

Free the local disk regardless (14 GB is too tight for a LaTeX build plus analysis) and keep only JSONL artifacts and figures locally.

### 6.1 SMInterface (house SageMaker substrate) — an explicit decision, not a default

The house substrate for compute and job submission is the local package at
**`/Users/johnyj02/workspaces/SMInterface`** (Brazil workspace; package at `src/SMInterface`, python module
`sm_interface`). It wraps a local source dir + hardware spec into a SageMaker Training Job and returns a handle
to poll and download artifacts from. **The expert planner owns the HOW of job submission.**

**⚠ This conflicts with §6 above, and the conflict must be resolved explicitly rather than by drift.** §6 recommends renting RunPod/Verda and **avoiding AWS**, for verified reasons: 2–7× the cost, egress charges, and **every GPU vCPU quota defaults to zero** with days-to-weeks support lead time. Those reasons still stand. So:

**Decision rule — pick one on day 1 and record it:**
- **If AWS GPU *training* quota already exists** on the `obtmodel` account for the target instance type → use SMInterface. It gives idempotent re-attachable jobs, S3 artifacts, and no per-hour babysitting, which suits a batch sweep well.
- **If quota is zero or unknown** → **rent RunPod/Verda per §6 and do not block on quota.** Quota lead time is days-to-weeks against a compressed runway; this is the single most likely schedule killer. **Query quota in hour one** — it is a two-minute check with multi-day latency, so it belongs alongside the other hour-one external dependencies.

**If SMInterface is used, these defaults are fatal as shipped** (verified in `src/sm_interface/config.py`):
- **`max_run` defaults to 3600 s and silently kills the job at one hour.** A 20 h shard needs `max_run: 79200`.
- **`volume_size` defaults to 30 GB.** §6 measured ~136 GB of weights → set **200**.
- **Only `/opt/ml/model` survives** into `model.tar.gz`. Every field the analysis needs must land there — including `finish_reason` and `n_gen_tokens` per §5, which are **unreconstructable after the fact**.
- **Batch only** — no REPL, no mid-run inspection beyond CloudWatch. Budget throwaway smoke jobs.
- **No built-in fan-out** — one `JobConfig` = one job; a sweep needs a thin driver (this is a named deliverable).
- **Spot is hostile here**: `max_wait` auto-defaults to `max_run`, leaving no capacity-wait budget; preemption is uncompensated unless checkpointed.
- **`framework_version` defaults to "2.3"** (torch 2.3.x / CUDA 12.1) while modern vLLM builds against CUDA 12.8/12.9 and pulls its own torch — a day-1 dependency risk on a non-interactive substrate. Escape hatch (`framework: generic` + custom vLLM image in ECR) is a 1–2 day Docker task.
- **Per-job pip install is 8–15 min** × N jobs — bake the image or accept the tax explicitly.

**Delegated to the expert planner:** job granularity (which axes sweep *inside* one container vs *across* jobs — the highest-leverage cost/wall-clock/blast-radius decision); per-class `instance_type` / `max_run` / `volume_size`, derived from the **day-3 measured calibration run**, not guessed; the fan-out driver contract and how overrides travel (`hyperparameters` dict vs config staged to S3); deterministic job naming, idempotence and resume; retry policy; and weight provisioning (pre-staged pinned S3 weights vs in-container HF download, pinned by commit hash either way).

**Start from** `src/SMInterface/examples/lightweight-training/` (copy and mutate `job.yaml` / `train.py` /
`requirements.txt`; do not start from the SageMaker SDK). Also read `src/sm_interface/config.py` (authoritative
defaults), `runner.py` (`submit`/`attach`/`JobHandle`; note `download_model` has no credential refresh), and
`SMInterface/CLAUDE.md` — **edit on the Mac only**; ninja-dev-sync pushes to the dev desktop (~5 s), and SSH is
for running commands, never editing.

**Determinism caveat:** §4.6 requires `tensor_parallel_size=1` on a single GPU. Whichever substrate is chosen must pin one GPU per job — multi-GPU tensor parallelism breaks batch invariance (vLLM #51290).

---

## 7. Deliverable: the paper

8-page ACL long paper, LaTeX ACL template only (**Word submissions have been desk-rejected since March 2026**). Unlimited references and appendices; appendices must use the double-column format and must not carry load-bearing content — hiding substantive material there is itself a desk-reject ground, and reviewers are not expected to read them.

Working title direction: frame as **decomposition/attribution**, not debunking. Something like *"How Much of Reported Quantization Degradation Is Measurement? Decomposing Accuracy Loss into Competence, Compliance, and Truncation."*

**The precise claim to make** — narrower than "degradation is overstated," and the only version still open after Kurtic et al.:

> *The residual reported degradation, after quantization-side hyperparameters are properly tuned, is further attributable to evaluation-protocol factors — and protocol variance exceeds bit-width variance on reasoning tasks at 1–8B.*

**Three claims we must NOT make**, each already owned: "first to combine LLM-judge with quantization evaluation" (at least four papers predate us, earliest 2024); "first factorial quantization evaluation" (arXiv:2606.29581 has *Factorial* in its title and runs 144 configurations over ~2M responses); "first to show accuracy hides behavioral divergence in compressed models" (Dutta et al., NeurIPS 2024, owns "flips").

⚠️ **The field's center of gravity runs against us, and this belongs in the introduction, not the limitations.** The dominant published finding is that benchmarks **understate** quantization damage, not that they inflate it. The strongest instance: Marchisio et al. (Findings of EMNLP 2024, arXiv:2407.03211) found a **1.7% automatic drop in Japanese was a 16.0% drop according to human raters**, and that "mathematical reasoning degrades fastest." Also pushing this direction: Wen et al. (arXiv:2606.17609, MC benchmarks overstate pruned-model usability), Oladri et al. (arXiv:2607.09999), and Ren et al. (arXiv:2608.02442, which finds **8.2–44.1% of correct-graded answers are shortcut-hacked** — i.e. lenient extraction *over*-credits too). **Frame our contribution as *calibrating* extraction, not loosening it.** A paper that only argues "be more lenient" walks straight into this literature.

⚠️ **Also pre-commit on this:** "small models are genuinely more fragile to quantization" is replicated by at least four independent groups (arXiv:2409.11055, arXiv:2606.21257, arXiv:2607.09999, arXiv:2606.29581). If protocol factors explained small-model degradation away *entirely*, we would be contradicting all of them. **Pre-register that we report the residual effect after protocol correction, not merely its attenuation.** Reporting "protocol accounts for X of the gap, and Y remains real" is both more credible and more useful than a debunking.

**Adopt two methodological standards now visible to our reviewers:** the pre-registered frozen-budget sweep design from arXiv:2608.04160 (which hard-capped 540,000 decodes with Holm correction and treats the output cap as an independent variable — directly applicable to H1), and the paired McNemar test from Kübler et al. (ICLR 2026).

**Publication velocity is a real risk:** title-level quantization+reasoning papers went from 6 in all of 2025 to 19 in the first seven months of 2026, and two directly relevant papers appeared in the first five days of August 2026. Also monitor **SLiM-Eval** (OpenReview `ssrs45kran`, ARR Jan/Mar 2026, 2.4B–6.7B, GSM8K degrades most under quantization) — it is our nearest neighbor on model band and may surface at ACL 2026.

**Related work must differentiate precisely** — the framing risk is real and this section is where the paper is won or lost:
- **arXiv:2606.25519 vs Liu et al. (COLM 2025, arXiv:2504.04823)** — the live contradiction on whether quantization lengthens CoT. This pairing motivates H1; present it early as the opening puzzle.
- **arXiv:2606.00206** — the 52%-recoverable-failure finding we extend. Credit it clearly; our addition is the measurement question, not the observation.
- **Hua et al., EMNLP 2025 (arXiv:2509.01790)** — the same artifact move on prompt sensitivity. Position as "Hua et al. for quantization," and engage their MATH null result honestly.
- **arXiv:2608.03854** — protocol dominates the quantization effect at 7B for classification. Convergent evidence, not a gap we fill.
- **arXiv:2605.19537** — backend choice alone moves scores up to 16.6pp. Cite when stating that we hold the backend fixed.
- **arXiv:2601.14277** — published the strict/flexible numbers in an appendix without interpreting them. Cite as motivating evidence that the field reports these numbers without noticing them. Be generous and factual, not dismissive.
- **Jo et al., arXiv:2510.14773** — owns extraction sensitivity in general; we add the quantization interaction they don't test.
- **Kurtic et al., ACL 2025** — already argued degradation was overstated, via hyperparameters. We supply a different, orthogonal mechanism. **Cite prominently and early**; do not let a reviewer discover it.
- **"Accuracy is Not All You Need" (NeurIPS 2024, arXiv:2407.09141)** — argues the *opposite direction* (flat accuracy hides real drift; flips up to 13.6%). **Address in the intro, not buried in related work.** Both are true: accuracy deltas can simultaneously over-penalize formatting and under-detect behavioral change. Saying so plainly is a strength.
- **Jin/Lee et al., IJCAI 2025, arXiv:2409.11055** — quantization disrupts instruction-following. This is the strongest counterargument; our compliance/competence split is the direct answer.
- **"Silent Failures / Hollow Convergence" (arXiv:2607.09999, Jul 2026)** — verified real. Correct the title (the originating plan drops the trailing *"and Failure Mode Shifts"*). Caveat: unrefereed, 0 citations, and it tests **only FP32/FP16/NF4 — no GPTQ/AWQ**, so it does not support claims about those methods.
- **UniComp (arXiv:2602.09130)**, **SLMQuant (arXiv:2511.13023**, ACM DOI 10.1145/3746262.3761973 — peer-reviewed), **"Quantized but Deceptive?" (2025.emnlp-main.1548** — this is **truthfulness only, not safety**; the originating plan mischaracterizes it).
- **Schaeffer et al., "Mirage" (NeurIPS 2023** Outstanding Paper, confirmed) — methodological template. Cite the citation count with the index named (~1,160 Google Scholar, Aug 2026) or not at all; the number varies ~30× across indexes.

Also cite as evaluation-fragility background: FormatSpread (Sclar et al., ICLR 2024), xFinder (ICLR 2025), EleutherAI "Lessons from the Trenches" (arXiv:2405.14782), and the HF Open LLM Leaderboard v2 blog (which documents models scoring **0 on GSM8K purely from output-format mismatch** — a perfect real-world instance of our mechanism).

**Mandatory:** Limitations section (no page cost, absence = desk reject), and a complete Responsible NLP Checklist — since December 2024, incorrect or incomplete filings are themselves a desk-reject ground. Section C (compute, hyperparameters, error bars) and Section E (AI assistance) both apply to us directly. Disclose AI assistance per policy; metadata stays editable for 48h after the deadline.

---

## 8. Reproducibility artifact

Release: generation pipeline, quantization configs with calibration details, all four scorers, the analysis scripts, and **the full generation JSONL** (this is the genuinely reusable artifact — it lets anyone re-score with a new extractor without a GPU). Plus the human-annotated judge-validation set with agreement statistics, which stands alone as a citable judge-reliability reference set for quantization eval work.

Pin model commit SHAs (not tags), every library version, calibration dataset and sample counts, all decoding hyperparameters, and hardware. Version-pinning is load-bearing here given §4.6.

**arXiv preprinting is permitted** — ARR has had no anonymity period since Feb 2024. But the submission itself must be anonymized, and anonymous repos must not be on Dropbox or Google Drive (download tracking). Note there is an *optional binding* no-preprint election that grants extra award eligibility and stated preference for borderline papers; if elected, preprinting before meta-reviews is a desk reject. Default to **not** electing it unless the user asks.

---

## 9. Schedule — 9.5 weeks, Aug 6 → Oct 12

Front-load the gates. Every one of them can force a scope change, so none should be discovered late.

### 9.1 Week 1 (Aug 6–12) — infra, pilot, estimates
**Day 1, in this order:** (1) **verify** Llama-3.1 access with the four commands in §5 — the user believes access exists but there is no token on this machine, so confirm it rather than assume, and if it 401s, request it in a browser immediately; (2) confirm **both** authors' OpenReview profiles are complete and set an Oct 14 reminder (§9.6); (3) install `uv`; (4) if AWS is even a possibility, file GPU vCPU quota increases.

**Also week 1, since it gates §3 and involves another person's calendar:** draft the annotation rubric, do the ~10-item joint dry-run with the co-author, and freeze it. Booking a second person's time is the kind of dependency that silently slips; getting the rubric frozen early also means the judge validation in week 2 isn't waiting on it.

Then: stand up the GPU box, install and pin the toolchain in **two venvs** (quantization vs serving), verify `llmcompressor`/vLLM interoperate. Build generation + the four scorers with unit tests. Run one small cell end-to-end (Qwen2.5-3B, bf16, GSM8K 100 items) and **measure real generations/hour** with `vllm bench throughput` on the actual instance — the published throughput estimates are ±40% and no measured public figure exists for a 3B model.

**Also in week 1, before any main run: the determinism audit (§4.6).** Run vLLM's `tests/v1/determinism/test_batch_invariance.py` per quantization format and rerun one identical cell per format to measure the disagreement rate. This gates everything downstream — if GPTQ is nondeterministic while bf16 is not, differential numerical noise exists between arms and every later claim needs that floor. Do this before spending compute on runs whose deltas you can't trust.

Deliver the **loose estimate** in the first two days and the **tight estimate** once throughput is measured, each split by phase, with dollar cost and calendar time reported separately (per §0.1 of the originating plan). **Report immediately if the tight estimate doesn't fit the runway.**

### 9.2 Week 2 (Aug 13–19) — replication + judge gates
Retargeted replication (§1.2): Kurtic et al. primary, the 2601.14277 strict/flexible gap secondary. Judge validation (§3) including human annotation. Measure the noise floor (§4.5) and residual nondeterminism (§4.6). Commit the pre-registration (§4) **before** main runs.

### 9.3 Weeks 3–4 (Aug 20 – Sep 2) — main runs
All 15 cells × reference condition, then the generation-costing variants in **this** priority order: **token budget (H1, the paper's lead) → prompt format (H5) → self-consistency (decoding)**. Self-consistency is deliberately last and is the designated cut: it is ~57% of the compute, GSM8K is saturated as a voting target (Appendix A), and it supports the least novel arm. Score continuously as generations land; don't batch scoring to the end.

For the token-budget sweep, follow arXiv:2608.04160's design: **freeze the budget levels before looking at results**, run hard-capped decodes at each level, and treat the cap as an independent variable rather than an implementation detail. Log `finish_reason` and token counts on every generation (§5) or this analysis is impossible after the fact.

### 9.4 Week 5 (Sep 3–9) — analysis
GLMMs, the decomposition table, the powered-ness audit, all figures and tables. **Freeze results at the end of this week.** Figures should be near-final before drafting starts.

### 9.5 Weeks 6–7 (Sep 10–23) — draft
Full draft including Limitations and the checklist. Related work gets disproportionate attention (§7).

### 9.6 Weeks 8–9 (Sep 24 – Oct 7) — external review + revision
The co-author is the first line of review, but a co-author who has been inside the project all along cannot see it the way a reviewer will. **Line up at least one genuinely external reader by mid-September** (ideally someone who knows either quantization or evaluation methodology) and get them the draft by Sep 30. Ask them specifically to attack the related-work differentiation (§7) — the paper's largest rejection risk, since three published papers occupy adjacent ground — and the judge validation (§3). Revise. Freeze Oct 7.

### 9.7 Oct 8–12 — submission buffer
Anonymize, verify formatting compliance against every desk-reject ground, submit early. **Do not use this buffer for science.**

**Author obligations — resolved.** Confirmed 2026-08-06: **two authors, and the co-author will register** (§0). **Action in week 1, for both authors:** confirm each OpenReview profile is complete (affiliation, Semantic Scholar / DBLP / ACL Anthology links, working email) — an incomplete profile blocks registration, and registration within 48h of submission is what prevents the auto-desk-reject. Do this in week 1, not October: profile links can take time to resolve, and the deadline lands two days *after* submission when attention is lowest. Put a hard calendar reminder on **Oct 14** for both people.

**Author list is frozen at submission.** ARR requires justification to add an author afterward, and removal is near-impossible ("except in extremely rare circumstances") since it requires removing their content. So settle the co-author's involvement now, before the annotation work in §3 makes them a substantive contributor.

**Also confirm:** the COLING 2027 commitment deadline is listed as Dec 20, 2026 by ARR but Dec 23, 2026 on COLING's own site. Plan against Dec 20. The Oct 12 cycle feeds **NAACL 2027** (June 1–5, San Francisco) and COLING 2027; venue choice happens *after* meta-reviews, not at submission.

⚠️ **Correction to an earlier draft of this plan:** Oct 12 is **not** the last chance at a 2027 archival venue. The Aug 3, 2026 cycle has closed, so Oct 12 is the last route to *NAACL/COLING 2027* — but the **January 2027 cycle feeds ACL 2027** (Aug 17–22, Kyoto). A paper rejected in the October cycle can be revised and resubmitted in January and still reach a 2027 venue. **Do not treat Oct 12 as a one-shot deadline** (see §11).

---

## 10. Report-back triggers

Proceed autonomously on everything else; the decisions in this document are made.

1. **Tight estimate doesn't fit the runway** (§9.1) — report immediately with proposed cuts. Cut order: self-consistency arm → GPTQ-8bit → TriviaQA → BBH MC contrast tasks. **Never cut** the token-budget arm (H1 — now the paper's lead), the negative controls (H6), the bf16 double-run noise floor, or the determinism audit; all three are structural to the argument. Note the binding constraint is engineering wall-clock, not dollars — compute is ~$50–100.
2. **Replication gate fails** after reasonable debugging (§1.2) — most likely causes are calibration differences, prompt-template mismatch, or chat-template handling. If Kurtic et al. won't reproduce, that changes the paper's framing and needs a decision.
3. **No judge configuration passes §3**, *including* the fallback of reporting rungs 1–3 as primary. If the deterministic rungs work, this is not a blocker — proceed and caveat.
   Related, non-blocking: if inter-annotator agreement on the §3 overlap subset comes in low (say raw agreement below ~0.85), that is a **rubric problem, not a judge problem** — revise the rubric, re-run the dry-run together, and re-annotate before drawing any conclusion about the judge. Report it if it persists, since the human ceiling bounds what the judge can be held to.
4. **A HIGH-overlap paper appears.** Novelty in this subfield decays in weeks: ~1 new quantization-reasoning paper every 1–3 weeks. Re-run a dated novelty scan in **week 1** and again in **week 5 before drafting**. Specifically watch for anyone crossing extraction sensitivity with quantization.
5. **The interaction is null across all tasks and axes.** Not a blocker — this is a publishable confirmatory result (§1.1) and the truncation result (H1), the decomposition (H4), and the powered-ness audit (H7) still yield content. Report it so framing can be adjusted early, and do not let it read as failure.
6. **Quantized generation proves badly nondeterministic** (§4.6) — i.e. the per-format disagreement rate is large enough to swamp the effects we're trying to measure, and neither batch-invariant mode nor `VLLM_DISABLED_KERNELS` fixes it. Report with the measured numbers. This is not necessarily bad news — it may become a headline result rather than a blocker — but it changes what the paper claims, so surface it rather than deciding alone.
7. Genuinely new forks not covered here — characterize options with minimal recon, then report.

---

## 11. Realistic expectations — read this if the rejection risk feels high

Rejection is the modal outcome for *any* ACL submission, and that is a fact about the venue, not a verdict on this paper. The verified numbers (Aug 2026):

| Venue | Submissions | Main | Findings | **Combined** |
|---|---|---|---|---|
| ACL 2025 | 8,360 | 20.3% | 16.7% | **37.0%** |
| ACL 2026 | 12,148 | 18.9% | 17.8% | **36.7%** |
| EMNLP 2025 | 8,174 | 22.2% | 17.3% | **39.5%** |
| NAACL 2025 | 3,246 | 22% | 15% | **37%** |

**The number to plan against is ~37% combined, not ~19%.** Over half of all acceptances now land in Findings, and Findings meets every criterion in the project's stated goal: archival, ACL Anthology, Google Scholar-indexed, real DOI (`10.18653/v1/...`), ISBN, `@inproceedings` citation. The founding preface states outright that *"papers published in Findings count as full publications."* The honest asterisk is prestige, not status — Findings has no Google Scholar Metrics h5-index, and ACL's own Exec is currently debating its recognition. Note also that Findings **is** archival, so accepting it forecloses submitting that work elsewhere.

Measured against *committed* papers rather than all submissions, ACL 2026 was 35.0% main / **67.9% combined** — the ~19% headline includes withdrawn and desk-rejected papers, a large fraction of which never had a serious chance.

**October 12 is not a one-shot deadline.** A paper reviewed in the October cycle and not accepted can be revised and resubmitted — "you can resubmit a paper as many times as you like" — and the **January 2027 cycle feeds ACL 2027** (Kyoto, Aug 17–22). ARR reviews have **no published expiration**. So the realistic frame is *two shots at a 2027 venue*, not one. The constraints:

- Meta-review scores run **1–5 in half-point steps** (5 = award consideration, 4 = Conference, 3 = Findings, 2 = resubmit next cycle, 1.5 = resubmit after next cycle, 1 = never resubmit). **A score ≥2.0 permits next-cycle resubmission; 1.5 forces a wait; 1 is terminal.**
- Resubmission requires a link to the prior submission (omitting it is a desk reject) plus a point-by-point revisions PDF.
- **Withdrawing more than 48h after the deadline blocks resubmission until the second following cycle** — so if the paper is going nowhere, either leave it in review or withdraw within 48h.
- Oct 20 → Jan is a genuinely short revision window, which is the argument for finishing early rather than at the buffer.

**Two honest caveats I will not paper over.** First, nobody has published whether ARR resubmission actually improves acceptance odds; the one published analysis (ACL 2023 PC report) found resubmission status **not** significantly predictive (p = 0.112), and a controlled experiment found that merely *signalling* resubmission cost −0.78 points on a 10-point scale. So do not bank on "we'll just resubmit" as a strategy — bank on it as a fallback. Second, this paper's specific elevated risk is **not** methodology or execution; it is the related-work squeeze in §7 (three published papers on adjacent ground, one from two days before planning). That is a *writing* problem, which is the most controllable kind — which is exactly why §7 gets disproportionate attention and why an external reader is tasked to attack it.

**What actually reduces risk here, in order:** (1) nail the §7 differentiation, (2) lead with the truncation result (§1.1) since it is the least-occupied ground, (3) report the decomposition honestly including nulls, (4) ship the artifact — a released generation corpus that others can re-score is a durable contribution independent of the accept/reject outcome.

---

## Appendix A — verified facts worth not re-deriving

- **ARR Oct 2026 cycle:** submission Oct 12, 2026 (23:59 AoE); commitment Dec 20, 2026; feeds NAACL 2027 (June 1–5, San Francisco) and COLING 2027 (May 9–14, Macau). Internal cycle dates (reviews due, author response, meta-review) are still **TBA** on aclrollingreview.org/dates. The **January 2027 cycle feeds ACL 2027** (Kyoto, Aug 17–22) and is the fallback route (§11).
- ⚠️ **There is no NAACL 2026** — NAACL skips years when ACL is in North America (`2026.naacl.org` does not resolve). And **ACL 2026 already happened** (July 2–7, 2026, San Diego). Don't cite either as a target.
- **ARR volume:** 10,518 submissions (Jan 2026) → **17,087** (May 2026), against only 1,424 qualified ACs and 10,636 reviewers. Combined main+Findings acceptance has stayed remarkably stable at 36.7–39.5% across recent venues; Findings absorbed the growth (16.7% → 17.8%). Aug 2026 cycle volume not yet published.
- **Dataset sizes (queried from the HF datasets server):** GSM8K test **1319**; ARC-Challenge test **1172** (train 1119, val 299); TriviaQA `rc.nocontext` val **17944**; BBH — 250 items per subtask except `causal_judgement` 187, `snarks` 178, `penguins_in_a_table` 146.
- **GSM8K harness config** (v3.0): 5-shot, `doc_to_text: "Question: {{question}}\nAnswer:"`, greedy (`do_sample: false, temperature: 0.0`), `until: ["Question:", "</s>", "<|im_end|>"]`; `strict-match` = `#### (\-?[0-9\.\,]+)`; `flexible-extract` = `(-?[$0-9.,]{2,})|(-?[0-9]+)` with `group_select: -1`.
- **BBH CoT fewshot** (v4.0): `dataset_path: SaylorTwift/bbh`, 3-shot, `max_gen_toks: 1024`, `until: ["</s>", "Q", "\n\n"]`, extraction filter `(?<=the answer is )(.*)(?=.)` — case-sensitive.
- **ARC:** `arc_challenge` is `output_type: multiple_choice` with `acc`/`acc_norm`, no generation. `arc_challenge_chat` is `generate_until`, 0-shot, `max_gen_toks: 100`, `gen_prefix: "The best answer is"`.
- **TriviaQA:** `generate_until`, `exact_match` against `answer.aliases`, `ignore_case` + `ignore_punctuation`, `until: ["\n", ".", ","]`.
- **Self-consistency norms.** Wang et al. (ICLR 2023) used **no top-p at all**: T=0.5/top-k=40 (UL2, LaMDA), T=0.7/top-k=40 (PaLM-540B), T=0.7 untruncated (GPT-3); nucleus sampling appears only in their ablation. The T=0.6/top_p=0.95 convention is **Minerva's and DeepSeek-R1's**, not Wang's — don't attribute it to the self-consistency paper. Wang's ablation sweeps k ∈ {1,5,10,20,40} and **never goes beyond 40**; its saturation guidance is qualitative ("5 or 10 as a starting point"). The quantitative saturation result is Minerva's: **97% of large-k accuracy is reached at k=64 for MATH and k=16 for GSM8K.** Large Language Monkeys finds voting plateaus beyond ~100 samples. Modern defaults are k=64 (competition math) or k=8 (cost-constrained).
- **Use unweighted majority vote.** Wang et al. Table 1 (PaLM-540B, GSM8K): unweighted 74.4 vs normalized-weighted 74.1 vs unnormalized-weighted 59.9 vs weighted-*average* 22.1. Weighting buys nothing because the model is poorly calibrated (their footnote 2). "Let's Verify Step by Step" likewise found RM-weighted voting "did not noticeably improve performance."
- ⚠️ **Two lm-evaluation-harness footguns that could silently corrupt the decoding arm:**
  1. **`repeats: N` alone silently discards N−1 generations.** `lm_eval/api/task.py` falls back to a `take_first` filter with the log line *"No custom filters defined. Using default 'take_first' filter for handling repeats"* and an in-code `TODO: handle repeats in a more general way rather than just discarding`. You **must** attach an explicit `majority_vote` filter pipeline or you pay for 64 samples and score only the first.
  2. **`"[invalid]"` competes as a votable answer.** `RegexFilter` uses `fallback="[invalid]"` on no match, and `MajorityVoteFilter` runs a bare `Counter` with no exclusion — so extraction failures form a bucket that can **win the plurality**. Ties break by CPython insertion order, making the vote order-dependent at T>0.
- **The literature is silent on unparseable-answer handling, and implementations disagree in opposite directions.** Verified by grepping the full text of nine major papers (Wang et al., Minerva, Let's Verify, DeepSeekMath, Qwen2.5-Math, DeepSeek-R1, Monkeys, Snell et al., s1) for "unparseable," "no answer," "fail to parse," "abstain," and tie-breaking — **zero hits**, and no paper specifies tie-breaking either. Meanwhile `lm-eval` and `lighteval` let failures vote, while **`bigcode-evaluation-harness` explicitly drops them first** (`# filtering the failed generations to avoid influencing majority voting`). **This is a clean, cheap, self-contained contribution:** the choice is undocumented in papers, resolved oppositely in code, and interacts directly with extraction strictness. Report our rule explicitly and report results both ways.
- An existing harness task is worth reusing rather than rebuilding: **`gsm8k_cot_self_consistency`** (`lm_eval/tasks/gsm8k/gsm8k-cot-self-consistency.yaml`) ships `do_sample: true`, `repeats: 64`, and three filter pipelines (`score-first`, `maj@64`, `maj@8`). ⚠️ Note its **`temperature: 0.2`**, not the 0.7 the originating plan assumed — if you deviate, say so and why. Its own comment concedes the maj@8 estimator is crude (`# Using a better estimator would be optimal`). Note also that `minerva_math`, `math500`, and `aime*` do **not** use maj@k (they run greedy), which is further evidence the decoding arm is the least conventional axis.
- **GSM8K is saturated as a self-consistency target**, which further justifies demoting the decoding arm: voting headroom on Qwen2.5-Math-72B GSM8K is **+0.1 point** (95.9→96.0 at maj@8) versus +2.7 on MATH. Frontier reasoning papers dropped GSM8K entirely — grep count is zero in both DeepSeek-R1 and s1 — and label noise now binds before model error does (GSM8K test index 1042 has a documented wrong gold answer). GSM8K survives as a *single-answer* metric, which is how we use it.
- **Terminology hazards to define explicitly in the paper:** `pass@k` = solved if *any* of k samples is correct (an oracle/coverage ceiling, needs a verifier); `maj@k`/`cons@k` = single plurality-voted answer; `avg@k` = mean per-sample correctness, i.e. a variance-reduced pass@1 estimate and **not** a voting metric (the most common confusion); `rm@k`/best-of-n = reward-model reranking. Minerva's notation is `maj1@k`, crediting AlphaCode (arXiv:2203.07814) for the `n@k` form; it also defines `majn@k`. DeepSeek-R1 introduced `cons@64` **and redefines "pass@1" as a mean over k samples**, so R1's "pass@1" is neither Chen et al.'s pass@1 nor a vote. `avg@k`/`mean@k` are not defined notation in any primary paper.
- ⚠️ **DeepSeek-R1's own numbers differ between arXiv versions** — v1 gives R1-Zero AIME 2024 as 71.0 pass@1 → 86.7 cons@64, while the current version restates it as 15.6% → 77.9% pass@1 and its Figure 1(a) plots `cons@16`, not cons@64. **Cite the version you actually mean**, and note that the widely-quoted cons@64 column belongs to R1-Zero and the distills rather than to R1 itself.
- **Notation:** Minerva's is `maj1@k` (crediting AlphaCode, arXiv:2203.07814, not DiVeRSe). DeepSeek-R1 introduced `cons@64` and **redefines "pass@1" as a mean over k samples** — a genuine terminological hazard; define our terms explicitly.
- **Judge unreliability on math** (for the paper's own justification of the ladder): Zheng et al. (NeurIPS 2023) measure GPT-4-as-judge failing 14/20 elementary math gradings by default, 6/20 with CoT, 3/20 with a reference answer; swap-consistency 65%.
- **No family-clean *generative* judge exists on RewardBench 2** — every open-weight generative entry is Qwen- or Llama-based, and the family-clean entries (QRM-Gemma-2-27B, Skywork-Reward-Gemma-2-27B) are scalar sequence classifiers that take no reference answer and so cannot do answer equivalence. This is why §3 uses a purpose-built verifier plus an instruct model rather than a reward-model leaderboard pick. RewardBench 2's "Factuality" split is 4-way best-of-N *preference*, not reference-based binary verification — do not cite it as evidence of verification ability.
- **Judge licensing:** `xVerify-*` weights are CC-BY-NC-ND-4.0 (no redistribution, no derivatives). Granite 4.1, Olmo 3.1, Gemma 4, and Ministral 3 are Apache-2.0. `01.AI`/Yi is dormant (last upload Aug 2024); DeepSeek has no family-clean sub-32B instruct model.
- **vLLM batch invariance** (`VLLM_BATCH_INVARIANT=1`, shipped v0.11.1, beta): overrides `mm/addmm/matmul/linear/softmax/mean.dim/bmm` with Triton kernels, disables reduced-precision reduction, forces TF32 off. On SM90/SM100 it only needs a cuBLAS workspace config; on **SM80/Ampere it must swap in Triton GEMMs**, so A100 pays more overhead than H100. Docs say capability ≥8.0 (an `envs.py` comment saying ≥9.0 is stale — A100 works). Auto-sets `disable_cascade_attn`.
- **Open batch-invariance bugs to watch** (all open as of 2026-08-06): #51290 (BI broken under TP>1), #51187 (3 distinct outputs in 5 repeats at 44 co-resident sequences; `max_num_seqs=8` fixes it), #39389 (prefix caching breaks greedy determinism), #48271 (RMSNorm block size varies with token count, defeating BI for sequences ≥256 tokens), #27433 (tracking issue — prefix-caching support still unchecked).
- **Throughput reality** (Kurtic et al., ACL 2025, Table 6; Llama-3.1-8B, A6000, max-throughput continuous batching): BF16 baseline; **INT8 W8A8 = 1.38×**; **INT4 W4A16 = 1.08×**, and slower than BF16 on 2 of 4 tasks. At batch=1 INT4 wins 2.39×. W4A16 is memory-bound-optimized; large-batch decode is compute-bound.
- **Measured vLLM throughput anchors** (third-party, 512in/512out, batch 32): Llama-8B on A100 ≈ 2,083 output tok/s; H100 ≈ 3,008; L40S ≈ 1,153. No public measured figure exists for a 3B model — benchmark our own.

## Appendix B — corrections to carry into the paper

Do not propagate these errors from the originating plan:
1. Li et al.'s "32.39%" is **relative** percent, on **Llama-3.2-1B**, on **MATH/MATH-500** — not percentage points, not Llama-3.1-8B, not GSM8K. Quote it as "~32% *relative* accuracy loss on the smallest model at W4A16," or cite the 8B figure (~11–12% relative) as more representative.
2. "Silent Failures in Quantized LLM Reasoning" full title ends **"and Failure Mode Shifts."** It tests only FP32/FP16/NF4.
3. "Quantized but Deceptive?" is about **truthfulness only**, not safety.
4. UniComp is **six** compression techniques (per v5) across 40 datasets; arXiv:2602.09130.
5. `auto-gptq` **and** `autoawq` are both abandoned and GitHub-archived. Use `llmcompressor` (both methods) or `gptqmodel` (GPTQ).
6. Do not claim 4-bit quantization speeds up inference at batch scale. Measured W4A16 speedup is ~1.08× and is *negative* on some tasks (Kurtic et al., ACL 2025, Table 6); the large-batch winner is W8A8-INT8 at 1.38×.
7. Do not set `enforce_eager=True` for determinism — CUDA graphs became batch-invariance-compatible in vLLM PR #27660, so it is pure throughput loss. (This corrects an earlier draft of this very plan.)
