# Pre-registration — QuantArtifact

**Status:** DRAFT (task `C2`). Must be COMMITTED and frozen by **2026-09-02**, before any
`run_id=main-v1` generation exists (gate `PREREG`, `EXECUTOR-TASKS.md` §8).

**Authority:** `EXECUTOR-PLAN.md` §4 governs the science here. This document is the
timestamped instantiation of it. Its entire value is the timestamp: retro-fitting §1
below after seeing results is worth far less, and a reviewer who detects that it was
patched rather than pre-specified caps acceptance at ~25% (`EXECUTOR-PLAN.md` §4.0).

**What "frozen" means.** After the freeze commit, this file may only be appended to,
under a dated `## Amendments` heading, with the reason stated. No edit to §1–§9 below.
Any analysis not specified here is exploratory and must be labelled as such in the paper.

---

## 1. Identification fix — the load-bearing section

Verbatim obligation from `EXECUTOR-PLAN.md` §4.0. This is pre-registered in all three
parts. It cannot be added later.

### 1.0 The flaw being fixed

`EXECUTOR-PLAN.md` §2.3 defines **Competence** as `P(correct | committed)` — given the
answer the model actually committed to, is it right? But **whether the model commits is
itself affected by precision.** Comparing competence across precision arms therefore
conditions on a post-treatment variable: this is collider / selection bias, not a causal
contrast.

This is not a technicality. **The bias points toward this paper's own hypothesis**, so the
more weight the centrepiece decomposition table carries, the more the design looks
engineered to confirm itself. We fix it by construction rather than by argument.

### 1.1 Primary competence contrast = the always-compliant stratum only

The primary competence contrast is computed **only** on the *always-compliant stratum*:
the set of items for which **every** arm produced a committed answer.

- **Stratum definition.** Item `i` is in the stratum iff for all precision levels `p` in
  the cell's precision set and all scorers `s` in rungs 1–4,
  `compliance(i, p, s) ∈ {COMPLIANT, RECOVERABLE}` — i.e. never `ABSENT`.
- Membership is determined by the **conjunction across all arms**, which makes the
  subgroup pre-treatment-defined with respect to any single precision contrast, so the
  contrast within it is identified.
- The stratum is defined **per (model × task × budget_level × prompt_variant)**, and the
  precision set is the full set for that arm (5 levels in the reference condition, 3 in
  the sweep and zero-shot arms — `EXECUTOR-TASKS.md` §2.2).
- **The stratum's size is reported as a first-class number**, not a footnote: as a count
  and as a fraction of items, per cell, in the main paper. If the stratum is small, that
  is itself the finding, and the competence contrast is reported as low-precision
  accordingly.

Pre-committed interpretation rule: a small stratum does **not** license falling back to
the full-sample competence contrast as primary. It licenses the Manski bounds of §1.3
carrying more of the argument.

### 1.2 Compliance-transition matrix — reported as a finding, not conditioned away

For each **ordered precision pair** `(p_ref → p_alt)` — with `p_ref = bf16` as reference
and `p_alt` ranging over GPTQ-4bit, AWQ-4bit, bnb-NF4, GPTQ-8bit — report the 2×2:

|                          | committed under `p_alt` | abstained under `p_alt` |
|--------------------------|-------------------------|-------------------------|
| committed under `p_ref`  | n₁₁                     | n₁₂                     |
| abstained under `p_ref`  | n₂₁                     | n₂₂                     |

where *committed* = `compliance ∈ {COMPLIANT, RECOVERABLE}` and *abstained* = `ABSENT`.

- Reported per (model × task × scorer rung), with the off-diagonal asymmetry
  `n₁₂ − n₂₁` and an exact McNemar test on the discordant cells.
- **This is a substantive result.** Compliance shift is a real quantization effect and
  belongs in the paper. Prior work folds it into "reasoning accuracy"; we report it
  separately.
- `TRUNCATED` is recorded **orthogonally** and is not a level of this matrix — see §7.

### 1.3 Manski bounds on the full-sample competence effect

Report worst-case/best-case (Manski) bounds on the full-sample competence effect, to show
what the data can and cannot rule out without the stratum restriction.

- Construction: for items `ABSENT` under an arm, competence is unobserved. Bound it by
  imputing all such items **correct** (upper) and all **incorrect** (lower), for each arm
  independently, and propagate to the difference.
- Report the bound interval alongside the point estimate from §1.1.
- **Wide bounds are an honest result and will be reported as such.** We pre-commit to
  reporting the bounds even when they are uninformative, and to not substituting a
  narrower but unidentified estimate in the abstract.

### 1.4 Scope of the fix

This interacts with §2 below: the H3 interaction model is fit on **correctness**, which is
well-defined for all items regardless of compliance, so **H3 is unaffected** by the
selection problem. Only the H4 competence decomposition needs the stratum restriction.

---

## 2. H3 — interaction test (specification frozen)

Item-level logistic mixed-effects model, **fit separately per task**:

```
correct ~ precision * scoring + model + (1 | item)
```

- `precision` and `scoring` are fixed factors; `scoring` levels are the four rungs of §5.
- **`model` is a FIXED effect.** With only 3 models, 3 levels cannot support a variance
  estimate. Item is the only random effect.
- The `precision × scoring` interaction coefficient **is** the artifact estimate. Reported
  as an odds ratio **and** as a marginal effect in accuracy points with a 95% CI.
- **Headline contrast is rung 1 (`strict`) vs rung 2 (`permissive`)** — deterministic,
  auditable, and independent of the judge. Rungs 3–4 are robustness.
- Companion test: **McNemar on discordant pairs**, reported alongside the GLMM as an
  assumption-light check. The extraction axis re-scores *identical generations*, so the
  strict-vs-permissive comparison is perfectly paired at the generation level.
- Pre-committed null interpretation: **a null interaction is a publishable, confirmatory
  result** and will be reported as one. It shows extraction leniency is a level shift, not
  a delta artifact, which *strengthens* the existing degradation literature by ruling out
  its most obvious alternative explanation. It will not be reported as a failed experiment.

---

## 3. Frozen token-budget levels (H1)

**Frozen as absolute integers before any results exist.** Verified against
`lm-evaluation-harness` **v0.4.12** task configs on 2026-08-23; the vendored copies are in
`vendor/lm-eval-configs/` and the derivation is in `notes/harness-configs.md`.

| Task | 1× | 2× | 4× | Source of 1× |
|---|---|---|---|---|
| `gsm8k` / `gsm8k_cot` | **256** | **512** | **1024** | framework default (`lm_eval/models/vllm_causallms.py`, `max_gen_toks: int = 256`) — **not set in the task YAML** |
| `bbh_cot_fewshot_*` (all 6) | **1024** | **2048** | **4096** | `bbh/cot_fewshot/_cot_fewshot_template_yaml`, `generation_kwargs.max_gen_toks: 1024` |
| `arc_challenge_chat` | **100** | **200** | **400** | `arc/arc_challenge_chat.yaml`, `generation_kwargs.max_gen_toks: 100` |

- The token budget is an **independent variable**, not an implementation detail. Decodes
  are hard-capped at each level.
- ⚠️ **GSM8K's 1× is a framework default, not a task-config value.** It is therefore a
  property of the *backend*, not the benchmark, and any backend with a different default
  silently changes it. We set it explicitly in `configs/tasks.yaml` and state the value in
  the paper rather than inheriting it.
- The sweep arms use only 2× and 4×; **1× comes free from the reference run** via the
  prefix-nested GSM8K subsample (`EXECUTOR-TASKS.md` §2.2, O8).

---

## 4. Truncation — definition, and a correction to the naive one

**Pre-registered definition.** An generation is **budget-truncated** iff
`finish_reason == "length"`.

⚠️ **This is necessary but not sufficient, and we pre-register the wider definition.**
Verification of the v0.4.12 configs (2026-08-23) shows the harness `until` stop sequences
can terminate a generation mid-reasoning, which is *also* truncation but is recorded as
`finish_reason == "stop"`:

- `bbh_cot_fewshot_*`: `until: ["</s>", "Q", "\n\n"]` — a **bare capital `Q`** and a blank
  line. Any chain of thought containing either is cut.
- `gsm8k` / `gsm8k_cot`: `until: ["Question:" / "Q:", "</s>", "<|im_end|>"]`.
- `arc_challenge_chat`: `until: ["\n\n", "."]` — a **single period**, with
  `max_gen_toks: 100`.

We therefore define and report **three** mutually exclusive termination classes per
generation, from `finish_reason` and `stop_reason`:

1. `TRUNC_BUDGET` — `finish_reason == "length"`. Hit the token cap.
2. `TRUNC_STOPSEQ` — `finish_reason == "stop"` **and** the matched `stop_reason` is a
   harness `until` string **and** rung-2 extraction finds no committed answer. A
   premature stop-sequence kill.
3. `COMPLETE` — terminated at EOS, or at a stop sequence with a committed answer present.

**H1 is tested against `TRUNC_BUDGET` as primary** (it is the quantity the budget sweep
manipulates), with `TRUNC_STOPSEQ` reported alongside because it is a second, distinct
measurement artifact operating on the same mechanism, and because it does not respond to
the budget sweep — which makes it a useful internal control.

Rationale for pre-registering this now: `EXECUTOR-PLAN.md` §2.3 and `EXECUTOR-TASKS.md`
§7.2 define `TRUNCATED` as `finish_reason == "length"` alone. On BBH that definition would
undercount truncation substantially. Fixing it after seeing results would be a
specification search.

---

## 5. Scoring ladder — rung-2 permissive parser rules, enumerated

All four rungs are applied to **every** stored generation. Zero extra generation. Rungs are
pure functions over stored artifacts (`scoring/extract.py`, no I/O).

**Rung 1 — `strict`.** The published harness regex, **verbatim**, per task:
- `gsm8k` (5-shot, non-CoT): `#### (\-?[0-9\.\,]+)` then `take_first`.
- `gsm8k_cot` (8-shot CoT): `The answer is (\-?[0-9\.\,]+).` then `take_first`.
- `bbh_cot_fewshot_*`: `(?<=the answer is )(.*)(?=.)` then `take_first`.
  **Case-sensitive lookbehind, lowercase `the`.** Every capitalization-variant failure is
  logged and counted; the count is a reported finding, not a discarded error.
- `arc_challenge_chat`: `remove_whitespace` then `take_first`, with metric
  `ignore_case: true, ignore_punctuation: true`.
- ⚠️ The GSM8K task-variant question (which of `gsm8k` / `gsm8k_cot` is the **primary**
  reference) is an **open scientific decision** as of this draft — see
  `notes/harness-configs.md` §3. Both are pre-registered above; the primary must be named
  in this file before the freeze.

**Rung 2 — `permissive`.** Deterministic, documented, ordered. Applied in this order; the
first rule that yields a candidate wins:
1. `\boxed{...}` — innermost complete braced group.
2. Explicit commitment phrase, case-**insensitive**: `the answer is`, `answer:`,
   `final answer is`, `the best answer is`, `so the answer is`. Take the span after it.
3. `####` marker, if present.
4. Last number in the string, for numeric-answer tasks
   (`(-?[$0-9.,]{2,})|(-?[0-9]+)` with `group_select: -1`, matching the harness
   `flexible-extract`).
5. For multiple-choice tasks: last standalone option letter in `A`–`E`, optionally
   parenthesised.

Normalization applied to the candidate, in order: strip surrounding whitespace and
`$ € £ %`; remove thousands separators `,` and `_`; strip a single trailing `.`; strip
LaTeX `\!`, `\,`, `\;`, `\left`, `\right`, `\$`, `\%`, `\text{...}` wrappers; apply
Minerva `normalize_final_answer` rules; collapse internal whitespace. Empty result after
normalization ⇒ no candidate.

**Rung 3 — `math-verify`.** `math-verify==0.9.0`, symbolic/numeric equivalence, applied to
numeric-answer tasks only.
- ⚠️ `parse()`/`verify()` use `signal.alarm()` and **raise in worker threads**: called with
  `parsing_timeout=None, timeout_seconds=None` and an externally imposed timeout.
- It is intentionally **asymmetric** — gold/target argument order matters and is fixed as
  `verify(gold, prediction)`.
- Timeout, exception, or parse failure ⇒ recorded as `incorrect` with `notes` set, never
  silently dropped. The timeout count is reported.

**Rung 4 — `judge`.** See §6.

**Monotonicity.** Rungs are asserted **monotone** on the `tests/fixtures` set: anything
rung *n* marks correct, rung *n+1* also marks correct. A violation is either a bug or a
genuine finding; it is triaged and recorded, never ignored. Monotonicity is asserted on
fixtures, **not** on experimental data — an empirical violation on real generations is a
reportable result.

**Versioning.** Scorer outputs are keyed by `(cell, scorer_name, scorer_version)`. Any
change to a rung **bumps the version**; outputs are never overwritten.

---

## 6. Judge — gate thresholds and pre-specified fallback

- **Primary judge:** `IAAR-Shanghai/xVerify-9B-I`. Licensed CC-BY-NC-ND-4.0: run and
  report numbers, **never redistribute weights or derivatives** in the artifact.
- **Second judge:** `ibm-granite/granite-4.1-8b` (Apache-2.0).
- Label space: CompassVerifier **3-way** (`A` / `B` / `C`) with an explicit `INVALID`
  class, which maps onto `ABSENT`/`TRUNCATED`. Instruction borrows CompassVerifier's
  "compare **ONLY** the final answer".
- The reference answer is **always** supplied (reference-guided judging cuts GPT-4's
  math-grading failure rate from 70% to 15%).
- Greedy, seed fixed, single-token verdict. The judge's **output-parse rate** is measured
  and reported.

**Gate (pre-specified, evaluated once).** The judge is used as a primary rung iff **all
three** hold:
1. raw agreement with adjudicated human labels **≥ 0.90** (with Wilson 95% CI), **and**
2. **Gwet's AC1 ≥ 0.70**, **and**
3. judge error rates for **bf16 vs 4-bit** generations are **not significantly different**
   (two-proportion test, α = 0.05).

Condition 3 is the one a good reviewer asks for: a judge that is lenient toward
*quantized* outputs specifically would manufacture our result.

**Reported regardless:** raw agreement + Wilson CI, Cohen's κ, **Gwet's AC1 (the primary
statistic)**, and the confusion matrix with **FP and FN broken out separately**. κ is
never reported alone — these labels are heavily skewed and a judge at 95% raw agreement
can post κ≈0.3 purely from prevalence (kappa paradox: Feinstein & Cicchetti 1990; Byrt,
Bishop & Carlin 1993). Human IAA is reported alongside, because a judge cannot be held to
better agreement than the humans achieve with each other.

**Pre-specified fallback if the gate fails:** rungs 1–3 are primary; the judge is reported
as a **caveated robustness check**. The paper is intact either way, because the headline is
already rung 1 vs rung 2. This is explicitly *not* a blocking condition.

**Judge coverage (pre-specified cascade):** judge 1 over the full reference condition, plus
a documented cascade on the token sweep — judge **only** where `rung2 ≠ rung3`, or where
`rung2 == incorrect`. Judge 2 over the reference condition and the human-validation sample
only. The cascade rule is reported in the paper.

---

## 7. Compliance labels (orthogonal to correctness)

Per `EXECUTOR-PLAN.md` §2.3. Assigned to every generation:

- `COMPLIANT` — answer present in the canonical expected format (rung-1 extraction succeeds).
- `RECOVERABLE` — no canonical format, but rung 2 finds a committed final answer.
- `ABSENT` — no identifiable answer (degeneration, refusal, or truncation).

**Truncation is recorded separately and is orthogonal** — a generation can be both
truncated and `ABSENT`, and the two must not be conflated. See §4 for the three
termination classes.

**Compliance rate is a dependent variable in its own right.** If it drops under
quantization, that is a real, reportable, novel effect, and it is precisely what prior work
has been folding into "reasoning accuracy".

---

## 8. H2 — trace-scan definition and its false-positive guard

Both variants are computed and **both are reported**:

- **permissive:** normalised gold appears anywhere in `generation_text`.
- **conservative:** normalised gold appears in the **final 20%** of the generation **or**
  after a commitment phrase (`the answer is`, `therefore`, `so`, `\boxed{`), **and** the
  rung-2 extraction ≠ gold.

**Mandatory calibration — without it the number is meaningless.** Report the **base rate of
incidental gold appearance** in traces where extraction *succeeded and was wrong*. That is
the false-positive floor for this measurement. This is the specific measurement question we
claim arXiv:2606.00206's authors did not ask, so it has to be airtight: the headline H2
number is always reported **net of** this floor, and the floor is reported next to it.

Numeric golds with |gold| < 10 are additionally reported as a separate stratum, since
single-digit golds have a high incidental match rate.

---

## 9. Statistics, multiplicity, and the noise floor

**One pre-specified omnibus contrast per hypothesis.** No exhaustive pairwise slicing.
**Holm–Bonferroni within each hypothesis family** (H1, H3, H4, H5, H6 are separate
families). Effect sizes with CIs for everything. **Never a bare significance flag.**

**Noise floor (mandatory, `EXECUTOR-PLAN.md` §4.5).** The bf16 baseline is run **twice**
end-to-end with different seeds on every model. The spread between those two runs is the
pipeline's own noise floor.

> **Pre-committed rule: any effect smaller than the measured noise floor will not be
> claimed by us either.** This applies to our own results with the same force we apply it
> to the literature in H7. Reporting it is the point.

**Differential noise floor (H8).** If the `A7` determinism audit shows a quantized format
is bitwise nondeterministic while bf16 is not, then differential numerical noise exists
*between arms*, and that per-format disagreement rate is reported as an additional floor on
every precision contrast involving that format.

**Frozen serving configuration.** One engine, one version, one hardware target, across
**every** arm — backend choice alone moves benchmark scores up to 16.6 pp
(arXiv:2605.19537) and explains ~39% of out-of-the-box variance (arXiv:2608.04714), so a
floating serving config would make the precision effects unidentifiable. The following are
fixed before `A13` and reported in the paper: `engine` and version, `tensor_parallel_size=1`,
`VLLM_BATCH_INVARIANT`, `enable_prefix_caching=False`, attention backend, `seed=0`,
`temperature=0`, `add_bos_token`, `max_model_len`, `max_num_seqs`, `dtype`, GPU model, CUDA
and driver versions.

`max_num_seqs` is set to the **largest value shown bitwise-reproducible across repeats by
the `A7` audit on this hardware** (swept over {8, 16, 32, 64}), rather than to a value
inherited from a third-party issue report. It is then **frozen and identical across all
arms**. Both the selected value and the full frontier are reported. Choosing it by
measurement rather than assumption is a throughput decision, not an analysis decision, and
it is pre-registered here only so that it is on record as having been fixed before results
existed.

**H7 framing, pre-committed.** For each published degradation figure cited, compute the 95%
CI implied by that paper's own eval size and state whether the reported delta clears it.
Also report our own minimum detectable effect at 80% power. Wording is scrupulously
neutral: **"many papers report aggregate deltas without a paired test"** — **never** "the
deltas are noise". We adopt Kübler et al.'s (ICLR 2026) paired McNemar test rather than
arguing around it.

---

## 10. Annotation — adjudication rule, pre-specified before any disagreement is seen

- ≥250 items, stratified across 3 models × {bf16, 4-bit} × all tasks, **oversampling
  rung-1/2/3 disagreement cases**; agreement on the natural distribution is **also**
  reported. ≥120 double-annotated for IAA. Per-annotator counts reported.
- Source pool is `run_id=pilot-v1` (`EXECUTOR-TASKS.md` O11), drawn with the **final
  frozen** prompt templates and budgets.
- **Blinding.** The condition column is **omitted from the annotation file entirely**, not
  hidden — a hidden column can be un-hidden. Annotators never see the precision condition,
  the judge's verdict, or the parser's verdict. Pre-shuffled **independently per
  annotator**; the two orderings are asserted to differ; seeds published.
- **Adjudication rule, fixed now:** disagreements are resolved by applying the frozen
  written rubric to the item as a *rule lookup*, by both annotators jointly, and the
  governing rubric clause is recorded per adjudicated item. If the rubric is silent, the
  item is marked `RUBRIC_GAP` and **excluded** from the agreement statistic, with the count
  reported. **Disagreements are never resolved by deferring to the senior author** (Belz et
  al., Insights 2023, list biased aggregation as a documented reproducibility failure).
- **No judge pre-annotation, ever.** Anchoring is measured (Fort & Sagot, LAW IV 2010:
  the better annotator dropped 98.1→95.8 under low-quality pre-annotation). Note the trap:
  Mikulová et al. (LREC 2022) found pre-annotation *raised* IAA — **higher agreement under
  pre-annotation is shared anchoring, not higher validity**, and IAA is a headline number.
- A ~10-item blind practice round is used **instead of** attention checks, because
  excluding check-failures yields a biased subsample (van der Lee et al., INLG 2019).
- If IAA on the overlap subset is below ~0.85, that is treated as a **rubric problem, not a
  judge problem**: revise the rubric, re-run the dry run jointly, re-annotate, and draw no
  conclusion about the judge until then.

---

## 11. What is exploratory

Anything not specified above. In particular, and labelled as exploratory in the paper if
reported at all: any subgroup analysis not named here, any post-hoc task subsetting, any
alternative parser rule added after the freeze, and any analysis of the `TRUNC_STOPSEQ`
class beyond the descriptive rate reported under §4.

---

## Amendments

*(none — append dated entries below the freeze commit only)*
