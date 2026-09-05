# Pre-registration — QuantArtifact

**Status:** LIVE, and **not frozen**. This document is corrected **in place** when something in
it turns out to be wrong; see *How this document is maintained* at the end. Its value is that every
decision records whether it was made before or after seeing the relevant data, not that the file
stopped changing on a particular date.

**Authority:** `EXECUTOR-PLAN.md` §4 governs the science here. This document is the written
instantiation of it, and `EXECUTOR-PLAN.md` §4.0 is the reason it exists at all: a reviewer who
detects that §1's identification fix was patched rather than specified in advance caps acceptance
at ~25%. That risk is real because **the bias §1 corrects points toward this paper's own
hypothesis**, so the fix has to be visibly principled rather than convenient.

**How that is protected here, and it is one rule rather than three.**

🚨 **Every decision in this document states whether it was made BEFORE or AFTER seeing the
relevant data, in place, with the data quoted inline.** §3b-bis, §5a, §5b, §9a–§9d and §9f all do
it: each is labelled as a pre-specified extension of an exploratory finding, dated, with the
exploratory numbers given. Discovery followed by a written commitment is normal practice.
Presenting it as pre-data when the git history says otherwise is what costs.

⚠️ **The failure mode being guarded against is NOT "editing after a date".** It is *choosing an
analysis because you have seen which choice flatters the result.* A calendar cannot detect that;
stating each decision's provenance can, and the git history is the independent record either way.
The document is therefore **maintained rather than frozen** (see the end), and it is corrected in
place so that no reader can act on a superseded claim.

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

- **Stratum definition, AMENDED 2026-09-02 (R26.3).** Item `i` is in the always-compliant
  stratum iff for all precision levels `p` in the cell's **contrasted** precision set and all
  invocations (§9g), `compliance(i, p) ∈ {COMPLIANT, RECOVERABLE}` — i.e. never `ABSENT` — where
  compliance is a **generation-level** label that **does not depend on the scoring rung**.

  🚨 **The original wording quantified over "all scorers `s` in rungs 1–4", and that was a
  drafting error rather than loose phrasing. Recording WHY, because §1.1 is a §4.0 item and the
  reason is what makes the amendment defensible rather than convenient.**

  **If compliance depended on the scoring rung, stratum membership would depend on the scoring
  axis, which is one of the two factors in H3's interaction.** A stratum whose membership varies
  with the scorer makes part of the scoring contrast a *selection* contrast. So scorer-independent
  compliance is **required by the design**, not merely how the scorer happens to be implemented.
  The rejected reading does not just shrink the stratum; it breaks the identification §4.0 exists
  to protect.

  This also matches what the scorer computes: `label_compliance` returns one label per generation,
  and `RECOVERABLE` is inherently a statement about two rungs at once, meaning "permissive found
  what strict missed". There is no per-rung compliance to quantify over.

  ⚠️ **One thing the paper must pre-empt, because a sharp reviewer will probe it.** Rung 2 appears
  both in the definition of `RECOVERABLE` and as an arm of the H3 contrast. That is not circular,
  and the paper states why in one sentence: rung 2 is used **definitionally, once, identically
  across arms**, to establish commitment; the stratum restricts only H4's
  **competence-across-precision** contrast, which is not a scoring contrast; and §4.0's own note
  already establishes that **H3 is fit on all items** and so is unaffected by the stratum.
- Membership is determined by the **conjunction across the contrasted arms**, which makes
  the subgroup pre-treatment-defined with respect to any single precision contrast, so the
  contrast within it is identified.
- 🚨 **AMENDED 2026-08-29 (planner ruling R18.4.1): the conjunction is over the THREE
  CONTRASTED arms — `bf16`, `gptq-4bit`, `awq-4bit` — not all five.** The stratum must be
  pre-treatment-defined *with respect to the arms being compared*; requiring a committed
  answer from arms we do not contrast (`gptq-8bit`, `bnb-nf4`) shrinks it multiplicatively
  **for no identification benefit**. Under independence at 0.90 per-arm compliance that is
  0.90³ = 0.73 rather than 0.90⁵ = 0.59.

  ⚠️ **Registered BEFORE the re-scored numbers exist**, and that timing is the point. Job
  3's pilot showed `COMPLIANT` at 2–6% for Qwen, but those figures are known to be
  dominated by the BBH stop-sequence defect (§9b) and are being discarded and re-measured.
  Amending the rule now, while no admissible number exists for it to be tuned to, is what
  keeps this principled rather than convenient. `gptq-8bit` and `bnb-nf4` remain fully
  reported everywhere else; they are simply not part of this conjunction.
- The stratum is defined **per (model × task × budget_level × prompt_variant)**, and the
  contrasted precision set is `{bf16, gptq-4bit, awq-4bit}` in every condition
  (`EXECUTOR-TASKS.md` §2.2).
- **The stratum's size is reported as a first-class number**, not a footnote: as a count
  and as a fraction of items, per cell, in the main paper. If the stratum is small, that
  is itself the finding, and the competence contrast is reported as low-precision
  accordingly.

**⚠️ `committed` and `TRUNC_BUDGET` are NOT mutually exclusive, and must not be made so.**
A generation reading *"The answer is 42. Let me verify: 42 × 2 = "* is **both** committed and
budget-truncated. Truncation **after** commitment does not bias the competence contrast — the
model already told us its answer — so **those items belong in the stratum and are not
filtered out.** Only truncation that prevents commitment is a threat. Getting this backwards
would shrink the stratum for no inferential gain.

**Pre-committed interpretation rules for stratum size:**
- A small stratum does **not** license falling back to the full-sample competence contrast as
  primary. It licenses the Manski bounds of §1.3 carrying more of the argument.
- **Minimum viable stratum, originally pre-specified at 50%.** The stratum intersects over
  the three contrasted arms, so it shrinks multiplicatively — at 0.90 compliance per arm
  and independence it would be 0.90³ = 0.73, and higher in practice since compliance is
  correlated across arms.
- 🚨 **AMENDED 2026-08-31 (R24.3): the 50% threshold is DISSOLVED, not lowered.** Measured
  on re-scored `pilot-v2`, the stratum is below 50% in most cells — ARC 74–100%, BBH
  10–100%, **`gsm8k` 1–27%** — and §9g makes it strictly smaller again by requiring
  commitment in *all invocations*.

  **The threshold had a defect independent of whether it passes: it is PROPORTIONAL where it
  should have been ABSOLUTE.** A 27% stratum on 1,319 GSM8K items is **356 items**; a 60%
  stratum on a 150-item BBH subtask is **90 items**. The first is far more usable than the
  second, and the proportional rule ranks them the wrong way round.

  ⚠️ **Lowering a threshold because it failed would be indefensible, so it is not lowered.**
  Instead: **report the point estimate AND the Manski bounds for every cell, always, with
  the stratum size as a first-class number.** §4.0 parts 1 and 3 already require both
  jointly, so the "threshold" becomes a **presentation rule** — which of the two is primary
  in the table — rather than an analysis gate. That is strictly more informative than a
  switch, and it removes the incentive the threshold created.
- ⚠️ **BBH is reported PER SUBTASK**, and the format gradient is tested on the subtasks whose
  stratum is usable. A pooled BBH number would hide a 10% stratum inside a 100% one.
- **Wide bounds remain an honest result** (§1.3), and they cost less than they would have
  three weeks ago: H4 was the centrepiece under the old spine, but the spine is now the
  apparatus findings (ordering, invocation, truncation) and **H4 is a supporting
  decomposition.** Bounds plus an honestly reported stratum size is a defensible H4, not a
  failed one.
- 🚨 **The `gsm8k` hash-marker arm is EXCLUDED from H4's primary decomposition** (R24.3),
  exactly parallel to §9c excluding it from H3. A floored metric has no variance for an
  interaction to appear in; a 1–27% stratum has too few items for a competence contrast. It
  is retained as the **demonstration** arm. Same reasoning, same treatment, registered in
  the same way.

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
well-defined for all items regardless of compliance, so H3 is unaffected by the *selection*
problem as §1.0 poses it. Only the H4 competence decomposition needs the stratum restriction.

🚨 **AMENDED 2026-09-04 (planner ruling R35.Q1). The sentence above was FALSE as an unqualified
claim, and §4.0 is the section that cannot be patched later, so the correction is recorded here
with its reason rather than left to the analysis.**

**H3 is NOT unaffected in any cell with an arm-differential rate of configuration-caused
non-commitment.** The mechanism, and it is arithmetic rather than judgement:

An item whose scored prefix is **empty** scores wrong under **both** rungs, so it contributes
**zero** to that item's strict-minus-permissive gap. That looks like immunity and is the opposite.
If arm A has a higher empty-prefix rate than arm B, then **arm A has fewer items able to
contribute a nonzero gap, so its measured gap is deflated further**, and the interaction
`gap(alt) − gap(bf16)` is correspondingly **inflated**.

Measured, and the direction is the one that matters: on Llama BBH the empty-prefix rate is
**~21% for `bf16` against ~8% for `gptq-4bit`** (`main-v1-smoke3`, 24 items per cell, so
indicative only). `bf16` is the reference arm, so its gap is the one deflated most, and the
interaction is inflated **positive** — i.e. **toward this paper's own hypothesis**. That is the
third time an artifact in this study has pointed that way, after §4.0's original collider and
R10's found/committed leak, and it is why the note could not be left standing.

**Registered remedy, using machinery already specified rather than anything new.** Empty-prefix
items are **not committed**, so §1.1's always-compliant stratum **already excludes them**.
Therefore:

1. **The H3 interaction is reported BOTH full-sample and stratum-restricted, for every cell,
   always.** Both are free once the stratum is computed, and reporting only one invites the
   question of which was chosen.
2. 🚨 **The stratum-restricted version is PRIMARY wherever the arm-wise empty-prefix rate spread
   within a cell exceeds 5 pp.** The threshold is registered here, before the reference run
   exists, so it cannot be selected after seeing which cells it moves.
3. The empty-prefix rate **per arm** is reported alongside, since it is the quantity that decides
   which version is primary and a reader must be able to check the switch.

⚠️ This is a *presentation and primacy* rule, not a new estimator: both quantities were already
specified, and §1.1's stratum was already the instrument for exactly this class of problem. What
changed is the recognition that H3, not only H4, needs it.

---

## 2. H3 — interaction test (specification frozen)

Item-level logistic mixed-effects model, **fit separately per task**:

```
correct ~ precision * scoring + model + (1 | item)
```

- `precision` and `scoring` are fixed factors; `scoring` levels are the **five** rungs of §5
  (`strict`, `strict_last`, `permissive`, `math_verify`, `judge`).
- **On GSM8K the model additionally carries `answer_marker` and its interactions**, since the
  two variants are two levels of one factor on the same items:

  ```
  correct ~ precision * scoring * answer_marker + model + (1 | item)
  ```

  The pre-registered quantity of interest is the **three-way**
  `precision × scoring × answer_marker` term, which is the direct test of the compliance
  mechanism (§3's directional prediction). BBH and ARC have no `answer_marker` level and use
  the two-factor form above.
- **`model` is a FIXED effect.** With only 3 models, 3 levels cannot support a variance
  estimate. Item is the only random effect.
- The `precision × scoring` interaction coefficient **is** the artifact estimate. Reported
  as an odds ratio **and** as a marginal effect in accuracy points with a 95% CI.
- 🚨 **AMENDED 2026-09-02 (R25.2a): the ESTIMATOR of each is now named, because `(1 | item)`
  as written denotes a conditional maximum-likelihood GLMM and that model was NOT fit.**

  | quantity | estimator | status |
  |---|---|---|
  | marginal effect, accuracy points | difference of differences with a **cluster bootstrap over items** | **PRIMARY** |
  | odds ratio | **GEE**, exchangeable working correlation, cluster-robust SEs by item, population-averaged | secondary |
  | odds ratio, appendix only | `BinomialBayesMixedGLM` variational Bayes, random intercept for item | robustness; demonstrates the registered model was fitted |

  **Why the marginal is primary, and why this is continuous with §2 rather than a departure from
  it.** §2 *already* requires the marginal effect in accuracy points, so it is one of the two
  registered quantities and not a fallback. A conditional `(1 | item)` model yields a
  **conditional** odds ratio, which is not the quantity anyone interprets in accuracy points, and
  conditional and marginal odds ratios differ systematically. The cluster bootstrap is
  assumption-lighter than any GLMM here: no link function and no Gaussian random-intercept
  assumption, the latter being unverifiable on this data. And §2 already asks for McNemar as an
  *"assumption-light companion"*, so preferring an assumption-light interval follows the same
  logic §2 states for itself.

  **Why GEE rather than variational Bayes for the reported odds ratio.** VB's interval is
  **anti-conservative**, since variational Bayes understates posterior variance, and presenting an
  optimistic interval where a reader expects a confidence interval is the worse error. GEE is
  population-averaged, which **matches the marginal estimand**, so the odds ratio and the
  accuracy-point effect describe the same contrast on two scales with the same clustering. VB's
  conditional odds ratio is numerically larger and a reader comparing the two would find them
  inconsistent without being told why.

  **No R dependency.** Fitting the registered model exactly would need `lme4`/`glmmTMB` through
  `rpy2` or `pymer4`. Rejected: it changes only the secondary odds ratio, it is a new toolchain
  in week 5 of a 7-week runway, and it would put R into the artifact's pinned dependency set,
  which is a reproducibility liability in a paper about reproducibility.

  🚨 **REQUIRED DISCLOSURE.** The paper must carry a sentence stating that a conditional ML GLMM
  was not fit and why. A reviewer who knows `lme4` will see `(1 | item)` in this document and no
  GLMM in the results, so the omission is pre-empted rather than waited for.
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

## 3a. Item counts and the powered-ness of the format gradient

**BBH per-subtask counts are NOT uniform, and the reason is power, not convenience.**

Minimum detectable effects, paired binary, `SE(Δ) ≈ √(p_d/n)`, `MDE ≈ 2.8·SE` at 80% power:

| n | MDE at `p_d`=0.10 | MDE at `p_d`=0.30 |
|---|---|---|
| 1319 (GSM8K full) | **2.4 pp** | 4.2 pp |
| 900 (BBH pooled) | **2.9 pp** | 5.0 pp |
| 600 (4 free-form pooled) | 3.6 pp | 6.2 pp |
| 250 | 5.6 pp | 9.6 pp |
| **150 (one subtask)** | **7.2 pp** | **12.6 pp** |

The **format gradient** (free-form → boolean → multiple-choice) is a prediction of the
hypothesis, not six arbitrary tasks. But its three levels are: 4 free-form subtasks **pooled**
(n=600), `boolean_expressions` **alone**, and `logical_deduction_five_objects` **alone**. At
n=150 the two single-subtask levels could not detect anything under ~7 pp — so two of the three
gradient levels would have been unpowered.

**Pre-registered counts:** the two single-subtask **contrast** levels are raised to the full
**250** items available; the four pooled free-form subtasks stay at **150**. BBH is therefore
**1,100 items per cell**, not 900. ≈ **+8,400 generations (~6%)**, targeted precisely at the
underpowered levels. **No subtasks are added** — that would mean new configs and a wider
multiplicity family.

Pooled BBH (n=1,100) and full GSM8K (n=1,319) are adequately powered, and the paper's thesis
rests on the **pooled cross-task comparison**, not on any single subtask.

Note this is separate from **H7**, which is unaffected: H7(a) is arithmetic over *other
papers'* eval sizes and does not depend on our counts at all.

---

## 3b. Chat template — FIXED, not a factor ⚠️ SUPERSEDED BY §3b-bis

🚨 **Item 1 below ("chat template ON, FIXED, not a factor") NO LONGER HOLDS.** §3b-bis makes
the template a **two-level factor** (planner ruling R19.1). This section is retained rather
than rewritten because its *diagnosis* is still the basis of the design and because a reader
should be able to see what was decided first and why it changed. **Read §3b-bis before acting
on anything here.** Item 2's off-template probe survives, promoted from a calibration
measurement to the study's delta-invariance test; item 3's double-BOS assertion is unchanged
and still binding.

`--apply_chat_template` wraps the entire few-shot block into a single user turn, after which
instruction-tuned models tend to revert to their own prose format rather than continuing the
demonstrated pattern. This is the most likely explanation for arXiv:2601.14277 reporting F16
GSM8K strict 24.64 vs flexible 77.63 **despite** 5-shot exemplars that demonstrate `####`.

**Pre-registered decisions:**
1. **Chat template ON, single-turn, for every arm.** Stated in the paper with the citation. It
   is **not** a factor — making it one would roughly double the matrix for an axis that is a
   confound to pin, not a novelty.
2. **One off-template probe**, to quantify what we held fixed and to pre-empt *"why didn't your
   strict scores collapse like theirs?"*: one model (`qwen2.5-3b`), {bf16, GPTQ-4bit}, `gsm8k`,
   500 items, template **off**. ≈2,000 generations. Reported as a calibration measurement, not
   as a factor level.
3. ⚠️ **`add_bos_token=True` combined with a chat template produces a DOUBLE BOS.** Red Hat's
   `add_bos_token` recommendation is for non-template use. A silent double BOS shifts scores
   and would confound **every** arm simultaneously.
   **Mandatory assertion, checked in gate G2:** log the final token IDs of one rendered prompt
   per model and assert **exactly one leading BOS token**. A failure here blocks generation.

---

## 3b-bis. Chat template is a TWO-LEVEL FACTOR — supersedes §3b item 1

Registered **2026-08-29**, planner ruling R19.1/R19.2. **This replaces §3b's "chat template ON,
FIXED, not a factor"** and promotes §3b item 2 from a calibration probe to the study's
delta-invariance test.

**Why the change.** §3b fixed the template ON because making it a factor would double the
matrix. Two costs of that surfaced later. First, the depressed strict scores the paper reports
are plausibly *caused* by the template — R1.2 diagnosed `--apply_chat_template` as the mechanism
behind arXiv:2601.14277's 24.64-vs-77.63 collapse, and we then adopted it. Second, BBH's
terminators broke under it (§9b). Patching each consequence while holding the choice fixed
leaves the headline open to *"this is your configuration, not a property of quantization
evaluation as practised."* **A level artifact is not a delta artifact** — that is the paper's
own reframe, so the template belongs on the level axis and the claim belongs on the invariance.

| level | prompt | BBH terminators |
|---|---|---|
| **`chat`** (PRIMARY) | `apply_chat_template=True`, single-turn | `["</s>", "Q:", "<|im_end|>"]` |
| **`raw`** (robustness) | `apply_chat_template=False`, raw continuation | `["</s>", "Q", "\n\n"]` |

🚨 **The terminator set is part of each level's DEFINITION, not a constant.** At `raw`,
lm-eval's `_cot_fewshot` set is correct *by construction* — it was written for that format. At
`chat` it is not, so the level uses lm-eval's own `_cot_zeroshot` set. Declaring the terminators
per level is what makes the formatting assumption explicit and testable rather than inherited.

**`chat` remains primary**, because lm-eval's default is `apply_chat_template=False` yet
`evaluator.py:183` **warns the user to enable it for instruct/chat variants** — which all three
of our models are. Evaluating instruct models in a format their own harness warns against would
be the less defensible primary.

⚠️ **`raw` is a MAIN-TEXT robustness arm, not an appendix item**, reported as a test that could
have failed.

**Scope, frozen:** 3 models × 3 contrasted arms (`bf16`, `gptq-4bit`, `awq-4bit`) ×
{`gsm8k_cot` 500, BBH 1500} = **9 cells, ~18,000 generations.** Not one model on one task.

**🚨 DECISION RULE, PRE-REGISTERED BEFORE THE PROBE RUNS.** This is what makes it a test rather
than a defence, and it is recorded while no `raw`-level number exists:

1. **If the strict-vs-permissive interaction holds in SIGN and rough MAGNITUDE under both
   levels** → `chat` is primary, `raw` is reported as robustness, and the self-infliction
   objection is **answered empirically rather than argued**.
2. **If they DIVERGE** → the divergence **is the finding**: the prompt template is a
   **moderator** of the measured quantization effect. That is the *stronger* result, directly
   on-thesis, and is reported as the primary framing rather than buried.

**Both branches are publishable and neither is preferred in advance.** We pre-commit to
reporting whichever obtains, and to **not** relabelling branch 2 as a limitation if it occurs.

⚠️ **What this does NOT license.** The probe is 9 cells at the 1x budget, not a full factorial
replication. It licenses a statement about the *interaction's* invariance, not about every
hypothesis under both templates. H1's budget sweep, H4's decomposition and the annotation pool
remain `chat`-only.

---

## 3c. H6 negative control — THREE arms, and why two were not enough

`EXECUTOR-TASKS.md` §2.2 states *"the gap between the modes **is** the extraction cost,
measured directly."* As specified it was not, because the two harness tasks differ in **prompt
content and answer space**, not only in scoring mode: `arc_challenge` renders
`"Question: …\nAnswer:"` and scores log-likelihood over answer **texts**, while
`arc_challenge_chat` renders a full A/B/C/D instruction and scores a generated **letter**.

*(An earlier draft claimed the confound was shot count. Verified false 2026-08-25:
`arc_easy.yaml` contains no `num_fewshot` field at all, so **both** arms are 0-shot.)*

**Pre-registered three-arm design, same items throughout:**

| arm | prompt | scoring | role |
|---|---|---|---|
| **1** | chat prompt + `gen_prefix` | generate → extract | extraction surface **present** |
| **2** | **byte-identical to arm 1** | log-likelihood over `{" A"," B"," C"," D"}` | extraction surface **absent** |
| **3** | harness-verbatim `arc_challenge` | log-likelihood over answer texts | published-comparable baseline |

> **The H6 contrast is arm 1 vs arm 2.** The prompt is byte-identical; the only thing that
> varies is whether the answer is generated-then-extracted or read off a likelihood ranking.
> **That is the extraction cost, measured directly** — which is what §2.2 claimed and did not
> have.

Arm 3 preserves comparability with published ARC-C numbers and is the G3 cross-validation
target. **Arm 2 costs no generation** — four single-token scorings over an already-computed
prefix.

🚨 **The G3 ≤0.5 pp cross-validation runs against arm 3, lm-eval's NATIVE config — not arm 2.**
Arm 2 scores *option letters* on the *chat* prompt; native `arc_challenge` scores *full answer
texts* on the `Question:`/`Answer:` prompt. Different instruments answering different questions,
both legitimate, **not interchangeable**. Recorded before the run so the criterion cannot be
reinterpreted after seeing a number.

**Arm 2 continuations are space-prefixed: `" A"`, `" B"`, `" C"`, `" D"`.** `gen_prefix` ends
`…The best answer is` with no trailing space, so that is what the generative arm must emit, and the
log-likelihood arm is a valid counterfactual only if it scores the same thing. Scoring `"A"` would
measure a tokenization the model essentially never produces after `is`, and the penalty varies by
model *and* precision — exactly the artifact shape that would manufacture an H6 result on the
never-cut negative control. Corroboration: lm-eval's default `target_delimiter` is `" "`, so native
ARC-C log-likelihood already scores space-prefixed continuations.

⚠️ **Continuation token ids are obtained by tokenizing IN CONTEXT and differencing**, not by
encoding the continuation standalone. Measured 2026-08-26: standalone `" A"` is **one** token on
Qwen2.5 and Llama-3.1 (byte BPE, `ĠA`) but **two** on Phi-3.5 (SentencePiece, `['▁','▁A']`, because
`▁A` already encodes the space). In context all three yield a single token. "Always prefix a space"
is the wrong rule — the same shape as the double-BOS trap.

**Guards, both pre-registered:**
1. **Assert single-token-ness per model** and record the four token ids per model in the prompt
   manifest. If a model splits an option, that model gets explicit length normalization and the
   discrepancy is **reported, not absorbed**. (Verified: all three are single-token in context.)
2. **Validate empirically on `pilot-v1`:** measure the distribution of the actual first token
   emitted after `The best answer is` in the ARC-chat generative outputs. If it is overwhelmingly
   `▁A`/`▁B`/`▁C`/`▁D`, the choice is a measured fact rather than an argument.

⚠️ **4-choice restriction.** The chat template hard-codes A–D
(`{% if choices.text|length > 3 %}`), so a 5-choice ARC-Challenge item **silently loses its
fifth choice**, and if the gold answer is the dropped one the item is unanswerable — wrong for
every arm, which would surface as a spurious competence effect. **The 500-item sample is
restricted to 4-choice items, and the exclusion count is reported.**

---

## 3d. H5 — everything except prompt text is held fixed

As specified, H5 was **uninterpretable**. The harness's two BBH templates differ in far more
than prompt text:

| | `_cot_fewshot` (v4.0) | `_cot_zeroshot` (v3.0) |
|---|---|---|
| `max_gen_toks` | **1024** | **absent → default 256** |
| `until` | `["</s>", "Q", "\n\n"]` | `["</s>", "Q:", "<\|im_end\|>"]` |
| `filter_list` | `get-answer` regex | **none at all** |
| metric | `ignore_case` commented out | `ignore_case: true` + 5 `regexes_to_ignore` |

So **both** truncation mechanisms differed between the H5 arms — including a **4× generation
budget gap**, and the generation budget is **H1's own independent variable**. H5 would have
measured prompt format + budget + stop sequences + metric normalization simultaneously.

**Pre-registered fix — held fixed across both prompt variants:**
- `max_gen_toks` = **1024** (the few-shot value) for both.
- `until` = the **few-shot** stop list for both.
- All four rungs applied identically.
- **Only the prompt text varies.**

Recorded as one documented deviation: the zero-shot arm departs from harness-verbatim scoring
**and** generation config, enumerated in the appendix. G2 is unaffected — it tests prompt
rendering, not metric configuration.

**`answer_marker` is NESTED within few-shot, not crossed with prompt format.** The zero-shot arm
runs **`gsm8k_cot` only**, plus BBH. The zero-shot × `hash` cell is **deliberately absent and the
absence is declared here**, because it would be degenerate: the marker is carried *by the
exemplars*, so stripping them leaves nothing to convey `####` except an explicit instruction —
which is a different manipulation altogether (instruction-following rather than
exemplar-imitation). That cell would score ≈0 at rung 1 for a trivial reason, and **a floor effect
cannot support an interaction test.**

Pre-registered prediction, stated so the unbalanced design is explained rather than merely
observed: were the cell run, rung-1 accuracy under zero-shot × `hash` would be near zero for
**all** precisions, yielding no estimable interaction.

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

**`TRUNC_STOPSEQ` is a fourth component of the H4 decomposition, and it is tested for
differentiality by precision.** Pre-registered here so the test is not a post-hoc addition:

> `H4′: measured_strict_delta = competence_delta + compliance_delta + truncation_delta
> + stopseq_delta`

- **Pre-specified test:** is the `TRUNC_STOPSEQ` rate differential by precision (bf16 vs each
  quantized arm), tested by exact McNemar on the paired item-level indicator?
- **If yes**, quantized models emit bare `Q` / blank lines / premature periods more often, and
  this is a genuine compliance-delta mechanism that belongs in the decomposition table with a
  number and its own subsection.
- **If no**, it is a static configuration defect affecting all arms equally, reported as a
  measured rate and a limitation of the benchmark rather than as an effect of quantization.

**Emphasis, pre-committed: H1 remains the lead regardless of how large `TRUNC_STOPSEQ` turns
out to be.** H1 responds to a manipulation we control, has a dose-response curve across three
budget levels, and resolves a published contradiction. `TRUNC_STOPSEQ` has no counterfactual
and no dose-response — it is a bug report unless the differentiality test above is positive, in
which case it is a measured effect but still not a hypothesis with a manipulation behind it.
Recording this now prevents a large stop-sequence rate from tempting a restructure of the paper
around whichever number came out biggest.

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
**GSM8K — `answer_marker` is a two-level FACTOR, both levels confirmatory.**

Both harness GSM8K variants run on the same 1319 items. Neither is primary.

⚠️ **Correction of record.** An earlier draft designated `gsm8k_cot` primary on the grounds
that `gsm8k` was non-CoT and so could not test H1. **That was verified false on 2026-08-25:**
`gsm8k` has `fewshot_split: train` with `doc_to_target: "{{answer}}"`, and GSM8K train answers
are multi-step worked solutions with `<<…>>` calculator annotations ending `#### N` (confirmed
by fetching train rows). **Both variants are few-shot chain-of-thought, and `gsm8k`
demonstrates `####` five times.** Neither does `EXECUTOR-PLAN.md` §1.1's *"instruction-tuned
models essentially never emit `####`"* hold for this task as configured. The primary/secondary
split is withdrawn; nothing was generated under it.

| | `gsm8k` | `gsm8k_cot` |
|---|---|---|
| `answer_marker` | **`hash`** | **`phrase`** |
| shots | 5, sampled from `train` | 8, hardcoded in the YAML |
| CoT in exemplars | yes | yes |
| required marker | `#### N` — **arbitrary** | `The answer is N.` — **natural language** |

⚠️ **The two harness arms differ in FOUR ways, not one** — verified 2026-08-26:

| | difference | status |
|---|---|---|
| (a) | marker: `#### N` vs `The answer is N.` | **the intended manipulation** |
| (b) | shots: 5 vs 8 | confound |
| (c) | exemplar content: sampled from `train` vs 8 hardcoded | confound |
| (d) | `until`: `["Question:",…]` vs `["Q:",…]` | confound |

Items and gold **are** identical (the same 1319). But **(b) and (c) run toward our own
hypothesis:** eight hand-written clean exemplars should teach format compliance better than five
randomly sampled ones carrying `<<48/2=24>>` calculator noise — predicting higher `gsm8k_cot`
compliance for reasons unrelated to marker naturalness, *the same direction the marker prediction
points.* A caveat cannot fix a bias that runs our way.

**So the marker contrast is estimated from a synthetic MATCHED arm, and three arms are
pre-registered with distinct roles:**

| Arm | Kind | Role | In the marker contrast? |
|---|---|---|---|
| `gsm8k_cot` | harness | primary; natural marker, ecological | **yes** |
| `gsm8k_cot_hashmarker` | **synthetic** | the **identified** marker contrast | **yes** |
| `gsm8k` | harness | secondary: Anchor C, ecological claim, sign-agreement check | **no** |

**`gsm8k_cot_hashmarker` construction.** `gsm8k_cot`'s prompt **verbatim**, with **only** the
exemplar terminator rewritten `The answer is N.` → `#### N`. Shots, exemplar text, the `Q:`/`A:`
scaffold and `until` are all inherited from `gsm8k_cot`, so (b), (c) and (d) are controlled
simultaneously. Rung-1 regex becomes `#### (\-?[0-9\.\,]+)`. It is asserted that exactly
`num_fewshot` substitutions occur and never zero — a zero-substitution arm would be byte-identical
to its base, i.e. a null contrast that looks like a real one.

**This arm is NOT harness-verbatim and is labelled as such in the configs, this document, and the
paper.** That is the arm's job, and the label is not to be softened.

**Cost:** 500 items × 9 cells (3 models × {bf16, GPTQ-4bit, AWQ-4bit}) = **4,500 generations**
(~3%). It estimates one contrast whose published magnitude is 41–64 points (arXiv:2601.14277
App. B), so n=500 resolves it with room to spare; and being a marker × precision interaction it
does not need bnb-NF4 or GPTQ-8bit, whose role is dose-response on the reference condition.

**Why the confounded pair is still reported.** `strict_acc(gsm8k)` vs `strict_acc(gsm8k_cot)` is
confounded by (b)(c), but the **within-arm rung1→rung2 gap** is not, to first order: shots and
exemplar content move *competence*, which shifts both rungs together and differences out. The
gap-of-gaps was always the better-identified quantity; the matched arm closes only the residual
channel in which exemplar quality acts on compliance directly rather than through competence.

**Pre-committed division of labour:** the marker effect is estimated from the **matched pair
only**. `gsm8k` contributes Anchor C, the ecological claim ("published harness configs really do
differ this way"), and a **sign-agreement check** against the identified estimate — with (b)(c)(d)
named explicitly wherever it appears. Reporting both halves answers both reviewer attacks:
synthetic-only invites *"you invented a prompt"*, harness-only invites *"your comparison is
confounded."*

**`TRUNC_STOPSEQ` is reported per arm, not pooled across GSM8K**, because (d) means `gsm8k`'s stop
set differs.

**Pre-registered directional prediction:**

> `interaction(hash) > interaction(phrase) ≈ 0`

i.e. the `precision × scoring` interaction is larger under an arbitrary marker than a natural
one, and near zero under the natural one. **The difference between those two interactions is a
three-way interaction that directly instantiates the compliance mechanism.** A near-null on
`phrase` is therefore a **confirmed prediction**, not a disappointing primary result.

Both levels enter the same Holm–Bonferroni family. **Because neither is designated primary,
this is not a licence to report whichever level gives the larger effect** — the directional
prediction above is what is being tested, and it is recorded before any data exists.

`gsm8k` additionally makes **Anchor C** (arXiv:2601.14277) reproducible.

⚠️ **SCOPED 2026-08-30 after reading the source (R19.3, `notes/protocol-survey.md`).** Anchor C
is **llama.cpp GGUF K-quants on CPU** (`commit b7600`, dual Xeon 8488C), not vLLM with GPTQ/AWQ
on a GPU. So what is reproducible is its **extraction gap** — F16 `GSM8K (FE) EM 77.63` vs
`GSM8K (SM) EM 24.64`, one model under two filters, on a task config we also run (`gsm8k`,
5-shot, hash marker, chat template on, `lm_eval` v0.4.9.2). Its **quantized** deltas are *not*
comparable to ours in kind. The substrate difference must be stated wherever the anchor is
invoked. ✅ Its chat-template setting **is** reported (*"prompts were formatted using the default
Llama-3.1-8B-Instruct chat template"*), which confirms R1.2's diagnosis on evidence rather than
plausibility.

Scope cost: +1,319 × 15 ≈ **19,785 generations**, ~20% above the `EXECUTOR-TASKS.md` §2.2
matrix. Approved by the user 2026-08-23.

**Rung 1b — `strict_last`.** The *same* strict regex with `group_select: -1`, i.e. the **last**
match instead of the first.

Added because rung 1 vs rung 2 confounds two distinct things. Neither strict regex sets
`group_select`, so both take the **first** match, while `flexible-extract` sets `-1` and takes
the **last**. Self-correction is common in chain-of-thought (*"The answer is 6. Wait — The
answer is 8."*), so part of any measured strict-vs-permissive gap is **which candidate was
selected**, not how strict the pattern was. Verified: on that string rung 1 returns `6.` and
scores wrong, while rung 2 returns `8` and scores right — with no strictness involved.

The ladder therefore decomposes as:

| step | isolates |
|---|---|
| 1 → 1b | **position-selection** effect (first vs last match) |
| 1b → 2 | **strictness** effect (pattern rigidity) |

Pure re-scoring, zero extra generation. **Rung 1 remains verbatim** — 1b is an addition for
decomposition, never a replacement, and the headline rung-1-vs-rung-2 contrast is still
reported as such, now with its two components separated. For `arc_challenge_chat` rung 1b is
identical to rung 1 by construction (no regex, so no match position to select).

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

## 5a. Rung 2a, and the rung gap decomposed into RECOVERED and DESTROYED

Registered **2026-09-02**, planner ruling R27.1. ⚠️ A **pre-specified extension informed by pilot
data already seen**, with the exploratory numbers stated inline. Not pre-data.

### What was found

Rung monotonicity fails on real generations. Measured on `pilot-v2`, six generations have rung 1
**correct** and rung 2 **incorrect**:

**All six instances, across five distinct items.** The counts must line up because this section's
whole purpose is a corrected count:

| # | model / task / item | arm(s) | rung 1 | rung 2 | class |
|---|---|---|---|---|---|
| 1 | phi gsm8k idx-55 | awq-4bit | `14` ✓ | `Jean can fill 14 bags of lollipops` ✗ | A |
| 2 | phi gsm8k idx-61 | gptq-4bit | `1430` ✓ | `in dollar format with two decimal points…` ✗ | A |
| 3 | phi bbh/boolean_expressions idx-16 | awq-4bit | `False` ✓ | `True` ✗ | B |
| 4 | phi bbh/logical_deduction idx-12 | awq-4bit | `(A)` ✓ | `not (A), (B), or (E)` ✗ | B |
| 5, 6 | llama bbh/dyck_languages idx-0 | gptq-4bit **and** bnb-nf4 | `] ]` ✓ | `] ] ]` ✗ | **C** |

Five items, six instances, because item 5/6 violates on two arms. Classes A and B are the four
**destruction** cases; class C is not destruction (below).

🚨 **CORRECTED 2026-09-02, before this section bound anything.** A first pass described all six
as one mechanism, "the generation continues past its committed answer". Reading the **real**
generations rather than the traced summaries shows **three** mechanisms, and one of them runs the
opposite way, so the destruction claim rests on four of the six cases and not all six.

| class | cases | mechanism | leniency destroying? |
|---|---|---|---|
| **A** | phi gsm8k idx-55, idx-61 | `#### N` then trailing prose; rung 2's last-candidate rule reaches into it | **yes** |
| **B** | phi boolean_expressions idx-16, phi logical_deduction idx-12 | the model states **several** commitments and self-corrects to a wrong one; rung 1 takes the **first** (harness `take_first`), rung 2 takes the **last** | **yes** |
| **C** | llama dyck_languages idx-0 (gptq-4bit, bnb-nf4) | the model says `] ] ]`, which is **wrong**; rung 1 extracts `] ]` **only because BBH's regex drops the final character** | **NO, the reverse** |

**Class C is rung 1 being accidentally correct via a known harness defect**, already pinned as a
test and recorded under G3, and rung 2 is the rung reporting faithfully what the model said. It is
counted as a monotonicity violation because it is one, but it must not be counted as destruction.

⇒ **Flexible extraction is not a monotone relaxation of strict extraction. It is a different
selection rule with TWO-SIDED error.** Classes A and B are the leniency-side error. Class C is a
**strict-side** error in the opposite direction, and reporting both is what makes the claim about
*calibration* rather than about one rung being better.

### Rung 2 is NOT amended, and that is the decisive point

🚨 **Rung 2's last-candidate rule IS lm-eval's `flexible-extract` with `group_select: -1`.** It is
**published practice**, not a defect of ours, and §2.2's headline contrast is
strict-published-practice against flexible-published-practice. Amending rung 2 would replace the
thing being measured with an extractor of our own invention.

**Rung 2a is added instead, as an ADDITIVE robustness rung:** commitment-**anchored** permissive,
which takes the **first** committed answer and the hash marker, and drops the positional
last-number and last-option-letter fallbacks, since those are the rules that can reach past a
commitment. Additive means the pilot's numbers stay valid and nothing is re-run.

⚠️ **CORRECTED TWICE, and the rule list changed once. Recorded in full because the section's
purpose is a corrected count.**

**First correction.** A first version of rung 2a **dropped** the positional `last_number` and
`last_option_letter` fallbacks. Computed on the run rather than on fixtures, that produced **14**
monotonicity violations against rung 2's 6, of which **13 were NON-EXTRACTIONS on
`arc_challenge_chat`**: neither a commitment phrase nor a `####` marker appears there, and
`last_option_letter` was the only rule able to read a generation such as *"The best piece of
equipment ... is C"*. Ruled an **implementation error against §5a's own stated purpose** (R29):
refusing to reach *past* a commitment is not the same as dropping the only rule that can read a
commitment-free generation.

**Corrected rule (R29, option ii): the positional fallbacks are ANCHORED rather than dropped**,
restricted to the pre-commitment span when a commitment exists and applied to the whole generation
when none does. Every edge case is enumerated in `extract_permissive_anchored`'s docstring, because
the code has silently decided four such questions already.

**MEASURED after the correction, not predicted** (R29 explicitly forbade predicting it, having been
wrong once about exactly this):

| rung pair | recovered | destroyed | monotonicity violations | §9h magnitude |
|---|---|---|---|---|
| rung 1 vs **rung 2** | 914 | **6** | 6 | **2** |
| rung 1 vs **rung 2a** | 918 | **1** | **1** | **2** |

🔬 **Rung 2a eliminates all FOUR genuine destruction cases (classes A and B). The single remaining
violation is the class-C `dyck_languages` case**, which is not destruction at all: rung 1 is
accidentally correct there via the dropped final character, and rung 2 reports faithfully.

⚠️ **So §9h's magnitude for the rung-1-vs-rung-2a pair remains 2, and it is now driven ENTIRELY by a
strict-side harness defect rather than by leniency.** It was expected to fall to 1. It did not, the
bound stays wide, and that is reported as measured rather than adjusted.

🔬 **A finding kept from the failed variant** (R29): on `arc_challenge_chat` positional fallback is
the **only** rule doing any work. That narrows what H6's control is a control *for*: ARC-chat is
structurally constrained in its **answer space** (four letters), not in its **extraction surface**,
whereas §1.3 justified it assuming the latter. H6's test is unaffected (zero interaction on ARC,
non-zero on BBH), but it is stated because a reviewer reading the filter list will notice.

### The registered decomposition

The rung gap is a **net of two opposing effects**, and reporting only the net is imprecise. Per
item, at a rung pair `(low, high)`:

- **recovered** — low wrong, high right. Leniency rescued the item.
- **destroyed** — low **right**, high wrong. Leniency lost an answer the stricter rule captured.

Both directions are reported separately, per arm, with Wilson intervals. `net = recovered −
destroyed` is what the §2 companion reports. **Nobody reports the destruction rate of flexible
extraction**, and saying "leniency recovers X pp" silently nets destruction against recovery.

**Exploratory values on `pilot-v2`, stated inline:** 914 recovered against 6 destroyed over 5,100
generations, so recovery dominates by roughly 150 to 1 and destruction is 0.12% overall.

### 🚨 Pre-registered directional prediction, so it is confirmatory rather than post-hoc

**Prediction: the DESTRUCTION rate is differentially HIGHER on quantized arms than on `bf16`.**

The mechanism implies it: destruction requires a generation that continues past its committed
answer, and over-generation is a quantized-arm behaviour. Registered with the direction stated
**before** the reference run, at which n it is a real test either way.

⚠️ **The pilot is suggestive and under-powered, and is labelled as such.** Destroyed counts are
`bf16` **0**, `gptq-8bit` **0**, `gptq-4bit` 2, `awq-4bit` 3, `bnb-nf4` 1, over 900 items per arm.
The direction matches and `bf16` at exactly zero is notable, but n=6 supports no magnitude claim.

⚠️ **Note what this would mean if it holds.** It is a compliance-adjacent measurement artifact
running **opposite** to the paper's main hypothesis: it would mean flexible extraction penalises
quantized arms specifically, which *understates* rather than overstates their measured degradation.
That direction is reported with the same prominence as the rest, per §0's standing rule that both
directions can be true at once.

### Framing this supports

The sharpest support the paper has for framing the contribution as **calibrating** extraction rather
than loosening it. Ren et al. (arXiv:2608.02442) established that leniency **over-credits**, with
8.2–44.1% of correct-graded answers shortcut-hacked. This measures that leniency also **destroys**
correct answers the stricter rule captured. A paper arguing only "be more lenient" walks into both.

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

- `COMPLIANT` — answer present in the canonical expected format.
- `RECOVERABLE` — no canonical format, but rung 2 finds a **committed** final answer.
- `ABSENT` — no identifiable answer (degeneration, refusal, or truncation).

**⚠️ Extraction *succeeding* is not the same as the model *committing*, and the compliance
label uses the latter.** Two cases forced this distinction, both found while building the
scorers and both pre-registered here before any generations exist:

1. **Positional fallbacks are not commitment evidence.** Rung 2's last-rules
   (`last_number`, `last_option_letter`) are faithful to the harness's own
   `flexible-extract` and are kept for **scoring**. But a generation truncated
   mid-reasoning — *"Ann has 3 apples and buys"* — yields `3` under `last_number` while the
   model plainly never committed to anything. Only `boxed`, `commitment_phrase` and
   `hash_marker` count as commitment.
2. **`arc_challenge_chat`'s rung-1 filter has no regex.** It is `remove_whitespace` +
   `take_first`, so it accepts *any* non-empty text. Taken literally every ARC-chat
   generation would be `COMPLIANT`, the compliance column would be vacuous, and the H4
   decomposition would be meaningless on the task that is our **structural control (H6)**.
   The task ships `gen_prefix: "The best answer is"`, so a canonical continuation is the
   option letter alone; anything else is not canonical format. Rung-1 **scoring** stays
   harness-verbatim regardless.

**Why this is load-bearing rather than tidy-minded:** the §1.1 always-compliant stratum is
defined as *items where every arm produced a committed answer*. Under the looser reading,
truncated generations carrying an incidental mid-reasoning number would be admitted to the
stratum — which would corrupt precisely the contrast the identification fix exists to make
identifiable, and would do so **in the direction of our own hypothesis**. Both the
`found` and `committed` flags, and the rule that fired, are recorded on every score record
so the choice is auditable rather than buried.

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

`max_num_seqs` is set by **measurement, not assumption**. §3.3's value of 8 comes from vLLM
issue #51187 observed on *other* hardware; it is also our largest throughput cost, since it
deliberately starves GPU concurrency.

**Pre-registered selection rule:**
- Swept over `{8, 16, 32, 64}` inside the `A7` determinism audit, which already runs 3
  identical repeats per format, using the **same probe set throughout**.
- The selected value is the **largest value that is bitwise-stable across all 3 repeats for
  EVERY format** — i.e. a **minimum over formats**, because §3.3 requires one value identical
  across all arms. A value that is stable for bf16 but not GPTQ is not eligible.
- Then **frozen before `A13`** and identical across every arm thereafter.
- **The full frontier table is reported in the paper** — largest bitwise-stable batch size per
  quantization format. This is direct evidence for H8 at zero additional cost, and strictly
  more informative than reporting one inherited number.
- **If no value above 8 is stable, that is a finding, not a failed sweep**, and it is reported
  as such.

**Pre-specified contingency on the level count** (planner ruling R13.2 — registered here so the
number of levels measured is a plan, not a post-hoc choice). The sweep runs **descending**,
because the quantity wanted is the *largest* usable value:

- The first level stable for every format is selected, and **the level immediately below it is
  then also measured**, to rule out a non-monotonicity. "Stable at 64 but unstable at 32" would
  mean stability is not monotone in batch size, so "the largest stable value" would not be a
  well-defined threshold and the frontier would have to be reported as a **set** rather than a
  cut-off. That outcome is reported if it occurs.
- **The floor (8) is measured only when nothing above it was stable.** Its purpose is to
  guarantee that some value is stable; once a higher level is stable, that is established.
- Consequently the audit costs 2 levels when the top value is stable and up to 4 when it is not.
  This is a deliberate asymmetry: cheap when the answer is unremarkable, expensive only when the
  frontier is genuinely informative.

**Confirmation of the selected value varies ITEM ORDERING, not repeat count** (R13.1). Repeating
an identical probe in an identical order re-presents the scheduler with nearly the same batches,
so repeat count explores the failure surface weakly. Batch-invariance failures arise from batch
*composition*, so the selected value is confirmed with **three orderings of the same 500-item
probe** — ordering 0 being the sweep's own pass, and two independent shuffles at published seeds
(`20260827`, `20260828`). The probe **size is held at 500 throughout**, so disagreement rates
remain comparable between the sweep and the confirmation, and the confirmation is reported as a
**rate** rather than a pass/fail. If the value is stable under repetition but not under
reordering, the single-ordering frontier **overstates** reproducibility, and that is reported.

**Model generality** (R12.3.3). The *mechanism* is read from vLLM's kernel-dispatch source and is
therefore architecture-general; the *rates* are specific to this hardware and model. A
confirmation probe on **Phi-3.5-mini** (bf16 and one 4-bit arm, at the frozen value) tests whether
the rates transfer, chosen over Llama-3.1-8B deliberately because its fused gate+up projection
makes it the architectural outlier and therefore the higher-information second point.

**Registered generation budget for the audit: 38,000** — superseding both §2.6's ~13,500 and the
23,000 registered before the criterion below was retired.

⚠️ **Actual, recorded for honesty rather than revised upward after the fact:** the audit as run
used ~24,000 generations across 21 cells, and R17.3's confirmatory extension (§9a) added a
further 22,000 (4 arms × 500 items × 11 passes). The registered figure was an upper bound and
was not exceeded; it is left as written because a budget adjusted to match what happened is
not a budget.

**⚠️ The repeat-based selection criterion above is RETIRED (planner ruling R15.4).** The pilot
measurement showed why: at `max_num_seqs=8`, three repeats of an identical probe in an identical
order gave **0/20 disagreement for every arm**, while three *reorderings* of the same items changed
**65% of GPTQ-4bit's outputs and none of bf16's*. A criterion defined on repetition therefore
measures an axis with no variance and certifies as "stable" an arm that is grossly unstable on the
axis that matters.

**Replacement, pre-registered:** for each level in {8, 32, 64}, measure **order-induced
divergence** across three orderings of the same probe (seeds `0`, `20260827`, `20260828`), with all
passes executed **inside a single engine process** so that ordering is not confounded with process
state. The frozen value is the largest level order-invariant for every arm; if no level is
order-invariant, that is reported as the frontier and the freeze is made on it explicitly.

Two requirements when freezing under order sensitivity: the **item order is identical across
arms**, so ordering contributes zero *differential* noise between precisions, and **the order is
published** as part of the artifact.

**Three floors, three names** (R15.6), which must not be conflated:
1. **Repetition floor** — identical config, identical order, N repeats. **§9's pre-commitment binds
   to this one**, because §9 protects against claiming effects that are pipeline noise under the
   conditions actually run, and ordering is held fixed and published.
2. **Reordering sensitivity** — reported per arm at three levels (text, extracted answer,
   correctness). This is *not* a floor on our deltas; it is a reproducibility bound for third
   parties.
3. **Pipeline floor** (§4.5) — the bf16 end-to-end double run, unchanged.

If correctness-level reordering divergence exceeds 10 pp, the freeze is escalated before being
made.

🚨 **OUTCOME, AMENDED 2026-08-31 (R24.1): THE TRIGGER FIRED, AND THE FREEZE MOVED 64 → 32.**

The first measurement reported worst strict-correctness 5.4% and worst permissive-correctness
7.0%, and `max_num_seqs = 64` was frozen on that basis. **That measurement was taken while BBH's
few-shot terminators were truncating five of six subtasks to a median of 7 generated tokens
(§9b), so roughly two fifths of the probe was inert and contributed no variance at any level.**
Re-measured with the terminators corrected (job 33, 500 items, qwen2.5-3b, 3 orderings):

| arm | B=8 | B=32 | B=64 |
|---|---|---|---|
| gptq-4bit **permissive** | 5.8% | 8.2% | **11.2%** ← exceeds the trigger |
| gptq-4bit strict | 5.4% | 7.4% | 7.4% |
| awq-4bit permissive | 7.6% | 9.4% | 9.6% |
| gptq-8bit permissive | 6.8% | 7.0% | 7.8% |
| bnb-nf4 permissive | 6.6% | 0.2% | 0.0% |
| **bf16, every level** | **0.0%** | **0.0%** | **0.0%** |

**11.2% exceeds the 10 pp trigger, at the value that had been frozen.** §9 requires the freeze
to be *escalated* before being made, not forbidden — the escalation happened and ruled: **move
to 32.** `max_num_seqs = 32` is frozen.

⚠️ **The substantive point is not the threshold, it is that the original rationale is void.**
The freeze chose 64 because the level-dependence looked like noise (~0.6 SE). On corrected data
the same series is monotone across ~5.4 pp. The inert items were flattening exactly the
dependence the decision turned on.

⚠️ **Why 32 and not 8.** No value is clean; even B=8 sits at 5.8–7.6%. This is a point on a
frontier, not safe harbour. Under **fixed** ordering, order divergence contributes zero
*differential* noise to our deltas (R16.5/R17), so 8's extra ~1.8 pp of margin buys
defensibility on a quantity that does not contaminate our estimates, at 30–50 extra GPU-hours.

⚠️ **32's fragility is reported, not buried:** 9.4% is **three items of margin** at n=500 and
the trend across levels is monotone. The paper states the full frontier, that our originally
frozen value exceeded our own pre-registered threshold, and that we moved.

**What is and is not re-run:** `pilot-v2` is **not** re-run — R11/O11 established that the
annotation pool need not be byte-identical to `main-v1`, since judge validation is a claim about
the population of generations rather than about specific strings; it is recorded as having run at
B=64. ⚠️ **The 10-ordering H8 run did NOT span {8, 32, 64}**, contrary to what R24.1 asserted:
its `order_sweep_levels` was `[64]`, a single-level run at the value this amendment
abandoned, and it is the table the H8 section leads with. It was therefore **re-run at
B=32** (job 37), and the B=64 measurement is retained but **not pooled** with it (§9a).
**The template probe is likewise re-run at B=32** (job 36).

### 9h. Intervals when the observed variance is exactly zero (generic estimator rule)

Registered **2026-09-02**, planner ruling R25.1. ⚠️ **A generic rule, deliberately not an
ARC-specific patch**, registered while the pre-registration is still editable.

**What happens.** On `arc_challenge_chat` the strict and permissive extractions agree on every
item, because its `gen_prefix` is *"The best answer is"* and its `until` contains `"."`, so the
per-item contribution to the H3 interaction is identically zero for every item. A percentile
bootstrap then resamples zeros and returns `[0.00, 0.00]`. Measured on `pilot-v2`: all four arms
reported exactly `+0.00 pp [0.00, 0.00]` at n=240.

**Why that is not acceptable to report.** It is arithmetically correct and substantively
misleading. With `n` items and no observed variation the honest claim is that the effect is
**bounded near zero**, not **known to be zero**, and §9 requires an interval for everything while
forbidding a bare significance flag. A `[0, 0]` interval satisfies the letter of that while
asserting a precision the data does not support, and tables are read without their footnotes.

**The registered rule.** Where the observed variance of the per-item contribution is exactly
zero, the interval is the **Wilson upper limit on 0 successes out of `n`**, converted to accuracy
points by multiplying by the largest magnitude a single differing item can contribute, and
reported symmetrically about zero.

- 🚨 **The per-item magnitude is MEASURED, and it is a property of a RUNG PAIR rather than a
  study constant. AMENDED 2026-09-02 (R27.3), superseding R26.1's derivation.**

  R26.1 derived the magnitude as 1 from rung monotonicity: under it, permissive ⊇ strict, each gap
  in `(alt_perm − alt_strict) − (ref_perm − ref_strict)` lies in {0, +1}, so `|d_i| ≤ 1`.
  **That premise is empirically FALSE on our generations, and the derivation is void.** The
  planner's own words: *"the premise it rested on is empirically false, and measuring rather than
  assuming is what caught it."*

  **Measured on `pilot-v2`: rung monotonicity is violated on 6 of 5,100 generations**, so the
  observed maximum `|d_i|` for the rung 1 vs rung 2 pair is **2**. Mechanism, traced on every case:
  the generation **continues past its committed answer**, and rung 2's last-candidate rule then
  selects a later, wrong span that rung 1's marker-anchored rule never sees. Full account in §5a.

  | rung pair | magnitude | basis |
  |---|---|---|
  | rung 1 vs **rung 2** | **2** | observed; monotonicity violated at a measured rate |
  | rung 1 vs **rung 2a** (§5a) | observed maximum for that pair | anchoring removes the rules that can reach past a commitment |

  - **The rule is registered; the VALUE is frozen at the first `main-v1` measurement.** Re-deriving
    it after every re-analysis would invite drift.
  - ⚠️ **The bound scales LINEARLY in the magnitude**, so a reader who disagrees with the value can
    rescale it directly. That is stated in the paper, which removes any need to trust the choice.
  - Monotonicity violations are reported as a **count and a rate**, per arm, never absorbed
    silently into a wider interval.

- **The interval method is reported in the table itself, not only in prose.** A column that mixes
  bootstrap percentile intervals with zero-event bounds without marking which is which is exactly
  what a careful reviewer flags, so every reported effect carries its `interval_method`.
- **The same rule applies everywhere observed variance is exactly zero**, which makes the
  bootstrap path consistent with the zero-discordance path already used for the transition tables
  and the stop-sequence differential.

⚠️ **The bounded form is the STRONGER result, not a hedge**, because reporting `[0, 0]` against
`+3.33` invites the objection that the control is degenerate.

🚨 **But H6's test is TWO SEPARATE CLAIMS, not interval separation (R26.1).** An earlier framing
compared "+3.33 pp on BBH against bounded within ±1.6 pp on ARC" as a contrast, and the planner
withdrew it: a worst-case bound and a bootstrap percentile interval are **different objects and
must not be read against each other**, so whether they happen to overlap is not the criterion.
H6 predicts the interaction vanishes where extraction is structurally constrained, and the test is:

1. the ARC interaction is **indistinguishable from zero** (bounded, ±1.6 pp at pilot n), and
2. the BBH interaction is **distinguishable from zero** (+3.33 pp, [+0.83, +6.11] at pilot n).

Both are separate claims about separate tasks and both currently hold. A formal *between-task*
claim, if wanted, is reported as the **task × interaction contrast** directly where it is
estimable, rather than by eyeballing two intervals. At the reference n of 500 the ARC bound
tightens by about 1.44× to roughly ±1.1 pp.

### 9a. Ordering as an accuracy statistic — CONFIRMATORY EXTENSION, explicitly not pre-data

Registered **2026-08-29**, per planner ruling R17.4.

🚨 **This is not pre-data and is not presented as such.** The three-ordering result below was
already observed when this section was written, and the git history shows it. It is registered
as a **pre-specified confirmatory extension of an exploratory finding** — discovery followed by
pre-registered confirmation, which is normal practice. Dressing an already-seen result as
pre-data is the version that costs us; stating the sequence plainly is the version that reads as
integrity.

**The exploratory finding, stated inline so the comparison is auditable.** Signed accuracy
spread (max−min) across the three orderings above, 500 items, qwen2.5-3b:

| arm | strict @8/32/64 | permissive @8/32/64 |
|---|---|---|
| bf16 | 0.00 / 0.00 / 0.00% | 0.00 / 0.00 / 0.00% |
| gptq-8bit | 0.60 / 0.40 / 0.40% | 0.40 / 0.40 / 0.60% |
| gptq-4bit | 0.60 / 1.00 / 1.00% | 0.20 / 0.20 / 1.60% |
| awq-4bit | 0.80 / 0.40 / 1.00% | 1.40 / 0.40 / 1.00% |
| bnb-nf4 | 1.60 / 0.20 / 0.20% | 0.40 / 0.20 / 0.20% |

Phi-3.5-mini at `max_num_seqs=8`: bf16 0.00%/0.00%, gptq-4bit 0.60%/1.20%.

**Estimator, frozen.** **Standard deviation of accuracy across orderings is the inferential
quantity**; mean, min and max are reported as descriptive. ⚠️ **The range is NOT an estimand.**
It is unstable and *grows with sample size*, so it cannot carry a population claim at any *n* —
"accuracy has SD σ across random evaluation-set orderings" is defensible, "the spread is 1.6 pp"
never will be. The paper may state an observed range only as a description of the orderings run.

**Design, frozen.**

| | |
|---|---|
| orderings | **10** — the 3 above plus 7 additional. Seeds, PINNED HERE rather than promised: `0, 20260827, 20260828, 20260829, 20260830, 20260831, 20260901, 20260902, 20260903, 20260904`. The first three are the original seeds unchanged and in order, so the 3-ordering result stays exactly recoverable as a prefix. Defined in `quantartifact.determinism.ORDERING_SEEDS_EXTENDED`. |
| arms | bf16, gptq-4bit, awq-4bit, bnb-nf4 |
| `max_num_seqs` | **32 only** (the frozen serving value) — ⚠️ the completed run used **64**, the then-frozen value; see §9's amended outcome. The two are not pooled. |
| items | **500** — the same probe, unchanged |
| model | qwen2.5-3b (audit model) |
| primary rung | **rung 2 (permissive)**, with rung 1 (strict) reported alongside, per §2.2's headline contrast |
| generations | 4 arms × 10 orderings × 500 = 20,000 (14,000 new) |

- ⚠️ **Items are NOT traded for orderings.** Spread is *paired* across orderings, so sampling
  variance largely cancels — but resolution does not. At 250 items one item is 0.4 pp while the
  whole effect is ~8 items, so halving the probe would put the effect inside its own granularity.
- **bf16 receives the same 10 orderings.** Exactly-zero across ten is a materially stronger
  control than exactly-zero across three, and the control is what makes the effect attributable
  to quantization rather than to the engine.

🚨 **THIS RUN SUPERSEDES THE 3-ORDERING FIGURES RATHER THAN EXTENDING THEM, and the two
must not be pooled.** The 3-ordering result was measured before the BBH terminator change
(§9b), so roughly two fifths of its probe items produced no answer under any ordering. Items
that are wrong under every ordering contribute no variance and only enlarge the denominator,
so those spreads are **attenuated** — the 1.6 pp figure is a conservative floor. This run uses
the corrected terminators, so its SD is computed on a different (answerable) population.
Registered before the run so the supersession cannot later look like a choice made after
seeing which set of numbers was more favourable.

**Claims permitted on this statistic.** That the shift is (i) *deterministic* given an ordering
rather than sampling noise — the repetition floor is 0.000%, so it cannot be averaged away;
(ii) *bidirectional*, so no ordering is correct and it cannot be calibrated away; and (iii)
*attributable by intervention* to the Marlin/Machete kernel path, established by the paired
±`VLLM_DISABLED_KERNELS` contrast.

🚨 **Claim explicitly NOT permitted:** comparing our spread to published quantization deltas on
other models, tasks, or protocols (e.g. Kurtic's 405B figure, Red Hat's GSM8K w4a16 delta).
§0.1(b) is the claim that such figures are mutually incomparable, so building a headline on that
comparison would turn the paper's own thesis against it. The comparative sentence — *the ordering
spread is X% of the bit-width effect measured on the same items* — is permitted **only** against
our own reference runs on the same items.

**H7 framing, pre-committed.** For each published degradation figure cited, compute the 95%
CI implied by that paper's own eval size and state whether the reported delta clears it.

**Protocol-reporting column, added 2026-08-30 (R19.3).** The H7 audit table gains a column
recording, per cited source, whether it reports (a) the chat-template setting and (b) the
extraction mode. ⚠️ **Reported as N of M CHECKED, with M stated** — an impression that "most
papers omit it" would be the exact failure §0.1(b) diagnoses. Each row must be verified against
the primary source and recorded in `notes/protocol-survey.md`, where "not reported" means
*checked and absent*, never *not looked for*. **No counted claim about the field goes in the
paper until M covers the bib entries that report a delta.**
Also report our own minimum detectable effect at 80% power. Wording is scrupulously
neutral: **"many papers report aggregate deltas without a paired test"** — **never** "the
deltas are noise". We adopt Kübler et al.'s (ICLR 2026) paired McNemar test rather than
arguing around it.

### 9b. The BBH terminator change, and why the pilot's compliance figures are discarded

Registered **2026-08-29**, per planner ruling R18.1.

**What happened.** Job 3 (`run_id=pilot-v1`, 17 cells x 300 items) was generated with BBH's
`until` taken verbatim from lm-eval's `_cot_fewshot_template_yaml` (v4.0):
`["</s>", "Q", "\n\n"]`. Under our chat-templated prompts those terminators fire almost
immediately, and **five of six BBH subtasks scored 0.0% for every model and every arm,
including bf16 Llama-3.1-8B**:

| subtask | `stop_reason` | `n_gen_tokens` | generation |
|---|---|---|---|
| `word_sorting` | `"Q"` | 1 | *(empty)* |
| `boolean_expressions` | `"\n\n"` | 9 | "Let's evaluate the expression step by step." |
| `object_counting` | `None` | 120 | *(worked -- its bulleted output never emits a blank line)* |

**Mechanism, stated precisely** (the first version of this section got it wrong). `until`
works exactly as specified; the problem is what the strings *mean* in each prompt format. In
the exemplars, blank lines occur **only between Q/A pairs and never inside an answer** --
verified in the rendered prompt -- so `\n\n` marks exactly one event under raw continuation:
the model has finished and begun inventing the next question. Under chat templating,
`A: Let's think step by step.` is the last thing the *user* said, the assistant opens a fresh
turn with no obligation to continue that text, and its own **paragraphing** emits `\n\n`
mid-answer. Separately, every `"Q"` stop has `n_gen_tokens=1` and empty text: the model's first
token continued the Q/A **transcript** instead of answering. `object_counting` survives only
because its exemplar answers are long bulleted lists with single newlines, which the model
imitates. **The terminators encode a formatting assumption, and the chat template invalidates
it.**

**Change (R19.1): the terminator set becomes part of the definition of each template level**,
not a patched constant. At **chat-OFF** lm-eval's `_cot_fewshot` set is correct *by
construction*; at **chat-ON** BBH uses `["</s>", "Q:", "<|im_end|>"]`, which is lm-eval's own
`_cot_zeroshot_template_yaml` (v3.0) set. Both templates are vendored at
`vendor/lm-eval-configs/bbh/`, so the difference is auditable.

⚠️ **Not a "deviation" and not an "upstream fix inherited"** -- both were earlier
mischaracterisations. It is a consequence of the template being a declared factor: each level
carries the terminators its prompt format requires.

**Relevant fact about the library, cited by line rather than asserted:** lm-eval's default is
`apply_chat_template=False`, while `evaluator.py:183` **warns the user to enable it for
instruct/chat variants** -- which all three of our models are. So chat-ON follows lm-eval's own
recommendation, and the library simultaneously recommends chat templates for instruct models
and ships a `bbh_cot_fewshot` config incompatible with that recommendation. That is an
observation about two of its configs, **not** a claim about anyone's results: we never ran
lm-eval's own generation pipeline.

**G2 is unaffected.** G2 asserts byte parity of *rendered prompts*; `until` is a generation
parameter. (The executor initially read the gate as broader than it is.)

🚨 **Consequences for this pre-registration, stated so no post-hoc reading is available:**

1. **Every compliance, termination and truncation figure from `pilot-v1` is DISCARDED**, not
   reinterpreted. `TRUNC_STOPSEQ` 26-45%, `ABSENT` 40-76% and `COMPLIANT` 2-6% are all
   dominated by BBH producing empty generations across 6 of 8 task-conditions. BBH's pilot
   cells are re-run before the freeze and those are the figures of record.
2. **The apparent Qwen inversion is NOT interpreted** -- bf16 truncating more than AWQ
   (135 vs 84 of 300) is consistent with bf16 writing more prose and hitting `\n\n` sooner.
   Recorded here as a rejected reading, not a finding.
3. **H1's budget axis on BBH is only measurable after this change.** With the old set the
   stop sequence bound long before `max_gen_toks`, so the 1x/2x/4x sweep would have
   measured nothing on BBH. The fix is a **precondition of H1 on BBH**, not a cleanup.
4. **The zeroing is reported in METHOD, not the Introduction** (R19.4, correcting R18.1).
   It is **not** a hypothesis -- it is not differential across precision, so it makes no claim
   about quantization deltas -- and the corrected diagnosis does not support an
   introduction-grade universal claim. The Introduction's motivating extreme case is **H8's
   ordering result**, which is fully measured and mechanistically pinned by intervention.
   🚨 **No upstream bug report is filed.** The earlier plan to file one rested on a claim --
   *"anyone running BBH with a chat model silently gets zeros"* -- that cannot be supported
   without running lm-eval's own pipeline, which we did not do. At most, after Sep 2, a
   *question* issue asking whether `bbh_cot_fewshot` is intended for use with
   `apply_chat_template`.
5. **The residual `Q:` stops after the fix are DATA, not defect.** Verified on GPU: BBH median
   generated tokens rose 1-7 -> 294-618, and 2 of 6 llama `word_sorting` items still stop at
   token 2 with empty output because the model began writing the next question. Those are
   genuine `ABSENT` compliance failures and are scored as such. The old `\n\n` stop made every
   subtask look like this, which is why the two could not previously be distinguished.

### 9c. GSM8K's hash arm is a demonstration, NOT an interaction level

Registered **2026-08-29**, per planner ruling R18.2.

Measured in the pilot (100 items, bf16): strict vs permissive is 1.0% vs 62.0% on
qwen2.5-3b, 17.0% vs 75.0% on llama-3.1-8b, 30.0% vs 65.0% on phi-3.5-mini. Instruction-tuned
models do not emit `####` unless the exemplars compel it, so **rung 1 on this arm measures
marker compliance far more than competence.**

- **A 1.0% score is a floor, and a floored metric has no variance for an interaction to
  appear in.** The `gsm8k` (`answer_marker=hash`) arm is therefore **EXCLUDED as a level of
  the precision x scoring interaction (H3)**; `gsm8k_cot` carries the primary interaction, as
  already specified. This was previously argued; it is now empirically justified.
- The arm is **RETAINED as a demonstration** of the marker-naturalness gradient's extreme
  end, and reported as such.
- **Also to be reported, per R18.2:** `gsm8k_cot`'s own strict-vs-permissive gap (the
  gradient's other end, and the direct test of the monotonicity prediction), and the
  **rank inversion** -- strict orders phi > llama > qwen while permissive orders
  llama > phi > qwen, i.e. **the choice of scorer reorders the models.**

### 9d. AWQ reproducibility -- existence claim registered, magnitude deferred

Registered **2026-08-29**, per planner ruling R18.3.

Two byte-different AWQ checkpoints of the **same recipe** (`e7377d975178f192`,
`184c722115d8d9bb`), same 300 items, same engine: strict 20.0% vs 21.0%, permissive 39.67%
vs 39.67%. A third cell re-ran the **same** hash and reproduced bit-identically, which is
what makes the 1 pp attributable to the weights rather than to the pipeline.

- **Registered claim (existence only):** *AWQ quantization is not bitwise reproducible
  across runs of the same recipe, and the resulting weight difference moves strict accuracy
  while leaving permissive unchanged.* The same-hash control is reported alongside it.
- ⚠️ **No magnitude is claimed from 3/300.** `awq-4bit-nooffload` is added as an extra cell
  in the Qwen reference run (500 items, one arm) and the magnitude is quoted from there.
- **Job 1b** (re-quantize one AWQ arm at identical settings, ~30 min) separates *"AWQ is
  inherently nondeterministic"* from *"`offload_device` changes the weights"*. These are
  different claims and the ambiguity is not publishable now that the result is main text.

---

### 9f. Invocation noise — THREE estimand classes, and the estimator (not a number)

Registered **2026-08-31**, planner rulings R21–R23. ⚠️ **Registered as a pre-specified
extension informed by pilot data already seen** (§9b's category (b)), with the exploratory
numbers stated inline. Not pre-data, and the git history shows it.

**What was found.** `pilot-v1` and `pilot-v2` ran the same items in the same order with only
BBH's `until` differing. `gsm8k` and `arc_challenge_chat` should have been bit-identical. They
were not: llama diverged 41–45% of items in **all five arms**, phi on its two base-repo arms
(bf16 45%, bnb-NF4 55%) and not its three checkpoint arms, qwen 0% in all seven. Every recorded
config field on a differing pair is byte-identical. **This affects `bf16`, so it is not H8's
mechanism.** Full record and three refuted hypotheses: `notes/cross-invocation.md`.

🚨 **THREE CLASSES, because the noise behaves differently depending on what crosses an
invocation boundary.** A single floor would be wrong in both directions.

| class | quantities | invocation noise |
|---|---|---|
| **1. none** | a **PER-ITEM LABEL** given one fixed stored generation: a rung's correctness on that generation, its compliance label, its §8 trace-scan result, its found/committed split, its per-item rung agreement | **exactly zero** — no boundary crossed |
| **2. partially cancelling** | a **within-arm** difference of two rungs (the strict−permissive gap) | reduced: the two rungs share the same text, so the noise is common and largely cancels. Pilot max 1.67 pp vs 4.44 pp for raw strict accuracy |
| **3. compounding** | anything **contrasting arms**: precision main effects, §1.1's competence contrast, §1.3's Manski bounds, H4's components, **and H3's interaction** | two independent invocations, variances **add** |

🚨 **AMENDED 2026-09-02 (R28): class 1 is PER-ITEM LABELS, never CELL-LEVEL RATES over them.**
The original wording conflated the two and listed only examples of the first, which made the
second look exempt. The distinction:

* **A per-item label, given one fixed stored generation, is deterministic.** That is genuinely
  class 1, and it is a statement about **scorer** determinism, i.e. whether the analysis
  reproduces from the artifacts.
* **A cell-level aggregate RATE over those labels inherits invocation noise from the
  generations**, because the generations themselves differ between invocations, so the set of
  items carrying any given label differs too. That is **not** class 1.

⚠️ This applies to every example the class-1 row lists. The **compliance rates**, the
**trace-scan rate** and the **per-arm recovered/destroyed rates of §5a** are cell-level
quantities and are therefore **not** class 1, even though the underlying labels are.

⚠️ **And "it is not a difference, so nothing cancels" does NOT imply class 1.** Absence of
cancellation is absence of a *mitigation*: it means the noise does not cancel, which leaves the
quantity worse off rather than exempt. §5a's `recovered` and `destroyed` are therefore **class 2
as an interim label**, and their floor is to be **measured directly from the K=2 cells** rather
than inherited, since they move whenever an invocation flips one rung and not the other, which is
the same event that moves the gap. Because the gap is their *difference*, the measured gap
movement of 1.67 pp is a **LOWER** bound on the components, not an upper one.

⚠️ §5a's **differential prediction** (destruction higher on quantized arms than `bf16`) contrasts
two arms, so it is **class 3**, with the same floor treatment as H3's interaction.

⚠️ **H3's interaction is class 3, not class 1.** Its *outer* difference is between arms, so two
invocation noises subtract and their variances add. Measured pilot movement **2.78 pp**, worse
than either raw accuracy delta on that cell. An earlier ruling placed it in class 1 by
conflating "the scoring axis is within-invocation" with "the interaction is"; it is not.

**The effect is CHURN, not bias, and therefore shrinks as 1/√n.** Measured per-item correctness
churn is **1.7–5.6%** (mean 3.5%) with near-symmetric direction (44 flips, |net| 20 summed). A
resampling check confirms `SD(Δ) = 100·√(churn/n)` to within the finite-population correction
(observed/predicted 0.766; FPC predicts 0.58–0.92 over the tested range, observed 0.56–0.94).

🚨 **THEREFORE THE PREREGISTERED QUANTITY IS THE ESTIMATOR, NOT A NUMBER.** The pilot's floor was
measured at **n=180**; the reference condition is **n=4,238**, ~24× larger, so the floor falls by
~√24.

⚠️ **CORRECTED 2026-09-01: this said n=2,719.** That was the count before `gsm8k_cot` (1,319) was
added as a separate task alongside `gsm8k`; the real per-cell total is
`gsm8k 1319 + gsm8k_cot 1319 + arc 500 + BBH 1100 = 4,238`, which `sm_reference.reference_plan`
computes from `configs/tasks.yaml`. Corrected **before** the first `main-v1` generation, which is
which the git history records. The registered quantity is the *estimator*, so the derived floors
below are predictions and are unaffected in kind; they are restated at the right n. Baking the pilot figure into a threshold would cripple H3 with a number that does not
apply to it.

- 🚨 **Estimator: the CROSS-JOB per-arm invocation SD, measured at reference n**, pooled across
  cells. Class-3 claims carry **NO ×√2 discount**.

  ⚠️ **CORRECTED 2026-09-05, and the correction is here rather than in a log at the end of the
  file, precisely so a reader cannot act on the superseded version.** This originally read
  *"estimated from the K=2 cells at reference n"*, with a ×√2 credit for averaging the two
  invocations.

  **`main-v1` ran all ten registered K=2 cells and that measurement is DEGENERATE.** The two
  invocations have different pids, are separate engine loads, ran up to 4 hours apart, and differ
  on **0 of 4,238 items on every cell**: 20 invocations, **42,380 item-pairs, zero differences**,
  including `llama-3.1-8b|bf16`, the cell that diverged on **41–45%** of items *between* `pilot-v1`
  and `pilot-v2`. So **within-job invocations are bitwise reproducible and the nondeterminism is
  CROSS-JOB only**, and the registered estimator returned **0.000 pp** against a real cross-job
  noise of 1–3 pp.

  **Why 0.000 is the wrong floor.** Within-job determinism does not make the between-arm delta
  noise-free; it makes it **fixed given a job**. Cell X always yields `O_X` inside this job, so the
  delta is deterministic here — but re-run the job and `O'_X ≠ O_X`, and the delta moves. **A
  scientific claim is a claim about what another run would find, not about what our job found.**

  **Two consequences, and both make claims HARDER**, which is the direction a correction should
  move and is why it is stated here rather than buried:

  1. The floor is measured **across a job boundary**: a dedicated job re-running
     `llama-3.1-8b|bf16`, `llama-3.1-8b|gptq-4bit` and `qwen2.5-3b|bf16` under `main-v1`'s exact
     image, config and instance type, so the **only** difference is the job boundary. Driver and
     CUDA recorded on both sides. The pilot's 1/√n extrapolation is reported **alongside as a
     validation of the scaling law**, not as the estimate, since it spans 15×.
  2. 🚨 **The ×√2 credit is WITHDRAWN.** It rested on averaging two invocations halving the
     variance entering H3's interaction. **Averaging two identical runs reduces nothing**, and the
     cross-job variance was untouched because both invocations sat in one job. H3 binds to the
     **full** cross-job floor and its resolution is worse than first projected.

  ⚠️ **What the K=2 cells DID establish** is worth more than the floor they failed to give:
  *within a single job generation is bitwise reproducible; across jobs it is not, so K must be
  measured across job boundaries and not within them.* That has force precisely because repeating
  inside one process is what a careful person would do, and it is what this section originally
  specified. It also **localises the mechanism** to something compiled or cached once per job,
  the first testable hypothesis after four refutations (`notes/cross-invocation.md`).

  **Three dependents take the same cross-job estimate:** the class-2 floor below, §5a's
  destruction-rate floor, and the invocation floor for between-arm accuracy claims. **§9g is
  unaffected**, since with zero within-job divergence the stratum over all invocations equals the
  single-invocation stratum — noted only so that is not mistaken for evidence.
- **Pre-commitment:** no class-3 claim is made for an effect smaller than the pooled
  class-3 floor. Class-2 claims bind to the class-2 floor. Class-1 claims bind to the
  **repetition floor (0.000%)**, unchanged.
- **Preliminary estimate, cited as such with its n:** at n=180 the per-arm SD is 1.39 pp;
  extrapolated to n=4,238 it is **0.29 pp**. ⚠️ **CORRECTED 2026-09-05:** this bullet previously
  gave **0.20 pp** after K=2 averaging and a class-3 floor of 0.29 pp on the reasoning that the
  ×√2 for contrasting two independently-invoked arms cancels the √2 of the K=2 average. **Both
  halves of that are void** — the K=2 average reduces nothing (see the estimator above), so there
  is no √2 to cancel and the class-3 floor is **√2 × the per-arm cross-job SD**. These remain
  predictions, superseded by the cross-job measurement at full n.
- ⚠️ **The one cell that may be bias rather than churn:** phi/bnb-NF4 flipped 9 up and 1 down.
  If a cell's churn is systematically asymmetric its shift does **not** shrink with n. Per-cell
  flip asymmetry is therefore reported alongside the floor, and any cell failing a two-sided
  binomial check on direction is excluded from the pooled churn estimate and reported separately.

**K=2 repetition, frozen — coverage follows the ESTIMAND, not the empirical screen.**

**3 models × {bf16, gptq-4bit, awq-4bit} = 9 cells, plus llama/gptq-8bit = 10 cells at K=2**,
run inside `main-v1` so the two invocations are contemporaneous (days-apart reruns would add
driver and software drift as a confound).

- ⚠️ **Qwen is included despite showing 0.00%.** At n=180 one item is 0.56 pp, so "0.00" means
  *fewer than one item flipped* — not zero with confidence. Qwen also carries the largest rung
  gaps (19–23 pp) and therefore the most H3 signal, making it the worst possible place for an
  unmeasured floor.
- ⚠️ **Stratifying on the empirical screen was rejected**: it biases the pooled floor upward by
  measuring only where the effect was already seen, and leaves the signal-carrying cells
  unmeasured.
- llama/gptq-8bit is added because it is the **worst observed** interaction movement (2.78 pp),
  and leaving the observed worst case without a floor estimate is what a reviewer picks first.
- The remaining dose-response arms (bnb-NF4, other gptq-8bit cells) report against the **pooled**
  floor, stated as an assumption.

### 9g. The always-compliant stratum is itself invocation-dependent

Registered **2026-08-31** (R23). §1.1 defines the stratum as items where **every contrasted arm
committed** — but if invocation noise flips commitment, the stratum is not a fixed set of items
across invocations. §1.1 is a §4.0 identification item and explicitly cannot be patched later,
so the definition is pinned now.

🚨 **Frozen choice: the stratum is defined as items committed in ALL INVOCATIONS OF ALL
CONTRASTED ARMS** — the stricter, smaller, invocation-stable version — rather than on K=2-averaged
commitment. Chosen because it is the version that **cannot be accused of being
invocation-shopped**. Both stratum sizes are reported, so the cost of the stricter choice is
visible rather than hidden.

### 9e. Execution provenance — which registered plans have been run, and where the output is

⚠️ **This section records WHERE, not WHAT.** A pre-registration's value is the plan; importing
results into it would defeat the purpose. This is a pointer table so a reviewer can trace each
registered plan to the artifact it produced without taking anything on trust. Results live in
`notes/RUNLOG.md` (per-run, with cost) and in the paper.

| registered in | run | artifact |
|---|---|---|
| §9's frontier + freeze | `…-determinism` (2026-08-28) | `artifacts/determinism.json` |
| §9's signed accuracy delta | `…-determinism-reanalyze` | `artifacts/determinism_reanalysis.json` |
| **§9a** 10 orderings, SD estimand | `…-determinism-orderings-r2` (2026-08-30) | `artifacts/determinism_ord10.json` |
| **§3b-bis** template factor + decision rule | `…-template` (2026-08-30) | `artifacts/scores/scores_template-v1.json` |
| **§9b** corrected BBH terminators | `…-pilot` `run_id=pilot-v2` | `artifacts/scores/scores_pilot-v2.json` |
| **§9d** AWQ replicate | `…-quantize-replicate-r2` | checkpoint `…awq-4bit__e7377d975178f192-rep2` |
| §10's annotation pool | `run_id=pilot-v2` | `artifacts/generations/pilot-v2/` |

⚠️ **`run_id=pilot-v1` is retained and is NOT the annotation pool.** It was generated with the
defective BBH terminators (§9b) and exists as the evidence for that finding. Any analysis
drawing on it must say which run it used.

## 5b. Harness-discarded recoverable answers, and the two truncation classes on the SCORED prefix

Registered **2026-09-04**, planner ruling R35. ⚠️ **A pre-specified confirmatory extension
informed by pilot data already seen**, with the exploratory observation stated inline. Not
pre-data, and the git history shows it.

### 5b.1 Both truncation classes are defined on the SCORED PREFIX

R34 moved `until` from a generation-time stop to scoring time (§9b-bis), which made
`finish_reason` and `n_gen_tokens` describe the **full** generation. Left unaddressed that would
have redefined §4's classes by accident, so the definitions are pinned here:

| class | definition |
|---|---|
| `TRUNC_STOPSEQ` | a stop string fired, with **no committed answer before it** |
| `TRUNC_BUDGET` | **no** stop string fired **and** the full generation hit the cap |
| `COMPLETE` | EOS, or a stop string with a committed answer before it |

⚠️ The ordering is load-bearing: checking the budget first would classify a generation that ran
past its stop string to the cap as `TRUNC_BUDGET`, even though its scored prefix ended complete
and early. That over-counts the class H1's sweep manipulates.

**Two fields are recorded at generation time because they are unreconstructable afterwards**, the
same rule that already governs `finish_reason` (README §1, task A4): `stop_string_hit` (which
string terminated the prefix) and `n_gen_tokens_scored` (tokens up to it). Schema 1.5.0.

🚨 **H1's estimand is the SCORED length.** A15 names *"mean and median `n_gen_tokens` by
precision, with a CI"* as a citable contribution settling the 2606.25519-vs-2504.04823
contradiction about whether quantization lengthens chain-of-thought. That contradiction concerns
the **answer the model produced**, not text the harness discarded, so the scored length is
primary and the full length is reported beside it. Reporting only the full length would inflate
every arm by its overrun, **differentially**, since the overrun rate differs by arm.

⚠️ **§4's internal-control logic is unchanged and still holds:** `TRUNC_STOPSEQ` on the scored
prefix should **not** respond to the budget sweep. Any apparent trade-off between the two classes
observed before these definitions were in force is **not interpreted** (R35.Q3): the counts moved
because the definitions moved.

### 5b.2 The harness discards recoverable answers, at an arm-differential rate

**The exploratory observation, stated inline and dated.** On `main-v1-smoke3` (2026-09-04, 4 items
per task path, so 120 BBH records), **7 items had an empty scored prefix and 6 of those 7 yielded
a CORRECT answer from the preserved continuation**:

| model | arm | BBH n | empty scored prefix | of those, correct in the continuation |
|---|---|---|---|---|
| llama-3.1-8b | bf16 | 24 | 5 | **4** |
| llama-3.1-8b | gptq-4bit | 24 | 2 | **2** |
| qwen2.5-3b | bf16 | 48 | 0 | 0 |
| qwen2.5-3b | gptq-4bit | 24 | 0 | 0 |

🚨 **Found by the EXTRACTOR, not by a human reading reasoning**, which R35.Q2 required be stated
because it decides what may be claimed. All six were extracted by the `commitment_phrase` rule at
**rung 1 as well as rung 2** — a properly committed answer, e.g. `"...So the answer is syndrome
therefrom."` against gold `syndrome therefrom`. The seventh is a genuine model error (gold `True`,
model committed `False`), which is a useful sign the extractor is not matching indiscriminately.

⚠️ A first report of this described the continuation as ending mid-word, which was an artifact of
truncating the *display* at 220 characters. Corrected before this section bound anything.

**Registered estimand.** The fraction of items whose **scored prefix is empty** whose **preserved
continuation** yields a correct answer, **per arm**, under a named extractor.

- **Extraction rule:** rungs 1 and 2 as already specified in §5, applied to the full generation.
  Reported separately per rung, never pooled.
- 🚨 **§8's false-positive guard applies and is mandatory here.** This is H2's recoverable-failure
  shape with a **configuration** cause rather than a model cause, so it inherits H2's
  conservative/permissive pair and its incidental-match floor, and the rate is **always reported
  net of that floor**. A claim that leniency recovered an answer is worthless without it.
- **Registered directional prediction:** the discard rate is **higher on arms that continue the
  transcript more often**, which on the pilot is the *less* quantized arm (bf16 21% against
  gptq-4bit 8%). ⚠️ Note this runs **opposite** to the paper's main hypothesis, which is why it is
  registered rather than left exploratory.
- **Scope:** an existence demonstration at pilot n, and a rate only at reference n. n=7 supports
  no magnitude.

**Why it is worth registering.** *"Moving the terminator makes visible that the standard
configuration destroys recoverable answers, at a rate that differs by arm"* is on-thesis and, as
far as the survey in `notes/protocol-survey.md` reaches, unmeasured. ⚠️ It is only observable on
post-R34 data: every earlier run destroyed the evidence at generation time, so `pilot-v1`,
`pilot-v2`, `template-v1` and `template-v2` cannot be re-analysed for it.

---

## 10. Annotation — adjudication rule, pre-specified before any disagreement is seen

- ≥250 items, stratified across 3 models × {bf16, 4-bit} × all tasks, **oversampling
  rung-1/2/3 disagreement cases**; agreement on the natural distribution is **also**
  reported. ≥120 double-annotated for IAA. Per-annotator counts reported.
- Source pool is **`run_id=pilot-v2`** (`EXECUTOR-TASKS.md` O11), drawn with the **final
  ⚠️ **CHANGED from `pilot-v1` on 2026-08-29 (R18.1).** `pilot-v1` was generated with BBH's
  broken terminator set and 5 of its 6 subtasks produced empty generations (§9b), so it is
  not a usable annotation pool. `pilot-v1` is **retained, not deleted** — it is the evidence
  for the terminator finding. `pilot-v2` regenerates all 17 cells with the corrected `until`.
  `gsm8k` and `arc_challenge_chat` are expected to reproduce bit-identically, since `until`
  is per-task, which is a free end-to-end determinism check rather than wasted compute.
- (continued) drawn with the **final
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
alternative parser rule added after the first `main-v1` generation, any per-BBH-subtask contrast beyond the three
pre-specified format-gradient levels of §3a, and the stratum competence contrast for any
(model, task) cell whose stratum falls below the 50% threshold of §1.1.

**Now confirmatory, and no longer exploratory — in two distinct categories, and the
distinction is the point.**

**(a) Added before ANY generations existed**, so genuinely pre-data: the `TRUNC_STOPSEQ`
differential-by-precision test of §4, and the `answer_marker` three-way interaction of §2.

**(b) Added AFTER pilot/audit generations existed but BEFORE any `main-v1` generation**, and
therefore registered as *pre-specified confirmatory extensions of exploratory findings*, each
dated and each quoting the exploratory numbers inline: §9a (ordering as an accuracy
statistic), §3b-bis (the template factor and its decision rule), §9c (`gsm8k`'s hash arm
excluded as an interaction level), §9d (AWQ reproducibility). §9b records a configuration
correction and the discarding of the figures it invalidated.

🚨 **These two categories must never be presented as one.** Claiming (b) as pre-data would be
the exact failure this document exists to prevent, and the git history would contradict it.
Stating the sequence plainly is what makes discovery-then-confirmation a normal and
defensible research practice rather than a concealed one.

---

## How this document is maintained

**It is NOT frozen, and there is no amendment log.** When something here turns out to be wrong or
degenerate, **the section itself is corrected in place**, with a dated note saying what it used to
say and why it changed.

🚨 **The reason is readability** (the user's instruction, 2026-09-05). An append-only log means a
reader reaches §9f, believes it, and only discovers 700 lines later that it was superseded. The
document then contradicts itself and you have to read all of it to know what is true. A correction
is only useful where the claim is made.

⚠️ **What that does NOT license.** Every decision still records **whether it was made before or
after seeing the relevant data**, inline, with the numbers quoted. That property is what makes this
document worth anything and it is independent of freezing: §3b-bis, §5a, §5b, §9a–§9d and §9f all
carry it. Silently rewriting a section to match a result, with no note that it changed, is the one
thing this document exists to prevent.

**Corrections so far**, each recorded in the section it affects rather than here:

| section | what changed | direction |
|---|---|---|
| §1.1 | stratum over the three contrasted arms; the 50% threshold dissolved | neutral |
| §1.4 | the claim that H3 is unaffected by selection was **false** for cells with a differential configuration-caused non-commitment rate | **stricter** |
| §3b | superseded by §3b-bis: the template became a two-level factor | neutral |
| §9 | the escalation trigger **fired**; `max_num_seqs` moved 64 → 32 | **stricter** |
| §9f | the K=2 estimator was degenerate; the floor is now cross-job and the ×√2 credit is withdrawn | **stricter** |

Four of the five made a claim **harder** to sustain, which is the direction a correction should move.
