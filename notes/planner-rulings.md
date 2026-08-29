# Planner rulings — R1–R9

**Source:** planner session, via `./ask-planner`, 2026-08-25.
**Answers:** `notes/planner-queries.md` Q1–Q9.
**Authority:** authoritative on planning decisions. Where this contradicts my read of the
repo I must say so and ask again, not quietly pick my own way.

> ⚠️ **Two claims the planner explicitly flagged as UNVERIFIED and told me to check myself**
> (its own execution was denied): (a) `arc_challenge` resolving to 0-shot, and (b) the BBH
> `get-answer` last-character behaviour. Verification status is tracked in §Verification
> below. **Do not act on either until verified.**

**Headline: Q1 and Q4 rested on false premises of mine.** In both cases the correction
changes the action. Q2, Q3 and Q7 found real defects.

---

## R1 — Q1: my premise was wrong, and the correction makes this a better experiment

**What the planner verified.** `gsm8k.yaml` has **no `fewshot_config`** and
`fewshot_split: train`, with `doc_to_target: "{{answer}}"` — the *full* train answer field.
GSM8K train answers are multi-step worked solutions with `<<…>>` calculator annotations
ending in `#### N`.

**Therefore `gsm8k` at 5-shot IS chain-of-thought, and it demonstrates `####` five times.**
My claim that *"its prompt never asks for a chain of thought"* is **false**. Both variants are
few-shot CoT, and neither config sets `max_gen_toks`, so both inherit the same default.

> **D6's stated basis for the primary/secondary split ("H1 needs CoT to truncate") is VOID —
> it does not discriminate between the arms.**

`EXECUTOR-PLAN.md` §1.1's *"instruction-tuned models essentially never emit `####`"* is also
wrong for the 5-shot harness task. It describes zero-shot or chat-templated behaviour, not the
task it cites.

The real difference between the arms:

| | `gsm8k` | `gsm8k_cot` |
|---|---|---|
| shots | 5, from **train** | 8, **hardcoded** |
| CoT in exemplars | yes | yes |
| required marker | `#### N` — **arbitrary** | `The answer is N.` — **natural language** |
| `group_select` on strict | absent → **first match** | absent → **first match** |

The arms hold items, gold, task and competence approximately constant and vary **only marker
naturalness** — a near-clean manipulation of *compliance difficulty*, the exact quantity the
competence/compliance split is built on.

### R1.1 — Neither arm is primary. Register `answer_marker` as a two-level factor.

`answer_marker ∈ {hash, phrase}`, **both confirmatory**, on the same 1319 items.
Pre-register the directional prediction:

> `interaction(hash) > interaction(phrase) ≈ 0`

and note that the **difference between those two interactions is a three-way interaction that
directly instantiates the compliance mechanism.** This converts "which arm do we trust?" into
a test of our own mechanism, and makes a near-null on `phrase` a **confirmed prediction**
rather than a disappointing primary. Same design logic as the BBH format gradient, extended
within GSM8K. Cost: the +19,785 D6 already approved.

### R1.2 — §1.1 needs rewriting, but not onto BBH + `gsm8k`-secondary

Rewrite the mechanism as:

> *Strict extraction requires a marker; when that marker is arbitrary rather than natural,
> compliance becomes a capability the model can fail independently of competence, and
> quantization may degrade it differentially.*

**Why arXiv:2601.14277 got strict 24.64 vs flexible 77.63 on F16 despite 5-shot exemplars:**
almost certainly `--apply_chat_template`, which wraps the few-shot block into one user turn
and makes instruction-tuned models revert to their own prose format.

- **Do NOT add `chat_template` as a factor** — that is another ~2×.
- **(a)** Fix it **ON, single-turn** for all arms, stated with the citation.
- **(b)** Run one probe: one model, {bf16, GPTQ-4}, `gsm8k`, 500 items, template **off**,
  ≈2,000 generations. Quantifies what we held fixed and pre-empts *"why didn't your strict
  scores collapse like theirs?"*
- **(c)** ⚠️ **`add_bos_token=True` + chat template = DOUBLE BOS.** Red Hat's tip is for
  non-template use. **Log the final token IDs of one rendered prompt per model and assert
  exactly one leading BOS.** A silent double-BOS shifts scores and confounds every arm.

### R1.3 — H7 is fine; the format gradient is not

I conflated two things. **H7(a) is arithmetic over other people's eval sizes and is
independent of what our GSM8K arm does — not at risk.** H7(b) is a reported number, not a
falsifiable claim.

MDEs, paired binary, `SE(Δ) ≈ √(p_d/n)`, `MDE ≈ 2.8·SE` at 80% power:

| n | p_d=0.10 | p_d=0.30 |
|---|---|---|
| 1319 (GSM8K full) | **2.4 pp** | 4.2 pp |
| 900 (BBH pooled) | **2.9 pp** | 5.0 pp |
| 600 (4 free-form pooled) | 3.6 pp | 6.2 pp |
| 250 | 5.6 pp | 9.6 pp |
| **150 (one subtask)** | **7.2 pp** | **12.6 pp** |

Pooled BBH is adequately powered and the thesis rests on the pooled cross-task comparison, so
it is fine. **The weak link is the format gradient**: its three levels are 4 subtasks pooled
(n=600), `boolean_expressions` **alone (n=150)**, and `logical_deduction_five_objects`
**alone (n=150)**. Two of three levels cannot detect anything under ~7 pp.

> **Action: raise the two single-subtask contrast tasks to the full 250; leave the four
> free-form at 150.** All six have 250 available. ≈ **+8,400 generations (~6%)**, targeted
> precisely. **Do not add subtasks** — that is new configs we cannot afford.

### R1.4 — Free addition that protects the headline contrast

Both strict regexes lack `group_select` (→ **first** match) while `flexible-extract` has
`group_select: -1` (→ **last** match). So part of any strict/flexible gap is a
**position-selection** artifact, not a strictness artifact — and self-correction
(*"The answer is 6. Wait — The answer is 8."*) is common in CoT, so this bites the `phrase`
arm.

> **Add rung 1b = the strict regex with `group_select: -1`.** Pure re-scoring, zero compute.
> Decomposes the headline gap and closes an obvious attack.

---

## R2 — Q2: CONFIRMED, my strict reading is what §4.0 meant

The `found`/`committed` split is right and the reasoning is right: admitting `last_number`
fallbacks lets budget-truncated generations into the stratum, truncation is treatment-caused,
so the leak runs toward our own hypothesis — the precise failure §4.0 exists to prevent.
`COMMITMENT_RULES = {boxed, commitment_phrase, hash_marker}` with fallbacks scoring but not
conferring compliance is the right mechanization. The ARC-chat canonical-shape fix is also
right; without it the compliance column is vacuous on our H6 control.

**Two additions to pre-register:**

1. **`committed` and `TRUNC_BUDGET` are NOT mutually exclusive and must not be made so.**
   *"The answer is 42. Let me verify: 42 × 2 = "* is both. Truncation **after** commitment
   does not bias competence, so those items **belong in the stratum** — do not filter them.
2. **Pre-register a minimum viable stratum size.** The stratum intersects over 5 precision
   arms, so it shrinks multiplicatively (0.90⁵ = 0.59 if independent; higher in practice).
   Register: **if the stratum is <50% of items for a given (model, task), the stratum
   contrast is exploratory for that cell and Manski bounds become primary there.**

---

## R3 — Q3: scheme CONFIRMED. And I missed that this also breaks H5.

Three-class scheme confirmed, `TRUNC_BUDGET` primary, `TRUNC_STOPSEQ` as the control that
should not respond to the sweep. Good design.

**What I missed — the planner calls it the most consequential item in my report.** Comparing
the two BBH templates:

| | `_cot_fewshot` (v4.0) | `_cot_zeroshot` (v3.0) |
|---|---|---|
| `max_gen_toks` | **1024** | **absent → default (256)** |
| `until` | `["</s>", "Q", "\n\n"]` | `["</s>", "Q:", "<\|im_end\|>"]` |
| `filter_list` | `get-answer` regex | **none at all** |
| metric | `ignore_case` commented out | `ignore_case: true` + 5 `regexes_to_ignore` |

**Both truncation mechanisms differ between the H5 arms**: a **4× token-budget difference**
and a different stop set. As specified, H5 measures prompt format **+ budget + stop sequences
+ metric normalization** — and the budget is **H1's own independent variable**.

> **H5 is uninterpretable as specified.**

**Action:** for H5, hold **fixed across both variants** — `max_gen_toks` = 1024, the `until`
list = the few-shot list, and all four rungs. **Only prompt text varies.** Record as one
documented deviation for the zero-shot arm.

**Emphasis ruling (answers my open question):** **H1 stays the lead; `TRUNC_STOPSEQ` is a
reported component, not a competing lead.** H1 responds to a manipulation we control, has a
dose-response curve, and resolves a published contradiction. `TRUNC_STOPSEQ` is a static
config defect — striking, but no counterfactual and no dose-response, so it is a bug report
rather than a hypothesis test.

> **But convert it into a measured effect: test whether it is differential by precision.** If
> quantized models emit bare `Q` or blank lines more often, it **is** a compliance-delta
> mechanism and belongs in the H4 decomposition as a **fourth component**. Then it earns its
> own subsection with a number without displacing H1.

**One more BBH item — VERIFY, DO NOT TRUST.** `get-answer` is `(?<=the answer is )(.*)(?=.)`.
Reading the semantics, `(.*)` is greedy and `(?=.)` demands a character after the capture,
forcing a one-character backtrack — so the planner **believes it drops the final character
whenever the generation ends exactly at the answer**: `"the answer is 42"` → `"4"`, but
`"the answer is 42."` → `"42"`. That interacts with `until: [..., "Q", "\n\n"]`, which can end
text right at the answer. **The planner's execution check was denied, so this is unverified.**
Add both strings as G3 fixtures and report actual behaviour. If it holds it is a citable
brittleness squarely on-thesis and possibly differential by precision. **Rung 1 stays verbatim
either way.**

---

## R4 — Q4: premise FALSE, and my preferred fix does not address the real confound

**What the planner verified.** `arc_easy.yaml` (which `arc_challenge` includes) has **no
`num_fewshot` field at all** → resolves to **0** in lm-eval. `arc_challenge_chat` says
`num_fewshot: 0` explicitly.

> **Both arms are 0-shot. The mode contrast is NOT confounded with shot count.**
> Option **(b) would spend 7,500 generations fixing a confound that does not exist** while
> leaving the real one untouched.

⚠️ **Verify this myself in the G2 parity test** by rendering both prompts and counting
exemplars — published ARC-C numbers are often 25-shot via CLI override, a plausible source of
confusion.

**The real confound** is prompt content and answer space: `arc_challenge` is
`"Question: …\nAnswer:"` scored over **answer texts** (`doc_to_choice: choices.text`);
`arc_challenge_chat` is a full A/B/C/D instruction with a generated **letter**. Adding shots
changes neither.

**Ruling — run three arms on the same items:**

| arm | prompt | scoring | role |
|---|---|---|---|
| 1 | chat prompt + `gen_prefix` | generate → extract | extraction surface **present** |
| 2 | **byte-identical to arm 1** | log-likelihood over `{" A"," B"," C"," D"}` | extraction surface **absent** |
| 3 | harness-verbatim `arc_challenge` | log-likelihood over answer texts | published-comparable baseline |

> **Arm 1 vs arm 2 is a byte-identical-prompt contrast where the only thing varying is whether
> the answer is generated-then-extracted or read off a likelihood ranking. That is the
> extraction cost, measured directly** — what §2.2 claimed and did not have.

Arm 3 keeps published comparability. **Cost: arm 2 is 4 single-token scorings on an
already-computed prefix — no generation, cheaper than my (b).**

**Also required:** the chat template hard-codes A–D (`{% if choices.text|length > 3 %}`), so
5-choice ARC-C items **silently lose a choice**, and if the gold is the dropped one the item is
unanswerable. **Restrict the 500-item sample to 4-choice items and report exclusions.**

---

## R5 — Q5 registered (L40S/SM89): one ruling + one cheap hedge

1. **Registered.** BI overhead on SM89 unmeasured; `A6` measures it; "does BI work at all on
   SM89" is an `A7` finding, not a precondition. **Prior to attach:** vLLM gives SM90/SM100 a
   cheap cuBLAS-workspace path and requires **Triton GEMM substitution below that**. Ada (8.9)
   is below SM90, so **the 1.2–2× figure is a reasonable prior for L40S, not an inapplicable
   one.**
2. **H8 framing — billing unchanged (never-cut, may become a headline), claim narrowed, and
   the narrowing turned into the strength.** Frame it as: *the mechanism is in vLLM's source,
   not the GPU* — `auto_awq.py` has a BI guard, `auto_gptq.py` / `choose_mp_linear_kernel()`
   have none, which is **source-level and therefore architecture-independent**; only the
   measured *rates* are Ada-specific and we say so. That answers "why one GPU?" head-on.
   **Cheap hedge, take it:** if a spot `g5` (A10G, SM86) is available, run the vLLM
   determinism suite for **one** format for one hour (~$1). Turns the headline asymmetry from
   n=1 architecture into n=2 and kills the objection.
3. **Registered, no action.** §9 trigger 3 unchanged. One consequence: at `max_num_seqs=8` on
   864 GB/s we are decode-bound, so **the Q8 sweep is worth more on this substrate than on
   A100** — it buys back throughput we no longer have.

---

## R6 — Q6 registered: matrix amended, §2.6 table superseded

| Source | Δ |
|---|---|
| §2.6 baseline | 99,000 |
| D6 second GSM8K variant | +19,785 |
| R1.3 BBH contrast tasks → 250 | +8,400 |
| R1.2 chat-template probe | +2,000 |
| R4 ARC arm 2 (log-likelihood) | +0 |
| R1.4 rung 1b, R3 stop-seq analysis | +0 (re-scoring) |
| **Revised** | **≈ 129,000** |

Order-of-magnitude only. **`A6`'s measured throughput supersedes it**, and at 1.88× price with
~2.4× less bandwidth, §2.6's "order of magnitude of headroom" is now nearer **3–4×**. §9
trigger 3 is the operative constraint, not the dollar figure.

---

## R7 — Q7 registered, and the deviation must be WIDENED

My action (four rungs applied identically; BBH rung 1 uses the `cot_fewshot` regex for both) is
**correct and confirmed**. **Widen per R3 to cover `max_gen_toks` and `until`.** My instinct
about the blanket sentence is right — use this phrasing rather than "we use the standard
harness":

> Prompts are rendered byte-identically to `lm-evaluation-harness` v0.4.12 and verified by an
> automated parity test. Scoring is applied uniformly by us across all conditions rather than
> inherited per-task, because the harness's own BBH few-shot and zero-shot configurations
> differ in metric normalization, generation cap, and stop sequences — differences that would
> otherwise be absorbed into our prompt-format contrast. All deviations are enumerated in
> Appendix [X].

G2 unaffected — it tests prompts, not metric configs.

---

## R8 — Q8 registered: sweeping `max_num_seqs` beats the plan's fixed value

**Agreed.** §3.3's value came from #51187 on other hardware; measuring our own frontier
dominates, especially given R5.3. Two constraints:

1. **The chosen value is a MIN over formats** — §3.3 requires it identical across arms, so
   take the highest value bitwise-stable across 3 repeats for **every** format, same probe set
   throughout.
2. **Report the frontier table in the paper** — free contribution, direct evidence for H8.

If **no** value above 8 is stable, **that is also a finding, not a failed sweep.** Freeze
before `A13` as pre-registered.

---

## R9 — Q9 registered

Accepted; neither stale premise changed a decision. `tectonic` being present means **G6 no
longer depends on Overleaf** — §11.4 comes off the risk register; keep Overleaf as the
co-author surface but `make -C paper g6` is authoritative. Disk at 24 GB removes the §10 disk
risk; the rule that nothing on the user's machine gets deleted without their say-so stands.

---

## Actions for the executor

**Pre-register before Sep 2:**
1. `answer_marker` factor with the directional prediction (R1.1)
2. Chat template ON single-turn + off-probe + double-BOS assertion (R1.2)
3. BBH contrast tasks → 250 (R1.3)
4. Rung 1b, `group_select: -1` (R1.4)
5. `committed` ∧ `TRUNC_BUDGET` compatible; stratum <50% ⇒ Manski primary (R2)
6. `TRUNC_STOPSEQ` as a fourth decomposition component, tested differential-by-precision (R3)
7. H5 holds budget / `until` / rungs fixed (R3)
8. ARC three-arm design, 4-choice items only (R4)
9. `max_num_seqs` min-over-formats, frozen before `A13` (R8)

**Add to G3:**
- the two BBH `get-answer` fixtures (`"the answer is 42"` and `"the answer is 42."`)
- the `gsm8k_cot` first-vs-last self-correction fixture
- a one-leading-BOS assertion per model

**Verify rather than trust the planner:**
- `arc_challenge` resolving to 0-shot
- the BBH `get-answer` last-character behaviour

**Do NOT do:** Q4 option (b).

---

## Round 2 — R9–R11 (2026-08-26)

Asked after G2 passed, batched per D13. Answers R9–R11.

### R9 (Q-A) — Add a **synthetic matched arm**. Option (2), at **4,500 generations, not 19,800**.

**My cost estimate was 4.4× too high.** I assumed 1319 items × 15 cells. The arm estimates
exactly one contrast — the rung1→rung2 gap under `####` vs under `The answer is N.` — and
published evidence puts that gap at **41–64 points** (2601.14277 App. B), so n=500 resolves it
with room to spare. It is a marker × precision interaction, so it needs bf16 + the two
pre-quantized 4-bit methods only, not bnb-NF4 or GPTQ-8bit (whose role is dose-response on the
reference condition).

> **500 items × 9 cells (3 models × {bf16, GPTQ-4, AWQ-4}) = 4,500. ~3% of the matrix.**

**Construction, exactly:** take `gsm8k_cot`'s prompt **verbatim** and change **only** the
exemplar terminator `The answer is N.` → `#### N`. Inherit the 8 hardcoded exemplars, their
reasoning text, the `Q:`/`A:` scaffold, `num_fewshot: 8`, and
`until: ["Q:", "</s>", "<|im_end|>"]`. Rung-1 regex becomes `#### (\-?[0-9\.\,]+)`. **That
controls (b) shots, (c) exemplar content and (d) `until` simultaneously**, because all three are
inherited from `gsm8k_cot` rather than from `gsm8k`. Task id `gsm8k_cot_hashmarker`. Label it a
**synthetic, non-harness-verbatim arm** in configs, prereg and paper — do not soften that, it is
the arm's job.

**Why "accept and caveat" was not available:** my (b)/(c) confound **runs toward our own
hypothesis.** Eight hand-written clean exemplars should teach format compliance better than 5
randomly sampled ones carrying `<<48/2=24>>` calculator noise — predicting higher `gsm8k_cot`
compliance for reasons unrelated to marker naturalness, *the same direction the marker
prediction points*. That is the failure mode R2 exists to prevent, one factor over, and a
caveat does not fix a bias that runs our way.

**What the original design already bought** (worth stating in the preref, since it justifies
reporting the confounded pair at all): `strict_acc(gsm8k)` vs `strict_acc(gsm8k_cot)` is
confounded by (b)(c), but the **within-arm rung1→rung2 gap** is not, to first order — shots and
exemplar content move *competence*, which shifts both rungs together and differences out. The
gap-of-gaps was always the better-identified quantity. The matched arm closes only the
**residual** channel: exemplar quality acting on compliance directly rather than through
competence.

**Three arms with roles assigned — this is the ruling, not the arm alone:**

| Arm | Kind | Role | In the marker contrast? |
|---|---|---|---|
| `gsm8k_cot` | harness | primary; natural marker, ecological | **yes** |
| `gsm8k_cot_hashmarker` | **synthetic** | the identified marker contrast | **yes** |
| `gsm8k` | harness | secondary; Anchor C + evidence published configs differ | **no** |

**The marker effect is estimated from the matched pair ONLY.** `gsm8k` does three other jobs —
Anchor C, the ecological claim, and a **sign-agreement check** against the identified estimate —
with (b)(c)(d) named explicitly. This retires Q-A rather than caveating it: the confounded
comparison stops carrying inferential weight.

Both halves are needed: synthetic-only invites *"you invented a prompt"*; harness-only invites
*"your comparison is confounded."* Both, with a stated division of labour, answers both.

**Also:** report `TRUNC_STOPSEQ` **per arm**, since (d) means `gsm8k`'s `until` differs. Break it
out rather than pooling across GSM8K.

### R10 (Q-B) — Zero-shot runs `gsm8k_cot` **only**. `answer_marker` is **nested within few-shot**, not crossed with prompt format.

The zero-shot × `####` cell is **degenerate**. The marker is carried *by the exemplars*; strip
them and nothing conveys `####` except an explicit instruction — a different manipulation
(instruction-following, not exemplar-imitation). It would score ≈0 at rung 1 for a trivial
reason, and **a floor effect cannot support an interaction test.**

No cost change. **Declare the nesting in the prereg and state the floor-effect prediction as the
reason the cell is absent** — explaining an unbalanced design pre-empts the reviewer question
instead of inviting it.

### R11 (Q-C) — Score the **space-prefixed** continuation, and assert single-token-ness per model.

`gen_prefix` ends `…The best answer is` with **no trailing space**, so the continuation the
generative arm must produce begins with one. The log-likelihood arm is a valid counterfactual
only if it scores **what the generative arm would have to emit**.

Scoring `"A"` instead would measure a tokenization the model essentially never produces after
`is`, and that penalty **varies by model and precision** — exactly the artifact shape that would
manufacture an H6 result on the never-cut negative control.

Independent confirmation of the convention: `arc_easy.yaml` scores
`doc_to_choice: "{{choices.text}}"` against a `doc_to_text` ending `"Answer:"`, and lm-eval's
default `target_delimiter` is `" "` — native ARC-C log-likelihood already scores space-prefixed
continuations.

**Two guards, both required:**
1. **Assert single-token-ness per tokenizer** and record the four token IDs per model in the
   prompt manifest. If a model splits it, that model needs explicit length normalization and the
   discrepancy is **reported, not absorbed**.
2. **Validate empirically on the pilot:** measure the distribution of the actual first token
   emitted after `The best answer is` in the ARC-chat generative outputs. If it is overwhelmingly
   `▁A`/`▁B`/`▁C`/`▁D`, the choice is a measured fact rather than an argument. ~10 lines, and H6
   is never-cut.

### 🚨 R11 addendum — G3's ARC cross-validation criterion was pointed at the wrong instrument

Our ARC-ll arm scores **option letters** on the **chat prompt**; lm-eval's native
`arc_challenge` scores **full answer texts** on the `Question:`/`Answer:` prompt. Different
instruments answering different questions — both legitimate, **not interchangeable**.

> **The G3 ≤0.5 pp cross-validation must run against lm-eval's NATIVE `arc_challenge` config**
> (= our arm 3). Run against our arm 2, the criterion would have measured nothing.

Recorded in `notes/gates.md`; arm 3 exists precisely to be the cross-validation target.

### Round cost and the trim order

**+4,500 (~3%).** The +30% already approved stands; this does **not** add another 20%. The `A6`
contingency still governs — if measured SM89 throughput pushes the reference block past
**Sep 13**, the trim order is **ARC few-shot → zero-shot → `gsm8k` secondary → BBH last**, and
**BBH never drops below 250/subtask.**

---

## Verification

All three verified **2026-08-25**. **The planner was right on all three; I was wrong on two
premises.**

| Claim | Status | Evidence |
|---|---|---|
| BBH `get-answer` drops the final character when text ends at the answer | ✅ **CONFIRMED** | Executed `re.findall(r"(?<=the answer is )(.*)(?=.)", …)`. `"the answer is 42"` → `'4'`; `"So the answer is False"` → `'Fals'`; `"the answer is (B)"` → `'(B'`. With a trailing period all three are correct. Pinned in `tests/test_extract.py::test_bbh_get_answer_drops_final_character_without_trailing_punctuation` |
| `arc_challenge` resolves to 0-shot | ✅ **CONFIRMED** | `grep -c num_fewshot vendor/lm-eval-configs/arc/arc_easy.yaml` → **0**. `arc_challenge.yaml` is only `include: arc_easy.yaml` + task/dataset_name. So **my Q4 premise was false** and option (b) is correctly dropped. Still to re-confirm by rendering in the G2 parity test. |
| `gsm8k` 5-shot exemplars are CoT and demonstrate `####` | ✅ **CONFIRMED** | Fetched `openai/gsm8k` `train` rows via the datasets server. Answer field: `"Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"` — 3 lines, `<<>>` calculator annotations, ends `#### N`. With `doc_to_target: "{{answer}}"` and `fewshot_split: train`, **`gsm8k` at 5-shot is chain-of-thought and shows `####` five times. My Q1 premise was false and D6's stated basis is void.** |

### Additional harness quirk found while writing the fixtures

`gsm8k_cot`'s strict regex is `The answer is (\-?[0-9\.\,]+).` — `\.` sits **inside** the
character class, so the capture greedily swallows the sentence period:
`"The answer is 6."` → `"6."`, not `"6"`. The harness absorbs this at metric time via
`regexes_to_ignore: ["\\.$"]`; we absorb it via numeric comparison. Pinned as a test so it
cannot silently change.

Also pinned: on a self-corrected generation (*"The answer is 6. Wait… The answer is 8."*) rung
1 returns `6.` and scores **wrong**, while rung 2 returns `8` and scores right — a pure
**position-selection** artifact, which is exactly what R1.4's rung 1b exists to separate from
the strictness artifact.

### Round 2 verification (2026-08-26)

| Claim | Status | Evidence |
|---|---|---|
| R11: `" A"` is a single token per model | ⚠️ **TRUE FOR 2 OF 3 — guard #1 fired** | See below. Qwen and Llama yes; **Phi-3.5-mini NO**. |

**R11's guard #1 did its job on the first run.** Measured `encode(x, add_special_tokens=False)`:

| model | `" A"` | tokens | `"A"` | tokens |
|---|---|---|---|---|
| Qwen2.5-3B | `[362]` | `['ĠA']` ✅ 1 | `[32]` | `['A']` |
| Llama-3.1-8B | `[362]` | `['ĠA']` ✅ 1 | `[32]` | `['A']` |
| **Phi-3.5-mini** | **`[29871, 319]`** | **`['▁', '▁A']` ❌ 2** | **`[319]`** | **`['▁A']` ✅ 1** |

**Cause.** Phi-3.5 uses a **SentencePiece** tokenizer where `▁A` *already encodes the leading
space*. Encoding the literal `" A"` therefore emits a redundant standalone `▁` (29871) **followed
by** `▁A`. Qwen and Llama use byte-level BPE, where `ĠA` is the space-prefixed token and the bare
`"A"` is a genuinely different token.

**So "always prefix a space" is the wrong rule** — it is the same shape as the double-BOS trap:
a blanket convention that is right for two models and wrong for a third.

**Implementation that is correct for every tokenizer convention: tokenize in context and
difference.** Encode `prefix` and `prefix + continuation`, and take the trailing ids that differ.
That is exactly what the model sees, so it is correct by construction regardless of whether the
tokenizer is BPE or SentencePiece, and it needs no per-family special-casing.

R11's substantive ruling is **unaffected** — we still score the continuation the generative arm
would actually emit (space + letter). Only the mechanism for obtaining its token ids changes, and
the recorded per-model ids in the manifest now become the evidence that it was done right.

## R12 — Job 1 complete; calibration reporting, the AWQ reproducibility finding, job 2 scope

Asked after job 1 finished 9/9. Four rulings.

### R12.0 Calibration tokens — report MEASURED, per model, and prove corpus identity

Confirmed: report the measured count, not `num_samples × max_seq_length`. *"The product describes
a ceiling, not a corpus."* The prereg is not frozen until Sep 2, so this is a **pre-freeze
correction, not an amendment**. Registered phrasing:

> 512 sequences from `ultrachat_200k` `train_sft`, `max_seq_length` 2048, no padding,
> seed 20260823 → N tokens (55.3% of the 512×2048 ceiling, since sequences shorter than 2048
> contribute their own length). Identical corpus for the GPTQ and AWQ arms of each model.

Two requirements, because §2.5's normalisation claim rests on them:

1. **Report per MODEL, not as a range.** My 579,588–581,138 spread is across *tokenizers*, not
   methods — same text, different models. A range reads as calibration drift between arms.
2. **Assert corpus identity, don't infer it from the seed.** Hash the realised tokenized
   sequences and assert equality across the GPTQ and AWQ arms *within* each model. *"Seed
   equality implies it; a hash proves it, and this is the exact claim the '8× budget difference'
   warning exists to protect."*

### R12.1 AWQ non-reproducibility is a MAIN-TEXT finding — fund it, and widen it

Run the replicate. *"Both outcomes are results"*: inherent AWQ nondeterminism is novel and
adjacent to H8; attribution to `offload_device` is sharper still, since *"a pure data-movement
flag changing published weights is a striking claim about llmcompressor."*

Four changes to what I proposed:

- **Add a GPTQ-4bit Qwen replicate too** (~10 min). A one-method finding invites *"is this just
  AWQ?"*; two methods makes it a claim about the practice.
- **Emit per-file SHA-256 from inside the job** from now on. ETag + sampled windows already
  settle that the existing pair differs; new runs should be exact.
- **"The bytes differ" is NOT the reportable quantity.** 6/13 differing windows does not
  distinguish last-bit float noise from the grid search picking *different scale values*. Report
  max and median relative difference in the scale tensors and in the dequantized weights —
  *"that distinction is the entire scientific content."*
- **Answer whether it moves ACCURACY, for free.** The superseded `qwen2.5-3b/awq-4bit` is still
  in S3, so add it to the pilot as a 16th cell beside the current one, 300 items each
  (+600 generations). *"Weights differ, scores agree within the noise floor"* is reassuring and
  valuable; *"weights differ and scores differ"* is significant. Either beats a byte diff with no
  accuracy consequence attached.

**Placement: a subsection of the main-text determinism contribution — not Limitations, not an
appendix.** It makes H8 two-part: is the *artifact* reproducible, and is *inference on it*
reproducible. Numeric tables to the appendix. Scope explicitly to the models and methods measured.

### R12.2 Hashing `offload_device` was right — but the taxonomy I used was wrong

Keep it. The recipe-vs-resource frame is the wrong partition; the right one is **output-affecting
vs provably-NOT-output-affecting**, and `offload_device` is not in the second class: *"Moving
tensors between host and device changes reduction order, and reduction order changes
floating-point results. A 'pure data-movement setting' is not provably output-neutral on
floating-point hardware."*

The consequences are asymmetric: hashing unnecessarily costs a rebuild (~$12, already paid); not
hashing when it matters puts two different weight sets under one identifier in a released
artifact — an **unrecoverable provenance error**. Applying it to all three AWQ arms was also
correct, since otherwise Phi would differ from Qwen and Llama in an uncontrolled way on exactly
the arm whose architecture is already the outlier.

Follow-ons: **ship a human-readable manifest of which parameters enter the hash**, so a reader
sees the decision rather than reverse-engineering it. And **if the replicates show `offload_device`
is not the driver, do NOT un-hash it** — over-conservative and harmless beats another rebuild.

One sentence for the paper: **R12.2 and H8 are the same phenomenon.** Batch composition is
nominally "just scheduling" and changes inference output; `offload_device` is nominally "just data
movement" and changes quantization output.

### R12.3 Job 2 scope confirmed, with three additions

Confirmed as designed: probe composition (*"BBH inclusion is what surfaces #48271's RMSNorm
leak"*), descending sweep with early exit, floor always measured, Marlin-disabled fallback only
where an arm proved unstable. Descending is explicitly endorsed as **better than the ascending
version the plan originally specified** — it finds the highest usable value first and self-adapts.

1. **If early exit fires at 64, still measure 32.** *"Stable at 64, unstable at 32"* would be a
   non-monotonicity that invalidates the frontier framing, and it is cheap to rule out. Otherwise
   the frontier is two points and the H8 contribution is thin.
2. **10 repeats at the chosen value before freezing.** Three repeats *picks* the value; ten
   *confirms* it. Everything downstream depends on this one number.
   ⚠️ **This EXTENDS R8, which as recorded specifies 3 repeats and does not mention 10.** Not a
   contradiction — select-then-confirm is coherent — but noted rather than silently merged.
3. **One model establishes the MECHANISM, not the RATES — add a Phi-3.5-mini confirmation probe.**
   My source-level argument is accepted and is what the write-up should lead with. The hole:
   #48271's leak depends on token count and block-size selection, which are model-dependent, and
   Phi's fused 3072→16384 gate+up projection — *the same architectural quirk that broke the AWQ
   memory budget* — could take a different kernel path. Run bf16 + one 4-bit arm at the frozen
   `max_num_seqs`, 3 repeats: 2 × 500 × 3 = **3,000 generations**, ~30–40 min. Phi over Llama
   deliberately: the architectural outlier is the higher-information second point.

H8 can then be stated honestly: mechanism from vLLM's source (architecture-general), rates from
L40S/SM89 (architecture-scoped), model-generality checked on the outlier.

### R12.4 Sequencing under 1 slot and 20 min–2.5 h queue waits

The Q1 requantizes need the **llmcompressor image**, so they cannot ride inside the vLLM
determinism job (§4's two-image split).

| order | job | why here |
|---|---|---|
| **next** | **Job 2 — determinism**: Qwen 5 arms × sweep, + the 32-level check, + 10-repeat confirmation, + Phi 2-arm confirmation | Blocks everything: freezes `max_num_seqs`, delivers G4 and the tight estimate |
| **then** | **Job 3 — pilot**, +2 cells (superseded and current `qwen/awq-4bit`) | Answers R12.1's accuracy question with no extra job; closes G3, Anchor A, the `TRUNC_STOPSEQ` trigger |
| **opportunistic** | **Job 1b — two requantizes** (AWQ + GPTQ replicates, quant image, ~30 min) | Attribution only. **Off the critical path** — slot it whenever the queue is free |

This gets R12.1's most valuable component (does it move accuracy?) without spending a queue wait
on it, and keeps the Sep 2 freeze dependent only on jobs 2 and 3.

## R13 — Job 2 budget: none of my four options. The confirmation was testing the wrong axis.

I costed R12.3 at 50,500–65,500 generations (38–49% of the study's 133,500) and offered four
trims. All rejected, for a reason that is better science *and* cheaper.

### R13.1 The 10-repeat confirmation measures the wrong thing — vary ORDERING, not repeats

Batch-invariance failures happen because **batch composition** changes float reduction order. *"Ten
repeats of the same 500 items in the same order re-presents the scheduler with nearly the same
batches every time."* Timing jitter in continuous batching means it is not a pure no-op, but **per
generation spent, repeat count is a weak way to explore the failure surface** — and it was the axis
I was buying 22,500 generations of.

The axis that varies the mechanism is **item ordering**: different order → different co-residency →
exactly what #51187 was about (44 co-resident sequences diverging where 8 did not).

> **5 arms × 3 ORDERINGS × 500 items.** Ordering 1 is the sweep pass already run at the chosen
> level; orderings 2–3 are independently shuffled, **seeds published**. Marginal cost **5,000**,
> not 25,000.

Sequence-length diversity matters for the same reason (#48271's RMSNorm leak is token-count
dependent), which is also **a second argument against my option B**: shrinking 500 → 200 items
*"cuts the diversity that carries the signal and keeps the redundancy that doesn't."* Probe size
stays 500 throughout, so the rate-comparability concern dissolves and this is reported **as a
rate**, not as a pass/fail.

### R13.2 Make the level count CONDITIONAL on the finding

The floor exists so that if 64 is unstable we know *some* value is stable. If 64 is stable that
purpose is already served, so the floor is no longer unconditional:

* **64 stable** → measure 32 for the non-monotonicity check, **skip 8**. 2 levels = 15,000.
* **64 unstable** → walk down 32/16/8 as designed. Up to 4 levels = 30,000 — *"and in that world
  the frontier is genuinely informative, so the spend is earned."*

*"Cost becomes cheap when the answer is good news and expensive only when it's interesting. That's
the right shape for a gating audit."*

### R13.3 Revised job 2 budget

| stage | best | worst |
|---|---|---|
| sweep (2 or 4 levels) | 15,000 | 30,000 |
| ordering confirmation (marginal) | 5,000 | 5,000 |
| Phi model-generality | 3,000 | 3,000 |
| Marlin fallback (only if unstable) | 0 | 7,500 |
| **total** | **23,000** | **45,500** |

**17% of the study best case**, down from 38–49%. *"Coverage is not reduced — redundancy is."*

### R13.4 `max_run` = 172,800 (48 h). A generous cap is FREE.

My option C's justification was wrong: **you are billed for time used, not for the cap.** 48 h is
generous enough that a timeout is unlikely at any plausible rate and short enough that a hang caps
exposure at ~$134 rather than the ~$336 a 5-day cap allows. Per-cell checkpointing already makes a
timeout cost a queue wait rather than work.

### R13.5 Smoke first — but for a reason I did not cite

**No generation cell has ever executed on this substrate.** Job 1 was quantization. Engine init
under batch invariance on SM89, prompt materialization from the frozen artifact, JSONL writing and
checkpoint resume are *all* unexercised on the real box. Read the crude gen/hour off the smoke —
that is my option C's entire benefit for ~10 minutes of GPU on a step I owe anyway.

**A6's deliverable falls out of job 2's own per-arm wall times. It does not need its own job.**

### R13.6 The payoff is larger than I said

If the sweep finds 64 stable, the ~110k remaining generations run on a decode-bound L40S at
plausibly **3–5×** the throughput of `max_num_seqs=8` — *"roughly the difference between ~40 and
~10 GPU-hours downstream."* Job 2 pays for itself several times over, *"which is exactly why it
should be paid once, properly — and why I'm cutting redundancy rather than coverage."*

### R13.7 Planner owns the estimate miss; record 23,000 as the honest number

§2.6's ~13,500 was the planner's and *"was too low"* — it predated the non-monotonicity level, the
ordering confirmation and the model-generality probe. **Update `configs/tasks.yaml` and §2.6 to
23,000 rather than treating the overrun as drift**, and note in the prereg that the sweep's level
count is **conditional on the 64-level result**, so it reads as a pre-specified contingency rather
than a post-hoc expansion.

## R14 — Serving config: four BI coverage gaps, and assert the ENGINE not the intent

Raised after the first determinism smoke's engine log exposed what vLLM 0.26.0 actually does
versus what `configs/models.yaml` declares. Three questions, and the answer upgrades H8.

### R14.1 Disable flashinfer autotune — and measure the contrast on GPTQ-4bit, not bf16

**Disabling is correct on the merits, independent of the confound.** BI mode's premise is a
*fixed, known* kernel set; an autotuner selects kernels by **startup benchmark**, i.e. by a noisy
timing measurement, so the configuration is not identical across runs **by construction**. That
violates §4.6's "reproducibility holds only within an identical configuration."

**My proposed contrast arm was wrong.** I suggested bf16 to "isolate cleanly"; the ruling is
**GPTQ-4bit**, because it is the arm H8's claim is *about* — no BI guard, keeps Marlin. If GPTQ-4
is stable with autotune off and unstable with it on, that is a crisp quotable result:
*kernel autotuning alone is sufficient to break bitwise reproducibility.* Marginal cost **1,500**
(3 × 500 for the autotune-ON condition; autotune-OFF is already the sweep).

⚠️ **Bounding note that limits the damage either way:** autotune variance can only *inflate*
measured instability, never manufacture false stability. **Any arm that comes out stable is
genuinely stable** regardless of how this resolves.

`cudagraph_mode=FULL_AND_PIECEWISE` is fine and expected — CUDA graphs became BI-compatible in
PR #27660, which is why `enforce_eager` stays off. But it belongs in the frozen list and must be
**asserted, not assumed**.

### R14.2 `VLLM_USE_FLASHINFER_SAMPLER=0` — register it, and it is a FINDING

My reading confirmed on both grounds, plus **a third I had missed and which is the strongest**:
**sampling is the last step before the token is chosen.** At `temperature=0` the argmax and its
**tie-breaking on equal logits** are implementation-specific, so an unguarded sampler converts any
upstream numerical near-tie *directly* into a different token. That makes it arguably **more
load-bearing than the attention backend**.

Goes in the prereg frozen list and the Method. Also record that **our image lacks `nvcc`**, so our
results are conditional on the PyTorch-native sampler path — someone with the toolkit gets
flashinfer by default and may get different determinism properties. One sentence, real
reproducibility content.

#### 🔬 THE FRAMING UPGRADE: H8 is now a four-gap claim, not two

`VLLM_BATCH_INVARIANT` coverage gaps, **all undocumented**:

1. `auto_awq.py` **has** a guard ("Marlin kernels are not batch invariant")
2. `auto_gptq.py` / `choose_mp_linear_kernel()` have **none**
3. the **sampler** path is unguarded (R14.2)
4. **kernel autotuning** is unguarded (R14.1)

> *"A beta reproducibility feature ships with at least four undocumented coverage gaps, and
> published quantization deltas inherit all of them."*

**Restructure the H8 write-up around the enumerated gap list.** This is materially stronger than
the two-gap version.

### R14.3 Assert the ENGINE'S REPORTED STATE, not the value we passed

All three declared-vs-enforced failures share one shape: **we knew what we passed and never read
back what the engine did.** So the rule is to assert on what the engine reports.

My four (dtype as loaded, attention backend as selected, prefix caching as reported, quantization
method chosen) confirmed, **plus seven**:

1. **`VLLM_BATCH_INVARIANT` actually took effect** — that kernels were *substituted*, not merely
   that the env var is set. *"The single most important assertion in the study: no substitution ⇒
   H8 dead and every delta untrustworthy."* Check the **patched op registry**, not the environment.
2. **Triton actually compiled a kernel this run.** A silent fallback is the catastrophic case. The
   verify job proved the toolchain once; assert it **per run**.
3. **The selected linear kernel class per arm.** Load-bearing, not hygiene:
   `quantization=compressed-tensors` is the *format*, while **Marlin vs Machete vs fallback** is
   the variable H8's claim is about — and `VLLM_DISABLED_KERNELS` is **unverifiable** unless the
   chosen kernel can be observed. My item 4 was necessary but insufficient.
4. **Model revision SHA as LOADED**, not as requested — the HF cache can serve a different
   revision than the pin.
5. **Engine-reported `max_num_seqs` and KV-cache blocks.** If `max_model_len` outruns the cache,
   vLLM **silently reduces concurrency** — corrupting the exact quantity A7 freezes.
6. **`temperature` and `seed` as they reached `SamplingParams`** — identical failure shape to
   `async_scheduling`.
7. **Single-BOS on every generation** (first token id), not once per model at G2. Cheap, and a
   known-live failure on Llama.

#### The structural fix, which matters more than any single assertion

My new test catches "field exists and is forwarded" — that prevents instance four of *this* shape
but not the next shape. The generative fix: make the frozen config **ONE MANIFEST** that is
(i) declared once, (ii) forwarded programmatically, (iii) **read back from the engine**, (iv)
written into every generation record, and (v) **asserted equal across all three**. §7.1 already
puts `batch_invariant` / `max_num_seqs` / `disabled_kernels` on the record — extend to the whole
frozen set.

Then **generate the paper's config table from the manifest** rather than hand-writing it.
Declared-vs-enforced-vs-reported drift becomes unshippable, and the Method cannot silently
misreport a value the way it would have for `async_scheduling`.

**One-time audit, now:** for every key in `models.yaml`, `quant.yaml`, `tasks.yaml`,
`scoring.yaml`, confirm a code path consumes it. **Any orphan is decoration.** Finds instance four
before it costs generations.

`sm_pilot`'s refusal to run until A7 freezes `max_num_seqs` is endorsed as **the template for
anything not yet determined**.

## R15 — The ordering finding: break the confound before building on it

Asked after job 2's smoke found GPTQ-4bit 65% order-dependent while bf16 was
order-invariant. Raw answers preserved verbatim in `notes/planner-answers/`.

⚠️ **Process note: I lost the top of this answer by piping `ask-planner` through
`tail -60`.** Planner answers are now saved in full to `notes/planner-answers/`
before any filtering. A ruling I cannot read is a ruling I will violate.

### R15.1 🚨 My confound check was PLAUSIBLE BUT NOT SOUND

I argued that because bf16 has the identical process structure and showed 0/20,
cross-instance variation could not explain GPTQ's 13/20. **That does not follow.** It
would only hold if bf16 and GPTQ were *equally sensitive* to whatever varies across
processes — and they demonstrably are not equally sensitive to anything. *"GPTQ's
Marlin path could be sensitive to workspace addresses or cuBLAS handle state in ways
bf16 simply isn't."*

There is an H8 finding either way, but **the mechanism decides the write-up**, and the
two candidates point in different directions:

| observation | claim | consequence |
|---|---|---|
| reordering items changes outputs | **batch-composition** — the batch-invariance thesis | licenses the reporting-standards conclusion about unreported item ordering |
| restarting the process changes outputs | process state — **not** a batch-invariance failure | different advice to the field |

**Do not hang the headline on the confounded cell.** Fill the 2×2, ~80 generations:

| | same process | new process |
|---|---|---|
| **same order** | have: 0/20 | **need** |
| **new order** | **NEED — 3 orderings run sequentially inside ONE engine** | have: 13/20 |

*"The 'new order, same process' cell is the decisive one: if GPTQ still diverges with
the engine held constant, ordering is established as the cause and the
batch-composition reading holds."*

### R15.2 Run the ordering test on the FULL ARM LADDER — AWQ is the prediction test

*"The single highest-value measurement available right now, because it's the one that
converts H8 from an observation into a mechanism."* ~300 generations at smoke scale.

AWQ **has** the batch-invariance guard; GPTQ does not. So:

- **AWQ order-invariant** → the source-level asymmetry *causes* the behavioural
  asymmetry. *"The strongest form H8 can take, because it's a confirmed prediction
  rather than a post-hoc explanation."*
- **AWQ also order-sensitive** → the guard is **insufficient**. A different finding,
  also publishable, plus a substantive vLLM bug report.

Include `bnb-NF4` and `GPTQ-8bit` as well: the full ladder makes it a **dose-response
over kernel paths**.

### R15.3 65% is a TEXT-level number. Report three levels.

My cascade reasoning is accepted, but *"the corollary matters more than the
argument"*: report divergence at **text**, **extracted answer**, and **correctness**.

> *"65% text / 3% correctness and 65% text / 20% correctness are different papers, and
> you already have the generations to compute both."*

Free, and it is what Q2 actually hinges on.

### R15.4 R8's sweep criterion is DEAD — reorient the sweep to the ordering axis

The repeat-based sweep *"measured an axis with no variance"* — it printed "stable" for
an arm that is 65% unstable, and it will keep printing "stable" at every level.

**Replace: for each level in {8, 32, 64}, measure ORDER-induced divergence across 3
orderings.** That is the decision-relevant comparison, and it may hand us **64** —
which, given the measured throughput, is a 4× win on everything downstream.

### R15.5 Q1 — freeze on the ORDERING axis; (a)'s logic accepted but not by default

My option (a) is right in substance: ordering is fixed in our design, so our deltas are
reproducible and order-sensitivity is a *result*, not an obstacle. Option (b) *"would
halt the study over a standard we don't actually need — order-invariance is required
for third parties to reproduce our strings, not for our comparisons to be valid.
Different standards; don't conflate them."*

**But freezing 8 must be EARNED, not defaulted to.** There is currently no evidence
that 8 beats 64 on the axis that actually varies. If order-sensitivity is flat across
levels, 8 buys nothing and costs 4× throughput.

Two hard requirements when freezing under order-sensitivity:
1. **Item order identical ACROSS ARMS**, so ordering contributes zero *differential*
   noise between bf16 and GPTQ — precisely §4.6's differential-noise concern.
2. **The order must be published** as part of the artifact.

### R15.6 Q2 — §9 binds to the REPETITION floor. Three numbers, three names.

*"You were right not to choose this after seeing data."*

1. **Repetition floor** — identical config, identical order, N repeats. Measured **0**.
   **§9's pre-commitment binds to this**, because §9 protects against claiming effects
   that are pipeline noise *under the conditions we actually ran*, and we hold ordering
   fixed. Not a dodge **provided ordering is common across arms**.
2. **Reordering sensitivity** — the H8 headline. Not a floor on our deltas; a
   **reproducibility bound for third parties**. Report at all three levels per arm.
3. **Pipeline floor** (§4.5) — bf16 end-to-end double run, unchanged.

The sentence for the paper:

> *Our deltas are reproducible under our fixed, published item ordering; our generated
> strings are not reproducible under reordering for GPTQ; therefore any quantization
> study that does not fix and report item ordering carries an uncontrolled term.*

*"That last clause is the sharpest thing this paper will say. Nobody reports item
ordering."* It joins the §0.1(b) reporting-standards residue with a harder edge than
relative-vs-absolute units — and unlike that one, it is ours, measured, and
mechanistically explained.

⚠️ **Escalation trigger:** if correctness-level divergence exceeds ~10 pp, return to
the planner before the freeze — that would pressure reading (1) as sufficient.

### R15.7 Q3 — the Marlin probe must itself be an ORDERING test, and PAIRED

Confirmed: a repeat-based fallback probe *"would print 'stable' for the same reason the
sweep did and prove nothing."*

Design it **paired**: GPTQ-4bit, the same 3 orderings, with and without
`VLLM_DISABLED_KERNELS=MarlinLinearKernel,MacheteLinearKernel`. Order-divergence
dropping to 0 with Marlin disabled *"pins the mechanism to a NAMED kernel — H8's
mechanism nailed down rather than inferred."*

**And assert the selected linear kernel class in BOTH conditions.** Otherwise "disabled
kernels" may silently have changed nothing, or the fallback may be a kernel we did not
expect, *"and the probe proves the opposite of what you'd conclude."*

### R15.8 Q4 — the 4× is CONSISTENT with Kurtic, and it is a regime finding

**Pre-empt the objection**, because a reviewer who knows our own replication target will
raise it. Appendix A/B report W4A16 speedup ~1.08× and sometimes negative — but that is
Kurtic's **max-throughput continuous batching** (large batch, compute-bound decode). At
**batch=1** they measured INT4 winning **2.39×**. At `max_num_seqs=8` we are in the
small-batch, bandwidth-bound regime, so ~4× is consistent with their table, not against
it. **Report ours as a small-batch number with `max_num_seqs` stated in the same
sentence.**

It matters beyond scheduling, three ways:
1. **BI overhead is not one number — it is arm-dependent**, because AWQ falls off Marlin
   under BI while GPTQ does not. The BI-on/BI-off measurement is still owed (A6's other
   deliverable; the 1.2–2× figure is Ampere-only). Report per arm.
2. Arms running at different speeds experience different scheduler behaviour — *which is
   the ordering-sensitivity mechanism*. Worth a sentence connecting them.
3. A genuine actionable finding: **at the low `max_num_seqs` determinism requires, 4-bit
   is substantially faster. Determinism and 4-bit speedup are complementary, not
   opposed.** Reframes Appendix B item 6 as regime-dependent rather than wrong.

**No sequencing change. Keep `max_run: 172800`.** Job 2's ~23,000 generations blend to
~4–6 h. The reference block now looks like **~20 GPU-hours rather than §2.6's 22–35**,
because 4-bit is fast in this regime. *"The schedule is ahead of plan, which is the
first time I've been able to say that."*

## R16 — Job 2's result: freeze 64, skip the missing cell, H8 becomes co-lead

Raw answer: `notes/planner-answers/R16-job2-frontier-and-freeze.txt`.

### R16.1 Escalation read confirmed — but compute permissive correctness, don't infer it

5.4% strict-correctness against a >10 pp trigger does not fire, and §9 stays bound to the
repetition floor (0.000%).

⚠️ **Caveat I had missed:** my table had no *permissive*-correctness column, and §2.2 makes
the headline contrast rung 1 **vs rung 2** — so permissive correctness matters as much as
strict. Computed rather than inferred: **worst is 7.0% (awq-4bit @ 32)**, under the trigger.
The planner's scaled estimate of ~7–8% was right, but *"compute it rather than infer it —
it's the only number that could still fire."*

### R16.2 Freeze `max_num_seqs = 64`

GPTQ-4's correctness difference is noise (4.2% vs 4.8% = 21 vs 24 items of 500, ~0.6 SE);
NF4's is not (4.8% → 0.2%, a 23-item swing). Throughput applies to ~110k remaining
generations.

Two conditions, both now recorded in `configs/models.yaml`:
- **Text divergence genuinely IS worse at 64** (33.4 → 39.6%, ~3 SE, *not* noise). Report
  honestly; it does not bind the decision because our claims are about scores rather than
  strings, *"but don't let it look like you missed it."*
- **Report the full frontier at all three metric levels**, not just the frozen value —
  *"the frontier is the contribution; it tells other people what they can get away with."*

NF4 improving with batch size is backwards from #51187. Offer **as a hypothesis, not a
finding**: larger reductions may be more robust to permutation if the divergence comes from
a data-dependent reduction whose result concentrates.

### R16.3 Skip the missing `gptq-8bit` Marlin cell — the framing is worth more

Third confirmation of a mechanism already pinned by a clean 33.4→0.0 and 43.6→0.0, against
a rebuild plus a queue cycle on a wall-clock-bound schedule. *"Not close."*

Report instead: *"for W8A16 there is no non-Marlin path available in a standard vLLM image
without additional CUDA tooling"*, with the kernel named and the compile failure quoted —
**plus the three-instance pattern** (gcc/Triton, nvcc/flashinfer, nvrtc/humming):
reproducibility features that require runtime compilation which standard images lack.
*"That is on-thesis, not an aside."*

### R16.4 H8 is well past sufficient. Do NOT chase the NF4 mechanism.

*"A second distinct mechanism exists, is Marlin-independent, and behaves oppositely"* is an
honest, complete existence claim. Chasing it costs unbounded source-reading against the
Sep 2 freeze, on the arm with the **least external validity anyway** — vLLM's own RFC
#39583 proposes moving bitsandbytes out-of-tree with *"performance is not great."* Cite the
RFC, state two candidate explanations as explicitly unresolved, move on.

### R16.5 🔬 H8 BECOMES CO-LEAD WITH H1 — pending one number

Order-induced correctness divergence of 4–5% sits **above many of the published quantization
deltas this paper audits** (Kurtic's corrected 405B figure is 0.7 points; RedHat's GSM8K
w4a16 delta is 0.1). *"A nuisance parameter that nobody reports moving the measured quantity
more than the effect being measured IS the paper's thesis — delivered by H8 rather than by
the decomposition."*

**But the precise claim depends on the SIGNED ACCURACY DELTA between orderings, per arm,
which I had not reported.** Flip rate and accuracy shift are different quantities:

| outcome | claim |
|---|---|
| accuracy itself moves by points | *"reordering alone moves reported accuracy more than 4-bit does."* Devastating. |
| items churn, aggregate stable | *"item-level churn exceeds published deltas even though aggregates look fine."* Still strong, and plugs into Dutta et al.'s flips / "Accuracy is Not All You Need", which §7 already requires engaging. |

**Both publishable; they are different papers.** Free to compute from generations already
held. **Do not reorder the narrative until the signed delta lands.**

Then the intro's spine becomes *the apparatus moves the number more than the bit-width
does*, instantiated twice — ordering and truncation. Nothing is dropped; the narrative
reorders.

---

## R17 — The signed delta landed. Neither fork; build on the effect's CHARACTER, not its size

Raw answer: `notes/planner-answers/R17-signed-delta.txt`. Numbers:
`artifacts/determinism_reanalysis.json`, RUNLOG under `determinism-reanalyze`.

### R17.1 🚨 My comparative claim was the paper's own diagnosed sin. Withdrawn.

I proposed leading with *"1.6 pp exceeds Kurtic's 0.7 and is 16× RedHat's 0.1."* That is a
**cross-model, cross-task, cross-protocol** comparison — Qwen-3B on a 500-item mixed probe
against Llama-405B and against Llama-8B/GSM8K — and **§0.1(b) is the claim that such deltas
are mutually incomparable**. *"Build the headline on that comparison and a reviewer turns the
paper's own thesis against it."*

The internally-valid version needs **our own** 4-bit delta on the same models and probe,
which arrives with the reference runs. **Do not write the comparative sentence yet.**

### R17.2 The claim to make instead — three properties, no magnitude comparison

> Reordering the evaluation set — a parameter no published protocol reports — shifts 4-bit
> accuracy by up to 1.6 points while leaving bf16 unchanged in all twelve configurations
> tested. The shift is **deterministic** given an ordering rather than sampling noise, so it
> cannot be averaged away; it is **bidirectional**, so no ordering is correct and it cannot
> be calibrated away; and it is **attributable by intervention** to the Marlin kernel path,
> which vLLM's batch-invariant mode does not cover.

Stronger than either fork because it is about the effect's *character*. One sentence gets
added after the reference runs: *the ordering spread is X% of the bit-width effect we measure
on the same items.*

- Wording: **"up to 1.6 points"**, never "points" plural.
- The churn observation belongs in the **same paragraph**: churn at 3–13× the aggregate means
  **the aggregate understates the instability** — the honest direction, and the link to the
  flips literature §7 requires engaging.

### R17.3 The range is the wrong estimand at ANY n. Report the distribution.

My statistical worry was right, but the fix is not more samples of a bad estimator: range is
unstable and **grows with sample size**. So report mean, SD, min, max across orderings, with
**SD as the inferential quantity and range as descriptive**. *"Accuracy has SD σ across random
evaluation-set orderings"* is defensible as a population claim; *"the spread is 1.6 pp"* never
will be.

**Extra-orderings job, scoped:** 4 arms (bf16, gptq-4bit, awq-4bit, bnb-nf4), `max_num_seqs=64`
**only**, 500 items, **7 additional orderings → 10 total**. 14,000 generations, ~35 min at the
measured 24,800 gen/h.

- ⚠️ **Keep 500 items; do not trade items for orderings.** Spread is *paired* across orderings
  so sampling variance largely cancels — but resolution does not. At 250 items one item is
  0.4 pp and the whole effect is 8 items.
- **bf16 gets the same 10 orderings.** Exactly-zero across ten is a much better control than
  across three.

### R17.4 Register it — as a confirmatory extension, explicitly NOT pre-data

Before Sep 2, register: estimator (SD primary, range descriptive), 10 orderings, seeds, arms,
`max_num_seqs=64`, and **rung 2 primary with rung 1 alongside** (per §2.2's headline contrast).

🚨 **Do not file it as pre-data — the 3-ordering result is already seen.** Register as a
*pre-specified confirmatory extension of an exploratory finding*, dated, with the exploratory
numbers stated inline. *"Discovery followed by pre-registered confirmation is normal practice;
dressing it as pre-data when the git history shows otherwise is the version that costs you.
The distinction reads as integrity."*

**Sep 2 freeze holds.**

### R17.5 H8 IS CO-LEAD. Spine confirmed, magnitude claim provisional

**Spine:** *the measurement apparatus moves the reported number more than the model change
does*, instantiated by **two independent mechanisms** — ordering (H8) and truncation (H1).
H8 goes **first** because it is complete, clean and mechanistically pinned; H1 follows because
it is the larger effect and carries the live 2606.25519-vs-2504.04823 contradiction.
*"Not two leads — one thesis, two instantiations."* The comparative half stays **provisional**
until reference runs land: write H8, do not lock the intro's magnitude claim.

### R17.6 Sequencing — all three, in this order; they do not compete

1. **Submit job 3 NOW.** It closes G3, Anchor A, the `TRUNC_STOPSEQ` trigger and the
   annotation pool — three of which feed Sep 2. *"Don't idle the slot behind a nice-to-have."*
2. **Write the H8 section while it runs** (no GPU). ⚠️ **Source every statistic from a table
   file rather than inlining numbers**, so the 10-ordering results update mechanically.
3. **Extra-orderings job after job 3** — off the Sep 2 critical path, since the prereg needs
   the *plan*, not the result.

Noted: 24,800 gen/h makes the ~20 GPU-hour reference estimate conservative — the second time
the schedule has moved in our favour.

---

## R18 — Job 3 scored: BBH was zeroed by the apparatus, and G2 was never in the way

Raw answer: `notes/planner-answers/R18-pilot-scored.txt`. Scores:
`artifacts/scores/scores_pilot-v1.json`.

### R18.1 🚨 Two things I got wrong, and lm-eval had already fixed the problem

I reported BBH's stop sequences as a conflict between G2's parity gate and getting any
data. Both halves of that were wrong.

1. **G2 tests RENDERED PROMPTS, not `generation_kwargs`.** `until` is a generation
   parameter, so changing it does not touch the gate. *"You're treating the gate as
   broader than it is."*
2. **The fix already exists upstream.** lm-eval's own
   `_cot_zeroshot_template_yaml` (v3.0) uses `until: ["</s>", "Q:", "<|im_end|>"]` —
   colon-terminated `Q:`, no `\n\n`, and a chat EOS — while `_cot_fewshot` (v4.0) is a
   raw-continuation config that predates chat templating. Verified in
   `vendor/lm-eval-configs/bbh/`; both files are vendored so the difference is auditable.

**Adopted for the few-shot arm.** The deviation is documented as *inheriting an upstream
fix*, not authoring one. Dropping BBH was **not available**: it is 6 of 8 task-conditions
and carries the format gradient that R1.3 made §7's power backbone.

- 🔬 **And it is the paper's most extreme instance of the thesis** (not optional, per the
  planner): `bbh_cot_fewshot` silently scores ~0 under chat templating on 5/6 subtasks for
  every model **including bf16 Llama-3.1-8B**. *The apparatus zeroes the benchmark; the
  model is irrelevant.* Goes in the **introduction as the motivating extreme case**, not
  as a fourth hypothesis — it is not differential across precision, so it says nothing
  about quantization deltas directly.
- **File it upstream with lm-evaluation-harness** (reproducible repro, five subtasks,
  bf16). ⚠️ Deferred to the user: this is a public action inside the ARR anonymity window.
- ⚠️ **H1's budget axis on BBH survives ONLY after this fix** — the stop sequence bound
  long before `max_gen_toks`, so the sweep would have measured nothing. State that in the
  H1 section: it is a **precondition, not a cleanup**.
- **Re-run BBH's pilot cells before Sep 2.** R18.4 depends on it.

### R18.2 GSM8K's hash arm: keep as demonstration, EXCLUDE as an interaction level

A 1.0% strict score is a **floor, and floored metrics have no variance for an interaction
to appear in**. So the arm has two roles and only one is admissible:

- **Demonstration:** keep. qwen 1.0% strict vs 62.0% permissive is the marker-naturalness
  gradient's extreme end, measured. Headline-grade.
- **Level in the precision × scoring interaction:** exclude. `gsm8k_cot` carries the
  primary interaction — as already ruled, now *empirically* justified rather than argued.

**Two things I failed to extract:**
1. **`gsm8k_cot`'s strict-vs-permissive gap** — the gradient's *other* end and the direct
   test of the monotonicity prediction. I reported only the hash arm.
2. 🔬 **A RANK INVERSION.** Strict orders phi (30) > llama (17) > qwen (1); permissive
   orders llama (75) > phi (65) > qwen (62). **The scorer changes the model ranking.**
   That is FormatSpread/xFinder instantiated for quantization eval, it is a memorable
   table, and it is *stronger than the gap magnitude alone*.

### R18.3 AWQ: existence claim now, magnitude later. And job 1b is back on the schedule

State: *AWQ quantization is not bitwise reproducible across runs of the same recipe; the
resulting weight difference moves strict accuracy by ~1 pp and permissive by 0 pp on 300
items.* The **same-hash control cell stays in the text** — it is the reason the 1 pp is
attributable to weights rather than to the pipeline.

- **No magnitude on 3/300.** Add `awq-4bit-nooffload` as an extra cell in the Qwen
  reference run (500 items, one arm, trivial) and quote the magnitude from there.
- 🚨 **Job 1b joins the schedule** (~30 min, quant image). Now that this is main text,
  *"AWQ is inherently nondeterministic"* and *"`offload_device` changes the weights"* are
  **different claims and the ambiguity cannot be published.**
- Placement: main text, in the H8 section — *the artifact-production step is also not
  reproducible.*

### R18.4 Re-score before interpreting anything. Two stratum rulings to register by Sep 2

**Re-score after R18.1.** `TRUNC_STOPSEQ` 26–45%, `ABSENT` 40–76% and `COMPLIANT` 2–6%
are all dominated by the BBH defect — BBH is 6 of 8 task-conditions and 5 of its subtasks
produced empty generations. **Read nothing into the differential test yet.**

⚠️ **Do not interpret the qwen inversion** (bf16 truncating more than AWQ, 135 vs 84 of
300): almost certainly the same artifact, since bf16 writes more prose and hits `\n\n`
sooner.

**Register before Sep 2:**
1. **Define the always-compliant stratum over the three CONTRASTED arms** (bf16, GPTQ-4,
   AWQ-4), not all five. The stratum must be pre-treatment-defined with respect to the
   arms being compared; demanding commitment from arms we do not contrast shrinks it for
   no identification benefit. **Register now, before the re-scored numbers exist**, so it
   is principled rather than convenient.
2. **The 50% threshold STAYS.** Cells failing it report **Manski bounds and the
   compliance-transition matrix only, no point estimate** — both already pre-registered
   in §4.0 parts 2–3. That is the fallback working as designed, not a patch.

**Note for the paper:** if `ABSENT` stays high after the BBH fix, that is a headline in
its own right — *strict-match scoring discards the majority of generations.*
