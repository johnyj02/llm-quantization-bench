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
