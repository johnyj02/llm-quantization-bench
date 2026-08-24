# Queries for the planner agent

**From:** executor, 2026-08-23 (end of day 0)
**Re:** `EXECUTOR-TASKS.md` §0 — *"If you find a conflict not listed in §0.2, that is a
planning bug: stop and report it."*

Nothing here is a complaint about the plan. The plan has been unusually good to execute
against; everything below was found *because* it told me to verify rather than trust. Four
items (Q1–Q4) need a ruling **before the Sep 2 pre-registration freeze**, because
`analysis/preregistration.md` is worth only its timestamp and I do not want to be amending
§4.0 after generations exist. Q5–Q9 need registration rather than debate.

Each carries what I did, my reasoning, and what I need from you. Where I acted, I acted
under this document's execution authority and said so; where the question is scientific,
`EXECUTOR-PLAN.md` has authority and I have **not** acted.

---

## Q1 🔴 — `####` and "few-shot CoT" are mutually exclusive. This changes the expected headline effect size.

**The conflict, not in §0.2.**
- `EXECUTOR-TASKS.md` §2.2 locks the reference prompt format as *"few-shot CoT (reference,
  harness-verbatim)"*.
- `EXECUTOR-TASKS.md` §3.4 and `EXECUTOR-PLAN.md` §1.1/§2.2 all state the GSM8K strict
  regex is `#### (\-?[0-9\.\,]+)`.

Verified against `lm-eval` v0.4.12 (vendored in `vendor/lm-eval-configs/`): `####` belongs
to task **`gsm8k`**, which is **5-shot and non-CoT**. The harness's few-shot-CoT task is
**`gsm8k_cot`**, whose `strict-match` regex is **`The answer is (\-?[0-9\.\,]+).`** — and
whose 8 hardcoded exemplars all end *"The answer is N."*

So (a) reference = few-shot CoT, (b) strict regex = `####`, and (c) harness-verbatim cannot
all hold. Pick two.

**Why this is the most important item here.** `EXECUTOR-PLAN.md` §1.1 grounds the artifact's
existence in `####` collapsing chat-model scores toward zero — *"instruction-tuned models
essentially never emit it."* That is true of `gsm8k`. Under `gsm8k_cot` the model is
explicitly taught the exact string rung 1 matches, so **compliance should be high and the
rung-1-vs-rung-2 gap — the paper's stated headline contrast — should be small or absent on
GSM8K.** Conversely `gsm8k` gives the large gap and Anchor C, but its prompt never asks for a
chain of thought, which weakens H1's mechanism on our largest task.

**What the user decided (D6, approved 2026-08-23):** run **both**. `gsm8k_cot` primary
(H1 needs CoT to truncate), `gsm8k` pre-registered secondary (carries `####`, makes Anchor C
reproducible), and the comparison between them is itself a pre-registered result. Cost:
+19,785 generations, ~20% over the §2.2 locked matrix.

**What I need from you — three things, and the third is the real question:**
1. Confirm `gsm8k_cot` as primary, or override.
2. Confirm the framing consequence: **`EXECUTOR-PLAN.md` §1.1's "strict collapses toward
   zero" argument does not survive as written for the primary GSM8K arm.** It survives for
   `gsm8k` and for BBH's case-sensitive lookbehind. Does the intro need rewriting around
   BBH + `gsm8k`-secondary as the artifact demonstrations, with `gsm8k_cot` as the honest
   near-null?
3. **Does H7's powered-ness audit still work?** If the primary GSM8K effect is near-null,
   our minimum-detectable-effect claim and the "protocol variance exceeds bit-width
   variance" thesis (§7's precise claim) rest more heavily on BBH. Is BBH×6 at 150 items
   each enough to carry that, or does the item count need revisiting *before* Sep 2?

I have pre-registered both variants with roles assigned and a pre-committed expectation
(large gap on `gsm8k`, small on `gsm8k_cot`) so that either outcome is a result rather than a
surprise. **I have not touched the intro framing** — that is yours.

---

## Q2 🔴 — §4.0's always-compliant stratum: "produced a committed answer" needed an operational definition, and the obvious one is biased toward us

§4.0 defines the stratum as *items where **every** arm produced a committed answer*. Building
the scorers forced the question of what "committed" means mechanically, and the naive answer
is wrong in our favour.

**The problem.** Rung 2's ordered rules include the harness's own `flexible-extract`
fallback — last number in string, `group_select: -1`. On a generation truncated mid-reasoning
(*"Ann has 3 apples and buys"*) that yields `"3"`. Under a `value is not None` reading, that
item is `RECOVERABLE`, counts as "produced a committed answer", and is **admitted to the
always-compliant stratum**. Truncated generations would then populate the very stratum whose
purpose is to be free of post-treatment selection — and since truncation is precisely what
quantization is hypothesised to cause, **the leak runs in the direction of our own
hypothesis**, which is the exact failure mode §4.0 exists to prevent.

Second instance, same root: **`arc_challenge_chat`'s rung-1 filter has no regex.** It is
`remove_whitespace` + `take_first`, so it succeeds on *any* non-empty text. Taken literally
every ARC-chat generation is `COMPLIANT`, the compliance column is vacuous, and the H4
decomposition is meaningless **on the task that is our structural control for H6**.

**What I did** (and pre-registered in `analysis/preregistration.md` §7, before any
generations exist):
- Split `found` (a candidate string was produced → used for **scoring**) from `committed`
  (positive evidence the model committed → used for the **compliance label** and therefore
  stratum membership).
- `COMMITMENT_RULES = {boxed, commitment_phrase, hash_marker}`; `FALLBACK_RULES =
  {last_number, last_option_letter}` still score but do not confer compliance.
- ARC-chat requires *canonical shape* for the label — the task ships
  `gen_prefix: "The best answer is"`, so a canonical continuation is the option letter
  alone. Rung-1 **scoring** stays harness-verbatim.
- Both flags plus the rule that fired are on every score record, so the choice is auditable
  rather than buried.

**What I need from you:** confirm this is what §4.0 meant by "committed". If you intended the
looser reading, say so now — it materially changes stratum size, and stratum size is
specified as a first-class reported number. I am confident the strict reading is right, but
§4.0 is explicitly the thing that "cannot be patched later," so I want it ruled on rather
than assumed.

---

## Q3 🔴 — `TRUNCATED = finish_reason == "length"` undercounts truncation, and the miss is concentrated in BBH

`EXECUTOR-PLAN.md` §2.3 and `EXECUTOR-TASKS.md` §7.2 both define `TRUNCATED` as
`finish_reason == "length"`. Verified `until` lists say that is incomplete:

| Task | `until` |
|---|---|
| `bbh_cot_fewshot_*` | `["</s>", "Q", "\n\n"]` — **a bare capital `Q`**, and a blank line |
| `arc_challenge_chat` | `["\n\n", "."]` — **a single period**, with `max_gen_toks: 100` |
| `gsm8k` / `gsm8k_cot` | `["Question:" / "Q:", "</s>", "<\|im_end\|>"]` |

A BBH chain of thought containing any capital `Q` or any blank line is cut **mid-reasoning**
and reported as `finish_reason == "stop"`, not `"length"`. An ARC-chat generation containing
a period before its answer, under a 100-token cap, likewise. This is a second truncation
mechanism on **the same causal pathway as H1**, and the specified definition misses it
entirely.

**What I did** (pre-registered, `analysis/preregistration.md` §4): three mutually exclusive
termination classes — `TRUNC_BUDGET` (`finish_reason == "length"`), `TRUNC_STOPSEQ`
(`"stop"` + matched harness `until` + no committed answer), `COMPLETE` (EOS, or a stop
sequence *with* an answer). **H1's primary remains `TRUNC_BUDGET`**, since that is what the
budget sweep manipulates; `TRUNC_STOPSEQ` is reported alongside and doubles as an internal
control because it should *not* respond to the budget sweep.

**What I need from you:** confirm the scheme, and rule on one consequence I can see but have
not acted on — **if `TRUNC_STOPSEQ` turns out to be common on BBH, is it a finding we lead
with or a limitation we report?** It is arguably a cleaner, more legible artifact than
budget truncation (a bare `Q` silently truncating chain-of-thought in a standard benchmark
is a striking result), but it is *not* the H1 the plan is built around and it does not
respond to the sweep. I do not want to make that emphasis call myself.

---

## Q4 🟠 — H6's mode contrast is confounded with a shot-count contrast

`EXECUTOR-TASKS.md` §2.2 places ARC-Challenge in the reference condition, run under **both**
scoring modes on the same 500 items, and states *"the gap between the modes **is** the
extraction cost, measured directly."* H6 is marked **never cut**.

Verified: **`arc_challenge_chat` is `num_fewshot: 0` by design** (it is a `llama`-tagged task
carrying `gen_prefix: "The best answer is"`), while `arc_challenge` is
`output_type: multiple_choice` scored by log-likelihood over answer *texts*, where shot count
does not enter the same way.

So as specified, the mode gap is **mode + shot count**, not mode alone — which undercuts the
"measured directly" claim for our load-bearing negative control.

**What I have NOT done:** anything. This is a design question, and the options trade off
against different things:
- **(a)** Accept it, and report the gap as mode+format rather than pure extraction cost.
  Cheapest, weakest claim.
- **(b)** Add a few-shot `arc_challenge_chat` variant so the mode contrast is shot-matched.
  ~500 × 15 = 7,500 extra generations, and it stops being harness-verbatim.
- **(c)** Run `arc_challenge` log-likelihood at 0-shot to match. Cheap, but 0-shot
  log-likelihood on ARC is a weaker baseline and is not what published numbers use.

I lean **(b)** — H6 is never-cut and load-bearing, the cost is small next to the ~20% we just
added, and a confounded negative control is worse than no negative control. But it is a scope
change to a locked matrix, so it is your call.

---

## Q5 🟠 — Substrate is L40S/SM89, not A100-80G/SM80. Three consequences for the science.

Already approved by the user; recorded as an amendment to **O4** in `notes/compute.md` §2.
O4's *conclusion* is reinforced — SageMaker training-job quota is **0** for every usable GPU
in the user's account, so SMInterface was never viable. O4's *premise* that AWS quota takes
days-to-weeks is false here: EC2 On-Demand G quota is **768 vCPUs, already granted**, so we
launch into the user's existing account with zero signup latency. That removed one of the
three named deadline risks.

Substrate: **EC2 `g6e.2xlarge`, 1 × L40S 48 GB, SM89 (Ada), 64 GiB host RAM, $2.242/hr.**
AWS has no single-A100-80G instance at all — A100-80G exists only as `p4de.24xlarge` (8 GPUs,
~$40/hr, spot quota 0). §3.1 already names L40S as an approved substrate.

**Consequences I need you to register, and one to rule on:**
1. **The "1.2–2× BI wall-clock overhead" figure is Ampere-specific.** SM89 is unmeasured.
   `A6` measures it directly; I am treating *"does `VLLM_BATCH_INVARIANT` work at all on
   SM89"* as an `A7` **finding**, not a precondition. Register.
2. **H8's claim scope narrows to L40S/SM89.** §4.6 says reproducibility holds only within
   identical hardware. "Which quantized serving configurations are bitwise reproducible" is
   now answerable *for Ada*, not in general. **Does that weaken H8 enough to change its
   billing?** §2.1 marks H8 "may become a headline." A single-architecture determinism result
   is still novel and unreported, but a reviewer can ask why one GPU. Your call on framing.
3. **Bandwidth is ~864 GB/s vs A100-80G's ~2039.** At `max_num_seqs=8` we are decode-bound,
   so §2.6's *"compute fits with an order of magnitude of headroom"* verdict rests on a
   premise that moved by ~2.4× on bandwidth and 1.88× on price. The user has said cost is not
   a constraint (D9), so this is a **schedule** question only — and §9 trigger 3 still fires
   if `A6` pushes the reference block past Sep 13. Register; no action needed yet.

---

## Q6 🟡 — Locked matrix amended: +19,785 generations

§2.2 is marked **LOCKED**. D6 adds the second GSM8K variant, ~20% over the stated 99k total.
User-approved. Flagging because you locked it and the §2.6 volume table is now stale.

## Q7 🟡 — BBH few-shot and zero-shot use different metric normalization in the harness, which would confound H5

`cot_fewshot` has `ignore_case` **commented out** and no `regexes_to_ignore`; `cot_zeroshot`
has `ignore_case: true` **and** `regexes_to_ignore: ["\\.$", ",", "\\\\", "\n", '"']`, and no
`get-answer` filter. **Few-shot BBH scoring is strictly harsher than zero-shot BBH scoring.**
Inheriting each variant's own metric config would put part of any H5 few-shot-vs-zero-shot
difference into the *scorer*.

**Acted, under execution authority:** our four rungs are applied **identically** across
prompt variants; rung 1 for BBH uses the `cot_fewshot` `get-answer` regex for both. Recorded
as a documented deviation from harness-verbatim scoring **for the zero-shot arm only**. G2 is
unaffected — it tests prompts, not metric configs. Flagging because it slightly weakens the
blanket "we use the standard harness" sentence, and you may want the paper to phrase that
more carefully than "verbatim".

## Q8 🟡 — `max_num_seqs` is now swept in `A7` rather than assumed

§3.3 fixes `max_num_seqs=8`, citing vLLM #51187 (44 co-resident sequences → 3 distinct
outputs in 5 repeats; 8 → 5/5 identical) — **measured on other hardware.** It is also our
largest cost driver, since it deliberately starves GPU concurrency.

`A7` already runs 3 identical repeats per format, so I have folded in a sweep over
`{8, 16, 32, 64}`, taking the highest value that is still bitwise-identical across repeats.
Pre-registered as frozen-before-`A13` and identical across all arms. Reporting the
**frontier** strictly dominates reporting one inherited value, given §3.3 already requires the
value to appear in the paper. Register; object if you disagree.

## Q9 🟡 — Two premises in §1's environment audit were stale

Minor, recorded for accuracy: free disk was **5.2 GB**, not 7.6 (now 24 GB after archiving a
13 GB HF cache to the user's dev desktop), and `huggingface-cli` **was** present (old
version). Neither changed a decision.

Also, one §3.5 constraint dissolved: **`tectonic` is installed**, so the official ACL template
compiles locally and MacTeX's 7 GB is irrelevant. `make -C paper g6` now checks all four G6
criteria mechanically. Overleaf stays the collaborative surface but **G6 no longer depends on
it** — §11.4 is no longer schedule-critical.

---

## What I am doing while this is outstanding

Nothing above blocks: prompt rendering + the **G2 byte-parity test**, ARC log-likelihood
scoring for **G3**, `quantize.py`, `judge.py`, the `qa` CLI, and `C1` related work. All
zero-compute.

Per the user's standing instruction (**D10**), **no GPU job, quantization, or billable
compute starts without their explicit confirmation** — enforced in code, not just documented.
So Q1–Q4 can be answered without any run being wasted.

**The one hard deadline on your answers: Sep 2**, the pre-registration freeze. Q1–Q4 all
touch text in `analysis/preregistration.md`, and that document's entire value is its
timestamp.
