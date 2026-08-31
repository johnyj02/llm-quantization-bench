# Handoff prompt — branch `fixing package bug.` → mainline

Copy everything below the line into the mainline session.

---

You are resuming as **EXECUTOR** on QuantArtifact. Since your last mainline context, **I did a
large amount of work on a separate branch** (session titled *"fixing package bug."*). All of it
is **committed to `main` in this same working tree** — 34 commits from `8b3c2ae` to `ecc624f`.
Your understanding of this project is now materially out of date, and several things you believe
are settled have been **overturned**.

**Do not start any job or edit any file until you have read the files listed in §0.** Several
changes reverse earlier decisions, and acting on stale understanding will waste a queue slot or
corrupt data.

## §0. READ THESE FIRST, in this order

| # | file | why |
|---|---|---|
| 1 | `notes/RUNLOG.md` | 33 jobs now. Read from the `determinism-orderings` entry to the end — that's the new material, and every entry carries what it taught us |
| 2 | `notes/planner-rulings.md` **R17–R23** | Seven new rulings. R19 **reverses** R18.1. R23 **corrects** R22. Raw text in `notes/planner-answers/R17..R23*.txt` |
| 3 | `notes/cross-invocation.md` | **NEW.** The biggest open problem. Read all of it |
| 4 | `analysis/preregistration.md` **§3b-bis, §9a, §9b, §9c, §9d, §9e, §9f, §9g** | Seven new registered sections, plus §1.1 amended and §3b marked superseded |
| 5 | `notes/protocol-survey.md` | **NEW.** What the cited work does and doesn't report, N of M *checked* |
| 6 | `README.md` §4 and **§4a** | Eight stages now, not four. §4a states two rules that have cost real runs |
| 7 | `CLAUDE.md` §3 | Stage list and two new hard rules |
| 8 | `notes/gates.md` | **G4 is PASSED** |

Then `git log --oneline 8b3c2ae..ecc624f` — the commit messages carry the reasoning, deliberately.

---

## §1. Things you currently believe that are NO LONGER TRUE

1. **"BBH's terminators are a bug in lm-evaluation-harness."** ❌ Withdrawn. The user challenged
   it and was right. `until` works exactly as specified; the strings encode a *formatting
   assumption* (blank lines occur only between exemplars) that chat templating invalidates. The
   value came from lm-eval but **generation read our own `configs/tasks.yaml`** — lm-eval was
   never in the generation path. **No upstream bug report is filed.** (R19.0)
2. **"Chat template is FIXED, not a factor" (prereg §3b item 1).** ❌ Superseded by **§3b-bis**:
   it is now a **two-level factor** (`chat` primary, `raw` a *main-text* robustness arm), and
   **the terminator set is part of each level's definition**. §3b is retained but banner-marked.
3. **"The prereg freezes on Sep 2."** ❌ The user ruled, and §9f registers, that the **binding
   line is the first `run_id=main-v1` generation**. Sep 2 is the schedule target. Until main-v1
   exists, §1–§9 may be edited, because no edit can have been influenced by confirmatory results.
4. **"H8's spread is up to 1.6 pp over 3 orderings."** ❌ Superseded by the 10-ordering run.
   **bf16 is exactly 0.000 across all ten orderings at both rungs**; worst quantized SD is
   **1.14 pp**. The two runs **must not be pooled** (§9a).
5. **"Range is the wrong estimand" was a statistical argument.** ✅ It is now an **empirical
   demonstration**: the range grew **1.6 → 3.8 pp** purely from adding orderings.
6. **"`offload_device` changed the AWQ weights."** ❌ Exonerated. **AWQ is not reproducible
   run-to-run at a bit-identical recipe** — job 1b proved it (0 of 939 tensor offsets differ,
   13/13 weight windows differ). (R18.3, §9d)
7. **"24,800 gen/h."** ❌ Measured while BBH emitted nothing. Real rates: **BBH 7,099**, gsm8k
   **5,250**, ARC effectively free — and **generation was only 25% of billable time**;
   **2.6 min/cell of overhead** dominated. Cell count is a first-class cost driver.
8. **"`max_num_seqs=64` is the only frozen serving value that matters."** ❌ Sampling was
   under-specified; see §3.4.

---

## §2. New scientific results

### 2.1 H8, re-measured at ten orderings (R17.3, prereg §9a)

| arm | strict SD | perm SD | strict range | perm range |
|---|---|---|---|---|
| **bf16** | **0.000** | **0.000** | 0.00 | 0.00 |
| gptq-4bit | 1.001 | 0.675 | 3.20 | 2.40 |
| awq-4bit | 0.801 | **1.143** | 2.60 | **3.80** |
| bnb-nf4 | 0.063 | 0.063 | 0.20 | 0.20 |

bf16 identical under all ten orderings at both rungs is **the strongest single number in H8**.
`bnb-nf4` has 4.4% *text* divergence but SD 0.063 pp — churn that never reaches the score.

### 2.2 The template factor returned BRANCH 1 (prereg §3b-bis)

18 cells, 18,000 generations. The interaction's **sign is preserved 6/6** across template
levels, so the self-infliction objection is **answered empirically**. Two caveats that are in
the paper: the **magnitude** half of the test is weak (interactions are ~3 pp against a 5 pp
tolerance) so **sign preservation is what carries it**; and accuracy **levels** move enormously
(qwen bf16 strict **18.4% → 42.7%**), which makes the paper's *"a level artifact is not a delta
artifact"* reframe a **measurement rather than an argument**.

### 2.3 AWQ is not bitwise reproducible (R18.3, §9d)

Same recipe hash, same seed, ~3 days apart: identical size, different ETag, **0 of 939 tensor
offsets differ** (so layout is identical and the differing bytes are genuinely weights),
**13/13 sampled 64 KB windows differ**. Accuracy consequence from `pilot-v2`: strict −0.33 pp,
permissive +0.67 pp, with a same-hash control cell reproducing bit-identically.
⚠️ A byte diff cannot distinguish "different weights" from "same weights, moved" — parse the
header first.

### 2.4 `pilot-v2` re-scored; the stratum falls back

BBH fixed adds 8–14 pp strict. **But the always-compliant stratum is below 50% in most cells** —
ARC 74–100%, BBH 10–100%, **`gsm8k` 1–27%**. Per §1.1 those cells go exploratory with Manski
bounds primary. That is the registered fallback firing, but the *extent* is unresolved and H4
may rest largely on bounds.

### 2.5 Protocol survey (R19.3, `notes/protocol-survey.md`)

- **Kurtic et al.** — chat template **not reported**, extraction mode **not reported**, no eval
  code. Also ran MMLU/ARC-C as *"text-generation tasks rather than log-likelihood"* → **H6 cite**.
- **Red Hat** — fully reported. Their card says results were revised *"after Meta modified the
  chat template"* → **third-party evidence the template moves published numbers**. Their w4a16
  GSM8K is 82.8 → 82.9 (**+0.1**), verified at source.
- **Anchor C (2601.14277)** is **llama.cpp GGUF on CPU**, not vLLM/GPTQ. Only its *extraction
  gap* is reproducible; its quantized deltas are not comparable. Its own headline ranking rests
  on the lenient filter.
- **Pape et al. (2605.19537)** is the **nearest prior work** and is now differentiated in Related
  Work. They compare across engines at **batch size one**, *"to prevent batching-induced
  instability"* — the effect we make the object of study. Their 35,000-paper survey (36% of
  no-code papers name their backend) means **no field-level claim rests on our M=4**.

---

## §3. 🚨 THE BIGGEST OPEN PROBLEM — read `notes/cross-invocation.md`

Found by a **pre-flight check the user asked for** before committing to `main-v1`. That check
also verified all 9 checkpoint digests and 100% prompt identity, so it was worth doing.

**`pilot-v1` and `pilot-v2` ran the same items in the same order with only BBH's `until`
differing. `gsm8k` and ARC should have been bit-identical. They were not.**

| model | gsm8k divergence | note |
|---|---|---|
| qwen2.5-3b | **0/100 in all 7 cells** | |
| llama-3.1-8b | **41–45/100 in all 5 arms** | **includes bf16** |
| phi-3.5-mini | bf16 45, nf4 55, awq/gptq **0** | |

**It affects `bf16`, so it is not H8's mechanism.** Every recorded config field on a differing
pair is byte-identical.

### 3.1 Why it mattered

Worst signed accuracy delta **4.44 pp**; SD of deltas 1.20 pp. And it does **not cancel across
arms**: llama bf16 moved **+1.11** while gptq-4bit moved **−1.67**, swinging their delta
**2.78 pp from re-running alone**.

### 3.2 Why it is survivable — the 1/√n check

The effect is **churn, not bias**: 1.7–5.6% of items flip, near-symmetrically (44 flips, |net|
20). Resampling confirms `SD(Δ) = 100·√(churn/n)` to within the finite-population correction.
So the pilot's n=180 figure shrinks at the reference condition's n=2,719:

| | per-arm SD | class-3 interaction floor |
|---|---|---|
| pilot n=180 | 1.39 pp | 2.78 pp |
| reference n=2,719 | **0.36 pp** | 0.51 pp |
| + K=2 | **0.25 pp** | **0.36 pp** |

**H3 is safe.** Baking the pilot number into §9 would have crippled it with a figure that does
not apply.

### 3.3 THREE noise classes (§9f) — this reverses part of R22

1. **none** — re-scoring *one fixed generation set*: rung monotonicity, compliance labels, §8's
   trace scan, per-item rung agreement.
2. **partially cancelling** — a **within-arm** rung difference (the gap): max 1.67 pp vs 4.44
   for raw accuracy.
3. **compounding** — anything **contrasting arms**: between-arm deltas, H4's components, **and
   H3's interaction** (max 2.78 pp). R22 wrongly placed H3 in class 1; its *outer* difference is
   between arms so two invocation noises compound. **I corrected the planner and it agreed.**

§9f registers the **estimator, not a number**. §9g pins a §4.0 item neither of us had raised:
the always-compliant stratum is itself invocation-dependent, so it is defined as items committed
in **all invocations of all contrasted arms**.

### 3.4 What was ruled out, and what the sampling fix was

Four hypotheses **refuted with the numbers that killed them** — record them, they save work:

| hypothesis | killed by |
|---|---|
| different `num_gpu_blocks` per invocation | KV cache sizes **byte-identical** between jobs |
| `max_num_seqs` exceeding KV concurrency | stable at **12.36×**, differs at **10.70×** |
| base repo vs local checkpoint | explains phi, contradicted by llama and qwen |
| **temperature / top_p / top_k** (the user's hypothesis) | Qwen ships the **most aggressive** defaults and is the **only** perfectly reproducible model |

⚠️ **But the user's question exposed a real separate defect.** vLLM's `generation_config`
defaults to `"auto"`, which reads the model's own `generation_config.json` — Qwen ships
`repetition_penalty 1.05` and `top_k 20`, Phi ships nothing. `temperature=0` makes top_p/top_k
inoperative, but **`repetition_penalty` modifies the logits before the argmax**. Fixed at both
levels: all six sampling fields pinned and asserted, and `generation_config="vllm"` so the file
is never read. The `diagnose` job then showed the old config **was** neutral (qwen/bf16 40/40
identical), so **prior data is not corrupted** — but the pinning stays, because greedy-with-no-
penalties is a property the paper *reports*.

**The mechanism remains UNDIAGNOSED.** Report as observed, on this substrate, with the refuted
list. The planner ruled it **publishable unexplained** and **defensible to file upstream** with
vLLM (unlike the BBH framing) — not before Sep 2.

---

## §4. New and changed code

**New stages** (all registered in `scripts/submit_job.py`):

| file | purpose |
|---|---|
| `sm_template.py` | the template factor, both levels, 18 cells |
| `sm_reference.py` | **THE MAIN RUN**, `run_id=main-v1`, 15 cells × 4,238 items = 63,570 gens |
| `sm_diagnose.py` | one cheap job per specific open question |

**New scripts:** `scripts/score_pilot.py`, `scripts/score_probe.py` (runs in `.venv-harness`),
`scripts/score_template.py`, `scripts/make_h8_tables.py`, `scripts/project_budget.py`.

**🚨 The single most important code change — `decoding_fingerprint()` in
`src/quantartifact/generate.py`.** It hashes everything determining what a generation *is* and
`cell_tag` appends `dec<hash>`, so a config change **automatically** invalidates cached cells.
This is the root fix for a bug family hit **eight times**; `DECODING_VERSION` must be bumped
when anything affecting generation changes. ⚠️ `env_hash` is **not** this — it covers
`{gpu, cuda}` only, which is why it survived two config changes unchanged.

**Two rules now in README §4a and CLAUDE.md §3, because they cost real runs:**
1. A cached/resumed unit of work must be keyed on **everything** determining its output.
2. A comparison must hold **item COUNT** fixed — `run_cell` batches the whole list, and
   batch-sensitive arms diverge up to 71.6% under recomposition. My own diagnostic changed n
   from 100 to 40 and **measured H8 instead of the config change**.

**Tests: 265 → 340.** New: `test_sm_template.py`, `test_sm_reference.py`, `test_sm_quantize.py`,
plus classes in `test_sm_determinism.py` for the ordering mode, the decoding fingerprint, and
**`TestPreregAndCodeAgree`, which parses the ordering seeds out of the prereg and compares them
to the code** — the prereg is the one document where declared-vs-enforced drift is fatal.

---

## §5. WHERE THINGS STAND RIGHT NOW

**Ledger: 33 jobs · 15.11 h · $42.30.** Budget $370–440; all-in projects to ~$90. **The user
has explicitly said going over budget is acceptable**, so the 15%-of-remaining cap in R20.1 is
**lifted** (documented in `scripts/project_budget.py`). Wall clock, not money, is the constraint.

**RUNNING NOW:** `johnyj02-QA-v0-1-20260831-33b4267e-determinism` — the determinism **re-run**,
21 cells, ~24k gens, ~2.5 h. Check with `bash scripts/watch_job.sh`.

### Why four jobs are being re-run

The user asked whether to re-run rather than diagnose, and re-running is **better science**, not
just hygiene:

- job 2 and the 10-ordering run used the **old BBH terminators**, so ~2/5 of their probe items
  were inert and **H8's spreads are attenuated** — the paper currently carries a
  conservative-floor caveat that re-running removes;
- all four predate different subsets of the fixes;
- **it doubles as the missing cross-invocation test**, because they are new jobs at matched item
  counts, which the `diagnose` job could not do (both its invocations ran inside one job).

| # | job | ~cost | status |
|---|---|---|---|
| 1 | determinism sweep | $6.81 | **RUNNING** |
| 2 | 10-ordering re-run | $2.31 | queued behind |
| 3 | pilot re-run (annotation pool) | $2.79 | |
| 4 | template probe re-run | $5.05 | |
| 5 | **main-v1 + K=2 on 10 cells** | ~$33 | last — it is the freeze line |

⚠️ **Nothing is deleted.** Old runs keep their `run_id`s as provenance. **`pilot-v1` must be
retained** — it is the evidence for the BBH terminator finding the paper reports in Method.

**Expect H8's numbers to change.** The mechanism (Marlin, pinned by intervention) and the
control (bf16 exactly zero) should hold; magnitudes will likely rise. **Regenerate the tables
with `scripts/make_h8_tables.py` — never edit a figure in `paper/main.tex` by hand.**

---

## §6. What is LEFT beyond the re-runs

1. **K=2 on 10 cells inside `main-v1`** (§9f) — 3 models × {bf16, gptq-4bit, awq-4bit} plus
   llama/gptq-8bit. **Not yet implemented in `sm_reference.py`.** Coverage follows the
   *estimand*, not the empirical screen: qwen is included despite showing 0.00%, because at
   n=180 one item is 0.56 pp and qwen carries the largest rung gaps.
2. **`invocation_id` on every generation record** — committed to in R21, **not yet implemented**.
3. **The log-likelihood arms are NOT implemented.** `run_cell` is generation-only
   (`scoring_mode="generate"` fixed, `option_logprobs` always `None`), so **H6's arm 2 and arm 3
   do not exist**, and that also blocks **G3's ARC cross-validation**. Flagged in
   `sm_reference.py`'s docstring. **H6's mode contrast is incomplete until this is built.**
4. **The stratum extent** (§2.4) needs a planner ruling: if most cells fall back to Manski
   bounds, H4's centrepiece may rest on bounds the prereg admits may be uninformative.
5. **Paper:** 7 of 8 pages, 11 `\todo`. H8, Related Work and Method written; six Results
   subsections outstanding. `make -C paper g6` fails only on the abstract `\todo`, which is
   scheduled for W5 after the results freeze — **not a regression**.
6. **Extend the protocol survey** — 4 of ~14 bib entries reporting a delta are checked. No
   counted field-level claim until M covers them; cite Pape et al. for the general claim.
7. **Upstream vLLM report** on the cross-invocation nondeterminism — planner-approved, after
   Sep 2, with the four refuted hypotheses.

## §7. USER-OWNED, unchanged

- 🚨 **Oct 14 — both authors must register as ARR reviewers/ACs.** Per-author. **Missing it is
  an automatic desk reject.**
- The Midway cert for dev-desktop SSH expires ~daily; only the user can run `mwinit -s`.
- Optional: raising the SageMaker quota from 1 to 2 would remove submit-time rejections.

## §8. How the user works — worth knowing

The user **caught two of my errors this session by pushing back**, and both times they were
right:

1. They rejected my "library bug" framing and asked *"shouldn't we fix the assumption rather
   than say it was wrong?"* → became the template factor, a better design.
2. They asked *"can't it be temperature and sampling?"* → the hypothesis was wrong, but the
   question exposed the uncontrolled `generation_config`.

They also asked for a **pre-flight check before the main run**, which found §3's problem. **Take
their challenges seriously and verify rather than defend.** They prefer plain-language
explanations when asked, want the logs and README kept current, and want to be told when
something is genuinely open rather than given a confident guess.
