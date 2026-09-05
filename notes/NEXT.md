# What to do next

**Written 2026-09-05 at the user's request, at a clean pause.** Working tree clean at `722dbd4`,
519 tests passing, ruff clean, no unfilled RUNLOG entries, **nothing running and nothing queued**.

Ledger: **44 jobs · 34.81 h · $97.47**. Budget is not a constraint (D15). **Wall clock is** — ARR
is **2026-10-12** and there is one shared queue slot with waits of 1 h to 6 h 19 m.

---

## 1. Where the project actually is

**The main run is DONE.** `run_id=main-v1`, 105,950 generations across 25 units (15 cells plus 10
K=2 repeats), every unit at exactly 4,238 items, nothing skipped, schema 1.5.0 throughout, join key
unique. Scored on all four rungs with `math-verify` available. Rule 4b clean.

**So generation for the reference condition is complete.** What remains is analysis, three
unbuilt arms, and the paper.

---

## 2. 🚨 The one job that is READY TO SUBMIT

`sm_diagnose.py` is rewritten and tested but **not submitted**. It answers two things at once and
the planner approved both (R36):

| group | what | why it matters |
|---|---|---|
| **A** | cross-job invocation floor at full reference n. 3 cells (`llama bf16`, `llama gptq-4bit`, `qwen bf16`) × 4,238 items, `main-v1`'s exact image and config, so the **only** difference is the job boundary | §9f's registered floor **measured 0.000** and this replaces it. Four registered quantities depend on it |
| **B** | cold-`torch.compile` test. 2 invocations with **isolated** cache roots, plus a **shared**-cache negative control | The first testable mechanism after four refutations |

```bash
.venv/bin/python scripts/submit_job.py --stage diagnose --plan-only        # inspect, free
export HF_TOKEN=$(grep '^HF_TOKEN=' ~/workspaces/MedLLMSafety/.env | cut -d= -f2-)
.venv/bin/python scripts/submit_job.py --stage diagnose --smoke --i-have-confirmation
.venv/bin/python scripts/submit_job.py --stage diagnose --i-have-confirmation
```

⚠️ **Smoke it first.** `sm_diagnose.py` changed substantially: new probe tuple shape, per-invocation
`env` for the cache root, `BASELINE_RUN` moved to `main-v1`, `N_ITEMS` split into two, and the
subprocess timeout raised to 4 h for full-n cells. None of that has run on GPU.

⚠️ **It also has NOT been through the CODE REVIEWER**, and rule 6 requires that before it feeds a
billable job. Do that first: `./ask-reviewer "review the R36 diagnose rewrite. $(git diff HEAD~1 -- sm_diagnose.py)"`

**Expected outcomes and what each means:**
- Group A gives a non-zero cross-job SD → that becomes the class-2 and class-3 floor, §5a's
  destruction floor, and the between-arm floor. Report the pilot's 1/√n extrapolation alongside as
  a validation of the scaling law.
- Group B **isolated caches diverge** → the phenomenon is **diagnosed**: cross-job divergence is
  cold `torch.compile`, and the recommendation becomes *report whether your compile cache was
  warm*. Far more citable than an unexplained observation.
- Group B **isolated caches agree** → a fifth refuted hypothesis, recorded with its killing number,
  and the unexplained framing stands.

---

## 3. What blocks the paper, in the order it actually matters

### 3a. 🚨 Five paper sections are EMPTY, and one of them is a desk reject

`make -C paper g6` fails on exactly this. **Limitations being absent is an automatic desk reject**
(`A23`) and costs only writing. Introduction, Discussion, Conclusion and Ethics are also empty.
Related Work and Method are written (94 prose lines each), and H8's Results subsection is written.

**This is the largest acceptance risk on the board and no amount of compute touches it.** `D15`
says so explicitly so that unconstrained budget does not become a reason to prefer jobs over prose.

Paper is at 7 of 8 pages with six Results subsections still one-line `\todo`s, so the page budget
will bind. That is a scope conversation, not an execution one.

### 3b. Run the analysis layer on `main-v1`

```bash
.venv/bin/python scripts/analyze_run.py --run-id main-v1
```

⚠️ **Not yet done for `main-v1`** — only for `pilot-v2`. This produces H3's interaction, the
stratum, the transition matrices, the Manski bounds and the Holm families at reference n. Several
outputs are compared against the invocation floor, so §2's job informs them, but most of it does
not wait.

⚠️ Expect the **stratum** to be the interesting problem: on the pilot it was below 50% in most
cells and `gsm8k` was 1–27%. R24.3 dissolved the threshold and made bounds co-primary, so this is
handled, but H4 may rest largely on bounds.

### 3c. Three arms that do not exist

| arm | blocks | note |
|---|---|---|
| **H6's log-likelihood arms** | H6 (the "never cut" negative control) **and** G3's ARC cross-validation | Needs a build, not just a job: `run_cell` is generation-only, `scoring_mode` is fixed and `option_logprobs` is always `None`. R24.2 says build after `main-v1`, before G5 closes. **That is now.** |
| **`A15` token-budget sweep** | **H1, the LEAD hypothesis, marked "never cut"** | The instrument does not exist. Levels are in `configs/tasks.yaml` and consumed by nothing. ~25,000 generations. **The single largest acceptance risk money can fix** (D15). |
| **`A18` zero-shot arm** | H5 | `prompt_variant` is hardcoded `"fewshot_cot"` in all three generation stages |

Also unrun: **`A14` pipeline noise floor** (bf16 twice at different seeds; §4.5 calls it mandatory,
and it is a different quantity from the invocation floor).

### 3d. User-owned, and unbuyable

- 🚨 **Oct 14 — both authors register as ARR reviewers/ACs.** Per-author, within 48 h of
  submission. **Missing it is an automatic desk reject.**
- **`B5` annotation rubric** needs the co-author; it blocks `A11` annotation and `A12` judge
  validation, and `A10`'s judge harness does not exist either (rung 4 of a four-rung ladder).
- A draft must reach the external reader by **Sep 30**.

---

## 4. Findings from `main-v1` worth not losing

**K=2 measured exactly zero.** 20 invocations, 42,380 item-pairs, **zero differences**, including
the Llama cell that diverged on 41–45% of items *between* jobs. So within-job generation is bitwise
reproducible and the nondeterminism is **cross-job only**. Consequences are folded into §9f in place.

**The ×√2 credit never materialised.** Averaging two identical runs reduces nothing, so H3 binds to
the full cross-job floor and its resolution is worse than R23 projected.

**H1's sign flips on Llama** (from `main-v1-smoke4`, n=36/cell, indicative only): measured on the
full generation, quantization looks like it *shortens* CoT by 15 tokens; measured on the **scored
answer** it *lengthens* it by 8. Llama's bf16 arm discards 41.4 tokens per generation against 6.9
for its 4-bit arm. §5b.1 already registers the scored length as primary, so this is that rule being
vindicated. **Recompute on `main-v1` at full n.**

**The harness discards recoverable answers.** 6 of 7 empty-scored-prefix items had a *correct*
answer in the discarded continuation, found by the extractor at rung 1, arm-differentially.
Registered confirmatory in §5b.2 with §8's false-positive guard mandatory. **Only measurable on
post-R34 data.** Recompute on `main-v1`.

---

## 5. Process state a new session must know

- **D16, new today: documents are NOT frozen and there are no dated amendments.** Corrections go
  **in place, in the section that makes the claim**, dated, saying what it used to say. "Not frozen"
  means corrected **visibly**, never quietly. `CLAUDE.md` §3 and the prereg's own maintenance
  section carry the rule; `TestPreregIsMaintainedNotFrozen` enforces it.
- **D15: budget is not a constraint**, but D10's confirmation flag stays, `--smoke` first stays, and
  wall clock is the real limit.
- Rules that keep earning their cost: read 3 raw generations after **every** prediction job before
  quoting any number (rule 4b — it found the Llama defect); call the planner and CODE REVIEWER
  rather than announcing that they need calling; gate commits with `&&` never `;`.
- ⚠️ **A mistake to not repeat**, from today: I truncated the prereg from 1,828 lines to 30 with a
  scripted string replace, because `"## Amendments"` matched the phrase inside a paragraph rather
  than the section header. Recovered in full from git because it had been committed first. **Anchor
  scripted edits on line numbers, and commit before editing.**
- Rulings run to **R36** in `notes/planner-rulings.md`, raw answers in `notes/planner-answers/`.
  Seven CODE REVIEWER rounds in `notes/reviews/`, each with an OUTCOME block.
- `notes/scope-audit.md` is the full task-by-task audit against `EXECUTOR-TASKS.md` §6 and is the
  best single map of what is missing.

## 6. If you do only one thing

**Write the Limitations section.** It is an automatic desk reject if absent, it costs no compute, no
queue slot and nobody else's calendar, and `make -C paper g6` currently fails on it.
