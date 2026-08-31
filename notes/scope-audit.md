# Scope audit — every task in `EXECUTOR-TASKS.md` §6 against evidence on disk

**Run 2026-08-31**, at the user's request (*"the Research and delivery tracks have many things
missing"*). They were right. Checked by looking for the artifact each task names as its
*Done when* / `→` output, not by recollection.

⚠️ **Method:** a task is ✅ only if its named evidence exists. "Code exists that could produce
it" is 🟡. This is deliberately stricter than the gate log, which tracks gates rather than tasks.

---

## Summary

| | count |
|---|---|
| ✅ done | 11 |
| 🟡 partial | 8 |
| ❌ not started | 12 |

**The compute track is far ahead of the delivery track.** 33 GPU jobs and $52.77 have produced
H8 (complete, mechanistically pinned), the checkpoints, the pilot and the template factor — while
**the entire statistical layer, the entire judge/annotation track, and five of eight paper
sections do not exist.** Nothing downstream of "generate and score one arm" has been built.

🚨 **Two findings that are worse than "not started":**

1. **Five required paper sections are EMPTY and G6 passed anyway** — see `G6-BLIND` below.
2. **`src/quantartifact/analysis/` contains only `__init__.py`.** Every statistic the paper's
   claims rest on — the always-compliant stratum, the compliance-transition matrix, Manski
   bounds, the H3 GLMM, McNemar, Holm–Bonferroni — is unimplemented. `A19` is not "analysis
   pending results"; there is no analysis code at all.

---

## 🚨 G6-BLIND — the gate could not see five empty sections

`make -C paper g6` reported **PASS** on all four criteria while these carried a heading, a
`\label`, planning comments, and **zero prose**:

| section | prose lines |
|---|---|
| Introduction | **0** |
| Discussion | **0** |
| Conclusion | **0** |
| **Limitations** | **0** |
| Ethics Statement | **0** |
| Appendices ×3 | 0 (permitted — A23 requires they carry no load-bearing content) |

The Limitations check was `grep -q 'section\*{Limitations}'` — it tested that the *heading*
existed. **An empty Limitations section is an automatic desk reject exactly as a missing one is**
(A23). This is the project's own recurring failure — *declared versus enforced* — arriving at the
level of the instrument that checks the deliverable.

**Fixed 2026-08-31:** `scripts/check_sections.py`, wired into `g6`. Comments and bare `\todo{}`
do not count as prose; a section whose only content is a table does not count as written. G6 now
fails, correctly.

---

## Track B — people and external latency

| id | task | status | evidence |
|---|---|---|---|
| B1 | HF token + Llama licence | ✅ | `notes/access.md`; Llama-3.1-8B config downloaded, rev pinned |
| B2 | ARR profiles + Oct 14 reminder + AWS | 🟡 | `notes/arr-obligations.md` exists. **USER-OWNED and unverifiable by me**: both authors' OpenReview profiles complete, both Oct 14 reminders set. Missing it is an automatic desk reject |
| B3 | GPU account + test pod | ✅ | `notes/compute.md`; closed on verified quota/pricing/AZ instead of a pod (D10/D11) |
| B4 | External reader confirmed in writing | 🟡 | `notes/review-plan.md`; D12 records it closed. **Draft due to them Sep 30** |
| **B5** | **Annotation rubric, frozen, co-author signed off** | ❌ | **`annotation/rubric.md` DOES NOT EXIST.** Needs the co-author, was a day-2–4 task, and `A11` cannot start without it |

## Track A — W1

| id | task | status | evidence |
|---|---|---|---|
| A1 | Two venvs, pinned | ✅ | `.venv`, `.venv-harness`, `pyproject.toml` |
| A2 | Repo + compiling paper skeleton | ✅ | `paper/main.pdf`, 7 pp, tectonic |
| A3 | Four configs, SHA-pinned | ✅ | `configs/{models,quant,tasks,scoring}.yaml` |
| A4 | Generation core | ✅ | `src/quantartifact/generate.py`; resumable, `finish_reason`/`n_gen_tokens` on every record |
| A5 | Scorers rungs 1–3 + compliance + trace scan | ✅ | `scoring/extract.py`, `scoring/compliance.py`, 340 tests |
| A6 | Throughput measured ± `VLLM_BATCH_INVARIANT` | 🟡 | Rates measured (BBH 7,099 / gsm8k 5,250 gen/h) but ⚠️ **never measured WITH vs WITHOUT batch invariance** — that pairing is the paper's BI-overhead number. `notes/estimates.md` (the named output) **does not exist** |
| A7 | Determinism audit (H8) → **G4** | ✅ | G4 PASSED. Frontier ×3 levels ×3 metric levels, Marlin pinned by intervention, 10-ordering SD. ⚠️ Named outputs `artifacts/tables/determinism.csv` and `notes/determinism.md` **do not exist** (content is in `RUNLOG.md` + `artifacts/determinism*.json`) |
| A8 | Stratification pilot | 🟡 | `pilot-v2`, 17 cells × 300, scored. ⚠️ **The 250-item blinded annotation sample was never drawn** — that is half of A8's *Done when* |

## Track A — W2

| id | task | status | evidence |
|---|---|---|---|
| **A9** | **Anchors A–D → G5a** | ❌ | **`artifacts/tables/anchors.csv` DOES NOT EXIST.** Anchor A is *mandatory, ungated, BLOCKING*: our self-quantized GSM8K within ±2.0 pp of Qwen's official GPTQ-Int4/Int8/AWQ. **Never run.** Anchor B (Red Hat) verified only *at source* in `protocol-survey.md`, not reproduced. Anchor C scoped out as llama.cpp/CPU (R19.3) |
| **A10** | **Judge harness (xVerify + granite)** | ❌ | **`scoring/judge.py` DOES NOT EXIST.** Rung 4 of a four-rung ladder is absent. No prompt template, no 50-item validation, no output-parse-rate measurement |
| **A11** | **Human annotation ≥250 items, ≥120 double** | ❌ | **`annotation/` is EMPTY.** No labels, no IAA. Described in TASKS as *"the schedule's tightest constraint"*; blocked on B5 |
| **A12** | **Judge validation → gate** | ❌ | **`artifacts/tables/judge_validation.csv` DOES NOT EXIST.** Blocked on A10+A11. Non-blocking for the paper by design (headline is rung 1 vs 2) but it is a named gate |
| A13 | Main reference runs, 15 cells × 2,719 | 🟡 | `sm_reference.py` written, smoke passed. **HELD** — it is the prereg freeze line, and K=2 + `invocation_id` are unimplemented |
| **A14** | **Noise floor: bf16 twice, different seeds** | ❌ | **`artifacts/tables/noise_floor.csv` DOES NOT EXIST.** §4.5 calls it mandatory. Partly superseded in spirit by the invocation-noise work (§9f), but *that* measures re-invocation at one seed, **not two seeds end-to-end** — different quantity, and the prereg's noise-floor pre-commitment cites this one |

## Track A — W3–W4

| id | task | status | evidence |
|---|---|---|---|
| **A15** | **Token-budget sweep (H1 — "Lead. Never cut.")** | ❌ | **Only `1x` exists anywhere in the code** (`grep` finds `"1x"` three times, `2x`/`4x` zero). `configs/tasks.yaml` declares all three levels; **nothing consumes 2× or 4×.** H1 is the lead hypothesis and its instrument is unbuilt |
| A16/A17 | Score everything at all rungs, versioned | 🟡 | `score_pilot.py`, `score_probe.py`, `score_template.py` exist. ⚠️ Rung 3 (`math-verify`) **not installed in `.venv`** — every scoring run so far printed `math_verify UNAVAILABLE`. Rung 4 needs A10. Outputs are not keyed by `(cell, scorer_name, scorer_version)` |
| **A18** | **Zero-shot arm (H5), 9 cells × 1,400** | ❌ | `prompt_variant` is hardcoded `"fewshot_cot"` in all three generation stages. Zero-shot rendering was also listed as not-done under G2 |
| A19 | **Analysis** | ❌ | **`src/quantartifact/analysis/` = `__init__.py` only.** No stratum, no transition matrix, no Manski bounds, no GLMM, no McNemar, no Holm–Bonferroni. `statsmodels`/`scipy` are declared in `pyproject.toml` and unused |
| A20 | Powered-ness audit (H7) | 🟡 | `notes/protocol-survey.md`: **4 of ~14** cited papers checked. R19.3 forbids a counted field-level claim until M covers them |
| A21 | Figures and tables | 🟡 | `paper/tables/h8_*.tex` generated from reports (good pattern). **`artifacts/figures/` holds only `.gitkeep`.** The §4.2 decomposition table — *"the paper's centrepiece, design it FIRST"* — does not exist |

## Track A — W7

| id | task | status |
|---|---|---|
| A23 | Compliance sweep | ❌ Responsible NLP Checklist not started. §C (compute/hyperparameters/error bars) and §E (AI assistance) both apply. Incorrect/incomplete filing is itself a desk-reject ground |
| A24 | Artifact release | ❌ Not assembled. Note `pilot-v1` **must** be retained (it is the evidence for the BBH terminator finding) and xVerify weights must **never** be redistributed |

## Track C — writing

| id | task | status |
|---|---|---|
| C1 | Related Work | ✅ 94 prose lines, 23 verified citations, nearest-prior-work differentiated |
| C2 | Pre-registration | 🟡 Extensive (§1–§9g). Binding line is the first `main-v1` generation (§9f), not Sep 2 |
| C3 | Method | ✅ 94 prose lines across five subsections |
| **C4** | **Introduction** | ❌ **EMPTY.** Due W4 |
| **C5** | **Limitations + checklist** | ❌ **EMPTY.** Desk-reject ground |
| C6 | Submittable draft (G6, **Sep 20**) | ❌ Fails on five empty sections |
| C7 | Results + Discussion + Abstract | 🟡 H8 written; six Results subsections are one-line `\todo`s; Discussion, Conclusion, Abstract empty |
| C8 | Related-work revision vs Sep 14 novelty scan | ❌ Not due yet |

---

## What this changes about sequencing

The held `main-v1` is **not** the critical path. Even with it complete, the paper could not be
assembled: there is no code to compute H3's interaction, H4's decomposition, or the stratum, and
no judge to populate rung 4.

**Cheapest-first, and all of it needs zero GPU:**

1. **`A19` analysis module** — blocks H3, H4, H7 and the centrepiece table. Largest single gap.
2. **Install `math-verify`** — rung 3 has silently never run. One line, and it is a declared rung.
3. **`C5` Limitations + `C4` Introduction** — both cost nothing but writing, and Limitations is a
   desk-reject ground. G6 now enforces this.
4. **`B5` rubric** → unblocks `A11` → `A12`. Needs the co-author, so it has people-latency and
   should start before anything that only needs me.
5. **`A21` decomposition table** — TASKS says design it *first* and let it drive logging. It is
   being designed last, which is how a required field goes unlogged.

**Needs GPU, in dependency order:** `A9` Anchor A (blocking, never run) → `A14` noise floor →
`A15` budget sweep (H1's instrument) → `A18` zero-shot → `A13` `main-v1` → H6's log-likelihood
arms (R24.2: after `main-v1`, before G5).

⚠️ **`A15` deserves flagging separately.** H1 is *"Lead. Never cut."* and the sweep that measures
it does not exist in any form. 9 cells × 1,400 × 2 extra levels is ~25,000 generations — real
compute that is not in any current plan or budget line.
