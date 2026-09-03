# Gate log

`EXECUTOR-TASKS.md` §8: *"Report each gate to me with its artifact the day it closes."*
This is the running record. Evidence paths are repo-relative.

| Gate | Due | Status | Closed | Evidence |
|---|---|---|---|---|
| **G1** env + access | Aug 25 | ✅ **PASSED** | 2026-08-23 | `notes/access.md`, `notes/compute.md`, `pyproject.toml` |
| **G2** prompt parity | Aug 27 | ✅ **PASSED** | 2026-08-26 | `artifacts/prompts/MANIFEST.json` (27 cells, 0 failures) |
| **G3** scorers | Aug 28 | 🟡 **partially met** | — | **340** tests green. ⚠️ **ARC cross-validation OUTSTANDING, scheduled after `main-v1` and before G5 closes** (R24.2). It needs the log-likelihood arms, which do not exist — `run_cell` is generation-only. 🚨 **The ≤0.5 pp criterion and R11's addendum are NOT relaxed**: relaxing a gate because it is inconvenient is the declared-vs-enforced pattern reappearing at gate level, and this project has already paid for that three times. |
| **G4** determinism (H8) | Aug 30 | ✅ **PASSED** | 2026-08-29 | frontier at {8,32,64} × 3 levels; `max_num_seqs=64` frozen (R16.2). Superseded in magnitude by the 10-ordering run (bf16 exactly 0.000 across ten, worst quantized SD 1.14 pp); a re-run with fixed BBH terminators is in flight |
| **PREREG** | **Sep 2** *(schedule)* | 🟡 drafted | — | `analysis/preregistration.md`. 🚨 **The BINDING line is the first `run_id=main-v1` generation, not the calendar** (§9f) — until it exists, §1–§9 may be edited, because no edit can have been influenced by confirmatory results |
| **G5a** anchors | Sep 2 | ❌ **FAILED, decision discharged** | 2026-09-03 | `artifacts/tables/anchors.csv`. `gptq-8bit` PASS (+0.45 pp), `gptq-4bit` **FAIL** (+4.09 pp), `awq-4bit` dropped (vLLM's `awq_dequantize` is fp16-only). 🚨 The FAIL is permanent and not retroactively converted; the DECISION is discharged on R33.1's three grounds, with both alternative explanations checked and excluded (R33.3). `main-v1` unblocked. |
| **G5** reference runs | Sep 13 | 🟡 unblocked | — | substrate proven end-to-end; gated on G4's frozen `max_num_seqs` |
| **G6** submittable draft | **Sep 20** | 🟡 mechanism ready | — | `make -C paper g6` checks all four criteria |
| G7 results frozen | Sep 25 | — | — | |
| G8 external review | Oct 4 | 🟡 on track | — | `B4` closed — readers ready (D12); draft due Sep 30 |
| G9 submitted | Oct 10 | — | — | |

---

## G4 — determinism · PASSED 2026-08-29

**Job 2 is terminal and delivered all five of its criteria.** `max_num_seqs = 64` is frozen
(R16.2) and recorded in `configs/models.yaml` as `max_num_seqs_resolved_by_audit: true`, which
unblocks G5/G5a and job 3. The full frontier, the Marlin attribution and the per-arm throughput
are in `notes/RUNLOG.md` under `determinism · Failed · 8755 s`.

⚠️ **The job reads `Failed` and was not one.** Exit 1 is my own rule — non-zero when any cell
could not be *measured* — and it fired for **one cell of 21** (`gptq-8bit@8+marlin-off`, which
needs `libnvrtc-builtins.so.13.0`). The planner ruled to skip it (R16.3): the mechanism is
already pinned by a clean 33.4→0.0 and 43.6→0.0, and the absence is itself a reportable finding.

**One analysis outstanding, and it is not a gate condition.** R16.5 requires the **signed
accuracy delta** between orderings, per arm — flip rate and accuracy shift are different
quantities, and which one holds decides whether the claim is *"reordering moves reported
accuracy more than 4-bit does"* or *"item-level churn exceeds published deltas while aggregates
look fine."* Both publishable, different papers. Computable from generations already held via
`--stage determinism --reanalyze` (no generation, ~$0.30). **The narrative does not reorder
until it lands.**

### 🚨 R8's pass criterion was RETIRED before it could pass

The original criterion — largest `max_num_seqs` bitwise-stable across 3 repeats for every arm —
**measures an axis with no variance**. Measured at `max_num_seqs=8`, 20 items, qwen2.5-3b:

| arm | repetition floor (same order) | order divergence (3 orderings) |
|---|---|---|
| bf16 | 0.000% | **0.000%** — order-invariant |
| gptq-4bit | 0.000% | **65.0%** |
| awq-4bit | 0.000% | **75.0%** |

Repetition says every arm is perfectly reproducible. Reordering the same items changes 65–75%
of both quantized arms' outputs and none of bf16's. The criterion would have certified a
grossly unstable configuration as stable. Replaced (R15.4) with order-induced divergence swept
at {8, 32, 64}.

### The finding, and why AWQ matters most

**AWQ *has* the batch-invariance guard GPTQ lacks, and AWQ is worse.** So the guard does not
deliver batch invariance — a harder claim about the feature than a confirmed prediction would
have been, and a vLLM bug report. bf16's order-invariance under the identical structure makes
the effect quantization-specific rather than a general engine property.

**Confound broken by construction:** all passes run in ONE engine, so ordering is the only
variable. An earlier design ran each ordering in its own process; the argument that bf16's
stability ruled that out was *unsound*, since it assumed both arms are equally sensitive to
process state.

### Divergence collapses by the answer level (R15.3)

Smoke, 20 items — superseded in magnitude by the 500-item frontier below, but this is where the
three-level requirement came from:

| arm | text | strict answer | permissive answer |
|---|---|---|---|
| bf16 | 0.0% | 0.0% | 0.0% |
| gptq-4bit | 65.0% | 10.0% | 20.0% |
| awq-4bit | 75.0% | **0.0%** | 40.0% |

AWQ's traces differ 75% of the time and its committed answer never does. **And the permissive
rung is more order-sensitive than the strict one**, which cuts toward C1's "calibrate
extraction, don't loosen it" framing.

### ✅ All five criteria, delivered — 500 items, qwen2.5-3b unless noted

| # | criterion | result |
|---|---|---|
| 1 | frozen `max_num_seqs`, earned on the **ordering** axis at {8,32,64} (R15.5) | **64** (R16.2). GPTQ-4's correctness difference across levels is noise (4.2 vs 4.8% ≈ 0.6 SE); 4× throughput on ~110k remaining generations. ⚠️ *text* divergence genuinely is worse at 64 (33.4→39.6%, ~3 SE) — reported, does not bind, since the claims are about scores not strings. |
| 2 | does **correctness** divergence clear R15.6's 10 pp trigger? | **No.** Worst strict-correctness **5.4%** (gptq-4 @32); worst *permissive*-correctness **7.0%** (awq-4 @32) — computed, not inferred, per R16.1, since §2.2's headline is rung 1 **vs rung 2**. No escalation before the freeze. |
| 3 | **Marlin attribution**, paired ±`VLLM_DISABLED_KERNELS` | **Pinned to a named kernel.** gptq-4 33.4→**0.0%**, awq-4 43.6→**0.0%**. bnb-nf4 31.4→31.4% = a **second, distinct, Marlin-independent** mechanism that *improves* with batch size (31.4→1.4%), backwards from every other arm. |
| 4 | model generality on Phi-3.5-mini, the architectural outlier | **Same shape.** bf16 0.0%; gptq-4 37.2% text / 5.6% answer / 2.6% correctness. Rates transfer across architectures. |
| 5 | A6 throughput per arm | ⚠️ **This row's original number was WRONG and is retained only to show why** (R20). It read ~24,800 gen/h — measured while BBH's broken terminators emitted ~nothing on 5 of 6 subtasks, so most of the "generations" were 1–9 tokens. Real rates with BBH fixed: **BBH 7,099 gen/h, gsm8k 5,250, ARC effectively free**, and **generation is only ~25% of billable time** — **2.6 min/cell of engine-load overhead dominates**, making CELL COUNT a first-class cost driver rather than item count. |

Two results beyond the criteria: the **repetition floor is 0.000% in all 15 cells** (§9's
pre-commitment binds here), and **autotune is ruled out empirically** — 33.4% with it off, 33.4%
with it on, so R14.1's confound does not exist and the finding is not an autotune artifact.

**Text divergence overstates the correctness impact by ~8–12×** (AWQ 47.0% → 3.8%). That is
precisely why R15.3 ordered three levels, and it is the frontier — not the frozen value — that
is the contribution: *it tells other people what they can get away with*.

Full run-by-run record with costs: `notes/RUNLOG.md`. Planner rulings: R15, R16.

---

## G1 — environment and access · PASSED 2026-08-23

- Llama-3.1-8B `config.json` downloaded → token **and** gated licence both confirmed.
  Revision `0e9e39f249a16976918f6564b8830bc894c89659`. The Sep 1 substitution is void.
- Substrate available with **zero external latency** (EC2, user's account, quota granted).
- Local analysis env resolves; `uv.lock` equivalent pinning in `pyproject.toml`.
- Instead of a pod launch-and-destroy, `B3` closed on verified quota + pricing + AZ
  availability, because launching anything now would violate D10/D11.

## G2 — prompt parity · PASSED 2026-08-26

**Criterion** (`EXECUTOR-TASKS.md` §8): byte-for-byte equality with `lm-eval` v0.4.12 for all
task configs. Blocking, because a prompt mismatch invalidates every comparison to published
numbers.

**How it is satisfied — by construction, not by diff.** Rather than maintain a second renderer
and test it against lm-eval, **lm-eval renders the prompts and we freeze the result** as a
versioned artifact that generation reads exclusively (`src/quantartifact/tasks/materialize.py`).
Parity then cannot drift, and the exact prompt strings *and token ids* ship in the release so a
reviewer can verify them without running anything. lm-eval remains a validation instrument only
(O9) — it never generates. It imports without torch, so this runs on the laptop.

**Evidence:** `artifacts/prompts/MANIFEST.json` — **27 cells** (3 models × 9 task configs),
**0 failures**, rolling SHA-256 per cell, re-render byte-identity verified per cell.

### Three real problems G2 caught

**1. `gsm8k` prompts were not reproducible.** The first run failed with
*"re-render differs for 8 items"*. Cause: `gsm8k` has `fewshot_split: train` and **no**
`fewshot_config`, so lm-eval **samples** its 5 exemplars at random — two renders of the same
item gave different prompts. `gsm8k_cot` and all six BBH subtasks hardcode their exemplars
(`sampler: first_n`) and were unaffected.

This mattered because `gsm8k` is the `answer_marker=hash` level, a **confirmatory** arm: its
prompts would have been irreproducible, and had exemplars differed across precision arms the
contrast would have been confounded by exemplars rather than precision. Fixed by pinning
`FEWSHOT_SEED = 1234` (lm-eval's own default) via `task.set_fewshot_seed()`, seeded once and
rendered in index order — exactly how the harness behaves in a real run, so parity holds.
Recorded per prompt as `fewshot_seed` / `fewshot_seed_matters`. **`gsm8k` is the only
seed-sensitive cell of the nine.**

**2. The double-BOS hazard is real, and only on Llama.** Measured:

| model | `bos_token_id` | template embeds BOS | naive `add_special=True` | resolved |
|---|---|---|---|---|
| Qwen2.5-3B | `None` | no | `[151644, …]` | `add_special=True`, 0 BOS |
| Phi-3.5-mini | `1`, `add_bos_token=False` | no | `[32010, …]` | `add_special=True`, 0 BOS |
| **Llama-3.1-8B** | `128000` | **yes** | **`[128000, 128000, …]`** | **`add_special=False`, 1 BOS** |

Llama's chat template already inserts `<|begin_of_text|>`, so the naive path yields a **double
BOS** — which would have silently shifted every Llama score and confounded every arm at once.
Red Hat's `add_bos_token=True` tip is for *non-template* use; blanket-applying it alongside a
chat template is the bug. Resolved per (model, rendered text) and recorded on every prompt.

**3. The BOS assertion was initially too strict, and Phi exposed it.** Demanding exactly one
BOS failed all nine Phi cells. Phi legitimately runs with **zero** (`add_bos_token=False`), so
the assertion now derives the expected count from each tokenizer's own behaviour on a bare
string and additionally rejects any count >1 outright. This catches a double BOS without
imposing one model's convention on another. Tested against all three profiles in
`tests/test_bos.py` with fakes, so it runs with no network.

### Not yet done for G2

- Full-scale materialization (currently `--limit 25` per cell). Deferred deliberately: the
  final item selection depends on the ARC 4-choice filter and the nested GSM8K subsample, so
  materializing in full now would produce an artifact we would immediately discard.
- Zero-shot variant rendering (H5 arm), which per prereg §3d must reuse the **few-shot**
  `max_gen_toks` and `until`.

## G3 — scorers · PARTIAL

**Done:** 305 tests green, ruff clean. Every fixture `EXECUTOR-TASKS.md` `A5` names is covered,
plus rung monotonicity on a fixture set, the `found`/`committed` split, the three termination
classes, the H2 false-positive floor, and rung 1b.

Two harness defects are pinned as tests rather than patched:
- **BBH `get-answer` drops the final character** when a generation ends exactly at its answer
  (`"the answer is 42"` → `'4'`). Fires in practice because `until` contains a bare `Q`.
- **`gsm8k_cot`'s strict regex captures the sentence period** (`\.` is inside the character
  class), so `"The answer is 6."` → `"6."`.

**Outstanding:** the ARC log-likelihood cross-validation — our `acc`/`acc_norm` within **0.5 pp**
of `lm-eval`'s on one bf16 model. That needs model weights on a GPU, so it is gated on D10/D11.
Everything else in G3 is closed.

🚨 **The criterion was pointed at the wrong instrument, and is now corrected** (planner ruling
R11 addendum). Our ARC-ll arm (**arm 2**) scores *option letters* on the *chat* prompt;
lm-eval's native `arc_challenge` (**arm 3**) scores *full answer texts* on the
`Question:`/`Answer:` prompt. Different instruments answering different questions — both
legitimate, **not interchangeable**.

> **The ≤0.5 pp cross-validation must run against lm-eval's NATIVE config, i.e. our arm 3.**
> Run against arm 2 it would have measured nothing.

Arm 3 exists precisely to be this target. Recorded before the run, so the criterion cannot be
quietly reinterpreted after seeing a number.
