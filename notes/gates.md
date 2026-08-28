# Gate log

`EXECUTOR-TASKS.md` §8: *"Report each gate to me with its artifact the day it closes."*
This is the running record. Evidence paths are repo-relative.

| Gate | Due | Status | Closed | Evidence |
|---|---|---|---|---|
| **G1** env + access | Aug 25 | ✅ **PASSED** | 2026-08-23 | `notes/access.md`, `notes/compute.md`, `pyproject.toml` |
| **G2** prompt parity | Aug 27 | ✅ **PASSED** | 2026-08-26 | `artifacts/prompts/MANIFEST.json` (27 cells, 0 failures) |
| **G3** scorers | Aug 28 | 🟡 partial | — | **265** tests green; ARC log-likelihood cross-val outstanding |
| **G4** determinism (H8) | Aug 30 | 🟢 **measuring** | — | job 2 running (38,000 gen). Smoke evidence below is already substantive |
| **PREREG** | **Sep 2** | 🟡 drafted | — | `analysis/preregistration.md`, all nine R-items written |
| **G5a** anchors | Sep 2 | 🟡 unblocked | — | 9 quantized checkpoints exist (job 1 complete, 45.86 GB) |
| **G5** reference runs | Sep 13 | 🟡 unblocked | — | substrate proven end-to-end; gated on G4's frozen `max_num_seqs` |
| **G6** submittable draft | **Sep 20** | 🟡 mechanism ready | — | `make -C paper g6` checks all four criteria |
| G7 results frozen | Sep 25 | — | — | |
| G8 external review | Oct 4 | 🟡 on track | — | `B4` closed — readers ready (D12); draft due Sep 30 |
| G9 submitted | Oct 10 | — | — | |

---

## G4 — determinism · MEASURING (job 2 in flight)

**The substrate is proven and the audit is running.** Job 1 delivered 9 checkpoints; job 2
(38,000 generations) is measuring the frontier. The smoke already produced results that stand
on their own, and they changed the gate's own criterion.

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

| arm | text | strict answer | permissive answer |
|---|---|---|---|
| bf16 | 0.0% | 0.0% | 0.0% |
| gptq-4bit | 65.0% | 10.0% | 20.0% |
| awq-4bit | 75.0% | **0.0%** | 40.0% |

AWQ's traces differ 75% of the time and its committed answer never does. **And the permissive
rung is more order-sensitive than the strict one**, which cuts toward C1's "calibrate
extraction, don't loosen it" framing.

### What the running job must still deliver

1. The frozen `max_num_seqs`, **earned on the ordering axis** at {8, 32, 64} rather than
   defaulted to 8 (R15.5). If order-sensitivity is flat, 64 is a 4× throughput win downstream.
2. Whether **correctness**-level divergence clears 10 pp — R15.6's escalation trigger. At n=20
   it is 2/20 with a Wilson CI of roughly 1–31%, so undecidable; the 500-item probe decides.
3. **Marlin attribution**, paired ±`VLLM_DISABLED_KERNELS`, to pin the mechanism to a named
   kernel rather than inferring it.
4. Model generality on Phi-3.5-mini, the architectural outlier.
5. A6's throughput deliverable per arm — already measured at bf16 ~3,900 gen/h vs gptq-4bit
   ~15,600 gen/h, a ~4× small-batch advantage consistent with Kurtic at batch=1 (2.39×).

Full run-by-run record with costs: `notes/RUNLOG.md`.

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

**Done:** 142 tests green, ruff clean. Every fixture `EXECUTOR-TASKS.md` `A5` names is covered,
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
