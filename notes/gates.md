# Gate log

`EXECUTOR-TASKS.md` §8: *"Report each gate to me with its artifact the day it closes."*
This is the running record. Evidence paths are repo-relative.

| Gate | Due | Status | Closed | Evidence |
|---|---|---|---|---|
| **G1** env + access | Aug 25 | ✅ **PASSED** | 2026-08-23 | `notes/access.md`, `notes/compute.md`, `pyproject.toml` |
| **G2** prompt parity | Aug 27 | ✅ **PASSED** | 2026-08-26 | `artifacts/prompts/MANIFEST.json` (27 cells, 0 failures) |
| **G3** scorers | Aug 28 | 🟡 partial | — | 142 tests green; ARC log-likelihood cross-val outstanding |
| **G4** determinism (H8) | Aug 30 | ⛔ blocked | — | needs GPU — gated on D10/D11 user confirmation |
| **PREREG** | **Sep 2** | 🟡 drafted | — | `analysis/preregistration.md`, all nine R-items written |
| **G5a** anchors | Sep 2 | ⛔ blocked | — | needs GPU |
| **G5** reference runs | Sep 13 | ⛔ blocked | — | needs GPU |
| **G6** submittable draft | **Sep 20** | 🟡 mechanism ready | — | `make -C paper g6` checks all four criteria |
| G7 results frozen | Sep 25 | — | — | |
| G8 external review | Oct 4 | 🟡 on track | — | `B4` closed — readers ready (D12); draft due Sep 30 |
| G9 submitted | Oct 10 | — | — | |

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
