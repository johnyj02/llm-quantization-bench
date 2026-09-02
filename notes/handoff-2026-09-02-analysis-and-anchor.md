# Handoff: branch "analysis layer and reference run" → mainline

**Written 2026-09-02.** Branch point `8b1a221`, **18 commits** of which the last is this document
itself, so the code work ends at `471cea4`. 35 files touched, +28,722 lines. The working tree is
clean and the whole range is already committed on `main`, so there is nothing to merge. This document
exists so a session that did not watch the work happen can rebuild its context without re-reading 17
diffs.

**Suite state at the time of writing: 506 passed, 2 skipped, ruff clean.** Both were verified with
`&&` chaining rather than `;`, for the reason in §7 below.

---

## 0. What this branch was for, and what it actually delivered

The user's instruction opening the branch was: *"Go ahead and start on the analysis layer, after that
we will do the reference smoke and Anchor A."* Two of those three landed. The analysis layer (task
**A19**) was built, reviewed five times, and **run on real data**. Anchor A (task **A9**, gate
**G5a**) was built along with its free preflight, and the preflight has passed. **No GPU job was
submitted on this branch**, so the compute ledger in `notes/RUNLOG.md` is unchanged and the D10
confirmation gate was never exercised.

The reference smoke and `main-v1` did **not** happen. They still need fresh authorisation, and §6
lists exactly what remains.

Seven planner rulings were obtained and implemented (**R25 through R31**). Five CODE REVIEWER rounds
were actioned, fixing two CRITICALs and seventeen MODERATEs. One of those CRITICALs would have put a
fabricated statistic in the paper, which is the single best argument for the review discipline the
branch also codified.

---

## 1. Read these files first, in this order

If the mainline session reads nothing else, read the first four.

| file | why it matters now |
|---|---|
| `analysis/preregistration.md` | **Amended on this branch**, and it is the binding scientific document. §1.1, §2, §5a (new), §9, §9f and §9h all changed. Still editable until the first `main-v1` generation exists, append-only thereafter. |
| `notes/planner-rulings.md` | The distilled rulings. R25 through R31 were appended here; the raw answers are in `notes/planner-answers/`. |
| `CLAUDE.md` | Gained a **standing header above §0** carrying the two-agent routing table, the decisive test for which agent owns a question, and the "NEVER ASSERT A STATE YOU HAVE NOT SEEN" rule. |
| `src/quantartifact/analysis/` | The layer that did not exist at `8b1a221`. Six modules, 1,832 lines. |
| `scripts/analyze_run.py` | The caller. Without it the layer produced no numbers, which was a real defect the user caught by asking "did we do something with that?" |
| `artifacts/analysis/analysis_pilot-v2.json` | The layer's real output over 5,100 pilot records. Every number in §4 comes from here. |
| `sm_anchor.py`, `scripts/anchor_preflight.py`, `scripts/score_anchor.py` | Anchor A: the job, its free gate, and the verdict computation. |
| `notes/reviews/2026-09-02T*.md` | Five CODE REVIEWER rounds, unabridged. |

---

## 2. The analysis layer (task A19)

Before this branch `src/quantartifact/analysis/` held only an `__init__.py`. It is the module that
turns stored generations into the quantities `analysis/preregistration.md` registers, and its absence
was the largest single gap in the project.

The design principle throughout is that **the pre-registration is enforced structurally rather than
by convention**. Where §9 says a quantity must carry something, the dataclass refuses to be
constructed without it, so a later session cannot quietly report a number that the registration does
not permit.

### `results.py` (377 lines) — the `Effect` type every quantity flows through

`Effect` is frozen and its `__post_init__` refuses five things outright: an inverted interval, a NaN
estimate or limit, a p-value with no named test beside it, a `"ratio"` unit whose `null_value` is
still `0.0`, and an estimate lying outside its own interval. The last guard fired on real data almost
immediately (17.5 against an interval limit of 17.499999999999996), so it carries a `1e-9 * span`
tolerance for exactly that floating-point boundary case.

Two functions carry most of §9h:

- `observed_max_contribution(...)` returns `(magnitude, source, n_violations)` where source is either
  `"observed"` or `"monotonicity_floor"`. It uses `math.ceil` rather than `int()`, because truncating
  1.9999999 to 1 would understate a bound.
- `zero_variance_bound(n, ...)` gives the Wilson upper limit on 0 out of n scaled by the magnitude,
  centred on a caller-supplied `centre`. The `centre` argument exists because a degenerate all-`+1`
  contribution vector is just as uninformative as an all-zero one, and the first implementation
  detected only the latter.

`holm_bonferroni(effects, *, family, alpha=0.05)` takes `family` as keyword-only with **no default**,
a MODERATE fix: a default family name would let two unrelated sets of tests silently pool into one
correction.

### `stratum.py` (260 lines) — §1.1, with three amendments

`CONTRASTED_ARMS` is now three arms rather than five (R18.4.1). The 50% membership threshold was
**dissolved** by R24.3, and R26.3 removed the rung quantifier, so compliance is now rung-free.
`StratumResult` carries `n_items_unpaired`, and `fraction` deliberately returns `None` when any item
is unpaired, exposing `fraction_of_paired` as the well-defined alternative. Reporting a fraction over
a denominator that silently dropped items is the shape of error this prevents.

### `transitions.py` (240 lines) — §1.2

The 2×2 transition table plus `mcnemar_exact`. With zero discordant pairs it returns
`(None, "mcnemar_exact")` rather than `p = 1.0`, because "no evidence either way" and "strong evidence
of no effect" are different claims and only the first is true.

### `bounds.py` (239 lines) — §1.3 Manski bounds

The difference bound combines **opposite sides**: `lower = alt.lower - reference.upper` and
`upper = alt.upper - reference.lower`. Combining same sides would produce an interval narrower than
the truth, which is the failure mode worth naming here.

### `interaction.py` (651 lines) — §2, hypothesis H3

`HEADLINE_RUNGS` is `("strict", "permissive")`. `EXCLUDED_FROM_H3_INTERACTION` is `{"gsm8k"}` per §9c.

`_item_key(row)` is shared by every collector and returns
`f"{model_key}|{task}|{subtask}|{item_id}"`. **This is CRITICAL 1 from review round 1.** `item_id` is
doc-scoped, so keying on `(arm, item_id)` collided items across tasks and models. The verified
symptom was `n = 1` where four items existed, and `n_items_dropped_unpaired` read 0 because the
collision happens *before* the pairing check could see it. A statistic computed over one item would
have gone into the paper reported as a population estimate.

Five public functions: `interaction_marginal` (primary, bootstrap), `interaction_glmm` (secondary,
GEE by default per R25.2b), `rung_mcnemar` (§2's registered companion, strict against permissive
*within* an arm, noise class 2), `rung_gap_decomposition` (R27.1, see §4) and `study_wide_magnitude`
(§9h's magnitude, obtained by asserting monotonicity on real data rather than assuming it).

`interaction_glmm` raises on `not fit.converged` and on non-finite coefficients. It had never actually
executed until a review round forced it to: `n` was being set to the row count rather than the item
count, and a converged fit was carrying NaN in `bse`.

### `decompose.py` (337 lines) — H4′

`EXCLUDED_FROM_H4_PRIMARY` is `{"gsm8k"}` per R24.3. `stopseq_differential` has **no `rung`
parameter** at all, because §4 defines `TRUNC_STOPSEQ` at rung 2 and accepting a rung argument would
imply a choice the registration does not offer. It raises on a missing reference arm and on no shared
items rather than returning an empty result.

### `scripts/analyze_run.py` (420 lines) — the caller

It reads generations, scores them across the rungs, reshapes into per-item rows, and emits every
registered quantity to `artifacts/analysis/analysis_<run_id>.json`.

Two details worth carrying forward. First, H3 and H4 each keep **two separate lists**, a registered
one and a `_demonstration_only` one, because feeding an excluded task's result into the Holm family
would reintroduce the very exclusion the prereg makes. I did exactly that once, one layer up from the
fix. Second, `holm_block` derives its `status` from `m`, the number of testable effects, rather than
from whether the list is empty. That was the **fourth** appearance of the vacuous-truth bug shape in
this project, and it appeared *inside the fix for its own previous appearance*.

---

## 3. Anchor A (task A9, gate G5a)

A9 is **mandatory, ungated and blocking**, and it had never run. It reproduces the three official
Qwen quantized checkpoints' behaviour and compares them against our self-quantized equivalents at
matched settings, with a pass band of ±2.0 pp on GSM8K.

**The construction is what makes the comparison mean anything.** Both sides are measured through
*our* pipeline, at the frozen serving configuration, on the same items, in the same engine version.
Any pipeline error is therefore common to both sides and cancels in the difference, leaving the
weights isolated. Comparing our number against a number Qwen published elsewhere would confound the
weights with every protocol difference between us and them, which is the exact confound §0.1(b) says
makes published deltas incomparable.

`sm_anchor.py` runs six cells, three arms crossed with {official checkpoint, ours}, one engine load
per cell in its own subprocess. Our side resolves **by recipe hash**, tying the job to the artifacts
job 1 actually produced rather than to whatever happens to be on disk. Official cells carry
`quant_config_hash=None` so a record from that side can never be mistaken for one of ours. The
tokenizer comes from the **base** repo on both sides, since a quantized checkpoint's tokenizer files
are a copy and even trivially different copies would confound weights with prompts.

Each cell now also records `engine_state` and `batch_invariance`. That was a MODERATE from review
round 5, and the reasoning is worth keeping: the docstring claims both sides run at a fixed serving
configuration, but our GPTQ-Int4 checkpoint and Qwen's are the same bit width and *not* the same
on-disk size, so at a fixed `gpu_memory_utilization` they can yield different KV-cache sizes and
therefore a different `num_gpu_blocks`. This project's own H8 result is that scheduling changes
generated text by up to 59%. vLLM also selects its GPTQ kernel from the checkpoint config, so the two
sides could take Marlin and non-Marlin paths, and H8 pinned Marlin specifically as the order-sensitive
one. Neither is likely to bite. The defect was that the job could not have told us whether it had.

**The verdict is deliberately not computed in the job.** It is CPU work over stored generations, so
computing it inside a billable process would put a gate decision somewhere that cannot be re-run
without a queue slot. `scripts/score_anchor.py` computes it.

### `scripts/anchor_preflight.py` — RUN, AND PASSED 2026-09-02

Per R30 this gates the GPU job. It downloads three small JSON config files and no weights, so it is
free, and it tests the most likely failure mode directly: a settings mismatch. All three official
checkpoints agree with our recipe on `bits`, `group_size` 128, `desc_act` False on both GPTQ arms,
and AWQ's gemm kernel and `zero_point`. Report at `artifacts/anchor_preflight.json`.

It runs under **`.venv-harness`**, not `.venv`, because `huggingface_hub` lives only there.

⚠️ Matching settings do **not** imply matching quality. The official checkpoints were calibrated on
Qwen's data rather than our `ultrachat_200k` 512×2048, so Anchor A tests **comparable quality, not
reproduction**. R30 was explicit that this framing has to reach the paper, because a reviewer who
reads the anchor as a reproduction claim will find a claim it cannot support.

---

## 4. Real numbers the layer produced, and the finding that came out of them

Run over `pilot-v2`, 5,100 records, with `math-verify` installed so rung 3 executed for the first
time. Full output in `artifacts/analysis/analysis_pilot-v2.json`.

**H3 interaction on BBH**, marginal estimator, percentage points against bf16:

| arm | estimate | 95% CI | n items |
|---|---|---|---|
| gptq-8bit | +2.22 | [+0.56, +3.89] | 360 |
| gptq-4bit | +3.33 | [+0.83, +6.11] | 360 |
| awq-4bit | +1.39 | [−0.83, +3.89] | 360 |
| bnb-nf4 | −0.28 | [−2.50, +1.94] | 360 |

**ARC is degenerate**, all four arms at +0.00 with the zero-variance bound ±3.15 pp at pilot n. That
is the `zero_variance_bound` path doing its job rather than a failure. **gsm8k is demonstration
only** and inverts sharply, gptq-4bit at −13.33 [−19.33, −7.33], which is precisely why §9c excludes
it from the registered interaction.

Holm families resolved as: H1 corrected over m=18, H3 corrected over m=4, and H4 reporting *"H4's
registered tests are not yet implemented as hypothesis tests"*, which is an honest status derived
from m rather than an empty block passing silently.

### 🚨 The scientific finding: rung monotonicity is violated on real generations

The layer measured **6 violations** on `pilot-v2` where the prereg's derivation had assumed none.
R26.1's magnitude derivation rested on that assumption, and the planner conceded it void:
*"the premise it rested on is empirically false, and measuring rather than assuming is what caught
it."*

I initially reported one mechanism and then corrected myself: there are **three**. The one that
matters most for how the finding is stated is class C, two llama dyck cases where strict is
**accidentally correct** through the drops-final-character defect. So the destruction claim rests on
4 of the 6, not all 6.

**R27's ruling, and it is the most consequential science on this branch.** Rung 2's last-candidate
rule *is* lm-eval's `flexible-extract` with `group_select: -1`, so this is a property of the standard
extractor the field uses and not a defect in our parser. Changing it would replace published practice
with an extractor of our own invention and destroy the thing being measured. So rung 2 stays exactly
as registered, and **rung 2a is added as a new pre-registered robustness rung**: commitment-anchored
permissive, selecting the first committed answer rather than the last candidate. Additive rather than
substitutive, so the pilot's numbers stay valid and nothing re-runs.

The planner also extracted content I had missed: the strict-to-permissive gap is a **net of two
opposing effects**, and reporting both directions separately is free and strictly more informative.

| | recovered (strict wrong → permissive right) | destroyed (strict right → permissive wrong) |
|---|---|---|
| rung 2 | ~914 items | 6 items |
| rung 2a | ~918 items | **1 item** |

Nobody reports the destruction rate of flexible extraction. R27's framing: *"Flexible extraction is
not a monotone relaxation, it is a different selection rule with two-sided error. That is novel and
citable."* It is also the sharpest support the paper has for §7's mandate to frame the contribution as
**calibrating** extraction rather than loosening it.

A directional prediction is now **registered before the fact** in §5a: if over-generation causes
destruction, the destruction rate should be **differentially higher on quantized arms**, which would
be a compliance-adjacent artifact running *opposite* to the paper's main hypothesis.

Rung 2a needed a fix before it froze. Its first implementation had **more** violations than rung 2,
14 against 6, from 13 non-extractions on ARC after dropping `last_option_letter`. R29 ruled that an
implementation error rather than a property, and anchoring the positional fallback to the
pre-commitment span fixed it, taking destroyed from 6 down to 1.

---

## 5. R31, the last commit, and why it is worth reading

`scripts/score_anchor.py`'s verdict rule was rewritten. The CODE REVIEWER flagged that R30's interval
rule addressed only the pass side, so I asked the planner **before the gate ran** rather than after it
failed. The planner conceded the asymmetry as its own drafting error and then rejected all three
fixes I proposed, on arithmetic:

> At n=1319 with 10–20% discordance the paired SE is 0.9–1.2 pp, so the CI is roughly 3.6–4.8 pp wide
> against a band that is 4.0 pp wide. Requiring the whole CI inside the band means PASS needs
> `|point| + half-width ≤ 2.0`, i.e. `|point|` under about 0.2 pp. **A gate that always defers to a
> human is not a gate.**

The threshold and the achievable resolution were never matched, and that is the real defect the
asymmetry example exposed. The registered rule is now **PASS** when `|point| ≤ 2.0 pp`, **FAIL**
otherwise, and **INCONCLUSIVE** reserved for a run whose resolution cannot see gross error at all.
Both non-pass verdicts **block** `main-v1`, which is what makes symmetry free: being honest about the
evidence costs nothing in safety, and the two verdicts differ only in what to do next. R30 said
"investigate" and the planner recorded that as its own gap.

A PASS must also **state its resolution**, since certifying near-equality to 2 pp was never achievable
at this n and pretending otherwise is exactly what H7 criticises.

⚠️ **One encoding decision in `verdict_for` is mine rather than the ruling's, and it is commented as
such in the code.** R31's prose says INCONCLUSIVE covers a CI that "cannot distinguish `≤ 2.0` from
`≥ 4.0`". My first encoding read that as "the interval overlaps the band **and** reaches beyond twice
it", which fired on R31's own FAIL example (−2.1, CI [−4.0, +0.5]) and would have made INCONCLUSIVE
the common outcome, defeating its purpose. Since R31 resolves that case as FAIL and calls INCONCLUSIVE
rare, the operative condition is the measurement's power: `half_width >= 2 × CRITERION_PP`. Verified
against all five worked examples R31 gives. If a later session disagrees with that reading, the place
to argue is the comment block in `verdict_for`, and it is a planner question.

R31 also ruled on Q2: one criterion serving both interval types is legitimate, because each is
compared to a fixed threshold rather than to each other, which was R26.1's actual objection. The
interval **type** is now recorded, since a bound says "cannot be outside" while a CI says "95%
coverage". And a **zero-variance result on the gate metric now raises rather than passing**, because
it would require both checkpoints to agree on every item, which is far more likely a wiring error with
both sides resolved to one path than two independently quantized models agreeing perfectly.

---

## 6. What is still open, and what needs the user

### Needs authorisation before anything runs

1. **Anchor A smoke, then the real run.** Roughly 3 hours and about $9. The free preflight has passed,
   so the remaining risk is the accuracy comparison itself. This **blocks `main-v1`** as gate G5a.
2. **Reference smoke, then `main-v1`.** Roughly 106,000 generations, about 17.5 hours, about $55. This
   is the **pre-registration freeze line**: `analysis/preregistration.md` §1 through §9 may be edited
   freely until the first `main-v1` generation exists, and is append-only from that job onward. K=2
   verification on the 10 invocation-floor cells is still outstanding.

Order matters. Anchor A is cheap and blocking, so it goes first.

### Built but not yet exercised on GPU

Two `sm_verify.py` fixes from `5ada832` remain unexercised: the `ctypes.CDLL` probe that replaced a
wrong `import vllm._C`, and the `importlib.metadata` version fallback. Fold a re-verify into the next
serving-image job rather than spending a queue slot on `verify` alone.

### Not built at all

- **Log-likelihood arms do not exist.** This blocks H6 arms 2 and 3 and G3's ARC cross-validation.
  R24.2 says build them after `main-v1` and before G5.
- **The judge and annotation track is absent**: A10, A11, A12 and B5. **B5's rubric needs the
  co-author**, so it is not unblockable by me.
- **A14** pipeline noise floor, **A15** budget sweep (about 25,000 uncosted generations) and **A18**
  zero-shot are all unbuilt.

### Paper

Seven of eight pages. **Introduction, Discussion, Conclusion, Limitations and Ethics are all empty.**
An empty Limitations section is a desk reject, and `make g6` reported PASS over it until
`scripts/check_sections.py` was added on the previous branch.

### Declined to the planner, still unresolved

Whether Anchor A and gate G5a belong in the pre-registration's §9e provenance table.

### 🚨 User-owned and hard-dated

**Oct 14 ARR reviewer and AC registration for both authors.** Missing it is an automatic desk reject.
ARR deadline is **2026-10-12, 23:59 AoE**.

---

## 7. Process rules this branch added, and why each one exists

These are in `CLAUDE.md`'s new standing header and in
`~/.claude/projects/.../memory/ask-planner-and-reviewer.md`. Every one of them came from a mistake
made on this branch, so a session that ignores them will most likely reproduce the mistake.

**Always say "CODE REVIEWER", never "the reviewer."** Shortened, the two advisor agents blur, and an
**ARR reviewer** is a third thing entirely, a person with an October obligation.

**Call the planner and the CODE REVIEWER automatically. Do not announce that they need calling.** The
user said this after I did it four times in one session: *"the whole point is for you to call them
automatically without telling me and not calling."* Calling either agent needs no approval, since
D10's confirmation gate is about billable compute and not about asking a question. Banned phrasings,
each of which I wrote instead of making the call: "this goes to the planner", "the CODE REVIEWER is
owed a pass", "flagged for the planner", "next: ask the planner about X".

**Asking is not enough, wait for the answer.** The analysis layer was written, committed and reported
to the user as built, and only then was the CODE REVIEWER consulted. The call was genuinely made and
still in flight when the work was presented as finished. Three consecutive rounds each found a
CRITICAL bug, so the interval between "asked" and "heard back" is where the risk lives.

**The decisive test for which agent owns a question**, because the hard cases look like code and are
science. Could a planner ruling or the pre-registration overturn the answer? Then it is science,
whatever surfaced it. Does the answer change a number or a claim that reaches the paper? Then it is
science, even when the fix is one line. What surfaced a question is not the owner of it.

**Both agents can be wrong.** R24.1 asserted a fact about our own data that was false, and R26.1
withdrew the planner's own framing. Verify factual claims about our artifacts before acting on them.

**🚨 NEVER ASSERT A STATE YOU HAVE NOT SEEN.** The worst mistake on this branch: I told the user "505
tests, ruff clean" **while the suite was red**. Two causes compounded. I ran `pytest ; echo exit=$?`
and then `git commit` chained with `;` rather than `&&`, so the commit ran regardless of the result
and the failure scrolled past. And a scripted `str.replace` silently did not match, because ruff had
already reordered the lines it was looking for. Commit `bffda77`, titled "ruff clean across the repo",
left ruff dirty. The three mitigations: chain with `&&` and never `;`, print a verdict rather than a
stream, and verify that a scripted edit actually landed.

**MINOR review items are listed and not actioned**, by the user's instruction of 2026-09-01.

**Write concisely but in full sentences, never telegraphic shorthand**, and **never use an em dash**.
The user rejected an earlier scope audit specifically because it *"has been shortened quite a lot to
the point where it is hard to understand and seems shorthand."*

---

## 8. The recurring bug family, now at more than ten instances

Worth stating plainly because it keeps costing real runs: **a cached or resumed unit of work must be
keyed on everything that determines what it produces.** The root fix is
`decoding_fingerprint()` in `src/quantartifact/generate.py`, appended to `cell_tag`, and
`DECODING_VERSION` is now **3** after `max_num_seqs` turned out to be missing from it, which was the
ninth instance. `env_hash` is *not* this: it covers `{gpu, cuda}` only.

A second shape has now appeared four times: **a guard is not a guard if the caller may pre-check its
condition.** `holm_block`'s status deriving from emptiness rather than from `m` is the latest, and it
surfaced inside the fix for its own previous appearance.

---

## 9. Constraints that have not changed and must not be relaxed

- 🚨 **No job starts without explicit user confirmation** (D10). Every billable command needs
  `--i-have-confirmation` and offers a zero-cost inspection mode.
- 🚨 **`--smoke` first, every time**, before any real submission.
- **`QA` is project initials and is NEVER expanded** (D11), not in a name, tag, S3 key or job id.
- **SMInterface is the only compute path** (D14). `scripts/box_*.sh` are provenance only.
- **`distributed` must stay `false`.** `tensor_parallel_size > 1` breaks batch invariance
  (vLLM #51290) and is the single most important determinism setting in the study.
- **`max_num_seqs` is frozen at 32**, moved from 64 by R24.1 when R15.6's escalation trigger fired.
- **`HF_TOKEN` lives in `~/workspaces/MedLLMSafety/.env` and is never copied into this repo.**
- **Read 3 raw generations after every prediction job**, before scoring or reporting any number.
- **Never delete anything of the user's without asking**, and nothing in this repo is pushed anywhere.
- **B2 (OpenReview) and B4 (external readers) are the user's, settled and closed** (D12). Do not
  re-raise them.
- **Do not redistribute xVerify weights or derivatives** (CC-BY-NC-ND). Qwen2.5-3B is
  `license:other`/qwen-research, not Apache-2.0.
- **Do not invoke the shell's `mwinit` alias** on this machine, it pipes a hardcoded password. The
  user runs `mwinit -s` themselves.
