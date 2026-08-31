# Scope audit — every task in `EXECUTOR-TASKS.md` §6 against evidence on disk

**Run 2026-08-31**, at the user's request (*"the Research and delivery tracks have many things
missing"*). They were right.

**How this was checked.** For each task, the artifact it names as its *Done when:* criterion or
its `→` output path was looked for on disk. A task is ✅ only if that artifact exists. "Code
exists that could produce it" is 🟡, never ✅. This is deliberately stricter than
`notes/gates.md`, which tracks *gates*; several gates are green while tasks underneath them are
not.

**Structure.** Each entry restates the specification from `EXECUTOR-TASKS.md` §6 in full, then
records what exists, what is missing, and what the gap costs. The specification is restated rather
than cross-referenced because the whole failure this audit found is that compressed summaries lose
the acceptance criteria — and a criterion nobody can read is a criterion nobody checks.

---

## Summary

| | count | tasks |
|---|---|---|
| ✅ done | 11 | `B1` `B3` `A1` `A2` `A3` `A4` `A5` `A7` `C1` `C3` + G4 |
| 🟡 partial | 8 | `B2` `B4` `A6` `A8` `A13` `A16/A17` `A20` `A21` `C2` `C7` |
| ❌ not started | 12 | `B5` `A9` `A10` `A11` `A12` `A14` `A15` `A18` `A19` `A23` `A24` `C4` `C5` `C6` |

**The single-sentence version.** The compute track has run far ahead of the delivery track: 33 GPU
jobs and $52.77 have produced a complete, mechanistically-pinned H8, nine quantized checkpoints, a
scored pilot and the template factor — while the entire statistical layer, the entire
judge-and-annotation track, and five of eight required paper sections do not exist in any form.

**The consequence for sequencing, stated up front because it reverses the current plan.** The held
`main-v1` reference run is **not** the critical path. Even if it completed tonight, the paper could
not be assembled: there is no code that computes H3's interaction, H4's decomposition or the
always-compliant stratum, and no judge to populate rung 4 of a four-rung ladder. Finishing
generation would produce data nothing can analyse.

---

## 🚨 Finding 1 — gate G6 was blind to five empty sections

This is recorded first because it is the only finding that was actively reporting success.

**What G6 checked.** `paper/Makefile`'s `g6` target verified four criteria: the PDF compiles, it
is ≤8 pages, a Limitations section is present, and `\todo{}` markers appear only inside Results.
The Limitations check was:

```make
@grep -q 'section\*{Limitations}' $(MAIN).tex && echo "  PASS" || { echo "  FAIL"; exit 1; }
```

**What that actually tests.** That the *heading string* exists in the file. Nothing about content.

**What was found.** Five required sections carried a heading, a `\label`, some `%` planning
comments, and **zero lines of prose**:

| section | prose lines | due |
|---|---|---|
| Introduction | **0** | `C4`, W4 |
| Discussion | **0** | `C7`, W5 |
| Conclusion | **0** | `C7`, W5 |
| **Limitations** | **0** | `C5`, W4 |
| Ethics Statement | **0** | `C5`, W4 |
| Appendices ×3 | 0 | permitted — `A23` requires appendices carry no load-bearing content |

`make -C paper g6` reported **PASS on all four criteria** in this state.

**Why it matters.** `A23` lists a missing Limitations section as an **automatic desk reject**, and
notes it costs zero page budget — so there is no trade-off to think about. An empty Limitations
section is the same desk reject wearing a heading. The gate that exists specifically to catch
desk-reject grounds could not distinguish the two.

**This is the project's own recurring failure, one level up.** The RUNLOG records eight instances
of *declared versus enforced* — a value stated in a config and never forwarded, or forwarded and
then discarded by a cache that could not tell the difference. This is the ninth in spirit and the
first at the level of the **instrument that checks the deliverable**: the criterion was declared
in `A23`, and the check enforced something weaker that happened to pass.

*Fixed:* `scripts/check_sections.py`, wired into the `g6` target. It counts prose per required
section, treating `%` comments and bare `\todo{}` as not-prose and structural markup (`\begin`,
`\caption`, table rules) as not-prose, so a section containing only a table does not read as
written. Parent sections count their subsections' prose, so Method legitimately passes on its five
written subsections while its own body is empty. **G6 now fails, correctly.**

---

## 🚨 Finding 2 — there is no analysis code at all

**What exists.** `src/quantartifact/analysis/` contains exactly one file: `__init__.py`.

**What `A19` specifies.** Per `EXECUTOR-TASKS.md` §6 and `EXECUTOR-PLAN.md` §4, with §4.0's
identification fix mandatory and already pre-registered:

1. The **primary competence contrast on the always-compliant stratum only** — items where every
   contrasted arm produced a committed answer, a pre-treatment-defined subgroup so the contrast is
   identified. The stratum's size is a first-class reported number.
2. The **2×2 compliance-transition matrix** per precision pair (committed→committed,
   committed→abstained, abstained→committed, abstained→abstained), with the off-diagonal asymmetry
   and an exact McNemar test on the discordant cells. This is a finding, not a nuisance.
3. **Manski bounds** on the full-sample competence effect. Wide bounds are an honest result.
4. The **H3 model**: `correct ~ precision * scoring + model + (1 | item)`, fit per task, with
   model as a **fixed** effect because three levels cannot support a variance estimate. The
   interaction reported as an odds ratio *and* as a marginal effect in accuracy points with a 95%
   CI, plus **paired McNemar** on discordant pairs as an assumption-light companion.
5. **Holm–Bonferroni within each hypothesis family**, one pre-specified omnibus contrast per
   hypothesis, effect sizes with CIs everywhere and never a bare significance flag.

**What is missing.** All five. `statsmodels>=0.14` and `scipy>=1.13` are declared in
`pyproject.toml` and imported by nothing.

**Why it matters.** H3, H4 and H7 are three of the eight hypotheses, H4 is *"the centrepiece
table"*, and §4.0's identification fix is the section `EXECUTOR-PLAN.md` says caps acceptance at
~25% if a reviewer sees it was patched rather than pre-specified. The prereg for all of it is
written and detailed; the code that would honour it does not exist. This is the single largest
gap in the project.

---

## Track B — people and external latency

`EXECUTOR-TASKS.md` §4: *"Track B and C are things that will silently slip if they are treated as
later phases."* Each Track B item has multi-day external latency.

#### `B1` HF token + Llama licence grant — ✅ DONE

**Specification.** Install the HF CLI, authenticate with a `read`-scope token, confirm with
`hf auth whoami`, then `hf download meta-llama/Llama-3.1-8B-Instruct --include config.json`. On
401/403, request the licence in a browser the same hour and proceed with Qwen + Phi; if not granted
by **Sep 1**, execute the O-2.2 substitution to `Qwen2.5-7B-Instruct` without asking.

*Done when:* `config.json` downloads, confirming both the token and the licence grant; the token is
exported as an environment variable on the box and **never** committed to the repo; recorded in
`notes/access.md`.

→ `notes/access.md` — **exists.**

**Status.** Llama-3.1-8B `config.json` downloaded, revision pinned
`0e9e39f249a16976918f6564b8830bc894c89659`. The Sep 1 substitution is void. `HF_TOKEN` lives in
`~/workspaces/MedLLMSafety/.env` and is exported at submit time only.

#### `B2` ARR obligations + AWS — 🟡 PARTIAL, and the outstanding half is user-owned

**Specification.** Verify **both** authors' OpenReview profiles are complete: affiliation,
Semantic Scholar / DBLP / ACL Anthology links, working email. Set an **Oct 14** reminder for both
people. Separately run `aws sts get-caller-identity` and one GPU vCPU quota query — read-only, two
minutes, purely to document what was declined per O4.

*Done when:* both profiles verified complete, both reminders set, quota result recorded.

→ `notes/arr-obligations.md` — **exists.**

**What is missing.** 🚨 **I cannot verify either OpenReview profile or either calendar reminder.**
These are the two items whose failure mode is an **automatic desk reject**: reviewer/AC
registration is per-author and must happen within 48 h of submission, and an incomplete profile
blocks registration. The AWS half is done and documented.

**Consequence.** No amount of work on my side substitutes for this. It is listed again under
"user-owned" at the end.

#### `B3` GPU account + working payment — ✅ DONE

**Specification.** Provider account, working payment method, and one successful five-minute test
pod that runs `nvidia-smi`.

*Done when:* a pod has been launched and destroyed successfully.

→ `notes/compute.md` — **exists.**

**Status.** Closed differently from the specification and deliberately so: rather than launch and
destroy a pod, `B3` was closed on verified quota, verified pricing and verified AZ availability,
because launching anything at that point would have violated D10 (no job without explicit
confirmation) and D11 (reviewed resource names). Substrate has since been proven end-to-end by 33
real jobs, which is strictly stronger evidence than a test pod.

#### `B4` External reader — 🟡 PARTIAL

**Specification.** Identify and ask **two** candidates (one will decline) who know quantization
*or* evaluation methodology and are not the co-author. Tell them now: **draft arrives Sep 30,
feedback needed by Oct 4**, and the ask is a targeted attack on related-work differentiation and
judge validation, not a full read.

*Done when:* one has confirmed in writing.

→ `notes/review-plan.md` — **exists**; D12 records `B4` as closed and settled.

**What remains.** The dependency, not the recruitment: a draft has to reach them by **Sep 30**,
and on current state five sections are empty. The reader cannot attack judge validation because
there is no judge (`A10`).

#### `B5` Annotation rubric — ❌ NOT STARTED, and it blocks the schedule's tightest constraint

**Specification.** Day 2–4, **needs the co-author.** Draft the rubric; dry-run ~10 items
*together*; freeze it. It must give an explicit decision rule for every edge case: truncated
output, multiple candidate answers, correct-but-never-stated, refusals, degenerate repetition.
Rubric drift between raters is the main threat to a two-rater design. **Pre-specify the
adjudication rule before seeing any disagreements** — never resolve by deferring to the senior
author (Belz et al., Insights 2023, list biased aggregation as a documented reproducibility
failure).

*Done when:* frozen, committed verbatim, and both annotators have signed off on the 10-item dry
run.

→ `annotation/rubric.md` (published verbatim in the appendix and artifact) — **does not exist.**
The `annotation/` directory is empty.

**Consequence.** `A11` (human annotation) cannot start, and `A11` is described in TASKS as *"the
schedule's tightest constraint"* because it needs a second person's calendar. `A12` (judge
validation) then cannot start either. The adjudication rule *is* already written — it is in
`analysis/preregistration.md` §10 — but the rubric itself, the edge-case decision rules, and the
co-author's sign-off on a dry run are not.

**Why this should start before anything that only needs me.** It is the only remaining item with
another person's latency in it.

---

## Track A — W1 (foundation, determinism, pilot)

#### `A1` Environment — ✅ DONE

**Specification.** Install `uv`; create the two virtual environments per §3.2 with everything
pinned; commit the lock file. Also report the top local disk consumers to the user and **let them
decide** what to remove — 7.6 GB is tight for even a minimal analysis env. Do not delete anything
on the user's machine.

*Done when:* both venvs resolve, the lock is committed, and
`python -c "import vllm, math_verify, lm_eval"` passes in the serving env.

**Status.** `.venv` (analysis, Python 3.12) and `.venv-harness` (lm-eval + transformers) both
resolve; pinning is in `pyproject.toml`. ⚠️ The literal *Done when* import check cannot pass on
this machine and was never meant to: `vllm` has no macOS build (its CPU backend is x86-only), so
the serving environment exists only inside the container. That is the intended architecture, not a
gap — but note `math_verify` is *also* absent locally, and unlike `vllm` that one matters (see
`A16/A17`).

#### `A2` Repo + paper skeleton — ✅ DONE

**Specification.** Package layout per `EXECUTOR-PLAN.md` §5. Create the paper project from the
current ACL template and commit a compiling stub with every section header present and empty.

*Done when:* `paper/main.pdf` compiles from the stub.

**Status.** Compiles with `tectonic` at 7 pages; every §7 section header present. Overleaf was
dropped in favour of local `tectonic` (documented in `notes/paper-toolchain.md`), which removes an
external dependency from G6. ⚠️ Note that "every section header present and empty" was satisfied
literally, and five of them are **still** empty — see Finding 1.

#### `A3` Configs — ✅ DONE

**Specification.** `models.yaml` (repo IDs plus **commit SHAs, not tags**), `quant.yaml`
(method/bits/calibration set/size/seqlen/seed per cell, normalised per §2.5), `tasks.yaml` (item
counts, subsample seeds, prompt templates, the three resolved token-budget levels),
`scoring.yaml` (the four rungs declaratively).

*Done when:* all four exist, every model pinned by SHA, the GSM8K 500-item subsample verified to be
a **prefix-nested** subset of the 1319, and the three budget levels recorded as absolute integers.

**Status.** All four exist. Models pinned by SHA. Budget levels recorded as absolute integers
(GSM8K 256/512/1024, BBH 1024/2048/4096, ARC-chat 100/200/400). ⚠️ Recorded but **not consumed** —
see `A15`.

#### `A4` Generation core — ✅ DONE

**Specification.** `generate.py`, the only component that touches a GPU for generation. Emits the
§7.1 record for every generation. One vLLM engine per `(model, precision)`. All §3.3 settings
applied and asserted at startup, failing loudly if `VLLM_BATCH_INVARIANT` is unset or
`tensor_parallel_size != 1`. Append-only gzipped JSONL, one file per cell. Resumable: re-running a
partially-complete cell must skip already-generated `(item_id, sample_idx, budget_level)` keys.

*Done when:* a 20-item run produces valid JSONL where **every** record has non-null
`finish_reason` and `n_gen_tokens`, and a re-run of the same command generates zero new records.

**Status.** Satisfied and then hardened well past the specification: batch invariance is asserted
*engaged* rather than merely requested (`assert_batch_invariance_engaged`), the engine's reported
state is read back and compared against what was asked for (`read_back_engine_state`), prompts are
passed as frozen `prompt_token_ids` so vLLM cannot re-tokenize and double-BOS, and
`decoding_fingerprint()` now hashes everything determining what a generation is so a config change
invalidates cached cells automatically.

⚠️ **The resumability clause is the origin of the project's most expensive bug family.** "Skip
already-generated keys" was implemented, but the *key* repeatedly omitted something that changes
the output — nine instances now, most recently `max_num_seqs`, which our own H8 result shows moves
generated text by up to 59%.

#### `A5` Scorers, rungs 1–3 — ✅ DONE (gate G3 partial, see `A16/A17`)

**Specification.** `scoring/extract.py`, pure functions, no I/O. Rung 1 = the harness regexes
**verbatim**. Rung 2 = the documented pre-registered permissive parser (last-number-in-string,
`\boxed{}`, "answer is X", comma/currency/unit stripping, Minerva `normalize_final_answer`).
Rung 3 = `math-verify` 0.9.0, with the warning that `parse()`/`verify()` use `signal.alarm()` and
raise in worker threads, so pass `parsing_timeout=None, timeout_seconds=None` and impose your own
timeout; it is intentionally **asymmetric**, so gold/target order matters. Also
`scoring/compliance.py` for the §2.3 2×2 and the H2 trace-scan of §7.4.

*Done when:* **G3** — unit tests cover `####`-present, `####`-absent, boxed answer,
comma-thousands, currency, negative number, trailing period, "The answer is" **capitalized** (must
fail rung 1 for BBH — assert this explicitly, it is a finding), truncated mid-answer, degenerate
repetition, empty generation, multiple candidate answers, and a `math-verify` timeout path. Rungs
asserted **monotone** on a fixture set.

**Status.** 340 tests green, every named fixture covered, monotonicity asserted, and two harness
defects pinned as tests rather than patched (BBH's `get-answer` dropping the final character;
`gsm8k_cot`'s strict regex capturing the sentence period). The scorer code is the strongest part of
the codebase.

#### `A6` Smoke + tight estimate — 🟡 PARTIAL, and the missing half is a paper number

**Specification.** One cell end-to-end (Qwen2.5-3B, bf16, GSM8K 100 items). Then
`vllm bench throughput` **on the actual instance**, because published estimates are ±40% and no
public measured figure exists for a 3B model.

*Done when:* measured generations/hour recorded **with and without** `VLLM_BATCH_INVARIANT=1`
(*"this is also the honest BI-overhead number for the paper"*), and the §2.6 cost table re-derived
from measurements and committed.

→ `notes/estimates.md` — **does not exist.** The measurements live in `notes/RUNLOG.md` and
`scripts/project_budget.py` instead.

**What exists.** Real per-task rates, and they corrected an earlier figure that was badly wrong:
the first "24,800 gen/h" was measured while BBH's terminators were emitting almost nothing, so most
of those "generations" were 1–9 tokens. Honest rates are **BBH 7,099 gen/h, gsm8k 5,250, ARC
effectively free**, with **generation only ~25% of billable time** and **2.6 min/cell of
engine-load overhead** dominating — which makes *cell count*, not item count, the first-class cost
driver. The budget projection is re-derived in `scripts/project_budget.py`.

**What is missing.** ⚠️ **The with-versus-without `VLLM_BATCH_INVARIANT` pairing has never been
run.** That pairing is not a diagnostic — it is the batch-invariance overhead number the paper
reports, and §3.3's 1.2–2× figure is an **Ampere** number that does not transfer to our Ada L40S.
It requires one cheap job.

#### `A7` Determinism audit (H8) — ✅ DONE, gate G4 PASSED

**Specification.** Quantize Qwen2.5-3B into all four quantized formats. Run vLLM's own
`tests/v1/determinism/test_batch_invariance.py` per format. Rerun an identical 500-item probe
(GSM8K 200 + BBH 200 + ARC-chat 100) **3×** per format and measure the bitwise disagreement rate.
If GPTQ is nondeterministic while bf16 is not, **differential numerical noise exists between arms
and is a floor on every claim in the paper.** Workaround to test if so:
`VLLM_DISABLED_KERNELS=MarlinLinearKernel,MacheteLinearKernel`. Extend to Llama/Phi × {bf16,
GPTQ-4} in W2 as confirmation.

*Done when:* **G4** — a committed table of per-format bitwise disagreement rate over 3 repeats, the
vLLM test-suite pass/fail per format, and the `VLLM_DISABLED_KERNELS` result where relevant.

→ `artifacts/tables/determinism.csv` + `notes/determinism.md` — ⚠️ **neither exists by those
names.** Content is in `notes/RUNLOG.md`, `artifacts/determinism*.json` and
`paper/tables/h8_*.tex`.

**Status.** Substantively complete and well beyond the specification. The original pass criterion
was **retired before it could pass** (R15.4): three repeats in an identical order gave 0/20
disagreement for every arm, i.e. the criterion measured an axis with no variance, while three
*reorderings* changed 65–75% of both quantized arms' outputs and none of bf16's. Replaced with
order-induced divergence swept at {8, 32, 64} across three metric levels, then extended to ten
orderings for a stable SD.

Delivered: `bf16` **exactly 0.000%** at every level and every metric level; the mechanism **pinned
by intervention** to a named kernel (gptq-4bit 53.2→0.0%, awq-4bit 53.6→0.0% with Marlin
disabled); a **second, distinct, Marlin-independent mechanism** in bnb-NF4 that behaves oppositely;
autotune ruled out empirically; generality confirmed on Phi.

⚠️ Two clauses of the specification were **not** done: vLLM's own
`test_batch_invariance.py` was never run per format (we asserted engagement directly instead), and
the two named output files do not exist. The second is cosmetic; the first is a cheap independent
check we skipped.

#### `A8` Stratification pilot — 🟡 PARTIAL, and the missing half is half the task

**Specification.** All 15 cells × 300 items (GSM8K 100 + BBH 120 + ARC-chat 80), using the **final
frozen** prompt templates and budgets — *"if templates are not frozen, the annotation population is
wrong."* One run serves three purposes: the annotation source pool (O11), a throughput measurement
on real cells, and a noise-floor smoke test.

*Done when:* 4,500 generations stored **and scored at rungs 1–3**, and the stratified 250-item
annotation sample **drawn and written blinded** per `A11`.

**What exists.** `run_id=pilot-v2`, 17 cells × 300 = 5,100 generations (17 not 15, because the two
AWQ reproducibility replicate cells were added), scored at rungs 1–2. Throughput measured.

**What is missing.**
- ⚠️ **The stratified 250-item blinded annotation sample was never drawn.** That is explicitly half
  of the *Done when*, and it is the artifact `A11` consumes.
- Rung 3 was not applied — `math-verify` is not installed (see `A16/A17`).
- ⚠️ The templates were **not** final when `pilot-v2` ran: the freeze has since moved from
  `max_num_seqs` 64 to 32 (R24.1). The planner ruled the pool need not be re-run (R11/O11: judge
  validation is a claim about the *population* of generations, not specific strings), so this is
  recorded rather than repaired — but it must be *stated in the paper*, as O11 requires.

---

## Track A — W2 (anchors, judge, annotation, main runs)

#### `A9` Anchor validation — ❌ NOT STARTED, and Anchor A is *blocking*

**Specification.** The retargeted replication gate (override O3 — Kurtic was demoted from primary
gate to stretch because it needs gated access and can absorb a week).

- **Anchor A — mandatory, ungated, BLOCKING.** Reproduce the official
  `Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4` / `-GPTQ-Int8` / `-AWQ` checkpoints' behaviour and compare
  against our self-quantized equivalents at matched settings (`group_size:128, desc_act:false`).
  *Pass:* our self-quantized GSM8K accuracy is within **±2.0 pp** of the official checkpoint's at
  the same precision. **A larger gap means our calibration or serving config is wrong — debug
  before proceeding.**
- **Anchor B — mandatory if Llama access exists, non-blocking.**
  `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16` publishes measured accuracies (GSM8K
  82.8→82.9, ARC-C 81.4→80.2, BBH 30.1→28.9). *Pass:* our deltas land near theirs.
- **Anchor C — secondary, strategically valuable.** Reproduce the strict-vs-flexible GSM8K gap from
  arXiv:2601.14277 Appendix B. *"If we reproduce their gap, our pipeline is validated on exactly
  the phenomenon the paper is about."*
- **Anchor D — stretch only.** Kurtic et al.

*Done when:* **G5a** — a committed table of anchor / published value / our value / delta / verdict.

→ `artifacts/tables/anchors.csv` — **does not exist.**

**What is missing.** All of it, as a run. Anchor A has **never been executed**, and it is the one
labelled *mandatory, ungated, blocking* — its entire function is to catch a wrong calibration or
serving configuration **before** the expensive runs. `notes/gates.md` currently shows G5a as
"🟡 unblocked — 9 quantized checkpoints exist", which conflates *having the checkpoints* with
*having validated them*.

Anchor B has been verified **at source** (`notes/protocol-survey.md` confirms Red Hat's published
w4a16 GSM8K delta of +0.1) but not reproduced by us. Anchor C was **scoped out** on evidence: R19.3
established it is llama.cpp GGUF on CPU, not vLLM/GPTQ, so only its *extraction gap* is
reproducible and its quantized deltas are not comparable.

**Consequence.** This is the cheapest insurance in the project and it is unbought. We are holding a
$33 reference run behind a prereg question while the check designed to catch a wrong calibration
has never run. It is also directly on the critical path for a reviewer question: nine checkpoints
were built with a specific `actorder`/`desc_act` configuration whose default we already found
silently wrong once.

#### `A10` Judge harness — ❌ NOT STARTED

**Specification.** Primary judge `IAAR-Shanghai/xVerify-9B-I`, purpose-built for
`{question, llm_output, correct_answer}` → binary; beats GPT-4o at 97.19 vs 96.25 F1; family-clean
with respect to Qwen/Phi/Llama. ⚠️ **CC-BY-NC-ND-4.0 — run it and report numbers, but never
redistribute the weights or any derivative in our artifact.** Second judge
`ibm-granite/granite-4.1-8b` (Apache-2.0), the cleanest licence for artifact release.

Use CompassVerifier's **3-way label space** (`A`/`B`/`C` with an explicit `INVALID` class) rather
than pure binary, because our 3B models under 4-bit *will* emit truncated and repetitive output and
`INVALID` maps directly onto `ABSENT`/`TRUNCATED`. Borrow its *"compare **ONLY** the final answer"*
instruction. Always supply the reference answer — reference-guided judging cuts GPT-4's math-grading
failure rate from 70% to 15%. Greedy, seed fixed, single-token verdict.

*Done when:* the exact prompt template is validated on ~50 items **before scaling**, and the
judge's own **output-parse rate** is measured — VerifyBench found xVerify failing to emit its
required format under a foreign template, *"so this check is not optional."*

→ `src/quantartifact/scoring/judge.py` — **does not exist.**

**Consequence.** Rung 4 of a declared four-rung ladder is absent. This is survivable by design —
§2.2 makes the headline the H3 interaction at **rung 1 vs rung 2**, deterministic and independent
of the judge, which is exactly why `A12`'s gate is non-blocking. But the ladder is described as
four rungs in the Method section that is already written.

#### `A11` Human annotation window — ❌ NOT STARTED

**Specification.** ≥250 items stratified across 3 models × {bf16, 4-bit} × all tasks,
**oversampling rung-1/2/3 disagreement cases** (agreement only matters where scorers disagree;
also report agreement on the natural distribution). ≥120 double-annotated for IAA; split the
remainder to reach 250; report per-annotator counts.

**Blinding is load-bearing, not hygiene.** **Omit the condition column from the annotation file
entirely** rather than hiding it — a hidden column can be un-hidden. Annotators must never see the
judge's verdict, the parser's verdict, or the precision condition. Pre-shuffle **independently per
annotator**, assert the two orderings actually differ, publish the seeds. One binary question per
screen. A ~10-item blind practice round instead of attention checks, because excluding
check-failures yields a biased subsample (van der Lee et al., INLG 2019).

Justification to cite: Hróbjartsson et al. (*BMJ* 2012;344:e1119) — across 21 trials with binary
subjective outcomes, non-blinded assessors exaggerated odds ratios by **36%**, and the entire bias
came from misclassifying a median of **3%** of items while agreeing on 78%. *"Our effects are small
enough that a bias hiding behind high agreement could manufacture the whole finding."*

**Never pre-annotate with the judge and have humans correct it.** Anchoring is measured (Fort &
Sagot, LAW IV 2010: the better annotator dropped 98.1→95.8 under low-quality pre-annotation). Note
the trap: Mikulová et al. (LREC 2022) found pre-annotation *raised* IAA — **higher agreement under
pre-annotation is shared anchoring, not higher validity**, and IAA is a headline number.

Tooling: a ~50-line local script is genuinely the right answer for 250 binary labels.

*Done when:* 250 adjudicated labels, IAA reported on the overlap subset, the number of adjudicated
items reported, and the annotation design disclosed in Responsible NLP §D.

→ `annotation/labels.jsonl`, `annotation/iaa.md` — **neither exists.**

**Blocked on.** `B5` (rubric, needs co-author) and `A8`'s undrawn 250-item blinded sample.

**Consequence.** TASKS calls this *"the schedule's tightest constraint"* because it needs a second
person's calendar in a fixed window. Nothing about it has started.

#### `A12` Judge validation — ❌ NOT STARTED

**Specification.** Report **all** of: raw agreement with a **Wilson 95% CI**, Cohen's κ, **Gwet's
AC1** (the right primary statistic here), and the confusion matrix with FP and FN **broken out
separately**. Do **not** report κ alone — these labels are heavily skewed and a judge at 95% raw
agreement can post κ≈0.3 purely from prevalence (the kappa paradox: Feinstein & Cicchetti 1990;
Byrt, Bishop & Carlin 1993).

**The check a good reviewer will ask for:** report judge error rates **separately for bf16 vs
4-bit generations.** A judge that is lenient toward *quantized* outputs specifically would
manufacture our result.

*Gate:* proceed with the judge as a primary rung if raw agreement ≥0.90 **and** AC1 ≥0.7 **and**
the bf16-vs-4-bit error rates are not significantly different. Report IAA alongside, because a
judge cannot be held to better agreement than the humans achieve with each other.

*If it fails:* report rungs 1–3 as primary and the judge as a caveated robustness check. **This
keeps the paper intact and is therefore NOT a blocking report-back.**

→ `artifacts/tables/judge_validation.csv` — **does not exist.**

**Blocked on.** `A10` and `A11`.

#### `A13` Main reference runs — 🟡 WRITTEN, DELIBERATELY HELD

**Specification.** 15 cells × 2,719 items. **Blocked until the pre-registration is committed.**
Score continuously as generations land — do not batch scoring to the end.

**Status.** `sm_reference.py` written, `reference-smoke` passed ($0.91). **Held**, for three
reasons, in order of how hard they are to fix:

1. It is the **pre-registration freeze line** (§9f), and an open escalation would be resolved
   *after* data existed if it ran now.
2. **K=2 repetition on 10 cells is not implemented** (§9f registers the estimator and it *depends*
   on those cells existing).
3. **`invocation_id` per generation record is not implemented** (committed to in R21).

⚠️ And per Finding 2, completing it would produce data that no analysis code can consume.

#### `A14` Noise floor — ❌ NOT STARTED

**Specification.** §4.5, **mandatory.** Run the bf16 baseline **twice** end-to-end with different
seeds on every model. The spread between the two runs is the pipeline's own noise floor, and **any
effect smaller than that floor cannot be claimed by us either.** *"Reporting that is exactly the
self-application of the paper's own standard that earns reviewer trust."*

→ `artifacts/tables/noise_floor.csv` — **does not exist.**

**What might look like a substitute, and is not.** The cross-invocation work (§9f) measures
something adjacent and important — re-running the *same* configuration in a new job diverges on
41–45% of items for Llama, with a worst signed accuracy delta of 4.44 pp. But that is
**re-invocation at one fixed seed**, whereas `A14` is **two different seeds end-to-end**. Different
quantity, different claim. The prereg's noise-floor pre-commitment is written against three named
floors (repetition, reordering, pipeline) and *this* is the pipeline floor — the one §4.5 defines
and the one still unmeasured.

---

## Track A — W3–W4 (sweep, scoring, zero-shot, analysis)

#### `A15` Token-budget sweep — ❌ NOT STARTED, and it is the LEAD hypothesis

**Specification.** H1, *"the lead."* 9 cells × 1,400 items × {2×, 4×} — 1× comes free from the
reference run via the nested subsample. Follow arXiv:2608.04160's design: **freeze the levels
before looking at results**, hard-cap decodes at each level, and treat the cap as an **independent
variable** rather than an implementation detail.

*Done when:* every generation has `finish_reason` and `n_gen_tokens`, and truncation rate × budget
× precision is tabulated. Also settle the 2606.25519-vs-2504.04823 contradiction directly: report
mean and median `n_gen_tokens` by precision, with a CI. *"That is a citable contribution on its
own."*

**What exists.** The levels are frozen and recorded as absolute integers in `configs/tasks.yaml`
(GSM8K 256/512/1024; BBH 1024/2048/4096; ARC-chat 100/200/400), which satisfies the
freeze-before-results requirement. `generate.py` accepts `--budget {1x,2x,4x}`.

**What is missing.** ⚠️ **Nothing consumes 2× or 4×.** All three generation stages hardcode
`budget_level="1x"`; a repo-wide search finds `"2x"`/`"4x"` only in that one argparse choice list.
So H1 — the hypothesis TASKS marks *"Lead. Never cut."* — has **no instrument**.

**Consequence, including one nobody has costed.** This is ~9 cells × 1,400 items × 2 levels ≈
**25,000 generations** that appear in no current plan, no queue estimate and no budget line. At the
honest rates from `A6` and with 2.6 min/cell of overhead, it is real compute. It also has a
dependency the current plan hides: BBH at 4× is `max_gen_toks=4096`, and H1's budget axis on BBH
only became measurable at all *after* the terminator fix (§9b), because before it the stop sequence
bound long before the budget did.

#### `A16`/`A17` Score everything at all four rungs — 🟡 PARTIAL

**Specification.** Apply all four rungs plus compliance and trace-scan labels. Pure functions over
stored artifacts, no GPU. Scorer outputs go to separate files keyed by
`(cell, scorer_name, scorer_version)`; **bump the version on any change, never overwrite.**

**What exists.** `scripts/score_pilot.py`, `scripts/score_probe.py` (runs in `.venv-harness`) and
`scripts/score_template.py`. Rungs 1, 1b and 2 applied; compliance and termination labels applied.

**What is missing.**
- ⚠️ **Rung 3 has silently never run.** `math-verify` is not installed in `.venv`, so every scoring
  run to date printed `math_verify UNAVAILABLE` and degraded to three rungs. That degradation is
  deliberate and correct behaviour in the script — but it means a declared rung of the ladder has
  produced no numbers, and nobody noticed because the message scrolled past in a `tail`.
- Rung 4 needs `A10`.
- ⚠️ Outputs are **not** keyed by `(cell, scorer_name, scorer_version)`. They are written to
  `artifacts/scores/scores_<run_id>.json`, so **re-scoring overwrites** rather than versioning —
  the exact thing the specification says never to do, and the same cache-identity family as the
  nine generation-side instances.

#### `A18` Zero-shot arm — ❌ NOT STARTED

**Specification.** H5. 9 cells × 1,400 items. Per prereg §3d, everything except the prompt text is
held fixed — the zero-shot variant must reuse the **few-shot** `max_gen_toks` and `until`.

**What is missing.** `prompt_variant` is hardcoded `"fewshot_cot"` in all three generation stages.
Zero-shot prompt rendering was also listed as outstanding under G2 ("not yet done for G2"), so
neither the prompts nor the generation path exist.

#### `A19` Analysis — ❌ NOT STARTED

See **Finding 2** above for the full specification and consequence. This is the largest gap.

#### `A20` Powered-ness audit — 🟡 PARTIAL

**Specification.** H7. For each published degradation figure we cite, compute the 95% CI implied by
that paper's own eval size and state whether the reported delta clears it. Also compute our own
minimum detectable effect at 80% power. **Scrupulously neutral wording** — this is an observation
about statistical resolution, not an accusation. Frame as *"many papers report aggregate deltas
without a paired test"*, never as *"the deltas are noise."*

**What exists.** `notes/protocol-survey.md`, structured as a counted survey — *N reported of M
checked*, each verified at source, with "not reported" meaning *checked and absent* rather than
*not looked for*. Findings so far are substantive: Kurtic et al. report neither the chat-template
setting nor the extraction mode and released no eval code; Red Hat document theirs fully and their
model card records results revised *"after Meta modified the chat template"*, which is
**third-party evidence that the template moves published numbers**; Pape et al. is established as
the nearest prior work and differentiated.

**What is missing.** **4 of ~14** cited papers reporting a delta have been checked. R19.3 forbids
any counted field-level claim until M covers them, and directs us to cite Pape et al.'s
35,000-paper survey for the general claim instead. The CI arithmetic and our own MDE are not
computed — both need `A19`.

#### `A21` Figures and tables — 🟡 PARTIAL, and the centrepiece is missing

**Specification.** The §4.2 decomposition table is the paper's centrepiece — **design it first and
let it drive what gets logged.** Near-final before drafting starts.

**What exists.** A good pattern, established for H8: `paper/tables/h8_*.tex` are **generated** from
the job reports by `scripts/make_h8_tables.py`, with every statistic sourced from a file and none
inlined, so a re-run updates the paper mechanically. That pattern has already caught two
stale-number bugs.

**What is missing.**
- ⚠️ **The §4.2 decomposition table does not exist**, and the instruction was to design it
  *first*, precisely so it would drive what gets logged. It is being designed last, which is how a
  required field turns out to have gone unlogged.
- `artifacts/figures/` contains only `.gitkeep`. No figures at all.

---

## Track A — W7 (compliance, release)

#### `A23` Compliance sweep — ❌ NOT STARTED

**Specification.** Check against **every** desk-reject ground: LaTeX ACL template; ≤8 pages;
**Limitations section present** (absence = automatic desk reject, and it costs zero page budget);
complete **and correct** Responsible NLP Checklist — incorrect or incomplete filings have
themselves been a desk-reject ground since Dec 2024, and §C (compute / hyperparameters / error
bars) and §E (AI assistance) both apply to us; anonymized submission; no anonymous repo on Dropbox
or Google Drive (download tracking); appendices in double-column format carrying **no load-bearing
content**.

**arXiv preprinting is permitted** (no ARR anonymity period since Feb 2024). Do **not** elect the
optional binding no-preprint option unless the user asks.

**Status.** Template ✅, page count ✅ (7 of 8). **Limitations present but EMPTY** — see Finding 1;
G6 now catches it. **Responsible NLP Checklist not started.** Note §C requires compute and error
bars, which need `A19`, and §D requires the annotation-design disclosure, which needs `A11`.

⚠️ Worth correcting a claim I made earlier in this project: I flagged the ARR anonymity period as a
reason to delay filing an upstream bug report. `A23` states there has been **no ARR anonymity
period since Feb 2024**. The upstream-filing decision is still the user's, but it is not an
anonymity question.

#### `A24` Artifact release — ❌ NOT STARTED

**Specification.** Pipeline, quantization configs with full calibration details, all four scorers,
analysis scripts, **the full generation JSONL** (*"the genuinely reusable artifact — it lets anyone
re-score with a new extractor without a GPU"*), `pilot-v1` included, and the human-annotated
judge-validation set with its agreement statistics (which stands alone as a citable judge-reliability
reference for quantization eval). Pin model commit SHAs, every library version, calibration dataset
and counts, all decoding hyperparameters, GPU model, CUDA and driver versions.

🚨 **Do not redistribute xVerify weights or derivatives** (CC-BY-NC-ND). Note Qwen2.5-3B is
`license:other`/qwen-research, **not** Apache-2.0.

**Status.** Not assembled. Two constraints to carry forward: `pilot-v1` **must be retained** — it
is the evidence for the BBH terminator finding the paper reports in Method — and the "all four
scorers" clause currently cannot be met, because the fourth does not exist.

---

## Track C — writing

`EXECUTOR-TASKS.md` §4 calls Track C *"the counter-intuitive one"*: Related Work, Introduction
framing, Method, Limitations and the checklist depend on **zero results**, so drafting them in
weeks 1–2 costs almost nothing and removes the largest single rejection risk. *"Do not defer
writing to week 5."*

#### `C1` Related work — ✅ DONE

**Specification.** W1–W2, *"the highest-value writing task in the project."* Differentiate
precisely against every paper in `EXECUTOR-PLAN.md` §7 and §0.1, with a long list of required
positioning moves and **three claims the paper must never make** (first to combine an LLM judge
with quantization eval; first factorial quantization evaluation; first to show accuracy hides
behavioural divergence — each is already owned).

**Status.** 94 prose lines, 23 citations all fetched from the arXiv API rather than recalled, the
live CoT-length contradiction opened with, Kurtic cited early and prominently, Hua et al.'s MATH
null engaged honestly, and the nearest prior work (Pape et al.) differentiated on the specific
ground that they ran at batch size one *to prevent* the instability we make the object of study.

#### `C2` Pre-registration — 🟡 SUBSTANTIVE, correctly not yet frozen

**Specification.** **Commit before any main-run generation (hard, Sep 2.)** Must contain at
minimum: §4.0's identification fix in all three parts; the frozen token-budget levels as absolute
integers; the rung-2 permissive parser rules **enumerated**; the H3 model specification; the
omnibus contrast per hypothesis and the Holm–Bonferroni families; the judge gate thresholds and
pre-specified fallback; the annotation adjudication rule; the noise-floor rule; and the H2
trace-scan definition including its false-positive guard. *"Retro-fitting this after results is
worth far less — the value is entirely in the timestamp."*

**Status.** All nine required items written, plus registered extensions §3b-bis and §9a–§9g. The
binding line was correctly re-pegged from the calendar to **the first `run_id=main-v1`
generation** (§9f), on the reasoning that a date cannot detect the actual failure mode — *choosing
an analysis because you have seen which choice flatters the result* — while stating each decision's
provenance can.

⚠️ Every decision informed by data already seen says so in place, with the exploratory numbers
inline. That is the property that protects it, and it is worth not losing.

#### `C3` Method — ✅ DONE

94 prose lines across five subsections: experimental design, quantization and calibration, the
scoring ladder, compliance/competence/truncation, and serving configuration and determinism.

#### `C4` Introduction — ❌ EMPTY

**Specification.** W4. Must contain, per `EXECUTOR-PLAN.md` §7: the reframe that **an artifact in
a LEVEL is not an artifact in a DIFFERENCE** (everyone knows lenient extraction raises absolute
scores; nobody has established whether it changes the quantization *delta*); the decomposition
identity `measured_strict_delta = competence + compliance + truncation`; the fact that **the
field's centre of gravity runs against us** — the dominant finding is that benchmarks *understate*
quantization damage (Marchisio et al.: a 1.7% automatic drop in Japanese was 16.0% to human
raters) — **placed here, not in Limitations**, because both directions are true at once and saying
so plainly is a strength; the framing as **calibrating** extraction rather than loosening it; and
Kurtic et al. cited early so a reviewer does not discover it.

**Status.** Zero prose. The planning comments enumerating all of the above are present, so the
content is specified and unwritten.

#### `C5` Limitations + Responsible NLP checklist — ❌ EMPTY, desk-reject ground

Zero prose in Limitations, and the checklist is not started. Both were due W4 and both depend on
zero results. **This is the highest-value writing available** and the one whose absence is
automatically fatal.

#### `C6` Submittable draft assembly — ❌ FAILS (gate G6, **Sep 20**)

Currently fails on five empty sections. Compiles at 7 of 8 pages, which is the tighter constraint
than it looks: H8 alone occupies about a page and six Results subsections are still one-line
`\todo`s.

#### `C7` Results + Discussion + Abstract — 🟡 PARTIAL

H8's subsection is fully written (88 prose lines) with every statistic drawn from generated tables.
Six Results subsections are one-line `\todo`s. Discussion, Conclusion and Abstract are empty. The
Abstract is scheduled for W5 after the results freeze, so it is not late.

#### `C8` Related-work revision against the Sep 14 novelty scan — ❌ NOT DUE YET

---

## Sequencing this implies

**Zero-GPU, and it is the real critical path:**

1. **`A19` analysis module.** Blocks H3, H4, H7 and the centrepiece table. Largest single gap.
2. **Install `math-verify`.** One line. A declared rung has silently never run.
3. **`C5` Limitations, then `C4` Introduction.** Both cost only writing; Limitations is a
   desk-reject ground and G6 now enforces it.
4. **`B5` rubric.** Start before anything that only needs me, because it needs the co-author —
   then `A11` → `A12`.
5. **`A21` decomposition table.** Design it now and let it drive logging, as specified, while
   `main-v1` is still held and its logging can still change.
6. **Version scorer outputs** by `(cell, scorer_name, scorer_version)` before re-scoring overwrites
   something load-bearing.

**Needs GPU, in dependency order:** `A9` **Anchor A** (mandatory, blocking, never run) → `A6`'s
batch-invariance overhead pairing (one cheap job, and it is a paper number) → `A14` pipeline noise
floor → `A15` budget sweep (H1's missing instrument, ~25,000 generations, uncosted) → `A18`
zero-shot → `A13` `main-v1` with K=2 and `invocation_id` → H6's log-likelihood arms (R24.2: after
`main-v1`, before G5 closes).

## User-owned, and unverifiable by me

- 🚨 **Oct 14 — both authors register as ARR reviewers/ACs.** Per-author, within 48 h of
  submission. **Missing it is an automatic desk reject.** Also: both OpenReview profiles complete,
  since an incomplete profile blocks registration (`B2`).
- **`B5` needs the co-author** for the rubric dry-run and sign-off.
- **`A11` needs the co-author** for ≥120 double-annotated items in a fixed window.
- **A draft must reach the external reader by Sep 30** (`B4`).
- The Midway certificate for dev-desktop SSH expires roughly daily; only the user can run
  `mwinit -s`.
