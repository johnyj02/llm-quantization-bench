# QuantArtifact

*How much of reported quantization degradation is measurement?*
Decomposing accuracy loss into **competence**, **compliance**, and **truncation**.

**Target:** ARR **2026-10-12, 23:59 AoE** → NAACL 2027 primary, Findings automatic.

---

## Read order

| Document | Authority |
|---|---|
| `EXECUTOR-TASKS.md` | **Execution** — scope, substrate, schedule, task order, acceptance criteria, gates |
| `EXECUTOR-PLAN.md` | **Scientific design** — hypotheses, factors, scoring semantics, identification strategy, statistics |
| `analysis/preregistration.md` | The frozen analysis plan. **Freeze deadline 2026-09-02.** Its value is the timestamp. |
| `notes/harness-configs.md` | Verified `lm-eval` v0.4.12 facts, and three problems found in the plan's premises |

Where the two plans disagree, TASKS wins on execution and PLAN wins on science. TASKS §0.2
lists eleven intentional overrides; a conflict not listed there is a planning bug.

## The one architectural rule

**Generate once, score many times.** Generation is expensive, GPU-bound and
nondeterministic. Scoring is cheap, deterministic, and will be revised repeatedly. They are
never coupled. Every generation record is complete enough that *any* future scorer can run
over it with no GPU — which is also exactly what the artifact release needs.

Three things that are unreconstructable after the fact and must never be dropped:

1. **`finish_reason` and `n_gen_tokens` on every generation record.** H1 is impossible
   without them.
2. **The engine, its version, and the full generation config, frozen across all arms.**
   Backend choice alone moves benchmark scores up to 16.6 pp (arXiv:2605.19537); if it
   floats, the effects are not identifiable.
3. **The §4.0 identification fix, pre-registered before any generations exist.** It cannot
   be retrofitted.

## Layout

```
configs/          models.yaml quant.yaml tasks.yaml scoring.yaml   (all pinned by SHA)
src/quantartifact/
  generate.py     the ONLY component that touches a GPU for generation
  quantize.py     llmcompressor wrappers (GPTQ and AWQ both — one calibration path)
  tasks/          dataset loading, subsampling, prompt rendering
  scoring/        extract.py (rungs 1-3, pure) · judge.py (rung 4) · compliance.py
  analysis/       GLMM, decomposition, Manski bounds, plots, tables
scripts/          thin CLI entry points; one job = one command, resumable, idempotent
artifacts/        generations/ scores/ (gitignored, S3 + release) · tables/ figures/ (committed)
tests/            scorer unit tests are the highest-value tests in this repo
annotation/       rubric.md (verbatim in the appendix) · labels.jsonl · iaa.md
notes/            access · compute · arr-obligations · review-plan · harness-configs · estimates
paper/            LaTeX, ACL template, mirrored from Overleaf via git
vendor/           pinned upstream configs, vendored for provenance
```

## Substrate

**EC2 `g6e.2xlarge`** — 1 × L40S 48 GB (SM89), 8 vCPU, 64 GiB RAM — in AWS account
`670060057797`, us-east-1, profile `obtmodel`. Two venvs on the box: `venv-quant`
(`llmcompressor`) and `venv-serve` (`vllm`, `lm-eval`, `math-verify`). `tensor_parallel_size=1`
is **required** — batch invariance is broken under TP>1.

The Mac is a dev, analysis and writing box only. No local model inference: GPTQ/AWQ/bnb do
not run usefully on Apple Silicon and vLLM's CPU backend is x86-only.

See `notes/compute.md` for the decision record, including the amendment to override O4.

## CLI contract

Every command is resumable and idempotent — re-running with identical arguments produces
zero new work and exits 0.

```
qa generate    --cell <cell_id> --task <task> --budget 1x --variant fewshot_cot --run-id main-v1
qa score       --run-id main-v1 --scorer permissive
qa judge       --run-id main-v1 --judge xverify --cascade none|rung23
qa determinism --format gptq-4bit --repeats 3
qa analyze     --hypothesis h3 --task gsm8k
qa report      --table decomposition
```

Join key across generations and every scorer, enforced as a uniqueness constraint:
`(run_id, cell_id, task, subtask, item_id, budget_level, prompt_variant, sample_idx, repeat_idx)`.

## Gates

| Gate | Due | What closes it |
|---|---|---|
| `G1` env + access | Aug 25 | `notes/access.md`, lockfiles, `notes/compute.md` |
| `G2` prompt parity | Aug 27 | **byte-for-byte** equality vs `lm-eval` for all task configs |
| `G3` scorers | Aug 28 | all fixtures pass; rung monotonicity; ARC ll within 0.5 pp of `lm-eval` |
| `G4` determinism (H8) | Aug 30 | per-format bitwise disagreement over 3 repeats |
| **PREREG** | **Sep 2** | `analysis/preregistration.md` committed — **hard block on `A13`** |
| `G5a` anchors | Sep 2 | Anchor A within ±2.0 pp |
| `G5` reference runs | Sep 13 | 15 cells generated and scored |
| **`G6` submittable draft** | **Sep 20** | compiles, ≤8 pages, every section written, `\todo{}` only in Results |
| `G7` results frozen | **Sep 25** | no new runs or analyses after this |
| `G8` external review | Oct 4 | reader's notes |
| `G9` submitted | Oct 10 | OpenReview confirmation |

Two standing rules: **an always-submittable artifact from Sep 20**, and **no new science
after Sep 25**.

⚠️ **Wed Oct 14: both authors must register as ARR reviewers/ACs.** The obligation is
per-author, not per-paper, and missing it is an automatic desk reject.
