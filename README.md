# QuantArtifact

*How much of reported quantization degradation is measurement?*
Decomposing accuracy loss into **competence**, **compliance**, and **truncation**.

**Target:** ARR **2026-10-12, 23:59 AoE** → NAACL 2027 primary, Findings automatic.

This file is the **operator's guide**: what to run, where, and in what order.
`CLAUDE.md` is the companion — it holds the *reasons* and the traps, and it is the
authority where the two overlap. **Deliberately not duplicated here:** the gate
statuses (`notes/gates.md`) and the per-run cost ledger (`notes/RUNLOG.md`). A summary
that can drift will drift — that already happened once with the ledger.

---

## 0. Read order

| Document | Authority |
|---|---|
| `EXECUTOR-TASKS.md` | **Execution** — scope, substrate, schedule, task order, gates |
| `EXECUTOR-PLAN.md` | **Scientific design** — hypotheses, factors, scoring semantics, statistics |
| `analysis/preregistration.md` | The frozen analysis plan. **Freeze 2026-09-02.** Its value is the timestamp. |
| `CLAUDE.md` | **How to actually run things**, and every trap paid for once already |
| `notes/planner-rulings.md` | Planner's authoritative rulings (R1–R16), raw text in `notes/planner-answers/` |
| `notes/RUNLOG.md` | Every billable run: what it cost and what it taught us |
| `notes/gates.md` | Live gate status |

TASKS wins on execution, PLAN wins on science. TASKS §0.2 lists eleven intentional
overrides; a conflict *not* listed there is a planning bug — stop and report it.

### Asking the planner

Scientific questions go to the **planner**, not the user (D13):

```bash
./ask-planner "question"                    # resumes its session — it remembers everything
FORK=1 ./ask-planner "throwaway question"   # reads its context without writing to its history
```

Opus 5 (1M context) at `--effort max`, resuming session `54ba15b1-…`. Two rules:

- ⚠️ **Batch the questions.** Each call re-sends the planner's whole context, so five
  separate calls cost five times what one does.
- ⚠️ **Save the answer unabridged to `notes/planner-answers/`, never read it through
  `tail -n`.** A `tail -60` silently truncated R15 and lost two of its points. Then
  distil it into `notes/planner-rulings.md` as an R-number.

## 1. The one architectural rule

**Generate once, score many times.** Generation is expensive, GPU-bound and
nondeterministic; scoring is cheap, deterministic, and will be revised repeatedly. They
are never coupled. Three things are unreconstructable after the fact:

1. **`finish_reason` and `n_gen_tokens` on every record.** H1 is impossible without them.
2. **The engine, version, and full generation config, frozen across arms.** Backend
   choice alone moves scores up to 16.6 pp; if it floats, nothing is identifiable.
3. **The §4.0 identification fix, pre-registered before any generation exists.**

---

## 2. AWS access — do this first, every session

| | |
|---|---|
| Account | **`654654594225`** (⚠️ **not** the old `670060057797`) |
| Provider | **CONDUIT**, not Isengard |
| Role | `IibsAdminAccess-DO-NOT-DELETE` |
| Profile | `qa-gpu` |
| Region | `us-east-1` (where the ECR images live) |

```bash
ada credentials update --provider conduit --account 654654594225 \
    --role IibsAdminAccess-DO-NOT-DELETE --profile qa-gpu --once

AWS_PROFILE=qa-gpu aws sts get-caller-identity      # verify
```

⚠️ **Both `--provider conduit` and `--role` are required.** Without the provider, ada
defaults to Isengard and returns `AWSAccountNotFoundException: Account not found`, which
looks like a wrong account ID rather than a wrong provider.

⚠️ **Credentials expire in ~1 hour.** Long-running watchers must refresh, not swallow the
error — `scripts/watch_job.sh` does.

## 3. Substrate

**SageMaker training jobs via SMInterface. This is the ONLY compute path** (decision
D14 — EC2 is withdrawn; `scripts/box_*.sh` are kept for provenance, do not run them).

`ml.g6e.2xlarge` — 1 × L40S (**46068 MiB**, not the nominal 48 GB), SM89 Ada, 8 vCPU,
64 GiB RAM. `tensor_parallel_size=1` is **required**: batch invariance is broken under
TP>1 (vLLM #51290).

🚨 **Quota is 1, the account is SHARED, and quota is enforced at submit time — it does
not queue.** A `Pending` job holds the slot, so a second submission is refused outright
with `ResourceLimitExceeded`. One job at a time, to a terminal state.

The Mac is dev/analysis/writing only. No local model inference: GPTQ/AWQ/bnb do not run
usefully on Apple Silicon and vLLM's CPU backend is x86-only.

---

## 4. Running a job

Stages, in dependency order:

| stage | entry point | what it does |
|---|---|---|
| `verify` | `sm_verify.py` | GPU, compute capability, C compiler, Triton JIT, CUDA linkage |
| `throughput` | `sm_throughput.py` | A6 (now largely subsumed — job 2 reports per-arm rates) |
| `quantize` | `sm_quantize.py` | **job 1** — 9 checkpoints at the normalised calibration budget |
| `determinism` | `sm_determinism.py` | **job 2** — order-divergence frontier, H8, gate G4 |
| `pilot` | `sm_pilot.py` | **job 3** — 17 cells × 300, the annotation pool |

```bash
# 1. Inspect at ZERO cost — submits nothing, validates everything
.venv/bin/python scripts/submit_job.py --stage determinism --plan-only

# 2. Rehearse. MANDATORY before every real submission (see §5)
export HF_TOKEN=$(grep '^HF_TOKEN=' ~/workspaces/MedLLMSafety/.env | cut -d= -f2-)
.venv/bin/python scripts/submit_job.py --stage determinism --smoke --i-have-confirmation

# 3. The real thing
.venv/bin/python scripts/submit_job.py --stage determinism --i-have-confirmation

# 4. Watch to completion, then print why it ended
bash scripts/watch_job.sh                       # resolves the newest job itself
bash scripts/watch_job.sh <job-name> 120         # or a specific one, 120s interval

# 5. Read the log with the JSON dump and tqdm spam stripped
bash scripts/job_log.sh <job-name>               # last 60 useful lines
bash scripts/job_log.sh <job-name> 200 head      # FIRST 200 (preflight, config echo)

# 6. Record it — do this AS SOON AS the job goes terminal
bash scripts/logrun.sh <job-name> "one-line outcome"
bash scripts/logrun.sh                           # running total + refresh the ledger
```

🚨 **No billable job starts without explicit user confirmation** (D10). Every command
offers a zero-cost inspection mode. `--i-have-confirmation` is deliberately awkward.

⚠️ **Job names embed the UTC date and must be unique per account.** UTC rolls over
mid-session, so never hardcode it; `submit_job.py` appends `-r2`, `-r3` on collision and
`watch_job.sh` resolves the name itself.

⚠️ **A non-zero exit is not always a failed run.** `sm_verify.py` and `sm_quantize.py`
exit non-zero if *any* check or arm failed, so SageMaker marks the job `Failed` even when
it produced its full report. **Read the report before concluding the run was lost.**

`sm_determinism.py` no longer does this: it separates **load-bearing** cells (the sweep —
the frontier *is* the contribution) from **corroborating** ones (a further Marlin arm once
the mechanism is pinned, Phi generality), and only the former fail the job. Both lists are
printed and stored in `determinism.json` (`unmeasured` / `unmeasured_optional`), so exit 0
never means "everything ran". Job 2 read `Failed` on the strength of one optional cell of
21 while the whole sweep had completed — three separate times a finished run has had to be
talked back out of looking like a lost one.

## 5. Why `--smoke` always comes first

20 items, capped `max_run`, `-smoke` name suffix, isolated outputs. Roughly **11% of
spend so far, and it has caught**: two llmcompressor API errors, a cross-arm GPU memory
leak, a genuine AWQ OOM, a missing `nvcc`, a silently-overridden `async_scheduling`, a
double-BOS re-tokenization bug, and one bug of mine in the ordering comparison. Each
would otherwise have surfaced inside a multi-hour queued run.

Two rules the smoke must obey, both learned the hard way:
- **Cover every distinct code path.** A GPTQ-only smoke said nothing about AWQ, which
  uses a genuinely different llmcompressor API.
- **Never skip on resume.** A rehearsal that resumes a previous rehearsal reports success
  while running nothing.

---

## 6. Docker — required, not optional

Two images, because `llmcompressor` and `vllm` **cannot coexist**: llmcompressor pins
`compressed-tensors==0.17.1` exactly and `transformers<=5.10.1`, which conflicts with
vllm 0.26.0. This is §3.2's two-venv split expressed as two images.

| file | tag | used by |
|---|---|---|
| `Dockerfile` | `johnyj02-qa:v0.1-<date>` | verify, throughput, determinism, pilot |
| `Dockerfile.quant` | `johnyj02-qa:v0.1-<date>-quant` | quantize |

The stock SageMaker DLC **cannot** work: `vllm==0.26.0` pins `torch==2.11.0` exactly
while the newest DLC is 2.10.0 — and pip-installing over it does not help, because the
CUDA runtime vLLM's precompiled `.so` links is not controlled by the Python torch pin.

### Build and push — ON THE DEV DESKTOP, not the Mac

```bash
bash scripts/build_image_remote.sh                                    # plan only
bash scripts/build_image_remote.sh --apply                            # serve image
bash scripts/build_image_remote.sh --apply --dockerfile=Dockerfile.quant   # quant image
```

It rsyncs the repo, refreshes Conduit credentials *there*, creates the ECR repo if
absent, builds `--platform linux/amd64` natively, pushes, and writes the resulting
`image_uri` back into `configs/sm_job.yaml`.

| | Mac | dev desktop |
|---|---|---|
| arch | **arm64** → emulated cross-build, very slow | **x86_64** → native `linux/amd64`, which is what SageMaker needs |
| disk | ~24 GB free | ~1.5 TB |
| network | home uplink | Amazon network |

### Dev desktop access

```bash
DEV=dev-dsk-johnyj02-2a-53853bae.us-west-2.amazon.com
ssh-keygen -L -f ~/.ssh/id_ecdsa-cert.pub | grep Valid   # check expiry
mwinit -s                                                 # YOU run this — interactive
ssh $DEV
```

⚠️ **The Midway cert expires roughly daily (~20 h).** `Permission denied (publickey)` is
almost always an expired cert, not a key problem. Do **not** use the shell's `mwinit`
alias on this machine — it pipes a hardcoded password.

⚠️ `ada` lives in `~/.toolbox/bin`, which is **not** on a non-interactive SSH PATH.

### Changing an image

1. Edit `Dockerfile` / `Dockerfile.quant`. Keep the build-time asserts — they fail the
   *build* instead of a queued job.
2. Rebuild on the dev desktop (above). The tag is date-stamped, so a rebuild on a new day
   is a new tag.
3. Confirm both tags exist and are **distinct**:
   ```bash
   AWS_PROFILE=qa-gpu aws ecr describe-images --repository-name johnyj02-qa \
     --region us-east-1 --query 'imageDetails[].[imageTags[0],imageSizeInBytes]' --output text
   ```
4. `configs/sm_job.yaml` → `images.serve` / `images.quant` select per stage.

**Three rules the image must satisfy** (full reasoning in `CLAUDE.md` §4):
1. **`build-essential` STAYS INSTALLED.** vLLM JIT-compiles Triton kernels at *runtime*;
   `VLLM_BATCH_INVARIANT` works by substituting batch-invariant **Triton** kernels, so no
   compiler ⇒ no determinism ⇒ H8 dead. `gcc` alone is insufficient (no libc headers).
2. **torch from the CUDA wheel index**, not plain PyPI.
3. **`llmcompressor` in a SEPARATE image.**

⚠️ There is **no `nvcc`** in either image (torch's cu130 wheels ship the CUDA *runtime*,
not the toolkit). That is why `VLLM_USE_FLASHINFER_SAMPLER=0` is required — flashinfer
JIT-compiles a CUDA module at runtime. It is also the *correct* setting: batch invariance
substitutes Triton kernels, and the flashinfer sampler is a separate unguarded path.

---

## 7. Where everything goes

| what | where |
|---|---|
| source of truth | this repo (local git, **no remote** — by design) |
| uploaded code | `s3://sagemaker-us-east-1-654654594225/<job-name>/source/sourcedir.tar.gz` → `/opt/ml/code` |
| quantized checkpoints | `s3://…/johnyj02/QA/v0.1/checkpoints/<model>__<arm>__<recipe-hash>/` |
| smoke checkpoints | `s3://…/johnyj02/QA/v0.1/checkpoints-smoke/` (isolated on purpose) |
| generations | `…/checkpoints/generations/<run_id>/<cell>/…jsonl.gz` (continuously synced) |
| job reports | `s3://…/johnyj02/QA/v0.1/` ← `/opt/ml/model`, uploaded only at job END |
| container images | ECR `654654594225.dkr.ecr.us-east-1.amazonaws.com/johnyj02-qa` |
| HF model cache | `/opt/ml/hf-cache` in-container (**on the 200 GB volume**, not the small root fs) |

⚠️ **Generations go to the checkpoint prefix, not `/opt/ml/model`.** `output_path` only
uploads when the job *ends*, so a timeout at cell 14 of 17 would lose everything;
`checkpoint_s3_uri` syncs continuously and restores on restart.

⚠️ **`submit_job.py` stages an ALLOWLIST**, not the repo root. The SageMaker SDK tars
*all* of `source_dir` and honours no ignore file — pointed at the root that is 575 MB, of
which 568 MB is `.venv*`, useless in the container and macOS **arm64** so actively wrong.

---

## 8. Local work (no GPU needed)

| env | purpose |
|---|---|
| `.venv` | analysis, scoring, tests. Python 3.12, `pytest`, `ruff`. No torch, no vLLM. |
| `.venv-harness` | `lm-eval` + `transformers`, for prompt materialization only |

```bash
.venv/bin/python -m pytest tests/ -q                     # 265 tests
.venv/bin/python -m ruff check src tests scripts sm_*.py
.venv/bin/python scripts/audit_configs.py                # config keys nothing consumes
make -C paper g6                                         # gate G6: compiles, ≤8pp, Limitations
```

The paper builds with **`tectonic`** (`/opt/homebrew/bin/tectonic`) — no MacTeX, and G6
does not depend on Overleaf.

⚠️ **BibTeX has two silent traps** (see `paper/custom.bib` header): `%` is not a comment
*inside* an entry, and an `@` anywhere in a comment starts a new entry. Both drop
citations while still producing a PDF. Always check the rendered entry count.

## 9. Secrets

- `HF_TOKEN` lives in `~/workspaces/MedLLMSafety/.env`. **Never copy it into this repo**;
  export it at submit time.
- Nothing here is pushed anywhere: `git remote -v` is empty by design.
- **Do not redistribute xVerify weights or derivatives** (CC-BY-NC-ND) — run and report
  numbers only. Qwen2.5-3B is `license:other`/qwen-research, **not** Apache-2.0.

## 10. Layout

```
configs/            models.yaml quant.yaml tasks.yaml scoring.yaml sm_job.yaml
sm_*.py             SageMaker entry points, one per stage
scripts/            submit_job.py · watch_job.sh · job_log.sh · logrun.sh
                    build_image_remote.sh · audit_configs.py
src/quantartifact/
  generate.py       the ONLY component that touches a GPU for generation
  quantize.py       llmcompressor wrappers — GPTQ and AWQ share one calibration path
  determinism.py    order-sensitivity, the two floors, the frontier
  records.py        GenerationRecord + the 9-field join key
  tasks/            dataset loading, subsampling, prompt rendering
  scoring/          extract.py (rungs 1–3) · judge.py (rung 4) · compliance.py
notes/              RUNLOG · gates · decisions · planner-rulings · planner-answers · compute
paper/              LaTeX + custom.bib (every entry fetched, not recalled)
tests/              scorer unit tests are the highest-value tests here
```

Join key, enforced as a uniqueness constraint across generations and every scorer:
`(run_id, cell_id, task, subtask, item_id, budget_level, prompt_variant, sample_idx, repeat_idx)`

## 11. Dates that do not move

- **Sep 2** — pre-registration freeze. Hard block on the main runs.
- **Sep 20** — submittable draft (G6). Always-submittable from here.
- **Sep 25** — results frozen. No new science after this.
- **Oct 12** — ARR submission.
- ⚠️ **Oct 14 — both authors must register as ARR reviewers/ACs.** Per-author, not
  per-paper. **Missing it is an automatic desk reject.**
