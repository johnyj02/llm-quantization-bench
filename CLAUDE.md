# CLAUDE.md — operational instructions for this project

**Read this before touching compute.** It records the things that are not derivable from the
code and that cost real time to rediscover. Scientific design lives in `EXECUTOR-PLAN.md`;
scope/schedule in `EXECUTOR-TASKS.md`; decisions in `notes/decisions.md`. This file is
**how to actually run things**.

---

## 0. Hard rules

1. **🚨 NO job starts without explicit user confirmation** (D10). Every billable command
   requires `--i-have-confirmation` and offers a zero-cost inspection mode
   (`--plan-only` / `--validate-only` / `--preflight-only`).
2. **Resource and job names must be user-reviewed and generic** (D11).
   Scheme: `johnyj02-QA-v<x.x>-<date>-<uuid>[-stage]`. **`QA` is project initials and is NEVER
   expanded** — not in a name, tag, S3 key, or job id. A human scanning resources should learn
   nothing about the research.
3. **SMInterface is the ONLY compute path** (D14, overriding `EXECUTOR-TASKS.md` O4).
   EC2 is withdrawn; `scripts/box_*.sh` are retained for provenance only — **do not run them**.
4. **Never delete anything of the user's without asking.**
5. **Route questions to the planner**, not the user: `./ask-planner "..."` (D13). Batch them;
   each call re-sends its whole context. `FORK=1` for throwaway questions.

---

## 1. AWS account and credentials

| | |
|---|---|
| Account | **`654654594225`** |
| Provider | **CONDUIT**, not Isengard |
| Role | `IibsAdminAccess-DO-NOT-DELETE` |
| Profile | `qa-gpu` |
| Region | `us-east-1` (ECR repos exist here) or `us-west-2` — quota is 1 in both |
| SageMaker role | `arn:aws:iam::654654594225:role/AmazonSageMakerAdminIAMExecutionRole` |

### Refresh credentials — BOTH flags are required

```bash
ada credentials update --provider conduit --account 654654594225 \
    --role IibsAdminAccess-DO-NOT-DELETE --profile qa-gpu --once
```

⚠️ **Without `--provider conduit`, ada defaults to Isengard and returns a misleading
`AWSAccountNotFoundException: Account not found : 654654594225`** — which looks like a wrong
account ID, not a wrong provider. And ada no longer falls back to that role name for Conduit,
so `--role` must be passed too.

⚠️ **SMInterface cannot refresh these credentials itself.** Its
`session.refresh_credentials()` hardcodes `ada credentials update --profile=<p> --once` with no
provider flag. Set `auto_refresh_credentials: false` in the job config and refresh first —
`scripts/submit_job.py` does this.

Verify: `AWS_PROFILE=qa-gpu aws sts get-caller-identity`

---

## 2. Checking instance availability

**AWS publishes no capacity API.** "Is there capacity right now" is only truly answerable by
attempting a launch. Spot price is the best public proxy: it floats with demand, so spot near
on-demand means the pool is squeezed.

```bash
# Full signal report: offering AZs, quota, spot as % of on-demand, recommended region
bash scripts/check_capacity.sh                       # default g6e.2xlarge, both regions
bash scripts/check_capacity.sh ml.g5.12xlarge        # another type
bash scripts/check_capacity.sh g6e.2xlarge us-west-2 # one region
```

### Raw commands

```bash
export AWS_PROFILE=qa-gpu

# Does the AZ carry this type at all? (hard fact)
aws ec2 describe-instance-type-offerings --location-type availability-zone \
  --filters "Name=instance-type,Values=g6e.2xlarge" \
  --query 'InstanceTypeOfferings[].Location' --output text --region us-west-2

# EC2 vCPU quota (hard fact). L-DB2E81BA = On-Demand G and VT.
aws service-quotas get-service-quota --service-code ec2 \
  --quota-code L-DB2E81BA --query 'Quota.Value' --output text --region us-west-2

# Contention proxy: spot price per AZ. Lower = more slack.
aws ec2 describe-spot-price-history --instance-types g6e.2xlarge \
  --product-descriptions "Linux/UNIX" --region us-west-2 \
  --query 'SpotPriceHistory[].[AvailabilityZone,SpotPrice]' --output text | sort -u -k1,1
```

### ⚠️ SageMaker quota: ALWAYS paginate fully

```bash
# CORRECT — no --max-items
aws service-quotas list-service-quotas --service-code sagemaker --region us-west-2 \
  --query 'Quotas[?contains(QuotaName,`g6e.2xlarge`)].[QuotaName,Value]' --output text
```

**Do NOT pass `--max-items`.** SageMaker has more quotas than any sane cap, and the result is
truncated **silently**. I capped at 400, saw no `ml.g6e.2xlarge for training job usage`, and
wrongly reported that we had no quota. We do:

| quota | us-west-2 | us-east-1 |
|---|---|---|
| `ml.g6e.2xlarge for training job usage` | 1 | 1 |
| `ml.g6e.2xlarge for spot training job usage` | 1 | 1 |

Note `for training warm pool usage` is a **retention allowance**, not permission to run jobs —
don't mistake one for the other.

### Is anything of ours running (and billing)?

```bash
aws ec2 describe-instances --region us-west-2 --profile qa-gpu \
  --filters "Name=tag:Owner,Values=johnyj02" "Name=instance-state-name,Values=pending,running" \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name]' --output text

aws sagemaker list-training-jobs --profile qa-gpu --region us-west-2 \
  --status-equals InProgress --query 'TrainingJobSummaries[].TrainingJobName' --output text
```

---

## 3. Running a job

**SageMaker training jobs QUEUE for capacity. EC2 `RunInstances` fails immediately.** That is
the whole reason for D14: `g6e.2xlarge` was unavailable in **10 AZs across three regions**, so
EC2 could not make progress at all.

### 🚨 CAPACITY queues. QUOTA does NOT. Only ONE job may exist at a time.

These are two different limits and only the first one waits:

| | behaviour |
|---|---|
| **capacity** (no free GPU right now) | job sits in `Pending` — *"Training job waiting for capacity"* — and starts when one frees up. This is D14's whole benefit. |
| **quota** (`ml.g6e.2xlarge for training job usage` = 1) | **rejected instantly** at `CreateTrainingJob` with `ResourceLimitExceeded: … current utilization of 1 Instances`. No queue, no retry. |

**A `Pending` job consumes the quota slot.** So with a limit of 1 you cannot stack a second job
behind the first — the submission is refused outright. Submit one, wait for it to reach a
terminal state, then submit the next.

⚠️ **`stop-training-job` does not free the slot immediately.** The job passes through `Stopping`
(~45 s observed) still holding the quota. Poll for `Stopped` before resubmitting, or the next
submit fails with the same `ResourceLimitExceeded`.

```bash
# Is our slot actually free? (checks the specific job, cheap)
aws sagemaker describe-training-job --profile qa-gpu --region us-east-1 \
  --training-job-name <name> --query '[TrainingJobStatus,SecondaryStatus]' --output text
```

⚠️ Do **not** poll with `list-training-jobs` in a tight loop — it throttles
(`ThrottlingException: Rate exceeded`) and, with `NameContains`, paginates the whole shared
account history without returning. Use `describe-training-job` on the exact name, ≥60 s apart.

```bash
# Inspect at zero cost — submits nothing
python scripts/submit_job.py --stage verify --plan-only

# Submit (requires user confirmation per D10)
export HF_TOKEN=$(grep '^HF_TOKEN=' ~/workspaces/MedLLMSafety/.env | cut -d= -f2-)
python scripts/submit_job.py --stage verify --i-have-confirmation

# Watch to completion, then print why it ended. Resolves the newest job itself.
bash scripts/watch_job.sh                    # newest johnyj02-QA-* job
bash scripts/watch_job.sh <job-name> 30      # a specific one, 30s interval

# Read the log with the JSON dump and the tqdm spam stripped out
bash scripts/job_log.sh <job-name>           # last 60 useful lines
bash scripts/job_log.sh <job-name> 200 head  # FIRST 200 (preflight, config echo)
```

⚠️ **Job names embed the UTC date, and UTC rolls over mid-session.** Never hardcode it in a
polling loop — a stale date matches nothing, `--training-job-name` gets an empty argument, and
the watcher dies instantly while the job runs unobserved. `watch_job.sh` resolves the name.

⚠️ **Credentials expire after ~1 h.** A loop that does `2>/dev/null || continue` swallows
`ExpiredTokenException` and reports a stale status forever. `watch_job.sh` refreshes and retries.

Stages, in order: `verify` → `throughput` → `quantize` → `determinism`. All four have entry
points and submit cleanly.

### `--smoke` FIRST, every time

**Mandatory before every real submission** (planner ruling). With one job slot and a queue,
submitting a broken multi-hour job is the most expensive mistake available.

```bash
python scripts/submit_job.py --stage quantize --smoke --i-have-confirmation
```

`--smoke` caps `max_run` at 1800 s, suffixes the job name `-smoke`, and **redirects
`checkpoint_s3_uri` to `…/checkpoints-smoke/`**. That last one matters: SageMaker restores the
*whole* checkpoint prefix into the container at every job start, so junk left in the study's
prefix is re-downloaded by every later job — and a deliberately-wrong-calibration checkpoint
sitting in this study's own tree is the last artefact to leave for a reviewer. Smoke weights can
never be *used* by a real job regardless, because the recipe hash differs and every consumer
resolves by hash.

### ⚠️ The account is SHARED — the one job slot is not ours alone

`aws sagemaker list-training-jobs` shows another user's jobs (`jathupra-*`) in account
`654654594225`. The `ml.g6e.2xlarge` quota of **1 is account-level**, so their job contends with
ours for the only slot. Submit whenever the slot is free rather than batching up work.

### A stage exiting non-zero is not always a failed run

`sm_verify.py` and `sm_quantize.py` exit non-zero if *any* check or arm failed, so SageMaker
marks the job `Failed` even when it produced its full report. **Read the report before concluding
the run was lost.** `sm_determinism.py` deliberately inverts this: nondeterminism exits 0 (it is a
finding), and only a failure to *measure* exits non-zero.

### Where the code actually lives when a job runs

| step | location |
|---|---|
| 1. source of truth | `/Users/johnyj02/workspaces/QuantArtifact` (local git, **no remote**) |
| 2. staged copy | a temp dir with only `src/ configs/ scripts/ vendor/ sm_*.py pyproject.toml` (~1 MB) |
| 3. uploaded tarball | `s3://sagemaker-us-east-1-654654594225/<job-name>/source/sourcedir.tar.gz` |
| 4. in the container | extracted to **`/opt/ml/code`**, entry point run from there |
| 5. results | written to `/opt/ml/model` → `output_path` in S3 |

⚠️ **`submit_job.py` stages an ALLOWLIST rather than uploading `source_dir` directly.**
The SageMaker SDK tars *all* of `source_dir` and honours no ignore file, so pointing it at the
repo root would upload **575 MB on every submission** — 568 MB of it `.venv-harness` + `.venv`,
which are useless in the container (it has its own Python) and are macOS **arm64** wheels, so
actively wrong for a linux/amd64 container. Allowlist, not denylist: a new large directory in
the repo must not silently start being uploaded.
Config: `configs/sm_job.yaml`. Overrides SMInterface's dangerous defaults —
`max_run` 3600 → per-stage (a 2,719-item cell dies in an hour), `volume_size` 30 → 200 GB.

**`distributed` MUST stay `false`.** `tensor_parallel_size>1` breaks batch invariance
(vLLM #51290) and is the single most important determinism setting in the study.

---

## 4. The Docker image — required, not optional

SMInterface must run `framework: generic` with a custom ECR image.

**Why the stock DLC cannot work:** `vllm==0.26.0` pins `torch==2.11.0` **exactly**, while the
newest SageMaker PyTorch DLC is torch 2.10.0. And pip-installing over it does not help, because
**the CUDA runtime vLLM's own precompiled `.so` links is not controlled by the Python torch
pin** — verified in MedLLMSafety with `objdump -p vllm/_C.abi3.so | grep NEEDED`.

### 🔨 BUILD ON THE DEV DESKTOP, not this Mac

```bash
DEV=dev-dsk-johnyj02-2a-53853bae.us-west-2.amazon.com
bash scripts/build_image_remote.sh              # plan only
bash scripts/build_image_remote.sh --apply      # rsync repo, build, push to ECR
```

**Why the dev desktop and not the laptop:**

| | Mac | dev desktop |
|---|---|---|
| arch | **arm64** → cross-build under emulation, very slow | **x86_64** → native `linux/amd64`, which is what SageMaker needs |
| disk | ~24 GB free; torch+vllm images are multi-GB | ~1.5 TB free |
| network | home uplink | Amazon network, much faster for the multi-GB pulls |

⚠️ **SSH to the dev desktop needs a live Midway cert**, and it expires roughly daily
(`Valid: from … to …`, ~20 h). When SSH returns `Permission denied (publickey)` that is
almost always an expired cert, not a key problem. **The user must refresh it** — it is
interactive:

```bash
ssh-keygen -L -f ~/.ssh/id_ecdsa-cert.pub | grep Valid   # check expiry
mwinit -s                                                 # user runs this; interactive
```

Do **not** invoke the shell's `mwinit` alias — on this machine it pipes a hardcoded password.

Local build (fallback only, if the dev desktop is unreachable):

```bash
bash scripts/build_image.sh          # build + push to ECR, then sets image_uri
```

### Three rules the image must satisfy

1. **`build-essential` STAYS INSTALLED — never purged after pip.** vLLM JIT-compiles Triton
   kernels at *runtime*. `MedLLMSafety/Dockerfile` purges it (that one is torch 2.6 and *not*
   our base); `Dockerfile.vllm` deliberately keeps it after a real job died with
   `"Failed to find C compiler"`. For us it is worse than a crash: `VLLM_BATCH_INVARIANT` works
   by substituting batch-invariant **Triton** kernels, so no compiler ⇒ no determinism ⇒ H8
   dead and every delta untrustworthy. `gcc` alone is insufficient (no libc headers);
   `build-essential` pulls in `libc6-dev`.
2. **torch from the CUDA wheel index**, not plain PyPI.
3. **`llmcompressor` lives in a SEPARATE image.** It pins `compressed-tensors==0.17.1` exactly
   and `transformers<=5.10.1`, conflicting with vllm 0.26.0. This is §3.2's two-venv split
   expressed as two images.

### ✅ CLOSED: the CUDA-13 risk. Measured 2026-08-26 — do not re-litigate

`vllm==0.26.0` links `libcudart.so.13`, so the host driver had to support CUDA 13. MedLLMSafety
hit `"The NVIDIA driver on your system is too old (found version 12080)"` on `ml.g5.2xlarge` and
fell back to `vllm==0.18.0`. **`ml.g6e.2xlarge` is fine.** The `verify` stage measured:

| | measured |
|---|---|
| driver | **595.91.07** |
| torch | `2.11.0+cu130`, built for CUDA 13.0, `available=True` |
| vLLM linkage | `libcuda.so.1`, **`libcudart.so.13`**, `libtorch_cuda.so` |
| gcc / Triton | Debian 14.2.0 / triton 3.6.0, JIT compiled and ran |
| GPU | L40S, **46068 MiB** (not the nominal 48 GB), SM 8.9 |

**The engine version stays frozen at 0.26.0. No planner escalation needed.** Full write-up in
`notes/compute.md` §9; the four measured facts are now in `configs/models.yaml:hardware`.

⚠️ Still open, and **A7 owns it**: `bi_supported_on_sm89` is `null`. Triton compiling a kernel is
necessary but *not sufficient* for batch invariance, and §3.3's 1.2–2× BI overhead is an **Ampere**
number that does not transfer to Ada.

⚠️ Two `sm_verify.py` fixes from `5ada832` are **unexercised on GPU** — the `ctypes.CDLL` probe
(replacing a wrong `import vllm._C`; 0.26.0 registers ops through `torch.ops`) and the
`importlib.metadata` version fallback. Fold a re-verify into the next serving-image job rather
than spending a queue slot on `verify` alone.

**If a future box ever reports driver < 13, changing the engine version is a PLANNER decision**,
not an executor fallback: the version is frozen and reported in the paper, and batch-invariance
behaviour is version-specific (PR #27660; the `auto_gptq.py`/`auto_awq.py` kernel guards H8 is
built on).

---

## 5. Local environments

| env | purpose | notes |
|---|---|---|
| `.venv` | analysis, scoring, tests | Python 3.12. `pytest`, `ruff`. No torch, no vLLM. |
| `.venv-harness` | `lm-eval` + `transformers` | For **G2 prompt materialization only**. Imports without torch, which is why G2 closed on a laptop. |

**No local model inference is possible.** This is Apple Silicon; GPTQ/AWQ/bnb do not run
usefully and vLLM's CPU backend is x86-only.

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m ruff check src tests scripts
make -C paper g6                      # gate G6: compiles, <=8pp, Limitations, todo scope
bash scripts/g2_prompt_parity.py      # run with .venv-harness/bin/python
```

Paper builds locally with **`tectonic`** (`/opt/homebrew/bin/tectonic`) — no MacTeX needed, and
G6 does not depend on Overleaf.

---

## 6. Secrets

- `HF_TOKEN` lives in `~/workspaces/MedLLMSafety/.env` (gitignored there). **Never copy it into
  this repo**; export it into the environment at submit time.
- Private keys go to `~/.ssh/`, mode 400, region-suffixed (EC2 key pairs are region-scoped).
- Nothing in this repo is pushed anywhere: `git remote -v` is empty by design.

---

## 7. Things I got wrong once — don't repeat them

| mistake | correction |
|---|---|
| `--max-items 400` on SageMaker quotas | Truncates silently. Paginate fully. |
| Assumed quota travels between accounts | It does not. Re-verify per account (D11). Old account had 768 vCPU / 0 SageMaker; new has 640 / 1. |
| `: "${AWS_REGION:=...}"` | Silently inherits an ambient region. Set it explicitly. |
| One `.pem` path for all regions | Key pairs are region-scoped; suffix the filename. |
| Pricing API without `regionCode` | Returned the SageMaker rate ($2.8035) as EC2's ($2.2421). |
| `run-instances` without `--subnet-id` | Tries one AZ and gives up. Iterate all of them. |
| Re-asking settled questions | `B2`/`B4` are the user's and closed (D12). Don't raise them. |
| `GPTQModifier(group_size=…, desc_act=…)` | Neither is a keyword in llmcompressor 0.12.0.1 — pydantic `extra_forbidden`. `group_size` lives in the preset scheme; `desc_act` became `actorder`. `notes/compute.md` §10. |
| Omitting `actorder` | **Silently gives static activation reordering.** `GPTQModifier.actorder` defaults to `Sentinel("static")`, which fills in `STATIC` whenever the scheme's is `None`. Only an explicit `actorder=None` means `desc_act: false`. llmcompressor's own example omits it, so the tutorial recipe is *not* what Anchor A compares against. |
| Smoking one arm | GPTQ and AWQ use different llmcompressor APIs (`AWQModifier` is a *function* returning two modifiers). A green GPTQ smoke says nothing about AWQ. `--smoke` now runs one arm per method. |
| Believing a `%` comment in a `.bib` | BibTeX has no top-level comment char: it ignores everything but `@`, so an `@` in a comment starts an entry. And inside an entry `%` is parsed as a field name. Both drop citations **silently** while the PDF still builds. |
