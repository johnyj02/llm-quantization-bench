# Compute substrate — task `B3`

**Status:** DECIDED and available. `B3` closes with **zero external latency**, which
removes one of the three named deadline risks (`EXECUTOR-TASKS.md` §2.6).

---

## 1. Decision

**EC2 `g6e.2xlarge` in AWS account `670060057797`, region `us-east-1`.**

| | |
|---|---|
| GPU | **1 × NVIDIA L40S, 48 GB** (`45776 MiB`), compute capability **SM89 (Ada Lovelace)** |
| vCPU / host RAM | 8 vCPU / **64 GiB** |
| On-demand price | **$2.2421 / hr** (us-east-1, Linux, shared tenancy, verified via Pricing API 2026-08-23) |
| Account / profile | `670060057797` / `obtmodel` (ada-managed; `ada credentials update --profile=obtmodel --once`) |
| Role | `IibsAdminAccess-DO-NOT-DELETE` (admin — see §5 safety note) |
| Storage | 200 GB **gp3 EBS** (~$16/mo). Quota 461 TiB, non-binding. |
| AZs available | `us-east-1a/b/c/d` |
| `tensor_parallel_size` | **1** — required; batch invariance is broken under TP>1 (vLLM #51290) |

## 2. Why this, and how it relates to override O4

`EXECUTOR-TASKS.md` **O4** decided *"rented persistent GPU box; SMInterface is a documented
non-goal"*, on two premises: (i) the determinism audit `A7` is inherently interactive and a
batch-only substrate is the worst possible fit, and (ii) AWS GPU quota latency is
days-to-weeks against a 7-week runway.

Verified 2026-08-23 in the user's account:

- **Premise (i) still holds, and is now *stronger* against SMInterface.** SageMaker
  **training-job** quota — which is what SMInterface submits — is **zero for every usable
  GPU**. `ml.g6e.xlarge`, `ml.g6e.2xlarge` and `ml.g5.xlarge` are not even enumerated under
  *"for training job usage"*. The only on-demand single-GPU training quota present is
  `ml.g6.2xlarge = 1` (L4 24 GB — wrong GPU: too little VRAM for the 9B judge, and slow).
  So SMInterface cannot serve this study without a quota-increase request, i.e. exactly the
  days-to-weeks latency O4 was written to avoid.
- **Premise (ii) is FALSE for this account.** EC2 GPU quota is **already granted**:
  On-Demand G and VT = **768 vCPUs**, On-Demand P = 768 vCPUs. No request needed.

So O4's **conclusion** (do not use SMInterface/SageMaker) stands and is reinforced. O4's
**premise** that AWS therefore requires renting third-party compute does not hold here —
the planner did not know the user already had an admin AWS account with GPU quota granted.
EC2 gives exactly what O4 required (an interactive, persistent, single-GPU box) inside an
account that already exists, with no signup and no new payment method.

**Reported to the user and confirmed 2026-08-23.** This is a recorded amendment to O4, not
a silent reversal.

### Options considered

| Option | GPU | $/hr | Interactive | Signup latency | Verdict |
|---|---|---|---|---|---|
| **EC2 `g6e.2xlarge`** | L40S 48 GB | 2.242 | yes | **none** | **CHOSEN** |
| EC2 `g6e.xlarge` | L40S 48 GB | 1.861 | yes | none | Rejected: 32 GiB host RAM is tight for `llmcompressor`'s sequential 512×2048 calibration pass on the 8B model; an OOM there costs more than the $0.38/hr saved |
| RunPod (O4's pick) | A100-80 GB | 1.19 | yes | **days** | Rejected by user: cheaper/hr *and* 2.4× the memory bandwidth, but needs a new account + card |
| SageMaker Studio CodeEditor | L40S 48 GB | ~2.2 | yes | none | Rejected: managed image, insufficient control over CUDA/driver/kernel pinning, which §4.6 requires pinning |
| SageMaker training job (SMInterface) | ml.g6.2xlarge L4 24 GB | ~1.6 | **no, batch** | none | Rejected: quota 0 for usable GPUs; batch-only |
| EC2 `p4de.24xlarge` | 8 × A100-80 GB | ~40 | yes | none | Rejected: no single-A100-80G instance exists on AWS at all; 8 GPUs invites a TP>1 mistake; spot quota 0 |
| EC2 `g5.xlarge` | A10G 24 GB | 1.006 | yes | none | Rejected: 24 GB too tight for the 9B judge |

## 3. Consequences to carry into the paper and into `A6`/`A7`

These are **deviations from the plan's assumed hardware** and must be pinned and stated
(`EXECUTOR-PLAN.md` §4.6: *"reproducibility holds only within identical hardware and
version"*).

1. **L40S 48 GB, not A100-80 GB.** `EXECUTOR-TASKS.md` §3.1 already names L40S as an
   approved substrate. VRAM is sufficient: largest weight footprint is Llama-3.1-8B bf16
   at 15.07 GB, and the largest judge (xVerify-9B-I) at ~18 GB, both comfortably inside
   48 GB at `max_num_seqs=8`.
2. **SM89 (Ada), not SM80 (Ampere).** The §3.3 budget of *"1.2–2× wall-clock for batch
   invariance on Ampere, because SM80 must swap in Triton GEMMs"* is an Ampere figure.
   **The SM89 BI overhead is unmeasured and must not be assumed** — `A6` measures it
   directly (with and without `VLLM_BATCH_INVARIANT=1`), and `A7` must verify that vLLM's
   batch-invariant kernels are actually supported on SM89 rather than silently degrading.
   Treat "does BI work at all on SM89" as an `A7` finding, not a precondition.
3. **Memory bandwidth is ~864 GB/s vs A100-80G's ~2039 GB/s.** At `max_num_seqs=8` the
   workload is decode-bound, so throughput could be materially lower than the §2.6 estimate
   assumed. **This is the main risk to the cost envelope** — see §4.
4. **Marlin kernels do run on SM89**, so the `auto_gptq.py`-has-no-BI-guard asymmetry that
   H8 is built on (§3.3) should still be observable. Verify in `A7`, do not assume.

## 4. Cost envelope — flagged risk

`EXECUTOR-TASKS.md` §2.6 estimated **$370–440 all-in** on A100-80G at $1.19/hr, including a
**$210 idle-billing** line for a persistent rented box.

On EC2 the two lines move in opposite directions:

- **Against us:** $2.242/hr vs $1.19/hr (1.88×), and possibly ~2× lower decode throughput
  from the bandwidth gap. Worst case that is ~3.8× the compute cost of the A100 assumption.
- **For us:** the **$210 idle line largely disappears.** An EC2 instance can be *stopped*,
  after which only EBS bills (~$16/mo for 200 GB gp3). `scripts/box_down.sh` will stop
  aggressively, and unlike RunPod the volume price does not double while stopped.

**Net: the envelope is at risk and is not yet resolvable by arithmetic.** `A6` produces the
measured generations/hour on this exact instance, which is the only honest input. Per
`EXECUTOR-TASKS.md` §9 trigger 3, if the measured figure pushes the reference-run block past
**Sep 13** it is an immediate report-back with proposed cuts, in the pre-decided cut order:
stretch list → zero-shot arm → BBH MC contrast task → GSM8K back to 500. **Never** the
token-budget arm (H1), the H6 negative controls, the noise floor, or the determinism audit.

## 4a. Cost reduction — what is available, and what is forbidden

The user has flagged cost minimisation as important. The honest answer has two halves,
because **this study deliberately buys identifiability with throughput.** Details and
provenance in `notes/reuse.md`.

### Available now, zero scientific cost

1. **Cap `max_model_len` at 8192.** Llama-3.1-8B's *full* trained context is 131072, which
   needs ~16 GiB of KV cache; the sibling project hit a hard `ValueError` on this. Our worst
   arm (BBH 3-shot CoT at 4× = ~1.5k prompt + 4096 gen) needs ~5.6k. Capping frees a large
   amount of KV cache. **Held fixed across all arms** — it is part of the frozen serving
   config.
2. **Resume + flush-to-S3 per batch**, adopted from `MedLLMSafety/run_batch_generation.py`.
   A crash 80% through a 2,719-item cell must not cost that 80%. Biggest single saver.
3. **One `generate()` call per batch**, letting vLLM's continuous batching schedule —
   short sequences do not wait on the batch's longest, unlike a padded batch.
4. **Stop the instance aggressively.** Stopped EC2 bills EBS only (~$16/mo), and unlike
   RunPod the volume price does not double while stopped.
5. **The judge cascade**, already in the plan: judge only where `rung2 ≠ rung3` or
   `rung2 == incorrect`. Judging is ~107k calls and the largest line after generation.
6. **Quantize once, reuse.** ~10–30 min per arm, and only ~1 GB VRAM (sequential layer
   onloading — the 16 GB requirement is *host* RAM, which is why 64 GiB was chosen).

### Forbidden — these would make the effects unidentifiable

| Knob | Throughput gain | Why we cannot |
|---|---|---|
| `enable_prefix_caching=True` | **Large for us** — 8-shot CoT prompts share a long identical prefix | Breaks greedy determinism (vLLM #39389); explicitly not covered by batch invariance |
| `VLLM_BATCH_INVARIANT=0` | 1.2–2× | It is the entire basis of H8 and of trusting any delta |
| `tensor_parallel_size>1` | scales | BI broken under TP>1 (#51290) |
| Sorting prompts by length | real | Changes batch composition, the exact thing #9567 shows moves outputs |

Turning these on to save money would forfeit the one thing that makes the spend worth
anything. Not a trade we can make.

### The one real opportunity: sweep `max_num_seqs` in `A7`

`max_num_seqs=8` is the **largest remaining cost driver** — it deliberately starves GPU
concurrency. But that value is inherited from a **vLLM issue report on someone else's
hardware** (#51187: 44 co-resident sequences → 3 distinct outputs in 5 repeats;
`max_num_seqs=8` → 5/5 identical), not from a measurement on our L40S/SM89 box.

**Proposal, folded into `A7` at near-zero marginal cost:** the determinism audit already
runs 3 identical repeats per format. Sweep `max_num_seqs ∈ {8, 16, 32, 64}` inside it and
take the **highest value that still yields bitwise-identical output across repeats.** If
batch invariance works properly on SM89, this could cut generation wall-clock several-fold.

This is strictly better than assuming the most conservative point: §3.3 already requires
`max_num_seqs` to be **reported in the paper**, and reporting the *frontier* — the largest
batch size at which quantized greedy decoding is bitwise reproducible, per format — is a
more useful result than reporting one arbitrary safe value. It converts a cost problem into
a finding the paper wants anyway.

**Sequencing note:** whatever value the sweep selects must be frozen **before** `A13` main
runs begin and held across every arm. Do not let it float per cell.

## 5. Safety and hygiene

- The `obtmodel` role is **admin** (`IibsAdminAccess-DO-NOT-DELETE`). Per the user's
  standing production-safety rules: prefer read-only operations, and **create only
  new resources**. **Nothing pre-existing in this account is modified, terminated, or
  deleted.**
- ⚠️ **SUPERSEDED by decision D11 (2026-08-25).** This section previously specified tagging
  `Project=QuantArtifact` and naming resources `quantartifact-*`. **That is withdrawn.** The
  user requires resource and job names to be **generic** and **reviewed by them before
  creation** — no paper, project, or venue identifiers in names or tag values. Names are to be
  *proposed* in the launch request, not applied.
- ⚠️ **The account is not settled** (D11). `670060057797` / profile `obtmodel` / `us-east-1` is
  where recon was done and quota verified, but the user may move this. Account, profile,
  region and bucket are **config, never hard-coded**. If the account changes, the §6 quota
  evidence must be re-verified — especially the 768-vCPU On-Demand G quota, which is the sole
  reason EC2 is viable here.
- All AWS calls made during `B2`/`B3` recon on 2026-08-23 were **read-only**
  (`sts get-caller-identity`, `service-quotas get/list`, `ec2 describe-*`,
  `pricing get-products`). No resource was created or changed.
- Termination protection to be enabled on the instance; the EBS root/data volume set to
  **not** delete on termination, so an accidental stop cannot lose generations.
- Generations are the artifact. Sync `artifacts/generations/*.jsonl.gz` to S3
  (bucket/prefix TBD per D11 -- generic naming, user-reviewed) after every cell completes.

## 6. AWS quota evidence (read-only, 2026-08-23, us-east-1)

```
EC2   On-Demand G and VT   L-DB2E81BA   768 vCPUs
EC2   On-Demand P          L-417A185B   768 vCPUs
EC2   Spot G and VT        L-3819A6DF    64 vCPUs
EC2   Spot P               L-7212CCBC   768 vCPUs
EBS   gp3 storage          L-7A658B76   461 TiB

SageMaker "for training job usage", single-GPU classes:
  ml.g6.2xlarge   1.0        <- only on-demand single-GPU training quota present
  ml.g6e.xlarge   (absent)
  ml.g6e.2xlarge  (absent)
  ml.g5.xlarge    (absent)
SageMaker other:
  Studio CodeEditor Apps on ml.g6e.xlarge   1.0
  ml.g6e.xlarge for endpoint usage          1.0
  ml.g6e.2xlarge for training warm pool     1.0
```

## 7. Provisioning requirements — non-negotiable

🚨 **A working C compiler must be installed AND KEPT on the instance.**
`VLLM_BATCH_INVARIANT=1` substitutes batch-invariant **Triton** kernels, and Triton
**JIT-compiles them at runtime**. No compiler ⇒ no BI kernels ⇒ no determinism ⇒ H8 dead and
every precision delta untrustworthy. The sibling project hit exactly this
(*"no C compiler" from Triton's JIT kernel build*) after a build toolchain was purged
post-`pip install`. So: install `build-essential`/`gcc`, do **not** clean it up, and have
`A4` assert a reachable compiler at startup alongside its existing checks that
`VLLM_BATCH_INVARIANT` is set and `tensor_parallel_size == 1`.

Also verify rather than assume, per `notes/reuse.md` §5: vLLM's own precompiled
`_C.abi3.so` links a specific CUDA runtime independent of the Python `torch` pin. Run
`objdump -p .../vllm/_C.abi3.so | grep NEEDED` on our pinned `vllm==0.26.0` and record the
result here before trusting the CUDA install.

## 8. Not yet done

- [ ] Launch the instance, pin CUDA + driver + torch + vLLM, record versions here.
- [ ] `nvidia-smi` output recorded (the `B3` "done when" criterion is a pod launched
      *and* destroyed; on EC2 the equivalent is launch → `nvidia-smi` → stop).
- [ ] `gcc` present and Triton JIT verified working (see §7).
- [ ] `objdump` CUDA-runtime check on `vllm==0.26.0`.
- [ ] Confirm batch invariance is actually supported on **SM89** — an `A7` finding, not a
      precondition.
- [ ] `scripts/box_up.sh` / `scripts/box_down.sh`.
- [ ] Confirm `g6e.2xlarge` capacity in at least two AZs at launch time (quota ≠ capacity).

---

## 9. SageMaker re-checked in the NEW account (2026-08-26) — correcting a gap

**My error.** §2 rejected SMInterface on verified evidence that SageMaker training-job quota
was **0 for every usable GPU** — but that was measured in the *old* account `670060057797`.
When the account moved to `654654594225` (D11) I re-verified **EC2** quota and **not**
SageMaker. D11 says a quota grant is account-specific; I applied that rule to one service and
not the other. The user caught it.

**Also worth stating plainly:** the decision not to use SMInterface was **not mine**. It is
`EXECUTOR-TASKS.md` **O4**, made by the planner before execution began. My contribution was
verifying its premise, which held for the old account.

### What the new account actually has (training-job quota, non-zero only)

| type | GPUs | VRAM | region | mode | quota |
|---|---|---|---|---|---|
| `ml.g6e.12xlarge` | 4 × L40S | 48 GB ea | both | **spot** | 1 |
| `ml.g5.12xlarge` | 4 × A10G | 24 GB ea | us-east-1 | on-demand | 2 |
| `ml.g6.2xlarge` | 1 × L4 | 24 GB | both | on-demand | 1 |
| `ml.g5.16xlarge` | 1 × A10G | 24 GB | both | spot | 1 |

⚠️ **CORRECTION (same day): `ml.g6e.2xlarge` DOES have training-job quota.**

| quota | us-west-2 | us-east-1 |
|---|---|---|
| `ml.g6e.2xlarge for training job usage` | **1** | **1** |
| `ml.g6e.2xlarge for spot training job usage` | **1** | **1** |

**My error, and it was the same mistake twice.** I queried with `--max-items 400`; SageMaker has
far more quotas than that, so the result was silently truncated and I concluded there was no
quota. The user spotted it in the console. Lesson: `list-service-quotas` for `sagemaker` must be
paginated fully — never capped.

**So no quota request is needed.** We have exactly the approved configuration: one
`ml.g6e.2xlarge` = 1 × L40S 48 GB, on-demand, with spot also available as a fallback.

SageMaker on-demand for `ml.g6e.2xlarge` is **$2.8035/hr** — a ~25% premium over EC2's $2.2421,
which buys queueing instead of `InsufficientInstanceCapacity`.

### The genuine advantage of the SageMaker path

**SageMaker training jobs QUEUE for capacity; EC2 `RunInstances` fails immediately.** That is a
real and material difference given L40S is currently exhausted in 10 AZs, and it is not
something O4 weighed — O4's objections were the 3600 s `max_run` default, the 30 GB volume
default, and batch-vs-interactive, all of which are configurable or workable.

### Consequences if we take it

- **`ml.g6e.12xlarge` gives the approved GPU** (L40S 48 GB) but bills 4 to use 1. Must pin
  `CUDA_VISIBLE_DEVICES=0` **and assert it at startup** — `tensor_parallel_size>1` breaks batch
  invariance (vLLM #51290), and a 4-GPU box is exactly where that mistake happens.
- **Spot means interruption.** Survivable: `A4` already checkpoints and resumes per batch, and
  the resume index is keyed on the full join tuple.
- **`max_run` must be raised well above the 3600 s default**, per O4's own warning.
- **A7 becomes a submitted script rather than an SSH session.** This is O4's real objection and
  it stands — but a queued batch job that runs beats an interactive box that never launches.

### Container: a custom ECR image is required, not optional

Verified 2026-08-26:

| | version |
|---|---|
| `vllm==0.26.0` pins | **`torch==2.11.0`**, `torchvision==0.26.0`, `torchaudio==2.11.0` (exact `==`) |
| Newest SageMaker PyTorch training DLC | **`torch 2.10.0`**, py313, **CUDA 13.0** |

So the stock DLC cannot host our pinned vLLM: torch is one minor version short and
pip-installing over it is the failure mode the sibling project already paid for — vLLM's
**precompiled** `_C.abi3.so` links a specific libcudart regardless of what the Python torch
package says (`notes/reuse.md` §5), and the DLC ships CUDA 13.0 against vLLM's `.so.12`
expectation.

**Two paths, and the choice is scientific, not just operational:**

1. **Custom image, `framework: generic`** — CUDA 12.x base, `torch==2.11.0`, `vllm==0.26.0`,
   build toolchain **retained** (Triton JIT-compiles the batch-invariant kernels at runtime, so
   no compiler means no determinism). Keeps `EXECUTOR-TASKS.md` §3.2's pinned engine exactly.
   Cost: build + ECR push.
2. **Downgrade to `vllm==0.18.0`**, which pins `torch==2.10.0` and matches the DLC — the version
   the sibling project actually ran. Cheaper operationally, but the engine version is **frozen
   and reported in the paper** (§3.2), and `VLLM_BATCH_INVARIANT` behaviour differs across
   versions (PR #27660 made CUDA graphs BI-compatible; the `auto_gptq.py`/`auto_awq.py` kernel
   guards H8 is built on are version-specific). **Changing it is a planner decision, not mine.**

Path 1 is the default unless the planner rules otherwise.

---

## 9. `verify` stage — measured on the box, 2026-08-26

Job `johnyj02-QA-v0-1-20260826-33b4267e-verify`, `ml.g6e.2xlarge`, us-east-1. Ran 2 min.
**13 of 14 checks passed.** SageMaker reports the job `Failed` because `sm_verify.py` exits
non-zero on any failed check — that is the exit code, not a lost run; the report was written
and every measurement below is from it.

### 🎉 The CUDA-13 risk is RESOLVED — the engine version stays frozen

This was the one open risk that could have forced a planner decision (§4 of `CLAUDE.md`).
MedLLMSafety hit `"NVIDIA driver ... too old (found version 12080)"` on `ml.g5.2xlarge` and had
to fall back to `vllm==0.18.0`. We do not:

| fact | measured |
|---|---|
| GPU | NVIDIA L40S, **46068 MiB** (not the nominal 48 GB) |
| driver | **595.91.07** |
| compute capability | **8.9** — SM89 Ada, as planned |
| torch | `2.11.0+cu130`, built for CUDA **13.0**, `available=True` |
| vLLM linkage | `libcuda.so.1`, **`libcudart.so.13`**, `libtorch_cuda.so` |
| gcc | Debian **14.2.0**-19 at `/usr/bin/gcc` |
| Triton | **3.6.0** — compiled and ran a kernel |

So `vllm==0.26.0` is kept as frozen and reported. No engine downgrade, and H8's
version-specific kernel guards stay valid.

**The C compiler survived into the runtime image**, which was the point of not purging
`build-essential`: `VLLM_BATCH_INVARIANT` works by substituting batch-invariant *Triton*
kernels, JIT-compiled at runtime, so no compiler would mean no determinism.

⚠️ Triton compiling a kernel is **necessary but not sufficient** for batch invariance.
`models.yaml:bi_supported_on_sm89` stays `null` — **A7 still owns that finding.** The 1.2–2×
BI overhead figure in `EXECUTOR-TASKS.md` §3.3 is an Ampere number and does not transfer to SM89.

### The one failure was my check, not the box

`vllm_extension_loads` did `import vllm._C` → `ModuleNotFoundError`. **vLLM 0.26.0 does not
expose its extension at that import path** (ops register through `torch.ops`), and `objdump` had
already read the `.so`'s `NEEDED` entries successfully in the check immediately above — so the
file existed and linked fine. Fixed in `5ada832`: load the `.so` directly with `ctypes.CDLL`,
which is precisely "does this binary resolve against the installed CUDA runtime and driver"
with no assumption about how vLLM surfaces it in Python. Same commit fixed the version table
printing `?` for `math_verify` (no `__version__`; use `importlib.metadata`).

🚧 **Both fixes are UNEXERCISED ON GPU.** They gate jobs 2+ (which serve with vLLM), not job 1
(the quant image has no vLLM). Fold a re-verify into the next serving-image job rather than
spending a queue slot on `verify` alone.

### The account is SHARED — the one job slot is not ours alone

`list-training-jobs` shows another user's jobs (`jathupra-*`) in this account. The
`ml.g6e.2xlarge` quota of **1 is account-level**, so a concurrent job from them contends with
ours for the only slot. Nothing was in flight when job 1 was submitted, but this is a real
scheduling hazard and an argument for submitting promptly whenever the slot is free.

---

## 10. `llmcompressor` 0.12.0.1 recipe API — verified against the pinned tag

The first `quantize` smoke run (196 billable seconds, ~$0.15) failed with:

```
ValidationError: 2 validation errors for GPTQModifier
group_size   Extra inputs are not permitted [input_value=128]
desc_act     Extra inputs are not permitted [input_value=False]
```

Everything before quantization worked: model fetch (434 weights), `ultrachat_200k` download,
8-example calibration map. **This is the smoke run paying for itself** — the same failure in the
real job would have cost a 3.5-hour queue-and-run cycle.

All of the below was read from the source at tag **`0.12.0.1`** (and
`compressed-tensors` **`0.17.1`**), not inferred.

### Where `group_size` actually lives

Neither modifier takes it. It is inside the preset scheme:

| preset | resolved weight args |
|---|---|
| `W4A16` | `num_bits=4, type=int, strategy=group, group_size=128, symmetric=True` |
| `W8A16` | `num_bits=8, type=int, strategy=group, group_size=128, symmetric=True` |
| `W4A16_ASYM` | as `W4A16` but `symmetric=False` |

Those happen to match `configs/quant.yaml` exactly — but the values we *hash* into
`quant_config_hash` come from our YAML, so a future `compressed-tensors` could change what
`W4A16` means and the weights would change while the hash did not. `assert_preset_matches_spec`
therefore resolves the preset and **asserts** it equals the config.

### 🚨 `desc_act` does not exist, and its replacement fails SILENTLY

The nearest equivalent is `actorder`, and there are **two** of them with **different defaults**:

| | default |
|---|---|
| `QuantizationArgs.actorder` (in the scheme) | `None` |
| `GPTQModifier.actorder` (the modifier) | **`Sentinel("static")`** |

`resolve_quantization_config` treats the sentinel as *deferential*: it injects
`ActivationOrdering.STATIC` whenever the scheme's actorder is `None`. Since "explicitly `None`"
and "never set" are indistinguishable to it, **omitting `actorder` yields static activation
reordering** — which is neither `desc_act: false` nor the `desc_act: true` we reject.

`llmcompressor`'s own W4A16 example omits `actorder`, so **the canonical tutorial recipe
produces static-actorder weights.** Anchor A is a *blocking* gate comparing our accuracy against
Qwen's official `desc_act: false` checkpoint, so inheriting that default would have quietly
invalidated it with every test still green. `gptq_actorder()` therefore always passes `actorder`
explicitly, and `None` is the only value meaning `desc_act: false`.

Semantics: `ActivationOrdering.GROUP` (alias `"dynamic"`) is classic `desc_act=True` with a
runtime group index; `STATIC` is an alias for `WEIGHT`, a statically applied permutation.

Passing `actorder` on *both* the modifier and the scheme raises `ValueError` if they differ, so
it is set in exactly one place: the modifier.

### `AWQModifier` is not a class

In 0.12.0.1 `llmcompressor.modifiers.awq.AWQModifier` is a **compatibility function** that splits
its kwargs and returns `[AWQTransformModifier(...), QuantizationModifier(...)]`. It routes
`config_groups`, `targets`, `ignore`, `scheme`, `kv_cache_scheme` and observer options to the
quantization half. `build_recipe` therefore returns a **list** for every method, so callers never
depend on which arm produced how many modifiers.

Non-deprecated import paths at this tag: `llmcompressor.modifiers.gptq` (the
`modifiers.quantization.gptq` path is a warning shim) and `llmcompressor.modifiers.transform.awq`.

### Process correction: a smoke must cover every code path

The first smoke ran `gptq-4bit` only. GPTQ and AWQ use genuinely different APIs here, so a
green GPTQ smoke would have said nothing about AWQ — which would then have failed nine arms into
the real job. `sm_quantize.py --smoke` now selects **one arm per method** and asserts it covers
every offline method present in the config.

### Job 1, attempt 1: 5 of 9 arms, then GPU OOM — one process per arm

`johnyj02-QA-v0-1-20260826-33b4267e-quantize`, 4546 s (76 min, ~$3.54). **Not a timeout**:
`max_run` is 21600 s and only 4546 s was used.

Produced (and persisted to S3, 14.55 GB): `qwen2.5-3b` × {gptq-8bit, gptq-4bit, awq-4bit},
`phi-3.5-mini` × {gptq-8bit, gptq-4bit}. Failed with `OutOfMemoryError: Sequential pipeline ran
out of memory`: `phi-3.5-mini/awq-4bit` and all three `llama-3.1-8b` arms.

**The checkpointing earned its keep.** All five successful arms survived in
`checkpoint_s3_uri`, are restored on the next run, and are skipped by recipe hash — so the retry
does 4 arms, not 9.

#### It was a leak, not a capacity limit

The log reported `Accelerator 0 | usage: 95.70% | total memory: 47.7 Gb`. A *fresh* Llama-8B GPTQ
pass should need roughly **10 GB** of that 47.7 GB card:

| | |
|---|---|
| one decoder layer, bf16 | ~436 MB (8.03 B params − 2 × 525 M embedding, over 32 layers) |
| Hessians for that layer | ~1.2 GB (6 × 4096² + 14336², float32) |
| cached calibration activations | ~8.6 GB (512 × 2048 × 4096, bf16) |

So 96% was accumulated state from earlier arms. The failure *order* confirms it: every arm fine,
then phi's AWQ (the most memory-hungry), then Llama's 8-bit got partway, then the last two failed
within seconds — the signature of a monotonic leak, not of a model that is simply too big.

Python cannot reliably release it: `del model` leaves torch's caching allocator holding blocks,
llmcompressor installs hooks on the model, and one arm had already logged
`Exception during finalizing modifier: Failed to compress 7 modules`, so the next arm inherits a
poisoned lifecycle. **Process exit is the only teardown that always works.**

`sm_determinism.py` and `sm_throughput.py` already used one-process-per-cell for exactly this
reason. Applying that to vLLM but not to llmcompressor was an inconsistency, and it cost 76
minutes and four arms.

⚠️ **Deliberately did NOT set `sequential_targets="Linear"`**, which the error message suggests.
That changes the granularity at which GPTQ accumulates Hessians within a decoder layer, and so
potentially the weights. Process isolation is semantically neutral; a granularity change is not,
and the recipe is frozen and reported in the paper.

#### Two smaller fixes the same failure exposed

- **Subprocess output is INHERITED, not captured.** `capture_output=True` buffers until the child
  exits, so a 30-minute arm would emit nothing to CloudWatch and look like a hang — and a killed
  job would lose the child's output entirely, which is the diagnostic that found this bug.
- **`--smoke` never skips.** A previous rehearsal had already written those exact hashes, so
  normal resume semantics would have skipped every arm, reported success, and proven nothing —
  worse than not running, because it looks like a pass.
