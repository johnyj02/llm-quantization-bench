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

## 5. Safety and hygiene

- The `obtmodel` role is **admin** (`IibsAdminAccess-DO-NOT-DELETE`). Per the user's
  standing production-safety rules: prefer read-only operations, and **create only
  new, clearly-namespaced resources**. Everything this project creates is tagged
  `Project=QuantArtifact` and named `quantartifact-*`. **Nothing pre-existing in this
  account is modified, terminated, or deleted.**
- All AWS calls made during `B2`/`B3` recon on 2026-08-23 were **read-only**
  (`sts get-caller-identity`, `service-quotas get/list`, `ec2 describe-*`,
  `pricing get-products`). No resource was created or changed.
- Termination protection to be enabled on the instance; the EBS root/data volume set to
  **not** delete on termination, so an accidental stop cannot lose generations.
- Generations are the artifact. Sync `artifacts/generations/*.jsonl.gz` to S3
  (`s3://sagemaker-us-east-1-670060057797/quantartifact/`) after every cell completes.

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

## 7. Not yet done

- [ ] Launch the instance, pin CUDA + driver + torch + vLLM, record versions here.
- [ ] `nvidia-smi` output recorded (the `B3` "done when" criterion is a pod launched
      *and* destroyed; on EC2 the equivalent is launch → `nvidia-smi` → stop).
- [ ] `scripts/box_up.sh` / `scripts/box_down.sh`.
- [ ] Confirm `g6e.2xlarge` capacity in at least two AZs at launch time (quota ≠ capacity).
