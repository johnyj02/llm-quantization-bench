# Decision log

Decisions confirmed by the user, with dates. Anything here overrides my own reading of the
two plan documents. Amendments to `EXECUTOR-TASKS.md` overrides are noted as such.

---

## 2026-08-23

### D1 — Substrate: EC2 `g6e.2xlarge`, user's AWS account
1 × L40S 48 GB (SM89), 8 vCPU, 64 GiB RAM, account `670060057797`, us-east-1, profile
`obtmodel`. **Amends the premise of override O4** (AWS quota latency), while reinforcing its
conclusion (SMInterface/SageMaker is not viable — training-job quota is 0 for every usable
GPU). Full record: `notes/compute.md` §2.

### D2 — Annotation window confirmed
Co-author can commit **~6 hours between Aug 26 and Sep 5**. `A11` runs at **full design**:
250 adjudicated labels, ≥120 double-annotated, Gwet's AC1 primary. No fallback to the
150+80 reduced design.

### D3 — NC-licensed judge weights approved for use
`IAAR-Shanghai/xVerify-9B-I` (CC-BY-NC-ND-4.0) may be **run** with numbers reported;
weights and derivatives **never redistributed**. Remains primary judge;
`ibm-granite/granite-4.1-8b` (Apache-2.0) second. `A24` must exclude xVerify weights from
the artifact release.

### D4 — Local disk: regenerable caches may be cleared
Standing permission for `~/.cache` and `~/Library/Caches` only. **Never** `~/Desktop`,
`~/workspaces`, `~/Documents`. Reclaimed 3.9 GB on 2026-08-23 (5.2 → 11 GB free).

### D5 — HF model cache: archive to dev desktop, then clear
13 GB of `BioMistral-7B` + BERT variants from other projects. **Copy to
`dev-dsk-johnyj02-2a-53853bae.us-west-2.amazon.com:~/hf-cache-archive/`, verify, then
delete locally.** Verification before deletion is mandatory, not optional.

### D6 — GSM8K: run BOTH task variants (option C)
**The decision stands. Its stated basis is VOID — superseded by ruling R1 on 2026-08-25.**

- ✅ **Still true:** both variants run, on the same 1319 items; +19,785 generations (~20%)
  above the locked §2.2 matrix, explicitly approved.
- ❌ **Void:** the primary/secondary split and its justification. I claimed `gsm8k` (5-shot)
  was non-CoT so H1 needed `gsm8k_cot` as primary. **Verified false 2026-08-25:** `gsm8k` has
  `fewshot_split: train` with `doc_to_target: "{{answer}}"`, and GSM8K train answers are
  multi-step worked solutions with `<<…>>` annotations ending `#### N`. **So `gsm8k` at 5-shot
  is also chain-of-thought and demonstrates `####` five times.** "H1 needs CoT" does not
  discriminate between the arms.
- ✅ **Replaced by (R1.1):** neither arm is primary. Register **`answer_marker ∈ {hash,
  phrase}`** as a two-level factor, **both confirmatory**, with the pre-registered directional
  prediction `interaction(hash) > interaction(phrase) ≈ 0`. The arms hold items, gold, task
  and competence ~constant and vary **only marker naturalness** — a near-clean manipulation of
  *compliance difficulty*, which is the quantity the whole competence/compliance split rests
  on. A near-null on `phrase` becomes a **confirmed prediction** rather than a disappointing
  primary.

See `notes/planner-rulings.md` R1 and its verification table.

### D7 — Author list frozen at two
The user plus one co-author. Settled **before** annotation begins, as required. Both must
register as ARR reviewers/ACs by **Oct 14** or the paper is automatically desk-rejected.

### D8 — Co-author contact goes through the user
I do not contact the co-author directly. **Everything the co-author needs must be produced
as a self-contained, hand-off-ready document** for the user to relay — notably the `B5`
annotation rubric, the 10-item dry-run instructions, and the annotation tool with its
instructions. Write them for someone who has not read either plan document.

### D9 — Cost: track and estimate, do not optimize
The ~$400 envelope in `EXECUTOR-TASKS.md` §11.2 is **not a binding constraint for now**.
Maintain a running spend figure and a forward estimate in `notes/estimates.md`; do not trade
scientific quality for cost, and do not block on cost. **The forbidden-knobs table in
`notes/compute.md` §4a still stands** — those are excluded for identifiability, not for
money. `A6` still produces the measured estimate.
⚠️ Still report per `EXECUTOR-TASKS.md` §9 trigger 3 if measured throughput pushes the
reference-run block past **Sep 13** — that is a *schedule* trigger, not a cost trigger, and
it survives this decision.

### D10 — 🚨 HARD GATE: no job starts without explicit user confirmation
> **"after writing all code and before starting the job you need to ask confirmation from
> me"**

Binding standing rule. **All code is written and reviewable BEFORE any GPU job, any
generation run, any quantization run, or any billable compute is started.** I ask, the user
confirms, then I run. This applies to:
- launching the EC2 instance
- `A6` smoke test and throughput measurement
- `A7` determinism audit
- `A8` pilot run
- `A9` anchors
- `A13` main reference runs, and every arm after it

Zero-compute work continues without asking: writing code, unit tests over fixtures, configs,
prose, prereg, related work, and read-only AWS/HF queries.

## 2026-08-25

### D11 — 🚨 The D10 ask must come BEFORE naming anything, and the account may change
> **"before you run any job you must ask me the human. Cause I want to make sure the job names
> are generic etc.. and possibly change the account"**

Extends D10 with two specifics:

1. **Resource and job names must be reviewed by the user before creation, and must be
   generic.** ⚠️ This **overrides** `notes/compute.md` §5, which currently proposes tagging
   `Project=QuantArtifact` and naming resources `quantartifact-*`. That is exactly the kind of
   project-identifying name the user wants to vet. **Propose names, do not apply them.**
   Default proposal going forward: neutral names with no paper/project/venue identifiers, and
   no tag values that describe the research.
2. **The AWS account is not settled.** `670060057797` (profile `obtmodel`) is where the recon
   was done and where quota was verified, but the user may move this elsewhere. So:
   - Do **not** hard-code the account, profile, region, or bucket anywhere. They are config.
   - `notes/compute.md`'s quota evidence is **account-specific** and must be re-verified if the
     account changes — in particular the 768-vCPU On-Demand G quota, which is the entire reason
     EC2 is viable here.
   - The S3 sync target (`s3://sagemaker-us-east-1-670060057797/...`) is provisional.

**Consequence for the first ask:** the launch request I bring to the user must include the
proposed instance name, tags, key-pair name, security-group name, volume name, and S3 prefix
**as a reviewable list**, plus the account/region/profile in use — not as a fait accompli.

### D11 RESOLVED 2026-08-26 — account moved, naming scheme specified

**Account: `654654594225`** (not `670060057797`). ⚠️ **CONDUIT-managed, not Isengard.**
`ada` defaults to `isengard` and returns a misleading *"Account not found"*; it also no longer
falls back to the role name for Conduit, so both flags are required:

```
ada credentials update --provider conduit --account 654654594225 \
    --role IibsAdminAccess-DO-NOT-DELETE --profile qa-gpu --once
```

**Quota re-verified in the new account, and it DID differ** — which is exactly why D11 requires
re-verification rather than assuming the grant travels:

| region | On-Demand G | `g6e.2xlarge` AZs | SageMaker default bucket |
|---|---|---|---|
| **us-east-1 (chosen)** | **640** vCPUs | 4 | **already exists** |
| us-west-2 | 768 vCPUs | 4 | not yet created |

640 vs the previous account's 768. Both are far above the 8 vCPUs `g6e.2xlarge` needs, so it is
not binding — but the number is different and is now recorded against the right account.

**Naming scheme, user-specified:** `testing-QA-johnyj02-<thing>-<date>`, where **`QA` is the
project initials only and is NEVER expanded**. With `testing` in front, the whole string reads
as quality-assurance testing — a human scanning resource names, job ids, tags or S3 keys should
learn nothing about the research. Same rule for S3: `johnyj02/testing/QA/`.

**REVISED by the user later the same day to:** `johnyj02-QA-v<x.x>-<date>-<uuid>`

- instance `johnyj02-QA-v0.1-20260826-33b4267e`
- security group / key pair / volume: the same base plus `-sg` / `-key` / `-vol`
- S3 `s3://sagemaker-us-east-1-654654594225/johnyj02/QA/v0.1/<date>-<uuid>/`
- tags `Owner=johnyj02`, `Purpose=testing-QA`, `AutoStop=true`
- `v<x.x>` is **derived from `src/quantartifact/__init__.py`** so the version cannot drift from
  the package.

⚠️ **The UUID is generated ONCE and persisted to `.box-run-id` (gitignored).** A fresh UUID per
invocation would break `box_up.sh`'s idempotency: it finds an existing instance by name, so a
rolling name means every re-run launches *another* $2.24/hr instance while the first keeps
billing unnoticed. Two guards were added because of this:
1. `box_up.sh` **also** looks up by `Owner`+`Purpose` tags, ignoring the name, and **refuses**
   if a box exists under a different name — overridable only with an explicit
   `ALLOW_SECOND_BOX=1`. This catches a lost `.box-run-id`, and also simply running on a later
   date.
2. `box_down.sh` looks up **by tag, not by name**, and stops *all* matches. A stop script that
   silently finds nothing because the date rolled over is how an instance runs all weekend.

**Do not expand "QA", and do not add a project/paper/venue/model/benchmark name to any name,
tag, path or job id.** This supersedes the withdrawn `Project=QuantArtifact` / `quantartifact-*`
proposal.

**GPU, since the user asked:** `g6e.2xlarge` = **1 × NVIDIA L40S 48 GB** (SM89 Ada), 8 vCPU,
64 GiB RAM, $2.242/hr. Not an A100: AWS has no single-A100 instance, and A100-80G exists only as
`p4de.24xlarge` (8 GPUs, ~$40/hr, spot quota 0). L40S 48 GB is the largest single-GPU option and
`EXECUTOR-TASKS.md` §3.1 already names L40S as approved.

## 2026-08-26

### D12 — `B2` and `B4` are the user's, settled, and CLOSED. Do not re-raise.
The user has confirmed both. I raised them repeatedly after they were answered; that stops.

**The user is also factually right that neither is needed now**, and the plan's day-1 framing
overstated the urgency:

- **`B4` external readers — CLOSED, readers are ready.** The draft does not go out until
  **Sep 30**, with feedback due **Oct 4**. The only reason `EXECUTOR-TASKS.md` §4 put it on day
  1 was calendar latency for a third person's late-September time. That risk is retired.
- **`B2` OpenReview — the real deadline is Oct 14**, within 48 h of submission, and it is
  entirely post-experiment. Profile completeness is a prerequisite for registration, not a
  precondition for any experimental work. The §4 day-1 framing was about the desk-reject
  consequence, not about timing.

**Still true and still non-negotiable:** missing the **Oct 14** reviewer registration is an
**automatic desk reject**, and the obligation is per-author. It stays on the calendar in
`notes/arr-obligations.md` §5 as a hard date. But it is the user's to execute and needs no
further prompting from me.

### D13 — Route questions to the PLANNER, not to the user
> **"Any questions and issues you have ask the planner."**

Default channel for anything I cannot settle from the repo: `./ask-planner`. Reserve the user
for (a) job/compute approval under D10/D11, (b) decisions only they can make (money, people,
accounts, credentials), and (c) reporting results.

Cost discipline, from the tool's own instructions: **each call re-sends the planner's entire
context, so batch related questions into one call** rather than firing several, and do not ask
it anything answerable by reading the repo. Use `FORK=1 ./ask-planner` for throwaway or
exploratory questions so its history stays clean.

Treat its answers as **authoritative on planning decisions**. If one contradicts my read of the
repo, say so and ask again — never quietly pick my own way. And per its own standing
instruction, **verify claims it flags as unverified** (it was right on all three in R1–R4, but
two of my own premises were wrong, so the check runs both ways).

### D14 — 🚨 SMInterface ONLY. Override O4. (user, 2026-08-26)
> **"We will only use SMInterface override the planner change all references and codes etc.. That
> is a must. with respect to quota if we dont have it i can get it that is not a issue."**

**`EXECUTOR-TASKS.md` O4 is OVERRIDDEN by user decision.** O4 declared the rented/EC2 box the
substrate and SMInterface "a documented non-goal". That is reversed. SMInterface is now the
**only** compute path.

**Rationale that makes this correct rather than merely instructed:** SageMaker training jobs
**queue for capacity**; EC2 `RunInstances` fails immediately. `g6e.2xlarge` was unavailable in
**10 AZs across three regions**, so the EC2 path had no way to make progress. O4 never weighed
queueing — its objections were the 3600 s `max_run` default, the 30 GB volume default, and
batch-vs-interactive, all of which are configurable or workable.

**Quota is not a blocker — the user will obtain it.** Needed:
`ml.g6e.2xlarge for training job usage` ≥ 1 (currently only a warm-pool retention allowance).

**Consequences, all now mandatory:**
- EC2 provisioning is WITHDRAWN. `scripts/box_*.sh` are retained for provenance but are
  **non-goals**; do not run them. `notes/compute.md` §1–§8 describes a withdrawn plan.
- `A7`'s determinism audit becomes a **submitted batch job**, not an SSH session. This was O4's
  real objection and it stands as a cost, not a blocker.
- `max_run` must be raised far above SMInterface's 3600 s default, and `volume_size` above its
  30 GB default (we need ~200 GB).
- ⚠️ **SMInterface's `session.refresh_credentials()` hardcodes
  `ada credentials update --profile=<p> --once` with NO `--provider conduit`.** This account is
  Conduit-managed, so auto-refresh WILL fail. Set `auto_refresh_credentials: false` and refresh
  manually before submitting.
- Role in the new account is **`AmazonSageMakerAdminIAMExecutionRole`**, not the
  `SageMakerExecutionRole` in SMInterface's own example (that was the old account).
- vLLM 0.26.0 pins torch exactly, and the PyTorch DLC ships a different torch. The sibling
  project hit this and solved it with a custom image (`MedLLMSafety/Dockerfile.vllm`). Expect
  `framework: generic` + an ECR image rather than the stock DLC.
