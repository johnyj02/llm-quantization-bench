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
