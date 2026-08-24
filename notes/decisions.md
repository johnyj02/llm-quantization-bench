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
Resolves the conflict in `notes/harness-configs.md` §3.
- **`gsm8k_cot` (8-shot CoT) is the PRIMARY reference** — `EXECUTOR-TASKS.md` §2.2 locks the
  reference format as few-shot CoT, and H1 (the lead hypothesis) needs chains of thought to
  truncate.
- **`gsm8k` (5-shot, non-CoT) is a pre-registered SECONDARY variant** — carries the `####`
  regex, makes **Anchor C** reproducible.
- The comparison between them is itself a pre-registered result.
- **Scope increase: +19,785 generations (~20%) above the locked §2.2 matrix.** Explicitly
  approved. Recorded in `analysis/preregistration.md` §5.

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
