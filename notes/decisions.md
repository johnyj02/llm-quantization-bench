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

**Quota is already in place — no request needed.** `ml.g6e.2xlarge for training job usage = 1`
in both us-west-2 and us-east-1 (plus spot = 1). I initially reported otherwise because I capped
`list-service-quotas` at `--max-items 400` and SageMaker has more quotas than that, silently
truncating the result; the user caught it in the console. **Always paginate that call fully.**

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

---

## 2026-09-04

### D15 — 🚨 BUDGET IS NOT A CONSTRAINT. Spend to raise acceptance odds. (user, 2026-09-04)

The user's words: *"dont worry about the budget; its more important to increase the chances of
acceptance hence feel free to make as many submissions as required"*.

**What this changes.** Cost stops being a reason to hesitate, to batch work into fewer jobs, or to
ask before a rehearsal. Ledger at the time: 42 jobs, 25.70 h, **$71.95** against the $370–440
envelope of `EXECUTOR-TASKS.md` §2.6, so roughly 18% used. R20.1's *"probe gets ≤15% of remaining
budget"* rule was already lifted; this goes further and lifts cost as a consideration entirely.

⚠️ **What it does NOT change, and the distinction matters.**

1. **D10 still governs.** Every billable command still requires `--i-have-confirmation` and still
   offers a zero-cost inspection mode. D15 is a standing *authorisation to submit*, not permission
   to remove the gate: the flag exists so a job cannot start by accident, including by me.
2. **`--smoke` first, every time.** That rule was never about money. With one queue slot, a broken
   multi-hour job costs a queue cycle and a day, and no budget makes that back.
3. 🚨 **`main-v1` is still special, and NOT because of cost.** It is the pre-registration freeze
   line: from its first generation, `analysis/preregistration.md` §1–§9 become append-only. That
   is a **scientific** irreversibility and D15 does not touch it. Anything to be registered must
   land first.
4. **Wall clock is now the binding constraint**, and it always really was. ARR is **2026-10-12**,
   queue waits have run 1 h to 6 h 19 m on a shared single slot, and only one job can exist at a
   time. So the freedom D15 grants is best spent on jobs that *unblock* work rather than on jobs
   that merely add polish.

**How to spend it, in the order that most affects acceptance** (from `notes/scope-audit.md`):

| priority | why it moves acceptance |
|---|---|
| **`A15` token-budget sweep** | H1 is the **lead hypothesis, "never cut"**, and its instrument does not exist. ~25,000 generations, previously uncosted. This is the single largest acceptance risk that money can fix. |
| **H6's log-likelihood arms** | The **"never cut" negative control** (§3c). Also unblocks G3's ARC cross-validation. Needs a build, not just a job. |
| **`A18` zero-shot arm (H5)** | 9 cells, currently `prompt_variant` is hardcoded to few-shot. |
| **`A14` pipeline noise floor** | §4.5 calls it mandatory, and it is the self-application of our own standard that earns reviewer trust. |
| **Extra orderings / replicates** | Cheap precision on H8, which is already the strongest result. |

⚠️ **The largest acceptance risks are NOT purchasable.** An empty Limitations section is an
automatic desk reject and costs only writing; the judge and annotation track needs the co-author's
calendar; the Oct 14 reviewer registration is per-author and unbuyable. D15 does not help with any
of them, and it must not become a reason to prefer compute over writing.

### D16 — 🚨 NO DOCUMENT FREEZING, and no dated amendments (user, 2026-09-05)

The user's words: *"stop freezing documents is what I meant; which means that dont claim something
then after a while of the user reading the doc say oh we made a change to what we claimed in the
begining and make a dated ammendement"*.

**The objection is about READABILITY, not about rigour**, and it identifies a real defect I had
created. `analysis/preregistration.md` said in §9f that the invocation floor came from the K=2
cells. That turned out to be degenerate, and I recorded the correction as an appended amendment
**700 lines later**. So a reader reached §9f, believed it, and had no way of knowing it was
superseded without reading to the end. The document contradicted itself.

**The rule now:** a correction goes **in place, in the section that makes the claim**, with a dated
note saying what it used to say and why it changed. §1.4 already did this correctly; the appended
amendment was the wrong pattern and has been folded into §9f and deleted.

⚠️ **What this does NOT relax, and the distinction is the whole point.** Every decision still
states **whether it was made before or after seeing the relevant data**, inline, with the numbers
quoted. That property is independent of freezing:

- The failure mode being guarded against is *choosing an analysis because you have seen which
  choice flatters the result.* A calendar never detected that; stating provenance does, and the git
  history is an independent record either way.
- 🚨 **"Not frozen" means corrected VISIBLY, never corrected quietly.** Silently rewriting a section
  to match a result, with no note that it changed, is the one thing the document exists to prevent.

**Why this is defensible rather than a weakening.** Five corrections have been made so far and
**four made a claim harder to sustain**: §1.4 admitted H3 was not immune to a selection effect, §9
recorded that an escalation trigger fired, and §9f moved a floor **up from zero**. A process whose
corrections consistently tighten the standard is not one that needs a calendar to be trusted.

⚠️ **The residual risk, stated plainly.** `EXECUTOR-PLAN.md` §4.0 warns that a reviewer who judges
§1's identification fix to have been patched rather than specified in advance caps acceptance at
~25%, and that risk is real because the bias §1 corrects points **toward** this paper's own
hypothesis. Without a freeze, the defence is the inline provenance notes plus the git history
rather than a single timestamp. I raised this once; the user's decision stands and this records it.

**And REPEATING AN EXPERIMENT is acceptable, which is what makes the rest of this rule usable**
(user, 2026-09-05: *"its fine if we need to repeat some experiments"*).

This completes D16 rather than decorating it. The pressure that produces a hedged document is
wanting to avoid a correction **because the correction would invalidate a run**. If a re-run is
expensive or unwelcome, there is a standing incentive to write a claim loosely enough that no
result can contradict it, and a loosely written claim is worth nothing to a reviewer. Because
repeating an experiment is acceptable, a claim can be written **precisely** and corrected
**precisely**, which is the only version of "not frozen" that is an improvement rather than a
weakening.

⚠️ **So the correct response to discovering a claim is wrong is never to soften the claim.** It is
to correct the section, state what it used to say, and re-run whatever the correction invalidates.
Three of the five corrections already recorded worked this way: R24.1 moved `max_num_seqs` 64 → 32
and re-ran the template probe and the 10-ordering H8 measurement rather than reporting numbers from
the abandoned setting, and R34 changed where `until` is applied and re-ran the reference rehearsal
rather than reinterpreting records taken under the old semantics.

⚠️ **What this does not license.** D10 still requires explicit confirmation for every billable job,
`--smoke` still comes first, and **wall clock is still the binding constraint** with ARR on
2026-10-12 and one shared queue slot. So a re-run is acceptable on *cost* grounds and still has to
be justified on *schedule* grounds. A correction that forces a re-run of the reference condition
late in September is a different proposition from one that forces a $2 probe, and the honest move is
to say which it is when raising it.

### D17 — A gate's check must name the intent it protects, inline (R42.3, 2026-09-05)

**The rule.** Every automated gate criterion carries, next to the code, the specific failure it
exists to prevent. Not a description of what it measures, but a statement of what goes wrong if
it passes when it should not.

**Why, and this is the fourth instance of one pattern rather than a new concern.** This project's
recurring defect is *declared versus enforced*: a value stated in a config and never forwarded
(`desc_act`, `async_scheduling`), a value forwarded but absent from the identity used to skip work
(`max_num_seqs` in `decoding_fingerprint`), a terminator table retyped from config and left to
drift (`HARNESS_UNTIL`). Gate G6 then produced the same shape **one level up, in the instrument
that checks the deliverable**:

- it compared **total** PDF pages against a limit that applies to the body only, which
  over-counted the moment Limitations existed and nearly caused body content to be cut to make
  room for content that never competed for the space;
- its Limitations check was `grep -q 'section\*{Limitations}'`, testing that the *heading*
  existed, and it reported PASS while five required sections carried zero prose. An empty
  Limitations section is a desk reject exactly as a missing one is.

In both cases the intent was right and the implementation tested something weaker that happened
to pass. A comment naming the intent is what lets the next reader notice the gap by reading,
rather than by the gate failing at the moment it matters.

**What this does not license.** It is not a substitute for the check being correct. Two of the
checks added under this rule were themselves wrong on first writing: the `\todo` counter reported
6 where 4 were real, because it subtracted one literal comment form instead of stripping comments
per line. A comment explaining the intent would not have caught that; running it against ground
truth did. Both are required.

**Corollary, from the same ruling.** Where a gate protects an *external* rule, verify the rule at
its source rather than from recall, including the planner's recall. R42 asked the planner whether
the 8-page limit was tradeable; it agreed with me and then declined to be the authority, on the
grounds that *"the symmetric error is cutting content against a constraint that isn't there."*
Checking the CFP directly then falsified two beliefs we both held: that Acknowledgments is
excluded from the count, and that a ninth page is granted on acceptance.
