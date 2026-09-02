# CLAUDE.md — operational instructions for this project

**Read this before touching compute.** It records the things that are not derivable from the
code and that cost real time to rediscover. Scientific design lives in `EXECUTOR-PLAN.md`;
scope/schedule in `EXECUTOR-TASKS.md`; decisions in `notes/decisions.md`. This file is
**how to actually run things**.

---

# 🚨🚨 ASK THE PLANNER AND THE CODE REVIEWER. DO NOT SKIP THEM. DO NOT PRE-EMPT THEM. 🚨🚨

**TWO AGENTS EXIST AND BOTH ARE MANDATORY IN THEIR OWN DOMAIN.**

| WHEN | WHO | HOW |
|---|---|---|
| **ANY SCIENCE QUESTION**: a hypothesis, an estimator, a threshold, a scope change, a registered design, anything that becomes a number or a claim in the paper | **THE PLANNER** (D13, rule 5) | `./ask-planner "..."` |
| **ANY CODE CHANGE**, before it feeds a billable job AND before you call it done | **THE CODE REVIEWER** (rule 6) | `./ask-reviewer "..."` |

## 🚨 THE TEST, for when it is not obvious which one

The category list above is not enough, because the hard cases look like code and are science. Two
questions decide it, in this order:

1. **Could a planner ruling or `analysis/preregistration.md` overturn the answer?** If yes it is
   **SCIENCE**, whatever surfaced it. If the only thing that could be wrong is the code, it is
   **CODE**.
2. **Does the answer change a number or a claim that reaches the paper?** If yes it is **SCIENCE**,
   even when the fix is one line of code.

⚠️ **The thing that surfaced a question is not the owner of it.** A defect found by running code,
or reported by the CODE REVIEWER, can still be a science question. That is the mistake, not a
hypothetical one:

| what happened | what I did | what was right |
|---|---|---|
| A bootstrap returned `[0, 0]` on ARC | Called it a CODE REVIEWER matter because code produced it | The code was correct; *what interval to report* is the **planner's** (R25.1) |
| Asked "is `zero_variance_bound` applied at the right magnitude?" | Sent one question mixing wiring (code) with whether 2 is the right magnitude (science) | **Split it.** The CODE REVIEWER then volunteered a ruling on the magnitude, and R26.1 later overturned it |
| §4's stopseq scope was ambiguous | Picked "the conservative reading" in code | **Ask.** R26.2 reversed it: my conservative choice over-corrected in the *wrong family* |

## 🚨 CALL THEM. DO NOT ANNOUNCE THAT THEY NEED CALLING.

The user's instruction of 2026-09-02, given after I did this **four times in one session**:

> *"please dont ask or report to me that code reviewer or planner needs a pass. The whole point is
> for you to call them automatically without telling me and not calling!"*

**Calling either agent is not a decision that needs the user's approval.** They cost nothing the
user has to authorise, they hold no job submission, and D10's confirmation gate is about **billable
compute**, not about asking a question. So there is nothing to check in first.

**BANNED PHRASES.** If you are about to write any of these, you are filing instead of asking, and
the correct action is to make the call in the same turn:

- *"this goes to the planner"* / *"I will raise this with the planner"*
- *"the CODE REVIEWER is owed a pass"* / *"needs a review before I call it done"*
- *"flagged for the planner"* in a comment, a commit message, or a note
- *"next: ask the planner about X"*

⚠️ Every one of those is a sentence I have written **instead of** making a call that takes one
command. Announcing an intention reads to the user as progress while producing none, and it wastes
their turn on a decision that was never theirs.

**The only thing to report is the ANSWER**, or that a call is currently running. Report what came
back and what you did about it, never that you are about to ask.

**Three signatures that mean STOP AND ASK THE PLANNER:**

- You are about to **"pick the conservative option"** on anything registered. Conservative is not
  a substitute for correct: R26.1 showed the conservative magnitude described a range the
  registered monotonicity property *forbids*, and R26.2 showed a conservative scope choice
  over-corrected in the wrong hypothesis family.
- The **CODE REVIEWER volunteers on science.** It does, in good faith, and its reasoning is often
  right. Do not treat it as settling anything registered. Record that you declined it and why.
- You find yourself writing **"flagged for the planner"** in a comment. That is not flagging, it is
  filing. Send it, or accept that the code has decided it.

🚨 **ALWAYS CALL IT THE "CODE REVIEWER", NEVER JUST "THE REVIEWER".**

The user's instruction of 2026-09-02, and it exists to prevent a specific mistake rather than for
tidiness. Shortened to "the reviewer" the two agents blur together, and they had already blurred:
a conceptual question about a registered estimator was routed to the CODE REVIEWER instead of the
planner, and the CODE REVIEWER then volunteered a ruling on a science decision that was not its to
make. Say "CODE REVIEWER" for code and "planner" for science, every time, in chat and in files.
⚠️ An **ARR reviewer** is a different thing entirely: a person, and the Oct 14 registration
obligation. Do not conflate the two.

## 🚨 NEVER ASSERT A STATE YOU HAVE NOT SEEN

On 2026-09-02 I committed `bffda77` titled *"Hoist a test import; ruff clean across the repo"* and
told the user "505 tests, ruff clean". **Both were false: ruff had one error and the suite was red.**
The CODE REVIEWER found it, and the test that was broken was the cell-construction test for a
**mandatory blocking gate that has never run**, so the thing most needing to be pinned was unpinned.

**The mechanism, because it will recur otherwise.** I chained
`pytest ... ; echo "exit=$?"` and then `git commit` in one call using `;` rather than `&&`, so the
commit ran regardless of the result, the output scrolled past unread, and I reported a state I had
never looked at. Twice in the same sequence: an earlier `str.replace` silently did not match, because
ruff had reordered the lines, and I did not check that the edit landed either.

**Three rules, all cheap:**

1. **Gate the commit on the check.** `pytest ... && ruff ... && git commit`, never `;`. A failing
   check must make the commit not happen.
2. **Print a verdict, not a stream.** `&& echo "PYTEST GREEN" || { echo "PYTEST RED"; ... }`. A
   truncated wall of dots is indistinguishable from a truncated wall of dots with an F in it.
3. **Verify an edit landed.** After a scripted `replace`, assert the new text is present. A pattern
   that does not match fails silently and looks identical to success.

⚠️ **A suite you believe is green while it is red is how a red suite becomes normal.** The claim is
the deliverable here as much as the code is: the user cannot re-run every check, so an unverified
"tests pass" is worse than saying nothing.

🚨 **ASKING IS NOT ENOUGH. WAIT FOR THE ANSWER BEFORE CALLING THE WORK DONE.**

Recorded because it happened on 2026-09-01: the analysis layer (`A19`) was written, committed,
and reported to the user as built, and *then* the CODE REVIEWER was consulted. The call was genuinely
made and was still in flight when the work was presented as finished. **That is close to not
asking at all**, because the user was shown completed work they had no reason to doubt. The
previous two reviews found a CRITICAL bug each, including one that would have put a fabricated
statistic in the paper, so the gap between "asked" and "heard back" is precisely where the risk
lives.

**The obligation in one line: ask, WAIT, act on CRITICAL and MODERATE, record the outcome in
`notes/reviews/` including what you declined and why, and only then say it is done.**

⚠️ Neither agent writes code and neither holds a job submission. Their authority is advisory;
the reason to act on them is that their findings are verified, not that they ranked something
highly. Full detail in rule 5 (planner) and rule 6 (CODE REVIEWER) below.

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
4a. **Log every terminal job to `notes/RUNLOG.md`** — the append-only ledger of what ran, what it
    cost, and what it taught us. Use the script so the numbers come from the API, never recall:
    ```bash
    bash scripts/logrun.sh <job-name> "one-line outcome"   # appends status/seconds/cost
    bash scripts/logrun.sh                                  # running total only
    ```
    Then fill in the `TODO` line with what changed as a result. Do this **as soon as a job reaches
    a terminal state**, not at the end of a session — the details that matter (which check failed,
    what the log said) are the ones that evaporate first.
4b. **🚨 READ A SAMPLE OF 3 GENERATIONS, EVERY TIME PREDICTIONS ARE PRODUCED.** Not optional,
    and not only when something looks wrong.
    ```bash
    .venv/bin/python scripts/peek_generations.py --run-id main-v1            # 3 at random
    .venv/bin/python scripts/peek_generations.py --run-id main-v1 --task bbh # per task
    ```
    Do this as soon as a generating job reaches a terminal state, before scoring it and before
    reporting any number from it.

    **Why this is a rule.** Three defects in this project were plainly visible in the raw
    generation text and were found weeks later by a scorer instead, because nobody looked:

    | defect | what a single record showed |
    |---|---|
    | the BBH terminator zeroing | `generation_text` was `""` with `n_gen_tokens=1` and `stop_reason="Q"`, on five of six subtasks, for every arm including bf16 Llama-8B |
    | the double BOS on Llama | `prompt_text` began with two `<\|begin_of_text\|>` tokens |
    | GSM8K's unreduced gold | `gold` was the whole worked solution ending `#### 18` while the extractor produced `18`, so every item scored wrong |

    Each was a **wrong number that looked plausible in aggregate and was obvious in one
    record**. That is the whole argument: aggregates hide precisely what a sample reveals, so
    this check is not redundant with the scorers or the tests.

    What to actually check is printed by the script, but the two that have caught the most are:
    is the text non-empty and a plausible length, and does `n_gen_tokens` sit far below
    `max_gen_toks` on a chain-of-thought task, which usually means a stop sequence fired early.

    ⚠️ **"My script cannot read that format" is NOT an exemption.** I claimed one for the
    `determinism` stage on 2026-09-01, on the grounds that it stores `raw_*.json` probe passes
    rather than generation JSONL, and the user caught it. The stage produces 22,000 model outputs
    and the obligation applies; `peek_generations.py` not reading the format is a limitation of
    the script. Read those files directly:
    ```bash
    aws s3 cp s3://…/checkpoints/determinism/raw_<cell>.json .   # then sample passes[0]["outputs"]
    ```
    🚨 **But note the probe stores TEXT ONLY.** Its per-pass keys are `ordering_seed`, `outputs`
    and `seconds`: **no `finish_reason` and no `n_gen_tokens`.** So on that stage the check cannot
    distinguish "the model stopped" from "hit the cap" from "a stop sequence fired", and the best
    available proxy is the fraction of outputs not ending in terminal punctuation (measured 21.4%
    on gptq-4bit at B=32, consistent with the 256-token cap but not evidence of it).

    That omission also sits against this project's own architectural rule: `README.md` §1 and task
    `A4` both state those two fields are unreconstructable afterwards and that H1 is impossible
    without them. It is defensible for the probe, whose purpose is comparing text across
    orderings, but it means the probe's data cannot be reused for anything else. Do not repeat the
    pattern in a new stage.

5. **🚨 ROUTE SCIENCE QUESTIONS TO THE PLANNER**, not to the user and not to the CODE REVIEWER:
   `./ask-planner "..."` (D13). Batch them, because each call re-sends its whole context.
   `FORK=1` for a throwaway question you do not want in its history.

   **The planner's remit**, and it is broader than "hypotheses": any hypothesis or its framing,
   any **estimator** or the interval around it, any **threshold** or decision rule, any scope or
   budget change, anything that amends `analysis/preregistration.md`, and **any number or claim
   that reaches the paper**. Also: what to do when a registered design cannot be implemented as
   written, which is a question about the registration rather than about the code.

   **Save the answer unabridged** to `notes/planner-answers/`, then distil it into
   `notes/planner-rulings.md` as an R-number. ⚠️ **Never read a ruling through `tail -n`** — doing
   that truncated R15 and lost two of its points.

   ⚠️ **The planner can be wrong, and has been.** R24.1 asserted the 10-ordering run spanned
   {8,32,64} when it measured `[64]` only, and R26.1 withdrew the planner's own H6 overlap framing
   as *"mine and it was wrong"*. Verify a factual claim about our data against the artifacts before
   acting on it; that is not insubordination, it is the check the planner expects.
6. **🚨 CONSULT THE CODE REVIEWER AFTER CODE CHANGES.** `./ask-reviewer "..."` — before the
   change feeds a billable job, and before it is called done.
   ```bash
   ./ask-reviewer "review <what changed>. $(git diff HEAD~1)"
   ```
   The planner rules on **science**; the CODE REVIEWER rules on **code**. Neither writes code.
   Tell it what the change is *for*, not just what it is, and point it at the authority
   document the change is supposed to satisfy — it checks against the registered design, not
   against your description of it.

   **Why this is a rule and not a suggestion.** Its first run found a wrong-number bug in
   `score_pilot.aggregate()` that every existing test passed over, and one place where a
   comment claimed a fix had landed in four callers when it had landed in one. Both were
   confirmed. Tests green and ruff clean are necessary and have repeatedly not been
   sufficient.

   Three obligations that come with it:
   - **Verify before acting.** The CODE REVIEWER labels what it *ran* separately from what it
     *read*; a reading-only claim gets checked before it changes code.
   - **Record the outcome** next to the raw answer in `notes/reviews/`, including findings
     **declined and why**. A finding dismissed without a recorded reason is one it re-raises
     every time its memory refreshes.
   - **Findings come classified CRITICAL, MODERATE or MINOR** (`notes/reviewer-charter.md`).
     Act on CRITICAL and MODERATE. 🚨 **MINOR items are NOT acted on**, by the user's
     instruction of 2026-09-01; the CODE REVIEWER still lists them in one line each so they are on
     record, and that list is the whole of what happens to them.

     ⚠️ The one risk in ignoring minors, and the charter tells the CODE REVIEWER to guard it: some
     correctness bugs look like hygiene. A swallowed exception, a type hint that lies, and a
     parameter accepted then ignored have all appeared here as apparent tidiness issues. The
     reviewer is instructed to **promote** such an item to MODERATE or CRITICAL rather than
     leave it as a MINOR with a warning attached, and to say when it has promoted something.
     If you find yourself arguing that a MINOR really matters, that is the signal it was
     misclassified: ask for a re-rank instead of quietly acting on it.
   - **It is ADVISORY.** It does not hold job submissions; that authority is the planner's
     and the user's. Act on its findings because they are verified, not because it ranked
     something `BLOCKER`.

## 0a. How to write to the user

Two rules, both stated by the user directly, and both about their own reading rather than
about the work. Neither is open to partial compliance.

1. **Be concise, but write in FULL SENTENCES. Never shorthand.**

   The user asked for brevity on 2026-08-26 and then clarified it on 2026-09-01, because
   brevity had been over-applied into telegraphic writing: sentence fragments, clipped noun
   phrases, and dense tables standing in for an explanation that needed a sentence. Their
   words were *"talk to the user concisely but in FULL sentences NOT SHORT hand"*. Both
   halves are the instruction, and the second half exists because the first was taken too far.

   In practice: lead with the answer or the blocker and then stop early, but let a proper
   sentence carry the point. Tables remain right for genuinely tabular data, such as a number
   per arm, and wrong as a substitute for prose. Do not narrate what the visible tool calls
   already show.

2. **🚨 NEVER USE AN EM DASH.** Stated flatly on 2026-09-01 as *"never use —"*.

   Rewrite rather than substitute a lookalike, because an en dash or a double hyphen reads as
   ignoring the instruction. An aside becomes a comma pair or parentheses, a pause before a
   conclusion becomes a colon or a new sentence, and two joined independent clauses become a
   semicolon or two sentences.

   This applies to every surface the user reads, which includes chat replies, commit messages,
   and prose written into repository files such as this one, `README.md`, and `notes/`. It does
   not apply to a double hyphen inside a flag such as `--i-have-confirmation`, which is code.

   ⚠️ This needs deliberate attention rather than good intentions. Em dashes are a strong
   default habit and slip in mid-sentence unnoticed, so re-read prose before sending it.

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

Stages, in dependency order: `verify` → `throughput` → `quantize` (job 1) → `determinism`
(job 2, H8) → `pilot` (job 3, the annotation pool) → `template` (the template factor) →
`reference` (**THE MAIN RUN**, `run_id=main-v1`). Plus `diagnose`, a general-purpose slot for
one cheap job answering a specific open question — a *measurement*, not a rehearsal. All eight
have entry points and submit cleanly; `README.md` §4 has the table.

🚨 **`reference` is the pre-registration freeze line.** `analysis/preregistration.md` §1–§9 may
be edited freely until the first `main-v1` generation exists, because until then no edit can
have been influenced by confirmatory results. From that job onward it is append-only. The Sep 2
date is the *schedule* target, not the binding constraint (§9f).

🚨 **A cached or resumed unit of work must be keyed on EVERYTHING that determines what it
produces.** This has cost real runs eight times — values declared and never forwarded
(`desc_act`, `async_scheduling`, `add_bos_token`), a value forwarded but missing from the cache
key (the pilot's per-level `until`), a path derived inside the callee (job 1b), a resume key
without the ordering count (the 10-ordering run silently reused 3-ordering results and exited
`Completed`), and a `run_id` without a probe index. Root fix: `decoding_fingerprint()` in
`src/quantartifact/generate.py`, appended to `cell_tag`. **Bump `DECODING_VERSION` when anything
affecting generation changes.** ⚠️ `env_hash` is NOT this — it covers `{gpu, cuda}` only, which
is why it stayed identical across two separate configuration changes.

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
