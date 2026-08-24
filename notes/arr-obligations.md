# ARR obligations and AWS quota record — task `B2`

**Submission target:** ARR **2026-10-12, 23:59 AoE (UTC−12)** → NAACL 2027 primary,
Findings automatic.

---

## 1. ⚠️ The automatic-desk-reject clause — read this first

**Both authors must register as ARR reviewers/ACs within 48 hours of submission (~Oct 14) or
the paper is automatically desk-rejected.** The obligation is **per-author, not per-paper**.

Registration is mandatory regardless of whether either author is *qualified* — ARR uses
qualification to decide assignments, not to waive the requirement. ARR's May 2026 policy
scales service load with submission count.

An **incomplete OpenReview profile blocks registration**, so profile completeness is a
prerequisite, not a formality.

## 2. Checklist — both authors

| Item | Author 1 (user) | Author 2 (co-author) |
|---|---|---|
| OpenReview profile exists | ⏳ | ⏳ |
| Affiliation filled in | ⏳ | ⏳ |
| Semantic Scholar link | ⏳ | ⏳ |
| DBLP link | ⏳ | ⏳ |
| ACL Anthology link | ⏳ | ⏳ |
| Working email verified | ⏳ | ⏳ |
| **Oct 14 calendar reminder set** | ⏳ | ⏳ |
| Registered as ARR reviewer/AC | ⏳ (due Oct 14) | ⏳ (due Oct 14) |

**Action for the user:** verify both profiles at <https://openreview.net/profile> and set the
Oct 14 reminder for both people. An agent cannot do either.

## 3. Author configuration

Two authors: the user plus one co-author; the co-author registers as an ARR reviewer.

⚠️ **Settle the author list before annotation begins (2026-08-26).** Annotation makes the
co-author a substantive contributor. ARR requires justification to add an author after
submission, and removal is near-impossible. This is `EXECUTOR-TASKS.md` §11.6 and it is
**still unconfirmed** — flagged to the user 2026-08-23.

Because there are two annotators, `A11` uses proper inter-annotator agreement rather than a
single-rater test–retest fallback. That is the stronger design and is what the plan
specifies.

## 4. Other desk-reject grounds (full sweep is `A23`, W7)

- **LaTeX ACL template only.** Word submissions have been desk-rejected since March 2026.
- **≤ 8 pages** for a long paper.
- **Limitations section is mandatory** — its absence is an automatic desk reject, and it
  costs **zero page budget**. Write it early (`C5`, W4).
- **Responsible NLP Checklist must be complete AND correct.** Incorrect or incomplete
  filings have themselves been a desk-reject ground since Dec 2024. §C
  (compute/hyperparameters/error bars) and §E (AI assistance) both apply to us; §D covers
  the annotation design (instructions, recruitment, and the fact that annotators were
  unpaid co-authors rather than crowdworkers).
- Anonymized submission; **no anonymous repo on Dropbox or Google Drive** (download
  tracking).
- Appendices in double-column format, carrying **no load-bearing content**.
- **arXiv preprinting is permitted** — there has been no ARR anonymity period since Feb
  2024. Do **not** elect the optional binding no-preprint option; if elected, preprinting
  before meta-reviews is a desk reject.

## 5. Post-submission calendar

| Date | Event |
|---|---|
| **Fri Oct 9 or Sat Oct 10** | Submit (gate `G9`). Oct 11–12 is buffer for **platform failure only**, not science. |
| **Wed Oct 14** | **Both authors register as ARR reviewers/ACs.** Automatic desk reject if missed. |
| Oct 14 (+48 h of submission) | Metadata stays editable for 48 h. Withdrawing >48 h after the deadline blocks resubmission until the *second* following cycle. |
| **Dec 20, 2026** | Commitment deadline. ARR lists Dec 20, NAACL's own site says Dec 23 — **plan to Dec 20**, re-verify in November. |
| Jan 2027 cycle | Feeds **ACL 2027** (Kyoto, Aug 17–22). A meta-review score ≥ 2.0 permits next-cycle resubmission. **Oct 12 is not one-shot.** |

## 6. AWS quota record — the O4 "document what we declined" item

`EXECUTOR-TASKS.md` §3.1 asks for a 2-minute read-only AWS quota check purely so the paper's
compute section is accurate and so we know what we declined. Done 2026-08-23; **full results
in `notes/compute.md` §6.**

Headline: `aws sts get-caller-identity` → account `670060057797`, and the check materially
changed the substrate decision rather than merely documenting a rejection. SageMaker
training-job quota is **zero** for every usable GPU, while EC2 On-Demand G quota is **768
vCPUs, already granted**. See `notes/compute.md` §2 for the recorded amendment to O4.
