# Annotation rubric — is the model's answer correct?

**Task `B5`. Status: FROZEN. Both annotators completed the 10-item dry run and accepted this rubric (sign-off table at the end, 09/01 and 09/07).** *(Corrected 2026-10-16: this line still read "DRAFT" after both sign-offs were recorded.)*

**Annotators:** A1 and A2.

**What you are judging, in one sentence.** For each item you see a question and a model's
answer. You decide whether the model's answer is **correct**, **incorrect**, or **no answer
given**. That is the whole task.

**What you are NOT judging.** Not the reasoning quality, not the style, not the length, not
whether you would have solved it that way, and not whether the model "should" have got it. Only
whether the answer it committed to matches the correct answer.

---

## 🚨 Why the rules below are this rigid

This rubric exists to validate an automated judge, and **the number it produces is a headline
number in the paper**. Two things follow.

First, **rubric drift between the two of us is the main threat to the whole design.** If we
each resolve edge cases our own way, our disagreement rate measures our inconsistency rather
than the judge's reliability. So every edge case has a written rule, decided *before* we see
any disagreements.

Second, and this is the uncomfortable one: **our effects are small enough that a bias hiding
behind high agreement could manufacture the entire finding.** Hróbjartsson et al. (*BMJ* 2012)
looked at 21 trials with binary subjective outcomes and found non-blinded assessors exaggerated
odds ratios by **36%**, and the whole bias came from misclassifying a median of **3% of items
while agreeing on 78%**. We could agree with each other 95% of the time and still be wrong in
the direction that flatters the paper. The blinding below is what prevents that, and it is
load-bearing rather than hygiene.

---

## The three labels

| Label | Key | Means |
|---|---|---|
| **CORRECT** | `c` | The model committed to an answer and it matches the reference answer. |
| **INCORRECT** | `i` | The model committed to an answer and it does not match. |
| **NO ANSWER** | `n` | The model never committed to a final answer at all. |

There is also `RUBRIC_GAP` (`g`), described at the end. **Use it sparingly and never to avoid a
decision this rubric already covers.**

---

## Decision procedure, in order

Work through these in sequence and stop at the first one that applies. **Do not skip ahead**,
because the order is what makes us consistent.

### Step 1. Find the committed answer

Read the generation and identify **the answer the model presents as its final answer**. Look
for a commitment marker: "The answer is X", "#### X", "So X", "Therefore X", a boxed answer, or
a final standalone statement of the result.

- If there is **no such commitment anywhere**, label **NO ANSWER** (`n`) and stop.
- If the generation is **empty or a fragment with no answer**, label **NO ANSWER** (`n`) and stop.

### Step 2. If the model commits more than once, take the LAST one

Models sometimes state an answer, reconsider, and state another. **Take the final commitment**,
because that is what the model is presenting as its answer.

⚠️ Note this deliberately differs from one of the automated scorers we are validating, which
takes the *first*. That difference is part of what we are measuring, so **do not try to match
what you think the scorer would say.** Apply this rule.

### Step 3. Compare to the reference answer

Compare the committed answer to the reference answer shown. Judge **equivalence of value, not
of formatting**.

**These all count as matching**, if the value is right:

| Reference | Model said | Verdict |
|---|---|---|
| `64` | `$64`, `64.00`, `64 dollars`, `sixty-four` | CORRECT |
| `1430` | `1,430`, `$1,430.00` | CORRECT |
| `8` | `8 bags`, `8 of them` | CORRECT |
| `False` | `false`, `FALSE`, `No` | CORRECT |
| `(A)` | `A`, `option A`, `answer A` | CORRECT |
| `] ]` | `]]`, `] ]` | CORRECT |

**These do not match:**

| Reference | Model said | Verdict |
|---|---|---|
| `64` | `63`, `64.5`, `640` | INCORRECT |
| `8` | `eight bags of lollipops, roughly` with no number | see step 4 |
| `(A)` | `(B)` | INCORRECT |
| `True` | `it depends` | see step 4 |

---

## Step 4. The edge cases, each with a fixed rule

These are the cases where we would otherwise drift. **Each has one answer.**

### 4.1 The generation is cut off mid-sentence

**If it committed to an answer before being cut off → judge that answer normally.**
*"The answer is 42. Let me verify: 42 × 2 ="* is a commitment to 42. Label on 42.

**If it was cut off before committing → NO ANSWER (`n`).**

⚠️ Truncation *after* commitment does not make an item unanswered. Getting this backwards
would systematically discard exactly the items the paper is about.

### 4.2 The correct answer appears in the reasoning but is not the final answer

**INCORRECT (`i`), or NO ANSWER if there is no final answer at all.**

*"...so we get 18. But wait, that seems wrong. The answer is 20."* with reference `18`
→ **INCORRECT.** The model committed to 20.

This is deliberate and it is one of the things the paper measures. Do **not** credit the model
for having the right number somewhere in its working.

### 4.3 The model gives multiple candidate answers and does not choose

*"The answer is either 6 or 8."*

→ **INCORRECT (`i`)**, even if one of them is right. A hedge is not a commitment to a value.

### 4.4 The model refuses, or says it cannot answer

→ **NO ANSWER (`n`).** A refusal is not an incorrect answer, it is an absent one.

### 4.5 Degenerate repetition

*"The answer is the answer is the answer is..."*

→ **NO ANSWER (`n`)** unless a specific value is clearly committed somewhere in it.

### 4.6 The model answers a different question than the one asked

→ **INCORRECT (`i`).** It committed to a value; the value is wrong for this question.

### 4.7 Units, currency, and formatting are wrong but the value is right

→ **CORRECT (`c`).** See the table in step 3. We judge value, not presentation.

### 4.8 The reference answer looks wrong to you

→ **Judge against the reference anyway**, and flag it. Press `f` to add a flag note. Do not
silently substitute your own judgement, and do not let it change the label. If a reference
answer is genuinely wrong that is a finding about the benchmark, and it needs to be visible
rather than absorbed into our labels.

### 4.9 You genuinely cannot tell

→ **RUBRIC_GAP (`g`)**, with a one-line note saying what was unclear.

⚠️ These items are **excluded from the agreement statistic and the count is reported**, so
using `g` is honest but it is not free: too many and the statistic loses power. Use it when the
rubric is silent, not when the item is merely hard.

---

## What you will never see, and why

The annotation file **omits** the precision condition, the automated parser's verdict, and the
judge's verdict. They are not hidden columns, they are **absent from the file**, because a
hidden column can be un-hidden.

You will also never be shown a pre-filled label to correct. **This is deliberate and it is not
about trusting you.** Fort and Sagot (LAW IV 2010) measured the better of two annotators
dropping from 98.1 to 95.8 under low-quality pre-annotation. And the trap is worse than it
looks: Mikulová et al. (LREC 2022) found pre-annotation *raised* inter-annotator agreement,
which sounds good and is not, because it is shared anchoring rather than higher validity, and
agreement is one of our headline numbers.

Your item order is **shuffled independently of the other annotator's**, with the two seeds
published and asserted to differ.

---

## Adjudicating disagreements

**Fixed before any disagreement has been seen, which is the point.**

1. Where our labels differ, we sit down **together** and apply this rubric to the item as a
   **rule lookup**: which numbered clause above decides it?
2. We record **the clause number** for each adjudicated item.
3. If no clause decides it, the item is marked **`RUBRIC_GAP`** and excluded from the agreement
   statistic, with the count reported.

🚨 **Disagreements are never resolved by deferring to the senior author.** Belz et al.
(Insights 2023) list biased aggregation as a documented reproducibility failure. If we cannot
agree by rule lookup, the rubric was incomplete, and that is the finding.

---

## The dry run, which is what B5 needs from you

**Before any real annotation:** we both label the same **10 items** blind, independently, then
compare and talk through every disagreement. The point is to find where this rubric is unclear
while it still costs nothing to fix.

This replaces attention checks deliberately. Excluding annotators who fail checks yields a
biased subsample (van der Lee et al., INLG 2019), so we would rather calibrate than filter.

**B5 is done when:** this file is frozen, committed verbatim, and **both of us have signed off
on the 10-item dry run.**

---

## If agreement comes out low

If our agreement on the overlap subset is below about **0.85**, that is a **rubric problem, not
a judge problem**. We revise this file, redo the dry run together, and re-annotate. **No
conclusion about the judge is drawn until then.** This is registered in advance so that a low
number cannot be reinterpreted after the fact.

---

## Sign-off

| Annotator | Dry run completed | Rubric accepted | Date  |
|---|---|-----------------|-------|
| A1 | Done | Done            | 09/01 |
| A2 | Done | Done               | 09/07 |
