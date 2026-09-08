# Annotation task — instructions for Syed Ahmed Danish

**You are annotator `A2`.** Everything you need is already set up. This document is
self-contained, so you do not need to read anything else in the repository first.

**Total time: about two hours**, split into a 5-minute read, a 10-minute practice round we do
together, and roughly 90 minutes of labelling that you can spread over several sittings.

---

## 1. What the project is, in one paragraph

We are measuring how much of the reported accuracy drop from *quantizing* a language model
(shrinking its weights to 4 bits so it runs cheaper) is really a property of the model, and how
much is an artifact of how we measure it. We found, for example, that simply changing the order
in which test questions are submitted moves a 4-bit model's measured accuracy while leaving the
full-precision model's accuracy exactly unchanged. A second finding is that the rule you use to
extract an answer from a model's output changes the size of the quantization effect you report.

## 2. What your task is, and why it needs a human

Part of our pipeline uses an automated "judge" model to decide whether a model's answer was
correct. **Nobody should believe an automated judge without checking it against human
judgement**, so your labels are the ground truth we validate it against.

You will see a question, a model's answer, and the correct reference answer. You decide whether
the model's answer is **correct**, **incorrect**, or **absent**. That is the whole task.

**You are not judging** the reasoning quality, the style, the length, whether you would have
solved it that way, or whether the model "should" have got it right. Only whether the answer it
committed to matches the reference.

### Why the rules are strict about edge cases

Two reasons, and the second is the important one.

First, if the two of us resolve edge cases differently, our disagreement rate measures *our*
inconsistency rather than the judge's reliability. So every edge case has a written rule, fixed
before either of us has seen any disagreements.

Second, the effects this paper measures are small, around one to three accuracy points. There is
a well-documented result that non-blinded assessors exaggerated effects by 36% across 21 medical
trials with yes/no outcomes, and the entire bias came from misclassifying about 3% of items while
agreeing on 78% of them. In other words, **we could agree with each other 95% of the time and
still both be biased in the direction that flatters our own paper.** That is why you will never
see which condition an answer came from.

---

## 3. Setup

You need the repository and Python 3.12. From the repository root:

```bash
cd ~/workspaces/QuantArtifact          # wherever you cloned it
.venv/bin/python --version             # should print Python 3.12.x
```

If `.venv` does not exist:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
```

Nothing you do needs a GPU, the internet, or any credentials.

---

## 4. Step 1: read the rubric (5 minutes)

Open **`annotation/rubric.md`** and read it once through. It is one page: three labels, a
five-step decision procedure, and nine numbered edge cases.

**The one rule most likely to catch you out** is rule 2. If the model states an answer,
reconsiders, and states a different one, **take the last answer**. This deliberately differs
from one of the automated scorers we are validating, which takes the first. That difference is
part of what we are measuring, so please do not try to guess what the scorer would say. Just
apply the rule.

The labels are:

| Key | Label | Means |
|---|---|---|
| `c` | CORRECT | It committed to an answer and it matches the reference |
| `i` | INCORRECT | It committed to an answer and it does not match |
| `n` | NO ANSWER | It never committed to a final answer at all |
| `g` | RUBRIC GAP | The rubric genuinely does not decide this case |

`g` is honest but not free: those items are excluded from the agreement statistic and the count
is reported, so please use it when the rubric is genuinely silent rather than when an item is
merely hard. The tool will ask you why.

---

## 5. Step 2: the practice round (10 minutes, then we talk)

**Do this before any real labelling.** It is the step that calibrates us against each other.

```bash
.venv/bin/python scripts/annotate.py --who A2 --practice
```

Ten items, labelled blind. I run the same command with `--who A1` on my own ten, so we practise
on *different* items, which is what makes this a test of the rubric rather than a rehearsal of
one shared sequence.

**Then we compare, together.** For every item where we disagreed, we look up which numbered
rubric clause decides it, rather than talking each other round. If no clause decides it, that
means the rubric is incomplete and I fix it. When we are both satisfied, we each tick the
sign-off table at the bottom of `annotation/rubric.md`.

We deliberately use this instead of hidden attention-check questions, because excluding people
who fail checks produces a biased sample. Calibrating is better than filtering.

---

## 6. Step 3: the real thing (about 90 minutes, resumable)

```bash
.venv/bin/python scripts/annotate.py --who A2
```

**You have 185 items.** One per screen, one keystroke each. At roughly 20 to 30 seconds an item
that is about 90 minutes, and there is no reason to do it in one sitting.

**It saves as you go.** Every label is written to disk immediately, so you can quit any time with
`q` or just close the terminal, and re-running the command picks up exactly where you left off.

Keys available at every item:

```
  c  correct        i  incorrect        n  no answer        g  rubric gap
  f  flag this item (e.g. the reference answer looks wrong), then label it anyway
  ?  show the rubric's decision procedure on screen
  s  skip for now (it stays in your queue)
  q  save and quit
```

If a reference answer looks wrong to you, press `f`, write a one-line note, and then **label the
item against the reference anyway**. Do not silently substitute your own judgement. If a
benchmark's reference answer is genuinely wrong that is a finding about the benchmark, and it
needs to be visible rather than absorbed into our labels.

---

## 7. What you will never see, and why

The file you read contains exactly four fields: an id, the question, the model's answer, and the
reference answer. **The precision condition, the automated parser's verdict and the judge's
verdict are absent from the file**, not hidden in it, because a hidden field is not hidden. The
tool physically cannot show them to you.

You will also never be shown a pre-filled label to correct. This is not about trusting you. It is
measured: in one study the better of two annotators dropped from 98.1% to 95.8% accuracy when
given low-quality pre-annotations. And there is a trap worth knowing, which is that another study
found pre-annotation *raised* inter-annotator agreement. That sounds like an improvement and is
not, because it is the two annotators anchoring on the same suggestion rather than judging
independently, and agreement is one of the numbers we report.

Your item order is shuffled independently of mine, with seeds `20260907` for me and `20260908`
for you, published in the repository, and the code asserts that our two orders actually differ.

---

## 8. What happens to your labels

| | |
|---|---|
| Sample | 250 items total, drawn across 48 strata of model, precision and task |
| Your share | 185 items |
| Overlap | **120 items are labelled by both of us**, and that overlap is what the inter-annotator agreement statistic is computed on |
| Disagreements | Resolved jointly by rubric-clause lookup, with the governing clause recorded per item, and **never** by deferring to the senior author |
| If agreement is low | If we agree on less than about 85% of the overlap, that is treated as a **problem with my rubric, not with your labelling.** I revise it, we redo the practice round, and we re-annotate. No conclusion about the judge is drawn until then |

Your name will appear as a co-author, and the annotation design, including the fact that
annotators were unpaid co-authors rather than paid crowdworkers, is disclosed in the paper's
Responsible NLP Checklist.

---

## 9. If something goes wrong

- **The command errors with "no queue at ..."** — the sample has not been drawn. Tell me and I
  will run `scripts/draw_annotation_sample.py`.
- **An item looks broken or empty** — that is often a real result rather than a bug. An empty
  answer is a legitimate `n`. If it looks genuinely corrupted, press `f` and note it.
- **You are unsure about a whole class of items** — stop and ask me rather than guessing
  consistently in one direction. A systematic guess is much worse than a `g`.

Ask me anything. Getting this right matters more than getting it fast.
