# External reader — task `B4`

**Why this is a day-1 task and not a week-5 task:** it needs a third person's calendar in
late September. That is external latency, and external latency does not compress.

**Status:** ⏳ **not started — needs the user.** An agent cannot recruit a human reader.

---

## 1. The ask

Identify and ask **two** candidates, on the assumption **one will decline**.

**Criteria:** knows quantization *or* evaluation methodology; **is not the co-author** (the
co-author is already inside the paper and will have lost the outside view by late
September).

**Tell them now, not in September:**

> - The draft arrives **Wed Sep 30**.
> - Feedback is needed by **Sun Oct 4**.
> - The ask is a **targeted attack on two specific things**, not a full read:
>   1. **Related-work differentiation** — is our contribution actually distinct from the
>      papers we cite, especially Kurtic et al. (ACL 2025), Hua et al. (EMNLP 2025), and
>      arXiv:2606.00206?
>   2. **Judge validation** — do you believe the judge numbers, and is the
>      bf16-vs-4-bit error-rate check convincing?

A 4-day window is realistic for a targeted attack on two sections. It is **not** realistic
for a full read, and asking for a full read is how you get no response at all.

## 2. Why these two targets specifically

`EXECUTOR-PLAN.md` §11 names the **related-work squeeze as *the* elevated rejection risk**
for this paper, and calls it *"a writing problem, which is the most controllable kind."* An
external reader is the highest-leverage instrument we have against it, because the failure
mode is invisible from inside: we know why we are different, and cannot tell whether the page
says so.

Judge validation is the second target because it is the most attackable empirical claim in
the paper and the one where a sympathetic reviewer will still push hardest.

## 3. Fallback, pre-decided

If the reader goes silent (gate `G8`, Oct 4), the **co-author performs a hostile read**
against `EXECUTOR-PLAN.md` §7's list. This is weaker — the co-author is not independent —
but it is not nothing, and it requires no new person's calendar.

## 4. Candidates

| # | Name | Expertise | Asked | Confirmed | Notes |
|---|---|---|---|---|---|
| 1 | | | ⏳ | ⏳ | |
| 2 | | | ⏳ | ⏳ | |

**`B4` is done when one candidate has confirmed in writing.**

## 5. Dates to give them

| Date | What |
|---|---|
| **Wed Sep 30** | Complete draft sent |
| **Sun Oct 4** | Feedback due (gate `G8`) |
| Oct 5–8 | We revise |
| **Thu Oct 8** | Freeze |
| Oct 9–10 | Submit |
