# Archived labels, 2026-10-14

Moved aside, not deleted, so the re-annotation starts clean. `annotate.py` is append-only and
skips any `annotation_id` it already finds, so a re-run with these in place would have labelled
nothing and reported "185 done, 0 remaining".

**Why they were redone.** Raw agreement was **1.000 on all 120 overlap items**, with Cohen's
kappa and Gwet's AC1 also 1.000. Zero disagreements is implausible for independent raters, and
the timestamps showed the two passes ran back to back (A2 20:56 to 21:12, A1 21:13 to 21:23) at
about 3.4 seconds per item. That measures self-consistency rather than inter-annotator agreement,
and `A12` reports IAA as a headline number, so it had to be redone rather than caveated.

⚠️ Both annotators share one machine, which the user has confirmed is acceptable. Independence
therefore rests on the two people not seeing each other's screen and on the per-annotator
shuffle, which the sampler already guarantees with published seeds 20260907 and 20260908.

`validation_from_archived_labels.json` is kept because one finding in it does **not** depend on
IAA and was reproduced from these labels: strict extraction agreed with the human verdict 76.0%
on bf16 against 56.8% on 4-bit, with false-negative rates of 24.0% and 40.4%. Worth comparing
against the re-annotation as a consistency check on the scorer-validation result itself.
