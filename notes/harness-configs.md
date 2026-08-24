# Harness configs — verified against lm-evaluation-harness v0.4.12

**Verified:** 2026-08-23, by direct fetch of the `v0.4.12` tag from
`github.com/EleutherAI/lm-evaluation-harness`. Vendored copies: `vendor/lm-eval-configs/`.

`EXECUTOR-TASKS.md` §2.2 says *"verify against the pinned harness configs in `A5`, do not
trust these numbers from memory."* This is that verification. Three of the plan's numbers
are confirmed correct; three of its premises are not.

---

## 1. Token-budget 1× levels — CONFIRMED, with one caveat

| Task | `max_gen_toks` | Where it comes from |
|---|---|---|
| `gsm8k` (5-shot, non-CoT) | **256** | ⚠️ **framework default**, not the task YAML |
| `gsm8k_cot` (8-shot CoT) | **256** | ⚠️ **framework default**, not the task YAML |
| `bbh_cot_fewshot_*` (all 6) | **1024** | `bbh/cot_fewshot/_cot_fewshot_template_yaml` |
| `arc_challenge_chat` | **100** | `arc/arc_challenge_chat.yaml` |

All three remembered values in `EXECUTOR-TASKS.md` §2.2 are **correct**.

⚠️ **Caveat that matters for the paper.** Neither GSM8K task YAML sets `max_gen_toks`. The
256 comes from `lm_eval/models/vllm_causallms.py`, `max_gen_toks: int = 256` — a *backend*
default. So GSM8K's generation cap is a property of the harness implementation, not of the
benchmark, and a different backend silently changes it. Given that arXiv:2605.19537
measures up to 16.6pp of benchmark movement from backend choice alone, this is a small but
real instance of our own thesis, and it is worth one sentence in the paper. We set it
explicitly in `configs/tasks.yaml` rather than inheriting it.

## 2. Rung-1 (strict) regexes — CONFIRMED verbatim

| Task | Filter name | Regex |
|---|---|---|
| `gsm8k` | `strict-match` | `#### (\-?[0-9\.\,]+)` |
| `gsm8k` | `flexible-extract` | `(-?[$0-9.,]{2,})\|(-?[0-9]+)`, `group_select: -1` |
| `gsm8k_cot` | `strict-match` | `The answer is (\-?[0-9\.\,]+).` |
| `gsm8k_cot` | `flexible-extract` | same as `gsm8k` |
| `bbh_cot_fewshot_*` | `get-answer` | `(?<=the answer is )(.*)(?=.)` |
| `arc_challenge_chat` | `remove_whitespace` | *(no regex; whitespace strip + `take_first`)* |

BBH's lookbehind is **case-sensitive and lowercase** (`the answer is`), as the plan
predicts. Note the few-shot exemplars do teach the lowercase form
(*"So the answer is False."*), so the capitalization failure mode is real but partially
mitigated by the prompt — `A5` must measure the rate rather than assume it. Confirmed as a
concrete, citable brittleness to log.

---

## 3. ⚠️ PROBLEM 1 — "few-shot CoT" and "`####` strict regex" are mutually exclusive

**Both plan documents assert the GSM8K strict regex is `#### (\-?[0-9\.\,]+)`**
(`EXECUTOR-TASKS.md` §3.4; `EXECUTOR-PLAN.md` §1.1, §2.2). That is correct — **for the task
`gsm8k`**, which is **5-shot and non-CoT**, whose `doc_to_target` is the raw GSM8K answer
string (which literally contains `#### N`), which is why the regex looks for it.

**`EXECUTOR-TASKS.md` §2.2 also specifies the reference prompt format as
"few-shot CoT (reference, harness-verbatim)."** The harness's few-shot-CoT GSM8K task is
`gsm8k_cot` — 8-shot, `doc_to_text: "Q: ...\nA:"`, exemplars ending *"The answer is 6."* —
and its `strict-match` regex is **`The answer is (\-?[0-9\.\,]+).`**, not `####`.

So within v0.4.12 these three requirements cannot all hold at once:

| | Requirement | Needed for |
|---|---|---|
| (a) | reference = few-shot **CoT** | **H1**, the lead hypothesis (truncation of chains of thought) |
| (b) | strict regex = **`####`** | the large H3 effect; **Anchor C** (arXiv:2601.14277) |
| (c) | **harness-verbatim** prompts | **G2** prompt parity; comparability to published numbers |

Pick any two.

**Why this is load-bearing, not pedantic.** `EXECUTOR-PLAN.md` §1.1 grounds the artifact's
existence in `####` collapsing chat-model scores toward zero ("instruction-tuned models
essentially never emit it"). Under `gsm8k_cot`, strict-match is `The answer is N`, which the
8 few-shot exemplars explicitly demonstrate — so compliance should be **high** and the
rung-1-vs-rung-2 gap (the paper's headline contrast) should be **much smaller**, possibly
near zero on GSM8K. Conversely `gsm8k` gives the large `####` gap and matches Anchor C, but
its prompt does not ask for a chain of thought, which weakens H1's mechanism on that task.

**This is not one of the eleven §0.2 overrides and is not resolvable at the execution
level** — it is a scientific-design fork, so `EXECUTOR-PLAN.md` has authority. **Reported to
the user 2026-08-23; awaiting decision.** Recommendation on record: run **both** GSM8K
variants in the reference condition (+1,319 × 15 ≈ 19.8k generations, ~20% volume increase,
compute is not the binding constraint), which converts the tension into a result —
the artifact measured under two harness-sanctioned prompt/regex pairings, one satisfiable by
chat models and one not — and preserves Anchor C. Pre-registration §5 currently enumerates
both and flags the primary as open.

## 4. ⚠️ PROBLEM 2 — `TRUNCATED = finish_reason == "length"` undercounts truncation

`EXECUTOR-PLAN.md` §2.3 and `EXECUTOR-TASKS.md` §7.2 define `TRUNCATED` as
`finish_reason == "length"`. The verified `until` stop sequences show that is incomplete:

| Task | `until` |
|---|---|
| `bbh_cot_fewshot_*` | `["</s>", "Q", "\n\n"]` — **a bare capital `Q`**, and a blank line |
| `gsm8k` | `["Question:", "</s>", "<\|im_end\|>"]` |
| `gsm8k_cot` | `["Q:", "</s>", "<\|im_end\|>"]` |
| `arc_challenge_chat` | `["\n\n", "."]` — **a single period**, with `max_gen_toks: 100` |

A BBH chain of thought containing any capital `Q` or any blank line is cut mid-reasoning and
reported with `finish_reason == "stop"`, not `"length"`. Same for an ARC-chat generation
containing a period before the answer — which, with a 100-token cap and the required
closing *"The best answer is X"*, is a live failure mode.

This is a **second, distinct truncation mechanism operating on the same causal pathway as
H1**, and the naive definition misses it entirely. Handled by pre-registering three
termination classes (`TRUNC_BUDGET` / `TRUNC_STOPSEQ` / `COMPLETE`) — see
`analysis/preregistration.md` §4. `TRUNC_STOPSEQ` is a useful internal control precisely
because it should *not* respond to the budget sweep.

Fixing this after seeing results would have been a specification search, which is why it is
pre-registered now.

## 5. ⚠️ PROBLEM 3 — BBH few-shot and zero-shot use different metric normalization

This confounds **H5** (few-shot CoT vs zero-shot) if the harness metric config is inherited
per-variant.

| | `cot_fewshot` template | `cot_zeroshot` template |
|---|---|---|
| `ignore_case` | **commented out** (i.e. off) | **`true`** |
| `ignore_punctuation` | commented out (off) | commented out (off) |
| `regexes_to_ignore` | **none** | `["\\.$", ",", "\\\\", "\n", '"']` |
| `filter_list` | `get-answer` regex | **none** |
| `target_delimiter` | `""` | *(default)* |

Few-shot BBH scoring is **strictly harsher** than zero-shot BBH scoring in the harness. If
we inherit each variant's own metric config, part of any H5 few-shot-vs-zero-shot difference
is the *scorer*, not the prompt.

**Execution decision (within this document's authority — scope/execution):** our four rungs
are applied **identically** across prompt variants. Rung 1 for BBH uses the `cot_fewshot`
`get-answer` regex for both variants, and normalization is held fixed at whatever rung 2
specifies. The harness's per-variant metric differences are reported as a documented
deviation from harness-verbatim scoring **for the zero-shot arm only**, with the reason. G2
prompt parity is unaffected — it tests *prompts*, not metric configs.

---

## 6. Other confirmed facts worth not re-deriving

- `arc_challenge_chat` is **`num_fewshot: 0`** by design and carries
  `gen_prefix: 'The best answer is'`. It is a `llama`-tagged task. So the H6 generative arm
  is inherently zero-shot; the log-likelihood `arc_challenge` arm has no shot count in the
  usual sense. **The H6 mode contrast is therefore not also a shot-count contrast only if we
  hold shots fixed** — record the resolved `num_fewshot` per arm in `configs/tasks.yaml` and
  state it in the paper.
- `arc_challenge` inherits from `arc_easy.yaml`: `output_type: multiple_choice`,
  `doc_to_text: "Question: {{question}}\nAnswer:"`, choices are the **answer texts** (not
  letters), metrics `acc` **and** `acc_norm`. This is the G3 cross-validation target.
- `gsm8k` metric strips `,`, `\$`, `(?s).*#### `, and `\.$` via `regexes_to_ignore`, with
  `ignore_case: true`, `ignore_punctuation: false`. The `(?s).*#### ` entry means the
  *metric* also tolerates a `####` prefix independent of the filter regex.
- BBH `dataset_path` is **`SaylorTwift/bbh`**, not the original BIG-Bench repo. Pin it.
- `gsm8k_cot` uses `fewshot_config.sampler: first_n` with **8 hardcoded exemplars** in the
  YAML — so the few-shot prompt is fully determined by the config, no sampling seed needed.
  Same for all BBH subtasks (3 hardcoded exemplars each). Good for G2 reproducibility.
- vLLM backend exposes `add_bos_token` (`vllm_causallms.py:70`), default `None`. Red Hat's
  tip is to pass `True` — quantized models are BOS-sensitive. Record the resolved value.
