# Model access — task `B1`

**Gate `G1` (due 2026-08-25): ✅ CLOSED 2026-08-23.**
Llama `config.json` downloaded successfully — token **and** license grant both confirmed.

---

## 1. Status

| Item | Status | As of |
|---|---|---|
| `hf` CLI installed | ✅ `uv tool install "huggingface_hub[cli]"` → `huggingface_hub 0.36.2` | 2026-08-23 |
| HF token | ✅ found at `~/workspaces/MedLLMSafety/.env` (`HF_TOKEN`), pointed out by the user | 2026-08-23 |
| `hf auth whoami` | ✅ `Jaybum01` | 2026-08-23 |
| **`meta-llama/Llama-3.1-8B-Instruct` access** | ✅ **GRANTED — `config.json` downloaded** | 2026-08-23 |
| `Qwen/Qwen2.5-3B-Instruct` (ungated) | ⏳ not yet fetched | — |
| `microsoft/Phi-3.5-mini-instruct` (ungated) | ⏳ not yet fetched | — |

**Consequences — all good news:**

- **Model 3 stays `meta-llama/Llama-3.1-8B-Instruct`.** The pre-decided Sep 1 substitution
  to `Qwen2.5-7B-Instruct` (§2.2) is **not needed** and is now void. Nothing to record in
  the paper's setup section beyond the fact that access was verified 2026-08-23.
- **Anchor B is live** (`RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16` publishes
  measured GSM8K 82.8→82.9, ARC-C 81.4→80.2, BBH 30.1→28.9 — a free sanity check on our
  harness config).
- **Anchor C is live**, which matters most: reproducing arXiv:2601.14277's strict-vs-flexible
  GSM8K gap on Llama-3.1-8B-Instruct validates our pipeline **on exactly the phenomenon the
  paper is about**. It is the most rhetorically useful anchor we have.
- **Anchor D** (Kurtic et al. full replication) is unblocked as a stretch item.

⚠️ **Token hygiene.** The token lives in another repo's gitignored `.env`
(`MedLLMSafety/.gitignore:12` covers it — verified). It is **not** copied into this repo.
For the GPU box, pass it as an environment variable at provision time; never commit it and
never bake it into an AMI or image layer.

`EXECUTOR-PLAN.md` §5's caution was well-founded but is now resolved: access did exist on
the account, and the environment simply could not reach it until the token was located.

⚠️ Note on the CLI version: the plan (§3.2) states `huggingface_hub ≥ 1.0.0` removed
`huggingface-cli` in favour of `hf`. `uv` resolved **0.36.2**, which ships **both**
executables. `hf` is present and is the command we use. Pin 0.36.2 in the lockfiles for
reproducibility; revisit only if a 1.x feature is needed.

## 2. How access was verified

```bash
set -a; . ~/workspaces/MedLLMSafety/.env; set +a     # HF_TOKEN
hf auth whoami                                       # -> Jaybum01
hf download meta-llama/Llama-3.1-8B-Instruct --include config.json
# -> Download complete. .../snapshots/0e9e39f249a16976918f6564b8830bc894c89659
```

A successful `config.json` fetch confirms **both** the token and the gated-license grant in
one shot, which is why it is the criterion rather than `whoami` alone.

Record the snapshot revision above: `A3`'s `models.yaml` must pin models by **commit SHA,
not tag**, and `0e9e39f249a16976918f6564b8830bc894c89659` is the resolved
`Llama-3.1-8B-Instruct` revision as of 2026-08-23. Re-confirm before the main runs.

## 3. Pre-decided substitution — NO LONGER NEEDED

Per `EXECUTOR-TASKS.md` §2.2 the substitution to `Qwen/Qwen2.5-7B-Instruct` was armed for
**2026-09-01** if Llama access was unconfirmed. **Access is confirmed, so this is void.**
Kept on record only so the decision trail is legible.

Why it is a good substitute: ungated; **Apache-2.0** (a *better* artifact story than
Qwen2.5-3B's `license:other`/qwen-research); has official Qwen GPTQ-Int4/Int8/AWQ
checkpoints, giving the same Anchor-A cross-check as model 1; and preserves a
3B → 3.8B → 7B size gradient.

The substitution is recorded in the paper's setup section **either way** — including if it
turns out not to be needed.

What Llama is actually needed for, so this stays in proportion:
- **Anchor B** (`RedHatAI/…w4a16` published accuracies) — mandatory *if* access exists,
  **non-blocking**.
- **Anchor C** (arXiv:2601.14277 strict-vs-flexible gap on Llama-3.1-8B) — secondary, but
  the most rhetorically useful anchor we have.
- **Anchor D** (Kurtic et al. full replication) — stretch only.
- The **blocking** gate `G5a` is **Anchor A**, which uses the *ungated* official Qwen
  checkpoints. So Llama is on no blocking path. `A1`–`A8` (pipeline, scorers, determinism
  audit, pilot) all proceed without it.

## 4. Repo trap — do not lose 17 GB to it

`meta-llama/Llama-3.1-8B-Instruct` totals **32 GB**, not 15, because it ships a duplicate
`original/consolidated.00.pth`. **Always pass `ignore_patterns=["original/*"]`**
(or `--exclude "original/*"` on the CLI).

## 5. Token handling

- The token is needed on the **GPU box**, not just locally.
- Export it as an environment variable there. **Never commit it to the repo.**
- `.gitignore` must cover `.env`, `*.token`, and `~/.cache/huggingface` is outside the repo
  anyway.

## 6. Timestamps

| Event | Date | Result |
|---|---|---|
| `hf` CLI installed | 2026-08-23 | `huggingface_hub 0.36.2` |
| Token located (`MedLLMSafety/.env`) | 2026-08-23 | ✅ |
| `hf auth whoami` | 2026-08-23 | ✅ `Jaybum01` |
| Llama `config.json` download | 2026-08-23 | ✅ **200, license granted** |
| Llama license request needed | — | not needed |
| Substitution to Qwen2.5-7B | — | **void, not needed** |
| Qwen2.5-3B + Phi-3.5-mini fetched to the GPU box | ⏳ | pending box launch |
