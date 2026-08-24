# Model access — task `B1`

**Gate `G1` due 2026-08-25.** Criterion: Llama `config.json` downloaded, **or** the license
requested with a recorded timestamp.

---

## 1. Status

| Item | Status | As of |
|---|---|---|
| `hf` CLI installed | ✅ `uv tool install "huggingface_hub[cli]"` → `huggingface_hub 0.36.2` | 2026-08-23 |
| HF token present | ❌ **absent** — `~/.cache/huggingface/token`, `~/.config/huggingface/`, `$HF_TOKEN` all absent | 2026-08-23 |
| `hf auth whoami` | ⏳ blocked on token | — |
| `meta-llama/Llama-3.1-8B-Instruct` access | ⏳ **UNVERIFIED** | — |
| `Qwen/Qwen2.5-3B-Instruct` (ungated) | ⏳ not yet fetched | — |
| `microsoft/Phi-3.5-mini-instruct` (ungated) | ⏳ not yet fetched | — |

Confirms `EXECUTOR-PLAN.md` §5's finding still holds: access may exist on the *account*
while this environment cannot use it.

⚠️ Note on the CLI version: the plan (§3.2) states `huggingface_hub ≥ 1.0.0` removed
`huggingface-cli` in favour of `hf`. `uv` resolved **0.36.2**, which ships **both**
executables. `hf` is present and is the command we use. Pin 0.36.2 in the lockfiles for
reproducibility; revisit only if a 1.x feature is needed.

## 2. ⚠️ ACTION REQUIRED FROM THE USER — the last multi-day-latency item

This is the only remaining `B1` step an agent cannot perform: the token paste is
interactive, and a license grant is a manual human approval on Meta's side that **can take
days** and sits on the critical path to Anchor B/C.

Run these four commands. In Claude Code, prefix with `!` to run them in-session so the
output lands in the conversation:

```bash
# 1. Get a token with `read` scope from https://huggingface.co/settings/tokens
hf auth login          # paste the token when prompted

# 2. Confirm the token works
hf auth whoami

# 3. Confirm the token AND the license grant in one shot
hf download meta-llama/Llama-3.1-8B-Instruct --include config.json
```

**Interpreting step 3:**
- **Succeeds** → token and license both good. Record the timestamp below; `B1` closes.
- **401 / 403** → the license is not granted. **Request it in a browser the same hour** at
  <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> and record the request
  timestamp below. Then proceed with Qwen + Phi, which are ungated.

## 3. Pre-decided substitution — do not ask, just execute

Per `EXECUTOR-TASKS.md` §2.2: **if Llama access is not confirmed by 2026-09-01**, drop Llama
and use **`Qwen/Qwen2.5-7B-Instruct`** as model 3. No permission needed.

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

## 6. Timestamps — fill in as they happen

| Event | Timestamp (UTC) | Result |
|---|---|---|
| `hf auth login` completed | | |
| `hf auth whoami` | | |
| Llama `config.json` download attempted | | |
| Llama license requested in browser (if 401/403) | | |
| Llama license granted | | |
| Substitution to Qwen2.5-7B executed (if Sep 1 passes) | | |
