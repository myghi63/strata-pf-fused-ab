# strata-pf-fused-ab

A small, self-contained A/B harness for **`STRATA_PF_FUSED`** (Strata's fused
int8 prompt-expert kernels) on a single GPU. It answers three questions with
numbers:

1. Does turning the fused prompt path on make the model fall into repetition
   loops, with greedy and with sampled decoding, in long reasoning?
2. Does it damage a very long prompt (~187K tokens)?
3. How far does it move the first-token distribution (KL), and how much prompt
   speed does it buy?

It was written to answer the maintainer's question in
[Strata PR #2392](https://github.com/Niko1221/Strata/pull/2392) and was run on
an RX 7800 XT (gfx1101). The measured numbers are in [RESULTS.md](RESULTS.md);
the first, lighter run (v1) and its corrections are in
[results/v1/](results/v1/RESULTS.md).

## What it measures

Two server starts, one per arm, with **everything identical except
`STRATA_PF_FUSED`**:

| arm | `STRATA_PF_FUSED` |
| --- | --- |
| 1 | `1` (fused) |
| 0 | `0` (MMQ) |

During the run the **server's own reasoning-loop detector is switched on**
(`"reasoning_loop_recovery": "stop"`, see below). Each arm runs, in order:

1. **First-token KL** — 6 prompts of 3.83–3.89K tokens, `max_tokens=4`. The
   engine dumps the logits of the first generated window
   (`STRATA_DUMP_FIRST_LOGITS`) and `kl.py` computes `KL(fused‖mmq)`,
   `KL(mmq‖fused)` and top-1 agreement (the layout of the repository's
   `tools/par/kl_logits.py`). Prompt speed is read from the same requests.
2. **Repeatability probe** — one prompt three times, greedy, 400 tokens. Shows
   whether greedy output is reproducible on this build at all, which decides
   whether a text difference between the arms means anything.
3. **Loop test** — 14 repetition-inviting prompts (~3.3K tokens each:
   counting, lists, tables, JSON, dialogue, cycles…), thinking on, up to
   **6,144 tokens**, once greedy and once sampled (Qwen's thinking-mode
   sampling: `temperature 0.6, top_p 0.95, top_k 20`, `seed 1`).
4. **Long context** — ~187K tokens of *Moby-Dick* (Project Gutenberg #2701,
   public domain, downloaded once into `data/`) with twelve inserted "ship's log
   notes". The task needs the whole text and a long answer: list the notes with
   their chapters, summarize every chapter in order, rank the longest chapters.
   Sampled, up to 60K output tokens, so prompt + output reach ~247K of the 256K
   context. Scored on note recall (code and chapter) and chapter coverage
   (missing / repeated / out of order), plus the same loop checks.

### What counts as a loop

A generation is flagged if **any** of these fires (reasoning and answer are
checked separately):

- the **server** cut it: `the thinking repeated the same passages`
  (`reasoning_loop_recovery`), `repeated reasoning detected` (recover mode) or
  `repeated one token` (`repeat_stop_tokens`, on by default at 256). `client.py`
  attributes each server-log line to the request that was running;
- an exactly periodic tail (period ≤ 300 characters, 3 repeats);
- one line 8+ times, one 10-word phrase 6+ times or one 6-word phrase 12+ times;
- a 4 KB window that compresses absurdly well (zlib ratio < 0.16).

Hitting `max_tokens` (`finish_reason=length`) alone is **not** a loop; it is
reported separately, together with whether the model got to write an answer.

### The server's loop detector

`serve/server.py` has an opt-in detector for reasoning that repeats whole
passages (#728): every 512 output tokens it measures how much of the last
2,000 words is covered by 12-word passages seen three times; at ≥ 25% it either
ends the reply (`"stop"`) or resumes it with the low-effort instruction
(`"recover"`). It is **off by default** and needs about 2,000 words of
reasoning before it can fire — which is why the loop test allows 6,144 tokens.

`run_all.sh` sets `"reasoning_loop_recovery": "stop"` in the model's
installed config (`~/.local/share/strata/install/strata-<model>.json`, the
file `strata launch` copies its settings from), checks that the run config
picked it up, and restores the original file when it exits — also on Ctrl-C
or an error. `"stop"` is used instead of `"recover"` because recover would
hide a loop by steering the model out of it. `LOOP_RECOVERY=off` leaves the
file alone.

## Why the prompts are 3.3K tokens or more

The fused prompt path only runs on a streamed chunk of `stream_all_min()`
tokens or more — 1024 by default (`src/prefill/prefill.cpp`,
`fused_layout()`). With ~600-token prompts **both** arms run MMQ, the KL comes
out exactly `0.0`, and the run measures nothing. Every short prompt is
therefore prefixed with the same neutral ~3.3K-token context, with the question
at the end. The script reports the fused-expert marker per arm and the KL
prompt sizes, and warns if the marker never appeared in arm 1.

Different prompts do not share the conversation cache (each one is read in
full); an identical prompt sent again is served from it, which is why the
repeatability probe isolates decoding from the prompt path.

## Requirements

- Strata installed and able to serve the model (`strata launch`).
- A saved launch preset that serves it with at least 256K context. The examples
  use `iq3_s-256k-final`; set `PRESET` to yours.
- One GPU, enough VRAM for the preset. The preset's VRAM reserve is not
  changed.
- `python3` with `numpy`; network access once, for the book.

`STRATA_PF_FUSED` and `STRATA_DUMP_FIRST_LOGITS` are per-start `--set`
overrides, so the preset file is untouched. The only file changed during the run
is the model's installed config (the loop detector), and it is restored.

## Quick start

```bash
# Stops the running server (if any), runs both arms, leaves the server stopped.
PRESET=iq3_s-256k-final bash run_all.sh --yes

# ... and starts the preset again at the end:
PRESET=iq3_s-256k-final bash run_all.sh --yes --restore
```

Without `--yes` the script only prints what it would do and exits.

Runtime on an RX 7800 XT: roughly 3–4 hours (two engine loads; per arm 6 KL
requests, 3 probe requests, 28 generations of up to 6,144 tokens and one
~187K-token prompt with up to 60K output tokens). `CLIENT_ARGS` passes options
to `client.py`, e.g. `CLIENT_ARGS="--loop-max-tokens 3072 --phases kl,loop"`
for a shorter run, or `--sampled-seeds 1,2,3` for more sampled passes.

The API key is read from the preset's `api_key` (or `STRATA_API_KEY`, or
`--key`). It is never written by this harness.

## What it writes

| file | contents |
| --- | --- |
| `report.md` / `report.json` | speed, server detector counts, loop table per prompt and mode, repeatability, long-context score |
| `kl.md` / `kl.json` | first-token KL in both directions, top-1, per-prompt table |
| `arm1.engine.log` / `arm0.engine.log` | the engine log lines of each arm |
| `arm1.server.log` / `arm0.server.log` | the server log lines of each arm |
| `dumps/fused1.*` / `dumps/fused0.*` | raw float32 logits (248,320 per file) |
| `out/arm*/{kl,probe,loop,long}/` | full requests, responses and the server events of each request |

## How to read it

- **Loops** — the number that matters is arm 1 vs arm 0, per mode. Both 0
  means the fused path did not cause a loop in this sample.
- **Long context** — 12/12 notes with the right chapter and no missing or
  repeated chapter means the ~187K-token prompt arrived intact in that arm.
- **KL** — `docs/AMD_HIP.md` lists "First-token KL vs MMQ: IQ3_S 0.00284 vs
  0.00332" for `STRATA_PF_FUSED` (pooled 24 prompts, Coder IQ1_M setup). The
  same order of magnitude means the fused path is not moving the distribution
  more than expected. **Exactly 0** means both arms ran the same path.
- **Top-1** — expected to be high; the first token is nearly always the same
  word because of the chat template, so the KL is the informative number.
- **Repeatability** — if greedy repeats on the same server already differ, the
  arm-to-arm text comparison is noise and only loop rates and scores count.
- **Speed** — `fused / mmq`. `docs/AMD_HIP.md` reports +19.1% at 4K
  (Coder IQ1_M, ROCm 7.14); a different card and model can gain much more.

## Caveats

- 6 KL prompts, not 24; first-token KL only, not teacher-forced over all
  positions (the engine's `--dump-logits` does that, outside the server).
- One run per arm; 28 short generations and one long one per arm is a sample,
  not a benchmark.
- The test runs with the production stack (dflash2 + speculative decoding). If a
  loop appears, repeat that prompt with `--drafting off` to separate the
  drafter's effect from `STRATA_PF_FUSED`.
- Single requests only: a multi-turn agent session with tool calls, where loops
  are often reported, is not simulated.
- `STRATA_HIP_PROMPT_F16` and `STRATA_PF_GEMM` stay on in both arms (the
  maintainer confirmed those are the gfx11 path). `docs/AMD_HIP.md` lists
  `STRATA_PF_FUSED` as a gfx12 default; on gfx11 it is opt-in.

## Files

| file | role |
| --- | --- |
| `run_all.sh` | orchestrates both arms: detector on, stop, launch with `--set`, client, logs, analysis, restore |
| `client.py` | sends the prompt set (kl, probe, loop, long), saves responses, timings, dump indices and server events |
| `prompts.py` | the 6 KL prompts, the 14 loop prompts, and the ~3.3K-token padding |
| `longctx.py` | builds and scores the ~187K-token Moby-Dick prompt |
| `report.py` | loop detector, server events, repeatability, long-context score, speed |
| `kl.py` | first-token KL from `STRATA_DUMP_FIRST_LOGITS` dumps |
| `results/` | the measured reports (v2 at the top level of RESULTS.md, v1 in `results/v1/`) |

## License

MIT — see [LICENSE](LICENSE). *Moby-Dick* is public domain (Project Gutenberg
#2701); the text is downloaded at run time, not shipped.
