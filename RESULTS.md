# Results — RX 7800 XT, IQ3_S (v2)

Measured with this harness on 2026-11-07, engine **0.1.57 (HIP)**. The first,
lighter run and its corrections are in [results/v1/](results/v1/RESULTS.md).

**Short version:** with the server's own loop detector on, `STRATA_PF_FUSED=1`
and `=0` behaved the same. Each arm had **one** degenerate reasoning loop in 28
short generations, both on the same prompt (list 120 Portuguese words starting
with "a", no repeats), one in greedy and one in sampled mode. With the server's
default sampling (the model card's, which is what a client such as opencode
gets) that prompt looped **0/8 times in each arm**. A ~187K-token prompt came
back intact in both arms. The fused path reads prompts **1.67× faster** at
~3.85K tokens and 1.45× at 187K, with decode unchanged.

## Setup

| item | value |
| --- | --- |
| GPU | AMD Radeon RX 7800 XT (gfx1101), 16 GB |
| CPU / RAM | Ryzen 7 5700X3D / 64 GB DDR4 |
| Model | Qwen3.8-Flash-Next, IQ3_S pack |
| Context | 262,144 tokens (native, no RoPE scaling) |
| Preset | `iq3_s-256k-final` |
| Drafting | dflash2, `--spec 6`, `--draft-conf 0.7` |
| Other | `--ple-io mmap`, `--conversation-cache-mib 4608`, `--vram-reserve-mib 1024`, hipBLASLt tuning table |
| Fixed in both arms | `STRATA_HIP_PROMPT_F16=1`, `STRATA_PF_GEMM=1` |
| Varied | `STRATA_PF_FUSED=1` (arm 1) vs `=0` (arm 0) |
| Server loop detector | `"reasoning_loop_recovery": "stop"` for the run (off by default) |

The switch engaged: the fused-expert marker
(`strata: prompt experts on the fused int8 kernels`) appeared in arm 1's engine
log and not in arm 0's.

## Loops, 14 prompts × 2 modes, up to 6,144 tokens

Prompts of ~3.3K tokens (each read in full, so the fused prompt path runs),
thinking on. Greedy = `temperature 0`; sampled = `temperature 0.6, top_p 0.95,
top_k 20, seed 1`.

| mode | arm | cut by the server's loop detector | of which a degenerate loop | wrote an answer |
| --- | --- | ---: | ---: | ---: |
| greedy | 1 (fused) | 1/14 | **1** | 13/14 |
| greedy | 0 (MMQ) | 1/14 | 0 | 13/14 |
| sampled | 1 (fused) | 1/14 | 0 | 13/14 |
| sampled | 0 (MMQ) | 1/14 | **1** | 13/14 |

Every cut was read by hand:

| prompt / mode | arm | coverage | what the reasoning was doing |
| --- | --- | ---: | --- |
| `loop_words_a` greedy | 1 | 1.000 | **degenerate**: `62 afogar? no.` repeated ~150 times |
| `loop_words_a` greedy | 0 | 0.318 | re-drafting the word list (new words, moving forward) |
| `loop_translate_30` sampled | 1 | 0.284 | re-writing the 30 translations to count them |
| `loop_words_a` sampled | 0 | 0.398 | **degenerate**: `Let's use "absurdo"? no.` over and over |

So the degenerate loop reproduces **in both arms**, on the same prompt, and
which arm falls into it changes with the decoding mode. The prompt is the
trigger (checking 120 words for repeats one by one), not the fused path.

The detector (coverage ≥ 0.25 of the recent reasoning inside 12-word passages
seen three times) also cuts legitimate re-drafts of long lists; three of the
four cuts above were that. A cut is therefore not a loop by itself.

The report's text heuristics also fired on outputs that are repetitive by
design and finished normally (a 50-item JSON list, a numbered poem, a Fibonacci
table, the five-cycle summary); they are listed in
[results/v2/report.md](results/v2/report.md) and none of them is a loop.

## The looping prompt with the server's default sampling, 8 runs per arm

`loop_words_a` sent 8 times per arm **without any sampling field**, so the
server applies the model card's sampling for thinking mode (`temperature 1.0,
top_p 0.95, top_k 20`; `serve/server.py` `MODEL_CARD_SAMPLING`) — what a client
that sends no sampling, such as opencode, gets.

| arm | runs | degenerate loops | cut by the detector (all re-drafts) | coverage of the cuts | wrote an answer |
| --- | ---: | ---: | ---: | --- | ---: |
| 1 (fused) | 8 | **0** | 2 | 0.349, 0.264 | 6/8 |
| 0 (MMQ) | 8 | **0** | 4 | 0.358, 0.250, 0.343, 0.329 | 4/8 |

All six cuts were read: each was the model re-writing or re-checking its list
of 120 words, with new words appearing. None of the 16 runs repeated a line the
way the two degenerate loops above did. Details:
[results/v2-rep/report.md](results/v2-rep/report.md).

## Long context, ~187K tokens

The beginning of *Moby-Dick* (chapters 1–80, Project Gutenberg #2701) with
twelve inserted "ship's log notes"; task: list the notes with their chapters,
summarize every chapter in order, rank the longest chapters. Sampled
(`temperature 0.6`), up to 60K output tokens.

| arm | prompt tokens | prompt tok/s | output tokens | decode tok/s | notes (code / chapter) | chapters missing / repeated | loop signals |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 (fused) | 186,578 | **1,044** | 20,427 | 45.0 | 12/12, 12/12 | 0 / 0 | none |
| 0 (MMQ) | 186,578 | 719 | 17,728 | 43.3 | 12/12, 12/12 | 0 / 0 | none |

Both arms read the whole prompt correctly and finished on their own.

## Prompt speed

| prompt | arm 1 (fused) | arm 0 (MMQ) | ratio |
| --- | ---: | ---: | ---: |
| 6 × ~3.85K tokens (mean) | 1,071 tok/s | 639 tok/s | **1.67×** |
| 186,578 tokens | 1,044 tok/s | 719 tok/s | **1.45×** |

Decode speed is the same in both arms (45.0 vs 43.3 tok/s on the long prompt).

## First-token KL

6 prompts of 3.83–3.89K tokens, `STRATA_DUMP_FIRST_LOGITS`, the layout of
`tools/par/kl_logits.py`.

| direction | mean | median | max |
| --- | ---: | ---: | ---: |
| KL(fused ‖ mmq) | **0.003182** | 0.000525 | 0.015959 |
| KL(mmq ‖ fused) | **0.002944** | 0.000527 | 0.014586 |

Top-1 same: **6/6**. The values are identical to the v1 run's, so the prompt
path is deterministic across server starts in both arms. `docs/AMD_HIP.md`
lists "First-token KL vs MMQ: IQ3_S 0.00284 vs 0.00332" for this switch; the
numbers here are in the same range.

## Greedy repeatability

The same prompt sent three times in a row, greedy, 400 tokens:

| arm | identical pairs | common prefix (chars) | draft accepted / drafted |
| --- | ---: | --- | --- |
| 1 | 0/2 | 222 / 648 | 234/255, 257/270, 203/217 |
| 0 | 0/2 | 222 / 222 | 251/261, 222/244, 231/239 |

The 2nd and 3rd requests are served from the conversation cache (5 new
tokens), so the divergence comes from decoding, not from the prompt path. The
draft acceptance differs on every request, which points at speculative
decoding; it was not isolated (`--drafting off`). Either way, a text difference
between the arms is not evidence about `STRATA_PF_FUSED`.

## Tool-name margin (agentic probe, engine 0.1.59)

The failure that matters in agent work — the model writing `bash` or `ash` where
the request declared `shell` — is rare: the server log has **2 rejected tool
calls in 2,371 generations (0.08%)**, both at long context (48,490 and 182,798
tokens), both recovered. `agentic.py` measures the same decision with the
server's `logprobs` at ~54K tokens, where the fused path is active.

| probe (~54K tokens) | margin arm1 | margin arm0 | Δlogp shell | Δlogp bash | KL (candidates) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `long_code` | 9.46 | 7.98 | +0.0018 | — | 0.00015 |
| `long_bash` | 5.02 | 3.72 | +0.0336 | −1.2699 | 0.00957 |

`margin` is `logprob(shell) − logprob(best wrong name)`. The largest on-path
change in the `shell` logprob is 0.034 nats; the only large move is `bash`
going **down** by 1.27 nats in the fused arm. To turn a 0.08% event into a
common one needs ~4.6 nats of bias, over 100× the measured effect. In 18 real
tool calls per arm (10 probes + 8 nontrivial commands), both arms were 18/18
`shell` with 18/18 `bash -n` clean, up to ~54K tokens.

The short probes are a control: `fused_layout()` needs at least 1,024 prompt
tokens, and `neutral` (119 tokens) returned identical logits in both arms.

Full table and caveats: [results/agentic/report.md](results/agentic/report.md).

## Interpretation

On this card, with this model and preset:

- `STRATA_PF_FUSED=1` did not make the model loop more: one degenerate loop per
  arm in 28 generations, on the same prompt, and none in 16 runs of that prompt
  with the model card's sampling;
- the loops that do occur appear with `temperature` 0 and 0.6, below the model
  card's 1.0 for thinking mode, and in both arms;
- a 187K-token prompt is read correctly with the fused path;
- the first-token KL is in the range the docs give for this switch;
- prompts are read 1.45–1.67× faster, decode unchanged.

## Caveats

- One card, one model, one run per arm. 28 + 8 short generations and one long
  one per arm is a sample, not a benchmark.
- 6 KL prompts (not 24), first-token KL only.
- The sampled loop pass and the long test used `temperature 0.6`, below the
  model card's 1.0 for thinking mode; only the 8-run pass used the server's
  default.
- dflash2 + speculative decoding were on in both arms; `PF_FUSED` was the only
  varied setting.
- Single requests only; a multi-turn agent session with tool calls was not
  simulated.
