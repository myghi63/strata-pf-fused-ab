# Tool-name margin: `STRATA_PF_FUSED=1` vs `=0`

Measured on 2026-11-14 with the harness in this repository, engine **0.1.59
(HIP)**, preset `iq3_s-256k-final` (256K context, dflash2, `--ple-io mmap`,
`--vram-reserve-mib 1024`), RX 7800 XT (gfx1101), Qwen3.8-Flash-Next IQ3_S.
Both arms start from the same preset; only `STRATA_PF_FUSED` changes
(`--set STRATA_PF_FUSED=1` / `=0`, fresh server each time). The fused-expert
marker is present in arm 1's engine log and absent in arm 0's.

## The question

The loop and KL tests answer "does it loop" and "how far did the distribution
move on average". A third failure mode matters in agent work: the model writing
`bash` (or `ash`) where the request declared `shell`. On this card the server
log has **2 rejected tool calls in 2,371 generations (0.08%)**, both at long
context (48,490 and 182,798 prompt tokens), one naming `ash` and one naming
`function`; both recovered on the next request.

Counting a 0.08% event directly would need thousands of requests per arm. This
probe measures the same decision where it is cheap.

## Method

With thinking off, the model writes the tool call as one line of JSON in the
answer, so the name token is normal content and the server's `logprobs` give its
whole distribution. The number is

    margin = logprob(shell) - logprob(best wrong name)

plus the KL over the candidate names only (`shell`, `bash`, `ash`, `sh`,
`function`, …), renormalised: the flag's effect on the decision itself.

Two forms show up and both are parsed: the JSON the probe asks for
(`"name": "shell"`) and the template's native call (`<function=shell>`), which
the model prefers at long context.

**The short probes are a control, not evidence.** `fused_layout()` needs at
least `stream_all_min()` (1,024) prompt tokens, so a 119–404-token prompt never
reaches the fused path. The `neutral` probe returning the *same* logits in both
arms (to four decimals) is the expected result and confirms the arms differ only
in the flag. Only the ~54K-token probes exercise it.

## Margins (fused path active, ~54K tokens)

| probe | prompt tokens (arm1 / arm0) | margin arm1 | margin arm0 | Δlogp shell | Δlogp bash | KL (candidates) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `long_code` | 53,627 / 53,330 | **9.46** | **7.98** | +0.0018 | — | 0.00015 |
| `long_bash` | 53,679 / 53,382 | **5.02** | **3.72** | +0.0336 | **−1.2699** | 0.00957 |

`long_code` is ~54K tokens of source files; `long_bash` is the same context
preceded by examples that call a `bash` tool. In both, `shell` wins by several
nats. In `long_bash` the fused arm makes `bash` **less** likely by 1.27 nats, so
the largest on-path KL (0.0096) has the protective sign.

Control probes (below the 1,024-token threshold):

| probe | margin arm1 | margin arm0 | Δlogp shell | KL (candidates) |
| --- | ---: | ---: | ---: | ---: |
| `neutral` | 7.93 | 7.93 | 0.0000 | 0.00000 |
| `prior_ash` | 8.27 | 8.45 | −0.0003 | 0.00002 |
| `prior_bash` | 3.89 | 3.75 | +0.0056 | 0.00027 |

The small differences there are run-to-run noise (the engine's streamed expert
cache is not bit-reproducible), which bounds the measurement noise at ~0.006
nats on `shell`.

## Real tool-call path (`tools` declared, `tool_choice: auto`)

| probe | arm1 | arm0 |
| --- | --- | --- |
| `neutral` | `shell`, `bash -n` ok | `shell`, `bash -n` ok |
| `prior_bash` | `shell`, `bash -n` ok | `shell`, `bash -n` ok |
| `prior_ash` | `shell`, `bash -n` ok | `shell`, `bash -n` ok |
| `long_code` (~54K) | `shell`, `bash -n` ok | `shell`, `bash -n` ok |
| `long_bash` (~54K) | `shell`, `bash -n` ok | `shell`, `bash -n` ok |

## Nontrivial commands (`bash -n`, no sampling fields)

Eight ordinary agent tasks per arm (find/sort/grep/du/git, with quoting and
pipes), sent through the real tool path with no sampling fields — what OpenCode
does:

| arm | calls | named `shell` | `bash -n` ok | syntax error | other name |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 (fused) | 8 | 8 | 8 | 0 | 0 |
| 0 (MMQ) | 8 | 8 | 8 | 0 | 0 |

So in 18 real tool calls per arm (10 probes + 8 syntax), up to ~54K tokens:
zero wrong names, zero syntax errors.

## What the numbers rule out

A perturbation of KL ~0.003 nats changes a logprob by ~0.003 nats. The measured
on-path KL is 0.00015–0.0096 nats and the largest change in the `shell` logprob
is **0.034 nats**. For the flag to turn a 0.08% event into a common one you
would need ~ln(100) ≈ **4.6 nats** of bias — over 100× the measured effect, and
in the wrong direction on the only probe where `bash` moved much.

To flip the name, the two names would have to be tied to within ~0.05 nats. At
temperature 1.0 (what OpenCode gets: it sends no sampling, so the server applies
the model card's `temperature 1.0, top_p 0.95, top_k 20`) such a near-tie would
flip often, not rarely. The observed margins are 3.7–9.5 nats.

Conclusion: on this card `STRATA_PF_FUSED` cannot explain the rare `ash`, nor a
systematic `bash`. The current log's two failures are different names, at long
context, both schema/prior echoes — context adherence and sampling, not a fixed
kernel bias.

## Caveats

- The logprob path conditions on a prompt that names `shell`, which inflates the
  margin; the real tool path is unbiased but exposes no logprobs for the name, so
  it can only count outcomes (10/10 `shell` per arm).
- 18 tool calls per arm cannot prove a 0.08% rate; they bound the *syntax* and
  *name* failures at short and ~54K context, not at 180K.
- Greedy decoding is not bit-reproducible here, so only rates and margins are
  evidence, not token-for-token equality.
