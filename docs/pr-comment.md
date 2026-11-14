Following up with the data I promised, now on engine **0.1.59** (the loop/KL run
was 0.1.57). Nothing here asks you to re-open anything — the concern is already
withdrawn — but the numbers are a useful record for anyone who finds the flag.

**Setup:** RX 7800 XT (gfx1101), IQ3_S, 256K native context, dflash2,
`STRATA_HIP_PROMPT_F16=1` and `STRATA_PF_GEMM=1` in both arms; only
`STRATA_PF_FUSED` changes, fresh server per arm, fused marker present in the
`=1` engine log only. Full harness and tables:
https://github.com/myghi63/strata-pf-fused-ab

**Loops (0.1.57).** 14 repetition-inviting prompts × (greedy + temperature
0.6), up to 6,144 tokens: the detector cut 1/14 per mode in each arm, and each
arm had exactly one degenerate loop, on the same prompt (list 120 Portuguese
words starting with "a", no repeats) — with fused in greedy, without fused in
sampled. The same prompt × 8 per arm with no sampling fields (so the server
applies the model card's temperature 1.0 / top_p 0.95 / top_k 20): **0/8
degenerate loops in both arms**. So the loop follows the prompt and the
sampling, not the flag.

**First-token KL (0.1.57, 6 prompts):** mean 0.00318 / 0.00294, top-1 6/6, same
range as the doc's IQ3_S figure.

**Tool-name decision (0.1.59).** The failure that matters in agent work is the
model writing `bash` or `ash` where the request declared `shell`. The server log
has 2 rejected calls in 2,371 generations (0.08%), both at long context (48,490
and 182,798 tokens), both recovered. Measuring the same decision with
`logprobs` at ~54K tokens (where the fused path is active):

| probe (~54K tokens) | margin fused | margin MMQ | Δlogp shell | Δlogp bash | KL (candidates) |
| --- | ---: | ---: | ---: | ---: | ---: |
| code context | 9.46 | 7.98 | +0.0018 | — | 0.00015 |
| code + `bash` prior | 5.02 | 3.72 | +0.0336 | −1.2699 | 0.00957 |

`margin` = `logprob(shell) − logprob(best wrong name)`. The largest on-path
change in the `shell` logprob is 0.034 nats; the only large move is `bash`
going **down** by 1.27 nats with fused. Turning a 0.08% event into a common one
needs ~4.6 nats of bias. In 18 real tool calls per arm (10 probes + 8
nontrivial commands), both arms were 18/18 `shell` with 18/18 `bash -n` clean.

**Speed (0.1.57):** 1.67× at ~3.85K tokens (1,071 vs 639 tok/s), 1.45× at 187K;
decode unchanged.

My read: on this card the flag is free at the output level, and the rare wrong
tool name is context adherence plus the client's sampling (OpenCode sends none,
so the server's temperature-1.0 default applies), not the fused kernel.
