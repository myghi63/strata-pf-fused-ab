# Comment for PR #2392 (fill in the placeholders if anything changes)

Tested `STRATA_PF_FUSED` on an RX 7800 XT (gfx1101, Ryzen 7 5700X3D), IQ3_S,
256K context, engine 0.1.57, everything else fixed (same preset, `--ple-io mmap`,
dflash2, `--spec 6`, conversation cache, hipBLASLt table). Only
`STRATA_PF_FUSED` changed between the two arms; each arm is a fresh server start.

Prompts are 3.83–3.89K tokens so the fused prompt path actually runs
(`fused_layout()` needs a streamed chunk >= `stream_all_min()`, 1024 tokens).
Verified: the fused-expert marker appears once in the `=1` log and never in the
`=0` log, and the prompt ring is 512 slots vs 384.

**First-token KL** (`STRATA_DUMP_FIRST_LOGITS`, same layout as
`tools/par/kl_logits.py`; 6 prompts):

| direction | mean | median | max |
| --- | ---: | ---: | ---: |
| KL(fused ‖ mmq) | 0.003182 | 0.000525 | 0.015959 |
| KL(mmq ‖ fused) | 0.002944 | 0.000527 | 0.014586 |

Top-1 same: 6/6. This is in the same range as the doc's IQ3_S number
(0.00284 / 0.00332, 24 prompts).

**Greedy loop test** (14 repetition-inviting prompts × up to 700 tokens × 2
repeats per arm): **0/28 loops with fused, 0/28 with MMQ**. The server's own
reasoning-loop recoveries (`the thinking repeated the same passages`): 0 vs 0.
So the loop did not reproduce on this card in this sample.

**Prompt speed at ~3.85K tokens**: 1066.6 tok/s fused vs 640.4 tok/s MMQ,
**1.67×**; decode speed unchanged (31.6–37.8 vs 31.4–37.9 tok/s). The doc
reports +19.1% at 4K on a 5070; this card gains much more from the fused prompt
path.

Caveats: 6 KL prompts (not 24), first-token KL only (not teacher-forced over all
positions), and nearly every loop generation hit the 700-token cap while still
thinking (27/28 and 28/28), so the loop test covers the reasoning text rather
than the final answer. Greedy decoding in this build is also not bit-reproducible
across requests (the streamed expert cache changes what is resident): 0 of 14
repeat pairs were identical within either arm, so only the loop *rate* is
evidence, not textual equality.
