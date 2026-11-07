> **Superseded — first run (v1, 700-token cap).** Kept for the record. Corrections found after publishing:
>
> - **"The server's reasoning-loop detector fired 0 times" means nothing.** `reasoning_loop_recovery` is off by
>   default (`serve/server.py`, #728) and was off in this run, so it could not fire. It also needs ~2,000 words of
>   reasoning before it can, which a 700-token reply never reaches. The harness only counted the "stop" message
>   anyway, not "repeated reasoning detected" (recover mode) or "repeated one token" (`repeat_stop_tokens`).
> - **The cause of the non-reproducible greedy output was a guess, not a measurement.** "The streamed expert cache
>   changes what is resident" was not tested. A later check repeated one prompt three times with the prompt served
>   from the conversation cache (5 new tokens) and the replies still diverged after 222 characters while the draft
>   acceptance counts differed; speculative decoding is the more likely source, still not isolated.
> - **Prompt sizes:** only the six KL prompts are 3.83-3.89K tokens; the loop prompts are ~3.3K.
> - **The "+19.1% at 4K on an RTX 5070" reference is wrong:** `docs/AMD_HIP.md` gives +19.1% for Coder IQ1_M on
>   ROCm 7.14 (an AMD card), and the doc's KL line reads "IQ3_S 0.00284 vs 0.00332" without saying the two numbers
>   are the two KL directions; that reading was mine.
> - 27/28 and 28/28 loop generations hit the 700-token cap while still thinking: the loop test covered only the
>   start of the reasoning.
>
> The KL, the fused-marker check and the prompt-speed numbers below stand. See the top-level RESULTS.md for v2.

# Results — RX 7800 XT, IQ3_S

Measured with this harness on 2026-11-07, engine **0.1.57 (HIP)**.

## Setup

| item | value |
| --- | --- |
| GPU | AMD Radeon RX 7800 XT (gfx1101), 16 GB |
| CPU | Ryzen 7 5700X3D |
| Model | Qwen3.8-Flash-Next, IQ3_S pack |
| Context | 262,144 tokens |
| Preset | `iq3_s-256k-final` |
| Drafting | dflash2, `--spec 6`, `--draft-conf 0.7` |
| Other | `--ple-io mmap`, `--conversation-cache-mib 4608`, `--vram-reserve-mib 1024`, hipBLASLt tuning table |
| Fixed in both arms | `STRATA_HIP_PROMPT_F16=1`, `STRATA_PF_GEMM=1` |
| Varied | `STRATA_PF_FUSED=1` (arm 1) vs `=0` (arm 0) |
| Prompts | 3,834–3,893 tokens (one 8,192-token chunk) |

The switch engaged: the fused-expert marker
(`strata: prompt experts on the fused int8 kernels`) appeared **once in arm 1
and never in arm 0**, and the prompt ring was 512 slots vs 384. This is the
check that the first version of the test failed.

## First-token KL

6 prompts, `STRATA_DUMP_FIRST_LOGITS`, same layout as
`tools/par/kl_logits.py`.

| direction | mean | median | max |
| --- | ---: | ---: | ---: |
| KL(fused ‖ mmq) | **0.003182** | 0.000525 | 0.015959 |
| KL(mmq ‖ fused) | **0.002944** | 0.000527 | 0.014586 |

Top-1 same: **6/6**.

| prompt | KL(fused‖mmq) | KL(mmq‖fused) | top-1 same |
| --- | ---: | ---: | :---: |
| kl_en_consensus | 0.015959 | 0.014586 | yes |
| kl_pt_navegacao | 0.000523 | 0.000529 | yes |
| kl_code_asyncio | 0.001288 | 0.001235 | yes |
| kl_math_svd | 0.000435 | 0.000426 | yes |
| kl_agent_transcript | 0.000527 | 0.000525 | yes |
| kl_json_api | 0.000362 | 0.000364 | yes |

Reference in `docs/AMD_HIP.md` for IQ3_S (24 prompts, both directions):
`0.00284` / `0.00332`. The mean here is in the same range; the distribution is
skewed by `kl_en_consensus` (0.016), and the median of the other five is
~0.0005.

## Loops

14 repetition-inviting prompts × up to 700 tokens × 2 repeats per arm, greedy:

| arm | loops | server reasoning-loop recoveries |
| --- | ---: | ---: |
| 1 (fused) | **0/28** | 0 |
| 0 (MMQ) | **0/28** | 0 |

No degenerate repetition in either arm. Nearly all generations hit the
700-token cap while still thinking (27/28 in arm 1, 28/28 in arm 0), so this
covers the reasoning text, not the final answer.

## Prompt speed

Same 6 requests (~3.85K tokens), `timings.prompt_per_second`:

| arm | mean tok/s | median tok/s | mean prompt ms |
| --- | ---: | ---: | ---: |
| 1 (fused) | **1066.6** | 1068.9 | 3618.8 |
| 0 (MMQ) | 640.4 | 641.9 | 6026.9 |

**1.67×** prompt speed. Decode speed is unchanged (31.6–37.8 vs 31.4–37.9
tok/s), as expected: the switch is a prompt-path optimization.

The Strata docs report +19.1% at 4K on an RTX 5070; this card gains much more
from the fused path. The measurement is a single run per arm, so treat the exact
ratio as indicative.

## Within-arm repeatability

| arm | identical repeat pairs | different |
| --- | ---: | ---: |
| 1 | 0/14 | 14 |
| 0 | 0/14 | 14 |

Greedy decoding here is not bit-reproducible across requests: the streamed
expert cache changes what is resident, which changes summation order. Therefore
the arm-to-arm text differences (all 28 pairs differ) are **not** evidence
against `STRATA_PF_FUSED`; only the loop rate is.

## Interpretation

On this card, with this model and preset:

- the fused prompt path changes the first-token distribution by about the same
  amount the project documents (mean KL ~0.003, top-1 unchanged);
- it did not produce a repetition loop in 28 greedy generations, and the
  server's own reasoning-loop detector fired zero times in both arms;
- it is 1.67× faster at ~3.85K-token prompts, with decode unchanged.

## Caveats

- 6 KL prompts (not 24), first-token KL only (not teacher-forced over all
  positions).
- One run per arm; the loop test is a sample of 28 generations per arm.
- The test runs with dflash2 + speculative decoding; `PF_FUSED` was the only
  varied setting.
- Nearly all loop generations hit the token cap while still thinking (27/28 in
  arm 1, 28/28 in arm 0).
