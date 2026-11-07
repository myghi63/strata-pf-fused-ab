Re-ran the `STRATA_PF_FUSED` A/B with a heavier test and corrected my first write-up (it claimed the server's loop detector fired 0 times, but `reasoning_loop_recovery` is off by default, and a 700-token cap never reaches the ~2,000 words it needs). Full numbers: https://github.com/myghi63/strata-pf-fused-ab/blob/main/RESULTS.md

Setup: RX 7800 XT (gfx1101), IQ3_S, 256K native context, engine 0.1.57, dflash2, `STRATA_HIP_PROMPT_F16=1` and `STRATA_PF_GEMM=1` in both arms; only `STRATA_PF_FUSED` changes, fresh server start per arm, `"reasoning_loop_recovery": "stop"` on for the run. Fused marker present in the `=1` engine log only.

**Loops, 14 repetition-inviting prompts × (greedy + temperature 0.6), up to 6,144 tokens:** the detector cut 1/14 per mode in each arm. Reading the texts, each arm had exactly **one degenerate loop**, on the same prompt (list 120 Portuguese words starting with "a", no repeats): with fused in greedy (`62 afogar? no.` ×150, coverage 1.000), without fused in sampled (`Let's use "absurdo"? no.` over and over). The other cuts were re-drafts of long lists (coverage 0.28–0.32).

**Same prompt × 8 per arm with no sampling fields** (so the server applies `MODEL_CARD_SAMPLING`, temperature 1.0 / top_p 0.95 / top_k 20): **0/8 degenerate loops in both arms**; the detector cut 2/8 (fused) and 4/8 (MMQ) runs, all of them the model re-writing its word list (coverage 0.25–0.36). So the detector also fires on legitimate re-drafts of list tasks.

**Long context:** 186,578-token prompt (Moby-Dick ch. 1–80 + 12 inserted notes), both arms recalled 12/12 notes with the right chapter and summarized all 80 chapters in order, no loop. Prompt 1,044 vs 719 tok/s.

**Speed:** 1.67× at ~3.85K tokens (1,071 vs 639 tok/s), 1.45× at 187K; decode unchanged.

**First-token KL** (6 prompts): 0.00318 / 0.00294, top-1 6/6, same range as the doc's IQ3_S figure.

So on this card I can reproduce a reasoning loop, but it happens with and without `STRATA_PF_FUSED`, only below the model card's temperature, and on a prompt that invites it. @skuridin, what sampling were you using when you saw the loops (greedy / temperature 0, or another client's defaults)? If you have a prompt that loops with fused and not with MMQ, I can run it through the same harness (`CLIENT_ARGS="--phases rep --rep-prompt …"`).
