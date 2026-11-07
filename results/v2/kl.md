# First-token KL, STRATA_PF_FUSED=1 vs =0

Positions (prompts): **6**

| direction | mean | median | max |
| --- | ---: | ---: | ---: |
| KL(fused \|\| mmq) | 0.003182 | 0.000525 | 0.015959 |
| KL(mmq \|\| fused) | 0.002944 | 0.000527 | 0.014586 |

**Top-1 same: 6/6**

| prompt | KL(fused\|\|mmq) | KL(mmq\|\|fused) | top-1 same |
| --- | ---: | ---: | :---: |
| kl_en_consensus | 0.015959 | 0.014586 | yes |
| kl_pt_navegacao | 0.000523 | 0.000529 | yes |
| kl_code_asyncio | 0.001288 | 0.001235 | yes |
| kl_math_svd | 0.000435 | 0.000426 | yes |
| kl_agent_transcript | 0.000527 | 0.000525 | yes |
| kl_json_api | 0.000362 | 0.000364 | yes |

Reference from docs/AMD_HIP.md (STRATA_PF_FUSED, pooled 24 prompts): "First-token KL vs MMQ: IQ3_S 0.00284 vs 0.00332" (the doc does not say what each number is).