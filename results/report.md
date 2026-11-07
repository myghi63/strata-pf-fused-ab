# STRATA_PF_FUSED A/B

## Prompt speed (from the KL requests)

| arm | prompts | mean tok/s | median tok/s | mean prompt ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 6 | 1066.6 | 1068.9 | 3618.8 |
| 0 | 6 | 640.4 | 641.9 | 6026.9 |

fused on/off speed ratio: **1.666x**

## Repetition loops (greedy, up to 700 tokens, 2 repeats)

| prompt | arm1 finish | arm1 signals | arm0 finish | arm0 signals | same text |
| --- | --- | --- | --- | --- | --- |
| loop_count_300.r0 | length | - | length | - | no (prefix 63 ch) |
| loop_count_300.r1 | length | - | length | - | no (prefix 534 ch) |
| loop_dialogue_40.r0 | length | - | length | - | no (prefix 344 ch) |
| loop_dialogue_40.r1 | length | - | length | - | no (prefix 0 ch) |
| loop_essay_reading.r0 | length | - | length | - | no (prefix 547 ch) |
| loop_essay_reading.r1 | length | - | length | - | no (prefix 65 ch) |
| loop_explain_compiler.r0 | length | - | length | - | no (prefix 180 ch) |
| loop_explain_compiler.r1 | length | - | length | - | no (prefix 39 ch) |
| loop_fibonacci.r0 | length | - | length | - | no (prefix 371 ch) |
| loop_fibonacci.r1 | length | - | length | - | no (prefix 371 ch) |
| loop_json_50.r0 | length | - | length | - | no (prefix 30 ch) |
| loop_json_50.r1 | length | - | length | - | no (prefix 560 ch) |
| loop_poem_sea.r0 | length | - | length | - | no (prefix 0 ch) |
| loop_poem_sea.r1 | length | - | length | - | no (prefix 460 ch) |
| loop_process_50.r0 | length | - | length | - | no (prefix 0 ch) |
| loop_process_50.r1 | length | - | length | - | no (prefix 561 ch) |
| loop_summarize_cycle.r0 | length | - | length | - | no (prefix 43 ch) |
| loop_summarize_cycle.r1 | length | - | length | - | no (prefix 43 ch) |
| loop_table_40.r0 | length | - | length | - | no (prefix 627 ch) |
| loop_table_40.r1 | length | - | length | - | no (prefix 413 ch) |
| loop_tcp_steps.r0 | length | - | length | - | no (prefix 267 ch) |
| loop_tcp_steps.r1 | length | - | length | - | no (prefix 267 ch) |
| loop_translate_30.r0 | length | - | length | - | no (prefix 306 ch) |
| loop_translate_30.r1 | length | - | length | - | no (prefix 222 ch) |
| loop_while_python.r0 | stop | - | stop | - | no (prefix 325 ch) |
| loop_while_python.r1 | stop | - | stop | - | no (prefix 166 ch) |
| loop_words_a.r0 | length | - | length | - | no (prefix 896 ch) |
| loop_words_a.r1 | length | - | length | - | no (prefix 896 ch) |

**Loops: arm1 (fused on) 0/28 generations, arm0 (fused off) 0/28.**

## Within-arm repeatability (r0 vs r1, same arm, same prompt)

| arm | identical repeats | different repeats |
| --- | ---: | ---: |
| 1 | 0 | 14 |
| 0 | 0 | 14 |

Greedy decoding here is not bit-reproducible across requests (the streamed expert cache changes what is resident), so an arm-to-arm text difference alone does not implicate `STRATA_PF_FUSED`. Only a difference in the *loop rate* is evidence.

## Outputs that looped (first 600 characters)

None. No repetition loop in either arm.
