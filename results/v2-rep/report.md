# STRATA_PF_FUSED A/B

## Prompt speed (the KL requests)

| arm | prompts | prompt tokens | mean tok/s | median tok/s | mean prompt ms |
| --- | ---: | ---: | ---: | ---: | ---: |

## Server loop detectors (whole server log of each arm)

| arm | reasoning loop, stop | reasoning loop, recover | one token repeated |
| --- | ---: | ---: | ---: |
| 1 | 2 | 0 | 0 |
| 0 | 4 | 0 | 0 |

## Repetition loops (short prompts)

### Summary

| mode | arm | cut by the server's loop detector | text heuristics fired | hit max_tokens | wrote an answer |
| --- | --- | ---: | ---: | ---: | ---: |

The server's detector is the primary signal. The text heuristics are deliberately sensitive and also fire on tasks whose correct output is repetitive (a 50-item JSON list, a numbered poem, a Fibonacci table); read the flagged texts below before calling one a loop.

## Greedy repeatability (same prompt, same server, 3 requests)

| arm | identical pairs | common prefix (chars) 0-1 / 1-2 | draft accepted / drafted |
| --- | ---: | --- | --- |

The 2nd and 3rd requests reuse the cached prompt (conversation cache), so a difference between them comes from decoding, not from the prompt path. The draft acceptance counts differ between requests; speculative decoding is the likely source, but this run does not isolate it (`--drafting off` would).

## Loop rate, one prompt repeated with the server's default sampling

No sampling fields in the request: the server applies the model card's (temperature 1.0, top_p 0.95, top_k 20 with thinking), as for a client such as opencode. `coverage` is the server detector's share of the recent reasoning inside passages seen three times (it cuts at 0.25).

| arm | runs | cut by the server | coverage of the cuts | other loop signals | wrote an answer |
| --- | ---: | ---: | --- | ---: | ---: |
| 1 | 8 | 2 | 0.349, 0.264 | 3 | 6/8 |
| 0 | 8 | 4 | 0.358, 0.250, 0.343, 0.329 | 2 | 4/8 |

## Long context (~187K-token prompt, sampled, up to 60K tokens)

| arm | prompt tokens | prompt tok/s | output tokens | decode tok/s | finish | notes found / right chapter | chapters missing / repeated | loop signals |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |

## Outputs flagged as loops (last 500 characters)

None.
