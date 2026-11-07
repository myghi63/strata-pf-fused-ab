# strata-pf-fused-ab

A small, self-contained A/B harness for **`STRATA_PF_FUSED`** (Strata's fused
int8 prompt-expert kernels) on a single GPU. It answers two questions with
numbers:

1. Does turning the fused prompt path on make the model fall into repetition
   loops?
2. How far does it move the model's first-token distribution (KL), and how much
   prompt speed does it buy?

It was written to answer the maintainer's question in
[Strata PR #2392](https://github.com/Niko1221/Strata/pull/2392) and was run on
an RX 7800 XT (gfx1101). The measured numbers are in [RESULTS.md](RESULTS.md).

## What it measures

Two server starts, one per arm, with **everything identical except
`STRATA_PF_FUSED`**:

| arm | `STRATA_PF_FUSED` |
| --- | --- |
| 1 | `1` (fused) |
| 0 | `0` (MMQ) |

Each arm runs the same fixed prompt set:

1. **First-token KL** — 6 long prompts, `max_tokens=4`. The engine dumps the
   logits of the first generated window (`STRATA_DUMP_FIRST_LOGITS`) and `kl.py`
   computes `KL(fused‖mmq)`, `KL(mmq‖fused)` and top-1 agreement. This is the
   same measurement and layout as the repository's `tools/par/kl_logits.py`.
2. **Greedy loop test** — 14 prompts that invite repetition (counting, lists,
   tables, JSON, dialogue, long essays…), greedy, up to 700 tokens, 2 repeats.
   `report.py` uses three independent signals so a single one cannot flag a
   prompt: exactly periodic tail, a line / 10-word n-gram repeated 8+ times, or
   an abnormally high zlib compression ratio. Hitting `max_tokens`
   (`finish_reason=length`) is reported separately — it is not a loop.
3. **Prompt speed** — read from the same 6 KL requests
   (`timings.prompt_per_second`).
4. **The server's own loop detector** — Strata counts reasoning-loop recoveries
   (`the thinking repeated the same passages`); the script copies that count per
   arm.
5. **Within-arm repeatability** — the two repeats of each prompt are compared.
   Greedy decoding in this build is **not** bit-reproducible across requests
   (the streamed expert cache changes what is resident), so a text difference
   between arms alone does not implicate `STRATA_PF_FUSED`. Only a difference in
   the loop *rate* is evidence.

## Why the prompts are ~4K tokens

The fused prompt path only runs on a streamed chunk of
`stream_all_min()` tokens or more — 1024 by default
(`src/prefill/prefill.cpp`, `fused_layout()`). With ~600-token prompts **both**
arms run MMQ, the first-token KL comes out exactly `0.0`, and the run measures
nothing. The first version of this harness fell into that trap; the committed
version prefixes every prompt with the same neutral ~3.3K-token context (the six
KL texts concatenated) and keeps the question / loop instruction at the end, so
every request is 3.8–3.9K tokens.

The script checks this for you: it reports the fused-expert marker per arm and
the KL prompt sizes, and warns if the marker never appeared in arm 1.

## Requirements

- Strata installed and able to serve the model (`strata launch`).
- A saved launch preset that serves it. The examples use `iq3_s-256k-final`;
  set `PRESET` to yours.
- One GPU, enough VRAM for the preset. Keep the preset's VRAM reserve
  (`--vram-reserve-mib`) as it is; the harness does not change it.
- `python3` with `numpy`.

The harness changes **nothing** in the preset: `STRATA_PF_FUSED` and
`STRATA_DUMP_FIRST_LOGITS` are passed as per-start `--set` overrides, so the
preset file is untouched.

## Quick start

```bash
# Stops the running server (if any), runs both arms, leaves the server stopped.
PRESET=iq3_s-256k-final bash run_all.sh --yes

# ... and starts the preset again at the end:
PRESET=iq3_s-256k-final bash run_all.sh --yes --restore
```

Without `--yes` the script only prints what it would do and exits, so it cannot
take down a server by accident.

Runtime: ~25–35 minutes on an RX 7800 XT (two engine loads + 6 KL requests +
28 generations per arm, each request with a ~4K-token prompt). Disk: ~15 MB
(the logits dumps).

The API key is read from the preset's `api_key` (or `STRATA_API_KEY`, or
`--key`). It is never written by this harness.

## What it writes

| file | contents |
| --- | --- |
| `report.md` / `report.json` | loop table per prompt, arm 1 vs arm 0, speed, within-arm repeatability |
| `kl.md` / `kl.json` | first-token KL in both directions, top-1, per-prompt table |
| `arm1.engine.log` / `arm0.engine.log` | the engine log lines of each arm |
| `arm1.server.log` / `arm0.server.log` | the server log lines of each arm (includes reasoning-loop recoveries) |
| `dumps/fused1.*` / `dumps/fused0.*` | raw float32 logits (248,320 per file) |
| `out/arm*/...` | full requests and responses |

## How to read it

- **KL** — compare with the reference for IQ3_S in `docs/AMD_HIP.md`
  (mean `0.00284` / `0.00332` over 24 prompts). The same order of magnitude
  (≈0.001–0.005) means the fused path is not moving the distribution more than
  expected. **Exactly 0** means both arms ran the same path (prompts too short).
- **Top-1** — how many prompts pick the same first token. It is expected to be
  high; on this model the first token is always the same word because of the
  chat template, so the KL is the informative number.
- **Loops** — the number that matters is arm 1 vs arm 0. If arm 1 loops and
  arm 0 does not, the report reproduces on your card. If both are 0, the fused
  path did not cause a loop in this sample.
- **Repeatability** — if the two repeats inside the same arm already differ,
  the arm-to-arm text comparison is noise; only the loop rate counts.
- **Speed** — `fused / mmq`. The docs report +19.1% at 4K and +8.4% at 16K on an
  RTX 5070; a different card can gain much more (see RESULTS.md).

## Caveats

- 6 KL prompts, not 24; 14 loop prompts × 2 repeats. This is a sample, not a
  benchmark.
- It is **first-token KL**, not teacher-forced KL over all positions. For all
  positions, the engine's `--dump-logits` writes one line per position, but that
  needs tokenizing the text and running the engine outside the server.
- The test runs with the production stack (dflash2 + speculative decoding). If a
  loop appears, repeat that prompt with `--drafting off` to separate the
  drafter's effect from `STRATA_PF_FUSED`.
- `STRATA_HIP_PROMPT_F16` and `STRATA_PF_GEMM` stay on in both arms: the
  maintainer confirmed those are the correct gfx11 path.
- With a 700-token cap the model may still be thinking when it hits the limit,
  so the loop test can cover the reasoning text but not the final answer.

## Files

| file | role |
| --- | --- |
| `run_all.sh` | orchestrates both arms: stop, launch with `--set`, client, logs, analysis |
| `client.py` | sends the fixed prompt set, saves responses, timings and dump indices |
| `prompts.py` | the 6 KL prompts, the 14 loop prompts, and the long-context padding |
| `report.py` | loop detector, arm comparison, speed, within-arm repeatability |
| `kl.py` | first-token KL from `STRATA_DUMP_FIRST_LOGITS` dumps |
| `results/` | the measured report and KL tables from the RX 7800 XT run |
| `docs/pr-comment.md` | the comment posted to PR #2392 |

## License

MIT — see [LICENSE](LICENSE).
