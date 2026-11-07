#!/usr/bin/env python3
"""First-token KL between the two arms, from the engine's STRATA_DUMP_FIRST_LOGITS files.

  python3 kl.py

Each request dumps one float32 row of vocabulary logits for the first generated
token (src/program/generate.cpp, `first_window`). The server's own warm-up
request dumps one too, so the index of a prompt's dump is NOT its position in
the list: client.py records it as `dump_index` (the number of dump files before
the request) and this script uses that.

Reports KL(fused||mmq), KL(mmq||fused), top-1 agreement, and the per-prompt table.
This is the same measurement and the same layout as tools/par/kl_logits.py.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prompts import KL  # noqa: E402

DUMPS = HERE / "dumps"


def logsoftmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = float(np.nanmax(x))
    return x - m - np.log(np.exp(x - m).sum())


def dump_index(arm: str, prompt_id: str, fallback: int) -> int:
    """The dump file number the client recorded for this prompt (warm-up included)."""
    p = HERE / "out" / f"arm{arm}" / "kl" / f"{prompt_id}.json"
    try:
        return int(json.loads(p.read_text()).get("dump_index", fallback))
    except Exception:  # noqa: BLE001
        return fallback


def main() -> int:
    rows = []
    for i, p in enumerate(KL):
        ia = dump_index("1", p["id"], i)
        ib = dump_index("0", p["id"], i)
        pa, pb = DUMPS / f"fused1.{ia}", DUMPS / f"fused0.{ib}"
        if not pa.exists() or not pb.exists():
            raise SystemExit(
                f"missing dump pair for {p['id']}: {pa} / {pb}. "
                f"Check that STRATA_DUMP_FIRST_LOGITS was set and that client.py ran with --dumps.")
        a = np.fromfile(pa, dtype="<f4")
        b = np.fromfile(pb, dtype="<f4")
        if a.size != b.size:
            raise SystemExit(f"{p['id']}: dump sizes differ ({a.size} vs {b.size})")
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            print(f"{p['id']}: non-finite logits, skipped")
            continue
        la, lb = logsoftmax(a), logsoftmax(b)
        kl_ab = float((np.exp(la) * (la - lb)).sum())
        kl_ba = float((np.exp(lb) * (lb - la)).sum())
        rows.append({"index": i, "id": p["id"], "dump1": ia, "dump0": ib,
                     "kl_fused_mmq": kl_ab, "kl_mmq_fused": kl_ba,
                     "top1_same": int(a.argmax() == b.argmax()),
                     "top1_fused": int(a.argmax()), "top1_mmq": int(b.argmax())})

    ab = np.array([r["kl_fused_mmq"] for r in rows])
    ba = np.array([r["kl_mmq_fused"] for r in rows])
    top = sum(r["top1_same"] for r in rows)
    md = ["# First-token KL, STRATA_PF_FUSED=1 vs =0", "",
          f"Positions (prompts): **{len(rows)}**", "",
          "| direction | mean | median | max |",
          "| --- | ---: | ---: | ---: |",
          f"| KL(fused \\|\\| mmq) | {ab.mean():.6f} | {np.median(ab):.6f} | {ab.max():.6f} |",
          f"| KL(mmq \\|\\| fused) | {ba.mean():.6f} | {np.median(ba):.6f} | {ba.max():.6f} |",
          "",
          f"**Top-1 same: {top}/{len(rows)}**", "",
          "| prompt | KL(fused\\|\\|mmq) | KL(mmq\\|\\|fused) | top-1 same |",
          "| --- | ---: | ---: | :---: |"]
    for r in rows:
        md.append(f"| {r['id']} | {r['kl_fused_mmq']:.6f} | {r['kl_mmq_fused']:.6f} | "
                  f"{'yes' if r['top1_same'] else 'no'} |")
    md.append("")
    md.append("Reference from docs/AMD_HIP.md, IQ3_S, pooled 24 prompts: "
              "PF_FUSED first-token KL 0.00284 vs 0.00332 (both directions).")
    (HERE / "kl.md").write_text("\n".join(md), encoding="utf-8")
    (HERE / "kl.json").write_text(json.dumps(
        {"rows": rows, "mean_fused_mmq": float(ab.mean()), "mean_mmq_fused": float(ba.mean()),
         "median_fused_mmq": float(np.median(ab)), "max_fused_mmq": float(ab.max()),
         "top1_same": top, "n": len(rows)}, indent=2), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
