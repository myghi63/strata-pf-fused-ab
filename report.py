#!/usr/bin/env python3
"""Analyse the two arms: repetition loops, arm-to-arm divergence and prompt speed.

  python3 report.py            # reads out/arm1 and out/arm0, writes report.md/report.json

A "loop" here is the degenerate repetition the maintainer described: the model
starts repeating a token, a line, a phrase or a whole block and does not leave
it. Three independent signals are used, so a single one cannot flag a prompt:
  * the tail is exactly periodic (period <= 300 characters, at least 3 repeats);
  * one line (or one 10-word phrase) repeats 8+ times;
  * the text compresses absurdly well (zlib ratio < 0.16 on 1,200+ characters).
`finish == "length"` is reported separately: hitting max_tokens is not a loop.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys
import zlib
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"


def tail_period(s: str, max_period: int = 400) -> int | None:
    n = len(s)
    for p in range(2, min(max_period, n // 3) + 1):
        block = s[-p:]
        if sum(ch.isalnum() for ch in block) < 15:
            continue
        if s[-3 * p:-2 * p] == s[-2 * p:-p] == block:
            return p
    return None


def ngram_top(words: list[str], n: int) -> int:
    if len(words) < n * 2:
        return 0
    c = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return c.most_common(1)[0][1]


def metrics(text: str) -> dict:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lc = Counter(lines)
    top_line, top_line_n = lc.most_common(1)[0] if lc else ("", 0)
    words = text.split()
    n4 = len(words) - 3
    distinct4 = (len({tuple(words[i:i + 4]) for i in range(n4)}) / n4) if n4 > 0 else 0.0
    ratio = len(zlib.compress(text.encode("utf-8"), 9)) / max(1, len(text))
    tp = tail_period(text)
    n10 = ngram_top(words, 10)
    n6 = ngram_top(words, 6)
    triggers = []
    if top_line_n >= 8 and len(top_line) >= 8:
        triggers.append(f"line x{top_line_n}")
    if tp is not None and tp <= 300:
        triggers.append(f"tail period {tp}")
    if n10 >= 6:
        triggers.append(f"10-gram x{n10}")
    if n6 >= 12:
        triggers.append(f"6-gram x{n6}")
    if ratio < 0.16 and len(text) > 1200:
        triggers.append(f"zlib {ratio:.3f}")
    return {
        "chars": len(text),
        "words": len(words),
        "zlib": round(ratio, 4),
        "tail_period": tp,
        "top_line_n": top_line_n,
        "top_line": top_line[:80],
        "ngram6": n6,
        "ngram10": n10,
        "distinct4": round(distinct4, 4),
        "triggers": triggers,
        "loop": bool(triggers),
    }


def text_of(rec: dict) -> tuple[str, str]:
    ch = rec["response"].get("choices") or [{}]
    msg = ch[0].get("message") or {}
    content = msg.get("content") or ""
    reason = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return content, reason


def finish_of(rec: dict) -> str:
    ch = rec["response"].get("choices") or [{}]
    return ch[0].get("finish_reason") or "?"


def timings_of(rec: dict) -> dict:
    t = rec["response"].get("timings") or {}
    return t if isinstance(t, dict) else {}


def load_arm(arm: str) -> dict:
    d = OUT / f"arm{arm}"
    loops, kls = {}, {}
    for p in sorted((d / "loop").glob("*.json")):
        loops[p.stem] = json.loads(p.read_text())
    for p in sorted((d / "kl").glob("*.json")):
        kls[p.stem] = json.loads(p.read_text())
    return {"loop": loops, "kl": kls}


def speed_table(kls: dict) -> dict:
    rows = []
    for name, rec in kls.items():
        t = timings_of(rec)
        pn = t.get("prompt_n")
        ms = t.get("prompt_ms")
        pps = t.get("prompt_per_second")
        if pn and ms:
            rows.append({"id": name, "tokens": pn, "prompt_ms": round(ms, 1),
                         "tok_s": pps if pps is not None else round(pn / (ms / 1000), 1),
                         "decode_tok_s": t.get("predicted_per_second")})
    return {"rows": rows,
            "mean_tok_s": round(statistics.mean(r["tok_s"] for r in rows), 1) if rows else None}


def main() -> int:
    if not (OUT / "arm1").exists() or not (OUT / "arm0").exists():
        raise SystemExit(f"missing {OUT}/arm1 or arm0; run run_all.sh first")
    arms = {a: load_arm(a) for a in ("1", "0")}
    md = ["# STRATA_PF_FUSED A/B", ""]

    # ---- speed (from the KL requests: long prompt, max_tokens=1)
    md += ["## Prompt speed (from the KL requests)", "",
           "| arm | prompts | mean tok/s | median tok/s | mean prompt ms |",
           "| --- | ---: | ---: | ---: | ---: |"]
    speeds = {}
    for a in ("1", "0"):
        s = speed_table(arms[a]["kl"])
        speeds[a] = s
        vals = [r["tok_s"] for r in s["rows"] if r["tok_s"]]
        med = round(statistics.median(vals), 1) if vals else None
        mean_ms = round(statistics.mean(r["prompt_ms"] for r in s["rows"]), 1) if s["rows"] else None
        md.append(f"| {a} | {len(vals)} | {s['mean_tok_s']} | {med} | {mean_ms} |")
    if speeds["1"]["mean_tok_s"] and speeds["0"]["mean_tok_s"]:
        md.append("")
        md.append(f"fused on/off speed ratio: **{speeds['1']['mean_tok_s'] / speeds['0']['mean_tok_s']:.3f}x**")
    md.append("")

    # ---- loops
    md += ["## Repetition loops (greedy, up to 700 tokens, 2 repeats)", "",
           "| prompt | arm1 finish | arm1 signals | arm0 finish | arm0 signals | same text |",
           "| --- | --- | --- | --- | --- | --- |"]
    loop_count = {"1": 0, "0": 0}
    total = 0
    detail = []
    names = sorted(set(arms["1"]["loop"]) | set(arms["0"]["loop"]))
    for name in names:
        r1 = arms["1"]["loop"].get(name)
        r0 = arms["0"]["loop"].get(name)
        if not r1 or not r0:
            continue
        c1, x1 = text_of(r1)
        c0, x0 = text_of(r0)
        m1 = metrics(c1 + ("\n" + x1 if x1 else ""))
        m0 = metrics(c0 + ("\n" + x0 if x0 else ""))
        total += 1
        for a, m in (("1", m1), ("0", m0)):
            if m["loop"]:
                loop_count[a] += 1
        f1, f0 = c1 + ("\n" + x1 if x1 else ""), c0 + ("\n" + x0 if x0 else "")
        same = "yes" if f1 == f0 else f"no (prefix {common_prefix(f1, f0)} ch)"
        md.append(f"| {name} | {finish_of(r1)} | {'; '.join(m1['triggers']) or '-'} | "
                  f"{finish_of(r0)} | {'; '.join(m0['triggers']) or '-'} | {same} |")
        detail.append({"id": name, "arm1": {**m1, "finish": finish_of(r1)},
                       "arm0": {**m0, "finish": finish_of(r0)}, "same_text": f1 == f0,
                       "prefix_chars": common_prefix(f1, f0)})
    md += ["", f"**Loops: arm1 (fused on) {loop_count['1']}/{total} generations, "
              f"arm0 (fused off) {loop_count['0']}/{total}.**", ""]

    srv = {}
    for a in ("1", "0"):
        p = OUT / f"arm{a}.server_loops"
        try:
            srv[a] = int(p.read_text().strip() or 0)
        except Exception:  # noqa: BLE001
            srv[a] = None
    if any(v is not None for v in srv.values()):
        md += [f"Server's own reasoning-loop recoveries (`the thinking repeated the same passages`): "
               f"arm1 {srv['1']}, arm0 {srv['0']}.", ""]

    # ---- within-arm determinism: how much of the arm-to-arm text difference is noise
    md += ["## Within-arm repeatability (r0 vs r1, same arm, same prompt)", "",
           "| arm | identical repeats | different repeats |", "| --- | ---: | ---: |"]
    det = {}
    for a in ("1", "0"):
        same = diff = 0
        for name, rec in arms[a]["loop"].items():
            if not name.endswith(".r0"):
                continue
            other = arms[a]["loop"].get(name[:-3] + ".r1")
            if not other:
                continue
            f0 = "".join(text_of(rec))
            f1 = "".join(text_of(other))
            if f0 == f1:
                same += 1
            else:
                diff += 1
        det[a] = {"same": same, "diff": diff}
        md.append(f"| {a} | {same} | {diff} |")
    md += ["", "Greedy decoding here is not bit-reproducible across requests (the streamed expert cache "
           "changes what is resident), so an arm-to-arm text difference alone does not implicate "
           "`STRATA_PF_FUSED`. Only a difference in the *loop rate* is evidence.", ""]

    # ---- the raw text of anything that looped, for eyeballing
    md += ["## Outputs that looped (first 600 characters)", ""]
    any_loop = False
    for row in detail:
        for a in ("1", "0"):
            if row[f"arm{a}"]["loop"]:
                any_loop = True
                rec = arms[a]["loop"][row["id"]]
                c, x = text_of(rec)
                md += [f"### {row['id']} - arm{a}", "```", (c or x)[:600].replace("```", "'''"), "```", ""]
    if not any_loop:
        md += ["None. No repetition loop in either arm.", ""]

    (HERE / "report.md").write_text("\n".join(md), encoding="utf-8")
    (HERE / "report.json").write_text(json.dumps(
        {"speeds": speeds, "loops": loop_count, "total": total, "detail": detail,
         "within_arm": det}, indent=2), encoding="utf-8")
    print("\n".join(md))
    return 0


def common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


if __name__ == "__main__":
    raise SystemExit(main())
