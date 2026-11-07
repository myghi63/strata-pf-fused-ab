#!/usr/bin/env python3
"""Analyse the two arms: repetition loops, run-to-run repeatability, long context and prompt speed.

  python3 report.py            # reads out/arm1 and out/arm0, writes report.md/report.json

A "loop" is degenerate repetition: the model repeats a token, a line, a phrase or a whole
block and does not leave it. A generation counts as looped if ANY of these fires:
  * the server's own detectors cut it (reasoning_loop_recovery "stop": "the thinking repeated
    the same passages"; repeat_stop_tokens: "repeated one token"), from the server-log lines
    client.py attributed to that request;
  * the reasoning or the answer has an exactly periodic tail (period <= 300 characters, 3 repeats);
  * one line repeats 8+ times, a 10-word phrase 6+ times or a 6-word phrase 12+ times;
  * the text compresses absurdly well (zlib ratio < 0.16 on 1,200+ characters).
Reasoning and answer are checked separately. `finish == "length"` alone is not a loop.
"""
from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys
import zlib
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# `python3 report.py [DIR]`: DIR holds out/ and arm*.server.log (default: this folder); report.md/json go there
BASE = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE
OUT = BASE / "out"
import longctx  # noqa: E402

SERVER_LOOP_RE = re.compile(r"the thinking repeated the same passages|repeated reasoning detected|repeated one token")


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


def worst_zlib(text: str, win: int = 4000, step: int = 2000) -> float:
    """Lowest compression ratio over 4 KB windows (a loop in the middle of a long text shows here)."""
    if len(text) <= win:
        return len(zlib.compress(text.encode("utf-8"), 9)) / max(1, len(text.encode("utf-8")))
    best = 1.0
    for i in range(0, len(text) - win + 1, step):
        seg = text[i:i + win].encode("utf-8")
        best = min(best, len(zlib.compress(seg, 9)) / len(seg))
    return best


def metrics(text: str) -> dict:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lc = Counter(lines)
    top_line, top_line_n = lc.most_common(1)[0] if lc else ("", 0)
    words = text.split()
    ratio = worst_zlib(text) if text else 1.0
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
    return {"chars": len(text), "words": len(words), "zlib_min": round(ratio, 4), "tail_period": tp,
            "top_line_n": top_line_n, "top_line": top_line[:80], "ngram6": n6, "ngram10": n10,
            "triggers": triggers}


def parts(rec: dict) -> tuple[str, str]:
    msg = ((rec["response"].get("choices") or [{}])[0].get("message")) or {}
    return msg.get("content") or "", msg.get("reasoning_content") or msg.get("reasoning") or ""


def finish_of(rec: dict) -> str:
    return ((rec["response"].get("choices") or [{}])[0].get("finish_reason")) or "?"


def tokens_of(rec: dict) -> int:
    return int((rec["response"].get("usage") or {}).get("completion_tokens") or 0)


def judge(rec: dict) -> dict:
    content, reasoning = parts(rec)
    mr, mc = metrics(reasoning), metrics(content)
    server = [ln for ln in rec.get("server_events", []) if SERVER_LOOP_RE.search(ln)]
    signals = ["server: " + re.sub(r"^.*?\[strata\] ", "", s)[:70] for s in server]
    signals += [f"reasoning {t}" for t in mr["triggers"]] + [f"answer {t}" for t in mc["triggers"]]
    return {"finish": finish_of(rec), "tokens": tokens_of(rec), "answered": bool(content.strip()),
            "reasoning": mr, "answer": mc, "server_events": rec.get("server_events", []),
            "signals": signals, "loop": bool(signals), "server_cut": bool(server),
            "heuristic": bool(mr["triggers"] or mc["triggers"])}


def load(arm: str, phase: str) -> dict:
    d = OUT / f"arm{arm}" / phase
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))} if d.exists() else {}


def common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def server_totals(arm: str) -> dict:
    p = BASE / f"arm{arm}.server.log"
    txt = p.read_text(errors="replace") if p.exists() else ""
    return {"loop_stop": txt.count("the thinking repeated the same passages"),
            "loop_recover": txt.count("repeated reasoning detected"),
            "token_repeat": txt.count("repeated one token"),
            "recovery_mode": (re.findall(r'reasoning_loop_recovery[^\n]*', txt) or [""])[0][:120],
            "log_found": p.exists()}


def main() -> int:
    if not (OUT / "arm1").exists() or not (OUT / "arm0").exists():
        raise SystemExit(f"missing {OUT}/arm1 or arm0; run run_all.sh first")
    md = ["# STRATA_PF_FUSED A/B", ""]
    js: dict = {}

    # ---- prompt speed (KL requests: ~3.85K tokens, max_tokens=4)
    md += ["## Prompt speed (the KL requests)", "",
           "| arm | prompts | prompt tokens | mean tok/s | median tok/s | mean prompt ms |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    speeds = {}
    for a in ("1", "0"):
        rows = []
        for name, rec in load(a, "kl").items():
            t = rec["response"].get("timings") or {}
            if t.get("prompt_n") and t.get("prompt_ms"):
                rows.append({"id": name, "tokens": t["prompt_n"], "cache_n": t.get("cache_n"),
                             "prompt_ms": t["prompt_ms"],
                             "tok_s": t.get("prompt_per_second") or t["prompt_n"] / (t["prompt_ms"] / 1000)})
        speeds[a] = rows
        if rows:
            v = [r["tok_s"] for r in rows]
            md.append(f"| {a} | {len(rows)} | {min(r['tokens'] for r in rows)}-{max(r['tokens'] for r in rows)} | "
                      f"{statistics.mean(v):.1f} | {statistics.median(v):.1f} | "
                      f"{statistics.mean(r['prompt_ms'] for r in rows):.1f} |")
    if speeds.get("1") and speeds.get("0"):
        ratio = statistics.mean(r["tok_s"] for r in speeds["1"]) / statistics.mean(r["tok_s"] for r in speeds["0"])
        md += ["", f"fused on/off prompt speed ratio: **{ratio:.3f}x**"]
        js["speed_ratio"] = ratio
    md.append("")
    js["speeds"] = speeds

    # ---- server detectors
    tot = {a: server_totals(a) for a in ("1", "0")}
    js["server_totals"] = tot
    md += ["## Server loop detectors (whole server log of each arm)", "",
           "| arm | reasoning loop, stop | reasoning loop, recover | one token repeated |",
           "| --- | ---: | ---: | ---: |"]
    for a in ("1", "0"):
        t = tot[a]
        md.append(f"| {a} | {t['loop_stop']} | {t['loop_recover']} | {t['token_repeat']} |"
                  if t["log_found"] else f"| {a} | (no server log) | | |")
    md.append("")

    # ---- loops
    js["loop"] = {}
    md += ["## Repetition loops (short prompts)", ""]
    loops = {"1": load("1", "loop"), "0": load("0", "loop")}
    names = sorted(set(loops["1"]) | set(loops["0"]))
    modes = sorted({n.rsplit(".", 1)[1] for n in names})
    summary = []
    for mode in modes:
        cnt = {"1": [0, 0, 0, 0, 0], "0": [0, 0, 0, 0, 0]}    # server cuts, total, hit length, answered, heuristic
        md += [f"### {mode}", "",
               "| prompt | arm1 finish | arm1 tokens | arm1 signals | arm0 finish | arm0 tokens | arm0 signals |",
               "| --- | --- | ---: | --- | --- | ---: | --- |"]
        for name in [n for n in names if n.endswith("." + mode)]:
            row = {}
            for a in ("1", "0"):
                rec = loops[a].get(name)
                if rec:
                    j = judge(rec)
                    row[a] = j
                    c = cnt[a]
                    c[0] += j["server_cut"]; c[1] += 1; c[2] += j["finish"] == "length"; c[3] += j["answered"]
                    c[4] += j["heuristic"]
            js["loop"][name] = row
            cell = lambda a: (f"{row[a]['finish']} | {row[a]['tokens']} | {'; '.join(row[a]['signals']) or '-'}"
                              if a in row else "- | - | -")
            md.append(f"| {name.rsplit('.', 1)[0]} | {cell('1')} | {cell('0')} |")
        md.append("")
        for a in ("1", "0"):
            l, n, ln, ans, h = cnt[a]
            summary.append(f"| {mode} | {a} | **{l}/{n}** | {h}/{n} | {ln}/{n} | {ans}/{n} |")
    md += ["### Summary", "",
           "| mode | arm | cut by the server's loop detector | text heuristics fired | hit max_tokens | wrote an answer |",
           "| --- | --- | ---: | ---: | ---: | ---: |"] + summary + ["",
           "The server's detector is the primary signal. The text heuristics are deliberately sensitive and also fire on "
           "tasks whose correct output is repetitive (a 50-item JSON list, a numbered poem, a Fibonacci table); read "
           "the flagged texts below before calling one a loop.", ""]

    # ---- repeatability probe
    md += ["## Greedy repeatability (same prompt, same server, 3 requests)", "",
           "| arm | identical pairs | common prefix (chars) 0-1 / 1-2 | draft accepted / drafted |",
           "| --- | ---: | --- | --- |"]
    js["probe"] = {}
    for a in ("1", "0"):
        recs = [r for _, r in sorted(load(a, "probe").items())]
        if len(recs) < 2:
            continue
        txt = ["".join(reversed(parts(r))) for r in recs]
        same = sum(txt[i] == txt[i + 1] for i in range(len(txt) - 1))
        pre = [common_prefix(txt[i], txt[i + 1]) for i in range(len(txt) - 1)]
        dr = [f"{(r['response'].get('timings') or {}).get('draft_n_accepted')}/"
              f"{(r['response'].get('timings') or {}).get('draft_n')}" for r in recs]
        cache = [(r["response"].get("timings") or {}).get("cache_n") for r in recs]
        js["probe"][a] = {"identical_pairs": same, "prefix": pre, "draft": dr, "cache_n": cache}
        md.append(f"| {a} | {same}/{len(txt) - 1} | {' / '.join(map(str, pre))} | {', '.join(dr)} |")
    md += ["", "The 2nd and 3rd requests reuse the cached prompt (conversation cache), so a difference between "
           "them comes from decoding, not from the prompt path. The draft acceptance counts differ between "
           "requests; speculative decoding is the likely source, but this run does not isolate it "
           "(`--drafting off` would).", ""]

    # ---- rep: one prompt N times with the server's default sampling
    reps = {a: load(a, "rep") for a in ("1", "0")}
    if any(reps.values()):
        md += ["## Loop rate, one prompt repeated with the server's default sampling", "",
               "No sampling fields in the request: the server applies the model card's (temperature 1.0, top_p 0.95, "
               "top_k 20 with thinking), as for a client such as opencode. `coverage` is the server detector's "
               "share of the recent reasoning inside passages seen three times (it cuts at 0.25).", "",
               "| arm | runs | cut by the server | coverage of the cuts | other loop signals | wrote an answer |",
               "| --- | ---: | ---: | --- | ---: | ---: |"]
        js["rep"] = {}
        for a in ("1", "0"):
            rows = [judge(r) for _, r in sorted(reps[a].items())]
            if not rows:
                continue
            cov = [float(m.group(1)) for j in rows for ev in j["server_events"]
                   for m in [re.search(r"coverage=([0-9.]+)", ev)] if m]
            cut = sum(any(SERVER_LOOP_RE.search(ev) for ev in j["server_events"]) for j in rows)
            other = sum(bool(j["reasoning"]["triggers"] or j["answer"]["triggers"]) for j in rows)
            js["rep"][a] = {"runs": rows, "cut": cut, "coverage": cov}
            md.append(f"| {a} | {len(rows)} | {cut} | {', '.join(f'{c:.3f}' for c in cov) or '-'} | {other} | "
                      f"{sum(j['answered'] for j in rows)}/{len(rows)} |")
        md.append("")

    # ---- long context
    md += ["## Long context (~187K-token prompt, sampled, up to 60K tokens)", "",
           "| arm | prompt tokens | prompt tok/s | output tokens | decode tok/s | finish | notes found / right chapter | "
           "chapters missing / repeated | loop signals |",
           "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |"]
    js["long"] = {}
    for a in ("1", "0"):
        rec = load(a, "long").get("long_mobydick")
        if not rec:
            continue
        j = judge(rec)
        content, _ = parts(rec)
        s = longctx.score(content, rec["meta"])
        t = rec["response"].get("timings") or {}
        js["long"][a] = {**j, "score": s, "timings": t}
        md.append(f"| {a} | {t.get('prompt_n')} | {t.get('prompt_per_second', 0):.0f} | {j['tokens']} | "
                  f"{t.get('predicted_per_second', 0):.1f} | {j['finish']} | {s['notes_found']}/12, "
                  f"{s['notes_chapter_ok']}/12 | {len(s['chapters_missing'])} / {len(s['chapters_repeated'])} | "
                  f"{'; '.join(j['signals']) or '-'} |")
    md.append("")

    # ---- the text of anything that looped
    md += ["## Outputs flagged as loops (last 500 characters)", ""]
    flagged = False
    for name, row in js["loop"].items():
        for a, j in row.items():
            if j["loop"]:
                flagged = True
                c, r = parts(loops[a][name])
                md += [f"### {name} - arm{a}", "", "Signals: " + "; ".join(j["signals"]), "", "```",
                       (r if not c.strip() else c)[-500:].replace("```", "'''"), "```", ""]
    for a, j in js["long"].items():
        if j["loop"]:
            flagged = True
            md += [f"### long_mobydick - arm{a}", "", "Signals: " + "; ".join(j["signals"]), ""]
    if not flagged:
        md += ["None.", ""]

    (BASE / "report.md").write_text("\n".join(md), encoding="utf-8")
    (BASE / "report.json").write_text(json.dumps(js, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
