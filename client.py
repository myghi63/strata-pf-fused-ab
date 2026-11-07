#!/usr/bin/env python3
"""Send the fixed prompt set to a running Strata server and save every reply.

The server must already be up (run_all.sh starts it). One arm per server start:
the arm is just the value of STRATA_PF_FUSED the engine was started with.

  python3 client.py --arm 1 --server-log <server.log> --dumps dumps/fused1
  python3 client.py --arm 0 --server-log <server.log> --dumps dumps/fused0

Phases (in this order):
  kl     6 long prompts, max_tokens=4, for the first-token logits dumps
  probe  the same prompt three times, greedy, to measure run-to-run repeatability
  loop   14 repetition-inviting prompts, greedy and sampled, up to --loop-max-tokens
  long   one ~187K-token prompt (longctx.py), sampled, up to 60K tokens
  rep    (opt-in) one loop prompt N times with the server's default sampling, for a loop rate

Every request records the server-log lines written while it ran (the server's loop
detector, its single-token repeat guard, budget wrap-ups), so a loop the server cut
short is attributed to the request that caused it.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prompts import KL, LOOP  # noqa: E402
import longctx  # noqa: E402

DEFAULT_PRESET = pathlib.Path.home() / ".config/strata/launch-presets/iq3_s-256k-final.json"
# Qwen's recommended sampling for thinking mode
SAMPLED = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0}
GREEDY = {"temperature": 0, "top_p": 1}
# server.py messages worth attributing to a request
EVENT_RE = re.compile(r"repeated reasoning detected|the thinking repeated the same passages|"
                      r"repeated one token|reasoning_close_retry|thinking budget|reasoning_loop_recovery")


def http_json(method: str, url: str, key: str | None, body: dict | None = None, timeout: float = 14400.0):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read().decode("utf-8"))


def wait_ready(base: str, key: str, timeout_s: float = 900.0) -> str:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout_s:
        try:
            m = http_json("GET", base + "/v1/models", key, timeout=10)
            ids = [x.get("id") for x in (m.get("data") or []) if isinstance(x, dict) and x.get("id")]
            if ids:
                return ids[0]
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise SystemExit(
                    f"{base} refused the API key (HTTP {e.code}). Set STRATA_API_KEY or fix the preset's api_key."
                ) from None
            last = f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001 - the server is still loading
            last = f"{type(e).__name__}: {e}"
        time.sleep(2)
    raise SystemExit(f"server did not become ready at {base} ({last})")


def post_chat(base: str, key: str, body: dict, tries: int = 6):
    for attempt in range(1, tries + 1):
        try:
            return http_json("POST", base + "/v1/chat/completions", key, body)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            if e.code in (429, 500, 502, 503, 504) and attempt < tries:
                print(f"    HTTP {e.code}, retry {attempt}/{tries - 1}: {detail}", flush=True)
                time.sleep(5 * attempt)
                continue
            raise SystemExit(f"HTTP {e.code}: {detail}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < tries:
                print(f"    {type(e).__name__}: {e}; retry {attempt}/{tries - 1}", flush=True)
                time.sleep(5 * attempt)
                continue
            raise
    raise AssertionError("unreachable")


def load_key(explicit: str | None, preset: pathlib.Path) -> str:
    if explicit:
        return explicit
    if os.environ.get("STRATA_API_KEY"):
        return os.environ["STRATA_API_KEY"]
    try:
        return json.loads(preset.read_text())["api_key"]
    except Exception:  # noqa: BLE001
        return ""


class ServerLog:
    """Reads the lines the server log gained since the last call."""

    def __init__(self, path: str | None):
        self.path = pathlib.Path(path) if path else None
        self.off = self.path.stat().st_size if self.path and self.path.exists() else 0

    def new_events(self) -> list[str]:
        if not self.path or not self.path.exists():
            return []
        with self.path.open("rb") as f:
            f.seek(self.off)
            data = f.read()
        self.off += len(data)
        return [ln for ln in data.decode("utf-8", "replace").splitlines() if EVENT_RE.search(ln)]


def ask(base: str, key: str, model: str, text: str, max_tokens: int, sampling: dict, slog: ServerLog) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens, "stream": False, **sampling}
    slog.new_events()                       # drop anything logged before this request
    t0 = time.time()
    resp = post_chat(base, key, body)
    time.sleep(0.5)                         # the server prints its verdict as the reply ends
    req = {k: v for k, v in body.items() if k != "messages"}
    req["prompt_chars"] = len(text)
    return {"elapsed_s": round(time.time() - t0, 3), "request": req, "response": resp,
            "server_events": slog.new_events()}


def save(path: pathlib.Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def dump_count(prefix: str | None) -> int | None:
    if not prefix:
        return None
    p = pathlib.Path(prefix)
    d, name = p.parent, p.name
    return len(list(d.glob(name + ".*"))) if d.exists() else 0


def brief(rec: dict) -> str:
    r = rec["response"]
    ch = (r.get("choices") or [{}])[0]
    u = r.get("usage") or {}
    ev = f" server:{len(rec['server_events'])}" if rec["server_events"] else ""
    return f"finish={ch.get('finish_reason')} tokens={u.get('completion_tokens')} {rec['elapsed_s']:.0f}s{ev}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--key", default=None)
    ap.add_argument("--preset", default=os.environ.get("PRESET", DEFAULT_PRESET.stem),
                    help="launch preset whose api_key is read when --key/STRATA_API_KEY are absent "
                         "(default: %(default)s)")
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--phases", default="kl,probe,loop,long")
    ap.add_argument("--loop-max-tokens", type=int, default=6144,
                    help="the server's loop detector needs ~2,000 words of reasoning before it can fire")
    ap.add_argument("--greedy-repeats", type=int, default=1)
    ap.add_argument("--sampled-seeds", default="1", help="comma-separated seeds for the sampled loop pass")
    ap.add_argument("--rep-prompt", default="loop_words_a", help="the loop prompt the 'rep' phase repeats")
    ap.add_argument("--rep-n", type=int, default=8, help="how many times the 'rep' phase sends it")
    ap.add_argument("--server-log", default=None, help="the server log of this launch (to attribute loop events)")
    ap.add_argument("--dumps", default=None,
                    help="dump prefix (e.g. dumps/fused1): the count of files before each KL request is "
                         "recorded as that request's dump_index, so kl.py lines the dumps up exactly "
                         "(the engine also dumps the server's own warm-up request)")
    a = ap.parse_args()
    phases = set(a.phases.split(","))

    key = load_key(a.key, pathlib.Path.home() / ".config/strata/launch-presets" / f"{a.preset}.json")
    model = wait_ready(a.base, key)
    slog = ServerLog(a.server_log)
    out = pathlib.Path(a.out) / f"arm{a.arm}"
    print(f"arm {a.arm}: model {model}, out {out}", flush=True)

    if "kl" in phases:
        # First, so the STRATA_DUMP_FIRST_LOGITS files line up with this list.
        for i, p in enumerate(KL):
            n_before = dump_count(a.dumps)
            rec = ask(a.base, key, model, p["text"], int(p.get("max_tokens", 4)), GREEDY, slog)
            rec.update(id=p["id"], index=i)
            if n_before is not None:
                rec["dump_index"] = n_before
            save(out / "kl" / f"{p['id']}.json", rec)
            print(f"  KL    [{i + 1}/{len(KL)}] {p['id']}", flush=True)

    if "probe" in phases:
        # Same prompt, greedy, three times in a row: are greedy replies reproducible on this build?
        p = LOOP[0]
        for k in range(3):
            rec = ask(a.base, key, model, p["text"], 400, GREEDY, slog)
            rec.update(id=p["id"], repeat=k)
            save(out / "probe" / f"{p['id']}.p{k}.json", rec)
            print(f"  PROBE {k} {p['id']} {brief(rec)}", flush=True)

    if "loop" in phases:
        passes = [("greedy", r, GREEDY) for r in range(a.greedy_repeats)]
        passes += [("sampled", int(s), {**SAMPLED, "seed": int(s)}) for s in a.sampled_seeds.split(",") if s]
        for mode, k, sampling in passes:
            for p in LOOP:
                rec = ask(a.base, key, model, p["text"], a.loop_max_tokens, sampling, slog)
                rec.update(id=p["id"], mode=mode, repeat=k)
                save(out / "loop" / f"{p['id']}.{mode}{k}.json", rec)
                print(f"  LOOP  {mode}{k} {p['id']} {brief(rec)}", flush=True)

    if "rep" in phases:
        # One prompt N times with NO sampling fields: the server applies its own default (the model card's
        # sampling, temperature 1.0 / top_p 0.95 / top_k 20 for Qwen3.8-Flash-Next with thinking), exactly
        # what a client like opencode that sends no sampling gets.  Measures the loop RATE per arm.
        p = next(x for x in LOOP if x["id"] == a.rep_prompt)
        for k in range(a.rep_n):
            rec = ask(a.base, key, model, p["text"], a.loop_max_tokens, {}, slog)
            rec.update(id=p["id"], mode="serverdefault", repeat=k)
            save(out / "rep" / f"{p['id']}.r{k}.json", rec)
            print(f"  REP   {k} {p['id']} {brief(rec)}", flush=True)

    if "long" in phases:
        longctx.fetch()
        b = longctx.build()
        rec = ask(a.base, key, model, b["text"], longctx.MAX_TOKENS, {**SAMPLED, "seed": 1}, slog)
        rec.update(id="long_mobydick", meta={"chapters": b["chapters"], "needles": b["needles"]})
        save(out / "long" / "long_mobydick.json", rec)
        print(f"  LONG  {brief(rec)}", flush=True)

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
