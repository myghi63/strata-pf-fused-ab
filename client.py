#!/usr/bin/env python3
"""Send the fixed prompt set to a running Strata server and save every reply.

The server must already be up (run_all.sh starts it). One arm per server start:
the arm is just the value of STRATA_PF_FUSED the engine was started with.

  python3 client.py --arm 1
  python3 client.py --arm 0

Writes out/arm<arm>/kl/<id>.json and out/arm<arm>/loop/<id>.r<k>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prompts import KL, LOOP  # noqa: E402

DEFAULT_PRESET = pathlib.Path.home() / ".config/strata/launch-presets/iq3_s-256k-final.json"

def http_json(method: str, url: str, key: str | None, body: dict | None = None, timeout: float = 3600.0):
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


def ask(base: str, key: str, model: str, text: str, max_tokens: int, logprobs: bool) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 5
    t0 = time.time()
    resp = post_chat(base, key, body)
    return {"elapsed_s": round(time.time() - t0, 3), "request": body, "response": resp}


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--key", default=None)
    ap.add_argument("--preset", default=os.environ.get("PRESET", DEFAULT_PRESET.stem),
                    help="launch preset whose api_key is read when --key/STRATA_API_KEY are absent "
                         "(default: %(default)s)")
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--only", choices=("all", "kl", "loop"), default="all")
    ap.add_argument("--dumps", default=None,
                    help="dump prefix (e.g. dumps/fused1): the count of files before each KL request is "
                         "recorded as that request's dump_index, so kl.py lines the dumps up exactly "
                         "(the engine also dumps the server's own warm-up request)")
    a = ap.parse_args()

    key = load_key(a.key, pathlib.Path.home() / ".config/strata/launch-presets" / f"{a.preset}.json")
    model = wait_ready(a.base, key)
    out = pathlib.Path(a.out) / f"arm{a.arm}"
    print(f"arm {a.arm}: model {model}, out {out}", flush=True)

    if a.only in ("all", "kl"):
        # First, so the STRATA_DUMP_FIRST_LOGITS files (.0 .1 ...) line up with
        # this list, one request per prompt and no repeats.
        for i, p in enumerate(KL):
            print(f"  KL   [{i + 1}/{len(KL)}] {p['id']}", flush=True)
            n_before = dump_count(a.dumps)
            rec = ask(a.base, key, model, p["text"], int(p.get("max_tokens", 4)), logprobs=False)
            rec["id"] = p["id"]
            rec["index"] = i
            if n_before is not None:
                rec["dump_index"] = n_before
            save(out / "kl" / f"{p['id']}.json", rec)

    if a.only in ("all", "loop"):
        for r in range(a.repeats):
            for p in LOOP:
                print(f"  LOOP r{r} {p['id']}", flush=True)
                rec = ask(a.base, key, model, p["text"], int(p.get("max_tokens", 600)), logprobs=True)
                rec["id"] = p["id"]
                rec["repeat"] = r
                save(out / "loop" / f"{p['id']}.r{r}.json", rec)

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
