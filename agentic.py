#!/usr/bin/env python3
"""Tool-name margin probe for STRATA_PF_FUSED.

Why this exists: the loop/KL harness measures the *average* effect of the fused
prompt path.  The failure that matters in agent work is a single decision -- the
model writing `bash` or `ash` where the request declared `shell` -- and that is
a rare event (2 in 2,371 generations on the test card).  Counting it directly
would need thousands of requests per arm.

This probe measures the same decision where it is cheap: with thinking off, the
model writes the tool call as one line of JSON in the answer, so the name token
is a normal content token and the server's `logprobs` give its whole
distribution.  The number that matters is the margin:

    logprob(shell) - logprob(best wrong name)

A kernel perturbation of KL ~0.003 nats changes a logprob by ~0.003 nats.  If
`shell` wins by several nats in both arms, the flag cannot flip the decision
unless the two names were already tied to within ~0.05 nats -- and then the flip
would be common at temperature 1.0, not rare.

It also sends the same prompts through the real tool-call path (`tools` declared,
`tool_choice: auto`) to check that the JSON probe agrees with what the server
actually does, and runs `bash -n` on the produced command.

  python3 agentic.py --arm 1 --preset iq3_s-256k-final --out agentic/out
  python3 agentic.py --analyze agentic/out
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_PRESET = pathlib.Path.home() / ".config/strata/launch-presets/iq3_s-256k-final.json"

SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Run a shell command in the project directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to run"},
                "description": {"type": "string", "description": "What the command does"},
            },
            "required": ["command"],
        },
    },
}

JSON_TEXT = (
    "You are OpenCode, a coding agent.  You have exactly one tool, named `shell`.\n"
    "There is no `bash` tool and no `ash` tool.  To call the tool, write exactly one\n"
    "JSON object on one line:\n"
    '  {"name": "shell", "arguments": {"command": "<command>", "description": "<what it does>"}}\n'
    "Do not write anything else.\n"
)

# The real tool-call path must not be told to write JSON: that instruction turns
# the call into content and the request never reaches the tool parser.  This one
# only states the tool, as an agent's system prompt does.
TOOL_TEXT = (
    "You are OpenCode, a coding agent.  You have exactly one tool, named `shell`.\n"
    "There is no `bash` tool and no `ash` tool.  Call the tool to do the task.\n"
)

TASK = "List the files in the current directory, sorted by size, and show the five largest."

# Nontrivial but ordinary agent commands: quoting, pipes, find, sort, git.  The
# point is not the task but that the command is long enough for a syntax slip
# to show up under `bash -n`, which is the failure the user sees as the model
# correcting itself.
SYNTAX_TASKS = [
    "List every file under the current directory modified in the last 2 days, sorted by modification time.",
    "Find all Python files that import requests and print each one with its line count.",
    "Show the five largest files under the current directory, excluding .git.",
    "Count the lines in every Markdown file and print the total.",
    "Find duplicate lines in README.md and print them with their counts.",
    "Print the disk usage of each directory directly under the current directory, sorted.",
    "Show the 20 most recent git commits as one line each.",
    "Find files larger than 1 MB and print their sizes in human-readable form.",
]

# Contexts that make the wrong name tempting.  The model's prior for `bash` is
# strong (most harnesses use that name), so the real failure is a schema-vs-prior
# conflict; these probes make the conflict explicit and repeat it at long range.
PRIOR_BASH = (
    "Examples from other agents (for reference only; this request declares `shell`):\n"
    "  bash(command=\"ls -la\")\n"
    "  bash(command=\"rg TODO\")\n"
    "  The bash tool runs commands.  bash is the usual name for this tool.\n"
)

PRIOR_ASH = (
    "Note: some minimal Linux systems use the `ash` shell.  The tool declared by\n"
    "this request is still `shell`; ash is a shell binary, not a tool name.\n"
)


def long_context() -> str:
    """~45K tokens of code-like text, built from this repository and the prompts."""
    parts = []
    for p in sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.sh")) + sorted(HERE.glob("*.md")):
        try:
            parts.append(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    blob = "\n\n".join(parts)
    # ~4 chars per token; 180K chars is ~45K tokens.
    return (blob * (180_000 // max(1, len(blob)) + 1))[:180_000]


PROBES = [
    ("neutral", "", ""),
    ("prior_bash", PRIOR_BASH, ""),
    ("prior_ash", PRIOR_ASH, ""),
    ("long_code", "", long_context()),
    ("long_bash", PRIOR_BASH, long_context()),
]


def http_json(url: str, key: str, body: dict, timeout: float = 1800.0):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read().decode("utf-8"))


def load_key(explicit: str | None, preset: pathlib.Path) -> str:
    if explicit:
        return explicit
    if os.environ.get("STRATA_API_KEY"):
        return os.environ["STRATA_API_KEY"]
    try:
        return json.loads(preset.read_text())["api_key"]
    except Exception:  # noqa: BLE001
        return ""


def messages(prefix: str, context: str, mode: str = "json") -> list[dict]:
    system = (JSON_TEXT if mode == "json" else TOOL_TEXT) + ("\n" + prefix if prefix else "")
    user = (context + "\n\n" if context else "") + TASK
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def name_position(content: str, entries: list[dict]) -> tuple[int, str] | None:
    """Index of the logprob entry whose token starts the tool name, and its form.

    Two forms show up: the JSON the probe asks for (`"name": "shell"`) and the
    template's native call (`<function=shell>`), which the model prefers at long
    context.  Both name the tool in normal content tokens, so both give logprobs.
    """
    text = "".join(e.get("token", "") for e in entries)
    m = re.search(r'"name"\s*:\s*"', text)
    form = "json"
    if not m:
        m = re.search(r"<function=", text)
        form = "native"
    if not m:
        return None
    start = m.end()
    pos = 0
    for i, e in enumerate(entries):
        nxt = pos + len(e.get("token", ""))
        if pos <= start < nxt:
            return (i, form) if pos == start else None
        pos = nxt
    return None


def probe_json(base: str, key: str, model: str, prefix: str, context: str) -> dict:
    body = {
        "model": model,
        "messages": messages(prefix, context, "json"),
        "max_tokens": 200,
        "logprobs": True,
        "top_logprobs": 20,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }
    t0 = time.time()
    r = http_json(base + "/v1/chat/completions", key, body)
    ch = r["choices"][0]
    msg = ch.get("message") or {}
    entries = ((ch.get("logprobs") or {}).get("content") or [])
    found = name_position(msg.get("content") or "", entries)
    at = None
    form = None
    if found is not None:
        i, form = found
        e = entries[i]
        at = {
            "token": e.get("token"),
            "logprob": e.get("logprob"),
            "top": {t.get("token"): t.get("logprob") for t in (e.get("top_logprobs") or [])},
        }
    return {
        "kind": "json",
        "elapsed_s": round(time.time() - t0, 2),
        "content": msg.get("content"),
        "finish": ch.get("finish_reason"),
        "form": form,
        "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"),
        "name_at": at,
        "entries": entries,
    }


def bash_syntax(command: str) -> str:
    if not command:
        return "empty"
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(command)
        path = f.name
    try:
        p = subprocess.run(["bash", "-n", path], capture_output=True, text=True, timeout=10)
        return "ok" if p.returncode == 0 else ("syntax: " + (p.stderr.strip().splitlines() or ["?"])[-1])
    finally:
        os.unlink(path)


def probe_tool(base: str, key: str, model: str, prefix: str, context: str) -> dict:
    body = {
        "model": model,
        "messages": messages(prefix, context, "tool"),
        "tools": [SHELL_TOOL],
        "tool_choice": "auto",
        "max_tokens": 96,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }
    t0 = time.time()
    r = http_json(base + "/v1/chat/completions", key, body)
    ch = r["choices"][0]
    msg = ch.get("message") or {}
    calls = msg.get("tool_calls") or []
    name, args, command = None, None, None
    if calls:
        name = (calls[0].get("function") or {}).get("name")
        args = (calls[0].get("function") or {}).get("arguments")
        try:
            command = (json.loads(args) or {}).get("command")
        except Exception:
            command = None
    return {
        "kind": "tool",
        "elapsed_s": round(time.time() - t0, 2),
        "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"),
        "finish": ch.get("finish_reason"),
        "name": name,
        "arguments": args,
        "command": command,
        "bash_n": bash_syntax(command) if name == "shell" else None,
        "rejected": r.get("strata_rejected_tool_calls"),
        "content": (msg.get("content") or "")[:400],
    }


def probe_syntax(base: str, key: str, model: str, task: str) -> dict:
    """One real tool call for a nontrivial task; no sampling fields, like OpenCode."""
    body = {
        "model": model,
        "messages": [{"role": "system", "content": TOOL_TEXT},
                     {"role": "user", "content": task}],
        "tools": [SHELL_TOOL],
        "tool_choice": "auto",
        "max_tokens": 160,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }
    t0 = time.time()
    r = http_json(base + "/v1/chat/completions", key, body)
    ch = r["choices"][0]
    msg = ch.get("message") or {}
    calls = msg.get("tool_calls") or []
    name, command = None, None
    if calls:
        fn = calls[0].get("function") or {}
        name = fn.get("name")
        try:
            command = (json.loads(fn.get("arguments") or "{}") or {}).get("command")
        except Exception:
            command = None
    return {
        "kind": "syntax",
        "task": task,
        "elapsed_s": round(time.time() - t0, 2),
        "finish": ch.get("finish_reason"),
        "name": name,
        "command": command,
        "bash_n": bash_syntax(command) if name == "shell" else None,
        "rejected": r.get("strata_rejected_tool_calls"),
        "content": (msg.get("content") or "")[:400],
    }


def run_syntax(arm: str, base: str, key: str, out: pathlib.Path, n: int) -> None:
    model = served_model(base, key)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        task = SYNTAX_TASKS[i % len(SYNTAX_TASKS)]
        try:
            rec = probe_syntax(base, key, model, task)
        except Exception as e:  # noqa: BLE001
            rec = {"kind": "syntax", "task": task, "error": f"{type(e).__name__}: {e}"}
        rec["arm"] = arm
        rec["n"] = i
        (out / f"arm{arm}.syntax{i:02d}.json").write_text(json.dumps(rec, ensure_ascii=False))
        print(f"  syntax{i:02d} name={rec.get('name')} bash_n={rec.get('bash_n')} "
              f"cmd={str(rec.get('command'))[:70]!r}", flush=True)


def served_model(base: str, key: str) -> str:
    req = urllib.request.Request(base + "/v1/models", headers={"Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=30) as f:
        return json.loads(f.read().decode())["data"][0]["id"]


def run(arm: str, base: str, key: str, out: pathlib.Path, only: str, pick: str) -> None:
    model = served_model(base, key)
    out.mkdir(parents=True, exist_ok=True)
    for pid, prefix, context in PROBES:
        if pick != "all" and pick != pid:
            continue
        for kind in ("json", "tool"):
            if only != "all" and only != kind:
                continue
            try:
                rec = probe_json(base, key, model, prefix, context) if kind == "json" else \
                      probe_tool(base, key, model, prefix, context)
            except Exception as e:  # noqa: BLE001
                rec = {"kind": kind, "error": f"{type(e).__name__}: {e}"}
            rec["id"] = pid
            rec["arm"] = arm
            rec["context_chars"] = len(context)
            (out / f"arm{arm}.{pid}.{kind}.json").write_text(json.dumps(rec, ensure_ascii=False))
            print(f"  {pid:10s} {kind:4s} {rec.get('elapsed_s', '?')}s "
                  f"prompt={rec.get('prompt_tokens')} "
                  f"name={rec.get('name') or (rec.get('name_at') or {}).get('token')}", flush=True)


CAND = ("shell", "bash", "ash", "sh", "function", "Shell", "Bash", "ASH", "BASH")


def cand_probs(top: dict) -> dict:
    """The distribution over the candidate names, renormalised (the decision)."""
    d = {k: v for k, v in top.items() if k in CAND and v is not None}
    if not d:
        return {}
    m = max(d.values())
    z = sum(math.exp(v - m) for v in d.values())
    return {k: math.exp(v - m) / z for k, v in d.items()}


def kl_cand(a: dict, b: dict) -> float | None:
    pa, pb = cand_probs(a), cand_probs(b)
    keys = [k for k in pa if k in pb]
    if not keys:
        return None
    sa = sum(pa[k] for k in keys)
    sb = sum(pb[k] for k in keys)
    if sa <= 0 or sb <= 0:
        return None
    return sum(pa[k] / sa * math.log((pa[k] / sa) / (pb[k] / sb)) for k in keys)


def fmt(x, nd: int = 4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def analyze(out: pathlib.Path) -> int:
    ids = sorted({p.name.split(".")[1] for p in out.glob("arm*.json")})
    print("# Tool-name margin, STRATA_PF_FUSED=1 vs =0\n")
    print("`margin` = logprob(chosen name) − logprob(best wrong name). "
          "`KL` is over the candidate names only (shell, bash, ash, sh, function, …), "
          "renormalised: it is the flag's effect on the decision itself.\n")
    print("| probe | arm | name | form | logp shell | logp bash | logp ash | margin |")
    print("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |")
    recs = {}
    for pid in ids:
        for arm in ("1", "0"):
            p = out / f"arm{arm}.{pid}.json.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            at = r.get("name_at") or {}
            if not at and r.get("entries"):
                found = name_position(r.get("content") or "", r["entries"])
                if found is not None:
                    e = r["entries"][found[0]]
                    at = {"token": e.get("token"), "logprob": e.get("logprob"),
                          "top": {t.get("token"): t.get("logprob") for t in (e.get("top_logprobs") or [])}}
                    r["form"] = found[1]
            top = at.get("top") or {}
            name = at.get("token")
            wrong = max((v for k, v in top.items() if k in CAND and k != name), default=None)
            margin = (top.get(name) - wrong) if (wrong is not None and top.get(name) is not None) else None
            recs[(pid, arm)] = (top, name, margin)
            print(f"| {pid} | {arm} | {name} | {r.get('form') or '-'} | {fmt(top.get('shell'))} | "
                  f"{fmt(top.get('bash'))} | {fmt(top.get('ash'))} | {fmt(margin)} |")
    print()
    print("| probe | Δlogp shell | Δlogp bash | Δlogp ash | KL (candidates) |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for pid in ids:
        a = recs.get((pid, "1"))
        b = recs.get((pid, "0"))
        if not a or not b:
            continue
        d = lambda k: (a[0].get(k) - b[0].get(k)) if (a[0].get(k) is not None and b[0].get(k) is not None) else None
        print(f"| {pid} | {fmt(d('shell'), 4)} | {fmt(d('bash'), 4)} | {fmt(d('ash'), 4)} | "
              f"{fmt(kl_cand(a[0], b[0]), 5)} |")
    print()
    print("A margin of several nats in both arms and deltas of ~0.00x nats mean the flag "
          "cannot flip the name unless the two names were already tied to within ~0.05 nats — "
          "and then a flip would be common at temperature 1.0, not rare.\n")
    print("## Real tool-call path (`tools` declared, `tool_choice: auto`)\n")
    print("| probe | arm | name | bash -n | rejected |")
    print("| --- | ---: | --- | --- | --- |")
    for pid in ids:
        for arm in ("1", "0"):
            p = out / f"arm{arm}.{pid}.tool.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            print(f"| {pid} | {arm} | {r.get('name') or r.get('rejected') or '-'} | "
                  f"{r.get('bash_n') or '-'} | {json.dumps(r.get('rejected')) if r.get('rejected') else '-'} |")
    print()
    print("## Nontrivial commands (`bash -n`, no sampling fields)\n")
    print("| arm | calls | named shell | `bash -n` ok | syntax error | other name |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for arm in ("1", "0"):
        files = sorted(out.glob(f"arm{arm}.syntax*.json"))
        if not files:
            continue
        recs = [json.loads(p.read_text()) for p in files]
        ok = sum(1 for r in recs if r.get("bash_n") == "ok")
        bad = sum(1 for r in recs if r.get("bash_n") and r.get("bash_n") != "ok")
        named = sum(1 for r in recs if r.get("name") == "shell")
        other = sum(1 for r in recs if r.get("name") and r.get("name") != "shell")
        print(f"| {arm} | {len(recs)} | {named} | {ok} | {bad} | {other} |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm")
    ap.add_argument("--analyze", default=None)
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--key", default=None)
    ap.add_argument("--preset", default=os.environ.get("PRESET", DEFAULT_PRESET.stem))
    ap.add_argument("--out", default=str(HERE / "agentic" / "out"))
    ap.add_argument("--only", choices=("all", "json", "tool"), default="all")
    ap.add_argument("--probe", default="all", help="probe id (default: all)")
    ap.add_argument("--syntax", type=int, default=0,
                    help="run N real tool calls on nontrivial tasks and bash -n each command")
    a = ap.parse_args()
    if a.analyze:
        return analyze(pathlib.Path(a.analyze))
    if not a.arm:
        ap.error("--arm is required unless --analyze is used")
    key = load_key(a.key, pathlib.Path.home() / ".config/strata/launch-presets" / f"{a.preset}.json")
    if a.syntax:
        run_syntax(a.arm, a.base, key, pathlib.Path(a.out), a.syntax)
    else:
        run(a.arm, a.base, key, pathlib.Path(a.out), a.only, a.probe)
    return 0


if __name__ == "__main__":
    sys.exit(main())
