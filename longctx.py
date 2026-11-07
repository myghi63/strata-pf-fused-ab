#!/usr/bin/env python3
"""Long-context test: ~187K tokens of Moby-Dick (Project Gutenberg #2701, public domain) with twelve
inserted "ship's log notes", then a task that needs the whole text and a long answer.

It checks two things at once:
  * the input: the notes must come back with the right code and chapter, and every chapter must be
    summarized once, in order (a corrupted or "looping" prefill shows up as missing / wrong notes or
    skipped / repeated chapters);
  * the output: the reasoning and the answer go through the same loop detector as the short prompts.

The book is downloaded once into data/ by run_all.sh (or `python3 longctx.py fetch`).
"""
from __future__ import annotations

import pathlib
import random
import re
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
BOOK = HERE / "data" / "pg2701.txt"
BOOK_URL = "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"
CHARS = 740_000          # ~3.89 chars/token on this tokenizer -> ~187K prompt tokens (fits 256K with 60K to answer)
MAX_TOKENS = 60_000
WORDS = ["amber", "falcon", "cobalt", "willow", "granite", "saffron",
         "harbor", "quartz", "ember", "juniper", "marlin", "tundra"]


def fetch() -> pathlib.Path:
    if BOOK.exists() and BOOK.stat().st_size > 1_000_000:
        return BOOK
    BOOK.parent.mkdir(parents=True, exist_ok=True)
    tmp = BOOK.with_suffix(".part")
    with urllib.request.urlopen(BOOK_URL, timeout=120) as r:
        tmp.write_bytes(r.read())
    tmp.replace(BOOK)
    return BOOK


def build() -> dict:
    """The prompt text, the number of chapters in it and the twelve (note, code, chapter) answers."""
    t = BOOK.read_text(encoding="utf-8")
    first = t.index("CHAPTER 1. Loomings.")
    s = t.index("CHAPTER 1. Loomings.", first + 10)          # skip the table of contents
    body = t[s:s + CHARS]
    body = body[:body.rindex("\nCHAPTER ")]                   # end on a chapter boundary
    heads = [m.start() for m in re.finditer(r"\nCHAPTER \d+\.", body)]
    nch = len(heads) + 1
    rnd = random.Random(7)
    needles = []
    for k in range(12):
        pos = body.index("\n\n", int(len(body) * (k + 0.5) / 12)) + 2
        ch = 1 + sum(1 for h in heads if h < pos)
        needles.append((pos, k + 1, f"{WORDS[k]}-{rnd.randint(1000, 9999)}", ch))
    for pos, n, code, _ in sorted(needles, reverse=True):
        body = body[:pos] + f"[Ship's log note {n}: the cargo code for crate {n} is {code}.]\n\n" + body[pos:]
    q = (f"Above is the beginning of Moby-Dick, chapters 1 to {nch}. Twelve \"Ship's log notes\" were inserted "
         "into the text.\n"
         "Task 1: list all twelve notes in order: note number, crate code, and the chapter number where it appears.\n"
         f"Task 2: for EVERY chapter from 1 to {nch}, in order, write one sentence summarizing what happens in it. "
         "Do not skip or merge chapters.\n"
         "Task 3: finally, list the five longest chapters you saw and explain how you judged their length.")
    return {"text": body + "\n\n---\n\n" + q, "chapters": nch,
            "needles": [{"note": n, "code": code, "chapter": ch} for _, n, code, ch in sorted(needles)]}


def score(answer: str, meta: dict) -> dict:
    """Needle recall (code and chapter) and chapter coverage of Task 2."""
    notes = []
    for nd in meta["needles"]:
        m = re.search(rf"\b{nd['note']}\b[^\n]*?{re.escape(nd['code'])}[^\n]*", answer)
        notes.append({**nd, "found": bool(m),
                      "chapter_ok": bool(m and re.search(rf"\b{nd['chapter']}\b", m.group(0)))})
    # Task 2 is the block between the "Task 2" and "Task 3" headings (when the model writes them)
    part = answer
    i2 = answer.find("Task 2")
    if i2 >= 0:
        i3 = answer.find("Task 3", i2)
        part = answer[i2:i3 if i3 > 0 else None]
    nums = [int(x) for x in re.findall(r"(?m)^\W*(?:Chapter\s+)?(\d{1,3})\b", part)]
    nums = [x for x in nums if 1 <= x <= meta["chapters"]]
    seen: dict[int, int] = {}
    for x in nums:
        seen[x] = seen.get(x, 0) + 1
    return {"notes_found": sum(n["found"] for n in notes),
            "notes_chapter_ok": sum(n["chapter_ok"] for n in notes),
            "notes": notes,
            "chapters_missing": [i for i in range(1, meta["chapters"] + 1) if i not in seen],
            "chapters_repeated": sorted(k for k, v in seen.items() if v > 1),
            "chapters_in_order": nums == sorted(nums)}


if __name__ == "__main__":
    if sys.argv[1:] == ["fetch"]:
        print(fetch())
    else:
        fetch()
        b = build()
        print(len(b["text"]), "chars,", b["chapters"], "chapters")
        for n in b["needles"]:
            print(n)
