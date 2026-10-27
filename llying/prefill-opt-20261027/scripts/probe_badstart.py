#!/usr/bin/env python3
"""Count probe outputs whose reasoning is corrupted right after the first token.

The probe's `garbled` flag looks at repetition / think tags and misses the short-prompt
failure where the reasoning starts "The" + junk ("Therenzcaff1:", "The9每小时...") but the
final answer is still right.

usage: probe_badstart.py PROBE.jsonl... [--show N]
"""
import argparse
import json
import os
import re

CJK = re.compile(r"[\u4e00-\u9fff]")


def bad_start(r):
    s = r.get("reasoning") or ""
    return bool(re.match(r"The[^ \n's]", s)) or bool(CJK.search(s[:60])) or len(s.strip()) < 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--show", type=int, default=3)
    a = ap.parse_args()
    for f in a.files:
        rows = [json.loads(line) for line in open(f)]
        bad = [r for r in rows if bad_start(r)]
        name = os.path.basename(f).removesuffix(".probe.jsonl")
        print(f"{name:45s} n={len(rows)} bad_start={len(bad):3d} "
              f"garbled={sum(r['garbled'] for r in rows):2d} correct={sum(r['correct'] for r in rows)}"
              + (" | " + " ; ".join(repr(r["reasoning"][:25]) for r in bad[:a.show]) if a.show else ""))


if __name__ == "__main__":
    main()
