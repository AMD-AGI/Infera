#!/usr/bin/env python3
"""T49's detector: a `PROBLEM:` line under a `passed` heading.

**T49 is the one class filed on 2026-09-04 that had no detector.** Seven
instances, four owners, one day, and *not one found by its author* — the author
reads the message against the code they just wrote and it always parses. Every
other entry that day either carries a check or says plainly that nothing detects
it; this one can be checked, and m5 is who noticed:

    the leader's instance is the only one of the seven found by a MACHINE rather
    than a person -- `checkpoint` counting `PROBLEM:` lines for an unrelated
    reason. Counting them under `passed` headings is a one-liner that would have
    caught it the day it landed.

**Why this shape and not a grep.** `PROBLEM:` is `write_report`'s marker for a
finding that *refuses*, so under a `passed` heading it is a contradiction in the
helper's own vocabulary. A bare `grep -c PROBLEM:` over a run finds eleven and
mis-keys — four are `check_no_regression`'s under `REFUSED` and are correct.
Only the pairing is the signal. (And `grep -c '^PROBLEM:'` finds **zero**: the
lines are indented two spaces. `checkpoint` was one step from recording that
the four expected reasons were absent from a run that carried them.)

**What it does NOT catch.** T49 is *right verdict, wrong explanation*, and this
finds one costume of it — the one where the wrongness is structural enough to
render. The other six are English sentences that are true of a broader condition
than the code tests, and no program finds those. This is a floor, not a sweep.

    python3 problem_under_passed.py <run-dir> [<run-dir> ...]

Exit 0 clean, 1 found, 2 nothing to read -- *cannot judge* kept apart from
*judged and clean*, which is the distinction the validators themselves draw.
"""
from __future__ import annotations

import collections
import os
import re
import sys

_NAME = re.compile(r"^# (\S+)")
_HEAD = re.compile(r"^## \S+: (\w[\w ]*)")


def scan(run: str) -> tuple[int, collections.Counter]:
    hits: collections.Counter = collections.Counter()
    seen = 0
    for root, _dirs, files in os.walk(run):
        if "validator_report.txt" not in files:
            continue
        seen += 1
        name = status = None
        path = os.path.join(root, "validator_report.txt")
        try:
            lines = open(path, errors="replace").read().splitlines()
        except OSError:
            continue
        for line in lines:
            m = _NAME.match(line)
            if m:
                name, status = m.group(1), None
                continue
            m = _HEAD.match(line)
            if m:
                status = m.group(1).strip()
                continue
            # `.strip()` deliberately: the lines are indented and anchoring to
            # line-start is how this check returned zero the first time it was
            # tried by hand.
            if line.strip().startswith("PROBLEM:") and (status or "").lower() == "passed":
                hits[name or "<unnamed>"] += 1
    return seen, hits


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip().splitlines()[-3], file=sys.stderr)
        return 2
    total_seen = 0
    total: collections.Counter = collections.Counter()
    for run in argv:
        seen, hits = scan(run)
        total_seen += seen
        total.update(hits)
        print(f"{run}: {seen} report(s) | {dict(hits) or 'clean'}")
    if not total_seen:
        print("no validator_report.txt anywhere -- CANNOT JUDGE, not clean", file=sys.stderr)
        return 2
    if total:
        print(f"\nT49: {sum(total.values())} PROBLEM: line(s) under a 'passed' heading")
        for k, v in total.most_common():
            print(f"  {v:4}  {k}")
        return 1
    print(f"\nclean across {total_seen} report(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
