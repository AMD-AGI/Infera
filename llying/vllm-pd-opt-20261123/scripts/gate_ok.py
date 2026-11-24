#!/usr/bin/env python3
"""Print "yes" if a `gate p4d4-...:` summary line is clean: all smoke requests ok, GSM8K >= 190/200, no garbled output,
at most one bad start, real acceptance length >= 3.0 (MTP K3). Same rule as the a4 / a7 once scripts.
  gate_ok.py "<summary line>"
"""
import re
import sys


def g(s, p, d=-1.0):
    m = re.search(p, s)
    return float(m.group(1)) if m else d


def main():
    s = sys.argv[1] if len(sys.argv) > 1 else ""
    smoke = re.search(r"smoke (\d+)/(\d+)", s)
    good = (smoke is not None and smoke.group(1) == smoke.group(2) and g(s, r"correct=(\d+)/200") >= 190
            and g(s, r"garbled=(\d+)", 99) == 0 and g(s, r"bad_start=(\d+)", 99) <= 1
            and g(s, r'"accept_len": ([0-9.]+)') >= 3.0)
    print("yes" if good else "no")


if __name__ == "__main__":
    main()
