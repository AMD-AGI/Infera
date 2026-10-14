#!/usr/bin/env python3
"""Verify that a /flush_cache emptied every prefill rank, device and HiCache host tier alike.

Usage: flush_check.py SERVER_LOG_DIR SINCE_ISO
After a successful flush the first "Prefill batch" on each (instance, DP rank) can hit
nothing, so its #cached-token must be 0. A prefill counts as live when its log has any line
since SINCE (a stopped instance logs nothing); each has PREFILL_DP ranks (default 4).
Exit 0 when every live rank has logged a batch and all are 0; 1 when any is non-zero; 2 when
some live rank has no batch since SINCE yet.
"""

import os
import re
import sys
from pathlib import Path

LINE = re.compile(
    r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)Z \[[^\]]*DP(\d+) TP\d+\] Prefill batch, #new-seq: \d+, "
    r"#new-token: (\d+), #cached-token: (\d+)")


def main() -> int:
    logs, since = Path(sys.argv[1]), sys.argv[2]
    first = {}
    live = 0
    for log in sorted(logs.glob("prefill-[0-9].log")):
        seen_line = False
        with open(log, errors="replace") as stream:
            for line in stream:
                if not seen_line and line[:19] >= since[:19] and line[:4].isdigit():
                    seen_line = True
                    live += 1
                m = LINE.match(line)
                if not m or m.group(1) < since:
                    continue
                key = (log.stem, int(m.group(2)))
                if key not in first:
                    first[key] = (m.group(1)[:19], int(m.group(3)), int(m.group(4)))
    bad = 0
    for key in sorted(first):
        ts, new, cached = first[key]
        bad += cached != 0
        print(f"{key[0]} DP{key[1]}  first batch {ts}  new={new}  cached={cached}"
              f"{'' if cached == 0 else '  <-- NOT EMPTY'}")
    expected = live * int(os.environ.get("PREFILL_DP", "4"))
    print(f"ranks seen {len(first)}/{expected}, non-empty first batches {bad}")
    if bad:
        return 1
    return 0 if len(first) == expected else 2


if __name__ == "__main__":
    sys.exit(main())
