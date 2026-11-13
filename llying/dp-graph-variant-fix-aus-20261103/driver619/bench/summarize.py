#!/usr/bin/env python3
"""DRIVER619-20261113: one table per node from repro_node.sh logs: headroom (largest accepted hipMalloc) before and
after registration, reported free, and the new-stream probe. Usage: summarize.py OUT_DIR"""
import glob
import json
import os
import sys

rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.log"))):
    steps, tail = {}, ""
    for line in open(f, errors="replace"):
        line = line.strip()
        if line.startswith("{"):
            try:
                d = json.loads(line)
                steps[d["step"]] = d
            except ValueError:
                pass
        elif line:
            tail = line
    s = steps.get("start", {})
    b, r, af = steps.get("before", {}), steps.get("registered", {}), steps.get("after", {})
    gp = steps.get("gem_pinned", {})
    err = steps.get("error", {})
    rows.append([os.path.basename(f)[:-4], s.get("driver", "?"), b.get("headroom_gib"), gp.get("headroom_gib", ""),
                 r.get("headroom_gib"), r.get("rep_free_gib"), r.get("real_free_gib"), r.get("mrs"), r.get("secs"),
                 af.get("stream", ""), (err.get("what", "") + " " + str(err.get("rc", err.get("errno", "")))).strip(),
                 tail if tail.startswith("exit") else tail[-80:]])
hdr = ["case", "driver", "headroom before", "headroom after GEM pin", "headroom after reg", "reported free after reg",
       "real free after reg", "MRs", "reg s", "new stream probe", "error", "last line"]
print("| " + " | ".join(hdr) + " |")
print("|" + "---|" * len(hdr))
for r in rows:
    print("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
