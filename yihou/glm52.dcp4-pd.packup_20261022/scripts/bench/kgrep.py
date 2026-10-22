#!/usr/bin/env python3
"""kgrep.py TRACE REGEX [REGEX ...]: count and total ms of GPU kernels whose name matches each regex (case-insensitive)."""
import gzip, json, re, sys
ev = json.load(gzip.open(sys.argv[1], "rt"))["traceEvents"]
ks = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
print("kernels", len(ks), "total ms %.2f" % (sum(k["dur"] for k in ks) / 1e3))
for rx in sys.argv[2:]:
    r = re.compile(rx, re.I)
    m = [k for k in ks if r.search(k["name"])]
    names = sorted({k["name"][:70] for k in m})
    print(f"{rx}: n={len(m)} ms={sum(k['dur'] for k in m) / 1e3:.2f} names={names[:4]}")
cats = sorted({e.get("cat") for e in ev if e.get("ph") == "X"})
print("cats", cats)
gl = [e for e in ev if e.get("ph") == "X" and "graphlaunch" in e.get("name", "").lower()]
print("graph launches", len(gl), sorted({e["name"] for e in gl}))
