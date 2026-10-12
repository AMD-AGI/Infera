# per-DP lines: step_ms = gap/40 vs own running-req and #transfer-req (in-flight incoming KV transfers)
import re, subprocess, datetime as dt, collections
import numpy as np
exec(open("step_regress.yihou.py").read().split("pat = re.compile")[0])
pat = re.compile(r"(\d{4}-\d\d-\d\dT(\d\d:\d\d:\d\d)\.\d+)Z \[[^\]]*DP(\d) TP\d\] Decode batch, #running-req: (\d+), #token: (\d+),.*?#prealloc-req: (\d+), #transfer-req: (\d+)")
for k, (sd, bd) in runs.items():
    s, e = window(f"{R}/{bd}/benchmark.log")
    out = subprocess.run(["strings", f"{R}/{sd}/server-logs/decode-0.a1.log"], capture_output=True, text=True).stdout
    per = collections.defaultdict(list)
    for m in pat.finditer(out):
        if not (s <= m.group(2) <= e): continue
        per[int(m.group(3))].append((dt.datetime.fromisoformat(m.group(1)[:26]).timestamp(), int(m.group(4)), int(m.group(7))))
    # global transfer activity: sum of #transfer-req over DPs at nearest time
    allrows = sorted((t, d, r, x) for d in per for (t, r, x) in per[d])
    ts = np.array([a[0] for a in allrows]); xs = np.array([a[3] for a in allrows])
    res = collections.defaultdict(list)
    for d in per:
        rows = sorted(per[d])
        for a, b in zip(rows, rows[1:]):
            g = b[0] - a[0]
            if g > 8: continue
            w = (ts > a[0]) & (ts <= b[0])
            gx = xs[w].mean() if w.any() else 0
            res[(b[1], gx > 0.5)].append(g / 40 * 1e3)
    line = []
    for r in range(3, 8):
        lo = res.get((r, False), []); hi = res.get((r, True), [])
        if len(lo) > 10 and len(hi) > 10:
            line.append(f"run{r}: noxfer {np.median(lo):.1f}({len(lo)}) xfer {np.median(hi):.1f}({len(hi)})")
    print(k, " | ".join(line))
