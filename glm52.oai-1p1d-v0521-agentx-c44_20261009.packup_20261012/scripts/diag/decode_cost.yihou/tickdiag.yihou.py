import re, subprocess, datetime as dt, collections
import numpy as np
exec(open("step_regress.yihou.py").read().split("INTERVAL = 40")[0])
for k, (sd, bd) in runs.items():
    s, e = window(f"{R}/{bd}/benchmark.log")
    out = subprocess.run(["strings", f"{R}/{sd}/server-logs/decode-0.a1.log"], capture_output=True, text=True).stdout
    per = collections.defaultdict(list)
    for m in pat.finditer(out):
        if not (s <= m.group(2) <= e): continue
        per[int(m.group(3))].append(dt.datetime.fromisoformat(m.group(1)[:26]).timestamp())
    msg = []
    for d in sorted(per):
        t = np.array(sorted(per[d])); g = np.diff(t)
        msg.append(f"DP{d}: n={len(t)} gap_med={np.median(g):.2f}s gaps>8s={int((g>8).sum())} sum_gaps>8s={g[g>8].sum():.0f}s")
    print(k, " | ".join(msg))
