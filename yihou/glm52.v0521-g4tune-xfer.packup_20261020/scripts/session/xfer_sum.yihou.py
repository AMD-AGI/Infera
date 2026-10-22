# Task g4tune 2026-10-19: per-DP, per-5-minute aggregate of YIHOU-XFER shard lines (read-only). Usage: xfer_sum.yihou.py <prefill-0.log>
import sys, re, collections
rx = re.compile(r"(\d\d):(\d\d):\d\d (DP\d) TP\d\] YIHOU-XFER shard=(\d) span=(\d+)s.*?wait_mean=([\d.]+)s.*?calls=(\d+) blocks=(\d+).*?GiB=([\d.]+) wall=([\d.]+)s")
agg = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0.0, set()])  # gib, span, wall, calls, wait*calls, shards
for l in open(sys.argv[1], errors="replace"):
    m = rx.search(l)
    if not m: continue
    hh, mm, dp, sh, span, wm, calls, blk, gib, wall = m.groups()
    key = (f"{hh}:{int(mm)//5*5:02d}", dp)
    a = agg[key]; a[0] += float(gib); a[1] += int(span); a[2] += float(wall); a[3] += int(calls); a[4] += float(wm) * int(calls); a[5].add(sh)
rows = collections.defaultdict(dict)
for (t, dp), a in agg.items():
    # GiB/s = total GiB / 300 s window (5 min); busy = wall / 300 s (can exceed 100% with >1 shard)
    rows[t][dp] = "%s sh=%d GiB/s=%.2f busy=%.0f%% wait=%.2f" % (dp, len(a[5]), a[0] / 300, 100 * a[2] / 300, a[4] / max(1, a[3]))
for t in sorted(rows):
    print(t, " | ".join(rows[t][d] for d in sorted(rows[t])))
