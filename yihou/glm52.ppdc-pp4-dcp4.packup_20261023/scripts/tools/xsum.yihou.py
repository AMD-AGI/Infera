# xsum (task dcpx8c 2026-10-22): aggregate [yihou-xfer] lines of a prefill log after a start time, per DP and total.
# Usage: python3 -I xsum.yihou.py <prefill-0.log> <start HH:MM:SS> [end HH:MM:SS]
import re, sys, collections as co
f, t0 = sys.argv[1], sys.argv[2]; t1 = sys.argv[3] if len(sys.argv) > 3 else "99:99:99"
keys = ["win", "calls", "blocks", "bytes", "t_xfer", "dcp_calls", "t_dcp", "pack_calls", "t_pack"]
agg = co.defaultdict(lambda: co.Counter()); hist = co.defaultdict(lambda: [0] * 6)
for line in open(f, errors="replace"):
    if "[yihou-xfer]" not in line: continue
    m = re.search(r"\[\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d) ((?:DP|PP)\d+)", line)
    if not m or not (t0 <= m.group(1) <= t1): continue
    dp = m.group(2); c = agg[dp]
    for k in keys:
        v = re.search(rf" {k}=([\d.]+)", line)
        if v: c[k] += float(v.group(1))
    h = re.search(r"hist\([^)]*\)=\[([\d, ]+)\]", line)
    if h: hist[dp] = [a + int(b) for a, b in zip(hist[dp], h.group(1).split(","))]
tot = co.Counter(); th = [0] * 6
for dp in sorted(agg):
    c = agg[dp]; tot.update(c); th = [a + b for a, b in zip(th, hist[dp])]
for name, c, h in [(dp, agg[dp], hist[dp]) for dp in sorted(agg)] + [("ALL", tot, th)]:
    w = c["win"] or 1
    print(f"{name}: win={c['win']:.0f}s GB={c['bytes']:.1f} rate={c['bytes']/w:.2f}GB/s calls={c['calls']:.0f} "
          f"t_xfer={c['t_xfer']:.0f}s ({c['bytes']/max(c['t_xfer'],1e-9):.1f}GB/s while in xfer) dcp_calls={c['dcp_calls']:.0f} "
          f"t_dcp={c['t_dcp']:.0f}s t_pack={c['t_pack']:.0f}s pack/dcp={c['t_pack']/max(c['t_dcp'],1e-9):.2f} "
          f"ms/pack={1e3*c['t_pack']/max(c['pack_calls'],1):.1f} ms/xfercall={1e3*c['t_xfer']/max(c['calls'],1):.1f} hist={h}")
