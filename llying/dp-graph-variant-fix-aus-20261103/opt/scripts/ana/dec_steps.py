"""Decode step time, accept length and DP imbalance from the decode log (lines every N steps per DP rank).
python3 dec_steps.py DECODE_LOG T0 T1 [N=40]"""
import re, sys, statistics as st
from datetime import datetime
log, T0, T1 = sys.argv[1:4]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 40
R = re.compile(r"^(\S+?)\d{3}Z \[\S+ \S+ DP(\d) TP\d\] Decode batch, #running-req: (\d+), #token: (\d+).*accept len: ([\d.]+).*#transfer-req: (\d+).*gen throughput \(token/s\): ([\d.]+)")
rows = {}
for line in open(log, errors="replace"):
    m = R.match(line)
    if not m: continue
    ts = m.group(1)
    if not (T0 <= ts[:19] < T1): continue
    t = datetime.fromisoformat(ts).timestamp()
    rows.setdefault(int(m.group(2)), []).append((t, int(m.group(3)), int(m.group(4)), float(m.group(5)), float(m.group(7))))
# step time per rank from consecutive lines (N steps apart); skip idle gaps > 10 s
per = []
for dp, v in rows.items():
    for a, b in zip(v, v[1:]):
        dt = b[0] - a[0]
        if 0 < dt < 10: per.append((b[0], dp, b[1], dt / N * 1e3, b[3], b[2]))
print(f"lines per rank: {', '.join(f'DP{k} {len(v)}' for k, v in sorted(rows.items()))}")
acc = [p[4] for p in per if p[2] > 0]; stp = [p[3] for p in per]
print(f"step ms mean {st.mean(stp):.1f} p50 {sorted(stp)[len(stp) // 2]:.1f}; accept len mean {st.mean(acc):.2f} (req-weighted {sum(p[4] * p[2] for p in per) / sum(p[2] for p in per):.2f})")
print(f"ms per output token (step/accept, req-weighted) {sum(p[3] / p[4] * p[2] for p in per if p[4] > 0) / sum(p[2] for p in per if p[4] > 0):.1f}")
# DP balance: at every log line, carry forward each rank's latest (running, step) seen within the last 3 s
ev = sorted(per); last = {}; imb_steps = []
for t, dp, run, sms, a_, tok in ev:
    last[dp] = (t, run, sms, tok)
    cur = [v for v in last.values() if t - v[0] < 3]
    if len(cur) == 4:
        r = [v[1] for v in cur]; imb_steps.append((max(r), st.mean(r), sms, st.mean(v[3] for v in cur), run, max(v[3] for v in cur), tok))
mx = [i[0] for i in imb_steps]; mn = [i[1] for i in imb_steps]
print(f"samples {len(imb_steps)}; running per rank mean {st.mean(mn):.1f}, busiest {st.mean(mx):.1f} (ratio {st.mean(mx) / st.mean(mn):.2f})")
xs = [i[0] for i in imb_steps]; ys = [i[2] for i in imb_steps]
mx_, my_ = st.mean(xs), st.mean(ys)
k = sum((x - mx_) * (y - my_) for x, y in zip(xs, ys)) / sum((x - mx_) ** 2 for x in xs); c = my_ - k * mx_
bal = st.mean(c + k * i[1] for i in imb_steps)
print(f"step ms ~ {c:.1f} + {k:.2f} * busiest; mean step {my_:.1f} ms, if perfectly balanced {bal:.1f} ms ({100 * (1 - bal / my_):.0f}% less)")
print(f"KV tokens per rank mean {st.mean(i[3] for i in imb_steps) / 1e3:,.0f}K")
for name, idx in (("own running", 4), ("busiest running", 0), ("mean running", 1), ("own KV tokens", 6), ("max KV tokens", 5), ("mean KV tokens", 3)):
    print(f"  corr(step, {name}) = {st.correlation([i[idx] for i in imb_steps], [i[2] for i in imb_steps]):.3f}")
