import re, sys, datetime, statistics as st
pat = re.compile(r"^(\S+)Z \[\S+ \S+ PP0\] Prefill batch.*?#new-token: (\d+), #cached-token: (\d+).*#queue-req: (\d+)")
t0, t1 = sys.argv[2], sys.argv[3]
rows = []
for line in open(sys.argv[1], errors="replace"):
    m = pat.match(line)
    if m and t0 <= m.group(1)[:16] <= t1:
        rows.append((datetime.datetime.fromisoformat(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))))
d = [(b[0] - a[0]).total_seconds() * 1e3 for a, b in zip(rows, rows[1:]) if a[3] > 0 and a[1] >= 8000]
d.sort(); n = len(d); tot = sum(d)
print(f"PP0 batches {len(rows)}, back-to-back full-chunk beats {n}: mean {tot/n:.0f} ms, median {d[n//2]:.0f}")
for lim in (450, 600, 800, 1200, 2000):
    s = [x for x in d if x > lim]
    print(f"  beats > {lim} ms: {100*len(s)/n:4.1f}% of beats, excess over median {sum(x - d[n//2] for x in s)/tot*100:4.1f}% of time")
print("  percentiles:", " ".join(f"p{p}={d[int(n*p/100)]:.0f}" for p in (10, 25, 50, 75, 90, 95, 99)))
print("beat by cached tokens of the batch (sum over its seqs):")
pairs = [((b[0] - a[0]).total_seconds() * 1e3, a[2]) for a, b in zip(rows, rows[1:]) if a[3] > 0 and a[1] >= 8000]
for lo, hi in ((0, 50e3), (50e3, 100e3), (100e3, 200e3), (200e3, 400e3), (400e3, 1e9)):
    v = sorted(x for x, c in pairs if lo <= c < hi)
    if v:
        k = len(v)
        print(f"  cached {lo/1e3:4.0f}k-{hi/1e3:5.0f}k: n {k:5d}  p10 {v[k//10]:5.0f}  p50 {v[k//2]:5.0f}  p90 {v[int(k*.9)]:5.0f}  mean {sum(v)/k:5.0f} ms")
