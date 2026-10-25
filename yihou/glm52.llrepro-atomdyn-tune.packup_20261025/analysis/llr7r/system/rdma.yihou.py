import sys, collections, statistics as st
# usage: rdma.yihou.py <rdma1s file> <col: tx|rx> ; line rate 400 Gb/s = 50 GB/s
LINE = 400e9/8
f, which = sys.argv[1], sys.argv[2]
rows = collections.defaultdict(list)
for l in open(f):
    p = l.split()
    if len(p) < 6: continue
    rows[p[1]].append((float(p[0]), int(p[2]), int(p[3]), int(p[4]), int(p[5])))
def pct(a, q):
    a = sorted(a); return a[min(len(a)-1, int(q*len(a)))]
print(f"# {f} col={which} line=50 GB/s; rates GB/s per sample interval")
summ = None; ts = None
series = {}
for d in sorted(rows):
    r = rows[d]
    rates = []
    for a, b in zip(r, r[1:]):
        dt = b[0]-a[0]; v = (b[1]-a[1]) if which == 'tx' else (b[2]-a[2])
        rates.append(v/dt/1e9)
    series[d] = rates
    tot = (r[-1][1]-r[0][1]) if which=='tx' else (r[-1][2]-r[0][2])
    dur = r[-1][0]-r[0][0]
    print(f"{d} n={len(rates)} dur={dur:.1f}s total={tot/1e9:.1f}GB mean={tot/dur/1e9:.2f} p50={pct(rates,.5):.2f} p90={pct(rates,.9):.2f} max={max(rates):.2f} "
          f">50%={sum(x>25 for x in rates)/len(rates):.3f} >80%={sum(x>40 for x in rates)/len(rates):.3f} zero(<0.01)={sum(x<0.01 for x in rates)/len(rates):.3f} "
          f"d_retx={r[-1][3]-r[0][3]} d_acktmo={r[-1][4]-r[0][4]}")
act = [d for d in series if sum(series[d]) > 1]
n = min(len(series[d]) for d in act)
s = [sum(series[d][i] for d in act) for i in range(n)]
print(f"SUM({','.join(act)}) mean={st.mean(s):.2f} p50={pct(s,.5):.2f} p90={pct(s,.9):.2f} max={max(s):.2f} GB/s; of {50*len(act)} GB/s")
# imbalance: per-sample max/mean over active rails, and totals ratio
imb = [max(series[d][i] for d in act)/(st.mean(series[d][i] for d in act)) for i in range(n) if s[i] > 1]
print(f"per-sample max/mean rail (samples with sum>1GB/s, n={len(imb)}): p50={pct(imb,.5):.2f} p90={pct(imb,.9):.2f}")
print("timeseries (t_rel, " + " ".join(act) + ", sum)")
t0 = rows[act[0]][0][0]
for i in range(n):
    print(f"{rows[act[0]][i+1][0]-t0:7.2f} " + " ".join(f"{series[d][i]:6.2f}" for d in act) + f" {s[i]:6.2f}")
