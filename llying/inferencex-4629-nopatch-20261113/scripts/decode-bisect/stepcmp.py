import re, sys, statistics as st
pat = re.compile(r"(DP\d+) TP\d+.*?Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+), accept len: ([\d.]+).*?gen throughput \(token/s\): ([\d.]+)")
names = sys.argv[1].split(",")
res = {}
for name, p in zip(names, sys.argv[2:]):
    rows = []
    for line in open(p, errors="ignore"):
        m = pat.search(line)
        if not m: continue
        dp, r, tok, use, acc, g = m.groups(); r=int(r); tok=int(tok); acc=float(acc); g=float(g)
        if g < 1 or r < 1: continue
        rows.append((dp, r, tok, acc, r*acc/g*1000))
    res[name] = rows
    print(f"{name}: {len(rows)} samples, accept len median {st.median(x[3] for x in rows):.2f}, running median {st.median(x[1] for x in rows)}, KV tokens median {st.median(x[2] for x in rows)/1e6:.2f}M, step median {st.median(x[4] for x in rows):.1f} ms")
print("\nstep ms (median, per DP rank) by running-req x KV-token bucket")
for lo,hi in [(1,4),(5,9),(10,14),(15,19),(20,24),(25,40)]:
  for tlo,thi in [(0,1e6),(1e6,2e6),(2e6,3e6),(3e6,6e6)]:
    vals=[]
    for n in names:
        xs=[x[4] for x in res[n] if lo<=x[1]<=hi and tlo<=x[2]<thi]
        vals.append((st.median(xs), len(xs)) if len(xs)>=5 else None)
    if all(v is None for v in vals): continue
    print(f"run {lo:>2}-{hi:<2} tok {tlo/1e6:.0f}-{thi/1e6:.0f}M: " + "  ".join(f"{n}={v[0]:6.1f}(n={v[1]:4d})" if v else f"{n}=     -        " for n,v in zip(names,vals)))
