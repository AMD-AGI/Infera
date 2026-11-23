import sys, glob, statistics as st
for d in sys.argv[1:]:
    rows = []
    for f in sorted(glob.glob(d + "/*.csv")):
        prev_end = None
        for line in open(f):
            if not line.startswith("S,"): continue
            c = line.rstrip("\n").split(",")
            wall, dt, cpu, dummy, toks, mode = float(c[1]), float(c[2]), float(c[3]), int(c[5]), int(c[7]), c[17]
            if dummy == 0 and toks >= 16000:
                rows.append((dt, cpu, toks, mode))
    if not rows:
        print(d, "no big steps"); continue
    dts = sorted(r[0] for r in rows if r[0] < 5000)
    toks = st.mean(r[2] for r in rows)
    p = lambda q: dts[int(q * (len(dts) - 1))]
    print(f"{d}: n={len(rows)} toks_mean={toks:.0f} cadence_ms p25/p50/p75 = {p(.25):.0f}/{p(.5):.0f}/{p(.75):.0f}  modes={sorted(set(r[3] for r in rows))}")
