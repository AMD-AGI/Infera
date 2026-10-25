import sys, json, statistics as st
T = sys.argv[1]
for h in ("135", "136"):
    v = [(l.split()[0], float(l.split()[1])) for l in open(f"{T}/probe.crsuse2-m2m-{h}.cpu")]
    b = sorted((x for _, x in v), reverse=True)
    print(f"== {h} cpu: n={len(v)} mean busy={st.mean(b):.1f}% sum-cores={sum(b)/100:.1f} cores; >90%:{sum(x>90 for x in b)} >50%:{sum(x>50 for x in b)} >20%:{sum(x>20 for x in b)}")
    print("   top10:", " ".join(f"{c}={x:.0f}" for c, x in sorted(v, key=lambda t: -t[1])[:10]))
    cur = None; d = {}
    txt = open(f"{T}/probe.crsuse2-m2m-{h}.gpu").read().split("== ")[1:]
    rows = []
    for blk in txt:
        t, _, js = blk.partition("\n")
        js = js.strip()
        if not js: continue
        try: j = json.loads(js)
        except Exception: continue
        rows.append((t.strip(), j))
    print(f"== {h} gpu: {len(rows)} samples {rows[0][0]}..{rows[-1][0]}")
    for c in ("card4","card5","card6","card7"):
        u = [float(j[c]["GPU use (%)"]) for _, j in rows]; p = [float(j[c]["Current Socket Graphics Package Power (W)"]) for _, j in rows]
        print(f"   {c}: use mean={st.mean(u):.1f} min={min(u):.0f} max={max(u):.0f} | power mean={st.mean(p):.0f} min={min(p):.0f} max={max(p):.0f} W | use series {' '.join(f'{x:.0f}' for x in u)}")
