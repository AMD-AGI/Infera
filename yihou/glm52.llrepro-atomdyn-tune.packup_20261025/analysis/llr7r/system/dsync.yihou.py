import sys, gzip, json, bisect
# per rank: GPU kernel-union busy fraction; host D2H sync (hipMemcpyWithStream) durations; GPU idle gap right after each sync returns
for f in sys.argv[1:]:
    j = json.load(gzip.open(f)); ev = [e for e in j["traceEvents"] if e.get("ph") == "X"]
    k = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"))
    # union
    U = []
    for a, b in k:
        if U and a <= U[-1][1]: U[-1][1] = max(U[-1][1], b)
        else: U.append([a, b])
    t0, t1 = U[0][0], U[-1][1]; busy = sum(b - a for a, b in U)
    gaps = sorted(((U[i+1][0] - U[i][1]), U[i][1]) for i in range(len(U)-1))
    gl = [g for g, _ in gaps]
    big = [g for g in gl if g > 50]
    sy = sorted((e["ts"], e["dur"]) for e in ev if e.get("cat") == "cuda_runtime" and e["name"] == "hipMemcpyWithStream")
    sel = sorted((e["ts"], e["dur"]) for e in ev if e.get("cat") == "python_function" and "graph_variants.py(32): select" in e["name"])
    ends = [u[1] for u in U]; starts = [u[0] for u in U]
    after = []
    for ts, d in sy:
        r = ts + d  # host returns; next GPU work start after r
        i = bisect.bisect_left(starts, r)
        # GPU idle at r? find whether r lies inside a busy interval
        j2 = bisect.bisect_right(starts, r) - 1
        if j2 >= 0 and U[j2][1] > r: after.append(0.0)
        elif i < len(starts): after.append(starts[i] - r)
    after.sort()
    sd = sorted(d for _, d in sy); sel_d = sorted(d for _, d in sel)
    q = lambda a, p: a[min(len(a)-1, int(p*len(a)))] if a else float('nan')
    print(f"{f.split('/')[-1][-30:]}: span {(t1-t0)/1e3:.1f} ms, GPU union busy {busy/(t1-t0):.3f}, gaps>50us n={len(big)} sum={sum(big)/1e3:.1f} ms, all gaps sum {sum(gl)/1e3:.1f} ms; "
          f"hipMemcpyWithStream n={len(sy)} sum={sum(sd)/1e3:.1f} ms p50={q(sd,.5)/1e3:.2f} max={sd[-1]/1e3:.2f} ms; select() n={len(sel_d)} sum={sum(sel_d)/1e3:.1f} ms p50={q(sel_d,.5)/1e3:.2f} ms; "
          f"GPU idle after sync return: p50={q(after,.5):.0f} us p90={q(after,.9):.0f} us sum={sum(after)/1e3:.2f} ms")
