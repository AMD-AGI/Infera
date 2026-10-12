# Summarize a kineto trace: GPU kernel busy vs gaps per stream, top kernels, runtime API calls (graph launches/syncs).
import gzip, json, sys, collections
for fn in sys.argv[1:]:
    d = json.load(gzip.open(fn))
    ev = [e for e in d.get("traceEvents", []) if e.get("ph") == "X"]
    cats = collections.Counter(e.get("cat") for e in ev)
    k = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    rt = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    print(f"== {fn.split('/')[-1]}  events={len(ev)} cats={dict(cats)}")
    if not k:
        continue
    t0 = min(e["ts"] for e in k); t1 = max(e["ts"] + e["dur"] for e in k)
    # union busy over all GPU streams
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in k)
    busy = 0; cs, ce = iv[0]
    gaps = []
    for s, e in iv[1:]:
        if s > ce:
            busy += ce - cs; gaps.append(s - ce); cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    span = t1 - t0
    g = sorted(gaps)
    print(f"  GPU span {span/1e3:.1f} ms, busy(union) {busy/1e3:.1f} ms ({100*busy/span:.1f}%), gaps n={len(g)} sum={sum(g)/1e3:.1f} ms, gaps>100us: n={sum(1 for x in g if x>100)} sum={sum(x for x in g if x>100)/1e3:.1f} ms, top gaps(us)={[int(x) for x in g[-8:]]}")
    streams = collections.Counter(e.get("args", {}).get("stream") for e in k)
    print(f"  streams: {dict(streams.most_common(6))}")
    byname = collections.defaultdict(float); cnt = collections.Counter()
    for e in k:
        n = e["name"][:90]; byname[n] += e["dur"]; cnt[n] += 1
    for n, t in sorted(byname.items(), key=lambda x: -x[1])[:18]:
        print(f"    {t/1e3:8.2f} ms ({100*t/busy:4.1f}%) x{cnt[n]:<5d} {n}")
    rts = collections.defaultdict(float); rc = collections.Counter()
    for e in rt:
        rts[e["name"]] += e["dur"]; rc[e["name"]] += 1
    print("  runtime API:", ", ".join(f"{n} x{rc[n]} {t/1e3:.1f}ms" for n, t in sorted(rts.items(), key=lambda x: -x[1])[:8]))
