import sys, gzip, json, collections
# summarize CPU-side ops of a decode torch trace: top names by total dur on CPU threads, sync-type ops, python frames
f = sys.argv[1]
j = json.load(gzip.open(f))
ev = [e for e in j["traceEvents"] if e.get("ph") == "X"]
cats = collections.Counter(e.get("cat") for e in ev)
print("cats", cats.most_common(12))
tot = collections.defaultdict(lambda: [0, 0.0]); 
for e in ev:
    if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"): continue
    k = (e.get("cat"), e["name"][:90]); tot[k][0] += 1; tot[k][1] += e.get("dur", 0)
ts = [e["ts"] for e in ev]; span = (max(ts) - min(ts)) / 1e3
print(f"span {span:.1f} ms")
for k, (n, d) in sorted(tot.items(), key=lambda t: -t[1][1])[:45]:
    print(f"{d/1e3:10.1f} ms n={n:6d} {k[0]} | {k[1]}")
for pat in ("graph_variants", "_resolve_attention_variant", "select", "Synchronize", "item", "Memcpy", "load_batch", "recv_requests", "broadcast"):
    m = [(k, v) for k, v in tot.items() if pat in k[1]]
    for k, (n, d) in sorted(m, key=lambda t: -t[1][1])[:4]:
        print(f"  [{pat}] {d/1e3:9.1f} ms n={n} {k[0]} | {k[1]}")
