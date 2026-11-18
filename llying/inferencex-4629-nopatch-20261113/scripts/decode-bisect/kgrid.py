import gzip, json, glob, sys, re, collections
def load(d, rank=0):
    f = sorted(glob.glob(f"{d}/profile/*TP-{rank}-*.json.gz"))[0]
    ev = json.load(gzip.open(f, "rt"))["traceEvents"]
    return [e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"]
pat = re.compile(sys.argv[3])
for d in sys.argv[1:3]:
    ks = load(d)
    agg = collections.defaultdict(list)
    for e in ks:
        if pat.search(e["name"]):
            a = e.get("args", {})
            agg[(e["name"][:60], str(a.get("grid")), str(a.get("block")))].append(e["dur"])
    print("==", d)
    for k, v in sorted(agg.items(), key=lambda kv: -sum(kv[1])):
        v.sort()
        print(f"  n={len(v):5d} med={v[len(v)//2]:8.1f}us p10={v[len(v)//10]:8.1f} p90={v[len(v)*9//10]:8.1f}  {k}")
