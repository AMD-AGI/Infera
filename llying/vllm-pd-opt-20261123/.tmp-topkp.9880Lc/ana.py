import collections, gzip, json, sys

d = json.load(gzip.open(sys.argv[1]))
ev = [e for e in d["traceEvents"] if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_op")]
ev.sort(key=lambda e: e["ts"])
names = collections.Counter()
for e in ev:
    n = e["name"]
    if any(s in n for s in ("topk", "top_k", "mqa_logits", "logits")):
        names[n[:110]] += 1
for n, c in names.most_common(30):
    print(c, n)
prev = collections.Counter()
grids = collections.Counter()
durs = collections.defaultdict(list)
last_logits = None
for e in ev:
    n = e["name"]
    if "mqa_logits" in n or "_logits_" in n:
        last_logits = (n[:70], str(e.get("args", {}).get("grid")))
    if "radix_topk_one_block" in n:
        prev[last_logits] += 1
        g = str(e.get("args", {}).get("grid"))
        grids[g] += 1
        durs[g].append(e["dur"])
print("-- preceding logits kernel (name, grid) of one-block topk")
for k, c in prev.most_common(15):
    print(c, k)
print("-- one-block topk grid: count, mean us, max us")
for g, c in sorted(grids.items(), key=lambda x: -x[1])[:25]:
    v = durs[g]
    print(c, g, round(sum(v) / len(v), 1), max(v))
for e in ev:
    if "radix_topk_one_block" in e["name"]:
        print("sample args", e.get("args"))
        break
ts = [e["ts"] for e in ev]
print("span s", (ts[-1] - ts[0]) / 1e6)
