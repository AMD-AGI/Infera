import collections, gzip, json, sys

d = json.load(gzip.open(sys.argv[1]))
ev = [e for e in d["traceEvents"] if e.get("ph") == "X"]
tot = collections.defaultdict(lambda: [0, 0.0])
for e in ev:
    if e.get("cat") == "kernel" and "radix_topk_one_block" in e["name"]:
        t = tot[e["name"][:60]]
        t[0] += 1
        t[1] += e["dur"]
for n, (c, s) in tot.items():
    print(f"{n}: {c} calls, {s/1e3:.1f} ms, mean {s/c:.1f} us")
cpu = collections.Counter()
shapes = collections.Counter()
for e in ev:
    n = e["name"]
    if e.get("cat") in ("cpu_op", "user_annotation") and ("top_k" in n or "topk" in n or "mqa_logits" in n):
        cpu[n] += 1
        a = e.get("args", {})
        dims = a.get("Input Dims")
        if dims is not None:
            shapes[(n, json.dumps(dims[:3]))] += 1
for n, c in cpu.most_common(20):
    print("cpu", c, n)
for (n, s), c in shapes.most_common(40):
    print("shape", c, n, s)
