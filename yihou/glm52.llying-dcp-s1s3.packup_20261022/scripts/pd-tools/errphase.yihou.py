import json, sys, collections
for p in sys.argv[1:]:
    c = collections.Counter(); first = {}
    for line in open(p):
        r = json.loads(line); md = r["metadata"]; ph = md.get("benchmark_phase"); e = "err" if r.get("error") else "ok"
        if r.get("error") and md.get("was_cancelled"): e = "err_cancelled"
        c[(ph, e)] += 1
        if e == "err" and ph not in first: first[ph] = md["request_start_ns"]
    print(p.split("/runs/")[1][:60], dict(c), {k: v // 10**9 for k, v in first.items()})
