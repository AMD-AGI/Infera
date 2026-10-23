import json, sys
n = big = 0; c = {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    m = r.get("metrics", {}).get("input_sequence_length")
    v = m.get("value") if isinstance(m, dict) else m
    if v is None: continue
    n += 1
    for t in (131072, 196608, 262144, 393216, 524288, 786432):
        if v > t: c[t] = c.get(t, 0) + 1
print("n", n, {k: f"{c.get(k,0)} ({100*c.get(k,0)/max(n,1):.1f}%)" for k in sorted(c)})
