"""Decode step breakdown from a kineto trace: per graph launch (draft / verify / draft_extend in step order) kernel
count, GPU kernel time and span; eager kernels; GPU idle inside the step; top kernels of each graph.
python3 dtrace.py TRACE.json.gz [TOPN=25]"""
import gzip, json, sys, re, collections, statistics as st
d = json.load(gzip.open(sys.argv[1])); topn = int(sys.argv[2]) if len(sys.argv) > 2 else 25
ev = [e for e in d["traceEvents"] if e.get("ph") == "X"]
gl = sorted((e for e in ev if e.get("name") == "hipGraphLaunch"), key=lambda e: e["ts"])
glc = {e["args"]["correlation"]: i for i, e in enumerate(gl)}
k = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
NAMES = ["draft", "verify", "draft_extend"]
per = collections.defaultdict(list)
eager = []
for e in k:
    i = glc.get(e["args"].get("correlation"))
    (per[i] if i is not None else eager).append(e)
nsteps = len(gl) // 3
print(f"graph launches {len(gl)} (~{nsteps} steps); kernels {len(k)}; eager {len(eager)}")
span0, span1 = k[0]["ts"], k[-1]["ts"] + k[-1]["dur"]
wall = (span1 - span0) / 1e3
busy = 0; cur = None
for e in k:
    s, t = e["ts"], e["ts"] + e["dur"]
    if cur and s <= cur[1]: cur[1] = max(cur[1], t)
    else:
        if cur: busy += cur[1] - cur[0]
        cur = [s, t]
busy += cur[1] - cur[0]
print(f"trace span {wall:.0f} ms = {wall / max(nsteps, 1):.1f} ms/step; GPU busy {busy / 1e3:.0f} ms ({100 * busy / 1e3 / wall:.0f}%)")
for g in range(3):
    idx = [i for i in range(len(gl)) if i % 3 == g]
    cnt = [len(per[i]) for i in idx]; kt = [sum(e["dur"] for e in per[i]) / 1e3 for i in idx]
    spans = [(per[i][-1]["ts"] + per[i][-1]["dur"] - per[i][0]["ts"]) / 1e3 for i in idx if per[i]]
    print(f"  {NAMES[g]:13s}: kernels/launch median {st.median(cnt):.0f} (min {min(cnt)}, launches with 0: {sum(c == 0 for c in cnt)}/{len(cnt)})"
          f"  kernel ms/launch {st.mean(kt):.2f}  span ms {st.mean(spans) if spans else 0:.2f}")
print(f"  eager kernels: {len(eager) / max(nsteps, 1):.0f}/step, {sum(e['dur'] for e in eager) / 1e3 / max(nsteps, 1):.2f} ms/step")
def norm(n):
    n = re.sub(r"<.*", "", n); n = re.sub(r"\(.*", "", n); return n[:90]
for g in range(3):
    c = collections.Counter(); n = collections.Counter()
    for i in range(g, len(gl), 3):
        for e in per[i]: c[norm(e["name"])] += e["dur"]; n[norm(e["name"])] += 1
    tot = sum(c.values())
    if not tot: continue
    print(f"--- {NAMES[g]} top kernels (ms/step, calls/step, share)")
    for name, t in c.most_common(topn):
        print(f"  {t / 1e3 / nsteps:7.3f} {n[name] / nsteps:6.1f}  {100 * t / tot:5.1f}%  {name}")
c = collections.Counter(); n = collections.Counter()
for e in eager: c[norm(e["name"])] += e["dur"]; n[norm(e["name"])] += 1
print("--- eager top kernels (ms/step, calls/step)")
for name, t in c.most_common(12): print(f"  {t / 1e3 / nsteps:7.3f} {n[name] / nsteps:6.1f}  {name}")
