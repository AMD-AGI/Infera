"""GPU idle time per decode step, by position (before draft / verify / draft_extend graphs), from kernel timestamps.
python3 dgaps.py TRACE.json.gz"""
import gzip, json, sys, statistics as st
d = json.load(gzip.open(sys.argv[1])); ev = [e for e in d["traceEvents"] if e.get("ph") == "X"]
gl = sorted((e for e in ev if e.get("name") == "hipGraphLaunch"), key=lambda e: e["ts"])
idx = {e["args"]["correlation"]: i for i, e in enumerate(gl)}
k = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
# idle gaps attributed to the next kernel's owner (graph i or eager)
gaps = {"->draft": [], "->verify": [], "->draft_extend": [], "->eager": []}
names = ["->draft", "->verify", "->draft_extend"]
end = k[0]["ts"] + k[0]["dur"]
for e in k[1:]:
    g = e["ts"] - end
    if g > 0:
        i = idx.get(e["args"].get("correlation"))
        gaps[names[i % 3] if i is not None else "->eager"].append(g)
    end = max(end, e["ts"] + e["dur"])
n = len(gl) // 3
print(" ".join(f"{k_}: {sum(v) / 1e3 / n:.2f} ms/step" for k_, v in gaps.items()), f"| total {sum(sum(v) for v in gaps.values()) / 1e3 / n:.2f}")
# CPU side: python step time, and time from verify graph launch call to its first kernel
