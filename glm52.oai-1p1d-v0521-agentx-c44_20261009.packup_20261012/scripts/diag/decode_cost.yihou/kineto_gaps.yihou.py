# For each GPU idle gap > 1 ms: CPU launch time (via correlation id) of the kernel that ends the gap, the CPU
# activity (user annotations / runtime syncs) just before that launch. Times relative to GPU idle start,
# launch->kernel latency L = kernel.ts - launch.ts (min over trace used as clock offset reference).
import gzip, json, sys, collections
fn = sys.argv[1]
d = json.load(gzip.open(fn)); ev = [e for e in d["traceEvents"] if e.get("ph") == "X"]
k = sorted([e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda e: e["ts"])
rt = [e for e in ev if e.get("cat") == "cuda_runtime"]
bycorr = {e["args"].get("correlation"): e for e in rt if "args" in e and e["args"].get("correlation") is not None}
lat = sorted(e["ts"] - bycorr[e["args"]["correlation"]]["ts"] for e in k if e.get("args", {}).get("correlation") in bycorr)
print(f"launch->start latency (us): min {lat[0]:.0f} p10 {lat[len(lat)//10]:.0f} p50 {lat[len(lat)//2]:.0f}")
ann = sorted([e for e in ev if e.get("cat") == "user_annotation"], key=lambda e: e["ts"])
syncs = sorted([e for e in rt if "Synchronize" in e["name"] or e["name"] == "hipMemcpyWithStream"], key=lambda e: e["ts"])
ce = k[0]["ts"] + k[0]["dur"]; tot = collections.defaultdict(float)
for e in k[1:]:
    g = e["ts"] - ce
    if g > 1000:
        c = e.get("args", {}).get("correlation"); r = bycorr.get(c)
        if r:
            launch_rel = (r["ts"] - ce) / 1e3  # CPU launch time minus GPU idle start (same clock domain? see latency)
            prev_sync = [s for s in syncs if s["ts"] + s["dur"] <= r["ts"]][-1:]
            ps = prev_sync[0] if prev_sync else None
            enclosing = [a["name"][:28] for a in ann if a["ts"] <= r["ts"] <= a["ts"] + a["dur"]]
            print(f"gap {g/1e3:5.2f} ms | launch of [{e['name'][:28]}] at idle+{launch_rel:6.2f} ms ({r['name']}) | "
                  f"prev sync {ps['name'] if ps else '-'} ended idle+{((ps['ts']+ps['dur'])-ce)/1e3 if ps else 0:6.2f} dur {ps['dur']/1e3 if ps else 0:.1f} | in {enclosing[-3:]}")
            tot[tuple(enclosing[-2:])] += g
    ce = max(ce, e["ts"] + e["dur"])
for k_, v in sorted(tot.items(), key=lambda x: -x[1]):
    print(f"  {v/1e3:7.1f} ms  {k_}")
