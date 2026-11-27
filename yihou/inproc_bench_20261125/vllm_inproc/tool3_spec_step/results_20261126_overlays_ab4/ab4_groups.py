import collections, glob, gzip, json, subprocess, sys, os
os.makedirs("logs_v3/ab4_groups", exist_ok=True)
res, gemm = {}, {}
GEMM_KEYS = ("mxpsh", "bpreshuffle", "bmm_a8w8", "cijk", "gemm_a8w8", "mxscale")
for arm in ("B3", "B4"):
    for f in sorted(glob.glob(f"work_v3/iterations/p_v3w_{arm}_u173k/profile/decode-*.trace.json.gz")):
        rank = "r0" if "DP-0" in f else "r1"
        out = subprocess.run([sys.executable, "logs_v3/kernel_groups.py", f, "5"], capture_output=True, text=True, check=True).stdout
        open(f"logs_v3/ab4_groups/{arm}_{rank}.json", "w").write(out)
        res[(arm, rank)] = json.loads(out)
        acc = collections.defaultdict(lambda: [0.0, 0])
        for e in json.load(gzip.open(f, "rt"))["traceEvents"]:
            if e.get("ph") == "X" and e.get("cat") == "kernel" and any(k in e["name"].lower() for k in GEMM_KEYS) and "moe" not in e["name"].lower():
                a = acc[e["name"]]; a[0] += e["dur"]; a[1] += 1
        gemm[(arm, rank)] = {k: (v[0] / 5000, v[1] / 5) for k, v in acc.items()}
groups = ["pa_mqa_logits", "radix_topk", "sparse_mla", "moe_stage1", "moe_stage2", "mega_mhc", "nccl", "engram", "other"]
print("group | B3 r0 / r1 | B4 r0 / r1 | B4 - B3 (mean of ranks)")
for g in groups:
    vals = {a: [res[(a, r)]["groups"].get(g, {}).get("ms_per_step", 0.0) for r in ("r0", "r1")] for a in ("B3", "B4")}
    m = {a: sum(v) / 2 for a, v in vals.items()}
    print(f"{g} | {vals['B3'][0]:.3f} / {vals['B3'][1]:.3f} | {vals['B4'][0]:.3f} / {vals['B4'][1]:.3f} | {m['B4'] - m['B3']:+.3f}")
print("\nGEMM-family kernels (non-MoE), ms/step mean of ranks, calls/step: B3 | B4 | delta")
names = sorted(set(k for d in gemm.values() for k in d), key=lambda n: -max(gemm[(a, r)].get(n, (0, 0))[0] for a in ("B3", "B4") for r in ("r0", "r1")))
tot = {"B3": 0.0, "B4": 0.0}
for n in names:
    m = {a: sum(gemm[(a, r)].get(n, (0, 0))[0] for r in ("r0", "r1")) / 2 for a in ("B3", "B4")}
    c = {a: gemm[(a, "r0")].get(n, (0, 0))[1] for a in ("B3", "B4")}
    tot["B3"] += m["B3"]; tot["B4"] += m["B4"]
    if max(m.values()) >= 0.05:
        print(f"{m['B3']:.3f} ({c['B3']:.0f}) | {m['B4']:.3f} ({c['B4']:.0f}) | {m['B4'] - m['B3']:+.3f} | {n[:110]}")
print(f"GEMM family total: B3 {tot['B3']:.3f} B4 {tot['B4']:.3f} delta {tot['B4'] - tot['B3']:+.3f}")
