import glob, json, subprocess, sys, os
os.makedirs("logs_v3/abc_groups", exist_ok=True)
res = {}
for arm in "ABC":
    for f in sorted(glob.glob(f"work_v3/iterations/p_v3x_{arm}_u173k/profile/decode-*.trace.json.gz")):
        rank = "r0" if "DP-0" in f else "r1"
        out = subprocess.run([sys.executable, "logs_v3/kernel_groups.py", f, "5"], capture_output=True, text=True, check=True).stdout
        open(f"logs_v3/abc_groups/{arm}_{rank}.json", "w").write(out)
        res[(arm, rank)] = json.loads(out)
groups = ["pa_mqa_logits", "radix_topk", "sparse_mla", "moe_stage1", "moe_stage2", "moe_sorting", "mega_mhc", "nccl", "engram", "other"]
print("group | " + " | ".join(f"{a} r0 / r1" for a in "ABC") + " | B-A (r0,r1 mean) | C-A")
for g in groups:
    vals = {a: [res[(a, r)]["groups"].get(g, {}).get("ms_per_step", 0.0) for r in ("r0", "r1")] for a in "ABC"}
    m = {a: sum(v) / 2 for a, v in vals.items()}
    print(f"{g} | " + " | ".join(f"{vals[a][0]:.3f} / {vals[a][1]:.3f}" for a in "ABC") + f" | {m['B'] - m['A']:+.3f} | {m['C'] - m['A']:+.3f}")
for a in "ABC":
    for r in ("r0",):
        ks = [k for k in res[(a, r)]["kernels"] if k["group"] in ("pa_mqa_logits", "radix_topk", "sparse_mla", "moe_stage1", "moe_stage2")]
        print(f"== {a} {r} kernels")
        for k in ks[:14]:
            print(f"  {k['ms_per_step']:.3f} {k['calls_per_step']:.1f} {k['group']} {k['name'][:90]}")
