import glob, json, subprocess, sys, os
os.makedirs("logs_v3/ab2_groups", exist_ok=True)
res = {}
for arm in ("A", "B2"):
    for f in sorted(glob.glob(f"work_v3/iterations/p_v3y_{arm}_u173k/profile/decode-*.trace.json.gz")):
        rank = "r0" if "DP-0" in f else "r1"
        out = subprocess.run([sys.executable, "logs_v3/kernel_groups.py", f, "5"], capture_output=True, text=True, check=True).stdout
        open(f"logs_v3/ab2_groups/{arm}_{rank}.json", "w").write(out)
        res[(arm, rank)] = json.loads(out)
groups = ["pa_mqa_logits", "radix_topk", "sparse_mla", "moe_stage1", "moe_stage2", "moe_sorting", "mega_mhc", "nccl", "engram", "other"]
print("group | A r0 / r1 | B2 r0 / r1 | B2 - A (mean of ranks)")
for g in groups:
    vals = {a: [res[(a, r)]["groups"].get(g, {}).get("ms_per_step", 0.0) for r in ("r0", "r1")] for a in ("A", "B2")}
    m = {a: sum(v) / 2 for a, v in vals.items()}
    print(f"{g} | {vals['A'][0]:.3f} / {vals['A'][1]:.3f} | {vals['B2'][0]:.3f} / {vals['B2'][1]:.3f} | {m['B2'] - m['A']:+.3f}")
for a in ("A", "B2"):
    ks = [k for k in res[(a, "r0")]["kernels"] if k["group"] in ("pa_mqa_logits", "radix_topk", "sparse_mla", "moe_stage1", "moe_stage2")]
    print(f"== {a} r0 kernels")
    for k in ks[:14]:
        print(f"  {k['ms_per_step']:.3f} {k['calls_per_step']:.1f} {k['group']} {k['name'][:90]}")
