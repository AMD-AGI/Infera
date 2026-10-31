"""Verify-graph time by category for each DP rank trace + one layer's kernel sequence.
python3 dtrace_cat.py TRACE.json.gz [...]"""
import gzip, json, sys, re, collections
CATS = [("moe_gemm", r"gemm1_a4w4|gemm2_a4w4|kernel_moe_gemm"), ("moe_misc", r"moe_sort|topk_reg|append_shared_experts|opus_moe"),
        ("allreduce_aiter", r"cross_device_reduce"), ("nccl", r"nccl"), ("attn_mla", r"sparse_mla|bmm_rope|batched_gemm_a8w8|rope_cat"),
        ("indexer", r"mqa_logits|_topk_|hadamard|kn_entry"), ("dense_gemm", r"Cijk|hgemm|_gemm_a16|bf16gemm"),
        ("norm", r"rmsnorm"), ("other", r".")]
for f in sys.argv[1:]:
    d = json.load(gzip.open(f)); ev = [e for e in d["traceEvents"] if e.get("ph") == "X"]
    gl = sorted((e for e in ev if e.get("name") == "hipGraphLaunch"), key=lambda e: e["ts"])
    ver = {e["args"]["correlation"] for i, e in enumerate(gl) if i % 3 == 1}
    ks = sorted((e for e in ev if e.get("cat") == "kernel" and e["args"].get("correlation") in ver), key=lambda e: e["ts"])
    n = len(ver); c = collections.Counter()
    for e in ks:
        for name, pat in CATS:
            if re.search(pat, e["name"]): c[name] += e["dur"]; break
    tot = sum(c.values())
    print(f"{f.split('/')[-1][:30]}: verify {tot / 1e3 / n:.2f} ms/step: " + ", ".join(f"{k} {c[k] / 1e3 / n:.2f}" for k, _ in CATS))
# one layer sequence (first verify launch, kernels 300..340) from the last file
corr0 = gl[1]["args"]["correlation"]
seq = [e for e in ks if e["args"]["correlation"] == corr0]
print("--- kernel sequence of one verify launch, kernels 200-232 (us)")
for e in seq[200:232]:
    nm = re.sub(r"<.*|\(.*", "", e["name"])[:80]
    print(f"  {e['dur']:7.1f}  {nm}")
