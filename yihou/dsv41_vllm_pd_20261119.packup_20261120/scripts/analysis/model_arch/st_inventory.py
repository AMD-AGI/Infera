"""Read safetensors headers (no tensor data) and aggregate bytes by component/dtype."""
import json, struct, sys, re, os, collections
M = sys.argv[1]; out = sys.argv[2]
idx = json.load(open(os.path.join(M, "model.safetensors.index.json")))
files = sorted(set(idx["weight_map"].values()))
tensors = {}
for f in files:
    with open(os.path.join(M, f), "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    for k, v in h.items():
        if k == "__metadata__": continue
        a, b = v["data_offsets"]
        tensors[k] = (v["dtype"], v["shape"], b - a, f)
json.dump(tensors, open(out, "w"))
def comp(k):
    if k.startswith("vision") or k.startswith("aligner") or k.startswith("image_"): return "vision"
    if k.startswith("embed"): return "embed"
    if k.startswith("head"): return "lm_head"
    if k == "norm.weight": return "final_norm"
    m = re.match(r"(layers|mtp)\.(\d+)\.(.*)", k)
    pre = "main" if m.group(1) == "layers" else "mtp"
    r = m.group(3)
    if r.startswith("engram.embed"): return pre + ".engram_table"
    if r.startswith("engram"): return pre + ".engram_other"
    if ".experts." in "." + r: return pre + ".routed_experts"
    if "shared_experts" in r: return pre + ".shared_expert"
    if r.startswith("ffn.gate"): return pre + ".router"
    if "indexer" in r: return pre + ".attn.indexer"
    if "compressor" in r: return pre + ".attn.compressor"
    if r.startswith("attn"): return pre + ".attn.core"
    if r.startswith("hc_"): return pre + ".hyperconn"
    if "markov" in r or "confidence" in r or "main_proj" in r or "main_norm" in r: return pre + ".dspark_heads"
    return pre + ".norms"
agg = collections.defaultdict(lambda: collections.Counter())
for k, (dt, sh, nb, f) in tensors.items():
    agg[comp(k)][dt] += nb
tot = 0
for c in sorted(agg):
    s = sum(agg[c].values()); tot += s
    print(f"{c:28s} {s/1e9:10.3f} GB  " + ", ".join(f"{d}:{b/1e9:.3f}" for d, b in sorted(agg[c].items())))
print("TOTAL", tot/1e9, "GB", "index total_size", idx["metadata"]["total_size"]/1e9)
