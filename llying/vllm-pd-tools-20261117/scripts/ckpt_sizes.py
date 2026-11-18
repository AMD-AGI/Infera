"""Per-category byte totals of a safetensors checkpoint from the file headers only (no tensor data read), and the
per-rank size of a DP-attention decode that shards routed experts over N ranks and replicates everything else.
   ckpt_sizes.py MODEL_DIR [N=4]"""
import glob, json, os, re, struct, sys, collections
d = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
cfg = json.load(open(os.path.join(d, "config.json")))
L = cfg.get("num_hidden_layers")
cat = collections.Counter()
def category(name):
    m = re.match(r"model\.layers\.(\d+)\.", name)
    layer = int(m.group(1)) if m else None
    mtp = layer is not None and L is not None and layer >= L
    pre = "mtp:" if mtp else ""
    if ".mlp.experts." in name: return pre + "routed_experts"
    if "shared_experts" in name: return pre + "shared_experts"
    if ".mlp.gate." in name or "e_score_correction" in name: return pre + "router"
    if ".mlp." in name: return pre + "dense_mlp"
    if "indexer" in name: return pre + "indexer"
    if "self_attn" in name: return pre + "attention"
    if "embed_tokens" in name: return pre + "embed"
    if "lm_head" in name or "shared_head" in name: return pre + "lm_head"
    if "norm" in name or "eh_proj" in name or "enorm" in name or "hnorm" in name: return pre + "norm/misc"
    return pre + "other"
dtypes = collections.Counter()
for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
    with open(f, "rb") as fh:
        hl = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hl))
    for k, v in hdr.items():
        if k == "__metadata__": continue
        b = v["data_offsets"][1] - v["data_offsets"][0]
        c = category(k); cat[c] += b
        dtypes[(c, v["dtype"])] += b
G = 1024 ** 3
tot = sum(cat.values())
print(f"layers {L}, total {tot / G:.1f} GiB")
for c, b in sorted(cat.items(), key=lambda x: -x[1]):
    print(f"  {c:24s} {b / G:8.2f} GiB  " + ", ".join(f"{dt} {bb / G:.1f}" for (cc, dt), bb in dtypes.items() if cc == c))
sharded = sum(b for c, b in cat.items() if c.endswith("routed_experts"))
print(f"per rank (routed experts / {n}, rest replicated): {(sharded / n + tot - sharded) / G:.1f} GiB; "
      f"(routed + shared experts / {n}): {((sharded + sum(b for c, b in cat.items() if c.endswith('shared_experts'))) / n + tot - sharded - sum(b for c, b in cat.items() if c.endswith('shared_experts'))) / G:.1f} GiB")
