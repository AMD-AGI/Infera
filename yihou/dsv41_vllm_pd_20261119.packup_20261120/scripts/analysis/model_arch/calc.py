"""Param counts from safetensors headers + KV/FLOP arithmetic for DSv4.1-Flash."""
import json, re, sys, collections
T = json.load(open(sys.argv[1]))
def nparams(dt, sh, nb):
    n = 1
    for s in sh: n *= s
    if dt == "I8": n *= 2  # packed fp4 e2m1 x2
    return n
cnt = collections.Counter()
for k, (dt, sh, nb, f) in T.items():
    if k.endswith(".scale") or k.endswith("bias"): continue
    m = re.match(r"(layers|mtp)\.(\d+)\.(.*)", k)
    if not m: cnt["top:" + k.split(".")[0]] += nparams(dt, sh, nb); continue
    pre, lid, r = m.group(1), int(m.group(2)), m.group(3)
    if ".experts." in "." + r:
        cnt[f"{pre}.routed"] += nparams(dt, sh, nb)
    elif r.startswith("engram.embed"):
        cnt[f"{pre}.engram_table"] += nparams(dt, sh, nb)
    else:
        part = r.split(".")[0] if not r.startswith("attn.") else "attn." + (r.split(".")[1] if r.split(".")[1] in ("indexer","compressor") else "core")
        cnt[f"{pre}.{part}"] += nparams(dt, sh, nb)
for k in sorted(cnt): print(f"{k:28s} {cnt[k]/1e9:9.3f} B")
tot = sum(cnt.values()); print("total params (B)", tot/1e9)
# shapes of one expert and attn
for k in ["layers.3.ffn.experts.0.w1.weight","layers.3.ffn.experts.0.w1.scale","layers.3.ffn.experts.0.w2.weight","layers.3.attn.wq_b.weight","layers.3.attn.wo_a.weight","layers.3.attn.wo_b.weight","layers.3.attn.wkv.weight","layers.3.attn.wq_a.weight","layers.20.attn.compressor.wkv.weight","layers.2.attn.compressor.wkv.weight","layers.2.attn.indexer.wq_b.weight","layers.1.engram.embed.weight","layers.1.engram.embed.scale","layers.1.engram.wkv.weight","mtp.0.ffn.experts.0.w1.weight","mtp.2.markov_head.embed.weight","mtp.0.main_proj.weight","layers.3.hc_attn_fn","embed.weight","head.weight"]:
    v = T.get(k); print(k, v[:3] if v else None)
n_routed_layer = cnt["layers.routed"] / 40 / 384
print("params per routed expert (M)", n_routed_layer/1e6)
