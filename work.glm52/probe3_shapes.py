"""Zero-GPU-alloc probe: what shapes does GLM-5.2 actually feed _forward_aiter?

Reads config.json + sglang's own helpers to compute, WITHOUT launching a server:
  - layer.head_dim / v_head_dim as RadixAttention sees them for a DSA MLA layer
  - pool kv_cache_dim under HIP+aiter
  - whether `kv_cache.view(-1, 1, 1, layer.head_dim)` in _forward_aiter can be valid
  - num_q_heads per GPU at TP8 and the resulting padded head count
"""
import json, sys

CFG = "/mnt/vast/xiaobo/models/GLM-5.2-MXFP4/config.json"
c = json.load(open(CFG))

kv_lora_rank = c["kv_lora_rank"]
qk_rope = c["qk_rope_head_dim"]
qk_nope = c["qk_nope_head_dim"]
head_dim_cfg = c.get("head_dim")
qk_head_dim = c.get("qk_head_dim")
v_head_dim = c["v_head_dim"]
nheads = c["num_attention_heads"]
topk = c["index_topk"]

print("=== config.json ===")
for k in ("architectures","num_attention_heads","head_dim","qk_head_dim","qk_nope_head_dim",
          "qk_rope_head_dim","kv_lora_rank","v_head_dim","index_topk","index_head_dim"):
    print(f"  {k:20s} = {c.get(k)}")

pool_dim = kv_lora_rank + qk_rope
print("\n=== derived ===")
print(f"  pool kv_cache_dim (HIP aiter/tilelang raw) = {kv_lora_rank} + {qk_rope} = {pool_dim}")

for tp in (8, 4, 2):
    nq = nheads // tp
    pad = nq < 16
    rf = 16 // nq if nq < 16 else 1
    print(f"  TP{tp}: num_q_heads={nq:3d}  need_pad_heads={pad}  repeat_factor={rf}  padded={nq*rf}")

print("\n=== the _forward_aiter view (dsa_backend.py) ===")
print("  code: kv_cache.view(-1, 1, 1, layer.head_dim)")
print("  MLA absorbed layer: head_dim = kv_lora_rank + qk_rope_head_dim = %d" % pool_dim)
print("                      v_head_dim = kv_lora_rank = %d" % kv_lora_rank)
print("  -> if RadixAttention.head_dim == %d, the view MATCHES the pool (OK)" % pool_dim)
print("  -> if it is config head_dim (%s), the view is WRONG by %sx"
      % (head_dim_cfg, None if not head_dim_cfg else round(pool_dim/head_dim_cfg, 3)))

print("\n=== NOTE ===")
print("  For DeepSeek-V3.2: kv_lora_rank+qk_rope = 512+64 = 576, and the MLA layer's")
print("  head_dim is also 576 -> aiter path consistent. Need to confirm what sglang")
print("  actually sets for GlmMoeDsa (qk_head_dim=%s, head_dim=%s)." % (qk_head_dim, head_dim_cfg))
