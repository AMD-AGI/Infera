#!/usr/bin/env python3
# Per-pool GiB decomposition of the DSv4 KV pool, and the gmu->pool master table.
GiB=1024**3
BPFT=13735.04
VRAM=287.98  # GiB/card

# From sglang source (agent-verified, v0.5.15.post1):
#  MLA/c4 per-token-per-layer = qk_nope(448 fp8)+qk_rope(64)*2 bf16 + scales = 584 B
#  indexer/c128 per-token-per-layer (fp8) = index_head_dim(128)+4 = 132 B
# bytes_per_full_token=13735.04 is the COMPOSITE of all pools over their layer subsets:
#   swa_ratio*584*L_total + (1/4)*584*L_ca4 + (1/128)*584*L_ca128
#   + (1/4)*132*L_ca4 + swa_ratio*c4_state_terms...  (see _get_bytes_per_full_token)
# We don't have the per-layer ca4/ca128 split from logs, so we report the pool
# TOKEN capacities (exact, logged) and the TOTAL KV bytes (exact, logged), which is
# what the task asks: "actual per-card KV memory vs gmu".

rows = [  # gmu, kv_avail_GB(logged), full_tok, swa, c4, c128, c4_state, availmem_GB, vram_used_GB
 (0.80, 91.24, 7053056,1057792,1763264,55102,66112, 67.92, 221.4),
 (0.85,105.08, 8135424,1220096,2033856,63558,76256, 55.95, 233.4),
 (0.88,113.39, 8784896,1317632,2196224,68632,82352, 48.84, 240.5),
 (0.90,118.93, 9217792,1382656,2304448,72014,86416, 44.02, 245.3),
 (0.92,124.47, 9650944,1447424,2412736,75398,90464, 39.31, 250.0),
]
print("gmu, KVpool_GB, KVpool/VRAM%, full_token, swa_tok, c4_tok, c128_tok, c4state_tok, c128state_GB, postpool_headroom_GB, VRAM_used_GB, VRAM_used/total%")
for g,kv,full,swa,c4,c128,c4s,am,vram in rows:
    print(f"{g:.2f}, {kv:6.2f}, {kv/VRAM*100:5.1f}, {full:8d}, {swa:8d}, {c4:8d}, {c128:6d}, {c4s:6d}, 1.01, {am:5.2f}, {vram:6.1f}, {vram/VRAM*100:4.1f}")

print("\n# gmu -> KVpool linear model (5-pt, R^2=1.000):")
print("#   KVpool_GB = 276.9*gmu - 130.3     (per +0.01 gmu: +2.770 GB, +216,470 full_tokens)")
print("#   full_token = KVpool_bytes / 13735.04 ; each pool = fixed fraction of full_token:")
print("#     swa = full*0.15 (swa_full_tokens_ratio) ; c4 = full/4 ; c128 = full/128 ; c4_logical = c128*32 = full/4")
print("# Implied static weights/card = VRAM_total - slope = 288 - 276.9 = 11.1 GiB (DSv4 fp8, TP8)")

print("\n# Runtime verify (Step 3, single-node mix gmu=0.90, real 8k/1k conc64 x128):")
print("#   peak full-attn token usage = 1%   peak swa token usage = 5%   peak #running = 13   RETRACT = 0")
print("#   -> runtime KV usage stays FAR inside the startup-fixed pool; pool is static, gmu is the only knob.")
print("#   -> swa pool (smaller, full*0.15) is the binding one (5% vs 1%), reached first under load.")
