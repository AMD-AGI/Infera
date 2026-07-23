#!/usr/bin/env python3
# Clean analysis of DSv4-Pro sglang 0.5.15 decode gmu sweep — KV pool vs gmu.
# Data hand-transcribed from DP0 raw log lines (all 8 DP ranks identical within rounding).
GiB = 1024**3
VRAM_TOTAL_B = 309220868096       # per card, from rocm-smi (288.0 GiB)
VRAM_TOTAL_GiB = VRAM_TOTAL_B / GiB
BPFT = 13735.04                    # bytes_per_full_token (constant, from log)

# gmu -> (kv_available_GB, full_token, swa, c4, c128, c4_state, c128_state_fixed_GB, avail_mem_GB_postpool, vram_used_GB)
# kv_available_GB / avail_mem / c128_state are GiB as sglang prints (it uses 1<<30).
data = {
 0.80: dict(kv=91.24,  full=7053056, swa=1057792, c4=1763264, c128=55102, c4s=66112, c128fix=1.01, availmem=67.92, vram=221.4),
 0.85: dict(kv=105.08, full=8135424, swa=1220096, c4=2033856, c128=63558, c4s=76256, c128fix=1.01, availmem=55.95, vram=233.4),
 0.88: dict(kv=113.39, full=8784896, swa=1317632, c4=2196224, c128=68632, c4s=82352, c128fix=1.01, availmem=48.84, vram=240.5),
 0.90: dict(kv=118.93, full=9217792, swa=1382656, c4=2304448, c128=72014, c4s=86416, c128fix=1.01, availmem=44.02, vram=245.3),
 0.92: dict(kv=124.47, full=9650944, swa=1447424, c4=2412736, c128=75398, c4s=90464, c128fix=1.01, availmem=39.31, vram=250.0),
}

print(f"VRAM total/card = {VRAM_TOTAL_GiB:.2f} GiB   bytes_per_full_token = {BPFT}\n")
print(f"{'gmu':>4} | {'KVpool GB':>9} | {'full_tok':>9} | {'swa':>8} {'c4':>8} {'c128':>7} | {'KVpool/VRAM':>11} | {'availmem GB':>11} | {'gmu*VRAM GB':>11}")
print("-"*100)
rows=[]
for g in sorted(data):
    d=data[g]
    kv=d['kv']; full=d['full']
    # verify: full_token * BPFT should equal kv_available bytes
    kv_from_full = full*BPFT/GiB
    kvpool_frac = kv/VRAM_TOTAL_GiB
    gmu_vram = g*VRAM_TOTAL_GiB
    rows.append((g,kv,full,kv_from_full,d['swa'],d['c4'],d['c128'],d['availmem'],d['vram'],gmu_vram,kvpool_frac))
    print(f"{g:>4} | {kv:>9.2f} | {full:>9} | {d['swa']:>8} {d['c4']:>8} {d['c128']:>7} | {kvpool_frac*100:>10.1f}% | {d['availmem']:>11.2f} | {gmu_vram:>11.2f}")

print("\n=== Linearity of KV pool vs gmu (per +0.01 gmu) ===")
gs=sorted(data)
for i in range(1,len(gs)):
    g0,g1=gs[i-1],gs[i]
    dkv=data[g1]['kv']-data[g0]['kv']; dg=g1-g0
    dfull=data[g1]['full']-data[g0]['full']
    print(f"  {g0}->{g1}: dKVpool={dkv:.2f}GB  => {dkv/(dg/0.01):.3f} GB per +0.01 gmu | dfull_token={dfull} => {dfull/(dg/0.01):.0f} tok/+0.01")

# slope via endpoints
dkv=data[0.92]['kv']-data[0.80]['kv']; dg=0.92-0.80
slope=dkv/dg
intercept_at_0 = data[0.80]['kv']-slope*0.80
print(f"\n  Linear fit: KVpool_GB ≈ {slope:.2f}*gmu + ({intercept_at_0:.2f})   [slope = weights-free VRAM ~= total-weights]")
print(f"  Implied model-weights/card ≈ VRAM_total - slope-per-unit... slope={slope:.1f}GB/unit gmu; VRAM_total={VRAM_TOTAL_GiB:.1f}")
print(f"  => weights/card ≈ {VRAM_TOTAL_GiB - slope:.1f} GiB (since KVpool = gmu*total - weights - overhead; d(KVpool)/d(gmu)=total only if overhead flat)")

print("\n=== Per-token KV bytes cross-check (decode: SWA c4 pool dominates full-ctx) ===")
# The multi-pool: swa (all layers, window) + c4 (full-attn latent) + c128 (indexer)
# bytes_per_full_token=13735 is the COMPOSITE cost of 1 full-context token across all pools.
# Sanity: a single 9472-ctx request's KV footprint:
ctx=9472
print(f"  1 request @ctx={ctx}: composite KV = {ctx*BPFT/1e6:.1f} MB (across swa+c4+c128, per rank)")
print(f"  full_token capacity @gmu0.90 = {data[0.90]['full']} tok = {data[0.90]['full']/ctx:.0f} full-ctx reqs worth")
