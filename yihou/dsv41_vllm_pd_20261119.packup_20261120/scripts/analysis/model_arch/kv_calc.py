"""KV bytes/token, PD transfer size, decode KV capacity and FLOPs/token arithmetic (DSv4.1-Flash)."""
GiB = 1024**3
# per-entry record sizes (bytes)
fmt = {
  "official (FP4 main 288 B + MXFP4 idx 68 B)": (288, 68),
  "vLLM ROCm default fp8_ds_mla V4 record 584 B + MXFP4 idx 68 B": (584, 68),
  "vLLM ROCm fp8_e4m3 plain 512 B + MXFP4 idx 68 B": (512, 68),
  "vLLM ROCm fp8_ds_mla 584 B + FP8 idx 132 B": (584, 132),
  "bf16 1024 B + FP8 idx 132 B": (1024, 132),
}
entries_per_token = 3 * (1 / 2) + 1 * (1 / 1)  # kv sources: 2,8,14 ratio 2; 20 ratio 1
print("compressed entries per token:", entries_per_token)
swa_layers = 40 + 3  # every backbone layer + 3 DSpark layers keep a 128-token SWA ring
for name, (main, idx) in fmt.items():
    per_tok = entries_per_token * (main + idx)
    swa_rec = 528 if main == 288 else (512 if main == 512 else (1024 if main == 1024 else 584))
    swa_fixed = swa_layers * 128 * swa_rec
    print(f"\n{name}: grow {per_tok:.0f} B/token; SWA fixed {swa_fixed/1e6:.2f} MB/request (128 slots x {swa_layers} layers x {swa_rec} B)")
    for isl in (50_000, 100_000, 200_000):
        print(f"  ISL {isl}: grow {per_tok*isl/1e6:7.1f} MB + SWA {swa_fixed/1e6:.1f} MB = {(per_tok*isl+swa_fixed)/1e6:7.1f} MB")
# decode KV capacity per GPU (upper bound; ignores activations, cudagraph pools, indexer logits buffer)
hbm = 288 * GiB; util = 0.90
w = {"TP1": None, "TP2": 145, "TP4": 81, "TP8": 36}  # GiB resident weights/rank, Engram offloaded (vLLM recipe comments)
full_w_offl = (510.286e9 - 202.758e9 - 0.971e9) / GiB
print(f"\nweights excl. engram+vision: {full_w_offl:.1f} GiB; budget 0.9x288 GiB = {util*288:.1f} GiB")
for tp, wg in w.items():
    if wg is None:
        print(f"{tp}: weights {full_w_offl:.0f} GiB > {util*288:.0f} GiB budget -> does not fit"); continue
    kv = util * hbm - wg * GiB
    for name, (main, idx) in list(fmt.items())[:3]:
        per_tok = entries_per_token * (main + idx)
        toks = kv / per_tok
        print(f"{tp}: KV budget {kv/GiB:.0f} GiB/rank; {name[:30]:30s}: {toks/1e6:6.1f} M tokens/rank = {toks/100_000:6.0f} x 100k-ctx")
# FLOPs per decode token
L = 40; H = 64; D = 512; topk = 512; win = 128
act_decode = 16.13e9; act_prefill = 7.9e9
attn = (L - 2) * H * (win + topk) * D * 4 + 2 * H * win * D * 4
for N in (50_000, 100_000, 200_000):
    idx = 3 * 32 * 128 * (N / 2) * 2 + 32 * 128 * N * 2 + 4 * 32 * 128 * min(16384, N) * 2
    idx_bytes = (3 * N / 2 + N) * 68 + 4 * min(16384, N) * 68
    print(f"N={N}: GEMM {2*act_decode/1e9:.1f} GFLOP + core attn {attn/1e9:.2f} + indexer {idx/1e9:.2f} GFLOP/decode token; indexer K read {idx_bytes/1e6:.1f} MB/token")
# engram lookup bytes per token
print("engram bytes/token:", 2 * 3 * 8 * (256 + 8), "B")
