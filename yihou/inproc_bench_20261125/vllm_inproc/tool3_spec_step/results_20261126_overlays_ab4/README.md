# kf overlays B3 vs B4 with vLLM tool 3 at shape A (run id stepE-overlay-ab4-20261126; 15:01-16:17 UTC, n02-21, ROCR 6,7 = rocm-smi GPU[5]/[4])

- **B3**: `indexer_4707282` + `sparse_mla_aa2a608` (6 files), container `yihou-inproc-v3xe` with its warm cache from A/B3.
- **B4**: B3 plus `gemm_wq_b_da32733` (2 files: `aiter/ops/flydsl/kernels/gemm/mxscale_preshuffle.py`,
  `aiter/ops/flydsl/mxscale_preshuffle_kernels.py`). New container `yihou-inproc-v3xf`, cold cache `cache_v3xF`.
- Shape A uniform as before (DP2 TP1, C448, ISL 173000, OSL 1024, synthetic 3.51, seed 1234, warmup 10).
- Six rounds with alternating order (B3-B4, B4-B3, ...), then one profiled run per arm (40 steps, window steps 20-24,
  all ranks). Scripts and log: `chain_ab4.sh`, `chain_ab4.out`.

## Gates
- **Module hashes**: all 8 in `yihou-inproc-v3xf` match (the 6 B3 prefixes plus mxscale_preshuffle e8ce8da140d385be and
  mxscale_preshuffle_kernels 6cd5d471bcf3ddb6). In `v3xe` the two GEMM files are stock (ac505dc4d9bab7a2, ce1cc937ba5008bf).
- **FlyDSL cache** (`flydsl_keys_{E,F}.txt`): B4 has 1 `launch_gemm` + 1 `launch_gemm_fused` key that B3 lacks. B3 has
  2 `launch_gemm` + 1 `launch_gemm_fused` keys that B4 lacks.
- **Profile, wq_b**: the 40-calls-per-step K=1280 MXFP8 GEMM changes kernel:
  `flydsl_bpreshuffle_mxfp8_8w_256x256x128_..._k1280` (B3) becomes `flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32`
  (B4). A 3-calls-per-step variant changes too: `flydsl_mxpsh_192x256x128_..._pf2` becomes `flydsl_mxpsh_192x128x128_...`.
  The trace carries no GEMM shapes. The N=32768 K=1280 attribution rests on the `_k1280` suffix and the 40 calls per
  step (one per layer).
- **Runs**: all 12 pass, 299 steps each, target and draft graph replays 299 on both ranks, realized accept
  3.514781414237936 in every run.

## Step time (rank 0 step p50; rank 1 agrees within 0.008 ms)

| round (order) | B3 ms | B4 ms | B4 vs B3 |
|---|---|---|---|
| 1 (B3, B4) | 102.527 | 101.766 | -0.74 % |
| 2 (B4, B3) | 101.727 | 102.250 | +0.51 % |
| 3 (B3, B4) | 102.236 | 102.372 | +0.13 % |
| 4 (B4, B3) | 102.720 | 102.424 | -0.29 % |
| 5 (B3, B4) | 102.467 | 102.165 | -0.30 % |
| 6 (B4, B3) | 102.493 | 102.010 | -0.47 % |
| mean | 102.362 | 102.165 | **-0.19 %** |
| range (spread) | 101.727-102.720 (0.99) | 101.766-102.424 (0.66) | |
| sd | 0.35 | 0.25 | |

- Paired per-round deltas: mean -0.19 %, sd 0.45 %, t = -1.06 (n = 6). B4 was faster in 4 of 6 rounds. **The difference
  is not resolved at this noise level**; a true effect of about 0.3 % would need roughly 3x more rounds to separate.
- B4r4 had one 160 ms step (step mean 103.75 ms); its p50 was not affected.

## Kernel attribution (profiled runs, ms/step, mean of the two ranks; `ab4_groups/summary.txt`)

| GEMM kernel (non-MoE) | calls/step | B3 | B4 | B4 - B3 |
|---|---|---|---|---|
| wq_b (K=1280): bpreshuffle `_k1280` -> mxpsh 128x256x128 | 40 | 2.949 | 2.977 | +0.028 |
| mxpsh 192x256 (B3) -> 192x128 (B4) | 3 | 0.191 | 0.190 | -0.001 |
| mxpsh 96x256x128 | 43 | 1.742 | 1.687 | -0.055 |
| mxpsh 96x128x128 | 8 | 0.248 | 0.300 | +0.052 |
| hipBLASLt MT224x384 | 1 | 1.094 | 1.301 | +0.207 |
| other bmm_a8w8 / mxpsh / Cijk (unchanged names) | | | | within +/-0.02 each |
| **GEMM family total** | | 16.768 | 16.983 | +0.215 |

| group | B3 | B4 | B4 - B3 |
|---|---|---|---|
| pa_mqa_logits | 3.439 | 3.441 | +0.002 |
| radix_topk | 0.647 | 0.643 | -0.004 |
| sparse_mla | 3.871 | 3.961 | +0.090 |
| moe stage1 | 22.068 | 21.868 | -0.201 |
| nccl | 24.728 | 25.227 | +0.498 |
| engram | 12.248 | 12.198 | -0.050 |

- **wq_b**: the overlay swaps the kernel but the new kernel takes the same time at this shape (2.95 vs 2.98 ms per
  step in one 5-step window each).
- **Other GEMM rows**: the hipBLASLt MT224x384 GEMM (1 call/step) is not touched by the overlay. Its +0.21 ms, and the
  nccl and moe moves, are within the profile-to-profile variation seen in earlier sessions.
- **Shared experts, engram**: the trace does not separate these GEMMs by layer role. The kernels whose names did not
  change moved by at most 0.06 ms/step each.
