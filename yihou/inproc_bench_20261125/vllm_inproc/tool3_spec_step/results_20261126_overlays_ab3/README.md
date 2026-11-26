# kf overlays A/B3 with vLLM tool 3 at shape A (run id stepE-overlay-ab3-20261126; 13:08-14:08 UTC, n02-21, ROCR 6,7 = rocm-smi GPU[5]/[4])

- **A**: stock `yihou-inproc-v3xa`, warm cache.
- **B3**: new container `yihou-inproc-v3xe` with a cold cache `cache_v3xE`. Six files are mounted read-only over their
  dist-packages targets: `indexer_4707282` (radix_topk, gluon pa_mqa_logits, triton/host pa_mqa_logits) and
  `sparse_mla_aa2a608` (gluon sparse_mla, sparse_mla_fp8, triton attention/sparse_mla).
- Protocol identical to A/B2 (`../results_20261126_overlays_ab2`): shape A uniform (DP2 TP1, C448, ISL 173000, OSL 1024,
  synthetic 3.51, seed 1234, warmup 10). Order A-B3, B3-A, A-B3, B3-A, then one profiled run per arm (MAX_STEPS=40,
  window steps 20-24, all ranks). Scripts and log: `chain_ab3.sh`, `chain_ab3.out`.

## Gates
- **Module hashes** inside `yihou-inproc-v3xe` match all 6 expected prefixes: 349ef025a759a0dd, 97c23bfaa537b3b1,
  86d26dc7d3e3c4f9, df10bb60a06a90b8, 2fcfde5a7fc768ac, 2ee4344b3dfd5731. A is stock: 37360c50, f5380c2e, db3cdb65,
  d7c7e731, d29d7d16, 156bfbfd.
- **Triton**: 51 cache keys hold `_pa_mqa_logits*` / `_sparse_mla*` kernels. None are shared with A or with B2
  (`triton_keys_E.txt`).
- **FlyDSL**: keys present in B3 but not in A: 3 `launch_radix_topk_one_block` + 1 `launch_topk_per_row_decode_kernel`.
  None of B3's radix_topk keys are shared with B2.
- **Kernel names**: B3 runs `radix_topk_one_block_decode_mixed_k512..._cap_band` (A: no suffix; B and B2: `_cap`).
- **Runs**: all 8 pass `verify_point.py`, with 299 steps, target and draft graph replays 299 on both ranks, and realized
  accept 3.514781414237936 in every run.

## Step time (rank 0 step p50; rank 1 agrees within 0.008 ms)

| round (order) | A ms | B3 ms | B3 vs A |
|---|---|---|---|
| 1 (A, B3) | 103.757 | 102.728 | -0.99 % |
| 2 (B3, A) | 103.692 | 102.849 | -0.81 % |
| 3 (A, B3) | 103.930 | 102.668 | -1.21 % |
| 4 (B3, A) | 103.900 | 102.208 | -1.63 % |
| mean | 103.820 | 102.613 | **-1.16 %** |
| range (spread) | 103.692-103.930 (0.24) | 102.208-102.849 (0.64) | |
| sd | 0.11 | 0.28 | |

- TPOT: A 30.31-30.53 ms, B3 29.96-30.10 ms (`ab3_table.txt`).
- B3 was faster than A in every round, in both orders. The two arms' ranges do not overlap.

## Kernel attribution (profiled runs, ms/step, rank 0 / rank 1; `ab3_groups/`)

| group | A | B3 | B3 - A | A/B2 delta (B2 - A) |
|---|---|---|---|---|
| pa_mqa_logits | 3.617 / 3.679 | 3.397 / 3.469 | -0.215 | -0.249 |
| radix_topk | 1.124 / 1.139 | 0.638 / 0.651 | -0.487 | -0.410 |
| sparse_mla | 4.446 / 4.497 | 3.863 / 3.896 | -0.591 | -0.464 |
| moe stage1 | 21.912 / 21.960 | 21.287 / 21.465 | -0.560 | +0.067 |
| moe stage2 | 11.888 / 11.908 | 11.957 / 11.911 | +0.036 | -0.009 |
| mega_mhc | 4.223 / 4.163 | 4.207 / 4.182 | +0.002 | +0.043 |
| nccl | 25.430 / 24.955 | 24.956 / 25.095 | -0.167 | +0.195 |
| engram | 12.275 / 12.294 | 12.021 / 12.378 | -0.086 | -0.136 |

- **Overlaid groups:** together -1.29 ms of kernel time per step (B2: -1.12 ms). Measured step delta -1.21 ms
  (-1.16 %); B2's was -1.05 ms (-1.01 %), in a different session.
- **Compared with A/B2, per kernel:** B3 saves more on radix_topk (-0.49 vs -0.41 ms/step) and on sparse_mla (-0.59 vs
  -0.46). pa_mqa_logits is about the same (-0.22 vs -0.25).
- **moe stage1 moved -0.56 ms/step in this pair** although no MoE file is overlaid (same kernel name in both arms).
  Single-window profile variation of that size was already seen in A/B/C (rank 0 vs rank 1 differed by 0.5 ms there).
  Treat moe, nccl and engram deltas here as noise, not overlay effect.
- B3 vs B2 was not interleaved in one session; the sessions' A means are 103.82 (A/B3) and 103.88 (A/B2) ms.
