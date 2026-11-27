# kf overlays B3 vs B5 with vLLM tool 3 at shape A (run id stepE-overlay-ab5-20261126; 16:19-17:33 UTC, n02-21, ROCR 6,7 = rocm-smi GPU[5]/[4])

- **B3**: `indexer_4707282` + `sparse_mla_aa2a608` (6 files), container `yihou-inproc-v3xe`, warm cache.
- **B5**: `sparse_mla_aa2a608` (3 files, as B3) + `indexer_0445826` (5 files). New container `yihou-inproc-v3xg`,
  cold cache `cache_v3xG`. Per the team lead's change, `gemm_wq_b_da32733` is not mounted and `indexer_4707282` is not
  mounted.
- Shape A uniform as before (DP2 TP1, C448, ISL 173000, OSL 1024, synthetic 3.51, seed 1234, warmup 10).
- Six rounds with alternating order (B3-B5, B5-B3, ...), then one profiled run per arm. All of it finished by 17:33Z,
  so the 18:25Z skip guard in `chain_ab5.sh` never fired.

## Gates
- **Module hashes**: all 8 in `v3xg` match: sparse_mla df10bb60a06a90b8 / sparse_mla_fp8 2fcfde5a7fc768ac / attention
  sparse_mla 2ee4344b3dfd5731, radix_topk b8bb092590da4de1, topk_per_row_decode 9b63abe1e4a37c8b, topk_per_row (host)
  c85bf177fc144216, gluon pa_mqa_logits 97c23bfaa537b3b1, triton pa_mqa_logits 86d26dc7d3e3c4f9. The GEMM files are stock
  in both containers.
  - B3 and B5 share the same pa_mqa_logits and sparse_mla files. The arms differ only in radix_topk and the two
    topk_per_row files (B3 has stock topk_per_row: cada44bf..., 1047901d...).
- **FlyDSL cache**: 3 `launch_radix_topk_one_block` + 1 `launch_topk_per_row_decode_kernel` keys are new in B5 relative to
  B3, and B3 has its own 3 + 1 that B5 lacks (`flydsl_keys_G.txt` vs `../results_20261126_overlays_ab4/flydsl_keys_E.txt`).
- **Profile kernel names**: radix_topk changes from `..._b1024_w64_v0_s0_cap_band` (B3) to `..._b512_w64_v0_s0_cap_band`
  (B5), for both k512 and k2048. No `topk_per_row` decode kernel appears in either arm's profile window at this shape.
- **Runs**: all 12 pass, 299 steps each, target and draft graph replays 299 on both ranks, realized accept
  3.514781414237936 in every run.

## Step time (rank 0 step p50; rank 1 agrees within 0.010 ms)

| round (order) | B3 ms | B5 ms | B5 vs B3 |
|---|---|---|---|
| 1 (B3, B5) | 102.423 | 102.440 | +0.02 % |
| 2 (B5, B3) | 102.365 | 102.884 | +0.51 % |
| 3 (B3, B5) | 102.378 | 102.354 | -0.02 % |
| 4 (B5, B3) | 102.176 | 102.233 | +0.06 % |
| 5 (B3, B5) | 102.593 | 102.549 | -0.04 % |
| 6 (B5, B3) | 102.460 | 102.172 | -0.28 % |
| mean | 102.399 | 102.439 | **+0.04 %** |
| range (spread) | 102.176-102.593 (0.42) | 102.172-102.884 (0.71) | |
| sd | 0.14 | 0.26 | |

- Paired per-round deltas: mean +0.04 %, sd 0.26 %, t = 0.39 (n = 6). **No step-level difference is resolved.**

## Kernel attribution (profiled runs, ms/step, mean of the two ranks; `ab5_groups/summary.txt`)

| group | B3 | B5 | B5 - B3 |
|---|---|---|---|
| radix_topk | 0.644 | 0.565 | -0.079 |
|   k512 kernel, 8 calls/step (rank 0) | 0.572 | 0.466 | -0.106 |
|   k2048 kernel, 1 call/step (rank 0) | 0.065 | 0.093 | +0.028 |
| pa_mqa_logits | 3.428 | 3.436 | +0.008 |
| sparse_mla | 3.902 | 3.873 | -0.029 |
| moe stage1 / stage2 | 21.661 / 11.953 | 21.681 / 11.975 | +0.020 / +0.021 |
| nccl | 25.107 | 25.145 | +0.038 |
| engram | 12.248 | 12.177 | -0.070 |

- **radix_topk** is the only group the overlay targets that moved: -0.08 ms per step, i.e. 0.08 % of a 102 ms step.
  That is below the run-to-run step noise here (sd 0.14-0.26 ms), which is consistent with the null step result.
- **Other groups** moved by less than 0.1 ms each, within profile-window variation.
