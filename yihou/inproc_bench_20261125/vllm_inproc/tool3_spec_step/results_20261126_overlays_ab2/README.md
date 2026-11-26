# kf overlays A/B2 with vLLM tool 3 at shape A (2026-11-26 09:50-10:47 UTC, n02-21, ROCR 6,7 = rocm-smi GPU[5]/[4])

- **A**: stock, container `yihou-inproc-v3xa`. It reuses its warm cache from the A/B/C run.
- **B2**: `indexer_656b7f4` (3 files) + `sparse_mla_dc44081` (3 files, including the new host file
  `aiter/ops/triton/attention/sparse_mla.py`). Each file is mounted read-only over its target under dist-packages.
  Container `yihou-inproc-v3xd`, cold cache `cache_v3xD`.
- Same shape and gates as `../results_20261126_overlays_abc` (DP2 TP1, C448, ISL 173000, OSL 1024, synthetic 3.51,
  seed 1234, warmup 10).
- Order alternates per round: A-B2, B2-A, A-B2, B2-A (`chain_ab2.sh`, `chain_ab2.out`). Then one profiled run per arm.

## Engagement gates
- **Module hashes** inside `yihou-inproc-v3xd`: all 6 match the expected prefixes (radix_topk 3cdc117db5b3e6ef,
  gluon pa_mqa_logits 3d98633a6ade92b9, triton pa_mqa_logits 7bec7de1a012042a, gluon sparse_mla a237ae28f0e68dad,
  sparse_mla_fp8 db42bc18e78e43a4, triton attention/sparse_mla be1bf700263bbea5). In A they are stock (attention/sparse_mla
  156bfbfd31eecd4a).
- **Triton cache**: the 47 keys holding `_pa_mqa_logits*` / `_sparse_mla*` kernels have none in common with A or with
  the earlier B arm (`triton_keys_D.txt`).
- **FlyDSL cache**: keys present in B2 but not in A: 3 `launch_radix_topk_one_block` + 1
  `launch_topk_per_row_decode_kernel`.
- **Profile**: B2 runs `radix_topk_one_block_decode_mixed_k512..._cap`.

## Runs (all 8 pass; 299 steps; target and draft replays 299 on both ranks; realized accept 3.514781414237936 in every run)

| round (order) | A step p50 ms | B2 step p50 ms | B2 vs A |
|---|---|---|---|
| 1 (A, B2) | 103.447 | 102.450 | -0.96 % |
| 2 (B2, A) | 104.092 | 102.833 | -1.21 % |
| 3 (A, B2) | 104.156 | 103.071 | -1.04 % |
| 4 (B2, A) | 103.823 | 102.950 | -0.84 % |
| mean | 103.880 | 102.826 | **-1.01 %** |
| range (spread) | 103.447-104.156 (0.71) | 102.450-103.071 (0.62) | |
| sd | 0.32 | 0.27 | |

- Step p50 is rank 0; rank 1 agrees within 0.007 ms. TPOT: A 30.34-30.49 ms, B2 29.94-30.19 ms (`ab2_table.txt`).
- B2 was faster than A in every round, in both run orders. The two arms' ranges do not overlap.
- A's mean here (103.88 ms) is 0.38 ms above its mean in the A/B/C session (103.50 ms). Compare arms within a session
  only.

## Kernel attribution (profiled runs, ms/step, rank 0 / rank 1; `ab2_groups/`)

| group | A | B2 | B2 - A (mean of ranks) |
|---|---|---|---|
| pa_mqa_logits | 3.652 / 3.676 | 3.406 / 3.423 | -0.249 |
| radix_topk | 1.125 / 1.133 | 0.720 / 0.718 | -0.410 |
| sparse_mla | 4.408 / 4.492 | 3.976 / 3.995 | -0.464 |
| moe stage1 | 21.418 / 21.827 | 21.407 / 21.972 | +0.067 |
| moe stage2 | 11.660 / 11.948 | 11.645 / 11.946 | -0.009 |
| mega_mhc | 4.198 / 4.155 | 4.207 / 4.233 | +0.043 |
| nccl | 25.732 / 24.421 | 25.944 / 24.599 | +0.195 |
| engram | 12.296 / 12.477 | 12.235 / 12.265 | -0.136 |

- The three overlaid groups together: -1.12 ms of kernel time per step. Measured step delta: -1.05 ms (-1.01 %).
- Against the earlier B arm (indexer_de75215 + sparse_mla_cf7a767, other session):
  - sparse_mla kernel saving: B2 -0.46 ms/step vs B -0.21 ms/step.
  - pa_mqa_logits and radix_topk savings: about the same in both.
  - Step deltas vs their own A: B2 -1.01 %, B -1.10 %.

  The larger sparse_mla kernel saving did not show up as a larger step saving. The two sessions were not
  interleaved with each other, so B vs B2 is not compared directly here.
