# kf combined overlays A/B/C with vLLM tool 3 at shape A (2026-11-26, n02-21, ROCR 6,7)

**Arms**
- **A**: stock image `infera-vllm-dsv41:0007cd8087f1-nightly-9c1d45c2` (id sha256:57ab3dc0...), container
  `yihou-inproc-v3xa`.
- **B**: A plus `indexer_de75215` (3 files) and `sparse_mla_cf7a767` (2 files), each file bind-mounted read-only over
  its target under `/usr/local/lib/python3.12/dist-packages/` (container `yihou-inproc-v3xb`).
- **C**: B plus `moe_488c3ad` (3 files) (container `yihou-inproc-v3xc`).

**Setup**
- Each arm has its own cache dir (`cache_v3x{A,B,C}`: TRITON_CACHE_DIR=/cache/triton, FLYDSL_RUNTIME_CACHE_DIR=/cache/flydsl,
  AITER_JIT_DIR=/cache/aiter_jit), all cold at start.
- Shape A uniform: DP2 TP1, C448 (224/rank), ISL 173000, OSL 1024, DSpark 5 synthetic 3.51, seed 1234, warmup 10,
  random KV fill, llying decode preset.
- Order A B C, repeated 4 times, run sequentially on the same GPUs (`chain_abc.sh`, `chain_abc.out`). Then one
  profiled run per arm (40 steps, window steps 20-24, both ranks).

**GPUs**: ROCR_VISIBLE_DEVICES 6,7 are rocm-smi GPU[5] and GPU[4]. A 60 GiB allocation on each container device showed
up on those two GPUs. kf's containers use HIP devices 0-3 and 5; the tuner on HIP 5 shows up as rocm-smi GPU[6].

## Engagement gates

- **Module hashes** (sha256[:16] of each overlaid module file inside the container): B and C match all expected
  prefixes for indexer + sparse_mla (3cdc117d, 14d077ba, 426c5a45, a3403949, da3c7e8c). C also matches the moe
  prefixes (0d72b4cb, c44384a1, 41af698c). B keeps the stock moe hashes (c317226d, 5b51a36a, bfd55e35). A is stock
  throughout.
- **Compile caches after round 1**:
  - The 47 Triton cache keys that hold `_pa_mqa_logits*` / `_sparse_mla*` kernels are all different between A and B
    (0 in common; `triton_keys_*.txt`).
  - FlyDSL keys present in B but not in A: 3 `launch_radix_topk_one_block` + 1 `launch_topk_per_row_decode_kernel`.
  - FlyDSL keys present in C but not in B: 16 `launch_mixed_moe_gemm1` + 1 `launch_gemm2` (`flydsl_keys_*.txt`).
- **Kernel names in the profiles**: B and C run `radix_topk_one_block_decode_mixed_k512..._cap` (A runs the version
  without `_cap`). C runs `mfma_moe1_..._ntdeal1_v2out_v33` (A and B run `..._gui_v2out_v33`).

## Runs (all 12 pass `verify_point.py`; 299 steps each; target and draft graph replays == 299 on both ranks; realized accept 3.514781414237936 in every run)

| round | A step p50 ms | B step p50 ms | C step p50 ms | B vs A | C vs A |
|---|---|---|---|---|---|
| 1 | 102.865 | 101.868 | 102.335 | -0.97 % | -0.52 % |
| 2 | 103.551 | 102.362 | 102.779 | -1.15 % | -0.75 % |
| 3 | 103.776 | 102.686 | 102.385 | -1.05 % | -1.34 % |
| 4 | 103.817 | 102.530 | 102.515 | -1.24 % | -1.25 % |
| mean | 103.502 | 102.361 | 102.503 | **-1.10 %** | **-0.97 %** |
| range (spread) | 102.865-103.817 (0.95) | 101.868-102.686 (0.82) | 102.335-102.779 (0.44) | | |
| sd | 0.44 | 0.35 | 0.20 | | |

- Step p50 is rank 0; rank 1 agrees within 0.012 ms in every run. Per-run step mean, max step and TPOT are in
  `abc_table.txt`. TPOT: A 30.13-30.45 ms, B 29.80-30.08 ms, C 29.99-30.10 ms.
- B was faster than A in all 4 rounds, and C was faster than A in all 4 rounds. Arm means differ by more than either
  arm's range, but A drifts upward across rounds (102.87 -> 103.82).
- C vs B, per round: +0.46, +0.41, -0.29, -0.01 %. This is inside the noise; the moe overlay's step effect is not
  resolved at this shape.

## Kernel attribution (profiled runs, ms per step, rank 0 / rank 1; `abc_groups/*.json`, `abc_groups/summary.txt`)

| group | A | B | C | B - A (mean of ranks) | C - A |
|---|---|---|---|---|---|
| pa_mqa_logits | 3.641 / 3.689 | 3.440 / 3.451 | 3.418 / 3.455 | -0.220 | -0.229 |
| radix_topk | 1.124 / 1.124 | 0.714 / 0.715 | 0.719 / 0.719 | -0.409 | -0.405 |
| sparse_mla | 4.428 / 4.478 | 4.247 / 4.239 | 4.232 / 4.248 | -0.210 | -0.213 |
| moe stage1 | 22.001 / 21.800 | 21.893 / 21.799 | 21.422 / 21.803 | -0.055 | -0.289 |
| moe stage2 | 12.015 / 11.984 | 12.015 / 11.587 | 12.077 / 11.629 | -0.199 | -0.146 |
| moe sorting | 1.034 / 1.037 | 1.038 / 1.033 | 1.030 / 1.037 | 0.000 | -0.002 |
| mega_mhc | 4.174 / 4.185 | 4.204 / 4.157 | 4.196 / 4.168 | +0.001 | +0.002 |
| nccl | 24.949 / 24.908 | 24.901 / 25.332 | 25.297 / 25.410 | +0.188 | +0.425 |
| engram | 11.499 / 11.292 | 11.182 / 10.948 | 11.495 / 11.026 | -0.331 | -0.135 |
| other | 29.378 / 29.789 | 29.256 / 29.578 | 29.371 / 29.704 | -0.166 | -0.046 |

- **Indexer and sparse MLA (B vs A):** the three overlaid kernel groups together save 0.84 ms of kernel time per
  step (pa_mqa_logits -0.22, radix_topk -0.41, sparse_mla -0.21). The measured step delta is -1.14 ms (-1.10 %).
- **MoE (C vs B):** the main stage-1 kernel is 20.707 ms with the overlay vs 21.198 ms without on rank 0, but on
  rank 1 it is 21.803 vs 21.799. Each profile is a single 5-step window; the per-rank difference is not explained.
- Groups sum to more than the step time because kernels run on several streams at once. Each profile is a single
  5-step window, so per-kernel differences below ~0.2 ms are within profile-to-profile variation (nccl and engram
  move by that much without any overlay touching them).
