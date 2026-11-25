# Shape A top kernels: vLLM tool 3, profiled run p_v3_A_u173k_graphon (2026-11-25, n03-33 GPUs 0,1)

Run: `bench/profile_decode.py` through `scripts/run_profile.sh` (MAX_STEPS=40, PROFILE_START=20, PROFILE_NUM=5,
PROFILE_RANKS=all, CAPTURE_PROFILE=1, graph ON). DP2 TP1, llying decode preset (aiter MoE, fp8_e4m3 KV, mxfp4 indexer,
FULL_DECODE_ONLY), C448 = 224 requests per DP rank, uniform ISL 173000, DSpark K=5, synthetic acceptance 3.51,
random KV fill. Window = measured steps 20-24 (5 steps), both ranks. A profiled run is not a performance
measurement. The unprofiled twin is v3_A_u173k_c448: step p50 104.30 ms.

How the numbers are made (`profile_shapeA/topk_from_trace.py`, from the chrome traces in `profile_shapeA/`): kernel
events (`cat == kernel`) in the window. ms/step = summed kernel duration / 5. "Share" = ms/step divided by the GPU
busy time per step (the union of all kernel intervals, 103.68 ms on rank 0 and 103.25 ms on rank 1). Kernels run on
3-4 streams, so the shares add up to more than 100 % (the kernel sum is 116.68 / 116.37 ms per step). The window
spans 107.62 / 107.57 ms per step from first kernel to last. Calls/step = count / 5.

| # | kernel | rank 0 ms/step | rank 1 ms/step | share of busy (max rank) | calls/step r0 / r1 |
|---|---|---|---|---|---|
| 1 | `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>) [clone .kd]` | 25.898 | 25.760 | 25.0 % | 86.0 / 86.0 |
| 2 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t64x128x256_pm1_fp8q_sort_async_gui_v2out_v33.kd` | 20.595 | 20.652 | 19.9 % | 40.0 / 40.0 |
| 3 | `_engram_lookup_kernel.kd` | 12.330 | 12.385 | 11.9 % | 2.0 / 2.0 |
| 4 | `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16lds_v2_` | 11.864 | 11.773 | 11.4 % | 40.0 / 40.0 |
| 5 | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at` | 6.050 | 6.125 | 5.9 % | 2.0 / 2.0 |
| 6 | `_sparse_mla_fp8_BLOCK_H_64_ROPE_TAIL_0_LAZY_LOG2_8_ADDR64_1_HEAD_BLOCKS_1.kd` | 4.497 | 4.464 | 4.3 % | 43.0 / 43.0 |
| 7 | `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_4x8_row_wb32x32_k1280.kd` | 2.986 | 2.954 | 2.9 % | 40.0 / 40.0 |
| 8 | `bmm_a8w8_mxscale_t128x256x128_w1x4_nb4_sk1_bd2_xcd_b1_s32_32x32.kd` | 2.715 | 2.680 | 2.6 % | 40.0 / 40.0 |
| 9 | `aiter mega_mhc_kernel<5120,10,false,true,false,16,0,2>` | 2.147 | 2.131 | 2.1 % | 43.0 / 43.0 |
| 10 | `bmm_a8w8_mxscale_t192x256x128_w1x4_nb4_sk1_bd2_xcd_b8_s32_32x32.kd` | 2.039 | 2.017 | 2.0 % | 40.0 / 40.0 |
| 11 | `aiter mega_mhc_kernel<5120,10,true,true,false,16,0,2>` | 1.980 | 1.976 | 1.9 % | 39.0 / 39.0 |
| 12 | `flydsl_mxpsh_96x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf3.kd` | 1.761 | 1.736 | 1.7 % | 43.0 / 43.0 |
| 13 | `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_128_BLOCK_KV_64_BLOCK_M_1_PRES...` | 1.434 | 1.508 | 1.5 % | 4.0 / 4.0 |
| 14 | `vllm::deepseek_v4_fused_ops::fusedDeepseekV4FullCacheKernel<c10::BFloat16, true, true, ...>` | 1.432 | 1.419 | 1.4 % | 43.0 / 43.0 |
| 15 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT448x192x64_MI16x16x1_...` (hipBLASLt) | 1.336 | 1.260 | 1.3 % | 1.0 / 1.0 |
| 16 | `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_64_BLOCK_KV_64_BLOCK_M_6_PRESH...` | 1.301 | 1.305 | 1.3 % | 3.0 / 3.0 |
| 17 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x384x64_MI16x16x1_...` (hipBLASLt) | 1.121 | 1.063 | 1.1 % | 1.0 / 1.0 |
| 18 | `bmm_a8w8_mxscale_t96x128x128_w1x4_nb4_sk1_bd2_xcd_b1_s32_32x32.kd` | 1.094 | 1.084 | 1.1 % | 43.0 / 43.0 |
| 19 | `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0.kd` | 1.059 | 1.061 | 1.0 % | 8.0 / 8.0 |
| 20 | `flydsl_mxpsh_128x128x128_F8_F8_B16_w4_x4_sk1_slrr_wb32x32_pf3.kd` | 0.951 | 0.945 | 0.9 % | 40.0 / 40.0 |

Full top-30 per rank, with exact names and both share columns: `profile_shapeA/decode-*.topk.json`.

## Engram lookup

- `_engram_lookup_kernel`: 2 launches per step, 6.07-6.31 ms each, **12.33 ms/step (rank 0) / 12.39 ms/step (rank 1)**.
  In every step one launch is on the main compute stream and one on a side stream (rank 0: streams 4 and 2; rank 1:
  streams 2 and 8). The main stream carries 7,480 / 7,460 of the window's kernels.
- Overlap (`profile_shapeA/engram_overlap.json`): kernels that are not engram kernels, on other streams, cover
  96.7-97.8 % of each lookup's duration, on both ranks, for both the main-stream and the side-stream launch. On its own
  stream a lookup is followed by `aiter::opus_moe_sorting_entry` (main-stream launch) or a hipBLASLt GEMM (side-stream
  launch). The trace shows the interval overlap only. Whether the main stream waits on the lookup before continuing,
  and therefore whether the lookup lengthens the critical path, was not measured.
- `_fused_engram_post_wkv_kernel`: 2 launches per step, about 0.034 ms each, on the main stream; no overlap.
- Reference: llying's serving decode trace (`dsv41_vllm_pd_20261119/results/llying_repro_steplog_20261124/e2e_trace/trace-d`)
  reports `_engram_lookup_kernel` at 4.479 ms average per call. Here it is about 6.2 ms per call. Batch, context and
  window differ between the two runs, so the numbers are not directly comparable.

## Stage timer (CUDA events around the bench's call sites, 40 measured steps, rank 0)

target_forward (FULL graph replay) 94.18 ms/step on average (3.767 s / 40), target_sample (target sampling +
synthetic rejection) 1.75 ms/step, draft_propose (DSpark drafter, its own FULL graph) 6.78 ms/step
(`profile_shapeA/device_timer_rank_*_yihou.json`).

Unexplained: kernel #5 (`at::native::elementwise_kernel_manual_unroll`, 2 calls per step, about 3 ms each) was not
attributed to a call site.
