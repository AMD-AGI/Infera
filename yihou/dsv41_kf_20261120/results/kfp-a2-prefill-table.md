# Calibrated-profiler hot-op tables (vLLM engines)

ms = summed kernel durations (streams overlap -> shares are of GPU kernel time, not wall time).

## role prefill: 36690.3 kernel-ms in the windows; mHC bucket rank 7, first mHC kernel rank 12

stage kernel-ms: MIXED 36381.9, SAMPLE_after_MIXED 308.4

| bucket | kernel ms | share |
|---|---|---|
| comm | 10658.51 | 29.0% |
| moe | 5651.49 | 15.4% |
| attention | 5603.84 | 15.3% |
| quant | 4987.49 | 13.6% |
| gemm | 4090.30 | 11.1% |
| misc | 2772.44 | 7.6% |
| mhc | 1657.73 | 4.5% |
| indexer | 725.69 | 2.0% |
| norm_rope_elem | 541.14 | 1.5% |
| sampling | 1.67 | 0.0% |

| # | kernel | ms | share | bucket |
|---|---|---|---|---|
| 1 | `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 10658.51 | 29.05% | comm |
| 2 | `_Z45opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernelI45opus_mla_v4_prefill_a16w16_16mx8_32nx1_traitsILi` | 5535.15 | 15.09% | attention |
| 3 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 5211.56 | 14.20% | moe |
| 4 | `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf16lds_v2_b` | 2892.99 | 7.88% | gemm |
| 5 | `_inverse_rope_mxfp8_quant_kernel` | 1420.84 | 3.87% | quant |
| 6 | `flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32` | 1085.72 | 2.96% | misc |
| 7 | `bmm_a8w8_mxscale_t256x256x128_w1x4_nb2_sk1_bd1_xcd_b8_s32_32x32` | 836.48 | 2.28% | gemm |
| 8 | `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_2x8_row_wb32x32_k8192` | 782.55 | 2.13% | quant |
| 9 | `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_64_BLOCK_KV_64_BLOCK_M_3_PRESHUFFLE` | 710.02 | 1.94% | indexer |
| 10 | `_engram_lookup_kernel` | 660.08 | 1.80% | misc |
| 11 | `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFloat16ELi64` | 647.64 | 1.77% | quant |
| 12 | `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb0ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16bS3_PKfS5_` | 647.09 | 1.76% | mhc |
| 13 | `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb1ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16bS3_PKfS5_` | 625.13 | 1.70% | mhc |
| 14 | `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_4x4_row_wb32x32_k5120` | 511.02 | 1.39% | quant |
| 15 | `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 331.99 | 0.90% | misc |
| 16 | `_dequantize_and_gather_k_one_pass_kernel` | 331.57 | 0.90% | quant |
| 17 | `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x8_row_wb32x32_k6144` | 313.72 | 0.86% | quant |
| 18 | `_kernel` | 274.87 | 0.75% | misc |
| 19 | `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x4_row_wb32x32_k2304` | 272.28 | 0.74% | quant |
| 20 | `mhc_pre_big_fuse_tilelang_kernel` | 267.78 | 0.73% | mhc |
| 21 | `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k5120` | 250.99 | 0.68% | quant |
| 22 | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi32ELb0ELi64ELb1ELb0ELi0ELb0ELb0ELb0` | 185.94 | 0.51% | quant |
| 23 | `kernel` | 149.38 | 0.41% | misc |
| 24 | `_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_15CUDAFunctor_addIN3c108BFloat16EEESt5arrayIPcL` | 129.39 | 0.35% | norm_rope_elem |
| 25 | `_ZN5aiter22opus_moe_sorting_entryINS_30MoeSortingMultiPhaseKernel_P23INS_19MoeSortingProblemMpIifhLi` | 127.88 | 0.35% | moe |
| 26 | `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT192x128x64_MI16x16x1_SN_LDSB1_AFC0_AG0_AGGSUA0_AGNT` | 118.18 | 0.32% | gemm |
| 27 | `_ZN4vllm3moe22topkGatingSoftplusSqrtILi6ELi384ELi4ELi8ELi64ELb0EiifEEvPKT7_PKbPfiPT5_PiiiibdPKfPKT6_` | 104.11 | 0.28% | moe |
| 28 | `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFloat16ELi64` | 95.23 | 0.26% | quant |
| 29 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 87.76 | 0.24% | moe |
| 30 | `__amd_rocclr_copyBuffer` | 78.80 | 0.21% | norm_rope_elem |

## prefill0 pid 680 dp_rank 1: window 9572 ms (steps), 39559 dispatches, unattributed 0 (0.0 ms), outside steps 11 (0.0 ms), missing symbols 0, causality violations 0
### MIXED: 20 records; reqs 2.0[2-3] tokens 16384.0[16384-16384] prefill_tokens 16384.0[16384-16384] prefill_reqs 2.0[2-3] spec_tokens 0.0[0-0] max_q 8192.0[8192-8192]; GPU span median 449.5821935; kernel ms/record 461.56; graph-replayed 10648/38808
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 135.249 | 2070.6 | comm | (2560, 1, 1) (3072, 1, 1) (7680, 1, 1) |
| `_Z45opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernelI45opus_mla_v4_prefill_a16w16_16mx8_32nx1` | 20.0 | 69.509 | 3705.4 | attention | (8388608, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 20.0 | 65.659 | 3279.0 | moe | (4608, 1920, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 20.0 | 36.293 | 1812.7 | gemm | (19630080, 1, 1) |
| `_inverse_rope_mxfp8_quant_kernel` | 22.0 | 17.938 | 894.2 | quant | (32768, 64, 1) (33280, 64, 1) (4194304, 64, 1) |
| `flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32` | 20.0 | 13.636 | 667.9 | misc | (32768, 128, 1) |
| `bmm_a8w8_mxscale_t256x256x128_w1x4_nb2_sk1_bd1_xcd_b8_s32_32x32` | 20.0 | 10.432 | 522.6 | gemm | (524288, 1, 1) |
| `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_64_BLOCK_KV_64_BLOCK_M_3_` | 25.6 | 10.245 | 428.0 | indexer | (101952, 1, 7) (11136, 1, 18) (11712, 1, 17) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_2x8_row_wb32x32_k8192` | 20.0 | 9.890 | 494.8 | quant | (655360, 1, 1) |
| `_engram_lookup_kernel` | 2.0 | 9.211 | 4646.0 | misc | (65536, 1, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 8.122 | 198.6 | quant | (34078720, 1, 1) (4352, 1, 1) (532480, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb0ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 20.0 | 8.052 | 401.7 | mhc | (262144, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb1ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 18.0 | 7.772 | 431.4 | mhc | (262144, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_4x4_row_wb32x32_k5120` | 20.0 | 6.423 | 319.8 | quant | (589824, 1, 1) |
| `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 26.6 | 4.554 | 185.6 | misc | (1045504, 1, 1) (1079296, 1, 1) (1115136, 1, 1) |
| `_dequantize_and_gather_k_one_pass_kernel` | 42.0 | 4.130 | 62.8 | quant | (256, 128, 1) (256, 8192, 1) (512, 128, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x8_row_wb32x32_k6144` | 2.0 | 3.977 | 1990.9 | quant | (3276800, 1, 1) |
| `mhc_pre_big_fuse_tilelang_kernel` | 1.0 | 3.759 | 4159.9 | mhc | (1572864, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x4_row_wb32x32_k2304` | 20.0 | 3.424 | 170.9 | quant | (655360, 1, 1) |
| `_kernel` | 131.0 | 3.066 | 11.2 | misc | (128, 10, 1) (128, 16, 1) (128, 3, 1) |

### SAMPLE_after_MIXED: 20 records; ; GPU span median 3.892193; kernel ms/record 3.87; graph-replayed 0/740
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 3.0 | 1.203 | 401.4 | quant | (34078720, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k15360` | 1.0 | 0.787 | 787.8 | quant | (655360, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_11FillFunctorIN3c108BFloat16EEESt5arr` | 3.0 | 0.431 | 143.9 | norm_rope_elem | (67108864, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k5120` | 3.0 | 0.378 | 125.8 | quant | (229376, 1, 1) |
| `_ZN2at6native12_GLOBAL__N_126CatArrayBatchedCopy_contigINS1_10OpaqueTypeILj2EEEjLi2ELi128E` | 1.0 | 0.365 | 365.7 | norm_rope_elem | (2097152, 3, 1) |
| `_kernel` | 4.0 | 0.296 | 53.6 | misc | (524288, 10, 1) (524288, 30, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x128_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGS` | 1.0 | 0.192 | 191.8 | gemm | (129280, 1, 1) |
| `__amd_rocclr_copyBuffer` | 3.0 | 0.072 | 4.6 | norm_rope_elem | (131072, 1, 1) (512, 1, 1) |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb0ELb0ELb1ELi1ELb0EEEvPT0_PT_PfS` | 1.0 | 0.051 | 50.9 | quant | (4194304, 1, 1) |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi64ELi8ELb0ELb0ELb1ELi1ELb0EEEvPT0_PT_PfS4_` | 3.0 | 0.021 | 7.1 | quant | (1048576, 1, 1) |
| `_prepare_dflash_inputs_kernel` | 3.0 | 0.018 | 5.5 | attention | (8704, 65, 1) (8960, 65, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_ker` | 2.0 | 0.009 | 4.5 | norm_rope_elem | (128, 1, 1) |
| `_gumbel_sample_kernel` | 1.0 | 0.006 | 6.4 | sampling | (512, 127, 1) (768, 127, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_ker` | 1.0 | 0.006 | 5.6 | norm_rope_elem | (128, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIiEESt5arrayIPcLm1EEEEvi` | 1.0 | 0.006 | 5.6 | norm_rope_elem | (256, 1, 1) |
| `_ZN2at6native24vectorized_gather_kernelILi16ElEEvPcS2_PT0_illllb` | 1.0 | 0.005 | 5.0 | misc | (512, 3, 1) (768, 3, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIlEESt5arrayIPcLm1EEEEvi` | 1.0 | 0.005 | 4.8 | norm_rope_elem | (256, 1, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_21index_put` | 1.0 | 0.005 | 4.8 | norm_rope_elem | (128, 1, 1) |
| `_gumbel_combine_kernel` | 1.0 | 0.005 | 4.8 | sampling | (512, 1, 1) (768, 1, 1) |
| `_get_num_sampled_and_rejected_kernel` | 1.0 | 0.004 | 4.2 | sampling | (512, 1, 1) (768, 1, 1) |

## prefill0 pid 681 dp_rank 0: window 9549 ms (steps), 40530 dispatches, unattributed 0 (0.0 ms), outside steps 2 (0.0 ms), missing symbols 0, causality violations 0
### MIXED: 20 records; reqs 2.0[2-3] tokens 16384.0[16384-16384] prefill_tokens 16384.0[16384-16384] prefill_reqs 2.0[2-3] spec_tokens 0.0[0-0] max_q 8192.0[8192-8192]; GPU span median 448.92368650000003; kernel ms/record 453.68; graph-replayed 10648/39788
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 135.984 | 1766.1 | comm | (2560, 1, 1) (3072, 1, 1) (7680, 1, 1) |
| `_Z45opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernelI45opus_mla_v4_prefill_a16w16_16mx8_32nx1` | 20.0 | 68.576 | 3652.4 | attention | (8388608, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 20.0 | 64.429 | 3217.3 | moe | (4608, 1920, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 20.0 | 35.978 | 1797.4 | gemm | (19630080, 1, 1) |
| `_inverse_rope_mxfp8_quant_kernel` | 22.0 | 17.928 | 894.2 | quant | (33024, 64, 1) (33280, 64, 1) (4194304, 64, 1) |
| `flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32` | 20.0 | 13.521 | 662.6 | misc | (32768, 128, 1) |
| `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_64_BLOCK_KV_64_BLOCK_M_3_` | 29.4 | 11.466 | 408.6 | indexer | (105216, 1, 7) (107776, 1, 7) (112960, 1, 2) |
| `bmm_a8w8_mxscale_t256x256x128_w1x4_nb2_sk1_bd1_xcd_b8_s32_32x32` | 20.0 | 10.432 | 520.9 | gemm | (524288, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_2x8_row_wb32x32_k8192` | 20.0 | 9.691 | 484.8 | quant | (655360, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb0ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 20.0 | 8.092 | 404.6 | mhc | (262144, 1, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 8.082 | 198.2 | quant | (34078720, 1, 1) (4352, 1, 1) (532480, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb1ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 18.0 | 7.842 | 435.9 | mhc | (262144, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_4x4_row_wb32x32_k5120` | 20.0 | 6.301 | 313.8 | quant | (589824, 1, 1) |
| `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 30.9 | 5.600 | 190.2 | misc | (1000448, 1, 1) (1008640, 1, 1) (1046528, 1, 1) |
| `_dequantize_and_gather_k_one_pass_kernel` | 42.0 | 4.944 | 58.5 | quant | (512, 128, 1) (512, 2810, 1) (512, 3322, 1) |
| `_engram_lookup_kernel` | 2.0 | 4.026 | 2008.2 | misc | (65536, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x8_row_wb32x32_k6144` | 2.0 | 3.868 | 1933.8 | quant | (3276800, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x4_row_wb32x32_k2304` | 20.0 | 3.355 | 167.5 | quant | (655360, 1, 1) |
| `_kernel` | 131.0 | 3.275 | 11.7 | misc | (128, 10, 1) (128, 16, 1) (128, 3, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k5120` | 21.0 | 2.796 | 125.8 | quant | (229376, 1, 1) |

### SAMPLE_after_MIXED: 20 records; ; GPU span median 3.852633; kernel ms/record 3.85; graph-replayed 0/740
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 3.0 | 1.201 | 400.9 | quant | (34078720, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k15360` | 1.0 | 0.781 | 781.7 | quant | (655360, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_11FillFunctorIN3c108BFloat16EEESt5arr` | 3.0 | 0.433 | 144.4 | norm_rope_elem | (67108864, 1, 1) |
| `_ZN2at6native12_GLOBAL__N_126CatArrayBatchedCopy_contigINS1_10OpaqueTypeILj2EEEjLi2ELi128E` | 1.0 | 0.374 | 373.3 | norm_rope_elem | (2097152, 3, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k5120` | 3.0 | 0.365 | 121.9 | quant | (229376, 1, 1) |
| `_kernel` | 4.0 | 0.296 | 53.5 | misc | (524288, 10, 1) (524288, 30, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x128_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGS` | 1.0 | 0.191 | 191.3 | gemm | (129280, 1, 1) |
| `__amd_rocclr_copyBuffer` | 3.0 | 0.069 | 4.7 | norm_rope_elem | (131072, 1, 1) (512, 1, 1) |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb0ELb0ELb1ELi1ELb0EEEvPT0_PT_PfS` | 1.0 | 0.052 | 52.6 | quant | (4194304, 1, 1) |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi64ELi8ELb0ELb0ELb1ELi1ELb0EEEvPT0_PT_PfS4_` | 3.0 | 0.021 | 7.0 | quant | (1048576, 1, 1) |
| `_prepare_dflash_inputs_kernel` | 3.0 | 0.019 | 6.1 | attention | (8704, 65, 1) (8960, 65, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_ker` | 2.0 | 0.009 | 4.5 | norm_rope_elem | (128, 1, 1) |
| `_gumbel_sample_kernel` | 1.0 | 0.006 | 6.5 | sampling | (512, 127, 1) (768, 127, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_ker` | 1.0 | 0.006 | 5.6 | norm_rope_elem | (128, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIiEESt5arrayIPcLm1EEEEvi` | 1.0 | 0.006 | 5.5 | norm_rope_elem | (256, 1, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_21index_put` | 1.0 | 0.005 | 4.7 | norm_rope_elem | (128, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIlEESt5arrayIPcLm1EEEEvi` | 1.0 | 0.005 | 4.8 | norm_rope_elem | (256, 1, 1) |
| `_gumbel_combine_kernel` | 1.0 | 0.005 | 4.7 | sampling | (512, 1, 1) (768, 1, 1) |
| `_get_num_sampled_and_rejected_kernel` | 1.0 | 0.004 | 4.1 | sampling | (512, 1, 1) (768, 1, 1) |
| `_post_update_kernel` | 1.0 | 0.003 | 3.2 | misc | (128, 1, 1) (192, 1, 1) |

## prefill1 pid 1535 dp_rank 1: window 9356 ms (steps), 37282 dispatches, unattributed 0 (0.0 ms), outside steps 2 (0.0 ms), missing symbols 0, causality violations 0
### MIXED: 20 records; reqs 2.0[2-3] tokens 16384.0[16384-16384] prefill_tokens 16384.0[16384-16384] prefill_reqs 2.0[2-3] spec_tokens 0.0[0-0] max_q 8192.0[8192-8192]; GPU span median 437.437552; kernel ms/record 451.87; graph-replayed 10650/36540
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 133.676 | 1954.7 | comm | (2560, 1, 1) (3072, 1, 1) (7680, 1, 1) |
| `_Z45opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernelI45opus_mla_v4_prefill_a16w16_16mx8_32nx1` | 20.0 | 69.138 | 3676.2 | attention | (8388608, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 20.0 | 65.128 | 3250.9 | moe | (4608, 1920, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 20.0 | 36.139 | 1806.4 | gemm | (19630080, 1, 1) |
| `_inverse_rope_mxfp8_quant_kernel` | 22.0 | 17.616 | 877.5 | quant | (33024, 64, 1) (4194304, 64, 1) |
| `flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32` | 20.0 | 13.620 | 670.6 | misc | (32768, 128, 1) |
| `bmm_a8w8_mxscale_t256x256x128_w1x4_nb2_sk1_bd1_xcd_b8_s32_32x32` | 20.0 | 10.492 | 524.6 | gemm | (524288, 1, 1) |
| `_engram_lookup_kernel` | 2.0 | 9.869 | 4897.1 | misc | (65536, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_2x8_row_wb32x32_k8192` | 20.0 | 9.746 | 487.9 | quant | (655360, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb0ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 20.0 | 8.217 | 410.7 | mhc | (262144, 1, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 8.098 | 197.3 | quant | (34078720, 1, 1) (4352, 1, 1) (532480, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb1ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 18.0 | 7.880 | 437.7 | mhc | (262144, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_4x4_row_wb32x32_k5120` | 20.0 | 6.385 | 317.8 | quant | (589824, 1, 1) |
| `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_64_BLOCK_KV_64_BLOCK_M_3_` | 15.8 | 5.707 | 414.6 | indexer | (100096, 1, 7) (113280, 1, 7) (113536, 1, 7) |
| `mhc_pre_big_fuse_tilelang_kernel` | 1.0 | 4.434 | 4450.2 | mhc | (1572864, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x8_row_wb32x32_k6144` | 2.0 | 3.897 | 1948.3 | quant | (3276800, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x4_row_wb32x32_k2304` | 20.0 | 3.399 | 169.7 | quant | (655360, 1, 1) |
| `_dequantize_and_gather_k_one_pass_kernel` | 42.0 | 3.120 | 57.7 | quant | (512, 128, 1) (512, 2101, 1) (512, 2613, 1) |
| `_kernel` | 131.0 | 3.072 | 11.3 | misc | (128, 10, 1) (128, 16, 1) (128, 3, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k5120` | 21.0 | 2.737 | 128.0 | quant | (229376, 1, 1) |

### SAMPLE_after_MIXED: 20 records; ; GPU span median 3.8868929999999997; kernel ms/record 3.86; graph-replayed 0/740
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 3.0 | 1.196 | 399.2 | quant | (34078720, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k15360` | 1.0 | 0.787 | 787.6 | quant | (655360, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_11FillFunctorIN3c108BFloat16EEESt5arr` | 3.0 | 0.432 | 143.9 | norm_rope_elem | (67108864, 1, 1) |
| `_ZN2at6native12_GLOBAL__N_126CatArrayBatchedCopy_contigINS1_10OpaqueTypeILj2EEEjLi2ELi128E` | 1.0 | 0.371 | 370.9 | norm_rope_elem | (2097152, 3, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k5120` | 3.0 | 0.368 | 122.5 | quant | (229376, 1, 1) |
| `_kernel` | 4.0 | 0.297 | 54.0 | misc | (524288, 10, 1) (524288, 30, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x128_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGS` | 1.0 | 0.196 | 195.7 | gemm | (129280, 1, 1) |
| `__amd_rocclr_copyBuffer` | 3.0 | 0.072 | 4.5 | norm_rope_elem | (131072, 1, 1) (512, 1, 1) |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb0ELb0ELb1ELi1ELb0EEEvPT0_PT_PfS` | 1.0 | 0.051 | 51.5 | quant | (4194304, 1, 1) |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi64ELi8ELb0ELb0ELb1ELi1ELb0EEEvPT0_PT_PfS4_` | 3.0 | 0.021 | 7.0 | quant | (1048576, 1, 1) |
| `_prepare_dflash_inputs_kernel` | 3.0 | 0.018 | 5.6 | attention | (8704, 65, 1) (8960, 65, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_ker` | 2.0 | 0.009 | 4.7 | norm_rope_elem | (128, 1, 1) |
| `_gumbel_sample_kernel` | 1.0 | 0.007 | 6.6 | sampling | (512, 127, 1) (768, 127, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIiEESt5arrayIPcLm1EEEEvi` | 1.0 | 0.006 | 5.5 | norm_rope_elem | (256, 1, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_ker` | 1.0 | 0.006 | 5.5 | norm_rope_elem | (128, 1, 1) |
| `_ZN2at6native24vectorized_gather_kernelILi16ElEEvPcS2_PT0_illllb` | 1.0 | 0.005 | 5.1 | misc | (512, 3, 1) (768, 3, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_21index_put` | 1.0 | 0.005 | 5.0 | norm_rope_elem | (128, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIlEESt5arrayIPcLm1EEEEvi` | 1.0 | 0.005 | 4.9 | norm_rope_elem | (256, 1, 1) |
| `_gumbel_combine_kernel` | 1.0 | 0.005 | 4.7 | sampling | (512, 1, 1) (768, 1, 1) |
| `_get_num_sampled_and_rejected_kernel` | 1.0 | 0.004 | 4.1 | sampling | (512, 1, 1) (768, 1, 1) |

## prefill1 pid 1536 dp_rank 0: window 9356 ms (steps), 38484 dispatches, unattributed 0 (0.0 ms), outside steps 2 (0.0 ms), missing symbols 0, causality violations 0
### MIXED: 20 records; reqs 2.0[2-3] tokens 16384.0[16384-16384] prefill_tokens 16384.0[16384-16384] prefill_reqs 2.0[2-3] spec_tokens 0.0[0-0] max_q 8192.0[8192-8192]; GPU span median 437.466895; kernel ms/record 451.98; graph-replayed 10650/37742
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 128.017 | 1867.7 | comm | (2560, 1, 1) (3072, 1, 1) (7680, 1, 1) |
| `_Z45opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernelI45opus_mla_v4_prefill_a16w16_16mx8_32nx1` | 20.0 | 69.535 | 3703.1 | attention | (8388608, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 20.0 | 65.362 | 3264.9 | moe | (4608, 1920, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 20.0 | 36.239 | 1812.8 | gemm | (19630080, 1, 1) |
| `_inverse_rope_mxfp8_quant_kernel` | 23.0 | 17.560 | 873.1 | quant | (33024, 64, 1) (33280, 64, 1) (4194304, 64, 1) |
| `flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32` | 20.0 | 13.509 | 660.5 | misc | (32768, 128, 1) |
| `bmm_a8w8_mxscale_t256x256x128_w1x4_nb2_sk1_bd1_xcd_b8_s32_32x32` | 20.0 | 10.468 | 523.3 | gemm | (524288, 1, 1) |
| `_engram_lookup_kernel` | 2.0 | 9.897 | 4984.5 | misc | (65536, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_2x8_row_wb32x32_k8192` | 20.0 | 9.800 | 490.2 | quant | (655360, 1, 1) |
| `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_64_BLOCK_KV_64_BLOCK_M_3_` | 20.9 | 8.083 | 423.9 | indexer | (101120, 1, 7) (109952, 1, 7) (11712, 1, 16) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 8.081 | 195.2 | quant | (34078720, 1, 1) (4352, 1, 1) (532480, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb0ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 20.0 | 7.993 | 400.1 | mhc | (262144, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb1ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 18.0 | 7.762 | 430.6 | mhc | (262144, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_4x4_row_wb32x32_k5120` | 20.0 | 6.442 | 321.1 | quant | (589824, 1, 1) |
| `mhc_pre_big_fuse_tilelang_kernel` | 1.0 | 4.407 | 4385.4 | mhc | (1572864, 1, 1) |
| `_dequantize_and_gather_k_one_pass_kernel` | 44.0 | 4.384 | 58.5 | quant | (512, 128, 1) (512, 2744, 1) (512, 3256, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x8_row_wb32x32_k6144` | 2.0 | 3.943 | 1977.2 | quant | (3276800, 1, 1) |
| `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 22.4 | 3.735 | 187.4 | misc | (1016832, 1, 1) (1048576, 1, 1) (1082368, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x4_row_wb32x32_k2304` | 20.0 | 3.436 | 171.8 | quant | (655360, 1, 1) |
| `_kernel` | 131.0 | 3.140 | 11.3 | misc | (128, 10, 1) (128, 16, 1) (128, 3, 1) |

### SAMPLE_after_MIXED: 20 records; ; GPU span median 3.842133; kernel ms/record 3.84; graph-replayed 0/740
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 3.0 | 1.162 | 387.6 | quant | (34078720, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k15360` | 1.0 | 0.789 | 789.7 | quant | (655360, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_11FillFunctorIN3c108BFloat16EEESt5arr` | 3.0 | 0.431 | 142.8 | norm_rope_elem | (67108864, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k5120` | 3.0 | 0.378 | 126.2 | quant | (229376, 1, 1) |
| `_ZN2at6native12_GLOBAL__N_126CatArrayBatchedCopy_contigINS1_10OpaqueTypeILj2EEEjLi2ELi128E` | 1.0 | 0.357 | 357.3 | norm_rope_elem | (2097152, 3, 1) |
| `_kernel` | 4.0 | 0.302 | 54.0 | misc | (524288, 10, 1) (524288, 30, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x128_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGS` | 1.0 | 0.192 | 192.3 | gemm | (129280, 1, 1) |
| `__amd_rocclr_copyBuffer` | 3.0 | 0.081 | 4.7 | norm_rope_elem | (131072, 1, 1) (512, 1, 1) |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb0ELb0ELb1ELi1ELb0EEEvPT0_PT_PfS` | 1.0 | 0.052 | 52.2 | quant | (4194304, 1, 1) |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi64ELi8ELb0ELb0ELb1ELi1ELb0EEEvPT0_PT_PfS4_` | 3.0 | 0.022 | 7.1 | quant | (1048576, 1, 1) |
| `_prepare_dflash_inputs_kernel` | 3.0 | 0.018 | 6.1 | attention | (8704, 65, 1) (8960, 65, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_ker` | 2.0 | 0.009 | 4.6 | norm_rope_elem | (128, 1, 1) |
| `_gumbel_sample_kernel` | 1.0 | 0.006 | 6.4 | sampling | (512, 127, 1) (768, 127, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_ker` | 1.0 | 0.006 | 5.7 | norm_rope_elem | (128, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIiEESt5arrayIPcLm1EEEEvi` | 1.0 | 0.006 | 5.6 | norm_rope_elem | (256, 1, 1) |
| `_ZN2at6native24vectorized_gather_kernelILi16ElEEvPcS2_PT0_illllb` | 1.0 | 0.005 | 5.1 | misc | (512, 3, 1) (768, 3, 1) |
| `_gumbel_combine_kernel` | 1.0 | 0.005 | 4.8 | sampling | (512, 1, 1) (768, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIlEESt5arrayIPcLm1EEEEvi` | 1.0 | 0.005 | 4.8 | norm_rope_elem | (256, 1, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_21index_put` | 1.0 | 0.005 | 4.7 | norm_rope_elem | (128, 1, 1) |
| `_get_num_sampled_and_rejected_kernel` | 1.0 | 0.004 | 4.1 | sampling | (512, 1, 1) (768, 1, 1) |

