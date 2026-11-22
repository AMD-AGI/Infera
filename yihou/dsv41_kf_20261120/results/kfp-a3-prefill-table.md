# Calibrated-profiler hot-op tables (vLLM engines)

ms = summed kernel durations (streams overlap -> shares are of GPU kernel time, not wall time).

## role prefill: 63044.0 kernel-ms in the windows; mHC bucket rank 7, first mHC kernel rank 13

stage kernel-ms: DUMMY 31729.6, MIXED 31046.9, SAMPLE_after_MIXED 267.5

| bucket | kernel ms | share |
|---|---|---|
| comm | 36121.05 | 57.3% |
| moe | 7307.61 | 11.6% |
| quant | 4830.10 | 7.7% |
| gemm | 4772.91 | 7.6% |
| attention | 4633.94 | 7.4% |
| misc | 2588.57 | 4.1% |
| mhc | 1235.90 | 2.0% |
| indexer | 987.07 | 1.6% |
| norm_rope_elem | 564.86 | 0.9% |
| sampling | 2.00 | 0.0% |

| # | kernel | ms | share | bucket |
|---|---|---|---|---|
| 1 | `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 36121.05 | 57.29% | comm |
| 2 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 6556.18 | 10.40% | moe |
| 3 | `_Z45opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernelI45opus_mla_v4_prefill_a16w16_16mx8_32nx1_traitsILi` | 4460.33 | 7.07% | attention |
| 4 | `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf16lds_v2_b` | 3672.87 | 5.83% | gemm |
| 5 | `_inverse_rope_mxfp8_quant_kernel` | 1147.15 | 1.82% | quant |
| 6 | `_dequantize_and_gather_k_one_pass_kernel` | 973.20 | 1.54% | quant |
| 7 | `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_64_BLOCK_KV_64_BLOCK_M_3_PRESHUFFLE` | 880.88 | 1.40% | indexer |
| 8 | `flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32` | 873.38 | 1.39% | misc |
| 9 | `bmm_a8w8_mxscale_t256x256x128_w1x4_nb2_sk1_bd1_xcd_b8_s32_32x32` | 664.06 | 1.05% | gemm |
| 10 | `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_2x8_row_wb32x32_k8192` | 616.20 | 0.98% | quant |
| 11 | `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFloat16ELi64` | 536.16 | 0.85% | quant |
| 12 | `_engram_lookup_kernel` | 516.43 | 0.82% | misc |
| 13 | `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb0ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16bS3_PKfS5_` | 503.79 | 0.80% | mhc |
| 14 | `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb1ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16bS3_PKfS5_` | 483.52 | 0.77% | mhc |
| 15 | `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 472.75 | 0.75% | misc |
| 16 | `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_4x4_row_wb32x32_k5120` | 400.79 | 0.64% | quant |
| 17 | `_kernel` | 267.56 | 0.42% | misc |
| 18 | `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x8_row_wb32x32_k6144` | 247.10 | 0.39% | quant |
| 19 | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi32ELb0ELi64ELb1ELb0ELi0ELb0ELb0ELb0` | 219.22 | 0.35% | quant |
| 20 | `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x4_row_wb32x32_k2304` | 213.96 | 0.34% | quant |
| 21 | `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k5120` | 199.89 | 0.32% | quant |
| 22 | `kernel` | 168.04 | 0.27% | misc |
| 23 | `_ZN5aiter22opus_moe_sorting_entryINS_30MoeSortingMultiPhaseKernel_P23INS_19MoeSortingProblemMpIifhLi` | 162.51 | 0.26% | moe |
| 24 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 143.42 | 0.23% | moe |
| 25 | `_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_15CUDAFunctor_addIN3c108BFloat16EEESt5arrayIPcL` | 119.70 | 0.19% | norm_rope_elem |
| 26 | `_ZN4vllm3moe22topkGatingSoftplusSqrtILi6ELi384ELi4ELi8ELi64ELb0EiifEEvPKT7_PKbPfiPT5_PiiiibdPKfPKT6_` | 113.76 | 0.18% | moe |
| 27 | `_sparse_attn_prefill_ragged_kernel` | 97.61 | 0.15% | attention |
| 28 | `mfma_moe1_silu_mul_afp8_wfp4_bf16_t64x128x256_pm1_async_gui_v33` | 93.93 | 0.15% | moe |
| 29 | `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT192x128x64_MI16x16x1_SN_LDSB1_AFC0_AG0_AGGSUA0_AGNT` | 92.73 | 0.15% | gemm |
| 30 | `_ZN5aiter21mxfp4_moe_sort_kernelILi256ELi32ELi24ELi32EEEvPhS1_PKiS3_iiiii` | 87.96 | 0.14% | moe |

## prefill0 pid 678 dp_rank 1: window 10361 ms (steps), 44138 dispatches, unattributed 0 (0.0 ms), outside steps 5 (0.0 ms), missing symbols 0, causality violations 0
### MIXED: 20 records; reqs 2.0[2-16] tokens 16384.0[16384-16384] prefill_tokens 16384.0[16384-16384] prefill_reqs 2.0[2-16] spec_tokens 0.0[0-0] max_q 8192.0[8192-8192]; GPU span median 463.27308; kernel ms/record 494.38; graph-replayed 10386/43393
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 147.416 | 3045.3 | comm | (2560, 1, 1) (3072, 1, 1) (7680, 1, 1) |
| `_Z45opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernelI45opus_mla_v4_prefill_a16w16_16mx8_32nx1` | 22.9 | 69.480 | 3669.2 | attention | (1061888, 1, 1) (4194304, 1, 1) (524800, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 20.0 | 65.554 | 3274.4 | moe | (4608, 1920, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 20.0 | 36.150 | 1805.0 | gemm | (19630080, 1, 1) |
| `_dequantize_and_gather_k_one_pass_kernel` | 55.7 | 19.726 | 67.8 | quant | (1024, 128, 1) (1024, 8192, 1) (1536, 128, 1) |
| `_inverse_rope_mxfp8_quant_kernel` | 25.0 | 17.729 | 874.6 | quant | (196864, 64, 1) (33280, 64, 1) (4194304, 64, 1) |
| `flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32` | 20.0 | 13.630 | 666.2 | misc | (32768, 128, 1) |
| `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_64_BLOCK_KV_64_BLOCK_M_3_` | 32.5 | 13.174 | 422.6 | indexer | (100864, 1, 7) (117888, 1, 6) (11904, 1, 16) |
| `bmm_a8w8_mxscale_t256x256x128_w1x4_nb2_sk1_bd1_xcd_b8_s32_32x32` | 20.0 | 10.405 | 522.0 | gemm | (524288, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_2x8_row_wb32x32_k8192` | 20.0 | 9.875 | 494.0 | quant | (655360, 1, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 7.975 | 210.2 | quant | (1597440, 1, 1) (1863680, 1, 1) (34078720, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb0ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 20.0 | 7.867 | 393.0 | mhc | (262144, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb1ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 18.0 | 7.587 | 421.0 | mhc | (262144, 1, 1) |
| `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 43.6 | 7.248 | 190.2 | misc | (1020928, 1, 1) (1025024, 1, 1) (1056768, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_4x4_row_wb32x32_k5120` | 20.0 | 6.419 | 319.4 | quant | (589824, 1, 1) |
| `_engram_lookup_kernel` | 2.0 | 4.045 | 2020.4 | misc | (65536, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x8_row_wb32x32_k6144` | 2.0 | 3.973 | 1984.9 | quant | (3276800, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x4_row_wb32x32_k2304` | 20.0 | 3.421 | 170.7 | quant | (655360, 1, 1) |
| `_kernel` | 131.0 | 3.256 | 11.4 | misc | (12288, 10, 1) (12288, 16, 1) (12288, 3, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k5120` | 21.0 | 2.847 | 127.7 | quant | (229376, 1, 1) |

### SAMPLE_after_MIXED: 20 records; ; GPU span median 3.8416635; kernel ms/record 3.84; graph-replayed 0/740
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 3.0 | 1.195 | 398.8 | quant | (34078720, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k15360` | 1.0 | 0.789 | 788.1 | quant | (655360, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_11FillFunctorIN3c108BFloat16EEESt5arr` | 3.0 | 0.429 | 143.3 | norm_rope_elem | (67108864, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k5120` | 3.0 | 0.368 | 122.6 | quant | (229376, 1, 1) |
| `_ZN2at6native12_GLOBAL__N_126CatArrayBatchedCopy_contigINS1_10OpaqueTypeILj2EEEjLi2ELi128E` | 1.0 | 0.352 | 351.8 | norm_rope_elem | (2097152, 3, 1) |
| `_kernel` | 4.0 | 0.294 | 53.7 | misc | (524288, 10, 1) (524288, 30, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x128_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGS` | 1.0 | 0.197 | 196.6 | gemm | (129280, 1, 1) |
| `__amd_rocclr_copyBuffer` | 3.0 | 0.071 | 4.7 | norm_rope_elem | (131072, 1, 1) (512, 1, 1) |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb0ELb0ELb1ELi1ELb0EEEvPT0_PT_PfS` | 1.0 | 0.052 | 51.8 | quant | (4194304, 1, 1) |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi64ELi8ELb0ELb0ELb1ELi1ELb0EEEvPT0_PT_PfS4_` | 3.0 | 0.022 | 7.2 | quant | (1048576, 1, 1) |
| `_prepare_dflash_inputs_kernel` | 3.0 | 0.018 | 6.0 | attention | (12288, 65, 1) (8704, 65, 1) (8960, 65, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_ker` | 2.0 | 0.009 | 4.6 | norm_rope_elem | (128, 1, 1) |
| `_gumbel_sample_kernel` | 1.0 | 0.007 | 6.8 | sampling | (1024, 127, 1) (1792, 127, 1) (4096, 127, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_ker` | 1.0 | 0.005 | 5.4 | norm_rope_elem | (128, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIiEESt5arrayIPcLm1EEEEvi` | 1.0 | 0.005 | 5.4 | norm_rope_elem | (256, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIlEESt5arrayIPcLm1EEEEvi` | 1.0 | 0.005 | 4.9 | norm_rope_elem | (256, 1, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_21index_put` | 1.0 | 0.005 | 4.8 | norm_rope_elem | (128, 1, 1) |
| `_gumbel_combine_kernel` | 1.0 | 0.005 | 4.8 | sampling | (1024, 1, 1) (1792, 1, 1) (4096, 1, 1) |
| `_get_num_sampled_and_rejected_kernel` | 1.0 | 0.004 | 4.1 | sampling | (1024, 1, 1) (1792, 1, 1) (4096, 1, 1) |
| `_post_update_kernel` | 1.0 | 0.003 | 3.2 | misc | (1024, 1, 1) (128, 1, 1) (192, 1, 1) |

## prefill0 pid 681 dp_rank 0: window 10362 ms (steps), 44295 dispatches, unattributed 0 (0.0 ms), outside steps 10 (0.0 ms), missing symbols 0, causality violations 0
### MIXED: 20 records; reqs 2.0[2-8] tokens 16384.0[16384-16384] prefill_tokens 16384.0[16384-16384] prefill_reqs 2.0[2-8] spec_tokens 0.0[0-0] max_q 8192.0[8192-8192]; GPU span median 463.289182; kernel ms/record 496.40; graph-replayed 10386/43545
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 150.508 | 3051.3 | comm | (2560, 1, 1) (3072, 1, 1) (7680, 1, 1) |
| `_Z45opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernelI45opus_mla_v4_prefill_a16w16_16mx8_32nx1` | 20.2 | 68.703 | 3647.9 | attention | (4194304, 1, 1) (8388608, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 20.0 | 64.234 | 3208.9 | moe | (4608, 1920, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 20.0 | 35.966 | 1797.2 | gemm | (19630080, 1, 1) |
| `_inverse_rope_mxfp8_quant_kernel` | 26.0 | 17.752 | 876.4 | quant | (164096, 64, 1) (197120, 64, 1) (33024, 64, 1) |
| `_dequantize_and_gather_k_one_pass_kernel` | 50.2 | 16.834 | 78.2 | quant | (1536, 128, 1) (1536, 8192, 1) (1792, 128, 1) |
| `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_64_BLOCK_KV_64_BLOCK_M_3_` | 37.6 | 15.075 | 411.9 | indexer | (11008, 1, 54) (111040, 1, 7) (11392, 1, 17) |
| `flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32` | 20.0 | 13.393 | 656.5 | misc | (32768, 128, 1) |
| `bmm_a8w8_mxscale_t256x256x128_w1x4_nb2_sk1_bd1_xcd_b8_s32_32x32` | 20.0 | 10.432 | 521.0 | gemm | (524288, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_2x8_row_wb32x32_k8192` | 20.0 | 9.686 | 484.4 | quant | (655360, 1, 1) |
| `_engram_lookup_kernel` | 2.0 | 9.527 | 4753.9 | misc | (65536, 1, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 8.144 | 202.7 | quant | (1597440, 1, 1) (1863680, 1, 1) (34078720, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb0ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 20.0 | 8.007 | 401.2 | mhc | (262144, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb1ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 18.0 | 7.739 | 429.3 | mhc | (262144, 1, 1) |
| `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 46.5 | 7.645 | 189.1 | misc | (1005568, 1, 1) (1022976, 1, 1) (1029120, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_4x4_row_wb32x32_k5120` | 20.0 | 6.271 | 312.4 | quant | (589824, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x8_row_wb32x32_k6144` | 2.0 | 3.869 | 1935.1 | quant | (3276800, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x4_row_wb32x32_k2304` | 20.0 | 3.357 | 167.3 | quant | (655360, 1, 1) |
| `_kernel` | 131.0 | 3.108 | 11.1 | misc | (12288, 10, 1) (12288, 16, 1) (12288, 3, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k5120` | 21.0 | 2.698 | 125.8 | quant | (229376, 1, 1) |

### SAMPLE_after_MIXED: 20 records; ; GPU span median 3.828576; kernel ms/record 3.81; graph-replayed 0/740
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 3.0 | 1.162 | 387.8 | quant | (34078720, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k15360` | 1.0 | 0.779 | 777.8 | quant | (655360, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_11FillFunctorIN3c108BFloat16EEESt5arr` | 3.0 | 0.430 | 143.1 | norm_rope_elem | (67108864, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k5120` | 3.0 | 0.370 | 123.1 | quant | (229376, 1, 1) |
| `_ZN2at6native12_GLOBAL__N_126CatArrayBatchedCopy_contigINS1_10OpaqueTypeILj2EEEjLi2ELi128E` | 1.0 | 0.363 | 363.3 | norm_rope_elem | (2097152, 3, 1) |
| `_kernel` | 4.0 | 0.295 | 53.7 | misc | (524288, 10, 1) (524288, 30, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x128_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGS` | 1.0 | 0.195 | 195.2 | gemm | (129280, 1, 1) |
| `__amd_rocclr_copyBuffer` | 3.0 | 0.072 | 4.7 | norm_rope_elem | (131072, 1, 1) (512, 1, 1) |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb0ELb0ELb1ELi1ELb0EEEvPT0_PT_PfS` | 1.0 | 0.051 | 50.8 | quant | (4194304, 1, 1) |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi64ELi8ELb0ELb0ELb1ELi1ELb0EEEvPT0_PT_PfS4_` | 3.0 | 0.021 | 7.1 | quant | (1048576, 1, 1) |
| `_prepare_dflash_inputs_kernel` | 3.0 | 0.018 | 5.6 | attention | (10240, 65, 1) (8704, 65, 1) (8960, 65, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_ker` | 2.0 | 0.009 | 4.2 | norm_rope_elem | (128, 1, 1) |
| `_gumbel_sample_kernel` | 1.0 | 0.007 | 6.8 | sampling | (1536, 127, 1) (2048, 127, 1) (512, 127, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIiEESt5arrayIPcLm1EEEEvi` | 1.0 | 0.005 | 5.4 | norm_rope_elem | (256, 1, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_ker` | 1.0 | 0.005 | 5.3 | norm_rope_elem | (128, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIlEESt5arrayIPcLm1EEEEvi` | 1.0 | 0.005 | 4.8 | norm_rope_elem | (256, 1, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_21index_put` | 1.0 | 0.005 | 4.7 | norm_rope_elem | (128, 1, 1) |
| `_gumbel_combine_kernel` | 1.0 | 0.005 | 4.7 | sampling | (1536, 1, 1) (2048, 1, 1) (512, 1, 1) |
| `_get_num_sampled_and_rejected_kernel` | 1.0 | 0.004 | 4.0 | sampling | (1536, 1, 1) (2048, 1, 1) (512, 1, 1) |
| `_post_update_kernel` | 1.0 | 0.003 | 3.3 | misc | (128, 1, 1) (192, 1, 1) (384, 1, 1) |

## prefill1 pid 1540 dp_rank 1: window 35026 ms (max_window_s_idle), 140704 dispatches, unattributed 0 (0.0 ms), outside steps 14621 (433.4 ms), missing symbols 0, causality violations 0
### DECODE: 80 records; reqs 0.0[0-0] tokens 0.0[0-0] prefill_tokens 0.0[0-0] prefill_reqs 0.0[0-0] spec_tokens 0.0[0-0] max_q 0.0[0-0]; GPU span median None; kernel ms/record 0.00; graph-replayed 0/0
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|

### DUMMY: 79 records; reqs 0[0-0] tokens 0[0-0] prefill_tokens 0[0-0] prefill_reqs 0[0-0] spec_tokens 0[0-0] max_q 0[0-0]; GPU span median 402.194421; kernel ms/record 401.64; graph-replayed 41556/111326
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 335.567 | 2305.3 | comm | (1280, 1, 1) (1792, 1, 1) (2560, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 20.0 | 35.200 | 1780.8 | moe | (4608, 1153, 1) (4608, 769, 1) (4608, 778, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 20.0 | 19.919 | 1008.8 | gemm | (11776000, 1, 1) (7843840, 1, 1) (7936000, 1, 1) |
| `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi32ELb0ELi64ELb1ELb0ELi0EL` | 22.8 | 1.036 | 51.8 | quant | (122880, 1, 1) (1311680, 1, 1) (1342080, 1, 1) |
| `_ZN5aiter22opus_moe_sorting_entryINS_30MoeSortingMultiPhaseKernel_P23INS_19MoeSortingProbl` | 22.8 | 0.844 | 39.7 | moe | (229376, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 17.7 | 0.664 | 16.1 | moe | (4608, 24, 1) (4608, 432, 1) (4608, 480, 1) |
| `_kernel` | 127.9 | 0.601 | 4.7 | misc | (12288, 10, 1) (12288, 16, 1) (12288, 3, 1) |
| `_ZN5aiter21mxfp4_moe_sort_kernelILi256ELi32ELi24ELi32EEEvPhS1_PKiS3_iiiii` | 22.8 | 0.477 | 23.6 | moe | (1180672, 1, 1) (122880, 1, 1) (135168, 1, 1) |
| `_ZN4vllm3moe22topkGatingSoftplusSqrtILi6ELi384ELi4ELi8ELi64ELb0EiifEEvPKT7_PKbPfiPT5_Piiii` | 40.0 | 0.367 | 9.4 | moe | (10240, 4, 1) (128, 4, 1) (2048, 4, 1) |
| `bmm_a8w8_mxscale_t16x32x256_w1x2_nb5_sk1_nt_xcd_sps_b8_s32_32x32` | 37.2 | 0.272 | 7.2 | gemm | (32768, 1, 1) |
| `kernel` | 46.0 | 0.222 | 4.8 | misc | (1024, 2, 1) (131072, 2, 1) (1536, 1, 1) |
| `flydsl_mxpsh_16x32x512_F8_F8_B16_w4_x0_sk1_multirow_slrr_wb32x32_pf5_nt` | 35.4 | 0.220 | 6.1 | norm_rope_elem | (128, 144, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_bf16_t64x128x256_pm1_async_gui_v33` | 0.8 | 0.220 | 284.2 | moe | (4608, 481, 1) (4608, 504, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi2ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 20.0 | 0.218 | 10.9 | mhc | (65536, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi2ELb1ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 18.0 | 0.217 | 12.0 | mhc | (65536, 1, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 40.0 | 0.216 | 5.3 | attention | (1536, 4, 4) (1536, 8, 4) (256, 32, 4) |
| `_sparse_attn_decode_reduce_kernel` | 40.0 | 0.209 | 5.2 | attention | (1536, 64, 1) (256, 64, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 0.203 | 4.6 | quant | (1064960, 1, 1) (12544, 1, 1) (1331200, 1, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 2.0 | 0.202 | 90.6 | gemm | (2396160, 1, 1) (2641920, 1, 1) (2887680, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_15CUDAFunctor_addIN3c108BFloat16EEESt` | 40.0 | 0.194 | 4.8 | norm_rope_elem | (1280, 1, 1) (163840, 1, 1) (245760, 1, 1) |

### DUMMY_BATCH: 79 records; ; GPU span median None; kernel ms/record 0.00; graph-replayed 0/0
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|

### MIXED: 6 records; reqs 1.0[1-2] tokens 13470.0[4162-16384] prefill_tokens 13470.0[4162-16384] prefill_reqs 1.0[1-2] spec_tokens 0.0[0-0] max_q 12288.0[4162-16384]; GPU span median 489.7178895; kernel ms/record 488.03; graph-replayed 3200/14535
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 177.808 | 2346.1 | comm | (2560, 1, 1) (3072, 1, 1) (7168, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 20.0 | 57.018 | 2962.9 | moe | (4608, 1348, 1) (4608, 1536, 1) (4608, 1647, 1) |
| `_Z45opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernelI45opus_mla_v4_prefill_a16w16_16mx8_32nx1` | 20.0 | 50.915 | 2394.9 | attention | (2130944, 1, 1) (4194304, 1, 1) (5404672, 1, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 20.0 | 31.985 | 1666.2 | gemm | (13772800, 1, 1) (15697920, 1, 1) (16834560, 1, 1) |
| `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_64_BLOCK_KV_64_BLOCK_M_3_` | 58.5 | 23.905 | 402.5 | indexer | (11712, 1, 60) (11776, 1, 60) (12224, 1, 58) |
| `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 66.2 | 15.145 | 236.3 | misc | (131072, 1, 1) (132096, 1, 1) (215040, 1, 1) |
| `_inverse_rope_mxfp8_quant_kernel` | 26.7 | 13.045 | 527.0 | quant | (1065472, 64, 1) (2097152, 64, 1) (2702336, 64, 1) |
| `_dequantize_and_gather_k_one_pass_kernel` | 51.3 | 11.473 | 114.6 | quant | (256, 128, 1) (256, 8192, 1) (512, 128, 1) |
| `flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32` | 20.0 | 10.077 | 580.8 | misc | (16384, 128, 1) (21248, 128, 1) (32768, 128, 1) |
| `_engram_lookup_kernel` | 2.0 | 8.750 | 4754.2 | misc | (65536, 1, 1) |
| `bmm_a8w8_mxscale_t256x256x128_w1x4_nb2_sk1_bd1_xcd_b8_s32_32x32` | 13.3 | 6.439 | 517.2 | gemm | (344064, 1, 1) (524288, 1, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 5.834 | 55.3 | quant | (1064960, 1, 1) (17039360, 1, 1) (21956608, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_2x8_row_wb32x32_k8192` | 10.0 | 4.856 | 486.3 | quant | (655360, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb0ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 13.3 | 4.688 | 392.2 | mhc | (131072, 1, 1) (262144, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb1ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 12.0 | 4.578 | 426.5 | mhc | (131072, 1, 1) (262144, 1, 1) |
| `mhc_pre_big_fuse_tilelang_kernel` | 1.0 | 3.805 | 4188.0 | mhc | (1013376, 1, 1) (1572864, 1, 1) (399552, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 20.0 | 3.530 | 193.0 | moe | (4608, 24, 1) (4608, 432, 1) (4608, 480, 1) |
| `kernel` | 141.0 | 3.403 | 23.8 | misc | (1024, 2, 1) (131072, 2, 1) (196608, 2, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_4x4_row_wb32x32_k5120` | 10.0 | 3.165 | 315.4 | quant | (589824, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_4x8_row_wb32x32_k8192` | 10.0 | 2.743 | 281.7 | quant | (174080, 1, 1) (327680, 1, 1) (430080, 1, 1) |

### SAMPLE_after_MIXED: 6 records; ; GPU span median 3.332249; kernel ms/record 2.98; graph-replayed 0/222
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 3.0 | 0.875 | 323.3 | quant | (17039360, 1, 1) (21956608, 1, 1) (34078720, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k15360` | 0.5 | 0.398 | 794.1 | quant | (655360, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_11FillFunctorIN3c108BFloat16EEESt5arr` | 3.0 | 0.310 | 114.9 | norm_rope_elem | (17047552, 1, 1) (33554432, 1, 1) (43237376, 1, 1) |
| `_ZN2at6native12_GLOBAL__N_126CatArrayBatchedCopy_contigINS1_10OpaqueTypeILj2EEEjLi2ELi128E` | 1.0 | 0.268 | 298.9 | norm_rope_elem | (2097152, 3, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k5120` | 2.0 | 0.240 | 124.2 | quant | (150528, 1, 1) (229376, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x4_row_wb32x32_k15360` | 0.5 | 0.228 | 462.2 | quant | (174080, 1, 1) (327680, 1, 1) (430080, 1, 1) |
| `_kernel` | 4.0 | 0.219 | 42.2 | misc | (133248, 10, 1) (133248, 30, 1) (262144, 10, 1) |
| `_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi2ELi16ELi8ELi2ELi1EEviiiiiiPKT_S3_S3_PS1_ii` | 0.8 | 0.162 | 194.2 | gemm | (16384, 16, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x8_row_wb32x32_k5120` | 1.0 | 0.061 | 58.4 | quant | (114688, 1, 1) (60928, 1, 1) |
| `__amd_rocclr_copyBuffer` | 3.0 | 0.058 | 4.6 | norm_rope_elem | (131072, 1, 1) (512, 1, 1) |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb0ELb0ELb1ELi1ELb0EEEvPT0_PT_PfS` | 1.0 | 0.038 | 40.6 | quant | (1065472, 1, 1) (2097152, 1, 1) (2702336, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x128_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGS` | 0.2 | 0.033 | 195.8 | gemm | (129280, 1, 1) |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi64ELi8ELb0ELb0ELb1ELi1ELb0EEEvPT0_PT_PfS4_` | 3.0 | 0.019 | 6.5 | quant | (1048576, 1, 1) (266368, 1, 1) (524288, 1, 1) |
| `_prepare_dflash_inputs_kernel` | 3.0 | 0.017 | 5.4 | attention | (8448, 129, 1) (8448, 33, 1) (8448, 65, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_ker` | 2.0 | 0.009 | 4.3 | norm_rope_elem | (128, 1, 1) |
| `_gumbel_sample_kernel` | 1.0 | 0.006 | 6.3 | sampling | (256, 127, 1) (512, 127, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_ker` | 1.0 | 0.006 | 5.6 | norm_rope_elem | (128, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIiEESt5arrayIPcLm1EEEEvi` | 1.0 | 0.006 | 5.5 | norm_rope_elem | (256, 1, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_ker` | 0.8 | 0.005 | 6.5 | norm_rope_elem | (1280, 1, 1) |
| `_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_21index_put` | 1.0 | 0.005 | 4.9 | norm_rope_elem | (128, 1, 1) |

## prefill1 pid 1541 dp_rank 0: window 8830 ms (steps), 42188 dispatches, unattributed 0 (0.0 ms), outside steps 3 (0.0 ms), missing symbols 0, causality violations 0
### MIXED: 20 records; reqs 2.0[2-5] tokens 16384.0[16384-16384] prefill_tokens 16384.0[16384-16384] prefill_reqs 2.0[2-5] spec_tokens 0.0[0-0] max_q 8192.0[8192-8192]; GPU span median 383.273182; kernel ms/record 415.15; graph-replayed 10652/38967
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 129.232 | 1846.7 | comm | (2560, 1, 1) (3072, 1, 1) (7168, 1, 1) |
| `_Z45opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernelI45opus_mla_v4_prefill_a16w16_16mx8_32nx1` | 20.0 | 69.559 | 3705.5 | attention | (8388608, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 20.0 | 41.877 | 1771.6 | moe | (4608, 1153, 1) (4608, 1348, 1) (4608, 1536, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 20.0 | 23.251 | 983.7 | gemm | (11776000, 1, 1) (13772800, 1, 1) (15697920, 1, 1) |
| `_inverse_rope_mxfp8_quant_kernel` | 26.0 | 17.963 | 890.3 | quant | (32768, 64, 1) (33024, 64, 1) (33280, 64, 1) |
| `flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32` | 20.0 | 13.623 | 666.9 | misc | (32768, 128, 1) |
| `bmm_a8w8_mxscale_t256x256x128_w1x4_nb2_sk1_bd1_xcd_b8_s32_32x32` | 20.0 | 10.435 | 523.6 | gemm | (524288, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_2x8_row_wb32x32_k8192` | 20.0 | 9.792 | 490.2 | quant | (655360, 1, 1) |
| `_engram_lookup_kernel` | 2.0 | 9.574 | 4909.1 | misc | (65536, 1, 1) |
| `_dequantize_and_gather_k_one_pass_kernel` | 50.0 | 8.658 | 70.2 | quant | (1024, 128, 1) (1024, 8192, 1) (1280, 128, 1) |
| `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_64_BLOCK_KV_64_BLOCK_M_3_` | 22.4 | 8.624 | 418.8 | indexer | (102912, 1, 7) (10944, 1, 18) (11392, 1, 55) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 8.128 | 201.6 | quant | (1064960, 1, 1) (266240, 1, 1) (34078720, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb0ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 20.0 | 7.910 | 395.4 | mhc | (262144, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb1ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 18.0 | 7.477 | 415.0 | mhc | (262144, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_4x4_row_wb32x32_k5120` | 20.0 | 6.400 | 318.6 | quant | (589824, 1, 1) |
| `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 25.6 | 4.201 | 184.3 | misc | (1003520, 1, 1) (1092608, 1, 1) (1187840, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x8_row_wb32x32_k6144` | 2.0 | 3.924 | 1957.6 | quant | (3276800, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x4_row_wb32x32_k2304` | 20.0 | 3.414 | 170.7 | quant | (655360, 1, 1) |
| `_kernel` | 131.0 | 2.932 | 11.1 | misc | (12288, 10, 1) (12288, 16, 1) (12288, 3, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k5120` | 21.0 | 2.730 | 127.0 | quant | (229376, 1, 1) |

### SAMPLE_after_MIXED: 20 records; ; GPU span median 5.227076500000001; kernel ms/record 4.84; graph-replayed 2128/3218
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 5.1 | 1.209 | 394.9 | quant | (25088, 1, 1) (34041344, 1, 1) (34053888, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k15360` | 1.0 | 0.785 | 785.9 | quant | (655360, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_11FillFunctorIN3c108BFloat16EEESt5arr` | 5.1 | 0.440 | 140.4 | norm_rope_elem | (49152, 1, 1) (67035136, 1, 1) (67059712, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k5120` | 3.0 | 0.379 | 127.2 | quant | (229376, 1, 1) |
| `_ZN2at6native12_GLOBAL__N_126CatArrayBatchedCopy_contigINS1_10OpaqueTypeILj2EEEjLi2ELi128E` | 1.7 | 0.355 | 347.9 | norm_rope_elem | (11520, 3, 1) (2097152, 3, 1) (7680, 3, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x128_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGS` | 1.7 | 0.337 | 197.5 | gemm | (129280, 1, 1) |
| `_kernel` | 13.1 | 0.330 | 5.0 | misc | (384, 10, 1) (384, 16, 1) (384, 3, 1) |
| `__amd_rocclr_copyBuffer` | 7.9 | 0.093 | 4.6 | norm_rope_elem | (1024, 1, 1) (11776, 1, 1) (131072, 1, 1) |
| `_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi4ELi16ELi8ELi2ELi2EEviiiiiiPKT_S3_S3_PS1_ii` | 3.0 | 0.065 | 21.6 | gemm | (16384, 16, 1) |
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 4.2 | 0.065 | 15.9 | comm | (2048, 1, 1) (2560, 1, 1) (2816, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_bf16_t16x128x256_pm1_async_gui_v33` | 2.1 | 0.057 | 26.9 | moe | (4608, 60, 1) (4608, 90, 1) |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb0ELb0ELb1ELi1ELb0EEEvPT0_PT_PfS` | 2.4 | 0.056 | 5.1 | quant | (2560, 1, 1) (3072, 1, 1) (3840, 1, 1) |
| `mfma_moe2_afp8_wfp4_bf16_cshuffle_t16x256x128_vscale_fix3_fp4opt_v1_pm1_abuffer` | 2.1 | 0.042 | 19.9 | moe | (5120, 60, 1) (5120, 90, 1) |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi64ELi8ELb0ELb0ELb1ELi1ELb0EEEvPT0_PT_PfS4_` | 5.1 | 0.031 | 6.9 | quant | (1047424, 1, 1) (1047808, 1, 1) (1048576, 1, 1) |
| `_ZN2at6native32elementwise_kernel_manual_unrollILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZ` | 7.0 | 0.031 | 4.5 | norm_rope_elem | (128, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi5ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 2.1 | 0.024 | 11.1 | mhc | (81920, 1, 1) |
| `_gumbel_sample_kernel` | 4.5 | 0.024 | 5.0 | sampling | (1024, 127, 1) (1280, 127, 1) (512, 127, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 2.1 | 0.023 | 10.7 | attention | (2560, 4, 4) (3840, 4, 4) |
| `bmm_a8w8_mxscale_t16x32x256_w1x2_nb5_sk4_nt_sps_b1_s32_32x32` | 3.9 | 0.021 | 5.3 | gemm | (28672, 1, 1) |
| `kernel` | 4.2 | 0.021 | 4.9 | misc | (2560, 1, 1) (3840, 1, 1) (5120, 2, 1) |

