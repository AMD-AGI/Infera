# Calibrated-profiler hot-op tables (vLLM engines)

ms = summed kernel durations (streams overlap -> shares are of GPU kernel time, not wall time).

## role agg: 61357.6 kernel-ms in the windows; mHC bucket rank 7, first mHC kernel rank 20

stage kernel-ms: DECODE 35584.2, MIXED 23374.8, SAMPLE_after_DECODE 1986.5, SAMPLE_after_MIXED 412.2

| bucket | kernel ms | share |
|---|---|---|
| comm | 20455.92 | 33.3% |
| moe | 11391.71 | 18.6% |
| attention | 8710.53 | 14.2% |
| gemm | 8414.34 | 13.7% |
| misc | 4939.28 | 8.0% |
| quant | 3653.62 | 6.0% |
| mhc | 1894.39 | 3.1% |
| norm_rope_elem | 985.56 | 1.6% |
| indexer | 794.52 | 1.3% |
| sampling | 117.77 | 0.2% |

| # | kernel | ms | share | bucket |
|---|---|---|---|---|
| 1 | `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 20455.92 | 33.34% | comm |
| 2 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 6534.03 | 10.65% | moe |
| 3 | `_sparse_attn_decode_gfx950_partial_kernel` | 4936.08 | 8.04% | attention |
| 4 | `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16lds_v2_biasa` | 3533.71 | 5.76% | gemm |
| 5 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 3139.13 | 5.12% | moe |
| 6 | `_Z45opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernelI45opus_mla_v4_prefill_a16w16_16mx8_32nx1_traitsILi` | 2917.93 | 4.76% | attention |
| 7 | `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf16lds_v2_b` | 1672.27 | 2.73% | gemm |
| 8 | `_engram_lookup_kernel` | 777.32 | 1.27% | misc |
| 9 | `_inverse_rope_mxfp8_quant_kernel` | 775.34 | 1.26% | quant |
| 10 | `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFloat16ELi64` | 656.59 | 1.07% | quant |
| 11 | `flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32` | 597.90 | 0.97% | misc |
| 12 | `_kernel` | 585.62 | 0.95% | misc |
| 13 | `_sparse_attn_decode_reduce_kernel` | 539.77 | 0.88% | attention |
| 14 | `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 489.77 | 0.80% | misc |
| 15 | `flydsl_mxpsh_192x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf2` | 441.39 | 0.72% | misc |
| 16 | `bmm_a8w8_mxscale_t96x128x128_w2x2_nb4_sk1_nt_xcd_b8_s32_32x32` | 436.29 | 0.71% | gemm |
| 17 | `mfma_moe1_silu_mul_afp8_wfp4_bf16_t64x128x256_pm1_async_gui_v33` | 425.26 | 0.69% | moe |
| 18 | `_dequantize_and_gather_k_one_pass_kernel` | 418.70 | 0.68% | quant |
| 19 | `bmm_a8w8_mxscale_t256x256x128_w1x4_nb2_sk1_bd1_xcd_b8_s32_32x32` | 415.22 | 0.68% | gemm |
| 20 | `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_PKfS5_S5_S` | 399.52 | 0.65% | mhc |
| 21 | `_ZN4vllm3moe22topkGatingSoftplusSqrtILi6ELi384ELi4ELi8ELi64ELb0EiifEEvPKT7_PKbPfiPT5_PiiiibdPKfPKT6_` | 394.69 | 0.64% | moe |
| 22 | `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb1ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_PKfS5_S5_S` | 369.68 | 0.60% | mhc |
| 23 | `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_2x8_row_wb32x32_k8192` | 315.52 | 0.51% | quant |
| 24 | `kernel` | 258.28 | 0.42% | misc |
| 25 | `_ZN5aiter22opus_moe_sorting_entryINS_30MoeSortingMultiPhaseKernel_P23INS_19MoeSortingProblemMpIifhLi` | 257.75 | 0.42% | moe |
| 26 | `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb0ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16bS3_PKfS5_` | 257.30 | 0.42% | mhc |
| 27 | `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb1ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16bS3_PKfS5_` | 245.95 | 0.40% | mhc |
| 28 | `mfma_moe2_afp8_wfp4_bf16_cshuffle_t64x128x128_vscale_fix3_fp4opt_v1_pm1_abuffer` | 242.86 | 0.40% | moe |
| 29 | `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_64_BLOCK_KV_64_BLOCK_M_3_PRESHUFFLE` | 236.24 | 0.39% | indexer |
| 30 | `_sparse_attn_prefill_ragged_kernel` | 234.67 | 0.38% | attention |

## agg0 pid 1061 dp_rank 1: window 4886 ms (steps), 127981 dispatches, unattributed 0 (0.0 ms), outside steps 0 (0.0 ms), missing symbols 0, causality violations 0
### DECODE: 77 records; reqs 30[29-34] tokens 180[174-204] prefill_tokens 0[0-0] prefill_reqs 0[0-0] spec_tokens 150[145-170] max_q 6[6-6]; GPU span median 42.341639; kernel ms/record 47.33; graph-replayed 95374/104955
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 11.684 | 90.1 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 38.4 | 10.062 | 256.5 | moe | (4608, 479, 1) (4608, 481, 1) (4608, 488, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 40.0 | 5.585 | 146.9 | attention | (44544, 1, 4) (46080, 1, 4) (64512, 1, 4) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 38.4 | 5.455 | 140.6 | gemm | (2391040, 1, 1) (2401280, 1, 1) (2437120, 1, 1) |
| `_sparse_attn_decode_reduce_kernel` | 40.0 | 0.720 | 17.8 | attention | (44544, 64, 1) (46080, 64, 1) (64512, 64, 1) |
| `flydsl_mxpsh_192x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf2` | 31.2 | 0.709 | 21.8 | misc | (1024, 128, 1) (512, 128, 1) |
| `_kernel` | 131.0 | 0.653 | 5.1 | misc | (10368, 10, 1) (10368, 12, 1) (10368, 16, 1) |
| `bmm_a8w8_mxscale_t96x128x128_w2x2_nb4_sk1_nt_xcd_b8_s32_32x32` | 33.2 | 0.640 | 19.2 | gemm | (49152, 1, 1) (65536, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 33.2 | 0.569 | 17.0 | mhc | (122880, 1, 1) |
| `_ZN4vllm3moe22topkGatingSoftplusSqrtILi6ELi384ELi4ELi8ELi64ELb0EiifEEvPKT7_PKbPfiPT5_Piiii` | 40.0 | 0.551 | 13.2 | moe | (11520, 4, 1) (11776, 4, 1) (2816, 4, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb1ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 30.8 | 0.547 | 17.8 | mhc | (122880, 1, 1) |
| `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT32x16x1024_MI16x16x1_SN_LDSB1_AFC0_AG0_AG` | 38.6 | 0.542 | 12.4 | gemm | (33792, 1, 1) (36864, 1, 1) (45056, 1, 1) |
| `flydsl_mxpsh_96x128x256_F8_F8_B16_w2_x4_sk2_slrr_wb32x32_pf3` | 24.9 | 0.536 | 21.5 | misc | (768, 40, 2) |
| `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 8.0 | 0.509 | 46.5 | misc | (178176, 1, 1) (184320, 1, 1) (258048, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_bf16_t64x128x256_pm1_async_gui_v33` | 1.3 | 0.472 | 352.3 | moe | (4608, 519, 1) (4608, 522, 1) (4608, 538, 1) |
| `flydsl_mxpsh_48x128x256_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf3` | 24.9 | 0.443 | 17.8 | misc | (1536, 36, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 0.426 | 10.2 | quant | (1064960, 1, 1) (1497600, 1, 1) (1530880, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 0.3 | 0.312 | 1206.1 | moe | (4608, 833, 1) |
| `_ZN5aiter22opus_moe_sorting_entryINS_30MoeSortingMultiPhaseKernel_P23INS_19MoeSortingProbl` | 35.3 | 0.277 | 7.6 | moe | (229376, 1, 1) |
| `flydsl_mxpsh_32x64x512_F8_F8_B16_w2_x4_sk1_slrr_wb32x32_pf5` | 24.9 | 0.277 | 11.0 | misc | (2304, 28, 1) |

### MIXED: 3 records; reqs 33[30-34] tokens 13138[2306-16384] prefill_tokens 12946[2132-16204] prefill_reqs 2[1-3] spec_tokens 150[145-160] max_q 10865[2132-11131]; GPU span median 367.874129; kernel ms/record 336.46; graph-replayed 1662/6522
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 103.026 | 730.3 | comm | (8192, 1, 1) |
| `_Z45opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernelI45opus_mla_v4_prefill_a16w16_16mx8_32nx1` | 20.0 | 44.660 | 2931.6 | attention | (1091584, 1, 1) (6628352, 1, 1) (8296448, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 20.0 | 30.737 | 1721.0 | moe | (4608, 1128, 1) (4608, 1248, 1) (4608, 676, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 13.3 | 13.465 | 1012.5 | gemm | (11520000, 1, 1) (12748800, 1, 1) |
| `_inverse_rope_mxfp8_quant_kernel` | 40.0 | 11.456 | 57.9 | quant | (32768, 64, 1) (3314176, 64, 1) (4148224, 64, 1) |
| `_engram_lookup_kernel` | 2.0 | 9.949 | 5208.7 | misc | (65536, 1, 1) |
| `flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32` | 20.0 | 9.084 | 547.1 | misc | (26368, 128, 1) (32768, 128, 1) (4864, 128, 1) |
| `_dequantize_and_gather_k_one_pass_kernel` | 78.0 | 8.653 | 51.1 | quant | (256, 128, 1) (256, 3510, 1) (256, 7019, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_bf16_t64x128x256_pm1_async_gui_v33` | 20.0 | 7.044 | 351.6 | moe | (4608, 504, 1) |
| `bmm_a8w8_mxscale_t256x256x128_w1x4_nb2_sk1_bd1_xcd_b8_s32_32x32` | 13.3 | 6.569 | 504.7 | gemm | (425984, 1, 1) (524288, 1, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 5.503 | 36.6 | quant | (1331200, 1, 1) (27327232, 1, 1) (34078720, 1, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 40.0 | 5.470 | 142.1 | attention | (44544, 1, 4) (46080, 1, 4) (49152, 1, 4) |
| `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_64_BLOCK_KV_64_BLOCK_M_3_` | 14.0 | 5.443 | 416.9 | indexer | (15680, 1, 45) (38784, 1, 17) (39168, 1, 18) |
| `mfma_moe2_afp8_wfp4_bf16_cshuffle_t64x128x128_vscale_fix3_fp4opt_v1_pm1_abuffer` | 20.0 | 3.990 | 199.7 | moe | (10240, 504, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn128_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 6.7 | 3.671 | 550.6 | gemm | (13783040, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_2x8_row_wb32x32_k8192` | 6.7 | 3.319 | 497.6 | quant | (655360, 1, 1) |
| `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 32.7 | 3.307 | 94.5 | misc | (131072, 1, 1) (136192, 1, 1) (178176, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_4x8_row_wb32x32_k8192` | 6.7 | 3.094 | 462.9 | quant | (532480, 1, 1) |
| `mhc_pre_big_fuse_tilelang_kernel` | 1.0 | 3.031 | 4198.4 | mhc | (1261248, 1, 1) (1572864, 1, 1) (221376, 1, 1) |
| `_kernel` | 131.0 | 2.854 | 5.5 | misc | (20480, 10, 1) (20480, 16, 1) (20480, 3, 1) |

### SAMPLE_after_DECODE: 77 records; ; GPU span median 3.028662; kernel ms/record 3.14; graph-replayed 11748/15829
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 6.0 | 0.520 | 62.8 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 3.0 | 0.251 | 85.7 | moe | (4608, 168, 1) (4608, 169, 1) (4608, 172, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x192x64_MI16x16x1_CMS_SN_LDSB0_AFC0_AG1_` | 0.9 | 0.217 | 235.0 | gemm | (129280, 1, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 3.0 | 0.137 | 45.2 | attention | (53760, 1, 4) (55040, 1, 4) (58880, 1, 4) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 0.5 | 0.136 | 274.7 | gemm | (129280, 1, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 3.0 | 0.134 | 46.2 | gemm | (839680, 1, 1) (844800, 1, 1) (860160, 1, 1) |
| `_gumbel_sample_kernel` | 5.0 | 0.076 | 15.3 | sampling | (10752, 127, 1) (11008, 127, 1) (11776, 127, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x48x64_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGSU` | 3.9 | 0.075 | 18.6 | gemm | (129280, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x224x64_MI16x16x1_CMS_SN_LDSB0_AFC0_AG1_` | 0.3 | 0.071 | 248.6 | gemm | (129280, 1, 1) |
| `flydsl_mxpsh_96x256x128_F8_F8_B16_w0_x4_sk4_slrr_wb32x32_pf3` | 3.0 | 0.070 | 20.7 | misc | (768, 20, 4) |
| `_kernel` | 13.0 | 0.062 | 5.1 | misc | (10368, 10, 1) (10368, 30, 1) (6784, 10, 1) |
| `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AG0_AG` | 3.0 | 0.059 | 18.8 | gemm | (28672, 1, 1) (30720, 1, 1) (32768, 1, 1) |
| `__amd_rocclr_copyBuffer` | 11.0 | 0.055 | 4.9 | norm_rope_elem | (10240, 1, 1) (10752, 1, 1) (11264, 1, 1) |
| `_ZN2at6native32elementwise_kernel_manual_unrollILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZ` | 10.0 | 0.049 | 4.9 | norm_rope_elem | (128, 1, 1) |
| `_sparse_attn_decode_reduce_kernel` | 3.0 | 0.047 | 15.6 | attention | (53760, 64, 1) (55040, 64, 1) (58880, 64, 1) |
| `bmm_a8w8_mxscale_t256x128x128_w4x2_nb3_sk1_nt_b1_s32_32x32` | 2.8 | 0.047 | 16.8 | gemm | (512, 256, 1) |
| `bmm_a8w8_mxscale_t128x64x128_w2x2_nb4_sk1_xcd_b8_s32_32x32` | 2.8 | 0.044 | 15.9 | gemm | (65536, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi6ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 2.8 | 0.040 | 14.6 | mhc | (131072, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC0_AG1_` | 0.1 | 0.037 | 259.6 | gemm | (129280, 1, 1) |
| `_ZN2at6native24vectorized_gather_kernelILi16ElEEvPcS2_PT0_illllb` | 7.0 | 0.036 | 5.1 | misc | (2688, 1, 1) (2752, 1, 1) (2944, 1, 1) |

### SAMPLE_after_MIXED: 3 records; ; GPU span median 5.887083; kernel ms/record 5.55; graph-replayed 459/675
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 6.0 | 0.783 | 27.5 | quant | (26753024, 1, 1) (33479680, 1, 1) (4222464, 1, 1) |
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 6.0 | 0.567 | 62.6 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x4_row_wb32x32_k15360` | 0.7 | 0.478 | 717.6 | quant | (522240, 1, 1) (645120, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_11FillFunctorIN3c108BFloat16EEESt5arr` | 6.0 | 0.288 | 12.0 | norm_rope_elem | (1130496, 1, 1) (1179648, 1, 1) (52682752, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 1.0 | 0.274 | 274.5 | gemm | (129280, 1, 1) |
| `_kernel` | 17.0 | 0.249 | 5.3 | misc | (411648, 10, 1) (411648, 30, 1) (515072, 10, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 3.0 | 0.249 | 83.9 | moe | (4608, 172, 1) (4608, 173, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k5120` | 2.0 | 0.242 | 122.1 | quant | (182784, 1, 1) (225792, 1, 1) |
| `_ZN2at6native12_GLOBAL__N_126CatArrayBatchedCopy_contigINS1_10OpaqueTypeILj2EEEjLi2ELi128E` | 2.0 | 0.233 | 26.5 | norm_rope_elem | (1299200, 3, 1) (176640, 3, 1) (184320, 3, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x192x64_MI16x16x1_CMS_SN_LDSB0_AFC0_AG1_` | 0.7 | 0.157 | 234.9 | gemm | (129280, 1, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 3.0 | 0.139 | 46.6 | attention | (58880, 1, 4) (61440, 1, 4) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 3.0 | 0.134 | 47.5 | gemm | (860160, 1, 1) (865280, 1, 1) |
| `flydsl_mxpsh_96x256x128_F8_F8_B16_w0_x4_sk4_slrr_wb32x32_pf3` | 4.0 | 0.102 | 21.2 | misc | (768, 20, 4) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x48x64_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGSU` | 5.0 | 0.096 | 18.7 | gemm | (129280, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AG0_AGGS` | 0.3 | 0.093 | 278.6 | gemm | (129280, 1, 1) |
| `__amd_rocclr_copyBuffer` | 12.0 | 0.092 | 5.0 | norm_rope_elem | (11264, 1, 1) (11776, 1, 1) (131072, 1, 1) |
| `_gumbel_sample_kernel` | 5.0 | 0.082 | 16.0 | sampling | (11776, 127, 1) (12288, 127, 1) |
| `bmm_a8w8_mxscale_t96x64x256_w2x2_nb3_sk1_bd2_xcd_b1_s32_32x32` | 3.0 | 0.063 | 20.7 | gemm | (55296, 1, 1) |
| `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AG0_AG` | 3.0 | 0.057 | 18.6 | gemm | (30720, 1, 1) |
| `bmm_a8w8_mxscale_t256x128x128_w4x2_nb3_sk1_nt_b1_s32_32x32` | 3.0 | 0.051 | 16.8 | gemm | (512, 256, 1) |

## agg0 pid 1062 dp_rank 0: window 4885 ms (steps), 128831 dispatches, unattributed 0 (0.0 ms), outside steps 0 (0.0 ms), missing symbols 0, causality violations 0
### DECODE: 73 records; reqs 45[42-52] tokens 270[252-312] prefill_tokens 0[0-0] prefill_reqs 0[0-0] spec_tokens 225[210-260] max_q 6[6-6]; GPU span median 42.317309; kernel ms/record 42.90; graph-replayed 91885/98449
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 40.0 | 10.433 | 255.1 | moe | (4608, 479, 1) (4608, 481, 1) (4608, 488, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 40.0 | 7.808 | 205.0 | attention | (64512, 1, 4) (66048, 1, 4) (70656, 1, 4) |
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 5.823 | 69.0 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 40.0 | 5.655 | 138.4 | gemm | (2391040, 1, 1) (2401280, 1, 1) (2437120, 1, 1) |
| `_sparse_attn_decode_reduce_kernel` | 40.0 | 0.736 | 18.1 | attention | (64512, 64, 1) (66048, 64, 1) (70656, 64, 1) |
| `flydsl_mxpsh_192x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf2` | 31.8 | 0.699 | 21.6 | misc | (512, 128, 1) |
| `bmm_a8w8_mxscale_t96x128x128_w2x2_nb4_sk1_nt_xcd_b8_s32_32x32` | 35.1 | 0.678 | 19.3 | gemm | (49152, 1, 1) (65536, 1, 1) |
| `_kernel` | 131.0 | 0.630 | 4.9 | misc | (10368, 10, 1) (10368, 12, 1) (10368, 16, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 35.1 | 0.611 | 17.3 | mhc | (122880, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb1ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 32.4 | 0.585 | 18.0 | mhc | (122880, 1, 1) |
| `flydsl_mxpsh_96x128x256_F8_F8_B16_w2_x4_sk2_slrr_wb32x32_pf3` | 26.3 | 0.572 | 21.7 | misc | (768, 40, 2) |
| `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT32x16x1024_MI16x16x1_SN_LDSB1_AFC0_AG0_AG` | 40.1 | 0.565 | 12.4 | gemm | (49152, 1, 1) (52224, 1, 1) (55296, 1, 1) |
| `_ZN4vllm3moe22topkGatingSoftplusSqrtILi6ELi384ELi4ELi8ELi64ELb0EiifEEvPKT7_PKbPfiPT5_Piiii` | 40.0 | 0.549 | 13.2 | moe | (4032, 4, 1) (4160, 4, 1) (4416, 4, 1) |
| `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 8.0 | 0.530 | 50.2 | misc | (258048, 1, 1) (264192, 1, 1) (282624, 1, 1) |
| `flydsl_mxpsh_48x128x256_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf3` | 26.3 | 0.468 | 17.8 | misc | (1536, 36, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 0.440 | 10.8 | quant | (524288, 1, 1) (536832, 1, 1) (574208, 1, 1) |
| `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_128_BLOCK_KV_64_BLOCK_M_1` | 4.0 | 0.345 | 85.7 | indexer | (64, 252, 9) (64, 258, 8) (64, 276, 8) |
| `flydsl_mxpsh_32x64x512_F8_F8_B16_w2_x4_sk1_slrr_wb32x32_pf5` | 26.3 | 0.290 | 11.0 | misc | (2304, 28, 1) |
| `_ZN5aiter22opus_moe_sorting_entryINS_30MoeSortingMultiPhaseKernel_P23INS_19MoeSortingProbl` | 35.1 | 0.262 | 7.4 | moe | (229376, 1, 1) |
| `bmm_a8w8_mxscale_t96x64x128_w2x2_nb5_sk1_bd2_xcd_b1_s32_32x32` | 26.3 | 0.257 | 9.8 | gemm | (61440, 1, 1) |

### MIXED: 7 records; reqs 45[43-50] tokens 2033[710-9381] prefill_tokens 1763[438-9129] prefill_reqs 2[1-3] spec_tokens 215[210-245] max_q 1185[438-8809]; GPU span median 174.589409; kernel ms/record 222.43; graph-replayed 5151/13826
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 98.253 | 168.4 | comm | (8192, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 11.4 | 16.610 | 1456.1 | moe | (4608, 1128, 1) (4608, 1248, 1) (4608, 676, 1) |
| `_Z45opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernelI45opus_mla_v4_prefill_a16w16_16mx8_32nx1` | 14.3 | 10.731 | 571.7 | attention | (1264640, 1, 1) (1867264, 1, 1) (4674048, 1, 1) |
| `_engram_lookup_kernel` | 2.0 | 9.752 | 4082.7 | misc | (65536, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_bf16_t64x128x256_pm1_async_gui_v33` | 22.9 | 8.105 | 344.0 | moe | (4608, 504, 1) (4608, 519, 1) (4608, 522, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 8.6 | 7.911 | 987.8 | gemm | (11520000, 1, 1) (12748800, 1, 1) (8499200, 1, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 40.0 | 7.373 | 201.4 | attention | (64512, 1, 4) (66048, 1, 4) (69120, 1, 4) |
| `_dequantize_and_gather_k_one_pass_kernel` | 78.0 | 5.950 | 44.2 | quant | (256, 128, 1) (256, 6308, 1) (256, 6789, 1) |
| `_sparse_attn_prefill_ragged_kernel` | 25.7 | 5.338 | 227.7 | attention | (112128, 4, 1) (117248, 4, 1) (32768, 4, 1) |
| `mfma_moe2_afp8_wfp4_bf16_cshuffle_t64x128x128_vscale_fix3_fp4opt_v1_pm1_abuffer` | 22.9 | 4.623 | 196.2 | moe | (10240, 504, 1) (10240, 519, 1) (10240, 522, 1) |
| `_inverse_rope_mxfp8_quant_kernel` | 40.0 | 3.153 | 20.5 | quant | (112128, 64, 1) (117248, 64, 1) (2337024, 64, 1) |
| `flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32` | 8.6 | 2.030 | 175.6 | misc | (18944, 128, 1) (5632, 128, 1) (7936, 128, 1) |
| `mhc_pre_big_fuse_tilelang_kernel` | 1.0 | 1.838 | 29.9 | mhc | (140832, 1, 1) (195168, 1, 1) (261312, 1, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 1.788 | 20.0 | quant | (1064960, 1, 1) (1331200, 1, 1) (1497600, 1, 1) |
| `kernel` | 97.6 | 1.770 | 5.3 | misc | (1040896, 2, 1) (11008, 1, 1) (11264, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 5.7 | 1.768 | 305.5 | moe | (4608, 576, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn128_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 2.9 | 1.531 | 534.6 | gemm | (13783040, 1, 1) |
| `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 20.6 | 1.206 | 19.2 | misc | (1213440, 1, 1) (131072, 1, 1) (1805312, 1, 1) |
| `bmm_a8w8_mxscale_t128x192x128_w1x4_nb2_sk1_bd1_b1_s32_32x32` | 8.6 | 1.190 | 105.2 | gemm | (18944, 24, 1) (5632, 24, 1) (7936, 24, 1) |
| `_kernel` | 131.0 | 1.048 | 5.2 | misc | (125056, 10, 1) (125056, 12, 1) (125056, 16, 1) |

### SAMPLE_after_DECODE: 73 records; ; GPU span median 3.046423; kernel ms/record 3.13; graph-replayed 11139/14982
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 6.0 | 0.366 | 60.1 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x320x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 1.0 | 0.305 | 317.7 | gemm | (129280, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 3.0 | 0.245 | 82.9 | moe | (4608, 168, 1) (4608, 169, 1) (4608, 172, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 3.0 | 0.159 | 52.0 | attention | (53760, 1, 4) (55040, 1, 4) (58880, 1, 4) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 0.5 | 0.139 | 281.0 | gemm | (129280, 1, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 3.0 | 0.137 | 47.2 | gemm | (839680, 1, 1) (844800, 1, 1) (860160, 1, 1) |
| `_gumbel_sample_kernel` | 5.0 | 0.089 | 17.8 | sampling | (10752, 127, 1) (11008, 127, 1) (11776, 127, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC0_AG1_` | 0.3 | 0.077 | 295.9 | gemm | (129280, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x48x64_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGSU` | 3.9 | 0.075 | 18.6 | gemm | (129280, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x224x64_MI16x16x1_CMS_SN_LDSB0_AFC0_AG1_` | 0.3 | 0.074 | 260.1 | gemm | (129280, 1, 1) |
| `flydsl_mxpsh_96x256x128_F8_F8_B16_w0_x4_sk4_slrr_wb32x32_pf3` | 3.0 | 0.071 | 20.9 | misc | (768, 20, 4) |
| `_kernel` | 13.0 | 0.061 | 5.0 | misc | (10368, 10, 1) (10368, 30, 1) (6784, 10, 1) |
| `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AG0_AG` | 3.0 | 0.060 | 19.2 | gemm | (28672, 1, 1) (30720, 1, 1) (32768, 1, 1) |
| `__amd_rocclr_copyBuffer` | 11.0 | 0.053 | 4.8 | norm_rope_elem | (10240, 1, 1) (10752, 1, 1) (11264, 1, 1) |
| `_ZN2at6native32elementwise_kernel_manual_unrollILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZ` | 10.0 | 0.048 | 4.8 | norm_rope_elem | (128, 1, 1) |
| `_sparse_attn_decode_reduce_kernel` | 3.0 | 0.048 | 15.8 | attention | (53760, 64, 1) (55040, 64, 1) (58880, 64, 1) |
| `bmm_a8w8_mxscale_t256x128x128_w4x2_nb3_sk1_nt_b1_s32_32x32` | 2.8 | 0.046 | 16.8 | gemm | (512, 256, 1) |
| `bmm_a8w8_mxscale_t128x64x128_w2x2_nb4_sk1_xcd_b8_s32_32x32` | 2.8 | 0.045 | 16.1 | gemm | (65536, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi6ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 2.8 | 0.040 | 14.6 | mhc | (131072, 1, 1) |
| `_compute_local_logits_stats_kernel` | 1.0 | 0.038 | 36.9 | misc | (64512, 16, 1) (66048, 16, 1) (67584, 16, 1) |

### SAMPLE_after_MIXED: 7 records; ; GPU span median 4.021509; kernel ms/record 4.67; graph-replayed 1068/1574
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 6.0 | 1.198 | 61.3 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 3.0 | 0.247 | 84.2 | moe | (4608, 169, 1) (4608, 172, 1) (4608, 173, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 6.0 | 0.230 | 11.8 | quant | (18938624, 1, 1) (2452480, 1, 1) (3629824, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 0.7 | 0.201 | 281.1 | gemm | (129280, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x320x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 0.6 | 0.185 | 323.1 | gemm | (129280, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC0_AG1_` | 0.6 | 0.164 | 286.3 | gemm | (129280, 1, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 3.0 | 0.157 | 52.7 | attention | (55040, 1, 4) (58880, 1, 4) (61440, 1, 4) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 3.0 | 0.139 | 47.8 | gemm | (844800, 1, 1) (860160, 1, 1) (865280, 1, 1) |
| `_kernel` | 17.0 | 0.115 | 5.2 | misc | (116224, 10, 1) (116224, 30, 1) (13696, 10, 1) |
| `_gumbel_sample_kernel` | 5.0 | 0.091 | 18.4 | sampling | (11008, 127, 1) (11776, 127, 1) (12288, 127, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_11FillFunctorIN3c108BFloat16EEESt5arr` | 6.0 | 0.090 | 6.8 | norm_rope_elem | (10018816, 1, 1) (1056768, 1, 1) (1130496, 1, 1) |
| `flydsl_mxpsh_96x256x128_F8_F8_B16_w0_x4_sk4_slrr_wb32x32_pf3` | 3.4 | 0.086 | 21.3 | misc | (768, 20, 4) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x48x64_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGSU` | 4.3 | 0.082 | 18.5 | gemm | (129280, 1, 1) |
| `_ZN2at6native12_GLOBAL__N_126CatArrayBatchedCopy_contigINS1_10OpaqueTypeILj2EEEjLi2ELi128E` | 2.0 | 0.072 | 15.3 | norm_rope_elem | (1116928, 3, 1) (1565440, 3, 1) (165120, 3, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x4_row_wb32x32_k15360` | 0.1 | 0.068 | 477.5 | quant | (368640, 1, 1) |
| `__amd_rocclr_copyBuffer` | 12.0 | 0.064 | 4.9 | norm_rope_elem | (10752, 1, 1) (11264, 1, 1) (11776, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k15360` | 0.3 | 0.062 | 215.4 | quant | (102400, 1, 1) (153600, 1, 1) |
| `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AG0_AG` | 3.0 | 0.060 | 18.9 | gemm | (28672, 1, 1) (30720, 1, 1) (32768, 1, 1) |
| `bmm_a8w8_mxscale_t256x128x128_w4x2_nb3_sk1_nt_b1_s32_32x32` | 3.0 | 0.052 | 17.2 | gemm | (512, 256, 1) |
| `bmm_a8w8_mxscale_t128x64x128_w2x2_nb4_sk1_xcd_b8_s32_32x32` | 3.0 | 0.051 | 16.5 | gemm | (65536, 1, 1) |

## agg1 pid 1180 dp_rank 1: window 13729 ms (steps), 134540 dispatches, unattributed 0 (0.0 ms), outside steps 0 (0.0 ms), missing symbols 0, causality violations 0
### DECODE: 68 records; reqs 49.0[42-58] tokens 294.0[252-348] prefill_tokens 0.0[0-0] prefill_reqs 0.0[0-0] spec_tokens 245.0[210-290] max_q 6.0[6-6]; GPU span median 48.285624999999996; kernel ms/record 130.99; graph-replayed 74279/93867
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 78.805 | 91.7 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 34.1 | 9.944 | 288.4 | moe | (4608, 492, 1) (4608, 499, 1) (4608, 506, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 4.7 | 8.397 | 1814.7 | moe | (4608, 1164, 1) (4608, 1165, 1) (4608, 1167, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 40.0 | 8.282 | 214.5 | attention | (64512, 1, 4) (66048, 1, 4) (73728, 1, 4) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 34.1 | 5.380 | 157.2 | gemm | (2457600, 1, 1) (2493440, 1, 1) (2529280, 1, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 4.7 | 4.755 | 1028.1 | gemm | (11888640, 1, 1) (11898880, 1, 1) (11919360, 1, 1) |
| `_sparse_attn_decode_reduce_kernel` | 40.0 | 0.831 | 20.8 | attention | (64512, 64, 1) (66048, 64, 1) (73728, 64, 1) |
| `bmm_a8w8_mxscale_t96x128x128_w2x2_nb4_sk1_nt_xcd_b8_s32_32x32` | 37.4 | 0.749 | 19.9 | gemm | (49152, 1, 1) (65536, 1, 1) |
| `flydsl_mxpsh_192x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf2` | 30.0 | 0.691 | 22.5 | misc | (1024, 128, 1) (512, 128, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 37.4 | 0.689 | 18.4 | mhc | (122880, 1, 1) |
| `_kernel` | 131.0 | 0.644 | 4.9 | misc | (10240, 10, 1) (10240, 12, 1) (10240, 16, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb1ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 34.7 | 0.631 | 18.2 | mhc | (122880, 1, 1) |
| `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 8.0 | 0.622 | 56.9 | misc | (258048, 1, 1) (264192, 1, 1) (294912, 1, 1) |
| `_ZN4vllm3moe22topkGatingSoftplusSqrtILi6ELi384ELi4ELi8ELi64ELb0EiifEEvPKT7_PKbPfiPT5_Piiii` | 40.0 | 0.584 | 14.3 | moe | (12288, 4, 1) (14336, 4, 1) (16384, 4, 1) |
| `bmm_a8w8_mxscale_t128x128x128_w1x4_nb4_sk2_bd2_b1_s32_32x32` | 18.8 | 0.568 | 30.0 | gemm | (61440, 1, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 0.501 | 12.3 | quant | (1064960, 1, 1) (1597440, 1, 1) (1863680, 1, 1) |
| `_ZN5aiter22opus_moe_sorting_entryINS_30MoeSortingMultiPhaseKernel_P23INS_19MoeSortingProbl` | 40.0 | 0.479 | 8.0 | moe | (229376, 1, 1) |
| `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi32ELb0ELi64ELb1ELb0ELi0EL` | 41.2 | 0.445 | 5.4 | quant | (103680, 1, 1) (107520, 1, 1) (113280, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_bf16_t64x128x256_pm1_async_gui_v33` | 1.2 | 0.419 | 351.4 | moe | (4608, 528, 1) (4608, 552, 1) (4608, 576, 1) |
| `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_128_BLOCK_KV_64_BLOCK_M_1` | 4.0 | 0.384 | 96.1 | indexer | (64, 252, 9) (64, 258, 8) (64, 288, 8) |

### MIXED: 12 records; reqs 55.5[44-60] tokens 16384.0[941-16384] prefill_tokens 16048.0[659-16132] prefill_reqs 2.0[1-10] spec_tokens 255.0[210-280] max_q 13395.5[659-16126]; GPU span median 397.7695285; kernel ms/record 379.06; graph-replayed 6699/24123
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 120.033 | 294.1 | comm | (7936, 1, 1) (8192, 1, 1) |
| `_Z45opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernelI45opus_mla_v4_prefill_a16w16_16mx8_32nx1` | 20.0 | 50.794 | 3403.4 | attention | (1121280, 1, 1) (3768320, 1, 1) (5464064, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 18.3 | 40.175 | 1819.6 | moe | (4608, 1045, 1) (4608, 1169, 1) (4608, 1267, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 18.3 | 22.214 | 1000.7 | gemm | (10670080, 1, 1) (11939840, 1, 1) (12943360, 1, 1) |
| `_inverse_rope_mxfp8_quant_kernel` | 33.3 | 13.469 | 254.7 | quant | (131072, 64, 1) (168704, 64, 1) (1884160, 64, 1) |
| `flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32` | 20.0 | 10.537 | 665.3 | misc | (15360, 128, 1) (22016, 128, 1) (28416, 128, 1) |
| `_engram_lookup_kernel` | 2.0 | 9.163 | 4663.0 | misc | (65536, 1, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 40.0 | 8.548 | 217.1 | attention | (64512, 1, 4) (66048, 1, 4) (66304, 1, 4) |
| `bmm_a8w8_mxscale_t256x256x128_w1x4_nb2_sk1_bd1_xcd_b8_s32_32x32` | 15.0 | 7.523 | 521.3 | gemm | (352256, 1, 1) (458752, 1, 1) (524288, 1, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 6.447 | 32.5 | quant | (1064960, 1, 1) (15970304, 1, 1) (1863680, 1, 1) |
| `_dequantize_and_gather_k_one_pass_kernel` | 64.7 | 6.167 | 32.3 | quant | (1024, 128, 1) (1024, 4005, 1) (1024, 8011, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_2x8_row_wb32x32_k8192` | 11.7 | 5.765 | 494.3 | quant | (655360, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb0ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 11.7 | 4.552 | 389.9 | mhc | (262144, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb1ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 10.5 | 4.407 | 419.4 | mhc | (262144, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 13.3 | 4.185 | 309.2 | moe | (4608, 528, 1) (4608, 576, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_4x4_row_wb32x32_k5120` | 11.7 | 3.739 | 320.2 | quant | (589824, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_bf16_t64x128x256_pm1_async_gui_v33` | 8.3 | 3.289 | 394.3 | moe | (4608, 552, 1) (4608, 560, 1) (4608, 561, 1) |
| `mhc_pre_big_fuse_tilelang_kernel` | 1.0 | 3.061 | 3571.8 | mhc | (1056768, 1, 1) (1353408, 1, 1) (1572864, 1, 1) |
| `_sparse_attn_prefill_ragged_kernel` | 13.3 | 2.705 | 129.2 | attention | (131072, 4, 1) (168704, 4, 1) (32768, 4, 1) |
| `_kernel` | 131.0 | 2.487 | 5.7 | misc | (12288, 10, 1) (12288, 16, 1) (12288, 3, 1) |

### SAMPLE_after_DECODE: 68 records; ; GPU span median 3.4949060000000003; kernel ms/record 4.30; graph-replayed 10272/13876
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 6.0 | 1.397 | 74.9 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x320x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 1.5 | 0.488 | 323.0 | gemm | (129280, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 3.0 | 0.282 | 95.5 | moe | (4608, 173, 1) (4608, 176, 1) (4608, 179, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 3.0 | 0.171 | 54.2 | attention | (61440, 1, 4) (65280, 1, 4) (69120, 1, 4) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 3.0 | 0.149 | 49.7 | gemm | (865280, 1, 1) (880640, 1, 1) (896000, 1, 1) |
| `_gumbel_sample_kernel` | 5.0 | 0.098 | 19.5 | sampling | (12288, 127, 1) (13056, 127, 1) (13824, 127, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x64x64_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGSU` | 4.7 | 0.097 | 20.0 | gemm | (129280, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC0_AG1_` | 0.3 | 0.070 | 261.9 | gemm | (129280, 1, 1) |
| `_kernel` | 13.0 | 0.063 | 4.9 | misc | (10368, 10, 1) (10368, 30, 1) (10752, 10, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x352x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 0.2 | 0.059 | 365.8 | gemm | (172544, 1, 1) |
| `_sparse_attn_decode_reduce_kernel` | 3.0 | 0.054 | 18.0 | attention | (61440, 64, 1) (65280, 64, 1) (69120, 64, 1) |
| `__amd_rocclr_copyBuffer` | 11.0 | 0.052 | 4.7 | norm_rope_elem | (11776, 1, 1) (12288, 1, 1) (131072, 1, 1) |
| `flydsl_mxpsh_192x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf2` | 2.2 | 0.047 | 21.7 | misc | (512, 128, 1) |
| `_ZN2at6native32elementwise_kernel_manual_unrollILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZ` | 10.0 | 0.046 | 4.7 | norm_rope_elem | (128, 1, 1) |
| `bmm_a8w8_mxscale_t96x128x128_w2x2_nb4_sk1_nt_xcd_b8_s32_32x32` | 2.2 | 0.042 | 19.4 | gemm | (49152, 1, 1) (65536, 1, 1) |
| `_compute_local_logits_stats_kernel` | 1.0 | 0.041 | 41.0 | misc | (64512, 16, 1) (66048, 16, 1) (67584, 16, 1) |
| `_resample_kernel` | 1.0 | 0.040 | 40.2 | sampling | (10752, 127, 1) (11008, 127, 1) (11264, 127, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 2.2 | 0.038 | 17.8 | mhc | (122880, 1, 1) |
| `bmm_a8w8_mxscale_t128x128x128_w2x4_nb4_sk2_b1_s32_32x32` | 1.3 | 0.038 | 29.4 | gemm | (122880, 1, 1) |
| `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AG0_AG` | 3.0 | 0.036 | 9.5 | gemm | (30720, 1, 1) (32768, 1, 1) (34816, 1, 1) |

### SAMPLE_after_MIXED: 12 records; ; GPU span median 6.932003; kernel ms/record 6.44; graph-replayed 1809/2674
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 6.0 | 0.932 | 14.8 | quant | (1283584, 1, 1) (15196672, 1, 1) (22123008, 1, 1) |
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 6.0 | 0.707 | 78.2 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x4_row_wb32x32_k15360` | 0.8 | 0.614 | 784.9 | quant | (296960, 1, 1) (430080, 1, 1) (552960, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x320x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 1.4 | 0.460 | 327.6 | gemm | (129280, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_11FillFunctorIN3c108BFloat16EEESt5arr` | 6.0 | 0.337 | 6.8 | norm_rope_elem | (1327104, 1, 1) (1376256, 1, 1) (1449984, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 3.0 | 0.290 | 97.0 | moe | (4608, 179, 1) (4608, 181, 1) (4608, 184, 1) |
| `_kernel` | 17.0 | 0.286 | 5.1 | misc | (10368, 10, 1) (10368, 30, 1) (10752, 10, 1) |
| `_ZN2at6native12_GLOBAL__N_126CatArrayBatchedCopy_contigINS1_10OpaqueTypeILj2EEEjLi2ELi128E` | 2.0 | 0.280 | 15.4 | norm_rope_elem | (1351680, 3, 1) (207360, 3, 1) (2097152, 3, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k5120` | 2.2 | 0.267 | 120.3 | quant | (150528, 1, 1) (193536, 1, 1) (225792, 1, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 3.0 | 0.186 | 65.8 | attention | (69120, 1, 4) (71680, 1, 4) (75520, 1, 4) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x352x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 0.5 | 0.182 | 364.4 | gemm | (172544, 1, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 3.0 | 0.155 | 54.2 | gemm | (896000, 1, 1) (906240, 1, 1) (921600, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x64x64_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGSU` | 5.0 | 0.104 | 20.1 | gemm | (129280, 1, 1) |
| `_gumbel_sample_kernel` | 5.0 | 0.102 | 21.1 | sampling | (13824, 127, 1) (14336, 127, 1) (15104, 127, 1) |
| `__amd_rocclr_copyBuffer` | 12.0 | 0.099 | 4.7 | norm_rope_elem | (131072, 1, 1) (13312, 1, 1) (13824, 1, 1) |
| `bmm_a8w8_mxscale_t128x128x128_w2x4_nb4_sk2_b1_s32_32x32` | 2.2 | 0.066 | 29.4 | gemm | (122880, 1, 1) |
| `flydsl_mxpsh_192x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf2` | 3.0 | 0.065 | 21.8 | misc | (512, 128, 1) |
| `bmm_a8w8_mxscale_t96x128x128_w2x2_nb4_sk1_nt_xcd_b8_s32_32x32` | 3.0 | 0.058 | 19.5 | gemm | (49152, 1, 1) (65536, 1, 1) |
| `_sparse_attn_decode_reduce_kernel` | 3.0 | 0.057 | 19.3 | attention | (69120, 64, 1) (71680, 64, 1) (75520, 64, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 3.0 | 0.054 | 18.0 | mhc | (122880, 1, 1) |

## agg1 pid 1181 dp_rank 0: window 13727 ms (steps), 140681 dispatches, unattributed 0 (0.0 ms), outside steps 0 (0.0 ms), missing symbols 0, causality violations 0
### DECODE: 56 records; reqs 57.0[47-60] tokens 342.0[282-360] prefill_tokens 0.0[0-0] prefill_reqs 0.0[0-0] spec_tokens 285.0[235-300] max_q 6.0[6-6]; GPU span median 47.6773145; kernel ms/record 70.84; graph-replayed 67551/75555
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 26.813 | 86.5 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 38.2 | 11.043 | 287.2 | moe | (4608, 492, 1) (4608, 499, 1) (4608, 506, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 40.0 | 8.803 | 231.9 | attention | (73728, 1, 4) (78336, 1, 4) (82944, 1, 4) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 38.2 | 5.826 | 150.4 | gemm | (2457600, 1, 1) (2493440, 1, 1) (2529280, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 1.4 | 2.426 | 1803.5 | moe | (4608, 1169, 1) (4608, 917, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 1.4 | 1.380 | 1038.2 | gemm | (11939840, 1, 1) (9359360, 1, 1) |
| `_sparse_attn_decode_reduce_kernel` | 40.0 | 0.858 | 22.0 | attention | (73728, 64, 1) (78336, 64, 1) (82944, 64, 1) |
| `bmm_a8w8_mxscale_t96x128x128_w2x2_nb4_sk1_nt_xcd_b8_s32_32x32` | 39.6 | 0.788 | 19.8 | gemm | (49152, 1, 1) (65536, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 39.6 | 0.731 | 18.4 | mhc | (122880, 1, 1) |
| `flydsl_mxpsh_192x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf2` | 30.4 | 0.692 | 22.3 | misc | (512, 128, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb1ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 36.7 | 0.673 | 18.3 | mhc | (122880, 1, 1) |
| `bmm_a8w8_mxscale_t128x128x128_w1x4_nb4_sk2_bd2_b1_s32_32x32` | 21.8 | 0.658 | 30.0 | gemm | (61440, 1, 1) |
| `_kernel` | 131.0 | 0.657 | 5.0 | misc | (10368, 10, 1) (10368, 12, 1) (10368, 16, 1) |
| `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 8.0 | 0.645 | 58.7 | misc | (294912, 1, 1) (313344, 1, 1) (331776, 1, 1) |
| `_ZN4vllm3moe22topkGatingSoftplusSqrtILi6ELi384ELi4ELi8ELi64ELb0EiifEEvPKT7_PKbPfiPT5_Piiii` | 40.0 | 0.570 | 14.2 | moe | (14336, 4, 1) (4608, 4, 1) (4928, 4, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 0.502 | 12.5 | quant | (1863680, 1, 1) (599040, 1, 1) (636672, 1, 1) |
| `flydsl_mxpsh_64x128x512_F8_F8_B16_w4_x4_sk1_slrr_wb32x32_pf2` | 21.8 | 0.438 | 20.0 | misc | (1536, 36, 1) |
| `flydsl_mxpsh_64x128x256_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf3` | 31.1 | 0.408 | 10.8 | misc | (1536, 36, 1) (1536, 40, 1) |
| `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_128_BLOCK_KV_64_BLOCK_M_1` | 4.0 | 0.405 | 105.3 | indexer | (64, 288, 8) (64, 306, 7) (64, 324, 7) |
| `_ZN5aiter22opus_moe_sorting_entryINS_30MoeSortingMultiPhaseKernel_P23INS_19MoeSortingProbl` | 40.0 | 0.362 | 7.9 | moe | (229376, 1, 1) |

### MIXED: 24 records; reqs 55.0[49-60] tokens 16384.0[795-16384] prefill_tokens 16060.0[489-16108] prefill_reqs 1.0[1-5] spec_tokens 270.0[230-295] max_q 16051.0[489-16096]; GPU span median 402.061329; kernel ms/record 401.07; graph-replayed 13427/48376
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 126.978 | 763.4 | comm | (7936, 1, 1) (8192, 1, 1) |
| `_Z45opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernelI45opus_mla_v4_prefill_a16w16_16mx8_32nx1` | 18.3 | 58.102 | 3602.1 | attention | (1066496, 1, 1) (5583872, 1, 1) (5774336, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 19.2 | 38.033 | 1797.3 | moe | (4608, 1045, 1) (4608, 1164, 1) (4608, 1165, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 19.2 | 21.020 | 989.4 | gemm | (10670080, 1, 1) (11888640, 1, 1) (11898880, 1, 1) |
| `_inverse_rope_mxfp8_quant_kernel` | 28.3 | 15.246 | 873.1 | quant | (125184, 64, 1) (131072, 64, 1) (131328, 64, 1) |
| `flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32` | 18.3 | 11.646 | 664.3 | misc | (22528, 128, 1) (23296, 128, 1) (32768, 128, 1) |
| `_engram_lookup_kernel` | 2.0 | 9.425 | 4713.6 | misc | (65536, 1, 1) |
| `bmm_a8w8_mxscale_t256x256x128_w1x4_nb2_sk1_bd1_xcd_b8_s32_32x32` | 17.5 | 8.967 | 525.0 | gemm | (360448, 1, 1) (376832, 1, 1) (524288, 1, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 40.0 | 8.888 | 230.5 | attention | (70656, 1, 4) (72192, 1, 4) (73728, 1, 4) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_2x8_row_wb32x32_k8192` | 15.8 | 7.786 | 491.7 | quant | (655360, 1, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 7.096 | 25.0 | quant | (1064960, 1, 1) (1597440, 1, 1) (1863680, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb0ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 15.8 | 6.206 | 391.1 | mhc | (262144, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb1ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 14.2 | 5.877 | 412.1 | mhc | (262144, 1, 1) |
| `_dequantize_and_gather_k_one_pass_kernel` | 54.7 | 5.714 | 60.6 | quant | (1024, 128, 1) (1024, 6126, 1) (1024, 8192, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_4x4_row_wb32x32_k5120` | 15.8 | 5.104 | 321.7 | quant | (589824, 1, 1) |
| `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_64_BLOCK_KV_64_BLOCK_M_3_` | 13.8 | 4.970 | 399.5 | indexer | (104512, 1, 6) (111872, 1, 7) (115392, 1, 7) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 14.2 | 4.199 | 290.7 | moe | (4608, 528, 1) (4608, 576, 1) |
| `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 26.2 | 3.105 | 138.4 | misc | (1173504, 1, 1) (1200128, 1, 1) (1272832, 1, 1) |
| `mhc_pre_big_fuse_tilelang_kernel` | 1.0 | 3.104 | 3976.2 | mhc | (1078080, 1, 1) (1114944, 1, 1) (112128, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x8_row_wb32x32_k6144` | 1.6 | 3.099 | 1960.0 | quant | (3276800, 1, 1) |

### SAMPLE_after_DECODE: 56 records; ; GPU span median 3.4779455; kernel ms/record 3.68; graph-replayed 8451/11390
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 6.0 | 0.691 | 73.1 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x320x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 1.0 | 0.331 | 319.4 | gemm | (129280, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 3.0 | 0.279 | 93.2 | moe | (4608, 173, 1) (4608, 176, 1) (4608, 179, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 3.0 | 0.190 | 65.2 | attention | (61440, 1, 4) (65280, 1, 4) (69120, 1, 4) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT176x384x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 0.4 | 0.166 | 386.0 | gemm | (188160, 1, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 3.0 | 0.148 | 48.2 | gemm | (865280, 1, 1) (880640, 1, 1) (896000, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x352x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 0.3 | 0.118 | 368.1 | gemm | (172544, 1, 1) |
| `_gumbel_sample_kernel` | 5.0 | 0.103 | 20.8 | sampling | (12288, 127, 1) (13056, 127, 1) (13824, 127, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x64x64_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGSU` | 4.6 | 0.093 | 19.4 | gemm | (129280, 1, 1) |
| `_kernel` | 13.0 | 0.065 | 5.1 | misc | (10368, 10, 1) (10368, 30, 1) (10752, 10, 1) |
| `_sparse_attn_decode_reduce_kernel` | 3.0 | 0.055 | 18.5 | attention | (61440, 64, 1) (65280, 64, 1) (69120, 64, 1) |
| `__amd_rocclr_copyBuffer` | 11.0 | 0.054 | 4.9 | norm_rope_elem | (11776, 1, 1) (12288, 1, 1) (131072, 1, 1) |
| `flydsl_mxpsh_192x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf2` | 2.4 | 0.051 | 21.7 | misc | (512, 128, 1) |
| `bmm_a8w8_mxscale_t128x128x128_w2x4_nb4_sk2_b1_s32_32x32` | 1.7 | 0.048 | 29.0 | gemm | (122880, 1, 1) |
| `_ZN2at6native32elementwise_kernel_manual_unrollILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZ` | 10.0 | 0.048 | 4.8 | norm_rope_elem | (128, 1, 1) |
| `bmm_a8w8_mxscale_t96x128x128_w2x2_nb4_sk1_nt_xcd_b8_s32_32x32` | 2.4 | 0.046 | 19.4 | gemm | (49152, 1, 1) (65536, 1, 1) |
| `_compute_local_logits_stats_kernel` | 1.0 | 0.045 | 45.7 | misc | (72192, 16, 1) (73728, 16, 1) (75264, 16, 1) |
| `_resample_kernel` | 1.0 | 0.045 | 46.2 | sampling | (12032, 127, 1) (12288, 127, 1) (12544, 127, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 2.4 | 0.041 | 17.6 | mhc | (122880, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC0_AG1_` | 0.1 | 0.038 | 267.1 | gemm | (129280, 1, 1) |

### SAMPLE_after_MIXED: 24 records; ; GPU span median 6.803390500000001; kernel ms/record 6.61; graph-replayed 3630/5360
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 6.0 | 1.024 | 13.4 | quant | (1655808, 1, 1) (22622208, 1, 1) (23383552, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x4_row_wb32x32_k15360` | 0.9 | 0.678 | 786.8 | quant | (440320, 1, 1) (450560, 1, 1) (645120, 1, 1) |
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 6.0 | 0.632 | 71.5 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_11FillFunctorIN3c108BFloat16EEESt5arr` | 6.0 | 0.375 | 6.7 | norm_rope_elem | (1253376, 1, 1) (1327104, 1, 1) (1376256, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x320x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 1.2 | 0.375 | 321.3 | gemm | (129280, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k5120` | 2.6 | 0.326 | 125.7 | quant | (154112, 1, 1) (157696, 1, 1) (225792, 1, 1) |
| `_kernel` | 17.0 | 0.316 | 5.1 | misc | (10368, 10, 1) (10368, 30, 1) (10752, 10, 1) |
| `_ZN2at6native12_GLOBAL__N_126CatArrayBatchedCopy_contigINS1_10OpaqueTypeILj2EEEjLi2ELi128E` | 2.0 | 0.296 | 15.7 | norm_rope_elem | (1321728, 3, 1) (195840, 3, 1) (207360, 3, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 3.0 | 0.281 | 93.2 | moe | (4608, 176, 1) (4608, 179, 1) (4608, 181, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 3.0 | 0.187 | 64.9 | attention | (65280, 1, 4) (69120, 1, 4) (71680, 1, 4) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x352x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 0.5 | 0.168 | 364.0 | gemm | (172544, 1, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 3.0 | 0.149 | 48.6 | gemm | (880640, 1, 1) (896000, 1, 1) (906240, 1, 1) |
| `__amd_rocclr_copyBuffer` | 12.0 | 0.111 | 4.8 | norm_rope_elem | (12288, 1, 1) (131072, 1, 1) (13312, 1, 1) |
| `_gumbel_sample_kernel` | 5.0 | 0.102 | 20.2 | sampling | (13056, 127, 1) (13824, 127, 1) (14336, 127, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x64x64_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGSU` | 5.0 | 0.100 | 19.4 | gemm | (129280, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC0_AG1_` | 0.3 | 0.078 | 261.6 | gemm | (129280, 1, 1) |
| `_sparse_attn_decode_reduce_kernel` | 3.0 | 0.054 | 18.0 | attention | (65280, 64, 1) (69120, 64, 1) (71680, 64, 1) |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb0ELb0ELb1ELi1ELb0EEEvPT0_PT_PfS` | 3.0 | 0.053 | 5.4 | quant | (120576, 1, 1) (203776, 1, 1) (2784256, 1, 1) |
| `flydsl_mxpsh_192x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf2` | 2.1 | 0.047 | 22.0 | misc | (512, 128, 1) |
| `_ZN2at6native32elementwise_kernel_manual_unrollILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZ` | 10.0 | 0.045 | 4.6 | norm_rope_elem | (128, 1, 1) |

## agg2 pid 2280 dp_rank 1: window 5526 ms (steps), 127572 dispatches, unattributed 0 (0.0 ms), outside steps 0 (0.0 ms), missing symbols 0, causality violations 0
### DECODE: 77 records; reqs 42[38-48] tokens 252[228-288] prefill_tokens 0[0-0] prefill_reqs 0[0-0] spec_tokens 210[190-240] max_q 6[6-6]; GPU span median 48.242783; kernel ms/record 55.59; graph-replayed 94925/105068
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 15.201 | 109.9 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 39.0 | 11.598 | 294.0 | moe | (4608, 506, 1) (4608, 510, 1) (4608, 517, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 40.0 | 6.966 | 165.0 | attention | (61440, 1, 4) (64512, 1, 4) (82944, 1, 4) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 39.0 | 6.312 | 160.0 | gemm | (2529280, 1, 1) (2549760, 1, 1) (2585600, 1, 1) |
| `_sparse_attn_decode_reduce_kernel` | 40.0 | 0.876 | 22.2 | attention | (61440, 64, 1) (64512, 64, 1) (82944, 64, 1) |
| `bmm_a8w8_mxscale_t128x128x128_w1x4_nb4_sk2_bd2_b1_s32_32x32` | 27.5 | 0.833 | 30.1 | gemm | (61440, 1, 1) |
| `bmm_a8w8_mxscale_t96x128x128_w2x2_nb4_sk1_nt_xcd_b8_s32_32x32` | 38.4 | 0.760 | 19.7 | gemm | (65536, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 38.4 | 0.728 | 18.8 | mhc | (122880, 1, 1) |
| `_kernel` | 131.0 | 0.681 | 5.1 | misc | (10368, 10, 1) (10368, 12, 1) (10368, 16, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb1ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 35.6 | 0.668 | 18.7 | mhc | (122880, 1, 1) |
| `flydsl_mxpsh_192x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf2` | 27.5 | 0.634 | 22.4 | misc | (512, 128, 1) |
| `_ZN4vllm3moe22topkGatingSoftplusSqrtILi6ELi384ELi4ELi8ELi64ELb0EiifEEvPKT7_PKbPfiPT5_Piiii` | 40.0 | 0.580 | 14.3 | moe | (10240, 4, 1) (3840, 4, 1) (4032, 4, 1) |
| `flydsl_mxpsh_64x128x512_F8_F8_B16_w4_x4_sk1_slrr_wb32x32_pf2` | 27.5 | 0.553 | 20.0 | misc | (1536, 36, 1) |
| `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 8.0 | 0.517 | 48.5 | misc | (245760, 1, 1) (258048, 1, 1) (331776, 1, 1) |
| `flydsl_mxpsh_64x128x256_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf3` | 38.6 | 0.503 | 10.9 | misc | (1536, 36, 1) (1536, 40, 1) (2048, 32, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 0.492 | 12.1 | quant | (1064960, 1, 1) (1331200, 1, 1) (499200, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 0.3 | 0.466 | 1791.5 | moe | (4608, 1164, 1) |
| `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_128_BLOCK_KV_64_BLOCK_M_1` | 4.0 | 0.334 | 81.6 | indexer | (64, 240, 9) (64, 252, 9) (64, 324, 7) |
| `_ZN5aiter22opus_moe_sorting_entryINS_30MoeSortingMultiPhaseKernel_P23INS_19MoeSortingProbl` | 40.0 | 0.331 | 8.0 | moe | (229376, 1, 1) |
| `flydsl_mxpsh_48x64x512_F8_F8_B16_w2_x4_sk1_slrr_wb32x32_pf3` | 27.5 | 0.327 | 11.8 | misc | (2048, 28, 1) |

### MIXED: 3 records; reqs 43[39-48] tokens 3859[764-4121] prefill_tokens 3619[536-3839] prefill_reqs 1[1-3] spec_tokens 200[190-235] max_q 1918[536-3839]; GPU span median 163.686432; kernel ms/record 218.22; graph-replayed 1702/6140
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 94.595 | 719.3 | comm | (8192, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 20.0 | 19.329 | 800.0 | moe | (4608, 596, 1) (4608, 625, 1) (4608, 918, 1) |
| `_Z45opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernelI45opus_mla_v4_prefill_a16w16_16mx8_32nx1` | 13.3 | 9.820 | 770.8 | attention | (1852928, 1, 1) (1965568, 1, 1) |
| `_engram_lookup_kernel` | 2.0 | 9.031 | 4887.6 | misc | (65536, 1, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 40.0 | 6.802 | 168.8 | attention | (58368, 1, 4) (61440, 1, 4) (72192, 1, 4) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn128_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 13.3 | 6.274 | 470.5 | gemm | (12144640, 1, 1) (12738560, 1, 1) |
| `_dequantize_and_gather_k_one_pass_kernel` | 78.0 | 5.966 | 24.8 | quant | (256, 128, 1) (256, 3034, 1) (256, 6067, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_bf16_t64x128x256_pm1_async_gui_v33` | 13.3 | 4.937 | 367.1 | moe | (4608, 504, 1) (4608, 576, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 6.7 | 4.872 | 730.1 | gemm | (9369600, 1, 1) |
| `_sparse_attn_prefill_ragged_kernel` | 26.7 | 4.760 | 78.1 | attention | (137216, 4, 1) (32768, 4, 1) (98304, 4, 1) |
| `_inverse_rope_mxfp8_quant_kernel` | 40.0 | 3.046 | 20.3 | quant | (137216, 64, 1) (32768, 64, 1) (926464, 64, 1) |
| `mfma_moe2_afp8_wfp4_bf16_cshuffle_t64x128x128_vscale_fix3_fp4opt_v1_pm1_abuffer` | 13.3 | 2.879 | 213.2 | moe | (10240, 504, 1) (10240, 576, 1) |
| `flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32` | 13.3 | 2.413 | 181.5 | misc | (7936, 128, 1) (8448, 128, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 6.7 | 2.261 | 336.9 | moe | (4608, 576, 1) |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb0ELb0ELb1ELi1ELb0EEEvPT0_PT_PfS` | 2.0 | 1.791 | 13.7 | quant | (1054976, 1, 1) (195584, 1, 1) (987904, 1, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 1.763 | 23.6 | quant | (1064960, 1, 1) (1331200, 1, 1) (1589248, 1, 1) |
| `mhc_pre_big_fuse_tilelang_kernel` | 1.0 | 1.627 | 268.6 | mhc | (370464, 1, 1) (395616, 1, 1) (73344, 1, 1) |
| `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_64_BLOCK_KV_64_BLOCK_M_3_` | 6.0 | 1.593 | 315.0 | indexer | (11392, 1, 17) (11456, 1, 18) (25024, 1, 17) |
| `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 22.7 | 1.490 | 29.3 | misc | (1198080, 1, 1) (131072, 1, 1) (1841152, 1, 1) |
| `bmm_a8w8_mxscale_t128x192x128_w1x4_nb2_sk1_bd1_b1_s32_32x32` | 13.3 | 1.435 | 107.8 | gemm | (7936, 24, 1) (8448, 24, 1) |

### SAMPLE_after_DECODE: 77 records; ; GPU span median 3.520746; kernel ms/record 3.59; graph-replayed 11616/15697
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 6.0 | 0.668 | 81.9 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x320x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 1.4 | 0.465 | 326.2 | gemm | (129280, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 3.0 | 0.283 | 94.4 | moe | (4608, 179, 1) (4608, 181, 1) (4608, 184, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 3.0 | 0.157 | 52.6 | attention | (69120, 1, 4) (71680, 1, 4) (75520, 1, 4) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 3.0 | 0.152 | 50.8 | gemm | (896000, 1, 1) (906240, 1, 1) (921600, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x64x64_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGSU` | 5.0 | 0.104 | 20.2 | gemm | (129280, 1, 1) |
| `_gumbel_sample_kernel` | 5.0 | 0.094 | 18.5 | sampling | (13824, 127, 1) (14336, 127, 1) (15104, 127, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC0_AG1_` | 0.3 | 0.089 | 273.0 | gemm | (129280, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 0.2 | 0.069 | 281.1 | gemm | (129280, 1, 1) |
| `_kernel` | 13.0 | 0.069 | 5.2 | misc | (10240, 10, 1) (10240, 16, 1) (10240, 3, 1) |
| `flydsl_mxpsh_192x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf2` | 3.0 | 0.065 | 21.6 | misc | (512, 128, 1) |
| `bmm_a8w8_mxscale_t128x128x128_w2x4_nb4_sk2_b1_s32_32x32` | 2.1 | 0.061 | 28.6 | gemm | (122880, 1, 1) |
| `bmm_a8w8_mxscale_t96x128x128_w2x2_nb4_sk1_nt_xcd_b8_s32_32x32` | 3.0 | 0.058 | 19.2 | gemm | (49152, 1, 1) (65536, 1, 1) |
| `_sparse_attn_decode_reduce_kernel` | 3.0 | 0.057 | 18.8 | attention | (69120, 64, 1) (71680, 64, 1) (75520, 64, 1) |
| `__amd_rocclr_copyBuffer` | 11.0 | 0.056 | 5.1 | norm_rope_elem | (131072, 1, 1) (13312, 1, 1) (13824, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 3.0 | 0.052 | 17.5 | mhc | (122880, 1, 1) |
| `_ZN2at6native32elementwise_kernel_manual_unrollILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZ` | 10.0 | 0.050 | 5.1 | norm_rope_elem | (128, 1, 1) |
| `flydsl_mxpsh_48x128x512_F8_F8_B16_w2_x4_sk1_slrr_wb32x32_pf3` | 2.1 | 0.040 | 18.5 | misc | (1792, 36, 1) |
| `_ZN2at6native24vectorized_gather_kernelILi16ElEEvPcS2_PT0_illllb` | 7.0 | 0.037 | 5.3 | misc | (3456, 1, 1) (3584, 1, 1) (3776, 1, 1) |
| `_compute_local_logits_stats_kernel` | 1.0 | 0.037 | 36.7 | misc | (58368, 16, 1) (59904, 16, 1) (61440, 16, 1) |

### SAMPLE_after_MIXED: 3 records; ; GPU span median 4.536913; kernel ms/record 5.03; graph-replayed 450/667
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 6.0 | 1.365 | 80.7 | comm | (7936, 1, 1) (8192, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x320x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 1.3 | 0.439 | 329.1 | gemm | (129280, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 3.0 | 0.290 | 97.2 | moe | (4608, 184, 1) (4608, 188, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 6.0 | 0.243 | 12.7 | quant | (7290624, 1, 1) (736512, 1, 1) (7773184, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k15360` | 0.7 | 0.184 | 276.5 | quant | (143360, 1, 1) (153600, 1, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 3.0 | 0.160 | 53.1 | attention | (75520, 1, 4) (81920, 1, 4) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 3.0 | 0.154 | 49.2 | gemm | (921600, 1, 1) (942080, 1, 1) |
| `_kernel` | 17.0 | 0.118 | 5.4 | misc | (10240, 10, 1) (10240, 16, 1) (10240, 3, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x64x64_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGSU` | 5.0 | 0.104 | 20.1 | gemm | (129280, 1, 1) |
| `_gumbel_sample_kernel` | 5.0 | 0.097 | 18.3 | sampling | (15104, 127, 1) (16384, 127, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 0.3 | 0.094 | 281.9 | gemm | (129280, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC0_AG1_` | 0.3 | 0.088 | 262.6 | gemm | (129280, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_11FillFunctorIN3c108BFloat16EEESt5arr` | 6.0 | 0.087 | 6.8 | norm_rope_elem | (14356480, 1, 1) (1449984, 1, 1) (15306752, 1, 1) |
| `bmm_a8w8_mxscale_t128x128x128_w2x4_nb4_sk2_b1_s32_32x32` | 3.0 | 0.086 | 28.8 | gemm | (122880, 1, 1) |
| `bmm_a8w8_mxscale_t128x256x128_w1x4_nb4_sk1_bd3_xcd_b1_s32_32x32` | 2.0 | 0.081 | 40.4 | gemm | (51200, 1, 1) (55296, 1, 1) |
| `_ZN2at6native12_GLOBAL__N_126CatArrayBatchedCopy_contigINS1_10OpaqueTypeILj2EEEjLi2ELi128E` | 2.0 | 0.071 | 13.5 | norm_rope_elem | (2097152, 3, 1) (226560, 3, 1) (245760, 3, 1) |
| `__amd_rocclr_copyBuffer` | 12.0 | 0.066 | 5.2 | norm_rope_elem | (131072, 1, 1) (14336, 1, 1) (15360, 1, 1) |
| `flydsl_mxpsh_192x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf2` | 3.0 | 0.065 | 21.7 | misc | (512, 128, 1) |
| `bmm_a8w8_mxscale_t96x128x128_w2x2_nb4_sk1_nt_xcd_b8_s32_32x32` | 3.0 | 0.058 | 19.4 | gemm | (65536, 1, 1) |
| `_sparse_attn_decode_reduce_kernel` | 3.0 | 0.057 | 19.0 | attention | (75520, 64, 1) (81920, 64, 1) |

## agg2 pid 2281 dp_rank 0: window 5526 ms (steps), 128273 dispatches, unattributed 0 (0.0 ms), outside steps 0 (0.0 ms), missing symbols 0, causality violations 0
### DECODE: 75 records; reqs 58[52-64] tokens 348[312-384] prefill_tokens 0[0-0] prefill_reqs 0[0-0] spec_tokens 290[260-320] max_q 6[6-6]; GPU span median 48.224293; kernel ms/record 50.03; graph-replayed 93837/101161
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 39.7 | 11.806 | 293.2 | moe | (4608, 506, 1) (4608, 510, 1) (4608, 517, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 40.0 | 8.963 | 233.2 | attention | (82944, 1, 4) (86016, 1, 4) (90624, 1, 4) |
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 8.175 | 86.8 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 39.7 | 6.428 | 159.7 | gemm | (2529280, 1, 1) (2549760, 1, 1) (2585600, 1, 1) |
| `_sparse_attn_decode_reduce_kernel` | 40.0 | 0.889 | 22.5 | attention | (82944, 64, 1) (86016, 64, 1) (90624, 64, 1) |
| `bmm_a8w8_mxscale_t128x128x128_w1x4_nb4_sk2_bd2_b1_s32_32x32` | 28.5 | 0.876 | 30.5 | gemm | (61440, 1, 1) |
| `bmm_a8w8_mxscale_t96x128x128_w2x2_nb4_sk1_nt_xcd_b8_s32_32x32` | 39.7 | 0.794 | 19.9 | gemm | (65536, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 39.7 | 0.749 | 18.7 | mhc | (122880, 1, 1) |
| `_kernel` | 131.0 | 0.689 | 5.2 | misc | (10368, 10, 1) (10368, 12, 1) (10368, 16, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb1ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 36.7 | 0.685 | 18.5 | mhc | (122880, 1, 1) |
| `flydsl_mxpsh_192x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf2` | 28.5 | 0.660 | 22.5 | misc | (512, 128, 1) |
| `flydsl_mxpsh_64x128x512_F8_F8_B16_w4_x4_sk1_slrr_wb32x32_pf2` | 28.5 | 0.577 | 20.1 | misc | (1536, 36, 1) |
| `_ZN4vllm3moe22topkGatingSoftplusSqrtILi6ELi384ELi4ELi8ELi64ELb0EiifEEvPKT7_PKbPfiPT5_Piiii` | 40.0 | 0.574 | 14.3 | moe | (5184, 4, 1) (5376, 4, 1) (5696, 4, 1) |
| `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 8.0 | 0.557 | 51.3 | misc | (331776, 1, 1) (344064, 1, 1) (362496, 1, 1) |
| `flydsl_mxpsh_64x128x256_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf3` | 39.8 | 0.521 | 11.0 | misc | (1536, 36, 1) (1536, 40, 1) (2048, 32, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 0.520 | 12.8 | quant | (1064960, 1, 1) (674048, 1, 1) (698880, 1, 1) |
| `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_128_BLOCK_KV_64_BLOCK_M_1` | 4.0 | 0.362 | 91.2 | indexer | (64, 324, 7) (64, 336, 7) (64, 354, 6) |
| `flydsl_mxpsh_48x64x512_F8_F8_B16_w2_x4_sk1_slrr_wb32x32_pf3` | 28.5 | 0.340 | 11.8 | misc | (2048, 28, 1) |
| `bmm_a8w8_mxscale_t128x128x128_w1x4_nb4_sk2_bd3_b1_s32_32x32` | 11.2 | 0.333 | 29.6 | gemm | (61440, 1, 1) |
| `_ZN5aiter22opus_moe_sorting_entryINS_30MoeSortingMultiPhaseKernel_P23INS_19MoeSortingProbl` | 40.0 | 0.316 | 7.8 | moe | (229376, 1, 1) |

### MIXED: 5 records; reqs 58[55-60] tokens 3068[1280-16384] prefill_tokens 2726[938-16066] prefill_reqs 2[1-5] spec_tokens 285[260-295] max_q 2726[681-13351]; GPU span median 163.63278; kernel ms/record 231.64; graph-replayed 2790/10744
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 64.073 | 281.3 | comm | (8192, 1, 1) |
| `_Z45opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernelI45opus_mla_v4_prefill_a16w16_16mx8_32nx1` | 16.0 | 25.356 | 753.1 | attention | (1395712, 1, 1) (5279232, 1, 1) (576000, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 12.0 | 15.714 | 1315.0 | moe | (4608, 1164, 1) (4608, 625, 1) (4608, 918, 1) |
| `_engram_lookup_kernel` | 2.0 | 9.860 | 5072.7 | misc | (65536, 1, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 40.0 | 8.835 | 231.3 | attention | (79872, 1, 4) (81408, 1, 4) (87552, 1, 4) |
| `_inverse_rope_mxfp8_quant_kernel` | 40.0 | 6.865 | 32.6 | quant | (155648, 64, 1) (240128, 64, 1) (2639616, 64, 1) |
| `_dequantize_and_gather_k_one_pass_kernel` | 78.0 | 6.849 | 25.9 | quant | (1280, 128, 1) (1280, 5392, 1) (1280, 8192, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 8.0 | 6.763 | 844.5 | gemm | (11888640, 1, 1) (9369600, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_bf16_t64x128x256_pm1_async_gui_v33` | 16.0 | 6.024 | 370.0 | moe | (4608, 504, 1) (4608, 547, 1) (4608, 576, 1) |
| `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_64_BLOCK_KV_64_BLOCK_M_3_` | 15.0 | 5.588 | 408.0 | indexer | (11840, 1, 47) (15232, 1, 47) (22464, 1, 9) |
| `_sparse_attn_prefill_ragged_kernel` | 24.0 | 5.564 | 145.8 | attention | (155648, 4, 1) (240128, 4, 1) (32768, 4, 1) |
| `flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32` | 16.0 | 5.387 | 286.1 | misc | (21248, 128, 1) (2560, 128, 1) (32768, 128, 1) |
| `bmm_a8w8_mxscale_t256x256x128_w1x4_nb2_sk1_bd1_xcd_b8_s32_32x32` | 8.0 | 3.605 | 462.2 | gemm | (344064, 1, 1) (524288, 1, 1) |
| `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 33.2 | 3.568 | 96.0 | misc | (1076224, 1, 1) (1145856, 1, 1) (1152000, 1, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 3.550 | 28.9 | quant | (1064960, 1, 1) (1331200, 1, 1) (2129920, 1, 1) |
| `mfma_moe2_afp8_wfp4_bf16_cshuffle_t64x128x128_vscale_fix3_fp4opt_v1_pm1_abuffer` | 16.0 | 3.450 | 214.1 | moe | (10240, 504, 1) (10240, 547, 1) (10240, 576, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 8.0 | 2.613 | 322.0 | moe | (4608, 576, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t64x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 4.0 | 2.409 | 575.9 | moe | (4608, 696, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn128_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 4.0 | 1.990 | 498.3 | gemm | (12738560, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_2x8_row_wb32x32_k8192` | 4.0 | 1.971 | 492.3 | quant | (655360, 1, 1) |

### SAMPLE_after_DECODE: 75 records; ; GPU span median 3.528826; kernel ms/record 3.55; graph-replayed 11313/15253
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 6.0 | 0.457 | 73.5 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x320x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 1.1 | 0.372 | 327.6 | gemm | (129280, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 3.0 | 0.284 | 95.2 | moe | (4608, 179, 1) (4608, 181, 1) (4608, 184, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 3.0 | 0.198 | 65.6 | attention | (69120, 1, 4) (71680, 1, 4) (75520, 1, 4) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT176x384x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 0.5 | 0.189 | 392.4 | gemm | (188160, 1, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 3.0 | 0.152 | 48.8 | gemm | (896000, 1, 1) (906240, 1, 1) (921600, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x352x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 0.4 | 0.144 | 371.2 | gemm | (172544, 1, 1) |
| `_gumbel_sample_kernel` | 5.0 | 0.105 | 21.0 | sampling | (13824, 127, 1) (14336, 127, 1) (15104, 127, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x64x64_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGSU` | 5.0 | 0.100 | 19.4 | gemm | (129280, 1, 1) |
| `_kernel` | 13.0 | 0.070 | 5.3 | misc | (10240, 10, 1) (10240, 16, 1) (10240, 3, 1) |
| `flydsl_mxpsh_192x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf2` | 3.0 | 0.066 | 21.8 | misc | (512, 128, 1) |
| `bmm_a8w8_mxscale_t128x128x128_w2x4_nb4_sk2_b1_s32_32x32` | 2.2 | 0.063 | 29.0 | gemm | (122880, 1, 1) |
| `bmm_a8w8_mxscale_t96x128x128_w2x2_nb4_sk1_nt_xcd_b8_s32_32x32` | 3.0 | 0.059 | 19.5 | gemm | (49152, 1, 1) (65536, 1, 1) |
| `_sparse_attn_decode_reduce_kernel` | 3.0 | 0.057 | 18.8 | attention | (69120, 64, 1) (71680, 64, 1) (75520, 64, 1) |
| `__amd_rocclr_copyBuffer` | 11.0 | 0.056 | 5.0 | norm_rope_elem | (131072, 1, 1) (13312, 1, 1) (13824, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 3.0 | 0.053 | 17.7 | mhc | (122880, 1, 1) |
| `_ZN2at6native32elementwise_kernel_manual_unrollILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZ` | 10.0 | 0.050 | 5.0 | norm_rope_elem | (128, 1, 1) |
| `_compute_local_logits_stats_kernel` | 1.0 | 0.047 | 46.6 | misc | (79872, 16, 1) (81408, 16, 1) (82944, 16, 1) |
| `_resample_kernel` | 1.0 | 0.046 | 46.6 | sampling | (13312, 127, 1) (13568, 127, 1) (13824, 127, 1) |
| `flydsl_mxpsh_48x128x512_F8_F8_B16_w2_x4_sk1_slrr_wb32x32_pf3` | 2.2 | 0.040 | 18.4 | misc | (1792, 36, 1) |

### SAMPLE_after_MIXED: 5 records; ; GPU span median 4.361672; kernel ms/record 5.08; graph-replayed 753/1115
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 6.0 | 0.540 | 73.6 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 6.0 | 0.491 | 17.2 | quant | (1926144, 1, 1) (21359616, 1, 1) (2302720, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x320x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 1.4 | 0.463 | 329.0 | gemm | (129280, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 3.0 | 0.285 | 94.6 | moe | (4608, 181, 1) (4608, 184, 1) (4608, 187, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x4_row_wb32x32_k15360` | 0.4 | 0.281 | 703.0 | quant | (419840, 1, 1) (645120, 1, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 3.0 | 0.198 | 65.3 | attention | (71680, 1, 4) (75520, 1, 4) (79360, 1, 4) |
| `_kernel` | 17.0 | 0.187 | 5.5 | misc | (10752, 10, 1) (10752, 30, 1) (11392, 10, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_11FillFunctorIN3c108BFloat16EEESt5arr` | 6.0 | 0.182 | 8.1 | norm_rope_elem | (11116544, 1, 1) (1376256, 1, 1) (1449984, 1, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 3.0 | 0.153 | 49.4 | gemm | (906240, 1, 1) (921600, 1, 1) (936960, 1, 1) |
| `_ZN2at6native12_GLOBAL__N_126CatArrayBatchedCopy_contigINS1_10OpaqueTypeILj2EEEjLi2ELi128E` | 2.0 | 0.150 | 19.3 | norm_rope_elem | (1736960, 3, 1) (2097152, 3, 1) (215040, 3, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x352x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 0.4 | 0.149 | 372.9 | gemm | (172544, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k5120` | 1.2 | 0.136 | 114.2 | quant | (146944, 1, 1) (225792, 1, 1) |
| `_gumbel_sample_kernel` | 5.0 | 0.104 | 20.9 | sampling | (14336, 127, 1) (15104, 127, 1) (15872, 127, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x64x64_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGSU` | 5.0 | 0.101 | 19.5 | gemm | (129280, 1, 1) |
| `__amd_rocclr_copyBuffer` | 12.0 | 0.081 | 5.2 | norm_rope_elem | (131072, 1, 1) (13824, 1, 1) (14336, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT176x384x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 0.2 | 0.078 | 391.8 | gemm | (188160, 1, 1) |
| `bmm_a8w8_mxscale_t128x128x128_w2x4_nb4_sk2_b1_s32_32x32` | 2.4 | 0.070 | 29.1 | gemm | (122880, 1, 1) |
| `flydsl_mxpsh_192x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf2` | 3.0 | 0.066 | 21.9 | misc | (512, 128, 1) |
| `bmm_a8w8_mxscale_t96x128x128_w2x2_nb4_sk1_nt_xcd_b8_s32_32x32` | 3.0 | 0.059 | 19.5 | gemm | (49152, 1, 1) (65536, 1, 1) |
| `_sparse_attn_decode_reduce_kernel` | 3.0 | 0.057 | 18.8 | attention | (71680, 64, 1) (75520, 64, 1) (79360, 64, 1) |

## agg3 pid 2911 dp_rank 1: window 6594 ms (steps), 132465 dispatches, unattributed 0 (0.0 ms), outside steps 0 (0.0 ms), missing symbols 0, causality violations 0
### DECODE: 70 records; reqs 46.5[41-50] tokens 279.0[246-300] prefill_tokens 0.0[0-0] prefill_reqs 0.0[0-0] spec_tokens 232.5[205-250] max_q 6.0[6-6]; GPU span median 43.738425500000005; kernel ms/record 44.11; graph-replayed 88399/94632
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 40.0 | 11.044 | 271.7 | moe | (4608, 479, 1) (4608, 481, 1) (4608, 488, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 40.0 | 7.786 | 202.7 | attention | (64512, 1, 4) (66048, 1, 4) (70656, 1, 4) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 40.0 | 6.009 | 149.7 | gemm | (2391040, 1, 1) (2401280, 1, 1) (2437120, 1, 1) |
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 5.894 | 70.3 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `flydsl_mxpsh_192x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf2` | 36.6 | 0.809 | 21.7 | misc | (512, 128, 1) |
| `_sparse_attn_decode_reduce_kernel` | 40.0 | 0.730 | 18.1 | attention | (64512, 64, 1) (66048, 64, 1) (70656, 64, 1) |
| `bmm_a8w8_mxscale_t96x128x128_w2x2_nb4_sk1_nt_xcd_b8_s32_32x32` | 36.6 | 0.711 | 19.4 | gemm | (49152, 1, 1) (65536, 1, 1) |
| `flydsl_mxpsh_96x128x256_F8_F8_B16_w2_x4_sk2_slrr_wb32x32_pf3` | 30.9 | 0.674 | 21.8 | misc | (768, 40, 2) |
| `_kernel` | 131.0 | 0.666 | 5.2 | misc | (8064, 10, 1) (8064, 12, 1) (8064, 16, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 36.6 | 0.627 | 17.2 | mhc | (122880, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb1ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 33.8 | 0.600 | 17.7 | mhc | (122880, 1, 1) |
| `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 8.0 | 0.591 | 55.3 | misc | (258048, 1, 1) (264192, 1, 1) (282624, 1, 1) |
| `flydsl_mxpsh_48x128x256_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf3` | 30.9 | 0.553 | 17.9 | misc | (1536, 36, 1) |
| `_ZN4vllm3moe22topkGatingSoftplusSqrtILi6ELi384ELi4ELi8ELi64ELb0EiifEEvPKT7_PKbPfiPT5_Piiii` | 40.0 | 0.540 | 13.2 | moe | (4032, 4, 1) (4160, 4, 1) (4416, 4, 1) |
| `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT32x16x1024_MI16x16x1_SN_LDSB1_AFC0_AG0_AG` | 40.1 | 0.539 | 12.2 | gemm | (49152, 1, 1) (52224, 1, 1) (55296, 1, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 0.442 | 11.0 | quant | (524288, 1, 1) (536832, 1, 1) (574208, 1, 1) |
| `flydsl_mxpsh_32x64x512_F8_F8_B16_w2_x4_sk1_slrr_wb32x32_pf5` | 30.9 | 0.345 | 11.1 | misc | (2304, 28, 1) |
| `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_128_BLOCK_KV_64_BLOCK_M_1` | 4.0 | 0.342 | 84.6 | indexer | (64, 252, 9) (64, 258, 8) (64, 276, 8) |
| `bmm_a8w8_mxscale_t96x64x128_w2x2_nb5_sk1_bd2_xcd_b1_s32_32x32` | 30.9 | 0.303 | 9.8 | gemm | (61440, 1, 1) |
| `_ZN5aiter22opus_moe_sorting_entryINS_30MoeSortingMultiPhaseKernel_P23INS_19MoeSortingProbl` | 36.6 | 0.279 | 7.6 | moe | (229376, 1, 1) |

### MIXED: 10 records; reqs 49.0[45-52] tokens 12011.5[4186-16384] prefill_tokens 11765.5[3910-16120] prefill_reqs 3.5[2-5] spec_tokens 225.0[205-250] max_q 4585.0[1920-11953]; GPU span median 327.14876649999997; kernel ms/record 327.02; graph-replayed 5760/21235
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 101.838 | 519.3 | comm | (8192, 1, 1) |
| `_Z45opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernelI45opus_mla_v4_prefill_a16w16_16mx8_32nx1` | 20.0 | 43.953 | 2496.4 | attention | (2001920, 1, 1) (2013184, 1, 1) (2019328, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 20.0 | 30.775 | 1594.9 | moe | (4608, 1044, 1) (4608, 1060, 1) (4608, 1163, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 18.0 | 15.907 | 891.7 | gemm | (10659840, 1, 1) (10823680, 1, 1) (11878400, 1, 1) |
| `_inverse_rope_mxfp8_quant_kernel` | 40.0 | 11.940 | 118.0 | quant | (1000960, 64, 1) (1006592, 64, 1) (1009664, 64, 1) |
| `_engram_lookup_kernel` | 2.0 | 9.571 | 4645.0 | misc | (65536, 1, 1) |
| `flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32` | 20.0 | 9.263 | 505.2 | misc | (12032, 128, 1) (24064, 128, 1) (24320, 128, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 40.0 | 7.861 | 204.8 | attention | (62976, 1, 4) (66048, 1, 4) (67584, 1, 4) |
| `_dequantize_and_gather_k_one_pass_kernel` | 78.0 | 6.431 | 47.2 | quant | (1024, 128, 1) (1024, 4434, 1) (1024, 5251, 1) |
| `bmm_a8w8_mxscale_t256x256x128_w1x4_nb2_sk1_bd1_xcd_b8_s32_32x32` | 12.0 | 5.879 | 530.6 | gemm | (385024, 1, 1) (393216, 1, 1) (524288, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_bf16_t64x128x256_pm1_async_gui_v33` | 16.0 | 5.780 | 357.8 | moe | (4608, 504, 1) (4608, 528, 1) (4608, 552, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 5.594 | 60.1 | quant | (1064960, 1, 1) (12421888, 1, 1) (1331200, 1, 1) |
| `_sparse_attn_prefill_ragged_kernel` | 20.0 | 5.035 | 252.1 | attention | (130816, 4, 1) (131072, 4, 1) (163840, 4, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_2x8_row_wb32x32_k8192` | 8.0 | 3.966 | 495.2 | quant | (655360, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb0ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 10.0 | 3.765 | 394.1 | mhc | (192768, 1, 1) (262144, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb1ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16` | 9.0 | 3.644 | 416.7 | mhc | (192768, 1, 1) (262144, 1, 1) |
| `mfma_moe2_afp8_wfp4_bf16_cshuffle_t64x128x128_vscale_fix3_fp4opt_v1_pm1_abuffer` | 16.0 | 3.346 | 207.9 | moe | (10240, 504, 1) (10240, 528, 1) (10240, 552, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_4x8_row_wb32x32_k8192` | 12.0 | 2.984 | 187.4 | quant | (174080, 1, 1) (245760, 1, 1) (481280, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_4x4_row_wb32x32_k5120` | 8.0 | 2.615 | 326.4 | quant | (589824, 1, 1) |
| `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_64_BLOCK_KV_64_BLOCK_M_3_` | 8.1 | 2.350 | 304.3 | indexer | (107584, 1, 7) (121984, 1, 2) (15488, 1, 12) |

### SAMPLE_after_DECODE: 70 records; ; GPU span median 3.0984425; kernel ms/record 3.16; graph-replayed 10680/14363
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 6.0 | 0.373 | 61.5 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x320x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 0.9 | 0.288 | 314.4 | gemm | (129280, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 3.0 | 0.254 | 85.9 | moe | (4608, 168, 1) (4608, 169, 1) (4608, 172, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 0.7 | 0.195 | 277.6 | gemm | (129280, 1, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 3.0 | 0.155 | 51.8 | attention | (53760, 1, 4) (55040, 1, 4) (58880, 1, 4) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 3.0 | 0.139 | 47.6 | gemm | (839680, 1, 1) (844800, 1, 1) (860160, 1, 1) |
| `_gumbel_sample_kernel` | 5.0 | 0.090 | 18.2 | sampling | (10752, 127, 1) (11008, 127, 1) (11776, 127, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x48x64_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGSU` | 4.3 | 0.082 | 18.6 | gemm | (129280, 1, 1) |
| `flydsl_mxpsh_96x256x128_F8_F8_B16_w0_x4_sk4_slrr_wb32x32_pf3` | 3.3 | 0.080 | 21.0 | misc | (768, 20, 4) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC0_AG1_` | 0.2 | 0.069 | 302.5 | gemm | (129280, 1, 1) |
| `_kernel` | 13.0 | 0.062 | 5.2 | misc | (6784, 10, 1) (6784, 16, 1) (6784, 3, 1) |
| `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AG0_AG` | 3.0 | 0.060 | 19.0 | gemm | (28672, 1, 1) (30720, 1, 1) (32768, 1, 1) |
| `__amd_rocclr_copyBuffer` | 11.0 | 0.056 | 5.1 | norm_rope_elem | (10240, 1, 1) (10752, 1, 1) (11264, 1, 1) |
| `bmm_a8w8_mxscale_t256x128x128_w4x2_nb3_sk1_nt_b1_s32_32x32` | 3.0 | 0.051 | 17.0 | gemm | (512, 256, 1) |
| `_ZN2at6native32elementwise_kernel_manual_unrollILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZ` | 10.0 | 0.050 | 5.0 | norm_rope_elem | (128, 1, 1) |
| `bmm_a8w8_mxscale_t128x64x128_w2x2_nb4_sk1_xcd_b8_s32_32x32` | 3.0 | 0.049 | 16.2 | gemm | (65536, 1, 1) |
| `_sparse_attn_decode_reduce_kernel` | 3.0 | 0.048 | 16.0 | attention | (53760, 64, 1) (55040, 64, 1) (58880, 64, 1) |
| `bmm_a8w8_mxscale_t96x64x256_w2x2_nb3_sk1_bd2_xcd_b1_s32_32x32` | 2.1 | 0.044 | 20.9 | gemm | (55296, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi6ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 3.0 | 0.042 | 14.1 | mhc | (131072, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x224x64_MI16x16x1_CMS_SN_LDSB0_AFC0_AG1_` | 0.2 | 0.041 | 260.2 | gemm | (129280, 1, 1) |

### SAMPLE_after_MIXED: 10 records; ; GPU span median 5.790803; kernel ms/record 5.81; graph-replayed 1515/2235
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 6.0 | 0.812 | 55.1 | quant | (11847680, 1, 1) (24363264, 1, 1) (24456704, 1, 1) |
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 6.0 | 0.604 | 64.5 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x4_row_wb32x32_k15360` | 0.7 | 0.472 | 781.5 | quant | (235520, 1, 1) (471040, 1, 1) (645120, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_11FillFunctorIN3c108BFloat16EEESt5arr` | 6.0 | 0.291 | 19.4 | norm_rope_elem | (1130496, 1, 1) (1253376, 1, 1) (1327104, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x320x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 0.9 | 0.284 | 316.2 | gemm | (129280, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 3.0 | 0.260 | 90.2 | moe | (4608, 172, 1) (4608, 176, 1) (4608, 179, 1) |
| `_kernel` | 17.0 | 0.252 | 5.3 | misc | (10368, 10, 1) (10368, 30, 1) (124160, 10, 1) |
| `_ZN2at6native12_GLOBAL__N_126CatArrayBatchedCopy_contigINS1_10OpaqueTypeILj2EEEjLi2ELi128E` | 2.0 | 0.243 | 51.3 | norm_rope_elem | (176640, 3, 1) (195840, 3, 1) (207360, 3, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k5120` | 1.8 | 0.213 | 121.0 | quant | (164864, 1, 1) (225792, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC0_AG1_` | 0.7 | 0.201 | 291.1 | gemm | (129280, 1, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 3.0 | 0.160 | 52.3 | attention | (58880, 1, 4) (65280, 1, 4) (69120, 1, 4) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 3.0 | 0.142 | 48.4 | gemm | (860160, 1, 1) (880640, 1, 1) (896000, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 0.4 | 0.112 | 279.9 | gemm | (129280, 1, 1) |
| `_gumbel_sample_kernel` | 5.0 | 0.095 | 19.4 | sampling | (11776, 127, 1) (13056, 127, 1) (13824, 127, 1) |
| `__amd_rocclr_copyBuffer` | 12.0 | 0.093 | 5.1 | norm_rope_elem | (11264, 1, 1) (12288, 1, 1) (131072, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k15360` | 0.3 | 0.081 | 271.8 | quant | (163840, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x64x64_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGSU` | 3.0 | 0.060 | 19.3 | gemm | (129280, 1, 1) |
| `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AG0_AG` | 3.0 | 0.053 | 18.3 | gemm | (30720, 1, 1) (32768, 1, 1) (34816, 1, 1) |
| `_ZN2at6native32elementwise_kernel_manual_unrollILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZ` | 10.0 | 0.050 | 5.0 | norm_rope_elem | (128, 1, 1) |
| `_sparse_attn_decode_reduce_kernel` | 3.0 | 0.050 | 16.5 | attention | (58880, 64, 1) (65280, 64, 1) (69120, 64, 1) |

## agg3 pid 2912 dp_rank 0: window 6594 ms (steps), 129589 dispatches, unattributed 0 (0.0 ms), outside steps 0 (0.0 ms), missing symbols 0, causality violations 0
### DECODE: 75 records; reqs 40[38-42] tokens 240[228-252] prefill_tokens 0[0-0] prefill_reqs 0[0-0] spec_tokens 200[190-210] max_q 6[6-6]; GPU span median 43.764729; kernel ms/record 64.18; graph-replayed 91229/102644
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 24.642 | 94.5 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 37.9 | 10.462 | 272.2 | moe | (4608, 479, 1) (4608, 481, 1) (4608, 488, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 40.0 | 6.155 | 159.1 | attention | (58368, 1, 4) (59904, 1, 4) (61440, 1, 4) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 37.9 | 5.629 | 145.9 | gemm | (2391040, 1, 1) (2401280, 1, 1) (2437120, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 1.3 | 2.038 | 1770.7 | moe | (4608, 1163, 1) (4608, 1164, 1) (4608, 592, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 1.1 | 1.036 | 1026.3 | gemm | (11878400, 1, 1) (11888640, 1, 1) (9799680, 1, 1) |
| `flydsl_mxpsh_192x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf2` | 34.7 | 0.774 | 21.6 | misc | (1024, 128, 1) (512, 128, 1) |
| `_sparse_attn_decode_reduce_kernel` | 40.0 | 0.726 | 18.0 | attention | (58368, 64, 1) (59904, 64, 1) (61440, 64, 1) |
| `_kernel` | 131.0 | 0.664 | 5.2 | misc | (16384, 10, 1) (16384, 16, 1) (16384, 3, 1) |
| `bmm_a8w8_mxscale_t96x128x128_w2x2_nb4_sk1_nt_xcd_b8_s32_32x32` | 34.1 | 0.661 | 19.3 | gemm | (49152, 1, 1) (65536, 1, 1) |
| `flydsl_mxpsh_96x128x256_F8_F8_B16_w2_x4_sk2_slrr_wb32x32_pf3` | 28.8 | 0.623 | 21.6 | misc | (768, 40, 2) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 34.1 | 0.592 | 17.3 | mhc | (122880, 1, 1) |
| `_ZN4vllm3moe22topkGatingSoftplusSqrtILi6ELi384ELi4ELi8ELi64ELb0EiifEEvPKT7_PKbPfiPT5_Piiii` | 40.0 | 0.583 | 14.1 | moe | (12288, 4, 1) (14336, 4, 1) (3648, 4, 1) |
| `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 8.0 | 0.561 | 51.0 | misc | (233472, 1, 1) (239616, 1, 1) (245760, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb1ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 31.6 | 0.560 | 17.8 | mhc | (122880, 1, 1) |
| `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT32x16x1024_MI16x16x1_SN_LDSB1_AFC0_AG0_AG` | 38.8 | 0.517 | 12.1 | gemm | (46080, 1, 1) (49152, 1, 1) (52224, 1, 1) |
| `flydsl_mxpsh_48x128x256_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf3` | 28.8 | 0.516 | 17.9 | misc | (1536, 36, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 0.431 | 10.4 | quant | (1064960, 1, 1) (1597440, 1, 1) (1863680, 1, 1) |
| `flydsl_mxpsh_32x64x512_F8_F8_B16_w2_x4_sk1_slrr_wb32x32_pf5` | 28.8 | 0.321 | 11.1 | misc | (2304, 28, 1) |
| `_ZN5aiter22opus_moe_sorting_entryINS_30MoeSortingMultiPhaseKernel_P23INS_19MoeSortingProbl` | 36.8 | 0.316 | 7.6 | moe | (229376, 1, 1) |

### MIXED: 5 records; reqs 40[40-42] tokens 4082[2078-10214] prefill_tokens 3848[1838-9986] prefill_reqs 1[1-2] spec_tokens 195[190-200] max_q 3848[1838-8107]; GPU span median 312.940799; kernel ms/record 310.17; graph-replayed 2930/10420
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 80.0 | 141.065 | 1095.6 | comm | (8192, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 20.0 | 31.150 | 1560.4 | moe | (4608, 1044, 1) (4608, 1060, 1) (4608, 1344, 1) |
| `_Z45opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernelI45opus_mla_v4_prefill_a16w16_16mx8_32nx1` | 20.0 | 21.816 | 894.2 | attention | (1591296, 1, 1) (1970176, 1, 1) (3419648, 1, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf` | 20.0 | 17.262 | 863.8 | gemm | (10659840, 1, 1) (10823680, 1, 1) (13731840, 1, 1) |
| `_engram_lookup_kernel` | 2.0 | 9.964 | 4897.1 | misc | (65536, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_bf16_t64x128x256_pm1_async_gui_v33` | 20.0 | 7.240 | 360.1 | moe | (4608, 504, 1) (4608, 528, 1) (4608, 552, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 40.0 | 6.404 | 167.4 | attention | (58368, 1, 4) (59904, 1, 4) (61440, 1, 4) |
| `_inverse_rope_mxfp8_quant_kernel` | 40.0 | 5.703 | 48.9 | quant | (1709824, 64, 1) (2556416, 64, 1) (32768, 64, 1) |
| `flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32` | 20.0 | 4.737 | 179.4 | misc | (14080, 128, 1) (20480, 128, 1) (4352, 128, 1) |
| `_dequantize_and_gather_k_one_pass_kernel` | 78.0 | 4.702 | 42.6 | quant | (256, 128, 1) (256, 4386, 1) (256, 8192, 1) |
| `mfma_moe2_afp8_wfp4_bf16_cshuffle_t64x128x128_vscale_fix3_fp4opt_v1_pm1_abuffer` | 20.0 | 4.156 | 205.9 | moe | (10240, 504, 1) (10240, 528, 1) (10240, 552, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 40.0 | 2.913 | 37.0 | quant | (1331200, 1, 1) (14379264, 1, 1) (1597440, 1, 1) |
| `mhc_pre_big_fuse_tilelang_kernel` | 1.0 | 2.818 | 2717.3 | mhc | (199488, 1, 1) (321408, 1, 1) (391872, 1, 1) |
| `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_64_BLOCK_KV_64_BLOCK_M_3_` | 7.2 | 2.586 | 391.2 | indexer | (18112, 1, 10) (26816, 1, 27) (37248, 1, 19) |
| `_sparse_attn_prefill_ragged_kernel` | 20.0 | 2.060 | 77.1 | attention | (32768, 4, 1) (65536, 4, 1) |
| `kernel` | 103.6 | 1.894 | 5.6 | misc | (10240, 1, 1) (10496, 1, 1) (1063936, 2, 1) |
| `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 21.8 | 1.885 | 70.1 | misc | (1284096, 1, 1) (131072, 1, 1) (1786880, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3` | 8.0 | 1.871 | 232.6 | mhc | (110848, 1, 1) (163584, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb1ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3` | 7.2 | 1.812 | 252.9 | mhc | (110848, 1, 1) (163584, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k8192` | 12.0 | 1.784 | 171.9 | quant | (143360, 1, 1) (163840, 1, 1) (92160, 1, 1) |

### SAMPLE_after_DECODE: 75 records; ; GPU span median 3.093823; kernel ms/record 3.38; graph-replayed 11439/15409
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 6.0 | 0.630 | 63.4 | comm | (7680, 1, 1) (7936, 1, 1) (8192, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 1.4 | 0.399 | 278.8 | gemm | (129280, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 3.0 | 0.259 | 85.4 | moe | (4608, 168, 1) (4608, 169, 1) (4608, 172, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 3.0 | 0.151 | 50.4 | attention | (53760, 1, 4) (55040, 1, 4) (58880, 1, 4) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 3.0 | 0.140 | 45.2 | gemm | (839680, 1, 1) (844800, 1, 1) (860160, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC0_AG1_` | 0.4 | 0.119 | 286.9 | gemm | (129280, 1, 1) |
| `_gumbel_sample_kernel` | 5.0 | 0.086 | 17.4 | sampling | (10752, 127, 1) (11008, 127, 1) (11776, 127, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x48x64_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGSU` | 4.1 | 0.078 | 18.5 | gemm | (129280, 1, 1) |
| `flydsl_mxpsh_96x256x128_F8_F8_B16_w0_x4_sk4_slrr_wb32x32_pf3` | 3.2 | 0.077 | 20.7 | misc | (768, 20, 4) |
| `_kernel` | 13.0 | 0.062 | 5.2 | misc | (10368, 10, 1) (10368, 30, 1) (6784, 10, 1) |
| `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AG0_AG` | 3.0 | 0.059 | 18.9 | gemm | (28672, 1, 1) (30720, 1, 1) (32768, 1, 1) |
| `__amd_rocclr_copyBuffer` | 11.0 | 0.056 | 5.1 | norm_rope_elem | (10240, 1, 1) (10752, 1, 1) (11264, 1, 1) |
| `_ZN2at6native32elementwise_kernel_manual_unrollILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZ` | 10.0 | 0.051 | 5.1 | norm_rope_elem | (128, 1, 1) |
| `bmm_a8w8_mxscale_t256x128x128_w4x2_nb3_sk1_nt_b1_s32_32x32` | 3.0 | 0.050 | 16.9 | gemm | (512, 256, 1) |
| `bmm_a8w8_mxscale_t128x64x128_w2x2_nb4_sk1_xcd_b8_s32_32x32` | 3.0 | 0.048 | 16.2 | gemm | (65536, 1, 1) |
| `_sparse_attn_decode_reduce_kernel` | 3.0 | 0.048 | 15.9 | attention | (53760, 64, 1) (55040, 64, 1) (58880, 64, 1) |
| `bmm_a8w8_mxscale_t96x64x256_w2x2_nb3_sk1_bd2_xcd_b1_s32_32x32` | 2.0 | 0.042 | 20.6 | gemm | (55296, 1, 1) |
| `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi6ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_` | 3.0 | 0.042 | 14.2 | mhc | (131072, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x224x64_MI16x16x1_CMS_SN_LDSB0_AFC0_AG1_` | 0.1 | 0.039 | 266.5 | gemm | (129280, 1, 1) |
| `_ZN2at6native24vectorized_gather_kernelILi16ElEEvPcS2_PT0_illllb` | 7.0 | 0.037 | 5.3 | misc | (2688, 1, 1) (2752, 1, 1) (2944, 1, 1) |

### SAMPLE_after_MIXED: 5 records; ; GPU span median 5.54588; kernel ms/record 5.65; graph-replayed 756/1116
| kernel | calls/rec | ms/rec | median us | bucket | grids |
|---|---|---|---|---|---|
| `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 6.0 | 1.617 | 66.0 | comm | (7680, 1, 1) (8192, 1, 1) |
| `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFl` | 6.0 | 0.409 | 22.4 | quant | (13742592, 1, 1) (20608768, 1, 1) (3748352, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGS` | 1.0 | 0.285 | 283.3 | gemm | (129280, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC0_AG1_` | 1.0 | 0.277 | 280.2 | gemm | (129280, 1, 1) |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 3.0 | 0.267 | 87.3 | moe | (4608, 172, 1) (4608, 176, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_3x4_row_wb32x32_k15360` | 0.4 | 0.209 | 522.0 | quant | (266240, 1, 1) (399360, 1, 1) |
| `_sparse_attn_decode_gfx950_partial_kernel` | 3.0 | 0.154 | 51.0 | attention | (58880, 1, 4) (65280, 1, 4) |
| `_kernel` | 17.0 | 0.150 | 5.4 | misc | (120832, 10, 1) (120832, 30, 1) (211456, 10, 1) |
| `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16ld` | 3.0 | 0.146 | 47.1 | gemm | (860160, 1, 1) (880640, 1, 1) |
| `_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_11FillFunctorIN3c108BFloat16EEESt5arr` | 6.0 | 0.146 | 11.4 | norm_rope_elem | (1130496, 1, 1) (1253376, 1, 1) (12582912, 1, 1) |
| `_ZN2at6native12_GLOBAL__N_126CatArrayBatchedCopy_contigINS1_10OpaqueTypeILj2EEEjLi2ELi128E` | 2.0 | 0.122 | 24.8 | norm_rope_elem | (1153280, 3, 1) (176640, 3, 1) (195840, 3, 1) |
| `_gumbel_sample_kernel` | 5.0 | 0.088 | 17.6 | sampling | (11776, 127, 1) (13056, 127, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k15360` | 0.4 | 0.086 | 215.2 | quant | (122880, 1, 1) (153600, 1, 1) |
| `__amd_rocclr_copyBuffer` | 12.0 | 0.074 | 5.1 | norm_rope_elem | (11264, 1, 1) (12288, 1, 1) (131072, 1, 1) |
| `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_1x4_row_wb32x32_k5120` | 0.6 | 0.064 | 107.3 | quant | (139776, 1, 1) |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x64x64_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGSU` | 3.0 | 0.060 | 19.8 | gemm | (129280, 1, 1) |
| `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AG0_AG` | 3.0 | 0.055 | 18.1 | gemm | (30720, 1, 1) (32768, 1, 1) |
| `bmm_a8w8_mxscale_t256x128x128_w4x2_nb3_sk1_nt_b1_s32_32x32` | 3.0 | 0.052 | 17.1 | gemm | (512, 256, 1) |
| `bmm_a8w8_mxscale_t128x64x128_w2x2_nb4_sk1_xcd_b8_s32_32x32` | 3.0 | 0.051 | 16.8 | gemm | (65536, 1, 1) |
| `_ZN2at6native32elementwise_kernel_manual_unrollILi128ELi4EZNS0_22gpu_kernel_impl_nocastIZZ` | 10.0 | 0.051 | 5.1 | norm_rope_elem | (128, 1, 1) |

