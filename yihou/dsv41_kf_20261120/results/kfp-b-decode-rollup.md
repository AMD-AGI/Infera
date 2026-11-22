stages=['DECODE', 'SAMPLE_after_DECODE'] ranks=8 step-records=571 kernel-ms=37570.7 exclude_comm=False
| bucket | ms | share |
|---|---|---|
| comm | 12588.7 | 33.5% |
| moe | 8249.8 | 22.0% |
| gemm | 6119.8 | 16.3% |
| attention | 4912.8 | 13.1% |
| misc | 2953.9 | 7.9% |
| mhc | 899.0 | 2.4% |
| quant | 648.7 | 1.7% |
| norm_rope_elem | 629.2 | 1.7% |
| indexer | 464.3 | 1.2% |
| sampling | 104.6 | 0.3% |

| # | kernel | ms | share | calls/step | bucket |
|---|---|---|---|---|---|
| 1 | `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 12588.7 | 33.51% | 86.0 | comm |
| 2 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 6319.5 | 16.82% | 41.5 | moe |
| 3 | `_sparse_attn_decode_gfx950_partial_kernel` | 4364.9 | 11.62% | 43.0 | attention |
| 4 | `gemm2_a4w4_port_hmax8192_imax1152_bm32_bn256_bk128_nt_atomic_a8_bhoist_apf_spart4x2_bf16lds_v2_` | 3418.1 | 9.10% | 41.5 | gemm |
| 5 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 935.7 | 2.49% | 1.0 | moe |
| 6 | `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf16lds` | 511.6 | 1.36% | 0.9 | gemm |
| 7 | `_sparse_attn_decode_reduce_kernel` | 482.7 | 1.28% | 43.0 | attention |
| 8 | `bmm_a8w8_mxscale_t96x128x128_w2x2_nb4_sk1_nt_xcd_b8_s32_32x32` | 426.3 | 1.13% | 38.0 | gemm |
| 9 | `flydsl_mxpsh_192x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32_pf2` | 421.2 | 1.12% | 32.7 | misc |
| 10 | `_kernel` | 414.2 | 1.10% | 144.0 | misc |
| 11 | `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_PKfS5` | 390.4 | 1.04% | 38.0 | mhc |
| 12 | `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb1ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_PKfS5` | 361.3 | 0.96% | 34.8 | mhc |
| 13 | `_ZN4vllm3moe22topkGatingSoftplusSqrtILi6ELi384ELi4ELi8ELi64ELb0EiifEEvPKT7_PKbPfiPT5_PiiiibdPKf` | 323.3 | 0.86% | 40.0 | moe |
| 14 | `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 321.4 | 0.86% | 8.0 | misc |
| 15 | `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFloat16` | 284.2 | 0.76% | 43.0 | quant |
| 16 | `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT32x16x1024_MI16x16x1_SN_LDSB1_AFC0_AG0_AGGSUA0` | 210.1 | 0.56% | 27.5 | gemm |
| 17 | `bmm_a8w8_mxscale_t128x128x128_w1x4_nb4_sk2_bd2_b1_s32_32x32` | 205.3 | 0.55% | 11.8 | gemm |
| 18 | `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_128_BLOCK_KV_64_BLOCK_M_1_PRES` | 193.7 | 0.52% | 4.0 | indexer |
| 19 | `flydsl_mxpsh_96x128x256_F8_F8_B16_w2_x4_sk2_slrr_wb32x32_pf3` | 193.5 | 0.51% | 15.6 | misc |
| 20 | `_ZN5aiter22opus_moe_sorting_entryINS_30MoeSortingMultiPhaseKernel_P23INS_19MoeSortingProblemMpI` | 191.6 | 0.51% | 39.3 | moe |

mHC: bucket rank 6, share 2.39%, kernels 15, first kernel rank 11
  #11 _ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_PKfS5_S5_S3_iiPDF16b 390.4 ms (1.04%) calls/step 38.0
  #12 _ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb1ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_PKfS5_S5_S3_iiPDF16b 361.3 ms (0.96%) calls/step 34.8
  #59 _ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi6ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_PKfS5_S5_S3_iiPDF16b 35.2 ms (0.09%) calls/step 4.2
  #65 _ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi6ELb1ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_PKfS5_S5_S3_iiPDF16b 30.1 ms (0.08%) calls/step 3.5
  #75 _ZN5aiter15mhc_post_kernelIDF16bLi4ELi4ELi1024ELb0EEEvPT_S2_S2_PfS3_iiiii 23.3 ms (0.06%) calls/step 6.0
  #96 _ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb1ELb0ELb0ELi16ELi0ELi2EEEvPKDF16bS3_PKfS5_S5_S3_iiPDF16b 15.4 ms (0.04%) calls/step 1.8
  #119 mhc_pre_big_fuse_tilelang_kernel 10.2 ms (0.03%) calls/step 1.0
  #122 _ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi9ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_PKfS5_S5_S3_iiPDF16b 8.7 ms (0.02%) calls/step 0.5
  #125 _ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi9ELb1ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_PKfS5_S5_S3_iiPDF16b 8.4 ms (0.02%) calls/step 0.4
  #161 _ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi8ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_PKfS5_S5_S3_iiPDF16b 3.8 ms (0.01%) calls/step 0.3
  #162 _ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi8ELb1ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_PKfS5_S5_S3_iiPDF16b 3.8 ms (0.01%) calls/step 0.3
  #166 _ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi6ELb1ELb0ELb1ELi16ELi0ELi2EEEvPKDF16bS3_PKfS5_S5_S3_iiPDF16b 3.7 ms (0.01%) calls/step 0.5
  #170 _ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi7ELb1ELb0ELb1ELi16ELi0ELi2EEEvPKDF16bS3_PKfS5_S5_S3_iiPDF16b 3.5 ms (0.01%) calls/step 0.5
  #217 _ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi6ELb1ELb0ELb0ELi16ELi0ELi2EEEvPKDF16bS3_PKfS5_S5_S3_iiPDF16b 1.1 ms (0.00%) calls/step 0.2
  #254 _ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi9ELb1ELb0ELb0ELi16ELi0ELi2EEEvPKDF16bS3_PKfS5_S5_S3_iiPDF16b 0.1 ms (0.00%) calls/step 0.0
