stages=['MIXED', 'SAMPLE_after_MIXED'] ranks=4 step-records=80 kernel-ms=36690.3 exclude_comm=False
| bucket | ms | share |
|---|---|---|
| comm | 10658.5 | 29.0% |
| moe | 5651.5 | 15.4% |
| attention | 5603.8 | 15.3% |
| quant | 4987.5 | 13.6% |
| gemm | 4090.3 | 11.1% |
| misc | 2772.4 | 7.6% |
| mhc | 1657.7 | 4.5% |
| indexer | 725.7 | 2.0% |
| norm_rope_elem | 541.1 | 1.5% |
| sampling | 1.7 | 0.0% |

| # | kernel | ms | share | calls/step | bucket |
|---|---|---|---|---|---|
| 1 | `_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE` | 10658.5 | 29.05% | 80.0 | comm |
| 2 | `_Z45opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernelI45opus_mla_v4_prefill_a16w16_16mx8_32nx1_trai` | 5535.2 | 15.09% | 20.0 | attention |
| 3 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t128x128x256_pm1_fp8q_sort_async_gui_v2out_v33` | 5211.6 | 14.20% | 20.0 | moe |
| 4 | `gemm2_a4w4_port_hmax8192_imax1152_bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf16lds` | 2893.0 | 7.88% | 20.0 | gemm |
| 5 | `_inverse_rope_mxfp8_quant_kernel` | 1420.8 | 3.87% | 22.2 | quant |
| 6 | `flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk1_slrr_wb32x32` | 1085.7 | 2.96% | 20.0 | misc |
| 7 | `bmm_a8w8_mxscale_t256x256x128_w1x4_nb2_sk1_bd1_xcd_b8_s32_32x32` | 836.5 | 2.28% | 20.0 | gemm |
| 8 | `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_2x8_row_wb32x32_k8192` | 782.5 | 2.13% | 20.0 | quant |
| 9 | `_pa_mqa_logits_mxfp4_kernel_NUM_HEADS_32_HEAD_SIZE_128_PAGE_SIZE_64_BLOCK_KV_64_BLOCK_M_3_PRESH` | 710.0 | 1.94% | 22.9 | indexer |
| 10 | `_engram_lookup_kernel` | 660.1 | 1.80% | 2.0 | misc |
| 11 | `_ZN4vllm21deepseek_v4_fused_ops47fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelIN3c108BFloat16` | 647.6 | 1.77% | 40.0 | quant |
| 12 | `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb0ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16bS3_P` | 647.1 | 1.76% | 20.0 | mhc |
| 13 | `_ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb1ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16bS3_P` | 625.1 | 1.70% | 18.0 | mhc |
| 14 | `flydsl_bpreshuffle_mxfp8_8w_256x256x128_F8_F8_B16_4x4_row_wb32x32_k5120` | 511.0 | 1.39% | 20.0 | quant |
| 15 | `radix_topk_one_block_decode_mixed_k512_b1024_w64_v0_s0` | 332.0 | 0.90% | 24.1 | misc |

mHC: bucket rank 7, share 4.52%, kernels 10, first kernel rank 12
  #12 _ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb0ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16bS3_PKfS5_S5_S3_iiPD 647.1 ms (1.76%) calls/step 20.0
  #13 _ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb1ELb1ELb0ELi16ELi3584ELi2EEEvPKDF16bS3_PKfS5_S5_S3_iiPD 625.1 ms (1.70%) calls/step 18.0
  #20 mhc_pre_big_fuse_tilelang_kernel 267.8 ms (0.73%) calls/step 1.0
  #37 _ZN5aiter15mhc_post_kernelIDF16bLi4ELi4ELi1024ELb1EEEvPT_S2_S2_PfS3_iiiii 42.4 ms (0.12%) calls/step 2.0
  #41 _ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi11ELb1ELb0ELb0ELi32ELi0ELi2EEEvPKDF16bS3_PKfS5_S5_S3_iiPDF16 38.5 ms (0.11%) calls/step 2.0
  #59 _ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi1ELb1ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_PKfS5_S5_S3_iiPDF16b 12.1 ms (0.03%) calls/step 15.5
  #60 _ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi1ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_PKfS5_S5_S3_iiPDF16b 11.9 ms (0.03%) calls/step 15.5
  #93 _ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi6ELb0ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_PKfS5_S5_S3_iiPDF16b 5.6 ms (0.02%) calls/step 4.5
  #94 _ZN5aiter12_GLOBAL__N_115mega_mhc_kernelILi5120ELi6ELb1ELb1ELb0ELi16ELi0ELi2EEEvPKDF16bS3_PKfS5_S5_S3_iiPDF16b 5.6 ms (0.02%) calls/step 4.5
  #119 _ZN5aiter15mhc_post_kernelIDF16bLi4ELi4ELi1024ELb0EEEvPT_S2_S2_PfS3_iiiii 1.7 ms (0.00%) calls/step 4.0
