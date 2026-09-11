# GPU kernel distribution

Sum of GPU kernel durations across selected ranks and traces; not wall-clock latency.

| Category | GPU time (%) | Summed GPU ms | Calls |
|---|---:|---:|---:|
| MoE expert GEMM | 29.62 | 1215.87 | 15360 |
| Communication / fused allreduce | 18.01 | 739.28 | 32760 |
| Dense GEMM / linear | 17.30 | 710.18 | 48000 |
| Attention / MLA | 16.40 | 673.13 | 23904 |
| DSA index / top-k | 6.91 | 283.73 | 11904 |
| MoE routing / sorting | 5.01 | 205.65 | 37440 |
| Tensor / elementwise / indexing | 3.83 | 157.36 | 16512 |
| Norm / RoPE / KV write | 2.07 | 84.94 | 14304 |
| Quantization / cast | 0.76 | 31.34 | 3936 |
| Sampling / speculative bookkeeping | 0.08 | 3.29 | 528 |

## Speculative stage attribution

Only ID-linked GPU events are attributed. CPU ranges are not GPU durations.

| Stage | GPU time (%) | Summed GPU ms |
|---|---:|---:|
| verify | 83.61 | 3431.97 |
| draft | 9.71 | 398.77 |
| draft_extend | 4.67 | 191.61 |
| unattributed | 2.01 | 82.42 |

## Top kernels

| Kernel | GPU time (%) | Mean us | Calls |
|---|---:|---:|---:|
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t32x128x256_pm1_async_v33` | 16.24 | 92.57 | 7200 |
| `main_kernel` | 13.54 | 36.64 | 15168 |
| `_ZN5aiter33reduce_scatter_cross_device_storeIDF16bLi4EEEvPNS_8RankDataENS_11RankSignalsEPNS_6SignalEiiii` | 13.04 | 34.84 | 15360 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x128x256_vscale_fix3_fp4opt_v1_persist_cu256` | 8.76 | 49.93 | 7200 |
| `hgemm_bf16_128x96x64x4_SPK2_W4x2x2_BLDS1_TN_AS1_0` | 4.15 | 22.49 | 7584 |
| `hgemm_bf16_128x64x64x4_SPK3_W2x4x1_BLDS1_TN_AS1_0` | 3.17 | 17.13 | 7584 |
| `hgemm_bf16_64x64x64x4_SPK1_W1x4x1_BLDS1_TN_AS1_0` | 2.70 | 11.42 | 9696 |
| `_fused_fp8_bmm_rope_cat_and_cache_mla_kernel_BLOCK_SIZE_M_64_BLOCK_SIZE_N_256_BLOCK_SIZE_K_128_GROUP_SIZE_M_1_QH_PER_KH_16_REUSE_FREQS_FRONT…` | 2.62 | 14.39 | 7488 |
| `void aiter::coop::coop_topk_kernel<2048u, 12u, 4096u, 1024u>(aiter::coop::CoopTopKParams<2048u>)` | 2.19 | 42.49 | 2112 |
| `_ZN2ck15kernel_moe_gemmINS_15GridwiseMoeGemmINS_13tensor_layout4gemm8RowMajorENS3_11ColumnMajorENS_5TupleIJS4_S5_S4_EEES4_DF16bDF16bffNS6_IJ…` | 2.09 | 223.55 | 384 |
| `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` | 2.04 | 39.56 | 2112 |
| `_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel_HAS_BIAS_0_BLOCK_SIZE_M_64_BLOCK_SIZE_N_256_BLOCK_SIZE_K_128…` | 1.95 | 10.68 | 7488 |
| `_ZN5aiter25local_device_load_rmsnormIDF16bLi512ELi2ELb0EEEvNS_11RankSignalsEPT_S3_S3_S3_fiiii` | 1.90 | 5.07 | 15360 |
| `_ZN5aiter13allgather_vecIDF16bLi4EEEvPNS_8RankDataENS_11RankSignalsEPNS_6SignalEPT_ii` | 1.75 | 124.97 | 576 |
| `_fused_dsa_target_verify_metadata_kernel` | 1.41 | 601.67 | 96 |
| `hgemm_bf16_32x64x128x5_SPK6_W1x4x1_BLDS1_TN_AS1_0` | 1.29 | 7.24 | 7296 |
| `_ZN5aiter26cross_device_reduce_2stageIDF16bLi4ELb0EEEvPNS_8RankDataES2_NS_11RankSignalsEPNS_6SignalEPT_ii` | 1.26 | 44.78 | 1152 |
| `_ZN5aiter30fused_mx_quant_moe_sort_kernelIDF16bN4opus5fp4_tELi256ELi32EEEvPT0_PhPKT_PKiSA_PKfiiiiiiiii` | 1.24 | 7.04 | 7200 |
| `_ZN5aiter30fused_mx_quant_moe_sort_kernelIDF16bN4opus5fp4_tELi256ELi8EEEvPT0_PhPKT_PKiSA_PKfiiiiiiiii` | 1.20 | 6.85 | 7200 |
| `_ZN2ck15kernel_moe_gemmINS_15GridwiseMoeGemmINS_13tensor_layout4gemm8RowMajorENS3_11ColumnMajorENS_5TupleIJS4_S5_S4_EEES4_DF16bDF16bffNS6_IJ…` | 1.11 | 118.82 | 384 |
