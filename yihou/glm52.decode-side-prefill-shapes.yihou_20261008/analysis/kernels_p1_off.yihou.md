# Kernel → op → input shapes — results/p1_bs16_isl110k_off/profile/decode-1791461625.299177-TP-0-DP-0.trace.json.gz

Per-iteration device time over 2 profiled iteration(s). Phase from host annotations.

| phase | kernel us / iter | share of total |
|---|---|---|
| TARGET_VERIFY (step[VERIFY]) | 51738.2 | 79.4% |
| draft DECODE x4 (draft > step[DRAFT]) | 6906.9 | 10.6% |
| DRAFT_EXTEND_V2 (draft_extend) | 3345.8 | 5.1% |
| draft glue (draft, outside steps) | 2843.4 | 4.4% |
| <none> | 348.1 | 0.5% |

## TARGET_VERIFY (step[VERIFY])

| kernel | launching op | input dims | grid | calls/iter | us/iter | % phase |
|---|---|---|---|---|---|---|
| `gemm1_a4w4_port_h6144_i512_ne257_bm16_iq_sep` | `aiter::fused_moe_` | c10::BFloat16[96, 6144]; c10::Float4_e2m1fn_x2[257, 1024, 3072]; c10::Float4_e2m1fn_x2[257, 6144, 256]; float[96, 9]; int[96, 9]; unsigned char[257, 1024, 192]; unsigned char[257, 6144, 16] | [1180, 1, 1] | 75 | 9034.7 | 17.5% |
| `gemm2_a4w4_port_hmax8192_imax512_bm16_bn128_bk256_nt_atomic_persist_cu256_bhoist_apf_sp...` | `aiter::fused_moe_` | c10::BFloat16[96, 6144]; c10::Float4_e2m1fn_x2[257, 1024, 3072]; c10::Float4_e2m1fn_x2[257, 6144, 256]; float[96, 9]; int[96, 9]; unsigned char[257, 1024, 192]; unsigned char[257, 6144, 16] | [12288, 1, 1] | 75 | 5817.3 | 11.2% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x32x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `aiter::gemm_a16w16 > aten::mm` | c10::BFloat16[24, 16384]; c10::BFloat16[16384, 6144] | [61440, 1, 1] | 78 | 4854.3 | 9.4% |
| `_sparse_mla_split_k_kernel` | `-` |  | [24, 4, 2] | 78 | 4551.4 | 8.8% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::reg_all_gather_into_tensor > record_param_comms` | c10::BFloat16[24, 6144] | [36, 1, 1] | 78 | 4386.1 | 8.5% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::reg_reduce_scatter_tensor > record_param_comms` | c10::BFloat16[96, 6144] | [48, 1, 1] | 78 | 4341.3 | 8.4% |
| `_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel_HAS_BIA...` | `-` |  | [64, 2, 1] | 156 | 3674.2 | 7.1% |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb1ELb0ELb1ELi1EEEvPT0_PT_PfS4...` | `aiter::rmsnorm2d_fwd_with_add > aiter::add_rmsnorm` | c10::BFloat16[24, 6144]; c10::BFloat16[24, 6144]; c10::BFloat16[24, 6144]; c10::BFloat16[24, 6144]; c10::BFloat16[6144] | [24, 1, 1] | 156 | 1691.5 | 3.3% |
| `hgemm_bf16_32x64x128x4_SPK4_W1x4x1_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[24, 6144]; c10::BFloat16[2624, 6144] | [41, 4, 1] | 78 | 1501.9 | 2.9% |
| `_gemm_a16_w16_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_NU...` | `aiter::gemm_a16w16 > aiter::gemm_a16w16_` | c10::BFloat16[24, 2048]; c10::BFloat16[16384, 2048] | [512, 1, 1] | 78 | 1361.4 | 2.6% |
| `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` | `-` |  | [480, 1, 1] | 21 | 1275.5 | 2.5% |
| `_ZN5aiter23fused_qk_rmsnorm_kernelIDF16bLi256ELi8ELb1ELi1EEEvPT_S2_PKS1_S4_S4_S4_ffiiiiiii` | `aiter::_fused_qk_rmsnorm_kernel` | c10::BFloat16[24, 2048]; c10::BFloat16[2048]; c10::BFloat16[24, 512]; c10::BFloat16[512]; c10::BFloat16[24, 2048]; c10::BFloat16[24, 512] | [24, 2, 1] | 78 | 799.7 | 1.5% |
| `hgemm_bf16_16x64x128x5_SPK8_W1x2x1_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[96, 6144]; c10::BFloat16[256, 6144] | [24, 8, 1] | 75 | 632.3 | 1.2% |
| `void aiter::grouped_topk_kernel<hip_bfloat16, float __vector(4), 1, true, true, false>(...` | `aiter::biased_grouped_topk_hip` | c10::BFloat16[96, 256]; c10::BFloat16[256]; float[96, 8]; int[96, 8] | [96, 1, 1] | 75 | 585.2 | 1.1% |
| `Cijk_SB_BiasS_HAS_ScaleAlphaVec_PostGSU16_VW4` | `aiter::gemm_a16w16 > aten::mm` | c10::BFloat16[24, 16384]; c10::BFloat16[16384, 6144] | [36864, 1, 1] | 78 | 571.6 | 1.1% |
| `_sparse_mla_reduce_kernel` | `-` |  | [24, 64, 8] | 78 | 536.6 | 1.0% |
| `void sgl::dsa_topk::coop_mb_scatter<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbPara...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 110080]; int[24]; int[24, 2048]; int[24, 110037]; int[25] | [24, 10, 1] | 10.5 | 409.7 | 0.8% |
| `void sgl::dsa_topk::coop_mb_scatter<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbPara...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 110080]; int[24]; int[24, 2048]; int[24, 110041]; int[25] | [24, 10, 1] | 10.5 | 383.3 | 0.7% |
| `void aiter::mxfp4_moe::moe_sort_quant::sort_quant_kernel_impl<257, 9, 16, 6144, 128, 10...` | `aiter::fused_moe_ > aiter::mxfp4_moe_sort` | int[96, 9]; float[96, 9]; int[4720]; int[295]; int[2]; int[864]; float[4720]; int[4720]; c10::BFloat16[96, 6144]; c10::BFloat16[0]; int[0] | [128, 1, 1] | 75 | 370.6 | 0.7% |
| `_fused_qk_rope_cat_and_cache_mla_kernel` | `aiter::fused_qk_rope_cat_and_cache_mla` | c10::BFloat16[24, 64, 512]; c10::BFloat16[24, 64, 64]; c10::BFloat16[24, 1, 512]; c10::BFloat16[24, 1, 64]; c10::Float8_e4m3fn[2334656, 1, 576]; long int[24]; long int[24]; c10::BFloat16[1048576, 1, 1, 32]; c10::BFloat16[1048576, 1, 1, 32] | [1536, 1, 1] | 78 | 350.0 | 0.7% |
| `_ZN7ck_tile6kentryILi1ENS_14Layernorm2dFwdINS_29Layernorm2dFwdPipelineOnePassINS_29Laye...` | `aiter::layernorm2d_fwd` | c10::BFloat16[24, 128]; c10::BFloat16[128]; c10::BFloat16[128] | [6, 1, 1] | 21 | 348.0 | 0.7% |
| `void aiter::kn_entry_2c_sbhd_cached_indirect_inplace<aiter::OpCachedFwd, 1, true, false...` | `aiter::rope_cached_positions_2c_fwd_impl` | c10::BFloat16[1, 24, 32, 64]; c10::BFloat16[1, 24, 1, 64]; c10::BFloat16[1, 24, 32, 64]; c10::BFloat16[1, 24, 1, 64]; c10::BFloat16[1048576, 1, 1, 32]; c10::BFloat16[1048576, 1, 1, 32]; long int[1, 24] | [3, 1, 1] | 21 | 340.8 | 0.7% |
| `act_quant_kernel__kernel` | `-` |  | [24, 1, 1] | 21 | 274.3 | 0.5% |
| `void sgl::dsa_topk::coop_mb_refine<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParam...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 110080]; int[24]; int[24, 2048]; int[24, 110041]; int[25] | [24, 1, 1] | 10.5 | 260.3 | 0.5% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `aiter::gemm_a16w16 > aten::mm` | c10::BFloat16[24, 6144]; c10::BFloat16[6144, 32] | [1024, 1, 1] | 21 | 236.3 | 0.5% |
| `_fused_append_shared_experts_kernel` | `-` |  | [96, 1, 1] | 75 | 234.9 | 0.5% |
| `void fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<16, 7, c10::B...` | `HadamardTransformFn` | c10::BFloat16[24, 128] | [24, 1, 1] | 21 | 227.5 | 0.4% |
| `void sgl::dsa_topk::coop_mb_refine<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParam...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 110080]; int[24]; int[24, 2048]; int[24, 110037]; int[25] | [24, 1, 1] | 10.5 | 222.2 | 0.4% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::reg_all_gather_into_tensor > record_param_comms` | c10::BFloat16[96, 38720] | [48, 1, 1] | 1 | 218.3 | 0.4% |
| `hgemm_bf16_64x64x128x3_SPK1_W2x2x2_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[96, 6144]; c10::BFloat16[6144, 6144] | [192, 1, 1] | 3 | 193.6 | 0.4% |
| `hgemm_bf16_16x64x128x4_SPK2_W1x2x1_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[24, 2048]; c10::BFloat16[4096, 2048] | [128, 2, 1] | 21 | 185.2 | 0.4% |
| `hgemm_bf16_16x64x128x4_SPK12_W1x2x1_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[24, 6144]; c10::BFloat16[128, 6144] | [4, 12, 1] | 21 | 125.3 | 0.2% |
| `void sgl::dsa_topk::coop_mb_hist1<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParams...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 110080]; int[24]; int[24, 2048]; int[24, 110041]; int[25] | [24, 10, 1] | 10.5 | 121.7 | 0.2% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x96x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `aten::mm` | c10::BFloat16[96, 6144]; c10::BFloat16[6144, 38720] | [51712, 1, 1] | 1 | 121.1 | 0.2% |
| `void fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<16, 7, c10::B...` | `HadamardTransformFn` | c10::BFloat16[24, 32, 128] | [768, 1, 1] | 21 | 119.7 | 0.2% |
| `void sgl::dsa_topk::coop_mb_hist0<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParams...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 110080]; int[24]; int[24, 2048]; int[24, 110037]; int[25] | [24, 10, 1] | 10.5 | 114.8 | 0.2% |
| `hgemm_bf16_64x64x128x3_SPK1_W4x1x1_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[96, 3072]; c10::BFloat16[6144, 3072] | [192, 1, 1] | 3 | 114.7 | 0.2% |
| `void sgl::dsa_topk::coop_mb_hist0<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParams...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 110080]; int[24]; int[24, 2048]; int[24, 110041]; int[25] | [24, 10, 1] | 10.5 | 108.3 | 0.2% |
| `_ZN5aiter32indexer_k_quant_and_cache_kernelIDF16bDB8_LN4vllm18Fp8KVCacheDataTypeE1ELi8E...` | `aiter::indexer_k_quant_and_cache` | c10::BFloat16[24, 128]; c10::Float8_e4m3fn[36479, 64, 132]; long int[24] | [3, 1, 1] | 21 | 78.1 | 0.2% |
| `_ZN7sgl_hip10activation18act_and_mul_kernelI14__hip_bfloat16TnPFT_RKS3_EXadL_Z4siluIS2_...` | `sgl_kernel::silu_and_mul` | c10::BFloat16[96, 3072]; c10::BFloat16[96, 6144] | [96, 1, 1] | 3 | 70.5 | 0.1% |

## draft DECODE x4 (draft > step[DRAFT])

| kernel | launching op | input dims | grid | calls/iter | us/iter | % phase |
|---|---|---|---|---|---|---|
| `_ZN2ck15kernel_moe_gemmINS_15GridwiseMoeGemmINS_13tensor_layout4gemm8RowMajorENS3_11Col...` | `aiter::fused_moe_ > aiter::ck_moe_stage1` | c10::BFloat16[16, 6144]; c10::BFloat16[257, 1024, 6144]; c10::BFloat16[257, 6144, 512]; int[8359]; int[262]; int[2]; c10::BFloat16[16, 9, 512] | [8, 144, 1] | 4 | 791.0 | 11.5% |
| `_ZN2ck15kernel_moe_gemmINS_15GridwiseMoeGemmINS_13tensor_layout4gemm8RowMajorENS3_11Col...` | `aiter::fused_moe_ > aiter::ck_moe_stage2` | c10::BFloat16[16, 9, 512]; c10::BFloat16[257, 1024, 6144]; c10::BFloat16[257, 6144, 512]; int[8359]; int[262]; int[2]; c10::BFloat16[16, 6144]; float[8359] | [48, 144, 1] | 4 | 503.2 | 7.3% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `aten::mm` | c10::BFloat16[16, 6144]; c10::BFloat16[6144, 38720] | [38912, 1, 1] | 4 | 497.4 | 7.2% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::reg_all_gather_into_tensor > record_param_comms` | c10::BFloat16[4, 6144] | [48, 1, 1] | 4 | 440.3 | 6.4% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::reg_all_gather_into_tensor > record_param_comms` | c10::BFloat16[16, 38720] | [44, 1, 1] | 4 | 376.2 | 5.4% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous name...` | `aten::cat` | TensorList[[4], [0]] | [1, 2, 1] | 20 | 349.8 | 5.1% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `aiter::gemm_a16w16 > aten::mm` | c10::BFloat16[4, 16384]; c10::BFloat16[16384, 6144] | [61440, 1, 1] | 4 | 287.2 | 4.2% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::reg_reduce_scatter_tensor > record_param_comms` | c10::BFloat16[16, 6144] | [48, 1, 1] | 4 | 277.0 | 4.0% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::inplace_all_reduce > record_param_comms` | TensorList[[16, 6144]] | [48, 1, 1] | 4 | 276.7 | 4.0% |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb0ELb0ELb1ELi1EEEvPT0_PT_PfS4...` | `aiter::rmsnorm2d_fwd > aiter::rmsnorm` | c10::BFloat16[4, 6144]; c10::BFloat16[4, 6144]; c10::BFloat16[6144] | [4, 1, 1] | 12 | 242.8 | 3.5% |
| `hgemm_bf16_16x128x128x3_SPK4_W1x4x1_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[4, 12288]; c10::BFloat16[6144, 12288] | [48, 4, 1] | 4 | 236.2 | 3.4% |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb1ELb0ELb1ELi1EEEvPT0_PT_PfS4...` | `aiter::rmsnorm2d_fwd_with_add > aiter::add_rmsnorm` | c10::BFloat16[4, 6144]; c10::BFloat16[4, 6144]; c10::BFloat16[4, 6144]; c10::BFloat16[4, 6144]; c10::BFloat16[6144] | [4, 1, 1] | 8 | 206.1 | 3.0% |
| `memcpy_triton_kernel` | `-` |  | [76, 1, 1] | 8 | 190.8 | 2.8% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymo...` | `aten::cat` | TensorList[[4, 1], [0, 1]] | [1, 2, 1] | 8 | 144.4 | 2.1% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_ke...` | `aten::copy_` | float[4, 154880]; c10::BFloat16[4, 154880] | [303, 1, 1] | 4 | 136.5 | 2.0% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymo...` | `aten::cat` | TensorList[[4, 6144], [0, 6144]] | [12, 2, 1] | 4 | 136.0 | 2.0% |
| `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16...` | `aten::fill_` | c10::BFloat16[4, 154880] | [303, 1, 1] | 4 | 135.2 | 2.0% |
| `hgemm_bf16_16x64x64x8_SPK6_W1x2x1_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[4, 6144]; c10::BFloat16[2624, 6144] | [41, 6, 1] | 4 | 134.6 | 1.9% |
| `_gemm_a16_w16_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_16_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_NU...` | `aiter::gemm_a16w16 > aiter::gemm_a16w16_` | c10::BFloat16[4, 2048]; c10::BFloat16[16384, 2048] | [1024, 1, 1] | 4 | 129.9 | 1.9% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymo...` | `aten::cat` | TensorList[[4, 154880], [0, 154880]] | [303, 2, 1] | 4 | 109.6 | 1.6% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_...` | `aten::bmm` | c10::BFloat16[64, 4, 192]; c10::BFloat16[64, 192, 512] | [32768, 1, 1] | 4 | 100.0 | 1.4% |
| `_sparse_mla_decode_reduce_kernel` | `-` |  | [4, 64, 8] | 4 | 94.5 | 1.4% |
| `void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::BFloat16, long, unsi...` | `aten::index_select` | c10::BFloat16[154880, 6144]; long int[4] | [12, 1, 1] | 4 | 93.7 | 1.4% |
| `_sparse_mla_decode_split_kernel` | `-` |  | [4, 4, 16] | 4 | 93.1 | 1.3% |
| `void aiter::grouped_topk_kernel<hip_bfloat16, float __vector(4), 1, true, true, false>(...` | `aiter::biased_grouped_topk_hip` | c10::BFloat16[16, 256]; c10::BFloat16[256]; float[16, 8]; int[16, 8] | [16, 1, 1] | 4 | 92.8 | 1.3% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymo...` | `aten::cat` | TensorList[[4, 6144], [4, 6144]] | [12, 2, 1] | 4 | 88.9 | 1.3% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<long>, std::a...` | `aten::fill_` |  | [1, 1, 1] | 4 | 83.0 | 1.2% |
| `hgemm_bf16_16x64x64x7_SPK12_W1x2x1_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[16, 6144]; c10::BFloat16[256, 6144] | [4, 12, 1] | 4 | 82.0 | 1.2% |
| `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_n...` | `aten::copy_` | c10::BFloat16[16, 4, 38720]; c10::BFloat16[16, 4, 38720] | [2420, 1, 1] | 4 | 80.9 | 1.2% |
| `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymou...` | `aten::_foreach_copy_` | TensorList[[4], [4], [4], [4], [4]]; TensorList[[4], [4], [4], [4], [4]] | [5, 1, 1] | 4 | 71.4 | 1.0% |
| `moe_sorting_oneshot_kernel_0` | `aiter::fused_moe_` | c10::BFloat16[16, 6144]; c10::BFloat16[257, 1024, 6144]; c10::BFloat16[257, 6144, 512]; float[16, 9]; int[16, 9] | [193, 1, 1] | 4 | 65.8 | 1.0% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `aten::bmm` | c10::BFloat16[64, 4, 512]; c10::BFloat16[64, 512, 256]; c10::BFloat16[64, 4, 256] | [16384, 1, 1] | 4 | 58.6 | 0.8% |
| `Cijk_SB_BiasS_HAS_ScaleAlphaVec_PostGSU16_VW4` | `aiter::gemm_a16w16 > aten::mm` | c10::BFloat16[4, 16384]; c10::BFloat16[16384, 6144] | [6144, 1, 1] | 4 | 57.6 | 0.8% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::...` | `aten::fill_` | float[2] | [1, 1, 1] | 4 | 51.0 | 0.7% |
| `_ZN5aiter23fused_qk_rmsnorm_kernelIDF16bLi256ELi8ELb1ELi1EEEvPT_S2_PKS1_S4_S4_S4_ffiiiiiii` | `aiter::_fused_qk_rmsnorm_kernel` | c10::BFloat16[4, 2048]; c10::BFloat16[2048]; c10::BFloat16[4, 512]; c10::BFloat16[512]; c10::BFloat16[4, 2048]; c10::BFloat16[4, 512] | [4, 2, 1] | 4 | 50.1 | 0.7% |
| `void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocprim::ROCPRIM_400200_NS::...` | `aten::cumsum` | long int[4] | [1, 1, 1] | 4 | 48.8 | 0.7% |
| `_fused_append_shared_experts_kernel` | `-` |  | [16, 1, 1] | 4 | 40.6 | 0.6% |
| `_fused_qk_rope_cat_and_cache_mla_kernel` | `aiter::fused_qk_rope_cat_and_cache_mla` | c10::BFloat16[4, 64, 512]; c10::BFloat16[4, 64, 64]; c10::BFloat16[4, 1, 512]; c10::BFloat16[4, 1, 64]; c10::Float8_e4m3fn[2334656, 1, 576]; long int[4]; long int[4]; c10::BFloat16[1048576, 1, 1, 32]; c10::BFloat16[1048576, 1, 1, 32] | [256, 1, 1] | 4 | 30.5 | 0.4% |
| `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16...` | `aten::fill_` | c10::BFloat16[16, 6144] | [48, 1, 1] | 4 | 22.8 | 0.3% |

## DRAFT_EXTEND_V2 (draft_extend)

| kernel | launching op | input dims | grid | calls/iter | us/iter | % phase |
|---|---|---|---|---|---|---|
| `_ZN2ck15kernel_moe_gemmINS_15GridwiseMoeGemmINS_13tensor_layout4gemm8RowMajorENS3_11Col...` | `aiter::fused_moe_ > aiter::ck_moe_stage1` | c10::BFloat16[96, 6144]; c10::BFloat16[257, 1024, 6144]; c10::BFloat16[257, 6144, 512]; int[9079]; int[284]; int[2]; c10::BFloat16[96, 9, 512] | [8, 284, 1] | 1 | 400.3 | 12.0% |
| `_ZN2ck15kernel_moe_gemmINS_15GridwiseMoeGemmINS_13tensor_layout4gemm8RowMajorENS3_11Col...` | `aiter::fused_moe_ > aiter::ck_moe_stage2` | c10::BFloat16[96, 9, 512]; c10::BFloat16[257, 1024, 6144]; c10::BFloat16[257, 6144, 512]; int[9079]; int[284]; int[2]; c10::BFloat16[96, 6144]; float[9079] | [48, 284, 1] | 1 | 203.7 | 6.1% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::reg_all_gather_into_tensor > record_param_comms` | c10::BFloat16[96, 38720] | [48, 1, 1] | 1 | 191.8 | 5.7% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x96x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `aten::mm` | c10::BFloat16[96, 6144]; c10::BFloat16[6144, 38720] | [51712, 1, 1] | 1 | 143.4 | 4.3% |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOp...` | `aten::argmax` | float[4, 154880] | [4, 1, 1] | 2 | 143.3 | 4.3% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous name...` | `aten::cat` | TensorList[[4], [0]] | [1, 2, 1] | 6 | 106.7 | 3.2% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::reg_reduce_scatter_tensor > record_param_comms` | c10::BFloat16[96, 6144] | [48, 1, 1] | 1 | 100.7 | 3.0% |
| `void at::native::(anonymous namespace)::cunn_SoftMaxForwardGmem<4, float, float, float,...` | `aten::_softmax` | float[4, 154880] | [4, 1, 1] | 1 | 84.7 | 2.5% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x32x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `aiter::gemm_a16w16 > aten::mm` | c10::BFloat16[24, 16384]; c10::BFloat16[16384, 6144] | [61440, 1, 1] | 1 | 79.4 | 2.4% |
| `memcpy_triton_kernel` | `-` |  | [454, 1, 1] | 2 | 68.1 | 2.0% |
| `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` | `-` |  | [480, 1, 1] | 1 | 61.6 | 1.8% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::inplace_all_reduce > record_param_comms` | TensorList[[96, 6144]] | [48, 1, 1] | 1 | 60.6 | 1.8% |
| `_sparse_mla_split_k_kernel` | `-` |  | [24, 4, 2] | 1 | 58.9 | 1.8% |
| `hgemm_bf16_32x128x128x3_SPK4_W2x2x2_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[24, 12288]; c10::BFloat16[6144, 12288] | [48, 4, 1] | 1 | 58.3 | 1.7% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous name...` | `aten::cat` | TensorList[[24], [0]] | [1, 2, 1] | 3 | 58.0 | 1.7% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::reg_all_gather_into_tensor > record_param_comms` | c10::BFloat16[24, 6144] | [36, 1, 1] | 1 | 57.1 | 1.7% |
| `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_n...` | `aten::copy_` | c10::BFloat16[96, 4, 38720]; c10::BFloat16[96, 4, 38720] | [14520, 1, 1] | 1 | 54.8 | 1.6% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::ar...` | `aten::fill_` | int[5] | [1, 1, 1] | 2 | 53.7 | 1.6% |
| `hgemm_bf16_16x64x128x5_SPK8_W1x2x1_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[96, 6144]; c10::BFloat16[256, 6144] | [24, 8, 1] | 1 | 46.7 | 1.4% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::compare_scalar_kernel<int...` | `aten::le` | int[4] | [1, 1, 1] | 1 | 46.2 | 1.4% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<long>, std::a...` | `aten::fill_` |  | [1, 1, 1] | 1 | 46.0 | 1.4% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x32x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `aten::bmm` | c10::BFloat16[64, 24, 512]; c10::BFloat16[64, 512, 256]; c10::BFloat16[64, 24, 256] | [16384, 1, 1] | 1 | 41.5 | 1.2% |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb1ELb0ELb1ELi1EEEvPT0_PT_PfS4...` | `aiter::rmsnorm2d_fwd_with_add > aiter::add_rmsnorm` | c10::BFloat16[24, 6144]; c10::BFloat16[24, 6144]; c10::BFloat16[24, 6144]; c10::BFloat16[24, 6144]; c10::BFloat16[6144] | [24, 1, 1] | 2 | 37.9 | 1.1% |
| `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16...` | `aten::fill_` | c10::BFloat16[24, 154880] | [1815, 1, 1] | 1 | 35.0 | 1.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_ke...` | `aten::copy_` | float[24, 154880]; c10::BFloat16[24, 154880] | [1815, 1, 1] | 1 | 33.8 | 1.0% |
| `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, lon...` | `aten::index` | c10::BFloat16[24, 6144] | [4, 3, 1] | 1 | 32.2 | 1.0% |
| `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, lon...` | `aten::index` | int[24, 2048] | [4, 2, 1] | 1 | 31.9 | 1.0% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | int[4]; long int[4] | [1, 1, 1] | 2 | 31.7 | 0.9% |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb0ELb0ELb1ELi1EEEvPT0_PT_PfS4...` | `aiter::rmsnorm2d_fwd > aiter::rmsnorm` | c10::BFloat16[24, 6144]; c10::BFloat16[24, 6144]; c10::BFloat16[6144] | [24, 1, 1] | 3 | 31.7 | 0.9% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x32x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_...` | `aten::bmm` | c10::BFloat16[64, 24, 192]; c10::BFloat16[64, 192, 512] | [65536, 1, 1] | 1 | 31.6 | 0.9% |
| `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_...` | `aten::arange` | int[0] | [14, 1, 1] | 2 | 31.3 | 0.9% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::la...` | `aten::clamp_min_` | float[4, 154880] | [605, 1, 1] | 1 | 31.3 | 0.9% |
| `Cijk_SB_BiasS_HAS_ScaleAlphaVec_PostGSU16_VW4` | `aiter::gemm_a16w16 > aten::mm` | c10::BFloat16[24, 16384]; c10::BFloat16[16384, 6144] | [36864, 1, 1] | 1 | 31.3 | 0.9% |
| `void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocprim::ROCPRIM_400200_NS::...` | `aten::cumsum` | int[4] | [1, 1, 1] | 2 | 30.3 | 0.9% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<lon...` | `aten::add` | long int[4] | [1, 1, 1] | 2 | 29.5 | 0.9% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::wh...` | `aten::where` | bool[4, 1]; long int[4, 1]; long int[4, 1] | [1, 1, 1] | 1 | 29.4 | 0.9% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::add` | long int[4]; int[4] | [1, 1, 1] | 1 | 28.3 | 0.8% |
| `void sgl::dsa_topk::coop_mb_hist1<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParams...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 110080]; int[24]; int[24, 2048]; int[24, 110037]; int[25] | [24, 10, 1] | 0.5 | 27.7 | 0.8% |
| `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymou...` | `aten::_foreach_copy_` | TensorList[[24], [24], [24], [4], [4]]; TensorList[[24], [24], [24], [4], [4]] | [5, 1, 1] | 1 | 26.9 | 0.8% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::ar...` | `aten::fill_` | int[25] | [1, 1, 1] | 1 | 26.8 | 0.8% |

## draft glue (draft, outside steps)

| kernel | launching op | input dims | grid | calls/iter | us/iter | % phase |
|---|---|---|---|---|---|---|
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOp...` | `aten::argmax` | float[4, 154880] | [4, 1, 1] | 8 | 480.7 | 16.9% |
| `void at::native::(anonymous namespace)::cunn_SoftMaxForwardGmem<4, float, float, float,...` | `aten::_softmax` | float[4, 154880] | [4, 1, 1] | 4 | 237.0 | 8.3% |
| `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_g...` | `aten::gather` | float[4, 154880]; long int[4, 1] | [1, 1, 1] | 8 | 188.1 | 6.6% |
| `void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_kernel<flo...` | `aten::exponential_` | float[4, 154880] | [2048, 1, 1] | 4 | 151.5 | 5.3% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_n...` | `aten::div` | float[4, 154880]; float[4, 1] | [1210, 1, 1] | 4 | 144.4 | 5.1% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, floa...` | `aten::div` | float[4, 154880]; float[4, 154880] | [605, 1, 1] | 4 | 143.7 | 5.1% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<lon...` | `aten::add_` | long int[4] | [1, 1, 1] | 4 | 131.6 | 4.6% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::wh...` | `aten::where` | bool[4, 1]; long int[4, 1]; long int[4, 1] | [1, 1, 1] | 4 | 118.4 | 4.2% |
| `triton_poi_fused_gather_max_view_1` | `triton_poi_fused_gather_max_view_1` | long int[4, 1]; long int[4, 1] | [1, 1, 1] | 4 | 115.2 | 4.1% |
| `triton_poi_fused_add_max_view_3` | `triton_poi_fused_add_max_view_3` | long int[4, 1] | [1, 1, 1] | 4 | 115.0 | 4.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, i...` | `aten::floor_divide` | int[4, 1720] | [7, 1, 1] | 4 | 86.3 | 3.0% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | long int[1720]; int[1720] | [4, 1, 1] | 4 | 84.1 | 3.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::la...` | `aten::clamp_min_` | float[4, 154880] | [605, 1, 1] | 4 | 75.0 | 2.6% |
| `void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocprim::ROCPRIM_400200_NS::...` | `aten::cumsum` | int[4] | [1, 1, 1] | 8 | 72.6 | 2.6% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::ar...` | `aten::fill_` | int[5] | [1, 1, 1] | 8 | 68.0 | 2.4% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | int[4]; long int[4] | [1, 1, 1] | 4 | 59.1 | 2.1% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::index` | int[17, 110031] | [860, 1, 1] | 2 | 57.7 | 2.0% |
| `triton_poi_fused_max_mul_unsqueeze_view_0` | `triton_poi_fused_max_mul_unsqueeze_view_0` | float[4, 1]; float[4, 1]; float[4, 1, 1]; float[4, 1] | [1, 1, 1] | 4 | 56.3 | 2.0% |
| `triton_poi_fused_add_arange_floor_divide_index_max_unsqueeze_view_2` | `triton_poi_fused_add_arange_floor_divide_index_max_unsqueeze_view_2` | c10::BFloat16[4, 6144]; c10::BFloat16[4, 6144] | [12, 1, 1] | 4 | 55.9 | 2.0% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_n...` | `aten::cat` | TensorList[[4, 1], [4, 5]] | [1, 1, 1] | 1 | 48.7 | 1.7% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::compare_scalar_kernel<int...` | `aten::le` | int[4] | [1, 1, 1] | 4 | 48.7 | 1.7% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::index` | int[4, 110031] | [14, 1, 1] | 2 | 46.1 | 1.6% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymo...` | `aten::cat` | TensorList[[4, 1], [4, 1], [4, 1], [4, 1], [4, 1]] | [1, 5, 1] | 1 | 42.6 | 1.5% |
| `void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::...` | `aten::fill_` | bool[2640984] | [645, 1, 1] | 0.5 | 27.2 | 1.0% |
| `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_...` | `aten::arange` | long int[0] | [1, 1, 1] | 1 | 26.7 | 0.9% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::la...` | `aten::clamp` | int[4] | [1, 1, 1] | 4 | 26.3 | 0.9% |
| `build_tree_efficient(long*, long*, long*, bool*, long*, long*, long*, long*, int, int, ...` | `sgl_kernel::build_tree_kernel_efficient` | long int[4, 5]; long int[4, 5]; long int[4]; bool[2640984]; long int[24]; long int[4, 6]; long int[4, 6]; long int[4, 6] | [4, 1, 1] | 0.5 | 25.6 | 0.9% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<long>, std::a...` | `aten::fill_` | long int[3, 4, 6] | [1, 1, 1] | 1 | 21.6 | 0.8% |
| `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_...` | `aten::arange` | int[0] | [14, 1, 1] | 4 | 15.0 | 0.5% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::index` | int[4, 110035] | [14, 1, 1] | 2 | 13.9 | 0.5% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymo...` | `aten::cat` | TensorList[[4, 154880], [4, 154880], [4, 154880], [4, 154880], [4, 154880]] | [303, 5, 1] | 1 | 13.0 | 0.5% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::index` | int[17, 110035] | [860, 1, 1] | 2 | 12.2 | 0.4% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<lon...` | `aten::add` | long int[4] | [1, 1, 1] | 4 | 11.7 | 0.4% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::cat` | TensorList[[4, 1], [4, 5]] | [1, 1, 1] | 1 | 8.9 | 0.3% |
| `build_tree_efficient(long*, long*, long*, bool*, long*, long*, long*, long*, int, int, ...` | `sgl_kernel::build_tree_kernel_efficient` | long int[4, 5]; long int[4, 5]; long int[4]; bool[2640888]; long int[24]; long int[4, 6]; long int[4, 6]; long int[4, 6] | [4, 1, 1] | 0.5 | 6.5 | 0.2% |
| `void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::...` | `aten::fill_` | bool[2640888] | [645, 1, 1] | 0.5 | 5.8 | 0.2% |
| `assign_draft_cache_locs_contiguous` | `-` |  | [4, 1, 1] | 1 | 2.4 | 0.1% |

## <none>

| kernel | launching op | input dims | grid | calls/iter | us/iter | % phase |
|---|---|---|---|---|---|---|
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `record_param_comms` | TensorList[[7]] | [1, 1, 1] | 1 | 160.7 | 46.2% |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOp...` | `aten::argmax` | float[24, 154880] | [24, 1, 1] | 1 | 56.4 | 16.2% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::add` | long int[4]; int[4] | [1, 1, 1] | 1 | 27.1 | 7.8% |
| `assign_extend_cache_locs_uniform` | `-` |  | [4, 1, 1] | 1 | 27.1 | 7.8% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::_index_put_impl_` | int[24]; int[4] | [1, 1, 1] | 1 | 19.4 | 5.6% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | int[4]; long int[4] | [1, 1, 1] | 2 | 6.6 | 1.9% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | long int[4, 3]; int[4, 3] | [1, 1, 1] | 1 | 5.6 | 1.6% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | long int[4, 6]; int[4, 6] | [1, 1, 1] | 1 | 5.4 | 1.5% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::add` | int[4, 1]; long int[4] | [1, 1, 1] | 1 | 4.9 | 1.4% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | int[4, 4]; long int[4, 4] | [1, 1, 1] | 1 | 4.4 | 1.3% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::ar...` | `aten::fill_` | int[4, 6] | [1, 1, 1] | 2 | 4.2 | 1.2% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | int[4, 3]; long int[4, 3] | [1, 1, 1] | 1 | 3.5 | 1.0% |
| `void VerifyTreeGreedy<int, long>(int*, int*, int*, long*, long*, long*, long*, long*, u...` | `sgl_kernel::verify_tree_greedy` | int[24]; int[4, 6]; int[4]; long int[4, 6]; long int[4, 6]; long int[4, 6]; long int[4, 6]; long int[4, 6] | [4, 1, 1] | 1 | 3.4 | 1.0% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::_index_put_impl_` | int[24]; int[4, 3] | [1, 1, 1] | 1 | 3.2 | 0.9% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::index` | int[24] | [1, 1, 1] | 1 | 3.2 | 0.9% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int...` | `aten::add` | int[4] | [1, 1, 1] | 1 | 3.0 | 0.9% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::ar...` | `aten::fill_` | int[24] | [1, 1, 1] | 1 | 2.8 | 0.8% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | long int[4]; int[4] | [1, 1, 1] | 1 | 2.6 | 0.8% |
| `fill_bonus_tokens` | `-` |  | [4, 1, 1] | 1 | 1.9 | 0.5% |
| `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_...` | `aten::arange` | long int[0] | [1, 1, 1] | 1 | 1.7 | 0.5% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::ar...` | `aten::fill_` | int[4] | [1, 1, 1] | 1 | 1.0 | 0.3% |
