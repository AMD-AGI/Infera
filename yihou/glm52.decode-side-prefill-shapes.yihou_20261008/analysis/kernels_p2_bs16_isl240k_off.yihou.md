# Kernel → op → input shapes — results/p2_bs16_isl240k_off/profile/decode-1791461638.465167-TP-0-DP-0.trace.json.gz

Per-iteration device time over 2 profiled iteration(s). Phase from host annotations.

| phase | kernel us / iter | share of total |
|---|---|---|
| TARGET_VERIFY (step[VERIFY]) | 137143.7 | 84.0% |
| draft DECODE x4 (draft > step[DRAFT]) | 14944.8 | 9.1% |
| DRAFT_EXTEND_V2 (draft_extend) | 9241.0 | 5.7% |
| draft glue (draft, outside steps) | 1160.7 | 0.7% |
| <none> | 855.9 | 0.5% |

## TARGET_VERIFY (step[VERIFY])

| kernel | launching op | input dims | grid | calls/iter | us/iter | % phase |
|---|---|---|---|---|---|---|
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::reg_all_gather_into_tensor > record_param_comms` | c10::BFloat16[24, 6144] | [36, 1, 1] | 78 | 62159.6 | 45.3% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::reg_reduce_scatter_tensor > record_param_comms` | c10::BFloat16[96, 6144] | [48, 1, 1] | 78 | 33142.4 | 24.2% |
| `gemm1_a4w4_port_h6144_i512_ne257_bm16_iq_sep` | `aiter::fused_moe_` | c10::BFloat16[96, 6144]; c10::Float4_e2m1fn_x2[257, 1024, 3072]; c10::Float4_e2m1fn_x2[257, 6144, 256]; float[96, 9]; int[96, 9]; unsigned char[257, 1024, 192]; unsigned char[257, 6144, 16] | [1180, 1, 1] | 75 | 11176.2 | 8.1% |
| `_sparse_mla_split_k_kernel` | `-` |  | [24, 4, 2] | 78 | 7142.8 | 5.2% |
| `gemm2_a4w4_port_hmax8192_imax512_bm16_bn128_bk256_nt_atomic_persist_cu256_bhoist_apf_sp...` | `aiter::fused_moe_` | c10::BFloat16[96, 6144]; c10::Float4_e2m1fn_x2[257, 1024, 3072]; c10::Float4_e2m1fn_x2[257, 6144, 256]; float[96, 9]; int[96, 9]; unsigned char[257, 1024, 192]; unsigned char[257, 6144, 16] | [12288, 1, 1] | 75 | 3476.0 | 2.5% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x32x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `aiter::gemm_a16w16 > aten::mm` | c10::BFloat16[24, 16384]; c10::BFloat16[16384, 6144] | [61440, 1, 1] | 78 | 3406.8 | 2.5% |
| `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` | `-` |  | [480, 1, 1] | 21 | 2284.9 | 1.7% |
| `Cijk_SB_BiasS_HAS_ScaleAlphaVec_PostGSU16_VW4` | `aiter::gemm_a16w16 > aten::mm` | c10::BFloat16[24, 16384]; c10::BFloat16[16384, 6144] | [36864, 1, 1] | 78 | 1688.9 | 1.2% |
| `_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel_HAS_BIA...` | `-` |  | [64, 2, 1] | 156 | 1425.1 | 1.0% |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb1ELb0ELb1ELi1EEEvPT0_PT_PfS4...` | `aiter::rmsnorm2d_fwd_with_add > aiter::add_rmsnorm` | c10::BFloat16[24, 6144]; c10::BFloat16[24, 6144]; c10::BFloat16[24, 6144]; c10::BFloat16[24, 6144]; c10::BFloat16[6144] | [24, 1, 1] | 156 | 1414.6 | 1.0% |
| `_gemm_a16_w16_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_NU...` | `aiter::gemm_a16w16 > aiter::gemm_a16w16_` | c10::BFloat16[24, 2048]; c10::BFloat16[16384, 2048] | [512, 1, 1] | 78 | 1390.6 | 1.0% |
| `_fused_append_shared_experts_kernel` | `-` |  | [96, 1, 1] | 75 | 947.1 | 0.7% |
| `hgemm_bf16_32x64x128x4_SPK4_W1x4x1_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[24, 6144]; c10::BFloat16[2624, 6144] | [41, 4, 1] | 78 | 885.9 | 0.6% |
| `void aiter::kn_entry_2c_sbhd_cached_indirect_inplace<aiter::OpCachedFwd, 1, true, false...` | `aiter::rope_cached_positions_2c_fwd_impl` | c10::BFloat16[1, 24, 32, 64]; c10::BFloat16[1, 24, 1, 64]; c10::BFloat16[1, 24, 32, 64]; c10::BFloat16[1, 24, 1, 64]; c10::BFloat16[1048576, 1, 1, 32]; c10::BFloat16[1048576, 1, 1, 32]; long int[1, 24] | [3, 1, 1] | 21 | 773.1 | 0.6% |
| `_sparse_mla_reduce_kernel` | `-` |  | [24, 64, 8] | 78 | 580.9 | 0.4% |
| `hgemm_bf16_16x64x128x5_SPK8_W1x2x1_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[96, 6144]; c10::BFloat16[256, 6144] | [24, 8, 1] | 75 | 507.8 | 0.4% |
| `void aiter::mxfp4_moe::moe_sort_quant::sort_quant_kernel_impl<257, 9, 16, 6144, 128, 10...` | `aiter::fused_moe_ > aiter::mxfp4_moe_sort` | int[96, 9]; float[96, 9]; int[4720]; int[295]; int[2]; int[864]; float[4720]; int[4720]; c10::BFloat16[96, 6144]; c10::BFloat16[0]; int[0] | [128, 1, 1] | 75 | 447.9 | 0.3% |
| `void aiter::grouped_topk_kernel<hip_bfloat16, float __vector(4), 1, true, true, false>(...` | `aiter::biased_grouped_topk_hip` | c10::BFloat16[96, 256]; c10::BFloat16[256]; float[96, 8]; int[96, 8] | [96, 1, 1] | 75 | 398.5 | 0.3% |
| `_ZN5aiter23fused_qk_rmsnorm_kernelIDF16bLi256ELi8ELb1ELi1EEEvPT_S2_PKS1_S4_S4_S4_ffiiiiiii` | `aiter::_fused_qk_rmsnorm_kernel` | c10::BFloat16[24, 2048]; c10::BFloat16[2048]; c10::BFloat16[24, 512]; c10::BFloat16[512]; c10::BFloat16[24, 2048]; c10::BFloat16[24, 512] | [24, 2, 1] | 78 | 373.2 | 0.3% |
| `_fused_qk_rope_cat_and_cache_mla_kernel` | `aiter::fused_qk_rope_cat_and_cache_mla` | c10::BFloat16[24, 64, 512]; c10::BFloat16[24, 64, 64]; c10::BFloat16[24, 1, 512]; c10::BFloat16[24, 1, 64]; c10::Float8_e4m3fn[2329664, 1, 576]; long int[24]; long int[24]; c10::BFloat16[1048576, 1, 1, 32]; c10::BFloat16[1048576, 1, 1, 32] | [1536, 1, 1] | 78 | 364.7 | 0.3% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::reg_all_gather_into_tensor > record_param_comms` | c10::BFloat16[96, 38720] | [48, 1, 1] | 1 | 349.8 | 0.3% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::inplace_all_reduce > record_param_comms` | TensorList[[96, 6144]] | [48, 1, 1] | 1 | 271.1 | 0.2% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `aiter::gemm_a16w16 > aten::mm` | c10::BFloat16[24, 6144]; c10::BFloat16[6144, 32] | [1024, 1, 1] | 21 | 250.1 | 0.2% |
| `void sgl::dsa_topk::coop_mb_scatter<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbPara...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 240064]; int[24]; int[24, 2048]; int[24, 240041]; int[25] | [24, 10, 1] | 10.5 | 187.0 | 0.1% |
| `void sgl::dsa_topk::coop_mb_scatter<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbPara...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 240064]; int[24]; int[24, 2048]; int[24, 240037]; int[25] | [24, 10, 1] | 10.5 | 184.6 | 0.1% |
| `hgemm_bf16_16x64x128x4_SPK2_W1x2x1_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[24, 2048]; c10::BFloat16[4096, 2048] | [128, 2, 1] | 21 | 158.4 | 0.1% |
| `void sgl::dsa_topk::coop_mb_refine<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParam...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 240064]; int[24]; int[24, 2048]; int[24, 240041]; int[25] | [24, 1, 1] | 10.5 | 138.7 | 0.1% |
| `void sgl::dsa_topk::coop_mb_refine<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParam...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 240064]; int[24]; int[24, 2048]; int[24, 240037]; int[25] | [24, 1, 1] | 10.5 | 137.6 | 0.1% |
| `hgemm_bf16_16x64x128x4_SPK12_W1x2x1_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[24, 6144]; c10::BFloat16[128, 6144] | [4, 12, 1] | 21 | 131.8 | 0.1% |
| `act_quant_kernel__kernel` | `-` |  | [24, 1, 1] | 21 | 121.3 | 0.1% |
| `_ZN7ck_tile6kentryILi1ENS_14Layernorm2dFwdINS_29Layernorm2dFwdPipelineOnePassINS_29Laye...` | `aiter::layernorm2d_fwd` | c10::BFloat16[24, 128]; c10::BFloat16[128]; c10::BFloat16[128] | [6, 1, 1] | 21 | 113.0 | 0.1% |
| `_ZN5aiter32indexer_k_quant_and_cache_kernelIDF16bDB8_LN4vllm18Fp8KVCacheDataTypeE1ELi8E...` | `aiter::indexer_k_quant_and_cache` | c10::BFloat16[24, 128]; c10::Float8_e4m3fn[36401, 64, 132]; long int[24] | [3, 1, 1] | 21 | 112.7 | 0.1% |
| `triton_poi_fused_mul_unsqueeze_0` | `triton_poi_fused_mul_unsqueeze_0` | c10::BFloat16[24, 32]; float[24, 32, 1]; float[24, 32, 1] | [6, 1, 1] | 21 | 98.6 | 0.1% |
| `void fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<16, 7, c10::B...` | `HadamardTransformFn` | c10::BFloat16[24, 128] | [24, 1, 1] | 21 | 92.6 | 0.1% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x96x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `aten::mm` | c10::BFloat16[96, 6144]; c10::BFloat16[6144, 38720] | [51712, 1, 1] | 1 | 92.1 | 0.1% |
| `void sgl::dsa_topk::coop_mb_hist0<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParams...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 240064]; int[24]; int[24, 2048]; int[24, 240041]; int[25] | [24, 10, 1] | 10.5 | 91.5 | 0.1% |
| `void sgl::dsa_topk::coop_mb_hist0<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParams...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 240064]; int[24]; int[24, 2048]; int[24, 240037]; int[25] | [24, 10, 1] | 10.5 | 90.3 | 0.1% |
| `void fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<16, 7, c10::B...` | `HadamardTransformFn` | c10::BFloat16[24, 32, 128] | [768, 1, 1] | 21 | 86.4 | 0.1% |
| `hgemm_bf16_64x64x128x3_SPK1_W2x2x2_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[96, 6144]; c10::BFloat16[6144, 6144] | [192, 1, 1] | 3 | 85.6 | 0.1% |
| `void sgl::dsa_topk::coop_mb_hist1<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParams...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 240064]; int[24]; int[24, 2048]; int[24, 240041]; int[25] | [24, 10, 1] | 10.5 | 56.3 | 0.0% |
| `hgemm_bf16_64x64x128x3_SPK1_W4x1x1_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[96, 3072]; c10::BFloat16[6144, 3072] | [192, 1, 1] | 3 | 53.1 | 0.0% |
| `void sgl::dsa_topk::coop_mb_hist1<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParams...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 240064]; int[24]; int[24, 2048]; int[24, 240037]; int[25] | [24, 10, 1] | 10.5 | 52.7 | 0.0% |
| `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_n...` | `aten::copy_` | c10::BFloat16[96, 4, 38720]; c10::BFloat16[96, 4, 38720] | [14520, 1, 1] | 1 | 24.3 | 0.0% |
| `_ZN7sgl_hip10activation18act_and_mul_kernelI14__hip_bfloat16TnPFT_RKS3_EXadL_Z4siluIS2_...` | `sgl_kernel::silu_and_mul` | c10::BFloat16[96, 3072]; c10::BFloat16[96, 6144] | [96, 1, 1] | 3 | 13.8 | 0.0% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous name...` | `aten::cat` | TensorList[[24], [0]] | [1, 2, 1] | 3 | 12.6 | 0.0% |
| `memcpy_triton_kernel` | `-` |  | [454, 1, 1] | 2 | 10.8 | 0.0% |
| `void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocprim::ROCPRIM_400200_NS::...` | `aten::cumsum` | long int[4] | [1, 1, 1] | 1 | 9.5 | 0.0% |
| `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_...` | `aten::arange` | int[0] | [30, 1, 1] | 2 | 8.5 | 0.0% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous name...` | `aten::cat` | TensorList[[4], [0]] | [1, 2, 1] | 2 | 8.4 | 0.0% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_n...` | `aten::copy_` | int[4, 6, 240041]; int[4, 6, 240041] | [11252, 1, 1] | 0.5 | 7.1 | 0.0% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | long int[3751]; int[3751] | [8, 1, 1] | 1 | 7.1 | 0.0% |
| `void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocprim::ROCPRIM_400200_NS::...` | `aten::cumsum` | int[4] | [1, 1, 1] | 2 | 6.8 | 0.0% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_n...` | `aten::copy_` | int[4, 6, 240037]; int[4, 6, 240037] | [11252, 1, 1] | 0.5 | 6.6 | 0.0% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | int[4]; long int[4] | [1, 1, 1] | 2 | 6.6 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::ar...` | `aten::fill_` | int[5] | [1, 1, 1] | 2 | 6.6 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_ke...` | `aten::copy_` | float[24, 154880]; c10::BFloat16[24, 154880] | [1815, 1, 1] | 1 | 6.2 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16...` | `aten::fill_` | c10::BFloat16[24, 154880] | [1815, 1, 1] | 1 | 5.7 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16...` | `aten::fill_` | c10::BFloat16[96, 6144] | [288, 1, 1] | 1 | 5.1 | 0.0% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::index` | int[17, 240041] | [1876, 1, 1] | 0.5 | 4.9 | 0.0% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::index` | int[17, 240037] | [1876, 1, 1] | 0.5 | 4.7 | 0.0% |
| `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, lon...` | `aten::gather` | c10::BFloat16[154880, 6144]; long int[24, 6144]; c10::BFloat16[24, 6144] | [24, 3, 1] | 1 | 4.4 | 0.0% |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb0ELb0ELb1ELi1EEEvPT0_PT_PfS4...` | `aiter::rmsnorm2d_fwd > aiter::rmsnorm` | c10::BFloat16[24, 6144]; c10::BFloat16[24, 6144]; c10::BFloat16[6144] | [24, 1, 1] | 1 | 4.4 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<lon...` | `aten::add` | long int[4] | [1, 1, 1] | 1 | 4.3 | 0.0% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | long int[4]; int[4] | [1, 1, 1] | 1 | 4.2 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<long>, std::a...` | `aten::fill_` |  | [1, 1, 1] | 1 | 4.1 | 0.0% |
| `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymou...` | `aten::_foreach_copy_` | TensorList[[24], [24], [24], [4], [4]]; TensorList[[24], [24], [24], [4], [4]] | [5, 1, 1] | 1 | 4.1 | 0.0% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::index` | int[24, 240037] | [176, 1, 1] | 0.5 | 3.2 | 0.0% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::index` | int[24, 240041] | [176, 1, 1] | 0.5 | 3.2 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, i...` | `aten::floor_divide` | int[24, 3751] | [88, 1, 1] | 1 | 3.2 | 0.0% |
| `void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocprim::ROCPRIM_400200_NS::...` | `aten::cumsum` | int[24] | [1, 1, 1] | 1 | 3.0 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::ar...` | `aten::fill_` | int[25] | [1, 1, 1] | 1 | 2.1 | 0.0% |
| `seqlens_expand_kernel` | `-` |  | [4, 1, 1] | 1 | 2.0 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::la...` | `aten::clamp` | int[24] | [1, 1, 1] | 1 | 2.0 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::...` | `aten::fill_` | float[156] | [1, 1, 1] | 1 | 1.9 | 0.0% |

## draft DECODE x4 (draft > step[DRAFT])

| kernel | launching op | input dims | grid | calls/iter | us/iter | % phase |
|---|---|---|---|---|---|---|
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::reg_all_gather_into_tensor > record_param_comms` | c10::BFloat16[4, 6144] | [48, 1, 1] | 4 | 5533.9 | 37.0% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::reg_reduce_scatter_tensor > record_param_comms` | c10::BFloat16[16, 6144] | [48, 1, 1] | 4 | 3321.0 | 22.2% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::inplace_all_reduce > record_param_comms` | TensorList[[16, 6144]] | [48, 1, 1] | 4 | 2118.2 | 14.2% |
| `_ZN2ck15kernel_moe_gemmINS_15GridwiseMoeGemmINS_13tensor_layout4gemm8RowMajorENS3_11Col...` | `aiter::fused_moe_ > aiter::ck_moe_stage1` | c10::BFloat16[16, 6144]; c10::BFloat16[257, 1024, 6144]; c10::BFloat16[257, 6144, 512]; int[8359]; int[262]; int[2]; c10::BFloat16[16, 9, 512] | [8, 144, 1] | 4 | 774.0 | 5.2% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::reg_all_gather_into_tensor > record_param_comms` | c10::BFloat16[16, 38720] | [44, 1, 1] | 4 | 726.3 | 4.9% |
| `hgemm_bf16_16x128x128x3_SPK4_W1x4x1_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[4, 12288]; c10::BFloat16[6144, 12288] | [48, 4, 1] | 4 | 690.5 | 4.6% |
| `_ZN2ck15kernel_moe_gemmINS_15GridwiseMoeGemmINS_13tensor_layout4gemm8RowMajorENS3_11Col...` | `aiter::fused_moe_ > aiter::ck_moe_stage2` | c10::BFloat16[16, 9, 512]; c10::BFloat16[257, 1024, 6144]; c10::BFloat16[257, 6144, 512]; int[8359]; int[262]; int[2]; c10::BFloat16[16, 6144]; float[8359] | [48, 144, 1] | 4 | 374.4 | 2.5% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `aten::mm` | c10::BFloat16[16, 6144]; c10::BFloat16[6144, 38720] | [38912, 1, 1] | 4 | 350.2 | 2.3% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `aiter::gemm_a16w16 > aten::mm` | c10::BFloat16[4, 16384]; c10::BFloat16[16384, 6144] | [61440, 1, 1] | 4 | 163.7 | 1.1% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous name...` | `aten::cat` | TensorList[[4], [0]] | [1, 2, 1] | 20 | 76.0 | 0.5% |
| `_gemm_a16_w16_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_16_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_NU...` | `aiter::gemm_a16w16 > aiter::gemm_a16w16_` | c10::BFloat16[4, 2048]; c10::BFloat16[16384, 2048] | [1024, 1, 1] | 4 | 71.5 | 0.5% |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb0ELb0ELb1ELi1EEEvPT0_PT_PfS4...` | `aiter::rmsnorm2d_fwd > aiter::rmsnorm` | c10::BFloat16[4, 6144]; c10::BFloat16[4, 6144]; c10::BFloat16[6144] | [4, 1, 1] | 12 | 50.3 | 0.3% |
| `_sparse_mla_decode_split_kernel` | `-` |  | [4, 4, 16] | 4 | 47.4 | 0.3% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `aten::bmm` | c10::BFloat16[64, 4, 512]; c10::BFloat16[64, 512, 256]; c10::BFloat16[64, 4, 256] | [16384, 1, 1] | 4 | 43.3 | 0.3% |
| `memcpy_triton_kernel` | `-` |  | [76, 1, 1] | 8 | 41.9 | 0.3% |
| `hgemm_bf16_16x64x64x8_SPK6_W1x2x1_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[4, 6144]; c10::BFloat16[2624, 6144] | [41, 6, 1] | 4 | 41.7 | 0.3% |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb1ELb0ELb1ELi1EEEvPT0_PT_PfS4...` | `aiter::rmsnorm2d_fwd_with_add > aiter::add_rmsnorm` | c10::BFloat16[4, 6144]; c10::BFloat16[4, 6144]; c10::BFloat16[4, 6144]; c10::BFloat16[4, 6144]; c10::BFloat16[6144] | [4, 1, 1] | 8 | 40.3 | 0.3% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_...` | `aten::bmm` | c10::BFloat16[64, 4, 192]; c10::BFloat16[64, 192, 512] | [32768, 1, 1] | 4 | 32.6 | 0.2% |
| `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_n...` | `aten::copy_` | c10::BFloat16[16, 4, 38720]; c10::BFloat16[16, 4, 38720] | [2420, 1, 1] | 4 | 31.2 | 0.2% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymo...` | `aten::cat` | TensorList[[4, 1], [0, 1]] | [1, 2, 1] | 8 | 29.1 | 0.2% |
| `Cijk_SB_BiasS_HAS_ScaleAlphaVec_PostGSU16_VW4` | `aiter::gemm_a16w16 > aten::mm` | c10::BFloat16[4, 16384]; c10::BFloat16[16384, 6144] | [6144, 1, 1] | 4 | 29.0 | 0.2% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymo...` | `aten::cat` | TensorList[[4, 154880], [0, 154880]] | [303, 2, 1] | 4 | 28.8 | 0.2% |
| `moe_sorting_oneshot_kernel_0` | `aiter::fused_moe_` | c10::BFloat16[16, 6144]; c10::BFloat16[257, 1024, 6144]; c10::BFloat16[257, 6144, 512]; float[16, 9]; int[16, 9] | [193, 1, 1] | 4 | 27.6 | 0.2% |
| `void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::BFloat16, long, unsi...` | `aten::index_select` | c10::BFloat16[154880, 6144]; long int[4] | [12, 1, 1] | 4 | 26.5 | 0.2% |
| `hgemm_bf16_16x64x64x7_SPK12_W1x2x1_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[16, 6144]; c10::BFloat16[256, 6144] | [4, 12, 1] | 4 | 21.7 | 0.1% |
| `void aiter::grouped_topk_kernel<hip_bfloat16, float __vector(4), 1, true, true, false>(...` | `aiter::biased_grouped_topk_hip` | c10::BFloat16[16, 256]; c10::BFloat16[256]; float[16, 8]; int[16, 8] | [16, 1, 1] | 4 | 21.7 | 0.1% |
| `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16...` | `aten::fill_` | c10::BFloat16[16, 6144] | [48, 1, 1] | 4 | 20.5 | 0.1% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymo...` | `aten::cat` | TensorList[[4, 6144], [0, 6144]] | [12, 2, 1] | 4 | 19.8 | 0.1% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymo...` | `aten::cat` | TensorList[[4, 6144], [4, 6144]] | [12, 2, 1] | 4 | 19.7 | 0.1% |
| `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16...` | `aten::fill_` | c10::BFloat16[4, 154880] | [303, 1, 1] | 4 | 19.1 | 0.1% |
| `_sparse_mla_decode_reduce_kernel` | `-` |  | [4, 64, 8] | 4 | 18.9 | 0.1% |
| `_fused_qk_rope_cat_and_cache_mla_kernel` | `aiter::fused_qk_rope_cat_and_cache_mla` | c10::BFloat16[4, 64, 512]; c10::BFloat16[4, 64, 64]; c10::BFloat16[4, 1, 512]; c10::BFloat16[4, 1, 64]; c10::Float8_e4m3fn[2329664, 1, 576]; long int[4]; long int[4]; c10::BFloat16[1048576, 1, 1, 32]; c10::BFloat16[1048576, 1, 1, 32] | [256, 1, 1] | 4 | 18.9 | 0.1% |
| `_ZN5aiter23fused_qk_rmsnorm_kernelIDF16bLi256ELi8ELb1ELi1EEEvPT_S2_PKS1_S4_S4_S4_ffiiiiiii` | `aiter::_fused_qk_rmsnorm_kernel` | c10::BFloat16[4, 2048]; c10::BFloat16[2048]; c10::BFloat16[4, 512]; c10::BFloat16[512]; c10::BFloat16[4, 2048]; c10::BFloat16[4, 512] | [4, 2, 1] | 4 | 17.7 | 0.1% |
| `_fused_append_shared_experts_kernel` | `-` |  | [16, 1, 1] | 4 | 17.3 | 0.1% |
| `void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocprim::ROCPRIM_400200_NS::...` | `aten::cumsum` | long int[4] | [1, 1, 1] | 4 | 17.2 | 0.1% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::...` | `aten::fill_` | float[2] | [1, 1, 1] | 4 | 17.1 | 0.1% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<long>, std::a...` | `aten::fill_` |  | [1, 1, 1] | 4 | 16.5 | 0.1% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_ke...` | `aten::copy_` | float[4, 154880]; c10::BFloat16[4, 154880] | [303, 1, 1] | 4 | 15.1 | 0.1% |
| `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymou...` | `aten::_foreach_copy_` | TensorList[[4], [4], [4], [4], [4]]; TensorList[[4], [4], [4], [4], [4]] | [5, 1, 1] | 4 | 14.1 | 0.1% |

## DRAFT_EXTEND_V2 (draft_extend)

| kernel | launching op | input dims | grid | calls/iter | us/iter | % phase |
|---|---|---|---|---|---|---|
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::reg_all_gather_into_tensor > record_param_comms` | c10::BFloat16[24, 6144] | [36, 1, 1] | 1 | 4785.6 | 51.8% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::reg_reduce_scatter_tensor > record_param_comms` | c10::BFloat16[96, 6144] | [48, 1, 1] | 1 | 2300.9 | 24.9% |
| `_ZN2ck15kernel_moe_gemmINS_15GridwiseMoeGemmINS_13tensor_layout4gemm8RowMajorENS3_11Col...` | `aiter::fused_moe_ > aiter::ck_moe_stage1` | c10::BFloat16[96, 6144]; c10::BFloat16[257, 1024, 6144]; c10::BFloat16[257, 6144, 512]; int[9079]; int[284]; int[2]; c10::BFloat16[96, 9, 512] | [8, 284, 1] | 1 | 402.4 | 4.4% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::reg_all_gather_into_tensor > record_param_comms` | c10::BFloat16[96, 38720] | [48, 1, 1] | 1 | 296.7 | 3.2% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `sglang::inplace_all_reduce > record_param_comms` | TensorList[[96, 6144]] | [48, 1, 1] | 1 | 257.0 | 2.8% |
| `_ZN2ck15kernel_moe_gemmINS_15GridwiseMoeGemmINS_13tensor_layout4gemm8RowMajorENS3_11Col...` | `aiter::fused_moe_ > aiter::ck_moe_stage2` | c10::BFloat16[96, 9, 512]; c10::BFloat16[257, 1024, 6144]; c10::BFloat16[257, 6144, 512]; int[9079]; int[284]; int[2]; c10::BFloat16[96, 6144]; float[9079] | [48, 284, 1] | 1 | 190.5 | 2.1% |
| `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` | `-` |  | [480, 1, 1] | 1 | 108.7 | 1.2% |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOp...` | `aten::argmax` | float[4, 154880] | [4, 1, 1] | 2 | 96.7 | 1.0% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x96x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `aten::mm` | c10::BFloat16[96, 6144]; c10::BFloat16[6144, 38720] | [51712, 1, 1] | 1 | 89.6 | 1.0% |
| `_sparse_mla_split_k_kernel` | `-` |  | [24, 4, 2] | 1 | 54.5 | 0.6% |
| `void at::native::(anonymous namespace)::cunn_SoftMaxForwardGmem<4, float, float, float,...` | `aten::_softmax` | float[4, 154880] | [4, 1, 1] | 1 | 50.6 | 0.5% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x32x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `aiter::gemm_a16w16 > aten::mm` | c10::BFloat16[24, 16384]; c10::BFloat16[16384, 6144] | [61440, 1, 1] | 1 | 44.5 | 0.5% |
| `hgemm_bf16_32x128x128x3_SPK4_W2x2x2_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[24, 12288]; c10::BFloat16[6144, 12288] | [48, 4, 1] | 1 | 31.7 | 0.3% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous name...` | `aten::cat` | TensorList[[4], [0]] | [1, 2, 1] | 6 | 24.4 | 0.3% |
| `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_n...` | `aten::copy_` | c10::BFloat16[96, 4, 38720]; c10::BFloat16[96, 4, 38720] | [14520, 1, 1] | 1 | 24.4 | 0.3% |
| `_gemm_a16_w16_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_NU...` | `aiter::gemm_a16w16 > aiter::gemm_a16w16_` | c10::BFloat16[24, 2048]; c10::BFloat16[16384, 2048] | [512, 1, 1] | 1 | 17.5 | 0.2% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x32x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `aten::bmm` | c10::BFloat16[64, 24, 512]; c10::BFloat16[64, 512, 256]; c10::BFloat16[64, 24, 256] | [16384, 1, 1] | 1 | 12.7 | 0.1% |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb0ELb0ELb1ELi1EEEvPT0_PT_PfS4...` | `aiter::rmsnorm2d_fwd > aiter::rmsnorm` | c10::BFloat16[24, 6144]; c10::BFloat16[24, 6144]; c10::BFloat16[6144] | [24, 1, 1] | 3 | 12.7 | 0.1% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous name...` | `aten::cat` | TensorList[[24], [0]] | [1, 2, 1] | 3 | 12.2 | 0.1% |
| `hgemm_bf16_32x64x128x4_SPK4_W1x4x1_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[24, 6144]; c10::BFloat16[2624, 6144] | [41, 4, 1] | 1 | 12.1 | 0.1% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `aiter::gemm_a16w16 > aten::mm` | c10::BFloat16[24, 6144]; c10::BFloat16[6144, 32] | [1024, 1, 1] | 1 | 11.3 | 0.1% |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb1ELb0ELb1ELi1EEEvPT0_PT_PfS4...` | `aiter::rmsnorm2d_fwd_with_add > aiter::add_rmsnorm` | c10::BFloat16[24, 6144]; c10::BFloat16[24, 6144]; c10::BFloat16[24, 6144]; c10::BFloat16[24, 6144]; c10::BFloat16[6144] | [24, 1, 1] | 2 | 11.0 | 0.1% |
| `memcpy_triton_kernel` | `-` |  | [454, 1, 1] | 2 | 10.6 | 0.1% |
| `void aiter::kn_entry_2c_sbhd_cached_indirect_inplace<aiter::OpCachedFwd, 1, true, false...` | `aiter::rope_cached_positions_2c_fwd_impl` | c10::BFloat16[1, 24, 32, 64]; c10::BFloat16[1, 24, 1, 64]; c10::BFloat16[1, 24, 32, 64]; c10::BFloat16[1, 24, 1, 64]; c10::BFloat16[1048576, 1, 1, 32]; c10::BFloat16[1048576, 1, 1, 32]; long int[1, 24] | [3, 1, 1] | 1 | 10.1 | 0.1% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | int[4]; long int[4] | [1, 1, 1] | 2 | 9.3 | 0.1% |
| `void sgl::dsa_topk::coop_mb_scatter<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbPara...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 240064]; int[24]; int[24, 2048]; int[24, 240037]; int[25] | [24, 10, 1] | 0.5 | 9.2 | 0.1% |
| `void sgl::dsa_topk::coop_mb_scatter<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbPara...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 240064]; int[24]; int[24, 2048]; int[24, 240041]; int[25] | [24, 10, 1] | 0.5 | 9.1 | 0.1% |
| `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_g...` | `aten::gather` | float[4, 154880]; long int[4, 1] | [1, 1, 1] | 2 | 8.3 | 0.1% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::ar...` | `aten::fill_` | int[5] | [1, 1, 1] | 2 | 8.3 | 0.1% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<lon...` | `aten::add` | long int[4] | [1, 1, 1] | 2 | 8.3 | 0.1% |
| `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_...` | `aten::arange` | int[0] | [30, 1, 1] | 2 | 8.2 | 0.1% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x32x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_...` | `aten::bmm` | c10::BFloat16[64, 24, 192]; c10::BFloat16[64, 192, 512] | [65536, 1, 1] | 1 | 7.9 | 0.1% |
| `void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocprim::ROCPRIM_400200_NS::...` | `aten::cumsum` | int[4] | [1, 1, 1] | 2 | 7.9 | 0.1% |
| `hgemm_bf16_16x64x128x4_SPK2_W1x2x1_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[24, 2048]; c10::BFloat16[4096, 2048] | [128, 2, 1] | 1 | 7.7 | 0.1% |
| `Cijk_SB_BiasS_HAS_ScaleAlphaVec_PostGSU16_VW4` | `aiter::gemm_a16w16 > aten::mm` | c10::BFloat16[24, 16384]; c10::BFloat16[16384, 6144] | [36864, 1, 1] | 1 | 7.6 | 0.1% |
| `void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_kernel<flo...` | `aten::exponential_` | float[4, 154880] | [2048, 1, 1] | 1 | 7.5 | 0.1% |
| `_sparse_mla_reduce_kernel` | `-` |  | [24, 64, 8] | 1 | 7.4 | 0.1% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | long int[3751]; int[3751] | [8, 1, 1] | 1 | 7.0 | 0.1% |
| `hgemm_bf16_16x64x128x5_SPK8_W1x2x1_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[96, 6144]; c10::BFloat16[256, 6144] | [24, 8, 1] | 1 | 6.8 | 0.1% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_n...` | `aten::copy_` | int[4, 6, 240041]; int[4, 6, 240041] | [11252, 1, 1] | 0.5 | 6.7 | 0.1% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_n...` | `aten::copy_` | int[4, 6, 240037]; int[4, 6, 240037] | [11252, 1, 1] | 0.5 | 6.7 | 0.1% |
| `void sgl::dsa_topk::coop_mb_refine<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParam...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 240064]; int[24]; int[24, 2048]; int[24, 240041]; int[25] | [24, 1, 1] | 0.5 | 6.7 | 0.1% |
| `void sgl::dsa_topk::coop_mb_refine<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParam...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 240064]; int[24]; int[24, 2048]; int[24, 240037]; int[25] | [24, 1, 1] | 0.5 | 6.5 | 0.1% |
| `void aiter::grouped_topk_kernel<hip_bfloat16, float __vector(4), 1, true, true, false>(...` | `aiter::biased_grouped_topk_hip` | c10::BFloat16[96, 256]; c10::BFloat16[256]; float[96, 8]; int[96, 8] | [96, 1, 1] | 1 | 6.0 | 0.1% |
| `hgemm_bf16_16x64x128x4_SPK12_W1x2x1_BLDS1_TN_AS1_0` | `aiter::gemm_a16w16` | c10::BFloat16[24, 6144]; c10::BFloat16[128, 6144] | [4, 12, 1] | 1 | 5.8 | 0.1% |
| `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16...` | `aten::fill_` | c10::BFloat16[24, 154880] | [1815, 1, 1] | 1 | 5.8 | 0.1% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_ke...` | `aten::copy_` | float[24, 154880]; c10::BFloat16[24, 154880] | [1815, 1, 1] | 1 | 5.5 | 0.1% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymo...` | `aten::cat` | TensorList[[24, 6144], [0, 6144]] | [72, 2, 1] | 1 | 5.5 | 0.1% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_n...` | `aten::div` | float[4, 154880]; float[4, 1] | [1210, 1, 1] | 1 | 5.3 | 0.1% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, floa...` | `aten::div` | float[4, 154880]; float[4, 154880] | [605, 1, 1] | 1 | 5.2 | 0.1% |
| `act_quant_kernel__kernel` | `-` |  | [24, 1, 1] | 1 | 5.1 | 0.1% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | long int[24]; int[24] | [1, 1, 1] | 1 | 5.0 | 0.1% |
| `triton_poi_fused_mul_unsqueeze_0` | `triton_poi_fused_mul_unsqueeze_0` | c10::BFloat16[24, 32]; float[24, 32, 1]; float[24, 32, 1] | [6, 1, 1] | 1 | 5.0 | 0.1% |
| `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16...` | `aten::fill_` | c10::BFloat16[96, 6144] | [288, 1, 1] | 1 | 4.8 | 0.1% |
| `_ZN5aiter32indexer_k_quant_and_cache_kernelIDF16bDB8_LN4vllm18Fp8KVCacheDataTypeE1ELi8E...` | `aiter::indexer_k_quant_and_cache` | c10::BFloat16[24, 128]; c10::Float8_e4m3fn[36401, 64, 132]; long int[24] | [3, 1, 1] | 1 | 4.8 | 0.1% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymo...` | `aten::cat` | TensorList[[24, 6144], [24, 6144]] | [72, 2, 1] | 1 | 4.7 | 0.1% |
| `void fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<16, 7, c10::B...` | `HadamardTransformFn` | c10::BFloat16[24, 128] | [24, 1, 1] | 1 | 4.7 | 0.1% |
| `_fused_append_shared_experts_kernel` | `-` |  | [96, 1, 1] | 1 | 4.7 | 0.1% |
| `p23_kernel_1` | `aiter::fused_moe_` | c10::BFloat16[96, 6144]; c10::BFloat16[257, 1024, 6144]; c10::BFloat16[257, 6144, 512]; float[96, 9]; int[96, 9] | [769, 1, 1] | 1 | 4.7 | 0.1% |
| `_ZN7ck_tile6kentryILi1ENS_14Layernorm2dFwdINS_29Layernorm2dFwdPipelineOnePassINS_29Laye...` | `aiter::layernorm2d_fwd` | c10::BFloat16[24, 128]; c10::BFloat16[128]; c10::BFloat16[128] | [6, 1, 1] | 1 | 4.6 | 0.1% |
| `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, lon...` | `aten::gather` | c10::BFloat16[154880, 6144]; long int[24, 6144]; c10::BFloat16[24, 6144] | [24, 3, 1] | 1 | 4.6 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, i...` | `aten::floor_divide` | int[24, 3751] | [88, 1, 1] | 1 | 4.6 | 0.0% |
| `_ZN5aiter23fused_qk_rmsnorm_kernelIDF16bLi256ELi8ELb1ELi1EEEvPT_S2_PKS1_S4_S4_S4_ffiiiiiii` | `aiter::_fused_qk_rmsnorm_kernel` | c10::BFloat16[24, 2048]; c10::BFloat16[2048]; c10::BFloat16[24, 512]; c10::BFloat16[512]; c10::BFloat16[24, 2048]; c10::BFloat16[24, 512] | [24, 2, 1] | 1 | 4.6 | 0.0% |
| `_fused_qk_rope_cat_and_cache_mla_kernel` | `aiter::fused_qk_rope_cat_and_cache_mla` | c10::BFloat16[24, 64, 512]; c10::BFloat16[24, 64, 64]; c10::BFloat16[24, 1, 512]; c10::BFloat16[24, 1, 64]; c10::Float8_e4m3fn[2329664, 1, 576]; long int[24]; long int[24]; c10::BFloat16[1048576, 1, 1, 32]; c10::BFloat16[1048576, 1, 1, 32] | [1536, 1, 1] | 1 | 4.5 | 0.0% |
| `void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocprim::ROCPRIM_400200_NS::...` | `aten::cumsum` | long int[4] | [1, 1, 1] | 1 | 4.5 | 0.0% |
| `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, lon...` | `aten::index` | c10::BFloat16[24, 6144] | [4, 3, 1] | 1 | 4.5 | 0.0% |
| `p0v2_kernel_0` | `aiter::fused_moe_` | c10::BFloat16[96, 6144]; c10::BFloat16[257, 1024, 6144]; c10::BFloat16[257, 6144, 512]; float[96, 9]; int[96, 9] | [257, 1, 1] | 1 | 4.5 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::...` | `aten::fill_` | float[2] | [1, 1, 1] | 1 | 4.4 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::la...` | `aten::clamp_min_` | float[4, 154880] | [605, 1, 1] | 1 | 4.4 | 0.0% |
| `void sgl::dsa_topk::coop_mb_hist0<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParams...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 240064]; int[24]; int[24, 2048]; int[24, 240037]; int[25] | [24, 10, 1] | 0.5 | 4.4 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::ar...` | `aten::fill_` | int[25] | [1, 1, 1] | 1 | 4.4 | 0.0% |
| `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, lon...` | `aten::index` | float[24, 154880] | [4, 152, 1] | 1 | 4.4 | 0.0% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::index` | int[17, 240037] | [1876, 1, 1] | 0.5 | 4.3 | 0.0% |
| `void sgl::dsa_topk::coop_mb_hist0<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParams...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 240064]; int[24]; int[24, 2048]; int[24, 240041]; int[25] | [24, 10, 1] | 0.5 | 4.3 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int...` | `aten::sub` | int[4] | [1, 1, 1] | 1 | 4.2 | 0.0% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::index` | int[17, 240041] | [1876, 1, 1] | 0.5 | 4.2 | 0.0% |
| `compute_position_kernel` | `-` |  | [4, 1, 1] | 1 | 4.2 | 0.0% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | long int[4]; int[4] | [1, 1, 1] | 1 | 4.2 | 0.0% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::add` | long int[4]; int[4] | [1, 1, 1] | 1 | 4.2 | 0.0% |
| `void fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<16, 7, c10::B...` | `HadamardTransformFn` | c10::BFloat16[24, 32, 128] | [768, 1, 1] | 1 | 4.1 | 0.0% |
| `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_...` | `aten::arange` | long int[0] | [1, 1, 1] | 1 | 4.1 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<long>, std::a...` | `aten::fill_` |  | [1, 1, 1] | 1 | 4.1 | 0.0% |
| `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, lon...` | `aten::index` | int[24, 2048] | [4, 2, 1] | 1 | 4.1 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::la...` | `aten::clamp` | int[24] | [1, 1, 1] | 1 | 4.1 | 0.0% |
| `void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocprim::ROCPRIM_400200_NS::...` | `aten::cumsum` | int[24] | [1, 1, 1] | 1 | 4.1 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::wh...` | `aten::where` | bool[4, 1]; long int[4, 1]; long int[4, 1] | [1, 1, 1] | 1 | 4.1 | 0.0% |
| `seqlens_expand_kernel` | `-` |  | [4, 1, 1] | 1 | 4.1 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::compare_scalar_kernel<int...` | `aten::le` | int[4] | [1, 1, 1] | 1 | 4.1 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<lon...` | `aten::sub` | long int[4] | [1, 1, 1] | 1 | 4.0 | 0.0% |
| `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymou...` | `aten::_foreach_copy_` | TensorList[[24], [24], [24], [4], [4]]; TensorList[[24], [24], [24], [4], [4]] | [5, 1, 1] | 1 | 3.5 | 0.0% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::index` | int[24, 240041] | [176, 1, 1] | 0.5 | 2.8 | 0.0% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::index` | int[24, 240037] | [176, 1, 1] | 0.5 | 2.7 | 0.0% |
| `void sgl::dsa_topk::coop_mb_hist1<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParams...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 240064]; int[24]; int[24, 2048]; int[24, 240037]; int[25] | [24, 10, 1] | 0.5 | 2.6 | 0.0% |
| `void sgl::dsa_topk::coop_mb_hist1<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParams...` | `sgl_kernel::fast_topk_transform_fused` | float[24, 240064]; int[24]; int[24, 2048]; int[24, 240041]; int[25] | [24, 10, 1] | 0.5 | 2.5 | 0.0% |

## draft glue (draft, outside steps)

| kernel | launching op | input dims | grid | calls/iter | us/iter | % phase |
|---|---|---|---|---|---|---|
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOp...` | `aten::argmax` | float[4, 154880] | [4, 1, 1] | 8 | 380.6 | 32.8% |
| `void at::native::(anonymous namespace)::cunn_SoftMaxForwardGmem<4, float, float, float,...` | `aten::_softmax` | float[4, 154880] | [4, 1, 1] | 4 | 208.9 | 18.0% |
| `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_g...` | `aten::gather` | float[4, 154880]; long int[4, 1] | [1, 1, 1] | 8 | 51.6 | 4.4% |
| `void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocprim::ROCPRIM_400200_NS::...` | `aten::cumsum` | int[4] | [1, 1, 1] | 8 | 47.4 | 4.1% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, floa...` | `aten::div` | float[4, 154880]; float[4, 154880] | [605, 1, 1] | 4 | 35.6 | 3.1% |
| `void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_kernel<flo...` | `aten::exponential_` | float[4, 154880] | [2048, 1, 1] | 4 | 33.3 | 2.9% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<lon...` | `aten::add_` | long int[4] | [1, 1, 1] | 4 | 30.0 | 2.6% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | long int[3751]; int[3751] | [8, 1, 1] | 4 | 23.7 | 2.0% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::index` | int[4, 240031] | [30, 1, 1] | 2 | 23.6 | 2.0% |
| `triton_poi_fused_add_arange_floor_divide_index_max_unsqueeze_view_2` | `triton_poi_fused_add_arange_floor_divide_index_max_unsqueeze_view_2` | c10::BFloat16[4, 6144]; c10::BFloat16[4, 6144] | [96, 1, 1] | 4 | 21.9 | 1.9% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::wh...` | `aten::where` | bool[4, 1]; long int[4, 1]; long int[4, 1] | [1, 1, 1] | 4 | 19.7 | 1.7% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_n...` | `aten::div` | float[4, 154880]; float[4, 1] | [1210, 1, 1] | 4 | 18.6 | 1.6% |
| `triton_poi_fused_max_mul_unsqueeze_view_0` | `triton_poi_fused_max_mul_unsqueeze_view_0` | float[4, 1]; float[4, 1]; float[4, 1, 1]; float[4, 1] | [1, 1, 1] | 4 | 18.3 | 1.6% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::la...` | `aten::clamp_min_` | float[4, 154880] | [605, 1, 1] | 4 | 18.2 | 1.6% |
| `triton_poi_fused_gather_max_view_1` | `triton_poi_fused_gather_max_view_1` | long int[4, 1]; long int[4, 1] | [1, 1, 1] | 4 | 18.1 | 1.6% |
| `triton_poi_fused_add_max_view_3` | `triton_poi_fused_add_max_view_3` | long int[4, 1] | [1, 1, 1] | 4 | 17.6 | 1.5% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::ar...` | `aten::fill_` | int[5] | [1, 1, 1] | 8 | 16.9 | 1.5% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::compare_scalar_kernel<int...` | `aten::le` | int[4] | [1, 1, 1] | 4 | 16.2 | 1.4% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, i...` | `aten::floor_divide` | int[4, 3751] | [15, 1, 1] | 4 | 15.2 | 1.3% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::index` | int[17, 240031] | [1876, 1, 1] | 2 | 14.2 | 1.2% |
| `void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::...` | `aten::fill_` | bool[5760888] | [1407, 1, 1] | 0.5 | 14.2 | 1.2% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::index` | int[17, 240035] | [1876, 1, 1] | 2 | 14.1 | 1.2% |
| `void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::...` | `aten::fill_` | bool[5760984] | [1407, 1, 1] | 0.5 | 13.2 | 1.1% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::index` | int[4, 240035] | [30, 1, 1] | 2 | 12.5 | 1.1% |
| `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_...` | `aten::arange` | int[0] | [30, 1, 1] | 4 | 11.0 | 1.0% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymo...` | `aten::cat` | TensorList[[4, 154880], [4, 154880], [4, 154880], [4, 154880], [4, 154880]] | [303, 5, 1] | 1 | 9.6 | 0.8% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | int[4]; long int[4] | [1, 1, 1] | 4 | 9.1 | 0.8% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::la...` | `aten::clamp` | int[4] | [1, 1, 1] | 4 | 7.8 | 0.7% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<lon...` | `aten::add` | long int[4] | [1, 1, 1] | 4 | 7.5 | 0.6% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::cat` | TensorList[[4, 1], [4, 5]] | [1, 1, 1] | 1 | 5.0 | 0.4% |
| `build_tree_efficient(long*, long*, long*, bool*, long*, long*, long*, long*, int, int, ...` | `sgl_kernel::build_tree_kernel_efficient` | long int[4, 5]; long int[4, 5]; long int[4]; bool[5760984]; long int[24]; long int[4, 6]; long int[4, 6]; long int[4, 6] | [4, 1, 1] | 0.5 | 5.0 | 0.4% |
| `build_tree_efficient(long*, long*, long*, bool*, long*, long*, long*, long*, int, int, ...` | `sgl_kernel::build_tree_kernel_efficient` | long int[4, 5]; long int[4, 5]; long int[4]; bool[5760888]; long int[24]; long int[4, 6]; long int[4, 6]; long int[4, 6] | [4, 1, 1] | 0.5 | 4.7 | 0.4% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<long>, std::a...` | `aten::fill_` | long int[3, 4, 6] | [1, 1, 1] | 1 | 4.6 | 0.4% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymo...` | `aten::cat` | TensorList[[4, 1], [4, 1], [4, 1], [4, 1], [4, 1]] | [1, 5, 1] | 1 | 4.2 | 0.4% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_n...` | `aten::cat` | TensorList[[4, 1], [4, 5]] | [1, 1, 1] | 1 | 4.2 | 0.4% |
| `assign_draft_cache_locs_contiguous` | `-` |  | [4, 1, 1] | 1 | 2.5 | 0.2% |
| `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_...` | `aten::arange` | long int[0] | [1, 1, 1] | 1 | 2.0 | 0.2% |

## <none>

| kernel | launching op | input dims | grid | calls/iter | us/iter | % phase |
|---|---|---|---|---|---|---|
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `record_param_comms` | TensorList[[7]] | [1, 1, 1] | 1 | 711.6 | 83.1% |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOp...` | `aten::argmax` | float[24, 154880] | [24, 1, 1] | 1 | 49.3 | 5.8% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::ar...` | `aten::fill_` | int[4, 6] | [1, 1, 1] | 2 | 9.3 | 1.1% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | int[4]; long int[4] | [1, 1, 1] | 2 | 8.6 | 1.0% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | int[4, 4]; long int[4, 4] | [1, 1, 1] | 1 | 5.8 | 0.7% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::add` | long int[4]; int[4] | [1, 1, 1] | 1 | 5.3 | 0.6% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::add` | int[4, 1]; long int[4] | [1, 1, 1] | 1 | 5.3 | 0.6% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | long int[4, 3]; int[4, 3] | [1, 1, 1] | 1 | 5.1 | 0.6% |
| `assign_extend_cache_locs_uniform` | `-` |  | [4, 1, 1] | 1 | 5.0 | 0.6% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::index` | int[24] | [1, 1, 1] | 1 | 4.9 | 0.6% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | long int[4, 6]; int[4, 6] | [1, 1, 1] | 1 | 4.8 | 0.6% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | int[4, 3]; long int[4, 3] | [1, 1, 1] | 1 | 4.5 | 0.5% |
| `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_...` | `aten::arange` | long int[0] | [1, 1, 1] | 1 | 4.4 | 0.5% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::ar...` | `aten::fill_` | int[4] | [1, 1, 1] | 1 | 4.1 | 0.5% |
| `fill_bonus_tokens` | `-` |  | [4, 1, 1] | 1 | 4.1 | 0.5% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | long int[4]; int[4] | [1, 1, 1] | 1 | 4.1 | 0.5% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::ar...` | `aten::fill_` | int[24] | [1, 1, 1] | 1 | 4.0 | 0.5% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::_index_put_impl_` | int[24]; int[4] | [1, 1, 1] | 1 | 4.0 | 0.5% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::_index_put_impl_` | int[24]; int[4, 3] | [1, 1, 1] | 1 | 4.0 | 0.5% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int...` | `aten::add` | int[4] | [1, 1, 1] | 1 | 4.0 | 0.5% |
| `void VerifyTreeGreedy<int, long>(int*, int*, int*, long*, long*, long*, long*, long*, u...` | `sgl_kernel::verify_tree_greedy` | int[24]; int[4, 6]; int[4]; long int[4, 6]; long int[4, 6]; long int[4, 6]; long int[4, 6]; long int[4, 6] | [4, 1, 1] | 1 | 3.9 | 0.5% |
