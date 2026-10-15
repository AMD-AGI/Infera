# Kernel → op → input shapes — results/p1g2_bs16_isl110k_on/profile/decode-1791463931.645337-TP-0-DP-0.trace.json.gz

Per-iteration device time over 5 profiled iteration(s). Phase from host annotations.

| phase | kernel us / iter | share of total |
|---|---|---|
| TARGET_VERIFY (step[VERIFY]) | 33079.6 | 94.9% |
| DRAFT_EXTEND_V2 (draft_extend) | 1517.7 | 4.4% |
| <none> | 229.3 | 0.7% |
| draft glue (draft, outside steps) | 36.6 | 0.1% |

## TARGET_VERIFY (step[VERIFY])

| kernel | launching op | input dims | grid | calls/iter | us/iter | % phase |
|---|---|---|---|---|---|---|
| `gemm1_a4w4_port_h6144_i512_ne257_bm16_iq_sep` | `-` |  | None | 75 | 6818.8 | 20.6% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `-` |  | None | 158 | 4435.5 | 13.4% |
| `_sparse_mla_split_k_kernel` | `-` |  | None | 78 | 4089.5 | 12.4% |
| `gemm2_a4w4_port_hmax8192_imax512_bm16_bn128_bk256_nt_atomic_persist_cu256_bhoist_apf_sp...` | `-` |  | None | 75 | 3320.0 | 10.0% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x32x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `-` |  | None | 78 | 2779.4 | 8.4% |
| `_gemm_a16_w16_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_NU...` | `-` |  | None | 78 | 1358.1 | 4.1% |
| `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` | `-` |  | None | 21 | 1077.9 | 3.3% |
| `_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel_HAS_BIA...` | `-` |  | None | 156 | 1070.9 | 3.2% |
| `hgemm_bf16_32x64x128x4_SPK4_W1x4x1_BLDS1_TN_AS1_0` | `-` |  | None | 78 | 821.2 | 2.5% |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb1ELb0ELb1ELi1EEEvPT0_PT_PfS4...` | `-` |  | None | 156 | 757.8 | 2.3% |
| `_sparse_mla_reduce_kernel` | `-` |  | None | 78 | 553.0 | 1.7% |
| `Cijk_SB_BiasS_HAS_ScaleAlphaVec_PostGSU16_VW4` | `-` |  | None | 78 | 514.2 | 1.6% |
| `hgemm_bf16_16x64x128x5_SPK8_W1x2x1_BLDS1_TN_AS1_0` | `-` |  | None | 75 | 502.3 | 1.5% |
| `void aiter::mxfp4_moe::moe_sort_quant::sort_quant_kernel_impl<257, 9, 16, 6144, 128, 10...` | `-` |  | None | 75 | 403.5 | 1.2% |
| `void aiter::grouped_topk_kernel<hip_bfloat16, float __vector(4), 1, true, true, false>(...` | `-` |  | None | 75 | 363.4 | 1.1% |
| `memcpy_triton_kernel` | `-` |  | None | 79 | 361.8 | 1.1% |
| `_fused_qk_rope_cat_and_cache_mla_kernel` | `-` |  | None | 78 | 359.9 | 1.1% |
| `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16...` | `-` |  | None | 79 | 350.9 | 1.1% |
| `_ZN5aiter23fused_qk_rmsnorm_kernelIDF16bLi256ELi8ELb1ELi1EEEvPT_S2_PKS1_S4_S4_S4_ffiiiiiii` | `-` |  | None | 78 | 347.0 | 1.0% |
| `_fused_append_shared_experts_kernel` | `-` |  | None | 75 | 334.1 | 1.0% |
| `void sgl::dsa_topk::coop_mb_scatter<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbPara...` | `-` |  | None | 21 | 277.9 | 0.8% |
| `void sgl::dsa_topk::coop_mb_refine<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParam...` | `-` |  | None | 21 | 261.7 | 0.8% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `-` |  | None | 21 | 233.6 | 0.7% |
| `void aiter::kn_entry_2c_sbhd_cached_indirect_inplace<aiter::OpCachedFwd, 1, true, false...` | `-` |  | None | 21 | 193.4 | 0.6% |
| `void fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<16, 7, c10::B...` | `-` |  | None | 42 | 179.8 | 0.5% |
| `hgemm_bf16_16x64x128x4_SPK2_W1x2x1_BLDS1_TN_AS1_0` | `-` |  | None | 21 | 158.2 | 0.5% |
| `void sgl::dsa_topk::coop_mb_hist0<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParams...` | `-` |  | None | 21 | 128.8 | 0.4% |
| `hgemm_bf16_16x64x128x4_SPK12_W1x2x1_BLDS1_TN_AS1_0` | `-` |  | None | 21 | 118.9 | 0.4% |
| `void sgl::dsa_topk::coop_mb_hist1<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParams...` | `-` |  | None | 21 | 118.3 | 0.4% |
| `act_quant_kernel__kernel` | `-` |  | None | 21 | 108.0 | 0.3% |
| `_ZN5aiter32indexer_k_quant_and_cache_kernelIDF16bDB8_LN4vllm18Fp8KVCacheDataTypeE1ELi8E...` | `-` |  | None | 21 | 98.3 | 0.3% |
| `_ZN7ck_tile6kentryILi1ENS_14Layernorm2dFwdINS_29Layernorm2dFwdPipelineOnePassINS_29Laye...` | `-` |  | None | 21 | 95.3 | 0.3% |
| `triton_poi_fused_mul_unsqueeze_0` | `-` |  | None | 21 | 93.3 | 0.3% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x96x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `-` |  | None | 1 | 90.4 | 0.3% |
| `__amd_rocclr_fillBufferAligned` | `-` |  | None | 21 | 89.3 | 0.3% |
| `hgemm_bf16_64x64x128x3_SPK1_W2x2x2_BLDS1_TN_AS1_0` | `-` |  | None | 3 | 86.0 | 0.3% |
| `hgemm_bf16_64x64x128x3_SPK1_W4x1x1_BLDS1_TN_AS1_0` | `-` |  | None | 3 | 48.8 | 0.1% |
| `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_n...` | `-` |  | None | 1 | 23.0 | 0.1% |
| `_ZN7sgl_hip10activation18act_and_mul_kernelI14__hip_bfloat16TnPFT_RKS3_EXadL_Z4siluIS2_...` | `-` |  | None | 3 | 12.6 | 0.0% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `-` |  | None | 2 | 8.5 | 0.0% |
| `void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocprim::ROCPRIM_400200_NS::...` | `-` |  | None | 2 | 8.2 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::ar...` | `-` |  | None | 1 | 5.2 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_ke...` | `-` |  | None | 1 | 5.2 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::...` | `-` |  | None | 1 | 4.8 | 0.0% |
| `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, lon...` | `-` |  | None | 1 | 4.4 | 0.0% |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb0ELb0ELb1ELi1EEEvPT0_PT_PfS4...` | `-` |  | None | 1 | 4.3 | 0.0% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<long>, std::a...` | `-` |  | None | 1 | 4.0 | 0.0% |

## DRAFT_EXTEND_V2 (draft_extend)

| kernel | launching op | input dims | grid | calls/iter | us/iter | % phase |
|---|---|---|---|---|---|---|
| `_ZN2ck15kernel_moe_gemmINS_15GridwiseMoeGemmINS_13tensor_layout4gemm8RowMajorENS3_11Col...` | `-` |  | None | 2 | 564.4 | 37.2% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `-` |  | None | 4 | 133.2 | 8.8% |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOp...` | `aten::argmax` | float[4, 154880] | [4, 1, 1] | 2 | 92.6 | 6.1% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `-` |  | None | 1 | 89.9 | 5.9% |
| `_sparse_mla_split_k_kernel` | `-` |  | None | 1 | 52.6 | 3.5% |
| `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` | `-` |  | None | 1 | 51.3 | 3.4% |
| `void at::native::(anonymous namespace)::cunn_SoftMaxForwardGmem<4, float, float, float,...` | `aten::_softmax` | float[4, 154880] | [4, 1, 1] | 1 | 50.8 | 3.3% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x32x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `-` |  | None | 2 | 49.1 | 3.2% |
| `hgemm_bf16_32x128x128x3_SPK4_W2x2x2_BLDS1_TN_AS1_0` | `-` |  | None | 1 | 34.6 | 2.3% |
| `_gemm_a16_w16_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_NU...` | `-` |  | None | 1 | 17.7 | 1.2% |
| `_fused_dsa_draft_extend_metadata_kernel` | `-` |  | [32773, 1, 1] | 1 | 16.5 | 1.1% |
| `void sgl::dsa_topk::coop_mb_scatter<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbPara...` | `-` |  | None | 1 | 13.2 | 0.9% |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb0ELb0ELb1ELi1EEEvPT0_PT_PfS4...` | `-` |  | None | 3 | 13.0 | 0.9% |
| `void sgl::dsa_topk::coop_mb_refine<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParam...` | `-` |  | None | 1 | 12.6 | 0.8% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::ar...` | `aten::fill_` | int[4] | [1, 1, 1] | 3 | 12.6 | 0.8% |
| `hgemm_bf16_32x64x128x4_SPK4_W1x4x1_BLDS1_TN_AS1_0` | `-` |  | None | 1 | 11.8 | 0.8% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1...` | `-` |  | None | 1 | 11.3 | 0.7% |
| `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb1ELb0ELb1ELi1EEEvPT0_PT_PfS4...` | `-` |  | None | 2 | 10.2 | 0.7% |
| `void aiter::kn_entry_2c_sbhd_cached_indirect_inplace<aiter::OpCachedFwd, 1, true, false...` | `-` |  | None | 1 | 9.5 | 0.6% |
| `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16...` | `-` |  | None | 2 | 9.4 | 0.6% |
| `memcpy_triton_kernel` | `-` |  | None | 2 | 9.0 | 0.6% |
| `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, lon...` | `-` |  | None | 2 | 8.7 | 0.6% |
| `void fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<16, 7, c10::B...` | `-` |  | None | 2 | 8.5 | 0.6% |
| `void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_kernel<flo...` | `aten::exponential_` | float[4, 154880] | [2048, 1, 1] | 1 | 8.1 | 0.5% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `-` |  | None | 2 | 8.1 | 0.5% |
| `void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<rocprim::ROCPRIM_400200_NS::...` | `-` |  | None | 2 | 8.0 | 0.5% |
| `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_g...` | `aten::gather` | float[4, 154880]; long int[4, 1] | [1, 1, 1] | 2 | 7.8 | 0.5% |
| `hgemm_bf16_16x64x128x4_SPK2_W1x2x1_BLDS1_TN_AS1_0` | `-` |  | None | 1 | 7.7 | 0.5% |
| `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_n...` | `-` |  | None | 1 | 7.6 | 0.5% |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x32x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_...` | `-` |  | None | 1 | 7.6 | 0.5% |
| `Cijk_SB_BiasS_HAS_ScaleAlphaVec_PostGSU16_VW4` | `-` |  | None | 1 | 7.4 | 0.5% |
| `_sparse_mla_reduce_kernel` | `-` |  | None | 1 | 7.3 | 0.5% |
| `hgemm_bf16_16x64x128x5_SPK8_W1x2x1_BLDS1_TN_AS1_0` | `-` |  | None | 1 | 7.0 | 0.5% |
| `void sgl::dsa_topk::coop_mb_hist0<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParams...` | `-` |  | None | 1 | 6.2 | 0.4% |
| `hgemm_bf16_16x64x128x4_SPK12_W1x2x1_BLDS1_TN_AS1_0` | `-` |  | None | 1 | 5.7 | 0.4% |
| `void sgl::dsa_topk::coop_mb_hist1<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopMbParams...` | `-` |  | None | 1 | 5.6 | 0.4% |
| `act_quant_kernel__kernel` | `-` |  | None | 1 | 5.3 | 0.3% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::ar...` | `-` |  | None | 1 | 5.2 | 0.3% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_n...` | `aten::div` | float[4, 154880]; float[4, 1] | [1210, 1, 1] | 1 | 5.2 | 0.3% |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymo...` | `-` |  | None | 1 | 4.9 | 0.3% |
| `_ZN5aiter32indexer_k_quant_and_cache_kernelIDF16bDB8_LN4vllm18Fp8KVCacheDataTypeE1ELi8E...` | `-` |  | None | 1 | 4.7 | 0.3% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::la...` | `aten::clamp_min_` | float[4, 154880] | [605, 1, 1] | 1 | 4.7 | 0.3% |
| `void aiter::grouped_topk_kernel<hip_bfloat16, float __vector(4), 1, true, true, false>(...` | `-` |  | None | 1 | 4.7 | 0.3% |
| `triton_poi_fused_mul_unsqueeze_0` | `-` |  | None | 1 | 4.7 | 0.3% |
| `_fused_qk_rope_cat_and_cache_mla_kernel` | `-` |  | None | 1 | 4.7 | 0.3% |
| `_ZN7ck_tile6kentryILi1ENS_14Layernorm2dFwdINS_29Layernorm2dFwdPipelineOnePassINS_29Laye...` | `-` |  | None | 1 | 4.6 | 0.3% |
| `p23_kernel_1` | `-` |  | None | 1 | 4.5 | 0.3% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, floa...` | `aten::div` | float[4, 154880]; float[4, 154880] | [605, 1, 1] | 1 | 4.5 | 0.3% |
| `_ZN5aiter23fused_qk_rmsnorm_kernelIDF16bLi256ELi8ELb1ELi1EEEvPT_S2_PKS1_S4_S4_S4_ffiiiiiii` | `-` |  | None | 1 | 4.5 | 0.3% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | long int[24]; int[24] | [1, 1, 1] | 1 | 4.4 | 0.3% |
| `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, lon...` | `aten::index` | int[24, 2048] | [4, 2, 1] | 1 | 4.4 | 0.3% |
| `p0v2_kernel_0` | `-` |  | None | 1 | 4.4 | 0.3% |
| `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymou...` | `aten::_foreach_copy_` | TensorList[[4], [4], [4]]; TensorList[[4], [4], [4]] | [3, 1, 1] | 1 | 4.3 | 0.3% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::...` | `-` |  | None | 1 | 4.3 | 0.3% |
| `__amd_rocclr_fillBufferAligned` | `-` |  | None | 1 | 4.2 | 0.3% |
| `_fused_append_shared_experts_kernel` | `-` |  | None | 1 | 4.2 | 0.3% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::add` | long int[4]; int[4] | [1, 1, 1] | 1 | 4.2 | 0.3% |
| `__amd_rocclr_copyBuffer` | `-` |  | None | 1 | 4.2 | 0.3% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<lon...` | `aten::add` | long int[4] | [1, 1, 1] | 1 | 4.2 | 0.3% |
| `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymou...` | `aten::_foreach_copy_` | TensorList[[24], [4], [24], [24], [4], [4]]; TensorList[[24], [4], [24], [24], [4], [4]] | [6, 1, 1] | 1 | 4.1 | 0.3% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::compare_scalar_kernel<int...` | `aten::le` | int[4] | [1, 1, 1] | 1 | 4.1 | 0.3% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::wh...` | `aten::where` | bool[4, 1]; long int[4, 1]; long int[4, 1] | [1, 1, 1] | 1 | 4.1 | 0.3% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int...` | `aten::sub` | int[4] | [1, 1, 1] | 1 | 4.1 | 0.3% |
| `compute_position_kernel` | `-` |  | [4, 1, 1] | 1 | 4.0 | 0.3% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<long>, std::a...` | `-` |  | None | 1 | 4.0 | 0.3% |
| `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_...` | `aten::arange` | long int[0] | [1, 1, 1] | 1 | 4.0 | 0.3% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<lon...` | `aten::sub` | long int[4] | [1, 1, 1] | 1 | 3.9 | 0.3% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_ke...` | `-` |  | None | 1 | 3.8 | 0.3% |

## <none>

| kernel | launching op | input dims | grid | calls/iter | us/iter | % phase |
|---|---|---|---|---|---|---|
| `_fused_dsa_target_verify_metadata_kernel` | `-` |  | [196633, 1, 1] | 1 | 81.0 | 35.4% |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOp...` | `aten::argmax` | float[24, 154880] | [24, 1, 1] | 1 | 46.5 | 20.3% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::ar...` | `aten::fill_` | int[4] | [1, 1, 1] | 3 | 12.9 | 5.6% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::ar...` | `aten::fill_` | int[4, 6] | [1, 1, 1] | 2 | 8.3 | 3.6% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | int[4]; long int[4] | [1, 1, 1] | 1.2 | 5.0 | 2.2% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::add` | long int[4]; int[4] | [1, 1, 1] | 1 | 4.8 | 2.1% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | int[4, 3]; long int[4, 3] | [1, 1, 1] | 1 | 4.3 | 1.9% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::index` | int[24] | [1, 1, 1] | 1 | 4.3 | 1.9% |
| `assign_extend_cache_locs_uniform` | `-` |  | [4, 1, 1] | 1 | 4.3 | 1.9% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int...` | `aten::add` | int[4] | [1, 1, 1] | 1 | 4.3 | 1.9% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::_index_put_impl_` | int[24]; int[4] | [1, 1, 1] | 1 | 4.2 | 1.8% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::add` | int[4, 1]; long int[4] | [1, 1, 1] | 0.8 | 4.2 | 1.8% |
| `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_...` | `aten::arange` | long int[0] | [1, 1, 1] | 1 | 4.2 | 1.8% |
| `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymou...` | `aten::_foreach_copy_` | TensorList[[24], [24], [24], [4], [4]]; TensorList[[24], [24], [24], [4], [4]] | [5, 1, 1] | 1 | 4.2 | 1.8% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | long int[4, 6]; int[4, 6] | [1, 1, 1] | 1 | 4.1 | 1.8% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::ar...` | `aten::fill_` | int[24] | [1, 1, 1] | 1 | 4.0 | 1.8% |
| `void VerifyTreeGreedy<int, long>(int*, int*, int*, long*, long*, long*, long*, long*, u...` | `sgl_kernel::verify_tree_greedy` | int[24]; int[4, 6]; int[4]; long int[4, 6]; long int[4, 6]; long int[4, 6]; long int[4, 6]; long int[4, 6] | [4, 1, 1] | 1 | 4.0 | 1.7% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | int[4, 4]; long int[4, 4] | [1, 1, 1] | 0.8 | 4.0 | 1.7% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | long int[4]; int[4] | [1, 1, 1] | 1 | 3.9 | 1.7% |
| `fill_bonus_tokens` | `-` |  | [4, 1, 1] | 1 | 3.9 | 1.7% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | long int[4, 3]; int[4, 3] | [1, 1, 1] | 0.8 | 3.5 | 1.5% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::_index_put_impl_` | int[24]; int[4, 3] | [1, 1, 1] | 0.8 | 3.2 | 1.4% |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | `record_param_comms` | TensorList[[7]] | [1, 1, 1] | 0.2 | 2.4 | 1.0% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::add` | int[4, 1]; long int[3] | [1, 1, 1] | 0.2 | 1.0 | 0.5% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | long int[4, 2]; int[4, 2] | [1, 1, 1] | 0.2 | 0.9 | 0.4% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::copy_` | int[4, 2]; long int[4, 2] | [1, 1, 1] | 0.2 | 0.8 | 0.4% |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::nati...` | `aten::_index_put_impl_` | int[24]; int[4, 2] | [1, 1, 1] | 0.2 | 0.8 | 0.3% |

## draft glue (draft, outside steps)

| kernel | launching op | input dims | grid | calls/iter | us/iter | % phase |
|---|---|---|---|---|---|---|
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<a...` | `aten::cat` | TensorList[[4, 1], [4, 5]] | [1, 1, 1] | 1 | 4.5 | 12.4% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<long>, std::a...` | `aten::fill_` | long int[3, 4, 6] | [1, 1, 1] | 1 | 4.2 | 11.4% |
| `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_n...` | `aten::cat` | TensorList[[4, 1], [4, 5]] | [1, 1, 1] | 1 | 4.2 | 11.4% |
| `void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::...` | `aten::fill_` | bool[2641752] | [645, 1, 1] | 0.2 | 3.0 | 8.1% |
| `void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::...` | `aten::fill_` | bool[2641560] | [645, 1, 1] | 0.2 | 2.9 | 8.0% |
| `void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::...` | `aten::fill_` | bool[2641824] | [645, 1, 1] | 0.2 | 2.9 | 8.0% |
| `void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::...` | `aten::fill_` | bool[2641920] | [645, 1, 1] | 0.2 | 2.9 | 8.0% |
| `void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::...` | `aten::fill_` | bool[2641656] | [645, 1, 1] | 0.2 | 2.9 | 7.8% |
| `build_tree_efficient(long*, long*, long*, bool*, long*, long*, long*, long*, int, int, ...` | `sgl_kernel::build_tree_kernel_efficient` | long int[4, 5]; long int[4, 5]; long int[4]; bool[2641560]; long int[24]; long int[4, 6]; long int[4, 6]; long int[4, 6] | [4, 1, 1] | 0.2 | 1.9 | 5.1% |
| `build_tree_efficient(long*, long*, long*, bool*, long*, long*, long*, long*, int, int, ...` | `sgl_kernel::build_tree_kernel_efficient` | long int[4, 5]; long int[4, 5]; long int[4]; bool[2641656]; long int[24]; long int[4, 6]; long int[4, 6]; long int[4, 6] | [4, 1, 1] | 0.2 | 1.8 | 5.0% |
| `build_tree_efficient(long*, long*, long*, bool*, long*, long*, long*, long*, int, int, ...` | `sgl_kernel::build_tree_kernel_efficient` | long int[4, 5]; long int[4, 5]; long int[4]; bool[2641920]; long int[24]; long int[4, 6]; long int[4, 6]; long int[4, 6] | [4, 1, 1] | 0.2 | 1.8 | 5.0% |
| `build_tree_efficient(long*, long*, long*, bool*, long*, long*, long*, long*, int, int, ...` | `sgl_kernel::build_tree_kernel_efficient` | long int[4, 5]; long int[4, 5]; long int[4]; bool[2641824]; long int[24]; long int[4, 6]; long int[4, 6]; long int[4, 6] | [4, 1, 1] | 0.2 | 1.8 | 4.9% |
| `build_tree_efficient(long*, long*, long*, bool*, long*, long*, long*, long*, int, int, ...` | `sgl_kernel::build_tree_kernel_efficient` | long int[4, 5]; long int[4, 5]; long int[4]; bool[2641752]; long int[24]; long int[4, 6]; long int[4, 6]; long int[4, 6] | [4, 1, 1] | 0.2 | 1.8 | 4.9% |
