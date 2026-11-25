# DeepSeek-V41-Flash MTP decode profile

2026-11-25 14:43 UTC

## Summary

| metric       | value        | note                                                                                                                                                                                                                                                                                           |
|--------------|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| forward time | 19.271 ms    | every model forward of one DSPARK step (draft proposal forward(s) and the target verify), each timed in place under its own graph and summed. Not the whole step: sampling, accept/reject and bookkeeping between them are only visible under --full-step (run automatically after this pass). |
| roll-up      | 19.125 ms    | sum of the per-op leaf rows below + the draft model row                                                                                                                                                                                                                                        |
| ratio        | 0.992        | roll-up / forward time; agreement does not prove coverage or numerical correctness; the draft is one row in both                                                                                                                                                                               |
| throughput   | 2914.2 tok/s | at CONC 16, accept len 3.51 from SGLANG_SIMULATE_ACC_LEN -- an assumption the server is also run under, not something measured here                                                                                                                                                            |
| ITL          | 5.490 ms     | forward time / accept len, per request -- a floor, the non-forward part is not counted                                                                                                                                                                                                         |

## Top kernels -- one step

| rank | API                                                                                                             | ms    | Kineto ms | share | calls | collective | target | hottest kernel                                                                                                                                                                                                                                                                                                                                                                      |
|------|-----------------------------------------------------------------------------------------------------------------|-------|-----------|-------|-------|------------|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | `aiter::fused_moe_ → aiter.ops.flydsl.kernels.tensor_shim._run_compiled:305`                                    | 8.907 | 8.145     | 46.6% | 80    |            | yes    | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33`                                                                                                                                                                                                                                                                                                    |
| 2    | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114`                     | 2.306 | 2.419     | 12.1% | 290   |            | yes    | `_gemm_afp8wfp8_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_FUSED_SPLITS_5_N_FIRST_0_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_CG_NUM_KSPLIT_5_SPLITK_BLOCK_SIZE_1024`                                                                                     |
| 3    | `sglang.srt.distributed.device_communicators.pynccl_wrapper.ncclGroupEnd:677`                                   | 1.539 | 13.806    | 8.0%  | 40    | yes        |        | `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)`                                                                                                                                                                                                                                                                                                                         |
| 4    | `sglang::reg_all_gather_into_tensor → aiter.dist.device_communicators.custom_all_reduce.all_gather_unreg:1871`  | 1.034 | 9.272     | 5.4%  | 80    | yes        |        | `void aiter::allgather_vec<std::bfloat16_t, 4>(aiter::RankData*, aiter::RankSignals, aiter::Signal*, std::bfloat16_t*, int, int)`                                                                                                                                                                                                                                                   |
| 5    | `aiter.ops.triton.attention.pa_decode_sparse._pa_decode_sparse_gfx950_gluon:536`                                | 0.672 | 0.683     | 3.5%  | 40    |            | yes    | `_sparse_mla_BLOCK_M_32_BLOCK_K_64_HEAD_SIZE_512_SPLIT_K_1_MAIN_FMT_FP8_DSV4_MLA_ROPE_SEPARATE_0`                                                                                                                                                                                                                                                                                   |
| 6    | `sglang.kernels.ops.layernorm.mhc_boundary_hip._hc_boundary_partials:622`                                       | 0.458 | 0.714     | 2.4%  | 80    |            | yes    | `_hc_boundary_partial_kernel`                                                                                                                                                                                                                                                                                                                                                       |
| 7    | `sglang.kernels.ops.layernorm.mhc_boundary_hip.rmsnorm_with_sinkhorn:388`                                       | 0.370 | 0.576     | 1.9%  | 80    |            | yes    | `_rmsnorm_sinkhorn_kernel`                                                                                                                                                                                                                                                                                                                                                          |
| 8    | `aiter.fused_moe._moe_sorting_impl:566 → aiter::moe_sorting_opus_fwd`                                           | 0.357 | 0.327     | 1.9%  | 80    |            | yes    | `void aiter::opus_moe_sorting_entry<aiter::MoeSortingMultiPhaseKernel_P23<aiter::MoeSortingProblemMp<int, float, int, 1, true, false, true> >, aiter::MoeSortingMultiPhaseKernel_P23<aiter::MoeSortingProblemMp<int, float, int, 1, true, false, true> >::Kargs>(aiter::MoeSortingMultiPhaseKernel_P23<aiter::MoeSortingProblemMp<int, float, int, 1, true, false, true> >::Kargs)` |
| 9    | `aiter.ops.opus.gemm_op_a8w8._launch_bmm_a8w8_mxscale_opus:689 → aiter::_opus_gemm_a8w8_mxscale_bmm_launch_raw` | 0.342 | 0.348     | 1.8%  | 40    |            | yes    | `_Z38gemm_a8w8_mxscale_flatmm_splitk_kernelI62opus_gemm_a8w8_mxscale_flatmm_splitk_bpreshuffle_traits_gfx950ILi256EN4opus3seqIJLi32ELi32ELi256EEEENS1_5tupleIJDB8_S5_ffhEEENS2_IJLi16ELi16ELi4EEEENS2_IJLi1ELi32ELi32EEEELi2ELb0ELb0ELb0EEDF16bLb0ELb0ELb1ELb0ELb0EEv35opus_gemm_scale_splitk_kargs_gfx950`                                                                         |
| 10   | `aiter::gemm_a16w16 → aiter.ops.flydsl.kernels.kernels_common.run_cached:206`                                   | 0.234 | 0.219     | 1.2%  | 40    |            |        | `hgemm_bf16_t32x32x64x9_kst4_w2x2x1_gm0_bias0_lnt_pft_0`                                                                                                                                                                                                                                                                                                                            |

Kineto ms of each timed leaf row scaled to its graph time (factors 0.64..1.33, 1 collective-heavy row(s) clamped); collectives take the row's residual. Targets are the top 5 non-collective groups plus ties.

## Config

| item        | value                                                                                                                                                                                                                                                                 |
|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model       | `DeepseekV4ForCausalLM`  /models/DeepSeek-V4.1-Flash                                                                                                                                                                                                                  |
| layers      | 40 layers -- moe+c0 1 (0), moe+c0+engram 1 (1), moe+c2+kvsrc+indexer 2 (2,8), moe+c2 15 (3-7,9-13,15-19), moe+c2+kvsrc+indexer+engram 1 (14), moe+c1+kvsrc+indexer+late 1 (20), moe+c1+late 15 (21-23,25-27,29-31,33-35,37-39), moe+c1+indexer+late 4 (24,28,32,36)   |
| parallel    | TP 4, attn-DP 4, attn-TP 1, MoE-EP 4                                                                                                                                                                                                                                  |
| batch       | CONC 16, 4 requests per rank                                                                                                                                                                                                                                          |
| context     | ISL 1024 per request                                                                                                                                                                                                                                                  |
| KV setup    | synthetic: native full allocation of the ISL context, random KV bytes (no prefill forward), target + draft pools                                                                                                                                                      |
| routing     | seeded balanced (seed 1234) on every MoE layer, hash layers included                                                                                                                                                                                                  |
| kv dtype    | `fp8_e4m3`                                                                                                                                                                                                                                                            |
| spec        | DSPARK, block 5, 6 verify tokens; simulated acceptance 3.51 (match-expected). One step is the draft proposal, the target verify and the accept/commit -- the per-op rows below are the verify's share, measured on the target stack only; the draft model is one row. |
| weights     | `auto` load, fp8 block 32x32 quantized, experts fp4, bfloat16 elsewhere                                                                                                                                                                                               |
| measurement | 5 graph replays per op; 20 calls per graph, more for a weight small enough to rotate over the 512 MB buffer, max over ranks (EP)                                                                                                                                      |

## Versions

| component | version                                         |
|-----------|-------------------------------------------------|
| bench     | no git files-sha256 f4cae327cf4d                |
| sglang    | 0.5.23.dev20261111+g6be0d75ae0 6be0d75ae0+dirty |
| aiter     | 285579b62+dirty                                 |
| torch     | 2.11.0+rocm10.0.0                               |
| rocm      | 7.15.26333                                      |
| triton    | 3.8.0                                           |
| flydsl    | 0.4.0                                           |
| tilelang  | 0.1.7.post3+cuda.gita55a8230                    |
| aklr      | absent                                          |

## Memory, per rank

| component                              | buffers | B/tok/buffer | B/tok | GB     |
|----------------------------------------|---------|--------------|-------|--------|
| swa_kv_pool                            | 40      | 104          | 4153  | 105.18 |
| kv_pools[1]                            | 1       | 585          | 585   | 14.82  |
| kv_pools[2]                            | 3       | 293          | 878   | 22.22  |
| index_pools[1]                         | 2       | 34           | 68    | 1.72   |
| index_pools[2]                         | 6       | 17           | 102   | 2.58   |
| **kv total**                           |         |              | 5786  | 146.52 |
| weights (target + draft)               |         |              |       | 85.71  |
| **HBM**                                |         |              |       | 287.98 |
| host-resident params/buffers (not HBM) |         |              |       | 188.83 |

- **capacity 108,771,328 full-KV tokens cluster-wide** = 27,192,832 per rank x 4 DP ranks = 106,222 requests at ISL 1024
- native pools: full KV 27,192,832, SWA 4,826,112, request slots 64; the B/tok columns divide every buffer by the full-KV token capacity (SWA holds its own, smaller count)
- `--mem-fraction-static 0.85` caps weights + kv at 245 GB of the 288 GB HBM; this run lands at 232 GB
- this run uses 4,096 context tokens, 0.0% of this rank's full-KV pool
- EP 4: this rank holds 96 routed experts at full intermediate width

## Hotspots -- one step, all layers

| op / kernel | ms/step | share | kernel calls/step | profile GPU ms/step | % of module GPU | Python / dispatcher API |
| --- | --- | --- | --- | --- | --- | --- |
| `mlp.experts` | 9.492 | 49.6% |  |  |  |  |
|   ↳ `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` |  |  | 40 | 5.426 | 62.52% | `aiter::fused_moe_ → aiter.ops.flydsl.kernels.tensor_shim._run_compiled:305` |
|   ↳ `gemm2_a4w4_port_hmax8192_imax2304_bm32_nt_atomic_a8_bhoist_apf_spart4x2_bf16lds_v2_biasabi7` |  |  | 40 | 2.719 | 31.32% | `aiter::fused_moe_ → aiter.ops.flydsl.kernels.tensor_shim._run_compiled:305` |
|   ↳ `_ZN5aiter30fused_mx_quant_moe_sort_kernelIDF16bDB8_Li256ELi32EEEvPT0_PhPKT_PKiS9_PKfiiiiiiiii` |  |  | 40 | 0.208 | 2.40% | `aiter.ops.quant.fused_dynamic_mx_quant_moe_sort:1088 → aiter::fused_dynamic_mx_quant_moe_sort_hip_bounded` |
|   ↳ `void aiter::opus_moe_sorting_entry<aiter::MoeSortingMultiPhaseKernel_P23<aiter::MoeSortingProblemMp<int, float, int, 1, true, false, true> >, aiter::MoeSortingMultiPhaseKernel_P23<aiter::MoeSortingProblemMp<int, float, int, 1, true, false, true> >::Kargs>(aiter::MoeSortingMultiPhaseKernel_P23<aiter::MoeSortingProblemMp<int, float, int, 1, true, false, true> >::Kargs)` |  |  | 40 | 0.166 | 1.92% | `aiter.fused_moe._moe_sorting_impl:566 → aiter::moe_sorting_opus_fwd` |
|   ↳ `void aiter::opus_moe_sorting_entry<aiter::MoeSortingMultiPhaseKernel_P0_v2<aiter::MoeSortingProblemMp<int, float, int, 1, true, false, true> >, aiter::MoeSortingMultiPhaseKernel_P0_v2<aiter::MoeSortingProblemMp<int, float, int, 1, true, false, true> >::Kargs>(aiter::MoeSortingMultiPhaseKernel_P0_v2<aiter::MoeSortingProblemMp<int, float, int, 1, true, false, true> >::Kargs)` |  |  | 40 | 0.160 | 1.85% | `aiter.fused_moe._moe_sorting_impl:566 → aiter::moe_sorting_opus_fwd` |
| `layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 4.282 | 22.4% |  |  |  |  |
|   ↳ `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` |  |  | 40 | 13.806 | 53.64% | `sglang.srt.distributed.device_communicators.pynccl_wrapper.ncclGroupEnd:677` |
|   ↳ `void aiter::allgather_vec<std::bfloat16_t, 4>(aiter::RankData*, aiter::RankSignals, aiter::Signal*, std::bfloat16_t*, int, int)` |  |  | 40 | 9.120 | 35.44% | `sglang::reg_all_gather_into_tensor → aiter.dist.device_communicators.custom_all_reduce.all_gather_unreg:1871` |
|   ↳ `_hc_boundary_partial_kernel` |  |  | 80 | 0.714 | 2.77% | `sglang.kernels.ops.layernorm.mhc_boundary_hip._hc_boundary_partials:622` |
|   ↳ `_rmsnorm_sinkhorn_kernel` |  |  | 80 | 0.576 | 2.24% | `sglang.kernels.ops.layernorm.mhc_boundary_hip.rmsnorm_with_sinkhorn:388` |
|   ↳ `_gemm_afp8wfp8_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_FUSED_SPLITS_5_N_FIRST_0_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_CG_NUM_KSPLIT_5_SPLITK_BLOCK_SIZE_1024` |  |  | 40 | 0.387 | 1.50% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
|   ↳ `_gemm_afp8wfp8_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_64_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_FUSED_SPLITS_3_N_FIRST_0_num_warps_4_num_stages_3_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_3_SPLITK_BLOCK_SIZE_768` |  |  | 40 | 0.289 | 1.12% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
|   ↳ `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::(anonymous namespace)::CompareFunctor<long> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<long> const&)::{lambda(int, bool)#2}>(int, at::native::gpu_kernel_impl<at::native::(anonymous namespace)::CompareFunctor<long> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<long> const&)::{lambda(int, bool)#2})` |  |  | 40 | 0.174 | 0.67% | `sglang.srt.models.deepseek_common.moe_transport._mask_dp_pad_rows:66 → aten::ge` |
|   ↳ `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16, bool)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16, bool)#1} const&)::{lambda(int, bool)#1})` |  |  | 40 | 0.171 | 0.67% | `sglang.srt.models.deepseek_common.moe_transport._mask_dp_pad_rows:66 → aten::masked_fill_` |
|   ↳ `Memcpy DtoD (Device -> Device)` |  |  | 40 | 0.152 | 0.59% | `sglang::reg_all_gather_into_tensor → aiter.dist.device_communicators.custom_all_reduce.all_gather_unreg:1871` |
|   ↳ `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>)` |  |  | 40 | 0.146 | 0.57% | `sglang.srt.models.deepseek_common.moe_transport.return_output:219 → aten::add` |
|   ↳ `_silu_and_mul_clamp_kernel` |  |  | 40 | 0.109 | 0.42% | `sglang.kernels.ops.activation.silu_and_mul_clamp_hip.silu_and_mul_clamp_triton:79` |
|   ↳ `void (anonymous namespace)::elementwise_kernel_with_index_grid_stride<int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, function_traits<at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>::result_type*)` |  |  | 40 | 0.087 | 0.34% | `sglang.srt.models.deepseek_common.moe_transport._mask_dp_pad_rows:66 → aten::arange` |
|   ↳ `void aiter::mhc_post_kernel<std::bfloat16_t, 4, 4, 1024, false>(std::bfloat16_t*, std::bfloat16_t*, std::bfloat16_t*, float*, float*, int, int, int, int, int)` |  |  | 1 | 0.004 | 0.02% | `aiter.ops.mhc.mhc_post:781 → aiter::_mhc_post` |
|   ↳ `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1})` |  |  | 1 | 0.003 | 0.01% | `sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc.hc_boundary:469 → aten::copy_` |
| `attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel` | 1.700 | 8.9% |  |  |  |  |
|   ↳ `_sparse_mla_BLOCK_M_32_BLOCK_K_64_HEAD_SIZE_512_SPLIT_K_1_MAIN_FMT_FP8_DSV4_MLA_ROPE_SEPARATE_0` |  |  | 38 | 0.663 | 38.38% | `aiter.ops.triton.attention.pa_decode_sparse._pa_decode_sparse_gfx950_gluon:536` |
|   ↳ `_Z38gemm_a8w8_mxscale_flatmm_splitk_kernelI62opus_gemm_a8w8_mxscale_flatmm_splitk_bpreshuffle_traits_gfx950ILi256EN4opus3seqIJLi32ELi32ELi256EEEENS1_5tupleIJDB8_S5_ffhEEENS2_IJLi16ELi16ELi4EEEENS2_IJLi1ELi32ELi32EEEELi2ELb0ELb0ELb0EEDF16bLb0ELb0ELb1ELb0ELb0EEv35opus_gemm_scale_splitk_kargs_gfx950` |  |  | 40 | 0.348 | 20.13% | `aiter.ops.opus.gemm_op_a8w8._launch_bmm_a8w8_mxscale_opus:689 → aiter::_opus_gemm_a8w8_mxscale_bmm_launch_raw` |
|   ↳ `void sglang::fused_k_norm_rope_flashmla<__hip_bfloat16, 512l, 64l, long, 8, (sglang::deepseek_v4::KVLayout)0, false, true>(sglang::FusedKNormRopeFlashMLAParams)` |  |  | 40 | 0.236 | 13.65% | `sglang.kernels.ops.attention.dsv4.norm_rope.fused_k_norm_rope_flashmla:304` |
|   ↳ `_gfx_sparse_decode_reduce_kernel` |  |  | 38 | 0.184 | 10.65% | `sglang.kernels.ops.attention.sparse_decode_reduce_hip.gfx_sparse_split_reduce:78` |
|   ↳ `_rmsnorm_fake_quant_fp8_kernel` |  |  | 40 | 0.139 | 8.07% | `sglang.kernels.ops.layernorm.rmsnorm_fake_quant_amd_gfx95.rmsnorm_fake_quant_fp8:160` |
|   ↳ `_ZN5aiter31inverse_rope_group_quant_kernelIDF16bDF16bLi512ELi64ELi32ELi8ELi1ELNS_11ScaleLayoutE0ELNS_9TierOrderE0ELb0ELb0EEEvPKT_PDB8_PhPKlPKT0_SD_iiiiiillliiibbii` |  |  | 40 | 0.119 | 6.86% | `aiter.ops.inverse_rope_group_quant.inverse_rope_group_quant:91 → aiter::_inverse_rope_group_quant_kernel` |
|   ↳ `_sparse_mla_BLOCK_M_16_BLOCK_K_64_HEAD_SIZE_512_SPLIT_K_0_MAIN_FMT_FP8_DSV4_MLA_ROPE_SEPARATE_0` |  |  | 2 | 0.020 | 1.16% | `aiter.ops.triton.attention.pa_decode_sparse._pa_decode_sparse_gfx950_gluon:536` |
|   ↳ `_mask_indices_by_length_kernel` |  |  | 8 | 0.019 | 1.09% | `sglang.kernels.ops.attention.dsv4.attn_glue_hip.mask_indices_by_length:57` |
| `draft model` | 1.094 | 5.7% |  |  |  |  |
|   ↳ kernel profile unavailable |  |  |  |  |  |  |
| `mlp.shared_experts` | 0.695 | 3.6% |  |  |  |  |
|   ↳ `_gemm_afp8wfp8_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_FUSED_SPLITS_5_N_FIRST_0_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_CG_NUM_KSPLIT_5_SPLITK_BLOCK_SIZE_1024` |  |  | 40 | 0.356 | 52.45% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
|   ↳ `_gemm_afp8wfp8_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_64_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_FUSED_SPLITS_3_N_FIRST_0_num_warps_4_num_stages_3_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_3_SPLITK_BLOCK_SIZE_768` |  |  | 40 | 0.218 | 32.17% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
|   ↳ `_silu_and_mul_clamp_kernel` |  |  | 40 | 0.104 | 15.38% | `sglang.kernels.ops.activation.silu_and_mul_clamp_hip.silu_and_mul_clamp_triton:79` |
| `attn.wo_b` | 0.584 | 3.1% |  |  |  |  |
|   ↳ `_gemm_afp8wfp8_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_BLOCK_SIZE_K_512_GROUP_SIZE_M_1_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_FUSED_SPLITS_3_N_FIRST_0_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_CG_NUM_KSPLIT_3_SPLITK_BLOCK_SIZE_3072` |  |  | 40 | 0.464 | 82.32% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
|   ↳ `_mxfp8_quant_kernel` |  |  | 40 | 0.100 | 17.68% | `sglang.kernels.ops.quantization.mxfp8_amd_gfx95._mxfp8_e4m3_quantize_triton:63` |
| `attn.wq_b` | 0.457 | 2.4% |  |  |  |  |
|   ↳ `_gemm_afp8wfp8_packed_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_64_BLOCK_SIZE_K_256_K_PACK_1_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_SPLITK_BLOCK_SIZE_1280_FUSED_SPLITS_1_B_CACHE_MODIFIER_NONE_N_32768_K_1280_LAUNCH_OPTIONS_NUM_WARPS_4_NUM_STAGES_3_WAVES_PER_EU_0_MATRIX_INSTR_NONKDIM_16` |  |  | 40 | 0.343 | 100.00% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
| `attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.250 | 1.3% |  |  |  |  |
|   ↳ `pa_mqa_logits_fp4_rowgroup_kernel_0` |  |  | 8 | 0.052 | 17.79% | `aiter.ops.flydsl.kernels.tensor_shim._run_compiled:305` |
|   ↳ `void sglang::topk_mode_1::topk_main_kernel<false, 0, (sglang::topk_mode_1::TopKMode)2>(sglang::topk_mode_1::TopKPagedParams)` |  |  | 8 | 0.050 | 17.07% | `sglang.kernels.ops.attention.dsv4.topk.topk_transform_paged_v2:325` |
|   ↳ `_index_q_pack_weights_kernel` |  |  | 8 | 0.039 | 13.49% | `sglang.kernels.ops.attention.dsv4.fp4_indexer_hip.index_q_rope_pack_weights_flydsl:1119` |
|   ↳ `_gemm_afp8wfp8_packed_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_16_BLOCK_SIZE_K_1024_K_PACK_1_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_SPLITK_BLOCK_SIZE_2048_FUSED_SPLITS_1_B_CACHE_MODIFIER_NONE_N_4096_K_1280_LAUNCH_OPTIONS_NUM_WARPS_2_NUM_STAGES_3_WAVES_PER_EU_2_MATRIX_INSTR_NONKDIM_16` |  |  | 8 | 0.036 | 12.27% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
|   ↳ `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB2048_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM3_NTn1_NTA4_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK5_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGRO0_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4_WGMXCCn1` |  |  | 3 | 0.030 | 10.25% | `sglang.kernels.ops.gemm.bf16_fp32._linear_bf16_fp32_cublas:108 → aten::mm` |
|   ↳ `_router_gemv_split_k_kernel` |  |  | 8 | 0.027 | 9.09% | `sglang.kernels.ops.gemm.router_gemv_hip.rocm_router_gemv_split_k:77` |
|   ↳ `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x256_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA1_DTLB1_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB1_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA32_LPB32_LPMXSA0_LPMXSB0_LPM0_LRVWn1_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS12_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR4_PLR1_PKA1_SGROB1_SIA3_SS0_SPO1_SRVW0_SSO1_SVW4_SK5_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN1_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS2_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGRO0_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_1_WGMXCCn1` |  |  | 4 | 0.018 | 6.14% | `torch.nn.modules.linear.forward:130 → aten::mm` |
|   ↳ `void sglang::flash_index_k_split_kernel<false, 128l, 64l, 64u, 2u, false, long>(sglang::IndexKSplitParams)` |  |  | 3 | 0.012 | 4.20% | `sglang.kernels.ops.attention.dsv4.fp4_indexer_rope_hip.index_k_norm_rope_pack_store_split:37` |
|   ↳ `void sglang::flash_c2_decode_kernel<(sglang::C2Mode)1, 512l, 64l, 7, long, long, (sglang::deepseek_v4::KVLayout)0, false>(sglang::Compress2DecodeParams)` |  |  | 3 | 0.011 | 3.81% | `sglang.kernels.ops.attention.dsv4.low_ratio_compress.c2_decode_norm_rope_store:135` |
|   ↳ `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB2048_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM3_NTn1_NTA4_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK5_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGRO0_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4_WGMXCCn1` |  |  | 1 | 0.009 | 3.22% | `torch.nn.modules.linear.forward:130 → aten::mm` |
|   ↳ `void sglang::flash_index_k_split_kernel<false, 128l, 64l, 64u, 1u, true, long>(sglang::IndexKSplitParams)` |  |  | 1 | 0.004 | 1.44% | `sglang.kernels.ops.attention.dsv4.fp4_indexer_rope_hip.index_k_norm_rope_pack_store_split:37` |
|   ↳ `void sglang::flash_c1_decode_kernel<512l, 64l, 8, long, long, (sglang::deepseek_v4::KVLayout)0, false>(sglang::Compress1DecodeParams)` |  |  | 1 | 0.004 | 1.23% | `sglang.kernels.ops.attention.dsv4.low_ratio_compress.c1_decode_norm_rope_store:52` |
| `attn.wqkv_a` | 0.249 | 1.3% |  |  |  |  |
|   ↳ `_gemm_afp8wfp8_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_64_BLOCK_SIZE_K_512_GROUP_SIZE_M_1_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_FUSED_SPLITS_5_N_FIRST_0_num_warps_8_num_stages_3_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_CG_NUM_KSPLIT_5_SPLITK_BLOCK_SIZE_1024` |  |  | 40 | 0.260 | 100.00% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
| `mlp.gate` | 0.234 | 1.2% |  |  |  |  |
|   ↳ `hgemm_bf16_t32x32x64x9_kst4_w2x2x1_gm0_bias0_lnt_pft_0` |  |  | 40 | 0.219 | 100.00% | `aiter::gemm_a16w16 → aiter.ops.flydsl.kernels.kernels_common.run_cached:206` |
| `logits` | 0.211 | 1.1% |  |  |  |  |
|   ↳ `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x32x128_MI16x16x1_SN_LDSB1_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB256_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT8_1_MXLIBL_MXSFNS_MO40_MGRIPM2_NTn1_NTA4_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW8_SK5_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGRO0_USI0_VSn1_VWA8_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1_WGMXCCn1` |  |  | 1 | 0.207 | 97.64% | `sglang.srt.layers.logits_processor._compute_lm_head:1373 → aten::mm` |
|   ↳ `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>)` |  |  | 1 | 0.005 | 2.36% | `sglang.srt.layers.logits_processor._copy_logits_to_buffer:1569 → aten::copy_` |
| `mlp.topk` | 0.209 | 1.1% |  |  |  |  |
|   ↳ `void aiter::topk_gating_kernel_opt<hip_bfloat16, hip_bfloat16, 384, true, 0>(hip_bfloat16 const*, hip_bfloat16 const*, float*, int*, unsigned long, int, int, float, int, int)` |  |  | 40 | 0.206 | 100.00% | `aiter.ops.topk.topk_gating:156 → aiter::topk_gating_fwd` |
| `engram :: Engram n-gram embedding gate (runs before the layer)` | 0.094 | 0.5% |  |  |  |  |
|   ↳ `_gemm_afp8wfp8_kernel_BLOCK_SIZE_M_64_BLOCK_SIZE_N_64_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_FUSED_SPLITS_1_N_FIRST_1_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_32_cache_modifier_CG_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_6144` |  |  | 2 | 0.066 | 71.55% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
|   ↳ `_engram_gather_kernel` |  |  | 2 | 0.013 | 14.20% | `sglang.kernels.ops.embeddings.engram_gather.engram_gather:129` |
|   ↳ `_engram_gate_kernel` |  |  | 2 | 0.008 | 8.61% | `sglang.kernels.ops.embeddings.engram_gate.fused_engram_gate:49` |
|   ↳ `_mxfp8_quant_kernel` |  |  | 2 | 0.005 | 5.64% | `sglang.kernels.ops.quantization.mxfp8_amd_gfx95._mxfp8_e4m3_quantize_triton:63` |
| `engram_hasher` | 0.005 | 0.0% |  |  |  |  |
|   ↳ `_engram_hash_kernel` |  |  | 1 | 0.005 | 100.00% | `sglang.kernels.ops.embeddings.engram_hash._launch_hash_kernel:223` |
| `norm` | 0.003 | 0.0% |  |  |  |  |
|   ↳ `void aiter::add_rmsnorm_quant_kernel<std::bfloat16_t, std::bfloat16_t, 256, 24, false, false, true, 1, false>(std::bfloat16_t*, std::bfloat16_t*, float*, std::bfloat16_t*, std::bfloat16_t*, std::bfloat16_t*, double, bool, int, int, int, int, int, int, int, bool, bool)` |  |  | 1 | 0.005 | 100.00% | `aiter.ops.rmsnorm._rms_norm_fwd_dispatch:401 → aiter::rmsnorm` |
| `hc_head :: final mHC combine` | 0.003 | 0.0% |  |  |  |  |
|   ↳ `_mhc_combine_kernel` |  |  | 1 | 0.002 | 100.00% | `sglang.kernels.ops.layernorm.mhc.mhc_combine:2800` |
| `embed_tokens` | 0.002 | 0.0% |  |  |  |  |
|   ↳ `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool)` |  |  | 1 | 0.003 | 100.00% | `torch.nn.functional.embedding:2455 → aten::gather` |
| `mlp.<rest> :: shared+routed add, post-expert collective` | -0.436 | -2.3% |  |  |  |  |
|   ↳ no exclusive GPU kernels (all in child rows) |  |  | 0 | 0.000 | n/a |  |

Leaf rows only, so the shares add to 100%. A `<rest>` row is its parent minus the children listed under it, and the `::` tag names the work that lands there -- inline code and fused kernels with no module to hook. It is a difference of two measurements rather than a measurement, so any error in the parent or in any child lands in it at full size and reversed sign: a `<rest>` small next to its parent is mostly that error, and can read negative. 

Indented rows list **all** Kineto GPU kernels/copies from rank 0's standalone module replays (one eager sample per layer kind, multiplied by its layer count; globals once). Kernel percentages sum to 100% **within each module**. Parent `<rest>` samples exclude separately measured children through module hooks. The API column comes from the kernel's Kineto runtime correlation, nearest recorded Python frame, and dispatcher external ID. ms/step and share are graph timings; the profile columns are diagnostic.

Packages these APIs were loaded from: `aiter` = `/sgl-workspace/aiter/aiter`, `sglang` = `/sgl-workspace/sglang/python/sglang`, `torch` = `/opt/venv/lib/python3.12/site-packages/torch`.

## Per layer kind -- one step

### moe+c0 -- x1 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.414   | 0.414   |
| `  attn`                                                                                 | 0.057   | 0.057   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.006   |
| `    attn.wq_b`                                                                          | 0.011   | 0.011   |
| `    attn.wo_b`                                                                          | 0.015   | 0.015   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.024   | 0.024   |
| `  mlp`                                                                                  | 0.249   | 0.249   |
| `    mlp.gate`                                                                           | 0.006   | 0.006   |
| `    mlp.topk`                                                                           | 0.005   | 0.005   |
| `    mlp.experts`                                                                        | 0.237   | 0.237   |
| `    mlp.shared_experts`                                                                 | 0.017   | 0.017   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.016  | -0.016  |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.108   | 0.108   |

### moe+c0+engram -- x1 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.406   | 0.406   |
| `  attn`                                                                                 | 0.057   | 0.057   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.006   |
| `    attn.wq_b`                                                                          | 0.011   | 0.011   |
| `    attn.wo_b`                                                                          | 0.015   | 0.015   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.025   | 0.025   |
| `  mlp`                                                                                  | 0.248   | 0.248   |
| `    mlp.gate`                                                                           | 0.006   | 0.006   |
| `    mlp.topk`                                                                           | 0.005   | 0.005   |
| `    mlp.experts`                                                                        | 0.237   | 0.237   |
| `    mlp.shared_experts`                                                                 | 0.017   | 0.017   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.016  | -0.016  |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.101   | 0.101   |
| `engram :: Engram n-gram embedding gate (runs before the layer)`                         | 0.047   | 0.047   |

### moe+c2+kvsrc+indexer -- x2 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.492   | 0.984   |
| `  attn`                                                                                 | 0.120   | 0.240   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.013   |
| `    attn.wq_b`                                                                          | 0.012   | 0.023   |
| `    attn.wo_b`                                                                          | 0.014   | 0.028   |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.040   | 0.080   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.048   | 0.096   |
| `  mlp`                                                                                  | 0.248   | 0.496   |
| `    mlp.gate`                                                                           | 0.006   | 0.012   |
| `    mlp.topk`                                                                           | 0.005   | 0.009   |
| `    mlp.experts`                                                                        | 0.236   | 0.473   |
| `    mlp.shared_experts`                                                                 | 0.017   | 0.033   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.015  | -0.031  |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.124   | 0.248   |

### moe+c2 -- x15 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.438   | 6.566   |
| `  attn`                                                                                 | 0.075   | 1.121   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.093   |
| `    attn.wq_b`                                                                          | 0.011   | 0.172   |
| `    attn.wo_b`                                                                          | 0.015   | 0.219   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.042   | 0.637   |
| `  mlp`                                                                                  | 0.258   | 3.864   |
| `    mlp.gate`                                                                           | 0.006   | 0.087   |
| `    mlp.topk`                                                                           | 0.005   | 0.076   |
| `    mlp.experts`                                                                        | 0.237   | 3.550   |
| `    mlp.shared_experts`                                                                 | 0.018   | 0.268   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.008  | -0.118  |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.105   | 1.581   |

### moe+c2+kvsrc+indexer+engram -- x1 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.473   | 0.473   |
| `  attn`                                                                                 | 0.120   | 0.120   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.006   |
| `    attn.wq_b`                                                                          | 0.011   | 0.011   |
| `    attn.wo_b`                                                                          | 0.014   | 0.014   |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.040   | 0.040   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.048   | 0.048   |
| `  mlp`                                                                                  | 0.248   | 0.248   |
| `    mlp.gate`                                                                           | 0.006   | 0.006   |
| `    mlp.topk`                                                                           | 0.005   | 0.005   |
| `    mlp.experts`                                                                        | 0.238   | 0.238   |
| `    mlp.shared_experts`                                                                 | 0.017   | 0.017   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.016  | -0.016  |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.105   | 0.105   |
| `engram :: Engram n-gram embedding gate (runs before the layer)`                         | 0.046   | 0.046   |

### moe+c1+kvsrc+indexer+late -- x1 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.474   | 0.474   |
| `  attn`                                                                                 | 0.121   | 0.121   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.006   |
| `    attn.wq_b`                                                                          | 0.011   | 0.011   |
| `    attn.wo_b`                                                                          | 0.014   | 0.014   |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.040   | 0.040   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.049   | 0.049   |
| `  mlp`                                                                                  | 0.249   | 0.249   |
| `    mlp.gate`                                                                           | 0.006   | 0.006   |
| `    mlp.topk`                                                                           | 0.005   | 0.005   |
| `    mlp.experts`                                                                        | 0.237   | 0.237   |
| `    mlp.shared_experts`                                                                 | 0.017   | 0.017   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.016  | -0.016  |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.104   | 0.104   |

### moe+c1+late -- x15 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.435   | 6.528   |
| `  attn`                                                                                 | 0.074   | 1.113   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.093   |
| `    attn.wq_b`                                                                          | 0.011   | 0.171   |
| `    attn.wo_b`                                                                          | 0.015   | 0.221   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.042   | 0.628   |
| `  mlp`                                                                                  | 0.255   | 3.820   |
| `    mlp.gate`                                                                           | 0.006   | 0.088   |
| `    mlp.topk`                                                                           | 0.006   | 0.086   |
| `    mlp.experts`                                                                        | 0.238   | 3.569   |
| `    mlp.shared_experts`                                                                 | 0.017   | 0.256   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.012  | -0.179  |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.106   | 1.594   |

### moe+c1+indexer+late -- x4 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.467   | 1.869   |
| `  attn`                                                                                 | 0.102   | 0.409   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.025   |
| `    attn.wq_b`                                                                          | 0.011   | 0.045   |
| `    attn.wo_b`                                                                          | 0.014   | 0.057   |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.023   | 0.091   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.048   | 0.192   |
| `  mlp`                                                                                  | 0.255   | 1.018   |
| `    mlp.gate`                                                                           | 0.006   | 0.024   |
| `    mlp.topk`                                                                           | 0.005   | 0.019   |
| `    mlp.experts`                                                                        | 0.238   | 0.951   |
| `    mlp.shared_experts`                                                                 | 0.017   | 0.069   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.011  | -0.044  |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.110   | 0.441   |

## Roll-up -- one step

| op                                 | x  | ms/call | ms/step |
|------------------------------------|----|---------|---------|
| moe+c0 layer                       | 1  | 0.414   | 0.414   |
| moe+c0+engram layer                | 1  | 0.406   | 0.406   |
| moe+c0+engram engram               | 1  | 0.047   | 0.047   |
| moe+c2+kvsrc+indexer layer         | 2  | 0.492   | 0.984   |
| moe+c2 layer                       | 15 | 0.438   | 6.566   |
| moe+c2+kvsrc+indexer+engram layer  | 1  | 0.473   | 0.473   |
| moe+c2+kvsrc+indexer+engram engram | 1  | 0.046   | 0.046   |
| moe+c1+kvsrc+indexer+late layer    | 1  | 0.474   | 0.474   |
| moe+c1+late layer                  | 15 | 0.435   | 6.528   |
| moe+c1+indexer+late layer          | 4  | 0.467   | 1.869   |
| embed_tokens                       | 1  | 0.002   | 0.002   |
| engram_hasher                      | 1  | 0.005   | 0.005   |
| hc_head                            | 1  | 0.003   | 0.003   |
| norm                               | 1  | 0.003   | 0.003   |
| logits                             | 1  | 0.211   | 0.211   |

## Draft model -- per op

| draft op                      | ms/call | share of draft layer |
|-------------------------------|---------|----------------------|
| `s1.mlp`                      | 0.113   |                      |
| `s0.mlp`                      | 0.113   |                      |
| `s2.mlp`                      | 0.113   |                      |
| `s1.mlp.experts`              | 0.107   |                      |
| `s2.mlp.experts`              | 0.106   |                      |
| `s0.mlp.experts`              | 0.106   |                      |
| `s1.attn`                     | 0.095   |                      |
| `s2.attn`                     | 0.095   |                      |
| `s0.attn`                     | 0.094   |                      |
| `s0.main_proj`                | 0.022   |                      |
| `s0.mlp.shared_experts`       | 0.019   |                      |
| `s2.mlp.shared_experts`       | 0.019   |                      |
| `s1.mlp.shared_experts`       | 0.019   |                      |
| `s2.attn.wq_a`                | 0.017   |                      |
| `s0.attn.wq_a`                | 0.017   |                      |
| `s1.attn.wq_a`                | 0.017   |                      |
| `s2.attn.wo_b`                | 0.015   |                      |
| `s0.attn.wo_b`                | 0.015   |                      |
| `s1.attn.wo_b`                | 0.015   |                      |
| `s1.attn.wq_b`                | 0.014   |                      |
| `s2.attn.wq_b`                | 0.014   |                      |
| `s0.attn.wq_b`                | 0.014   |                      |
| `s2.attn.wkv`                 | 0.007   |                      |
| `s1.attn.wkv`                 | 0.007   |                      |
| `s0.attn.wkv`                 | 0.007   |                      |
| `s0.mlp.gate`                 | 0.005   |                      |
| `s1.mlp.gate`                 | 0.005   |                      |
| `s2.mlp.gate`                 | 0.005   |                      |
| `s0.main_norm`                | 0.003   |                      |
| `s2.mlp.topk`                 | 0.003   |                      |
| `s1.post_attention_layernorm` | 0.003   |                      |
| `s1.mlp.topk`                 | 0.003   |                      |
| `s0.mlp.topk`                 | 0.003   |                      |
| `s2.norm`                     | 0.002   |                      |
| `s2.post_attention_layernorm` | 0.002   |                      |
| `s2.input_layernorm`          | 0.002   |                      |
| `s0.input_layernorm`          | 0.002   |                      |
| `s1.input_layernorm`          | 0.002   |                      |
| `s0.post_attention_layernorm` | 0.002   |                      |
| `embed_tokens`                | 0.002   |                      |

The `draft model` row in the roll-up is the draft forward(s) of one step timed in place under their own graphs (1.094 ms), not this table: the draft is not decomposed into the roll-up (the glm52 port keeps it one row too). These per-op rows are one call each, for ranking inside the draft.

## Coverage

| layer kind                  | hook not called          | possible containing row                            |
|-----------------------------|--------------------------|----------------------------------------------------|
| draft                       | markov_head              | unattributed (no parent residual)                  |
| draft                       | s0.attn.wo_a             | unattributed (no parent residual)                  |
| draft                       | s1.attn.wo_a             | unattributed (no parent residual)                  |
| draft                       | s2.attn.wo_a             | unattributed (no parent residual)                  |
| moe+c0+engram               | attn.wo_a                | attn.<rest>                                        |
| moe+c0+engram               | input_layernorm          | layer.<rest>                                       |
| moe+c0+engram               | post_attention_layernorm | layer.<rest>                                       |
| moe+c0                      | attn.wo_a                | attn.<rest>                                        |
| moe+c0                      | input_layernorm          | layer.<rest>                                       |
| moe+c0                      | post_attention_layernorm | layer.<rest>                                       |
| moe+c1+indexer+late         | attn.indexer             | attn.sources (timed through the backend call site) |
| moe+c1+indexer+late         | attn.wo_a                | attn.<rest>                                        |
| moe+c1+indexer+late         | input_layernorm          | layer.<rest>                                       |
| moe+c1+indexer+late         | post_attention_layernorm | layer.<rest>                                       |
| moe+c1+kvsrc+indexer+late   | attn.compressor          | attn.sources (timed through the backend call site) |
| moe+c1+kvsrc+indexer+late   | attn.indexer             | attn.sources (timed through the backend call site) |
| moe+c1+kvsrc+indexer+late   | attn.wo_a                | attn.<rest>                                        |
| moe+c1+kvsrc+indexer+late   | input_layernorm          | layer.<rest>                                       |
| moe+c1+kvsrc+indexer+late   | post_attention_layernorm | layer.<rest>                                       |
| moe+c1+late                 | attn.wo_a                | attn.<rest>                                        |
| moe+c1+late                 | input_layernorm          | layer.<rest>                                       |
| moe+c1+late                 | post_attention_layernorm | layer.<rest>                                       |
| moe+c2+kvsrc+indexer+engram | attn.compressor          | attn.sources (timed through the backend call site) |
| moe+c2+kvsrc+indexer+engram | attn.indexer             | attn.sources (timed through the backend call site) |
| moe+c2+kvsrc+indexer+engram | attn.wo_a                | attn.<rest>                                        |
| moe+c2+kvsrc+indexer+engram | input_layernorm          | layer.<rest>                                       |
| moe+c2+kvsrc+indexer+engram | post_attention_layernorm | layer.<rest>                                       |
| moe+c2+kvsrc+indexer        | attn.compressor          | attn.sources (timed through the backend call site) |
| moe+c2+kvsrc+indexer        | attn.indexer             | attn.sources (timed through the backend call site) |
| moe+c2+kvsrc+indexer        | attn.wo_a                | attn.<rest>                                        |
| moe+c2+kvsrc+indexer        | input_layernorm          | layer.<rest>                                       |
| moe+c2+kvsrc+indexer        | post_attention_layernorm | layer.<rest>                                       |
| moe+c2                      | attn.wo_a                | attn.<rest>                                        |
| moe+c2                      | input_layernorm          | layer.<rest>                                       |
| moe+c2                      | post_attention_layernorm | layer.<rest>                                       |

Hooked but never called during the recording forward. A name is missing either because this layer kind does not run that op, or because a fused kernel reads the module's weight directly (DSv4.1 on gfx950: the mHC boundary kernel applies `input_layernorm` / `post_attention_layernorm` itself). A parent residual can contain this work; missing globals are not covered by layer residuals.

## Not replayable

| layer kind | op         | its ms now sits in    | why it could not be replayed                                                                              |
|------------|------------|-----------------------|-----------------------------------------------------------------------------------------------------------|
| draft      | `s0.layer` | the `draft model` row | RuntimeError: The size of tensor a (24) must match the size of tensor b (20) at non-singleton dimension 0 |
| draft      | `s1.layer` | the `draft model` row | RuntimeError: The size of tensor a (24) must match the size of tensor b (20) at non-singleton dimension 0 |
| draft      | `s2.layer` | the `draft model` row | RuntimeError: The size of tensor a (24) must match the size of tensor b (20) at non-singleton dimension 0 |

These ops ran in the recording forward, so their cost is real and is in the parent row; what is missing is the row that would separate it out. Read the named `<rest>` row as an upper bound, not a measurement.
