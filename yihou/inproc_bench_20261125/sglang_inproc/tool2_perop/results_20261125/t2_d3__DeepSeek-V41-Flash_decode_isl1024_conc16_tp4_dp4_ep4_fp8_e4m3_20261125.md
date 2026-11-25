# DeepSeek-V41-Flash decode profile

2026-11-25 13:58 UTC

## Summary

| metric     | value        | note                                                                               |
|------------|--------------|------------------------------------------------------------------------------------|
| step time  | 9.914 ms     | whole synthetic model forward in one graph; not serving ITL                        |
| roll-up    | 9.776 ms     | sum of the per-op leaf rows below                                                  |
| ratio      | 0.986        | roll-up / forward time; agreement does not prove coverage or numerical correctness |
| throughput | 1613.8 tok/s | at CONC 16                                                                         |

## Top kernels -- one step

| rank | API                                                                                                            | ms    | Kineto ms | share | calls | collective | target | hottest kernel                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|------|----------------------------------------------------------------------------------------------------------------|-------|-----------|-------|-------|------------|--------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | `aiter::fused_moe_ → aiter.ops.flydsl.kernels.tensor_shim._run_compiled:305`                                   | 2.826 | 2.613     | 28.9% | 80    |            | yes    | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 2    | `sglang.srt.distributed.device_communicators.pynccl_wrapper.ncclGroupEnd:677`                                  | 1.784 | 1.785     | 18.3% | 40    | yes        |        | `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 3    | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114`                    | 1.531 | 2.145     | 15.7% | 287   |            | yes    | `_gemm_afp8wfp8_packed_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_64_BLOCK_SIZE_K_256_K_PACK_1_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_SPLITK_BLOCK_SIZE_1280_FUSED_SPLITS_1_B_CACHE_MODIFIER_NONE_N_32768_K_1280_LAUNCH_OPTIONS_NUM_WARPS_4_NUM_STAGES_2_WAVES_PER_EU_2_MATRIX_INSTR_NONKDIM_16`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 4    | `sglang::reg_all_gather_into_tensor → aiter.dist.device_communicators.custom_all_reduce.all_gather_unreg:1871` | 1.459 | 1.459     | 14.9% | 80    | yes        |        | `void aiter::allgather_vec<std::bfloat16_t, 4>(aiter::RankData*, aiter::RankSignals, aiter::Signal*, std::bfloat16_t*, int, int)`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 5    | `aiter.ops.triton.attention.pa_decode_sparse._pa_decode_sparse_gfx950_gluon:536`                               | 0.367 | 0.435     | 3.8%  | 40    |            | yes    | `_sparse_mla_BLOCK_M_16_BLOCK_K_64_HEAD_SIZE_512_SPLIT_K_1_MAIN_FMT_FP8_DSV4_MLA_ROPE_SEPARATE_0`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 6    | `aiter.fused_moe._moe_sorting_impl:566 → aiter::moe_sorting_opus_fwd`                                          | 0.353 | 0.326     | 3.6%  | 80    |            | yes    | `void aiter::opus_moe_sorting_entry<aiter::MoeSortingMultiPhaseKernel_P23<aiter::MoeSortingProblemMp<int, float, int, 1, true, false, true> >, aiter::MoeSortingMultiPhaseKernel_P23<aiter::MoeSortingProblemMp<int, float, int, 1, true, false, true> >::Kargs>(aiter::MoeSortingMultiPhaseKernel_P23<aiter::MoeSortingProblemMp<int, float, int, 1, true, false, true> >::Kargs)`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 7    | `aiter::batched_gemm_a8w8_mxscale_bpreshuffle → aiter.ops.flydsl.batched_gemm_a8w8_gfx950._launch:265`         | 0.267 | 0.317     | 2.7%  | 40    |            | yes    | `bmm_a8w8_mxscale_t16x32x256_w1x2_nb5_sk1_b8_s32_32x32_nt_xcd_sps`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 8    | `sglang.kernels.ops.attention.dsv4.norm_rope.fused_k_norm_rope_flashmla:304`                                   | 0.228 | 0.271     | 2.3%  | 40    |            |        | `void sglang::fused_k_norm_rope_flashmla<__hip_bfloat16, 512l, 64l, long, 8, (sglang::deepseek_v4::KVLayout)0, false, true>(sglang::FusedKNormRopeFlashMLAParams)`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 9    | `aiter.ops.quant.fused_dynamic_mx_quant_moe_sort:1088 → aiter::fused_dynamic_mx_quant_moe_sort_hip_bounded`    | 0.198 | 0.183     | 2.0%  | 40    |            |        | `_ZN5aiter30fused_mx_quant_moe_sort_kernelIDF16bDB8_Li256ELi32EEEvPT0_PhPKT_PKiS9_PKfiiiiiiiii`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 10   | `sglang.srt.layers.logits_processor._compute_lm_head:1373 → aten::mm`                                          | 0.194 | 0.192     | 2.0%  | 1     |            |        | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x128_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR0_CLS0_CADS0_DTLA1_DTLB1_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLL0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA4_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS16_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO1_SRVW0_SSO0_SVW4_SK5_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN1_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSMn1_ULSGRO1_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGRO0_USI0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1_WGMXCCn1` |

Kineto ms of each timed leaf row scaled to its graph time (factors 0.00..1.15, 1 collective-heavy row(s) clamped); collectives take the row's residual. Targets are the top 5 non-collective groups plus ties.

## Config

| item        | value                                                                                                                                                                                                                                                               |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model       | `DeepseekV4ForCausalLM`  /models/DeepSeek-V4.1-Flash                                                                                                                                                                                                                |
| layers      | 40 layers -- moe+c0 1 (0), moe+c0+engram 1 (1), moe+c2+kvsrc+indexer 2 (2,8), moe+c2 15 (3-7,9-13,15-19), moe+c2+kvsrc+indexer+engram 1 (14), moe+c1+kvsrc+indexer+late 1 (20), moe+c1+late 15 (21-23,25-27,29-31,33-35,37-39), moe+c1+indexer+late 4 (24,28,32,36) |
| parallel    | TP 4, attn-DP 4, attn-TP 1, MoE-EP 4                                                                                                                                                                                                                                |
| batch       | CONC 16, 4 requests per rank                                                                                                                                                                                                                                        |
| context     | ISL 1024 per request                                                                                                                                                                                                                                                |
| KV setup    | synthetic: native full allocation of the ISL context, random KV bytes (no prefill forward)                                                                                                                                                                          |
| routing     | seeded balanced (seed 1234) on every MoE layer, hash layers included                                                                                                                                                                                                |
| kv dtype    | `fp8_e4m3`                                                                                                                                                                                                                                                          |
| weights     | `auto` load, fp8 block 32x32 quantized, experts fp4, bfloat16 elsewhere                                                                                                                                                                                             |
| measurement | 10 graph replays per op; 20 calls per graph, more for a weight small enough to rotate over the 512 MB buffer, fastest of 2 passes, max over ranks (EP)                                                                                                              |

## Versions

| component | version                                         |
|-----------|-------------------------------------------------|
| bench     | no git files-sha256 0eefe1b4fcf5                |
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
| swa_kv_pool                            | 40      | 89           | 3577  | 104.63 |
| kv_pools[1]                            | 1       | 585          | 585   | 17.11  |
| kv_pools[2]                            | 3       | 293          | 878   | 25.67  |
| index_pools[1]                         | 2       | 34           | 68    | 1.99   |
| index_pools[2]                         | 6       | 17           | 102   | 2.98   |
| **kv total**                           |         |              | 5209  | 152.38 |
| weights                                |         |              |       | 83.25  |
| **HBM**                                |         |              |       | 287.98 |
| host-resident params/buffers (not HBM) |         |              |       | 188.83 |

- **capacity 125,638,656 full-KV tokens cluster-wide** = 31,409,664 per rank x 4 DP ranks = 122,694 requests at ISL 1024
- native pools: full KV 31,409,664, SWA 4,800,768, request slots 64
- `--mem-fraction-static 0.85` caps weights + kv at 245 GB of the 288 GB HBM; this run lands at 236 GB
- this run uses 4,096 context tokens, 0.0% of this rank's full-KV pool
- synthetic KV filled: `index_pools[1].index_k_payload_buffer` 1.87 GB, `index_pools[1].index_k_scale_buffer` 0.12 GB, `index_pools[2].index_k_payload_buffer` 2.81 GB, `index_pools[2].index_k_scale_buffer` 0.18 GB, `kv_pools[1].kv_buffer` 17.11 GB, `kv_pools[2].kv_buffer` 25.67 GB, `swa_kv_pool.kv_buffer` 104.63 GB
- EP 4: this rank holds 96 routed experts at full intermediate width

## Hotspots -- one step, all layers

| op / kernel | ms/step | share | kernel calls/step | profile GPU ms/step | % of module GPU | Python / dispatcher API |
| --- | --- | --- | --- | --- | --- | --- |
| `mlp.experts` | 3.377 | 34.5% |  |  |  |  |
|   ↳ `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_v2out_v33` |  |  | 40 | 1.626 | 52.06% | `aiter::fused_moe_ → aiter.ops.flydsl.kernels.tensor_shim._run_compiled:305` |
|   ↳ `gemm2_a4w4_port_hmax8192_imax2304_bm32_bn128_bk256_nt_atomic_a8_bhoist_apf_spart4x2_bf16lds_v2_biasabi7` |  |  | 40 | 0.988 | 31.63% | `aiter::fused_moe_ → aiter.ops.flydsl.kernels.tensor_shim._run_compiled:305` |
|   ↳ `_ZN5aiter30fused_mx_quant_moe_sort_kernelIDF16bDB8_Li256ELi32EEEvPT0_PhPKT_PKiS9_PKfiiiiiiiii` |  |  | 40 | 0.183 | 5.86% | `aiter.ops.quant.fused_dynamic_mx_quant_moe_sort:1088 → aiter::fused_dynamic_mx_quant_moe_sort_hip_bounded` |
|   ↳ `void aiter::opus_moe_sorting_entry<aiter::MoeSortingMultiPhaseKernel_P23<aiter::MoeSortingProblemMp<int, float, int, 1, true, false, true> >, aiter::MoeSortingMultiPhaseKernel_P23<aiter::MoeSortingProblemMp<int, float, int, 1, true, false, true> >::Kargs>(aiter::MoeSortingMultiPhaseKernel_P23<aiter::MoeSortingProblemMp<int, float, int, 1, true, false, true> >::Kargs)` |  |  | 40 | 0.165 | 5.28% | `aiter.fused_moe._moe_sorting_impl:566 → aiter::moe_sorting_opus_fwd` |
|   ↳ `void aiter::opus_moe_sorting_entry<aiter::MoeSortingMultiPhaseKernel_P0_v2<aiter::MoeSortingProblemMp<int, float, int, 1, true, false, true> >, aiter::MoeSortingMultiPhaseKernel_P0_v2<aiter::MoeSortingProblemMp<int, float, int, 1, true, false, true> >::Kargs>(aiter::MoeSortingMultiPhaseKernel_P0_v2<aiter::MoeSortingProblemMp<int, float, int, 1, true, false, true> >::Kargs)` |  |  | 40 | 0.162 | 5.18% | `aiter.fused_moe._moe_sorting_impl:566 → aiter::moe_sorting_opus_fwd` |
| `layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 3.243 | 33.2% |  |  |  |  |
|   ↳ `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` |  |  | 40 | 1.785 | 30.54% | `sglang.srt.distributed.device_communicators.pynccl_wrapper.ncclGroupEnd:677` |
|   ↳ `void aiter::allgather_vec<std::bfloat16_t, 4>(aiter::RankData*, aiter::RankSignals, aiter::Signal*, std::bfloat16_t*, int, int)` |  |  | 40 | 1.320 | 22.58% | `sglang::reg_all_gather_into_tensor → aiter.dist.device_communicators.custom_all_reduce.all_gather_unreg:1871` |
|   ↳ `_hc_boundary_partial_kernel` |  |  | 80 | 0.781 | 13.37% | `sglang.kernels.ops.layernorm.mhc_boundary_hip._hc_boundary_partials:622` |
|   ↳ `_rmsnorm_sinkhorn_kernel` |  |  | 80 | 0.563 | 9.63% | `sglang.kernels.ops.layernorm.mhc_boundary_hip.rmsnorm_with_sinkhorn:388` |
|   ↳ `_gemm_afp8wfp8_packed_kernel_BLOCK_SIZE_M_4_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_K_PACK_4_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_SPLITK_BLOCK_SIZE_1024_FUSED_SPLITS_5_B_CACHE_MODIFIER_NONE_N_4608_K_5120_LAUNCH_OPTIONS_NUM_WARPS_2_NUM_STAGES_2_WAVES_PER_EU_0_MATRIX_INSTR_NONKDIM_0` |  |  | 40 | 0.305 | 5.22% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
|   ↳ `_gemm_afp8wfp8_packed_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_32_BLOCK_SIZE_K_1024_K_PACK_1_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_SPLITK_BLOCK_SIZE_3072_FUSED_SPLITS_1_B_CACHE_MODIFIER_NONE_N_5120_K_2304_LAUNCH_OPTIONS_NUM_WARPS_2_NUM_STAGES_3_WAVES_PER_EU_0_MATRIX_INSTR_NONKDIM_0` |  |  | 40 | 0.253 | 4.33% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
|   ↳ `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16, bool)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16, bool)#1} const&)::{lambda(int, bool)#1})` |  |  | 40 | 0.185 | 3.16% | `sglang.srt.models.deepseek_common.moe_transport._mask_dp_pad_rows:66 → aten::masked_fill_` |
|   ↳ `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::(anonymous namespace)::CompareFunctor<long> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<long> const&)::{lambda(int, bool)#2}>(int, at::native::gpu_kernel_impl<at::native::(anonymous namespace)::CompareFunctor<long> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<long> const&)::{lambda(int, bool)#2})` |  |  | 40 | 0.180 | 3.07% | `sglang.srt.models.deepseek_common.moe_transport._mask_dp_pad_rows:66 → aten::ge` |
|   ↳ `Memcpy DtoD (Device -> Device)` |  |  | 40 | 0.139 | 2.38% | `sglang::reg_all_gather_into_tensor → aiter.dist.device_communicators.custom_all_reduce.all_gather_unreg:1871` |
|   ↳ `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>)` |  |  | 40 | 0.126 | 2.15% | `sglang.srt.models.deepseek_common.moe_transport.return_output:219 → aten::add` |
|   ↳ `_silu_and_mul_clamp_kernel` |  |  | 40 | 0.107 | 1.82% | `sglang.kernels.ops.activation.silu_and_mul_clamp_hip.silu_and_mul_clamp_triton:79` |
|   ↳ `void (anonymous namespace)::elementwise_kernel_with_index_grid_stride<int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, function_traits<at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>::result_type*)` |  |  | 40 | 0.094 | 1.60% | `sglang.srt.models.deepseek_common.moe_transport._mask_dp_pad_rows:66 → aten::arange` |
|   ↳ `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1})` |  |  | 1 | 0.004 | 0.08% | `sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc.hc_boundary:469 → aten::copy_` |
|   ↳ `void aiter::mhc_post_kernel<std::bfloat16_t, 4, 4, 1024, false>(std::bfloat16_t*, std::bfloat16_t*, std::bfloat16_t*, float*, float*, int, int, int, int, int)` |  |  | 1 | 0.004 | 0.07% | `aiter.ops.mhc.mhc_post:781 → aiter::_mhc_post` |
| `attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel` | 1.219 | 12.5% |  |  |  |  |
|   ↳ `_sparse_mla_BLOCK_M_16_BLOCK_K_64_HEAD_SIZE_512_SPLIT_K_1_MAIN_FMT_FP8_DSV4_MLA_ROPE_SEPARATE_0` |  |  | 38 | 0.413 | 28.54% | `aiter.ops.triton.attention.pa_decode_sparse._pa_decode_sparse_gfx950_gluon:536` |
|   ↳ `bmm_a8w8_mxscale_t16x32x256_w1x2_nb5_sk1_b8_s32_32x32_nt_xcd_sps` |  |  | 40 | 0.317 | 21.90% | `aiter::batched_gemm_a8w8_mxscale_bpreshuffle → aiter.ops.flydsl.batched_gemm_a8w8_gfx950._launch:265` |
|   ↳ `void sglang::fused_k_norm_rope_flashmla<__hip_bfloat16, 512l, 64l, long, 8, (sglang::deepseek_v4::KVLayout)0, false, true>(sglang::FusedKNormRopeFlashMLAParams)` |  |  | 40 | 0.271 | 18.73% | `sglang.kernels.ops.attention.dsv4.norm_rope.fused_k_norm_rope_flashmla:304` |
|   ↳ `_rmsnorm_fake_quant_fp8_kernel` |  |  | 40 | 0.142 | 9.84% | `sglang.kernels.ops.layernorm.rmsnorm_fake_quant_amd_gfx95.rmsnorm_fake_quant_fp8:160` |
|   ↳ `_ZN5aiter31inverse_rope_group_quant_kernelIDF16bDF16bLi512ELi64ELi32ELi2ELi1ELNS_11ScaleLayoutE0ELNS_9TierOrderE0ELb0ELb0EEEvPKT_PDB8_PhPKlPKT0_SD_iiiiiillliiibbii` |  |  | 40 | 0.139 | 9.64% | `aiter.ops.inverse_rope_group_quant.inverse_rope_group_quant:91 → aiter::_inverse_rope_group_quant_kernel` |
|   ↳ `_gfx_sparse_decode_reduce_kernel` |  |  | 38 | 0.123 | 8.52% | `sglang.kernels.ops.attention.sparse_decode_reduce_hip.gfx_sparse_split_reduce:78` |
|   ↳ `_sparse_mla_BLOCK_M_16_BLOCK_K_64_HEAD_SIZE_512_SPLIT_K_0_MAIN_FMT_FP8_DSV4_MLA_ROPE_SEPARATE_0` |  |  | 2 | 0.022 | 1.55% | `aiter.ops.triton.attention.pa_decode_sparse._pa_decode_sparse_gfx950_gluon:536` |
|   ↳ `_mask_indices_by_length_kernel` |  |  | 8 | 0.018 | 1.27% | `sglang.kernels.ops.attention.dsv4.attn_glue_hip.mask_indices_by_length:57` |
| `mlp.shared_experts` | 0.564 | 5.8% |  |  |  |  |
|   ↳ `_gemm_afp8wfp8_packed_kernel_BLOCK_SIZE_M_4_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_K_PACK_4_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_SPLITK_BLOCK_SIZE_1024_FUSED_SPLITS_5_B_CACHE_MODIFIER_NONE_N_4608_K_5120_LAUNCH_OPTIONS_NUM_WARPS_2_NUM_STAGES_2_WAVES_PER_EU_0_MATRIX_INSTR_NONKDIM_0` |  |  | 40 | 0.284 | 41.85% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
|   ↳ `_gemm_afp8wfp8_packed_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_32_BLOCK_SIZE_K_1024_K_PACK_1_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_SPLITK_BLOCK_SIZE_3072_FUSED_SPLITS_1_B_CACHE_MODIFIER_NONE_N_5120_K_2304_LAUNCH_OPTIONS_NUM_WARPS_2_NUM_STAGES_3_WAVES_PER_EU_0_MATRIX_INSTR_NONKDIM_0` |  |  | 40 | 0.247 | 36.38% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
|   ↳ `_silu_and_mul_clamp_kernel` |  |  | 40 | 0.148 | 21.77% | `sglang.kernels.ops.activation.silu_and_mul_clamp_hip.silu_and_mul_clamp_triton:79` |
| `attn.wo_b` | 0.516 | 5.3% |  |  |  |  |
|   ↳ `_gemm_afp8wfp8_packed_kernel_BLOCK_SIZE_M_4_BLOCK_SIZE_N_16_BLOCK_SIZE_K_512_K_PACK_4_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_SPLITK_BLOCK_SIZE_2048_FUSED_SPLITS_4_B_CACHE_MODIFIER_NONE_N_5120_K_8192_LAUNCH_OPTIONS_NUM_WARPS_4_NUM_STAGES_2_WAVES_PER_EU_0_MATRIX_INSTR_NONKDIM_0` |  |  | 40 | 0.398 | 77.23% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
|   ↳ `_mxfp8_quant_kernel` |  |  | 40 | 0.117 | 22.77% | `sglang.kernels.ops.quantization.mxfp8_amd_gfx95._mxfp8_e4m3_quantize_triton:63` |
| `attn.wq_b` | 0.406 | 4.2% |  |  |  |  |
|   ↳ `_gemm_afp8wfp8_packed_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_64_BLOCK_SIZE_K_256_K_PACK_1_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_SPLITK_BLOCK_SIZE_1280_FUSED_SPLITS_1_B_CACHE_MODIFIER_NONE_N_32768_K_1280_LAUNCH_OPTIONS_NUM_WARPS_4_NUM_STAGES_2_WAVES_PER_EU_2_MATRIX_INSTR_NONKDIM_16` |  |  | 40 | 0.352 | 100.00% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
| `attn.wqkv_a` | 0.223 | 2.3% |  |  |  |  |
|   ↳ `_gemm_afp8wfp8_packed_kernel_BLOCK_SIZE_M_4_BLOCK_SIZE_N_16_BLOCK_SIZE_K_256_K_PACK_4_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_SPLITK_BLOCK_SIZE_1024_FUSED_SPLITS_5_B_CACHE_MODIFIER_CG_N_1792_K_5120_LAUNCH_OPTIONS_NUM_WARPS_2_NUM_STAGES_2_WAVES_PER_EU_0_MATRIX_INSTR_NONKDIM_0` |  |  | 40 | 0.234 | 100.00% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
| `logits` | 0.198 | 2.0% |  |  |  |  |
|   ↳ `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x128_MI16x16x1_SN_LDSB0_AFC0_AG1_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR0_CLS0_CADS0_DTLA1_DTLB1_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLL0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA4_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS16_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO1_SRVW0_SSO0_SVW4_SK5_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN1_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSMn1_ULSGRO1_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGRO0_USI0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1_WGMXCCn1` |  |  | 1 | 0.192 | 98.00% | `sglang.srt.layers.logits_processor._compute_lm_head:1373 → aten::mm` |
|   ↳ `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>)` |  |  | 1 | 0.004 | 2.00% | `sglang.srt.layers.logits_processor._copy_logits_to_buffer:1569 → aten::copy_` |
| `attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.181 | 1.9% |  |  |  |  |
|   ↳ `pa_mqa_logits_fp4_rowgroup_kernel_0` |  |  | 5 | 0.044 | 19.53% | `aiter.ops.flydsl.kernels.tensor_shim._run_compiled:305` |
|   ↳ `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT16x16x256_MI16x16x1_SN_LDSB1_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSTI0_LBSPPA512_LBSPPB512_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM3_NTn1_NTA4_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK5_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGRO0_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4_WGMXCCn1` |  |  | 3 | 0.033 | 14.66% | `sglang.kernels.ops.gemm.bf16_fp32._linear_bf16_fp32_cublas:108 → aten::mm` |
|   ↳ `_index_q_pack_weights_kernel` |  |  | 5 | 0.028 | 12.11% | `sglang.kernels.ops.attention.dsv4.fp4_indexer_hip.index_q_rope_pack_weights_flydsl:1119` |
|   ↳ `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x256_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA1_DTLB1_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB1_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA32_LPB32_LPMXSA0_LPMXSB0_LPM0_LRVWn1_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA4_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS12_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR4_PLR1_PKA1_SGROB1_SIA3_SS0_SPO1_SRVW0_SSO1_SVW4_SK5_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN1_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS2_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGRO0_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_1_WGMXCCn1` |  |  | 4 | 0.020 | 9.00% | `torch.nn.modules.linear.forward:130 → aten::mm` |
|   ↳ `void sglang::topk_mode_1::topk_split_select<(sglang::topk_mode_1::TopKMode)2>(sglang::topk_mode_1::TopKPagedParams, sglang::topk_mode_1::SplitWorkspace)` |  |  | 5 | 0.020 | 8.90% | `sglang.kernels.ops.attention.dsv4.topk.topk_transform_paged_v2:325` |
|   ↳ `_gemm_afp8wfp8_packed_kernel_BLOCK_SIZE_M_4_BLOCK_SIZE_N_16_BLOCK_SIZE_K_256_K_PACK_4_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_SPLITK_BLOCK_SIZE_2048_FUSED_SPLITS_1_B_CACHE_MODIFIER_NONE_N_4096_K_1280_LAUNCH_OPTIONS_NUM_WARPS_4_NUM_STAGES_2_WAVES_PER_EU_0_MATRIX_INSTR_NONKDIM_16` |  |  | 5 | 0.018 | 7.96% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
|   ↳ `_router_gemv_split_k_kernel` |  |  | 5 | 0.015 | 6.49% | `sglang.kernels.ops.gemm.router_gemv_hip.rocm_router_gemv_split_k:77` |
|   ↳ `void sglang::flash_index_k_split_kernel<false, 128l, 64l, 64u, 2u, false, long>(sglang::IndexKSplitParams)` |  |  | 3 | 0.010 | 4.59% | `sglang.kernels.ops.attention.dsv4.fp4_indexer_rope_hip.index_k_norm_rope_pack_store_split:37` |
|   ↳ `sglang::topk_mode_1::topk_split_hist(sglang::topk_mode_1::TopKPagedParams, sglang::topk_mode_1::SplitWorkspace)` |  |  | 5 | 0.009 | 3.85% | `sglang.kernels.ops.attention.dsv4.topk.topk_transform_paged_v2:325` |
|   ↳ `(anonymous namespace)::deepseek_v4_topk_transform_kernel((anonymous namespace)::TopKParams)` |  |  | 3 | 0.007 | 2.94% | `sglang.kernels.ops.attention.dsv4.topk.topk_transform_paged:118 → sgl_kernel::deepseek_v4_topk_transform_512` |
|   ↳ `void sglang::flash_c2_decode_kernel<(sglang::C2Mode)0, 512l, 64l, 7, long, long, (sglang::deepseek_v4::KVLayout)0, false>(sglang::Compress2DecodeParams)` |  |  | 3 | 0.006 | 2.76% | `sglang.kernels.ops.attention.dsv4.low_ratio_compress.c2_decode_norm_rope_store:135` |
|   ↳ `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA1_DTLB1_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB1_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA4_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS16_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR4_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK5_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN1_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSMn1_ULSGRO1_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGRO0_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG32_4_1_WGMXCCn1` |  |  | 1 | 0.006 | 2.44% | `torch.nn.modules.linear.forward:130 → aten::mm` |
|   ↳ `void sglang::flash_index_k_split_kernel<false, 128l, 64l, 64u, 1u, true, long>(sglang::IndexKSplitParams)` |  |  | 1 | 0.004 | 1.83% | `sglang.kernels.ops.attention.dsv4.fp4_indexer_rope_hip.index_k_norm_rope_pack_store_split:37` |
|   ↳ `Cijk_SB_BiasS_HAS_ScaleAlphaVec_PostGSU16_VW1` |  |  | 1 | 0.003 | 1.53% | `torch.nn.modules.linear.forward:130 → aten::mm` |
|   ↳ `void sglang::flash_c1_decode_kernel<512l, 64l, 8, long, long, (sglang::deepseek_v4::KVLayout)0, false>(sglang::Compress1DecodeParams)` |  |  | 1 | 0.003 | 1.41% | `sglang.kernels.ops.attention.dsv4.low_ratio_compress.c1_decode_norm_rope_store:52` |
| `mlp.topk` | 0.149 | 1.5% |  |  |  |  |
|   ↳ `_router_gate_kernel` |  |  | 40 | 0.193 | 100.00% | `sglang.kernels.ops.moe.rocm_router_gate.rocm_router_gate:262` |
| `engram :: Engram n-gram embedding gate (runs before the layer)` | 0.069 | 0.7% |  |  |  |  |
|   ↳ `_gemm_afp8wfp8_packed_kernel_BLOCK_SIZE_M_4_BLOCK_SIZE_N_16_BLOCK_SIZE_K_512_K_PACK_4_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_SPLITK_BLOCK_SIZE_2048_FUSED_SPLITS_3_B_CACHE_MODIFIER_NONE_N_25600_K_6144_LAUNCH_OPTIONS_NUM_WARPS_4_NUM_STAGES_2_WAVES_PER_EU_0_MATRIX_INSTR_NONKDIM_0` |  |  | 2 | 0.054 | 69.86% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
|   ↳ `_engram_gate_kernel` |  |  | 2 | 0.009 | 11.43% | `sglang.kernels.ops.embeddings.engram_gate.fused_engram_gate:49` |
|   ↳ `_engram_gather_kernel` |  |  | 2 | 0.008 | 9.87% | `sglang.kernels.ops.embeddings.engram_gather.engram_gather:129` |
|   ↳ `_mxfp8_quant_kernel` |  |  | 2 | 0.007 | 8.83% | `sglang.kernels.ops.quantization.mxfp8_amd_gfx95._mxfp8_e4m3_quantize_triton:63` |
| `engram_hasher` | 0.005 | 0.0% |  |  |  |  |
|   ↳ `_engram_hash_kernel` |  |  | 1 | 0.006 | 100.00% | `sglang.kernels.ops.embeddings.engram_hash._launch_hash_kernel:223` |
| `embed_tokens` | 0.003 | 0.0% |  |  |  |  |
|   ↳ `void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::BFloat16, long, unsigned int, 2, 2, -2>(at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, long)` |  |  | 1 | 0.004 | 100.00% | `torch.nn.functional.embedding:2455 → aten::index_select` |
| `hc_head :: final mHC combine` | 0.002 | 0.0% |  |  |  |  |
|   ↳ `_mhc_combine_kernel` |  |  | 1 | 0.002 | 100.00% | `sglang.kernels.ops.layernorm.mhc.mhc_combine:2800` |
| `norm` | 0.002 | 0.0% |  |  |  |  |
|   ↳ `void aiter::add_rmsnorm_quant_kernel<std::bfloat16_t, std::bfloat16_t, 256, 24, false, false, true, 1, false>(std::bfloat16_t*, std::bfloat16_t*, float*, std::bfloat16_t*, std::bfloat16_t*, std::bfloat16_t*, double, bool, int, int, int, int, int, int, int, bool, bool)` |  |  | 1 | 0.003 | 100.00% | `aiter.ops.rmsnorm._rms_norm_fwd_dispatch:401 → aiter::rmsnorm` |
| `mlp.<rest> :: shared+routed add, post-expert collective` | -0.380 | -3.9% |  |  |  |  |
|   ↳ `_router_gemv_split_k_kernel` |  |  | 40 | 0.134 | 100.00% | `sglang.kernels.ops.gemm.router_gemv_hip.rocm_router_gemv_split_k:77` |

Leaf rows only, so the shares add to 100%. A `<rest>` row is its parent minus the children listed under it, and the `::` tag names the work that lands there -- inline code and fused kernels with no module to hook. It is a difference of two measurements rather than a measurement, so any error in the parent or in any child lands in it at full size and reversed sign: a `<rest>` small next to its parent is mostly that error, and can read negative. 

Indented rows list **all** Kineto GPU kernels/copies from rank 0's standalone module replays (one eager sample per layer kind, multiplied by its layer count; globals once). Kernel percentages sum to 100% **within each module**. Parent `<rest>` samples exclude separately measured children through module hooks. The API column comes from the kernel's Kineto runtime correlation, nearest recorded Python frame, and dispatcher external ID. ms/step and share are graph timings; the profile columns are diagnostic.

Packages these APIs were loaded from: `aiter` = `/sgl-workspace/aiter/aiter`, `sglang` = `/sgl-workspace/sglang/python/sglang`, `torch` = `/opt/venv/lib/python3.12/site-packages/torch`.

## Per layer kind -- one step

### moe+c0 -- x1 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.222   | 0.222   |
| `  attn`                                                                                 | 0.051   | 0.051   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.006   |
| `    attn.wq_b`                                                                          | 0.010   | 0.010   |
| `    attn.wo_b`                                                                          | 0.013   | 0.013   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.023   | 0.023   |
| `  mlp`                                                                                  | 0.091   | 0.091   |
| `    mlp.topk`                                                                           | 0.004   | 0.004   |
| `    mlp.experts`                                                                        | 0.084   | 0.084   |
| `    mlp.shared_experts`                                                                 | 0.014   | 0.014   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.011  | -0.011  |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.080   | 0.080   |

### moe+c0+engram -- x1 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.215   | 0.215   |
| `  attn`                                                                                 | 0.052   | 0.052   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.006   |
| `    attn.wq_b`                                                                          | 0.010   | 0.010   |
| `    attn.wo_b`                                                                          | 0.013   | 0.013   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.023   | 0.023   |
| `  mlp`                                                                                  | 0.091   | 0.091   |
| `    mlp.topk`                                                                           | 0.004   | 0.004   |
| `    mlp.experts`                                                                        | 0.084   | 0.084   |
| `    mlp.shared_experts`                                                                 | 0.014   | 0.014   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.011  | -0.011  |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.073   | 0.073   |
| `engram :: Engram n-gram embedding gate (runs before the layer)`                         | 0.034   | 0.034   |

### moe+c2+kvsrc+indexer -- x2 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.259   | 0.519   |
| `  attn`                                                                                 | 0.080   | 0.161   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.011   |
| `    attn.wq_b`                                                                          | 0.010   | 0.020   |
| `    attn.wo_b`                                                                          | 0.013   | 0.026   |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.018   | 0.037   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.034   | 0.067   |
| `  mlp`                                                                                  | 0.091   | 0.181   |
| `    mlp.topk`                                                                           | 0.004   | 0.007   |
| `    mlp.experts`                                                                        | 0.084   | 0.168   |
| `    mlp.shared_experts`                                                                 | 0.014   | 0.028   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.011  | -0.022  |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.088   | 0.176   |

### moe+c2 -- x15 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.233   | 3.488   |
| `  attn`                                                                                 | 0.059   | 0.878   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.084   |
| `    attn.wq_b`                                                                          | 0.010   | 0.152   |
| `    attn.wo_b`                                                                          | 0.013   | 0.194   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.030   | 0.448   |
| `  mlp`                                                                                  | 0.093   | 1.401   |
| `    mlp.topk`                                                                           | 0.004   | 0.056   |
| `    mlp.experts`                                                                        | 0.085   | 1.273   |
| `    mlp.shared_experts`                                                                 | 0.014   | 0.211   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.009  | -0.138  |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.081   | 1.210   |

### moe+c2+kvsrc+indexer+engram -- x1 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.247   | 0.247   |
| `  attn`                                                                                 | 0.080   | 0.080   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.006   |
| `    attn.wq_b`                                                                          | 0.010   | 0.010   |
| `    attn.wo_b`                                                                          | 0.013   | 0.013   |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.017   | 0.017   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.034   | 0.034   |
| `  mlp`                                                                                  | 0.091   | 0.091   |
| `    mlp.topk`                                                                           | 0.004   | 0.004   |
| `    mlp.experts`                                                                        | 0.084   | 0.084   |
| `    mlp.shared_experts`                                                                 | 0.014   | 0.014   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.011  | -0.011  |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.076   | 0.076   |
| `engram :: Engram n-gram embedding gate (runs before the layer)`                         | 0.034   | 0.034   |

### moe+c1+kvsrc+indexer+late -- x1 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.272   | 0.272   |
| `  attn`                                                                                 | 0.104   | 0.104   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.006   |
| `    attn.wq_b`                                                                          | 0.010   | 0.010   |
| `    attn.wo_b`                                                                          | 0.013   | 0.013   |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.039   | 0.039   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.036   | 0.036   |
| `  mlp`                                                                                  | 0.090   | 0.090   |
| `    mlp.topk`                                                                           | 0.004   | 0.004   |
| `    mlp.experts`                                                                        | 0.084   | 0.084   |
| `    mlp.shared_experts`                                                                 | 0.014   | 0.014   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.012  | -0.012  |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.078   | 0.078   |

### moe+c1+late -- x15 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.232   | 3.484   |
| `  attn`                                                                                 | 0.058   | 0.874   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.083   |
| `    attn.wq_b`                                                                          | 0.010   | 0.152   |
| `    attn.wo_b`                                                                          | 0.013   | 0.193   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.030   | 0.445   |
| `  mlp`                                                                                  | 0.093   | 1.401   |
| `    mlp.topk`                                                                           | 0.004   | 0.056   |
| `    mlp.experts`                                                                        | 0.084   | 1.264   |
| `    mlp.shared_experts`                                                                 | 0.014   | 0.212   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.009  | -0.130  |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.081   | 1.209   |

### moe+c1+indexer+late -- x4 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.263   | 1.051   |
| `  attn`                                                                                 | 0.087   | 0.347   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.022   |
| `    attn.wq_b`                                                                          | 0.010   | 0.041   |
| `    attn.wo_b`                                                                          | 0.013   | 0.051   |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.022   | 0.089   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.036   | 0.143   |
| `  mlp`                                                                                  | 0.091   | 0.363   |
| `    mlp.topk`                                                                           | 0.004   | 0.015   |
| `    mlp.experts`                                                                        | 0.084   | 0.336   |
| `    mlp.shared_experts`                                                                 | 0.014   | 0.057   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.011  | -0.044  |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.085   | 0.341   |

## Roll-up -- one step

| op                                 | x  | ms/call | ms/step |
|------------------------------------|----|---------|---------|
| moe+c0 layer                       | 1  | 0.222   | 0.222   |
| moe+c0+engram layer                | 1  | 0.215   | 0.215   |
| moe+c0+engram engram               | 1  | 0.034   | 0.034   |
| moe+c2+kvsrc+indexer layer         | 2  | 0.259   | 0.519   |
| moe+c2 layer                       | 15 | 0.233   | 3.488   |
| moe+c2+kvsrc+indexer+engram layer  | 1  | 0.247   | 0.247   |
| moe+c2+kvsrc+indexer+engram engram | 1  | 0.034   | 0.034   |
| moe+c1+kvsrc+indexer+late layer    | 1  | 0.272   | 0.272   |
| moe+c1+late layer                  | 15 | 0.232   | 3.484   |
| moe+c1+indexer+late layer          | 4  | 0.263   | 1.051   |
| embed_tokens                       | 1  | 0.003   | 0.003   |
| engram_hasher                      | 1  | 0.005   | 0.005   |
| hc_head                            | 1  | 0.002   | 0.002   |
| norm                               | 1  | 0.002   | 0.002   |
| logits                             | 1  | 0.198   | 0.198   |

## Coverage

| layer kind                  | hook not called          | possible containing row |
|-----------------------------|--------------------------|-------------------------|
| moe+c0+engram               | attn.wo_a                | attn.<rest>             |
| moe+c0+engram               | input_layernorm          | layer.<rest>            |
| moe+c0+engram               | mlp.gate                 | mlp.<rest>              |
| moe+c0+engram               | post_attention_layernorm | layer.<rest>            |
| moe+c0                      | attn.wo_a                | attn.<rest>             |
| moe+c0                      | input_layernorm          | layer.<rest>            |
| moe+c0                      | mlp.gate                 | mlp.<rest>              |
| moe+c0                      | post_attention_layernorm | layer.<rest>            |
| moe+c1+indexer+late         | attn.indexer             | attn.<rest>             |
| moe+c1+indexer+late         | attn.wo_a                | attn.<rest>             |
| moe+c1+indexer+late         | input_layernorm          | layer.<rest>            |
| moe+c1+indexer+late         | mlp.gate                 | mlp.<rest>              |
| moe+c1+indexer+late         | post_attention_layernorm | layer.<rest>            |
| moe+c1+kvsrc+indexer+late   | attn.compressor          | attn.<rest>             |
| moe+c1+kvsrc+indexer+late   | attn.indexer             | attn.<rest>             |
| moe+c1+kvsrc+indexer+late   | attn.wo_a                | attn.<rest>             |
| moe+c1+kvsrc+indexer+late   | input_layernorm          | layer.<rest>            |
| moe+c1+kvsrc+indexer+late   | mlp.gate                 | mlp.<rest>              |
| moe+c1+kvsrc+indexer+late   | post_attention_layernorm | layer.<rest>            |
| moe+c1+late                 | attn.wo_a                | attn.<rest>             |
| moe+c1+late                 | input_layernorm          | layer.<rest>            |
| moe+c1+late                 | mlp.gate                 | mlp.<rest>              |
| moe+c1+late                 | post_attention_layernorm | layer.<rest>            |
| moe+c2+kvsrc+indexer+engram | attn.compressor          | attn.<rest>             |
| moe+c2+kvsrc+indexer+engram | attn.indexer             | attn.<rest>             |
| moe+c2+kvsrc+indexer+engram | attn.wo_a                | attn.<rest>             |
| moe+c2+kvsrc+indexer+engram | input_layernorm          | layer.<rest>            |
| moe+c2+kvsrc+indexer+engram | mlp.gate                 | mlp.<rest>              |
| moe+c2+kvsrc+indexer+engram | post_attention_layernorm | layer.<rest>            |
| moe+c2+kvsrc+indexer        | attn.compressor          | attn.<rest>             |
| moe+c2+kvsrc+indexer        | attn.indexer             | attn.<rest>             |
| moe+c2+kvsrc+indexer        | attn.wo_a                | attn.<rest>             |
| moe+c2+kvsrc+indexer        | input_layernorm          | layer.<rest>            |
| moe+c2+kvsrc+indexer        | mlp.gate                 | mlp.<rest>              |
| moe+c2+kvsrc+indexer        | post_attention_layernorm | layer.<rest>            |
| moe+c2                      | attn.wo_a                | attn.<rest>             |
| moe+c2                      | input_layernorm          | layer.<rest>            |
| moe+c2                      | mlp.gate                 | mlp.<rest>              |
| moe+c2                      | post_attention_layernorm | layer.<rest>            |

Hooked but never called during the recording forward. A name is missing either because this layer kind does not run that op, or because a fused kernel reads the module's weight directly (DSv4.1 on gfx950: the mHC boundary kernel applies `input_layernorm` / `post_attention_layernorm` itself). A parent residual can contain this work; missing globals are not covered by layer residuals.
