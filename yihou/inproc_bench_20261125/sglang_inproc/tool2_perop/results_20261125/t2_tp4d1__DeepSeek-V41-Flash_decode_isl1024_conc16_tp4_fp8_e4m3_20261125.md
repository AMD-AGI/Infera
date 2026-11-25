# DeepSeek-V41-Flash decode profile

2026-11-25 14:14 UTC

## Summary

| metric     | value        | note                                                                               |
|------------|--------------|------------------------------------------------------------------------------------|
| step time  | 8.375 ms     | whole synthetic model forward in one graph; not serving ITL                        |
| roll-up    | 8.268 ms     | sum of the per-op leaf rows below                                                  |
| ratio      | 0.987        | roll-up / forward time; agreement does not prove coverage or numerical correctness |
| throughput | 1910.4 tok/s | at CONC 16                                                                         |

## Top kernels -- one step

| rank | API                                                                                                         | ms    | Kineto ms | share | calls | collective | target | hottest kernel                                                                                                                                                                                                                                                                                  |
|------|-------------------------------------------------------------------------------------------------------------|-------|-----------|-------|-------|------------|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | `aiter::fused_moe_ → aiter.ops.flydsl.kernels.tensor_shim._run_compiled:305`                                | 2.872 | 2.961     | 34.7% | 80    |            | yes    | `mfma_moe1_silu_mul_afp8_wfp4_bf16_t32x128x256_pm1_async_gui_v33`                                                                                                                                                                                                                               |
| 2    | `sglang::outplace_all_reduce → aiter.dist.device_communicators.custom_all_reduce.all_reduce:1662`           | 0.966 | 17.890    | 11.7% | 162   | yes        |        | `void aiter::cross_device_reduce_2stage<std::bfloat16_t, 4, false, false>(aiter::RankData*, aiter::RankData*, aiter::RankSignals, aiter::Signal*, std::bfloat16_t*, int, int)`                                                                                                                  |
| 3    | `sglang.kernels.ops.layernorm.mhc_boundary_hip._hc_boundary_partials:622`                                   | 0.839 | 0.743     | 10.1% | 80    |            | yes    | `_hc_boundary_partial_kernel`                                                                                                                                                                                                                                                                   |
| 4    | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114`                 | 0.637 | 0.689     | 7.7%  | 127   |            | yes    | `_gemm_afp8wfp8_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_64_BLOCK_SIZE_K_512_GROUP_SIZE_M_1_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_FUSED_SPLITS_5_N_FIRST_0_num_warps_8_num_stages_3_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_CG_NUM_KSPLIT_5_SPLITK_BLOCK_SIZE_1024` |
| 5    | `sglang.kernels.ops.layernorm.mhc_boundary_hip.rmsnorm_with_sinkhorn:388`                                   | 0.566 | 0.502     | 6.8%  | 80    |            | yes    | `_rmsnorm_sinkhorn_kernel`                                                                                                                                                                                                                                                                      |
| 6    | `aiter.ops.triton.attention.pa_decode_sparse._pa_decode_sparse_gfx950_gluon:536`                            | 0.402 | 0.418     | 4.9%  | 40    |            | yes    | `_sparse_mla_BLOCK_M_16_BLOCK_K_64_HEAD_SIZE_512_SPLIT_K_1_MAIN_FMT_FP8_DSV4_MLA_ROPE_SEPARATE_0`                                                                                                                                                                                               |
| 7    | `aiter.ops.quant.fused_dynamic_mx_quant_moe_sort:1088 → aiter::fused_dynamic_mx_quant_moe_sort_hip_bounded` | 0.380 | 0.391     | 4.6%  | 80    |            | yes    | `_ZN5aiter30fused_mx_quant_moe_sort_kernelIDF16bDB8_Li256ELi32EEEvPT0_PhPKT_PKiS9_PKfiiiiiiiii`                                                                                                                                                                                                 |
| 8    | `aiter::fused_moe_ → sglang.kernels.ops.moe.moe_sorting_small._run_small_sort:410`                          | 0.203 | 0.209     | 2.5%  | 40    |            |        | `_moe_sorting_small_kernel_sorted`                                                                                                                                                                                                                                                              |
| 9    | `aiter::batched_gemm_a8w8_mxscale_bpreshuffle → aiter.ops.flydsl.batched_gemm_a8w8_gfx950._launch:265`      | 0.201 | 0.208     | 2.4%  | 40    |            |        | `bmm_a8w8_mxscale_t16x32x256_w1x2_nb5_sk4_b2_s32_32x32_nt_xcd_sps`                                                                                                                                                                                                                              |
| 10   | `sglang.kernels.ops.attention.dsv4.norm_rope.fused_k_norm_rope_flashmla:304`                                | 0.153 | 0.159     | 1.9%  | 40    |            |        | `void sglang::fused_k_norm_rope_flashmla<__hip_bfloat16, 512l, 64l, long, 8, (sglang::deepseek_v4::KVLayout)0, false, true>(sglang::FusedKNormRopeFlashMLAParams)`                                                                                                                              |

Kineto ms of each timed leaf row scaled to its graph time (factors 0.81..1.13, 4 collective-heavy row(s) clamped); collectives take the row's residual. Targets are the top 5 non-collective groups plus ties.

## Config

| item        | value                                                                                                                                                                                                                                                               |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model       | `DeepseekV4ForCausalLM`  /models/DeepSeek-V4.1-Flash                                                                                                                                                                                                                |
| layers      | 40 layers -- moe+c0 1 (0), moe+c0+engram 1 (1), moe+c2+kvsrc+indexer 2 (2,8), moe+c2 15 (3-7,9-13,15-19), moe+c2+kvsrc+indexer+engram 1 (14), moe+c1+kvsrc+indexer+late 1 (20), moe+c1+late 15 (21-23,25-27,29-31,33-35,37-39), moe+c1+indexer+late 4 (24,28,32,36) |
| parallel    | TP 4, attn-DP 1, attn-TP 4, MoE-EP 1                                                                                                                                                                                                                                |
| batch       | CONC 16, 16 requests per rank                                                                                                                                                                                                                                       |
| context     | ISL 1024 per request                                                                                                                                                                                                                                                |
| KV setup    | synthetic: native full allocation of the ISL context, random KV bytes (no prefill forward)                                                                                                                                                                          |
| routing     | seeded balanced (seed 1234) on every MoE layer, hash layers included                                                                                                                                                                                                |
| kv dtype    | `fp8_e4m3`                                                                                                                                                                                                                                                          |
| weights     | `auto` load, fp8 block 32x32 quantized, experts fp4, bfloat16 elsewhere                                                                                                                                                                                             |
| measurement | 5 graph replays per op; 20 calls per graph, more for a weight small enough to rotate over the 512 MB buffer, min over ranks                                                                                                                                         |

## Versions

| component | version                                         |
|-----------|-------------------------------------------------|
| bench     | no git files-sha256 735122bf5973                |
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
| swa_kv_pool                            | 40      | 165          | 6594  | 107.84 |
| kv_pools[1]                            | 1       | 585          | 585   | 9.57   |
| kv_pools[2]                            | 3       | 293          | 878   | 14.35  |
| index_pools[1]                         | 2       | 34           | 68    | 1.11   |
| index_pools[2]                         | 6       | 17           | 102   | 1.67   |
| **kv total**                           |         |              | 8227  | 134.54 |
| weights                                |         |              |       | 84.62  |
| **HBM**                                |         |              |       | 287.98 |
| host-resident params/buffers (not HBM) |         |              |       | 188.83 |

- **capacity 17,560,320 full-KV tokens** = 17,148 requests at ISL 1024; the MQA KV is replicated on every TP rank
- native pools: full KV 17,560,320, SWA 4,948,224, request slots 256
- `--mem-fraction-static 0.85` caps weights + kv at 245 GB of the 288 GB HBM; this run lands at 219 GB
- this run uses 16,384 context tokens, 0.1% of this rank's full-KV pool
- synthetic KV filled: `index_pools[1].index_k_payload_buffer` 1.05 GB, `index_pools[1].index_k_scale_buffer` 0.07 GB, `index_pools[2].index_k_payload_buffer` 1.57 GB, `index_pools[2].index_k_scale_buffer` 0.10 GB, `kv_pools[1].kv_buffer` 9.57 GB, `kv_pools[2].kv_buffer` 14.35 GB, `swa_kv_pool.kv_buffer` 107.84 GB

## Hotspots -- one step, all layers

| op / kernel | ms/step | share | kernel calls/step | profile GPU ms/step | % of module GPU | Python / dispatcher API |
| --- | --- | --- | --- | --- | --- | --- |
| `mlp.experts` | 3.454 | 41.8% |  |  |  |  |
|   ↳ `mfma_moe1_silu_mul_afp8_wfp4_bf16_t32x128x256_pm1_async_gui_v33` |  |  | 40 | 1.972 | 55.38% | `aiter::fused_moe_ → aiter.ops.flydsl.kernels.tensor_shim._run_compiled:305` |
|   ↳ `mfma_moe2_afp8_wfp4_bf16_cshuffle_t32x128x128_vscale_fix3_fp4opt_v1_pm1_abuffer` |  |  | 40 | 0.989 | 27.76% | `aiter::fused_moe_ → aiter.ops.flydsl.kernels.tensor_shim._run_compiled:305` |
|   ↳ `_ZN5aiter30fused_mx_quant_moe_sort_kernelIDF16bDB8_Li256ELi32EEEvPT0_PhPKT_PKiS9_PKfiiiiiiiii` |  |  | 40 | 0.213 | 5.97% | `aiter.ops.quant.fused_dynamic_mx_quant_moe_sort:1088 → aiter::fused_dynamic_mx_quant_moe_sort_hip_bounded` |
|   ↳ `_moe_sorting_small_kernel_sorted` |  |  | 40 | 0.209 | 5.87% | `aiter::fused_moe_ → sglang.kernels.ops.moe.moe_sorting_small._run_small_sort:410` |
|   ↳ `_ZN5aiter30fused_mx_quant_moe_sort_kernelIDF16bDB8_Li128ELi8EEEvPT0_PhPKT_PKiS9_PKfiiiiiiiii` |  |  | 40 | 0.179 | 5.02% | `aiter.ops.quant.fused_dynamic_mx_quant_moe_sort:1088 → aiter::fused_dynamic_mx_quant_moe_sort_hip_bounded` |
| `layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 1.415 | 17.1% |  |  |  |  |
|   ↳ `_hc_boundary_partial_kernel` |  |  | 80 | 0.743 | 59.28% | `sglang.kernels.ops.layernorm.mhc_boundary_hip._hc_boundary_partials:622` |
|   ↳ `_rmsnorm_sinkhorn_kernel` |  |  | 80 | 0.502 | 40.02% | `sglang.kernels.ops.layernorm.mhc_boundary_hip.rmsnorm_with_sinkhorn:388` |
|   ↳ `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1})` |  |  | 1 | 0.005 | 0.39% | `sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc.hc_boundary:469 → aten::copy_` |
|   ↳ `void aiter::mhc_post_kernel<std::bfloat16_t, 4, 4, 1024, false>(std::bfloat16_t*, std::bfloat16_t*, std::bfloat16_t*, float*, float*, int, int, int, int, int)` |  |  | 1 | 0.004 | 0.31% | `aiter.ops.mhc.mhc_post:781 → aiter::_mhc_post` |
| `attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel` | 1.112 | 13.5% |  |  |  |  |
|   ↳ `_sparse_mla_BLOCK_M_16_BLOCK_K_64_HEAD_SIZE_512_SPLIT_K_1_MAIN_FMT_FP8_DSV4_MLA_ROPE_SEPARATE_0` |  |  | 38 | 0.399 | 34.56% | `aiter.ops.triton.attention.pa_decode_sparse._pa_decode_sparse_gfx950_gluon:536` |
|   ↳ `bmm_a8w8_mxscale_t16x32x256_w1x2_nb5_sk4_b2_s32_32x32_nt_xcd_sps` |  |  | 40 | 0.208 | 18.03% | `aiter::batched_gemm_a8w8_mxscale_bpreshuffle → aiter.ops.flydsl.batched_gemm_a8w8_gfx950._launch:265` |
|   ↳ `void sglang::fused_k_norm_rope_flashmla<__hip_bfloat16, 512l, 64l, long, 8, (sglang::deepseek_v4::KVLayout)0, false, true>(sglang::FusedKNormRopeFlashMLAParams)` |  |  | 40 | 0.159 | 13.78% | `sglang.kernels.ops.attention.dsv4.norm_rope.fused_k_norm_rope_flashmla:304` |
|   ↳ `_rmsnorm_fake_quant_fp8_kernel` |  |  | 40 | 0.132 | 11.39% | `sglang.kernels.ops.layernorm.rmsnorm_fake_quant_amd_gfx95.rmsnorm_fake_quant_fp8:160` |
|   ↳ `_gfx_sparse_decode_reduce_kernel` |  |  | 38 | 0.116 | 10.03% | `sglang.kernels.ops.attention.sparse_decode_reduce_hip.gfx_sparse_split_reduce:78` |
|   ↳ `_ZN5aiter31inverse_rope_group_quant_kernelIDF16bDF16bLi512ELi64ELi32ELi4ELi1ELNS_11ScaleLayoutE0ELNS_9TierOrderE0ELb0ELb0EEEvPKT_PDB8_PhPKlPKT0_SD_iiiiiillliiibbii` |  |  | 40 | 0.104 | 9.01% | `aiter.ops.inverse_rope_group_quant.inverse_rope_group_quant:91 → aiter::_inverse_rope_group_quant_kernel` |
|   ↳ `_mask_indices_by_length_kernel` |  |  | 8 | 0.019 | 1.62% | `sglang.kernels.ops.attention.dsv4.attn_glue_hip.mask_indices_by_length:57` |
|   ↳ `_sparse_mla_BLOCK_M_16_BLOCK_K_64_HEAD_SIZE_512_SPLIT_K_0_MAIN_FMT_FP8_DSV4_MLA_ROPE_SEPARATE_0` |  |  | 2 | 0.018 | 1.58% | `aiter.ops.triton.attention.pa_decode_sparse._pa_decode_sparse_gfx950_gluon:536` |
| `mlp.<rest> :: shared+routed add, post-expert collective` | 0.719 | 8.7% |  |  |  |  |
|   ↳ `void aiter::cross_device_reduce_2stage<std::bfloat16_t, 4, false, false>(aiter::RankData*, aiter::RankData*, aiter::RankSignals, aiter::Signal*, std::bfloat16_t*, int, int)` |  |  | 40 | 9.780 | 97.42% | `sglang::outplace_all_reduce → aiter.dist.device_communicators.custom_all_reduce.all_reduce:1662` |
|   ↳ `Memcpy DtoD (Device -> Device)` |  |  | 40 | 0.132 | 1.31% | `sglang::outplace_all_reduce → aiter.dist.device_communicators.custom_all_reduce.all_reduce:1662` |
|   ↳ `_router_gemv_split_k_kernel` |  |  | 40 | 0.127 | 1.27% | `sglang.kernels.ops.gemm.router_gemv_hip.rocm_router_gemv_split_k:77` |
| `attn.wo_b` | 0.554 | 6.7% |  |  |  |  |
|   ↳ `void aiter::cross_device_reduce_2stage<std::bfloat16_t, 4, false, false>(aiter::RankData*, aiter::RankData*, aiter::RankSignals, aiter::Signal*, std::bfloat16_t*, int, int)` |  |  | 40 | 7.864 | 95.89% | `sglang::outplace_all_reduce → aiter.dist.device_communicators.custom_all_reduce.all_reduce:1662` |
|   ↳ `_gemm_afp8wfp8_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_32_BLOCK_SIZE_K_1024_GROUP_SIZE_M_1_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_FUSED_SPLITS_1_N_FIRST_1_num_warps_2_num_stages_3_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_2048` |  |  | 40 | 0.165 | 2.01% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
|   ↳ `_mxfp8_quant_kernel` |  |  | 40 | 0.092 | 1.13% | `sglang.kernels.ops.quantization.mxfp8_amd_gfx95._mxfp8_e4m3_quantize_triton:63` |
|   ↳ `Memcpy DtoD (Device -> Device)` |  |  | 40 | 0.080 | 0.98% | `sglang::outplace_all_reduce → aiter.dist.device_communicators.custom_all_reduce.all_reduce:1662` |
| `attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.245 | 3.0% |  |  |  |  |
|   ↳ `void sglang::topk_mode_1::topk_split_select<(sglang::topk_mode_1::TopKMode)2>(sglang::topk_mode_1::TopKPagedParams, sglang::topk_mode_1::SplitWorkspace)` |  |  | 5 | 0.061 | 22.31% | `sglang.kernels.ops.attention.dsv4.topk.topk_transform_paged_v2:325` |
|   ↳ `pa_mqa_logits_fp4_rowgroup_kernel_0` |  |  | 5 | 0.058 | 21.29% | `aiter.ops.flydsl.kernels.tensor_shim._run_compiled:305` |
|   ↳ `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB2048_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM3_NTn1_NTA4_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK5_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGRO0_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4_WGMXCCn1` |  |  | 3 | 0.031 | 11.13% | `sglang.kernels.ops.gemm.bf16_fp32._linear_bf16_fp32_cublas:108 → aten::mm` |
|   ↳ `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x256_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA1_DTLB1_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB1_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA32_LPB32_LPMXSA0_LPMXSB0_LPM0_LRVWn1_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA4_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS12_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR4_PLR1_PKA1_SGROB1_SIA3_SS0_SPO1_SRVW0_SSO1_SVW4_SK5_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN1_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS2_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGRO0_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_1_WGMXCCn1` |  |  | 4 | 0.021 | 7.56% | `torch.nn.modules.linear.forward:130 → aten::mm` |
|   ↳ `_index_q_pack_weights_kernel` |  |  | 5 | 0.021 | 7.50% | `sglang.kernels.ops.attention.dsv4.fp4_indexer_hip.index_q_rope_pack_weights_flydsl:1119` |
|   ↳ `_gemm_afp8wfp8_packed_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_16_BLOCK_SIZE_K_512_K_PACK_2_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_SPLITK_BLOCK_SIZE_2048_FUSED_SPLITS_1_B_CACHE_MODIFIER_NONE_N_4096_K_1280_LAUNCH_OPTIONS_NUM_WARPS_2_NUM_STAGES_3_WAVES_PER_EU_1_MATRIX_INSTR_NONKDIM_16` |  |  | 5 | 0.018 | 6.55% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
|   ↳ `_router_gemv_split_k_kernel` |  |  | 5 | 0.013 | 4.92% | `sglang.kernels.ops.gemm.router_gemv_hip.rocm_router_gemv_split_k:77` |
|   ↳ `void sglang::flash_index_k_split_kernel<false, 128l, 64l, 64u, 2u, false, long>(sglang::IndexKSplitParams)` |  |  | 3 | 0.011 | 3.97% | `sglang.kernels.ops.attention.dsv4.fp4_indexer_rope_hip.index_k_norm_rope_pack_store_split:37` |
|   ↳ `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB2048_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM3_NTn1_NTA4_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK5_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGRO0_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4_WGMXCCn1` |  |  | 1 | 0.010 | 3.66% | `torch.nn.modules.linear.forward:130 → aten::mm` |
|   ↳ `sglang::topk_mode_1::topk_split_hist(sglang::topk_mode_1::TopKPagedParams, sglang::topk_mode_1::SplitWorkspace)` |  |  | 5 | 0.009 | 3.35% | `sglang.kernels.ops.attention.dsv4.topk.topk_transform_paged_v2:325` |
|   ↳ `(anonymous namespace)::deepseek_v4_topk_transform_kernel((anonymous namespace)::TopKParams)` |  |  | 3 | 0.007 | 2.58% | `sglang.kernels.ops.attention.dsv4.topk.topk_transform_paged:118 → sgl_kernel::deepseek_v4_topk_transform_512` |
|   ↳ `void sglang::flash_c2_decode_kernel<(sglang::C2Mode)0, 512l, 64l, 7, long, long, (sglang::deepseek_v4::KVLayout)0, false>(sglang::Compress2DecodeParams)` |  |  | 3 | 0.006 | 2.32% | `sglang.kernels.ops.attention.dsv4.low_ratio_compress.c2_decode_norm_rope_store:135` |
|   ↳ `void sglang::flash_index_k_split_kernel<false, 128l, 64l, 64u, 1u, true, long>(sglang::IndexKSplitParams)` |  |  | 1 | 0.004 | 1.55% | `sglang.kernels.ops.attention.dsv4.fp4_indexer_rope_hip.index_k_norm_rope_pack_store_split:37` |
|   ↳ `void sglang::flash_c1_decode_kernel<512l, 64l, 8, long, long, (sglang::deepseek_v4::KVLayout)0, false>(sglang::Compress1DecodeParams)` |  |  | 1 | 0.004 | 1.33% | `sglang.kernels.ops.attention.dsv4.low_ratio_compress.c1_decode_norm_rope_store:52` |
| `attn.wqkv_a` | 0.239 | 2.9% |  |  |  |  |
|   ↳ `_gemm_afp8wfp8_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_64_BLOCK_SIZE_K_512_GROUP_SIZE_M_1_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_FUSED_SPLITS_5_N_FIRST_0_num_warps_8_num_stages_3_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_CG_NUM_KSPLIT_5_SPLITK_BLOCK_SIZE_1024` |  |  | 40 | 0.247 | 100.00% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
| `attn.wq_b` | 0.187 | 2.3% |  |  |  |  |
|   ↳ `_gemm_afp8wfp8_packed_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_32_BLOCK_SIZE_K_1024_K_PACK_1_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_SPLITK_BLOCK_SIZE_2048_FUSED_SPLITS_1_B_CACHE_MODIFIER_CG_N_8192_K_1280_LAUNCH_OPTIONS_NUM_WARPS_2_NUM_STAGES_3_WAVES_PER_EU_1_MATRIX_INSTR_NONKDIM_16` |  |  | 40 | 0.196 | 100.00% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
| `mlp.topk` | 0.151 | 1.8% |  |  |  |  |
|   ↳ `_router_gate_kernel` |  |  | 40 | 0.176 | 100.00% | `sglang.kernels.ops.moe.rocm_router_gate.rocm_router_gate:262` |
| `logits` | 0.092 | 1.1% |  |  |  |  |
|   ↳ `void aiter::allgather_vec<std::bfloat16_t, 4>(aiter::RankData*, aiter::RankSignals, aiter::Signal*, std::bfloat16_t*, int, int)` |  |  | 1 | 0.225 | 76.06% | `sglang::reg_all_gather_into_tensor → aiter.dist.device_communicators.custom_all_reduce.all_gather_unreg:1871` |
|   ↳ `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x16x512_MI16x16x1_SN_LDSB1_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_1_MXLIBL_MXSFNS_MO40_MGRIPM3_NTn1_NTA4_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK5_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGRO0_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4_WGMXCCn1` |  |  | 1 | 0.054 | 18.25% | `sglang.srt.layers.logits_processor._compute_lm_head:1373 → aten::mm` |
|   ↳ `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1})` |  |  | 1 | 0.008 | 2.76% | `sglang.srt.distributed.parallel_state.all_gather:1398 → aten::copy_` |
|   ↳ `Memcpy DtoD (Device -> Device)` |  |  | 1 | 0.004 | 1.49% | `sglang::reg_all_gather_into_tensor → aiter.dist.device_communicators.custom_all_reduce.all_gather_unreg:1871` |
|   ↳ `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>)` |  |  | 1 | 0.004 | 1.45% | `sglang.srt.layers.logits_processor._copy_logits_to_buffer:1569 → aten::copy_` |
| `engram :: Engram n-gram embedding gate (runs before the layer)` | 0.083 | 1.0% |  |  |  |  |
|   ↳ `_gemm_afp8wfp8_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_A_SCALE_K_GROUP_32_B_SCALE_N_GROUP_32_B_SCALE_K_GROUP_32_FUSED_SPLITS_1_N_FIRST_1_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_CG_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_6144` |  |  | 2 | 0.064 | 73.04% | `aiter::gemm_a8w8_blockscale → aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8:114` |
|   ↳ `_engram_gather_kernel` |  |  | 2 | 0.010 | 12.00% | `sglang.kernels.ops.embeddings.engram_gather.engram_gather:129` |
|   ↳ `_engram_gate_kernel` |  |  | 2 | 0.008 | 9.06% | `sglang.kernels.ops.embeddings.engram_gate.fused_engram_gate:49` |
|   ↳ `_mxfp8_quant_kernel` |  |  | 2 | 0.005 | 5.90% | `sglang.kernels.ops.quantization.mxfp8_amd_gfx95._mxfp8_e4m3_quantize_triton:63` |
| `embed_tokens` | 0.008 | 0.1% |  |  |  |  |
|   ↳ `void aiter::cross_device_reduce_2stage<std::bfloat16_t, 4, false, false>(aiter::RankData*, aiter::RankData*, aiter::RankSignals, aiter::Signal*, std::bfloat16_t*, int, int)` |  |  | 1 | 0.032 | 86.91% | `sglang::outplace_all_reduce → aiter.dist.device_communicators.custom_all_reduce.all_reduce:1662` |
|   ↳ `_vocab_parallel_embedding_kernel` |  |  | 1 | 0.003 | 7.26% | `sglang.kernels.ops.embeddings.vocab_parallel_embedding.vocab_parallel_embedding:56` |
|   ↳ `Memcpy DtoD (Device -> Device)` |  |  | 1 | 0.002 | 5.83% | `sglang::outplace_all_reduce → aiter.dist.device_communicators.custom_all_reduce.all_reduce:1662` |
| `engram_hasher` | 0.005 | 0.1% |  |  |  |  |
|   ↳ `_engram_hash_kernel` |  |  | 1 | 0.006 | 100.00% | `sglang.kernels.ops.embeddings.engram_hash._launch_hash_kernel:223` |
| `norm` | 0.002 | 0.0% |  |  |  |  |
|   ↳ `void aiter::add_rmsnorm_quant_kernel<std::bfloat16_t, std::bfloat16_t, 256, 24, false, false, true, 1, false>(std::bfloat16_t*, std::bfloat16_t*, float*, std::bfloat16_t*, std::bfloat16_t*, std::bfloat16_t*, double, bool, int, int, int, int, int, int, int, bool, bool)` |  |  | 1 | 0.003 | 100.00% | `aiter.ops.rmsnorm._rms_norm_fwd_dispatch:401 → aiter::rmsnorm` |
| `hc_head :: final mHC combine` | 0.002 | 0.0% |  |  |  |  |
|   ↳ `_mhc_combine_kernel` |  |  | 1 | 0.002 | 100.00% | `sglang.kernels.ops.layernorm.mhc.mhc_combine:2800` |

Leaf rows only, so the shares add to 100%. A `<rest>` row is its parent minus the children listed under it, and the `::` tag names the work that lands there -- inline code and fused kernels with no module to hook. It is a difference of two measurements rather than a measurement, so any error in the parent or in any child lands in it at full size and reversed sign: a `<rest>` small next to its parent is mostly that error, and can read negative. 

Indented rows list **all** Kineto GPU kernels/copies from rank 0's standalone module replays (one eager sample per layer kind, multiplied by its layer count; globals once). Kernel percentages sum to 100% **within each module**. Parent `<rest>` samples exclude separately measured children through module hooks. The API column comes from the kernel's Kineto runtime correlation, nearest recorded Python frame, and dispatcher external ID. ms/step and share are graph timings; the profile columns are diagnostic.

Packages these APIs were loaded from: `aiter` = `/sgl-workspace/aiter/aiter`, `sglang` = `/sgl-workspace/sglang/python/sglang`, `torch` = `/opt/venv/lib/python3.12/site-packages/torch`.

## Per layer kind -- one step

### moe+c0 -- x1 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.183   | 0.183   |
| `  attn`                                                                                 | 0.044   | 0.044   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.006   |
| `    attn.wq_b`                                                                          | 0.005   | 0.005   |
| `    attn.wo_b`                                                                          | 0.014   | 0.014   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.019   | 0.019   |
| `  mlp`                                                                                  | 0.100   | 0.100   |
| `    mlp.topk`                                                                           | 0.004   | 0.004   |
| `    mlp.experts`                                                                        | 0.087   | 0.087   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | 0.010   | 0.010   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.039   | 0.039   |

### moe+c0+engram -- x1 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.177   | 0.177   |
| `  attn`                                                                                 | 0.044   | 0.044   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.006   |
| `    attn.wq_b`                                                                          | 0.005   | 0.005   |
| `    attn.wo_b`                                                                          | 0.014   | 0.014   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.019   | 0.019   |
| `  mlp`                                                                                  | 0.100   | 0.100   |
| `    mlp.topk`                                                                           | 0.004   | 0.004   |
| `    mlp.experts`                                                                        | 0.086   | 0.086   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | 0.010   | 0.010   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.033   | 0.033   |
| `engram :: Engram n-gram embedding gate (runs before the layer)`                         | 0.042   | 0.042   |

### moe+c2+kvsrc+indexer -- x2 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.221   | 0.442   |
| `  attn`                                                                                 | 0.076   | 0.152   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.012   |
| `    attn.wq_b`                                                                          | 0.005   | 0.010   |
| `    attn.wo_b`                                                                          | 0.014   | 0.028   |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.019   | 0.039   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.032   | 0.064   |
| `  mlp`                                                                                  | 0.107   | 0.215   |
| `    mlp.topk`                                                                           | 0.004   | 0.008   |
| `    mlp.experts`                                                                        | 0.086   | 0.173   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | 0.017   | 0.034   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.038   | 0.075   |

### moe+c2 -- x15 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.195   | 2.928   |
| `  attn`                                                                                 | 0.052   | 0.775   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.090   |
| `    attn.wq_b`                                                                          | 0.005   | 0.071   |
| `    attn.wo_b`                                                                          | 0.014   | 0.208   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.027   | 0.406   |
| `  mlp`                                                                                  | 0.109   | 1.629   |
| `    mlp.topk`                                                                           | 0.004   | 0.057   |
| `    mlp.experts`                                                                        | 0.086   | 1.291   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | 0.019   | 0.281   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.035   | 0.524   |

### moe+c2+kvsrc+indexer+engram -- x1 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.209   | 0.209   |
| `  attn`                                                                                 | 0.075   | 0.075   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.006   |
| `    attn.wq_b`                                                                          | 0.005   | 0.005   |
| `    attn.wo_b`                                                                          | 0.014   | 0.014   |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.019   | 0.019   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.031   | 0.031   |
| `  mlp`                                                                                  | 0.100   | 0.100   |
| `    mlp.topk`                                                                           | 0.004   | 0.004   |
| `    mlp.experts`                                                                        | 0.086   | 0.086   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | 0.010   | 0.010   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.034   | 0.034   |
| `engram :: Engram n-gram embedding gate (runs before the layer)`                         | 0.041   | 0.041   |

### moe+c1+kvsrc+indexer+late -- x1 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.251   | 0.251   |
| `  attn`                                                                                 | 0.114   | 0.114   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.006   |
| `    attn.wq_b`                                                                          | 0.005   | 0.005   |
| `    attn.wo_b`                                                                          | 0.014   | 0.014   |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.052   | 0.052   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.038   | 0.038   |
| `  mlp`                                                                                  | 0.100   | 0.100   |
| `    mlp.topk`                                                                           | 0.004   | 0.004   |
| `    mlp.experts`                                                                        | 0.087   | 0.087   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | 0.009   | 0.009   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.037   | 0.037   |

### moe+c1+late -- x15 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.196   | 2.934   |
| `  attn`                                                                                 | 0.051   | 0.769   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.090   |
| `    attn.wq_b`                                                                          | 0.005   | 0.069   |
| `    attn.wo_b`                                                                          | 0.014   | 0.208   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.027   | 0.402   |
| `  mlp`                                                                                  | 0.110   | 1.644   |
| `    mlp.topk`                                                                           | 0.004   | 0.056   |
| `    mlp.experts`                                                                        | 0.087   | 1.299   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | 0.019   | 0.289   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.035   | 0.520   |

### moe+c1+indexer+late -- x4 layers

| op                                                                                       | ms/call | ms/step |
|------------------------------------------------------------------------------------------|---------|---------|
| `layer`                                                                                  | 0.238   | 0.952   |
| `  attn`                                                                                 | 0.091   | 0.365   |
| `    attn.wqkv_a`                                                                        | 0.006   | 0.024   |
| `    attn.wq_b`                                                                          | 0.005   | 0.019   |
| `    attn.wo_b`                                                                          | 0.014   | 0.055   |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.034   | 0.135   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.033   | 0.133   |
| `  mlp`                                                                                  | 0.109   | 0.436   |
| `    mlp.topk`                                                                           | 0.004   | 0.015   |
| `    mlp.experts`                                                                        | 0.086   | 0.345   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | 0.019   | 0.075   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.038   | 0.152   |

## Roll-up -- one step

| op                                 | x  | ms/call | ms/step |
|------------------------------------|----|---------|---------|
| moe+c0 layer                       | 1  | 0.183   | 0.183   |
| moe+c0+engram layer                | 1  | 0.177   | 0.177   |
| moe+c0+engram engram               | 1  | 0.042   | 0.042   |
| moe+c2+kvsrc+indexer layer         | 2  | 0.221   | 0.442   |
| moe+c2 layer                       | 15 | 0.195   | 2.928   |
| moe+c2+kvsrc+indexer+engram layer  | 1  | 0.209   | 0.209   |
| moe+c2+kvsrc+indexer+engram engram | 1  | 0.041   | 0.041   |
| moe+c1+kvsrc+indexer+late layer    | 1  | 0.251   | 0.251   |
| moe+c1+late layer                  | 15 | 0.196   | 2.934   |
| moe+c1+indexer+late layer          | 4  | 0.238   | 0.952   |
| embed_tokens                       | 1  | 0.008   | 0.008   |
| engram_hasher                      | 1  | 0.005   | 0.005   |
| hc_head                            | 1  | 0.002   | 0.002   |
| norm                               | 1  | 0.002   | 0.002   |
| logits                             | 1  | 0.092   | 0.092   |

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
