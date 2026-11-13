# CODE-MAP: GLM-5.2-MXFP4 decode step on MI355X (TP8 / attention-DP8, EAGLE 3/1/4)

Read-only code study, 2026-11-12. Every row cites `file:line`; quotes are verbatim fragments.
"UNCONFIRMED" marks what the provided trees cannot prove.

## 0. Path legend and what actually runs

| Tag | Absolute path |
|---|---|
| `S/` | `/perf_apps/liyingli/bench_agentx/kreview-20261112/src/sglang/srt/` |
| `K/` | `/perf_apps/liyingli/bench_agentx/kreview-20261112/src/sglang/kernels/` |
| `A/` | `/perf_apps/liyingli/bench_agentx/kreview-20261112/src/aiter/` (the `aiter` python package root) |
| `FIX/` | `/perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/patches/fix/` |
| `OVK/` | `/perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/kit/overlay-kopt/` |
| `OV/` | `/perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/kit/overlay/` |
| `CFG/` | `/perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/config/` |
| `W` | `/perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/kit/engine/worker.sh` |
| `M/` | `/perf_apps/xiaobo/models/GLM-5.2-MXFP4/` |

The decode container does not run `S/` unmodified:

| Replacement | Evidence |
|---|---|
| 6 SGLang files are bind-mounted from `FIX/` (`graph_variants.py`, `decode_cuda_graph_runner.py`, `dp_attn.py`, `schedule_batch.py`, `forward_batch_info.py`, `two_batch_overlap.py`). `S/` holds the originals (`cmp` equal to `patches/orig/`). Line numbers below for these files are from `FIX/`. | `CFG/fix-mounts.sh:5-11` |
| `models/glm4_moe.py` is mounted from `/perf_apps/liyingli/bench_agentx/opt-aus-20261103/ptpc-fix/glm4_moe.py` (draft MoE runs as PTPC-FP8). | `CFG/arm-C-prod-ptpcfix.env:5` |
| KF overlay on `PYTHONPATH`: production arms mount `OV/`, MegaMoeTP arms mount `OVK/`. The two differ only in `sitecustomize.py` (kopt_megatp hook), `yihou_kf/__init__.py:251` (M_HI env) and `yihou_kf/kf_indexer.py:1757-1759` (group env). KF line numbers below are `OVK/`. | `CFG/recipe-e10.env:15-16`, `CFG/arm-C-1p1d-pp8-megatp.env:6` |
| Decode flags, later value wins: worker defaults `--speculative-num-steps 5 … --speculative-num-draft-tokens 6`, `--mem-fraction-static 0.92`, then `DECODE_EXTRA_ARGS` appends `--mem-fraction-static 0.95 … --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` and `index_share_for_mtp_iteration:true`. | `W:77-78`, `W:111-117`, `CFG/recipe-e10.env:12` |
| Topology: `--tp-size 8 --dp-size 8 --enable-dp-attention --cuda-graph-max-bs-decode 256`, `--max-running-requests 256`, `--ep-size 1`, `--kv-cache-dtype fp8_e4m3 --dsa-decode-backend triton`. `DECODE_CUSTOM_AR=1` drops `--disable-custom-all-reduce`. | `W:66-69`, `W:111-112`, `W:129-131`, `CFG/recipe-e10.env:17` |
| Env: `YIHOU_KF_SPARSE=1 YIHOU_KF_INDEXER=1 YIHOU_KF_MOE=1 SGLANG_OPT_USE_TOPK_V2=false SGLANG_GLM_NEXTN_MOE_PTPC=1`, `SGLANG_USE_ROCM700A=0`, `SGLANG_DP_USE_GATHERV=1`. `YIHOU_KF_FUSED_INDEXER`, `DSA_LPT_GROUP_ROWS`, `DSA_LPT_NG2` are not set in the listed chain. | `CFG/recipe-e10.env:16`, `CFG/arm-C-prod.env:5`, `W:42` |

Derived widths: `attn_dp_size=8` (`S/arg_groups/parallel_hook.py:56` "attn_dp_size=cfg.dp_size"), `attn_tp_size = 8//8//1 = 1`
(`S/runtime_context.py:76`), `moe_tp_size = 8//1//1 = 8` (`S/runtime_context.py:154`).

## 1. Scheduler step with EAGLE (K=3 steps, topk 1, 4 draft tokens)

| Item | What the code does | Evidence |
|---|---|---|
| Worker | EAGLE -> `EAGLEWorkerV2`. | `S/speculative/spec_info.py:348-351` ("return EAGLEWorkerV2") |
| Order per decode step | `draft` span -> `self.verify(...)` -> `draft_extend` span. | `S/speculative/eagle_worker_v2.py:1408-1414`, `:1417-1421`, `:1431-1439` |
| `draft` graph | One graph holding the whole loop: `for i in range(self.speculative_num_steps)` with `if i == self.speculative_num_steps - 1: break` before the forward -> **2 NextN forwards**. Rows per request = `speculative_eagle_topk` = **1**. Mode DECODE. | `S/speculative/eagle_worker_v2.py:762-802`; `S/speculative/eagle_draft_cuda_graph_runner.py:152,158`; `S/speculative/spec_utils.py:116-117` |
| `step[VERIFY bs=N]` | Target model (78 layers) in `ForwardMode.TARGET_VERIFY`; rows per request = `num_draft_tokens` = **4** (`return num_draft_tokens`); `raw_num_token = raw_bs * self.captured_req_width`. | `FIX/model_executor/runner/decode_cuda_graph_runner.py:303-305,311,1469`; `S/speculative/spec_info.py:290-300` |
| `draft_extend` graph | **1 NextN forward**, mode `DRAFT_EXTEND_V2`, rows per request = `speculative_num_draft_tokens` = **4** ("Draft-extend fills the whole tree width (num_draft_tokens) per req"); lm_head rows are selected by `select_index` (one per request). | `S/speculative/eagle_worker_v2.py:1098-1101,1150-1153,1188-1189`; `S/speculative/eagle_draft_extend_cuda_graph_runner.py:113,153`; `S/speculative/spec_utils.py:118-119` |
| Meaning of `bs=N` | The span is built before any graph padding from `bs = forward_batch.batch_size`, and `batch_size=len(batch.seq_lens)` -> **this rank's own requests** in the verify batch, not the padded/captured batch. The graph path does not touch the batch before replay ("Only the decode cuda-graph path above pre-pads its static buffers and returns early"). | `S/utils/profile_utils.py:486-497`; `S/model_executor/model_runner.py:1818-1824,1981-1994`; `FIX/model_executor/forward_batch_info.py:1022` |
| Draft indexer reuse | `index_share_for_mtp_iteration and self.topk == 1` -> `seed_dsa_topk_from_draft_extend`: the draft-extend indexer top-k at the last accepted row seeds the next `draft`; in `draft` the NextN layer gets `prev_topk_indices` and `should_run_indexer` returns False (`not self.skip_topk or (self.is_nextn and prev_topk_indices is None)`). The target layers do not read this flag. | `S/speculative/eagle_worker_v2.py:285-300,757-761,1171-1177`; `S/models/deepseek_common/attention_forward_methods/forward_mla.py:165` |
| NextN layer | `DeepseekV2DecoderLayer(config, 0, …, is_nextn=True)`; always sparse (`return is_nextn or (…)`); own indexer built (`if not self.skip_topk or is_nextn`). Checkpoint layer 78 is entirely BF16; routed experts are cast to per-channel FP8 at load. | `S/models/deepseek_nextn.py:126-131`; `S/models/deepseek_v2.py:2113,2762`; ptpc-fix `glm4_moe.py:1322-1356` (`_maybe_quant_glm_nextn_moe_to_ptpc`) |

## 2. DP attention and CUDA-graph batch selection

| Item | What the code does | Evidence |
|---|---|---|
| Generated list (spec decoding, `max_bs=256`, `--cuda-graph-bs` unset) | `list(range(1, 9, 1)) + list(range(10, 33, 2)) + list(range(40, 65, 4)) + list(range(72, 257, 8)) + list(range(272, max_bs + 1, 16))`, filtered `bs <= max_bs` -> 1..8, 10..32 step 2, 40..64 step 4, 72..256 step 8 (44 values). | `S/arg_groups/cuda_graph_hook.py:517-532`; called at `S/arg_groups/memory_hook.py:203` |
| List actually captured | Runner filters `capture_bs = [bs for bs in capture_bs if bs <= num_max_requests]` with `num_max_requests = model_runner.req_to_token_pool.size`; per DP worker `requested_per_worker = max_num_reqs // self.attn_dp_size` = 256//8 = **32** -> buckets **1,2,3,4,5,6,7,8,10,12,…,32** (20 graphs). UNCONFIRMED only in that the `infera.engine.sglang` wrapper could change `--max-running-requests`. | `S/model_executor/runner/base_cuda_graph_runner.py:77-102`; `S/mem_cache/kv_cache_configurator.py:2916-2919` |
| Bucket choice | `if self.require_mlp_tp_gather: max_batch_size = self._max_dp_batch_size(forward_batch); bs = self._pad_to_bucket(max_batch_size, self.capture_bs)`; `_max_dp_batch_size` = `max(request_counts)` over `original_global_num_tokens_cpu` (per-rank request counts from the MLP sync: `num_tokens = local_batch.batch_size()`). `_pad_to_bucket` = smallest bucket >= value. | `FIX/model_executor/runner/decode_cuda_graph_runner.py:700-707,1470-1475`; `FIX/model_executor/forward_batch_info.py:931`; `FIX/managers/scheduler_components/dp_attn.py:571`; `S/model_executor/runner/base_cuda_graph_runner.py:166-180` |
| `require_mlp_tp_gather` | True: attention DP on and `moe_dense_tp_size is None`. | `S/utils/common.py:4155,4166-4169` |
| Same bucket on every rank | Yes. Graph batch `bs_g = bucket(max_r N_r)`; every rank replays `padded_num_tokens = bs * self.captured_req_width` = `4*bs_g` rows; the DP sizes baked in are `[num_tokens] * self.num_dp_ranks`, `global_dp_buffer_len = sum(...)`. | `FIX/…/decode_cuda_graph_runner.py:604-609,1018-1021,1475,1525` |
| Padding mode in graphs | `get_default_mode_in_cuda_graph()` returns MAX_LEN unless `SGLANG_USE_ROCM700A` is set (it is set to 0) -> **MAX_LEN**. Gather = `tp_group.all_gather_into_tensor(global_tokens, local_tokens)` (attn_tp_size == 1); return = `tp_group.reduce_scatter_tensor(output, input)`. `SGLANG_DP_USE_GATHERV` is inert here (`and not _DpGatheredBufferWrapper.is_dp_max_padding()`). | `S/layers/dp_attention.py:139,195-201,688-697,871-877,1096-1097`; `FIX/…/decode_cuda_graph_runner.py:1079,1271-1276` |
| Row counts | Per-rank attention rows = `4 * bs_g`. Global MoE rows **M = 8 * 4 * bs_g = 32 * bs_g** (128 … 1024). |  derived from the rows above |
| Pad rows in the MoE | Not masked: `moe_num_token_non_padded` returns None under attention DP ("dp_gather concatenates one padded slot per attention-DP rank"); the dispatcher's pad mask is under `elif not self.use_aiter_moe_runner:`. All M rows are routed and computed. | `FIX/model_executor/forward_batch_info.py:1377-1379`; `S/layers/moe/token_dispatcher/standard.py:235-248` |
| Pad requests in attention | Padded requests carry `seq_len` fill value 1 (`return 1`), so their 4 rows have causal length 2..5. | `S/layers/attention/dsa_backend.py:4006-4008`; `:1682-1684,1713-1719` |
| Variant | With the fix, a TARGET_VERIFY capture has one graph per bucket ("a target-verify capture runs the full indexer in both, so one graph serves"). | `/perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/patches/fix.diff` (graph_variants.py hunk) |

## 3. Weights: placement, precision, bytes

Checkpoint dtypes are from the safetensors headers in `M/` (header-only read) and `M/config.json`
`quantization_config.exclude` (1319 entries: `lm_head`, every `self_attn.*`, `indexer.*`, dense `mlp.*_proj` of layers 0-2,
every `mlp.gate`, all of layer 78). MXFP4 (`dtype: fp4`, `group_size: 32`, `scale_format: e8m0`) applies only to
`mlp.experts.*` and `mlp.shared_experts.*` of layers 3-77.

| Weight | Module / group | Shape per rank | Precision as run | Bytes per rank | Evidence |
|---|---|---|---|---|---|
| `fused_qkv_a_proj_with_mqa` | `ReplicatedLinear` | [2624, 6144] | BF16 | 32,243,712 | `S/models/deepseek_v2.py:2066-2072` |
| `q_b_proj` | `ColumnParallelLinear(parallel_group="attn_tp")`, attn_tp=1 | [16384, 2048] | BF16 | 67,108,864 | `S/models/deepseek_v2.py:2074-2081` |
| `w_kc` (from `kv_b_proj`) | split of [28672, 512] | [64, 192, 512] | **FP8 e4m3, per-tensor scale** ("GLM ships kv_b_proj as bf16 … Quantize to per-tensor e4m3fn") | 6,291,456 | `S/models/deepseek_common/deepseek_weight_loader.py:724-736,752-754` |
| `w_vc` (from `kv_b_proj`) | same | [64, 512, 256] | FP8 e4m3 | 8,388,608 | same, `:755-758` |
| `o_proj` | `RowParallelLinear(parallel_group="attn_tp")` | [6144, 16384] | BF16 | 201,326,592 | `S/models/deepseek_v2.py:2150-2158` |
| `indexer.wq_b` (21 layers + NextN) | `ReplicatedLinear` | [4096, 2048] | BF16 | 16,777,216 | `S/layers/attention/dsa/dsa_indexer.py:337-343` |
| `indexer.wk` | `ReplicatedLinear` | [128, 6144] | BF16 | 1,572,864 | `:354-360` |
| `indexer.weights_proj` | `ReplicatedLinear`, `params_dtype=torch.bfloat16` | [32, 6144] | BF16 | 393,216 | `:361-367` |
| RMSNorm weights | input / post-attn / final | [6144] | BF16 | 12,288 | `S/models/deepseek_v2.py:2652-2655,2979` |
| MoE `gate.weight` | `nn.Parameter` (replicated) | [256, 6144] | BF16 (`e_score_correction_bias` F32) | 3,145,728 | `S/models/deepseek_v2.py:503-512,515-524` |
| MoE `w13_weight` (gate+up) | `intermediate_size // self.moe_tp_size` = 256 | [257, 512, 3072] u8 | **MXFP4** packed | 404,226,048 (1,572,864 per expert) | `S/layers/moe/fused_moe_triton/layer.py:410`; `S/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py:297-312` |
| MoE `w13_weight_scale` | | [257, 512, 192] u8 | E8M0 | 25,264,128 (98,304 per expert) | same, `:342-350` |
| MoE `w2_weight` (down) | | [257, 6144, 128] u8 | MXFP4 packed | 202,113,024 (786,432 per expert) | same, `:318-334` |
| MoE `w2_weight_scale` | | [257, 6144, 8] u8 | E8M0 | 12,633,088 (49,152 per expert) | same, `:355-363` |
| Dense MLP `gate_up_proj` (layers 0-2) | `parallel_group = "replicated" if is_dense_ffn_fully_dp() else "tp"` -> tp (moe_dense_tp_size None) | [3072, 6144] | BF16 | 37,748,736 | `S/models/deepseek_v2.py:2639-2650,299-306`; `S/layers/layer_boundary/layout.py:83-84` |
| Dense MLP `down_proj` | TP8 | [6144, 1536] | BF16 | 18,874,368 | `S/models/deepseek_v2.py:307-315` |
| `lm_head` | `ParallelLMHead(…, use_attn_tp_group=get_parallel().enable_dp_lm_head)` -> full TP8 group | [19360, 6144] | BF16 | 237,895,680 | `S/models/deepseek_v2.py:3264-3270` |
| `embed_tokens` | vocab-parallel inside the attention-TP group (size 1) -> full table | [154880, 6144] | BF16 | 12,288 per looked-up row | `S/layers/vocab_parallel_embedding.py:185` |

Confirmed claims: 256 routed + 1 fused shared expert (`num_experts_for_moe = config.n_routed_experts + self.num_fused_shared_experts`,
`top_k_for_moe = config.num_experts_per_tok + self.num_fused_shared_experts` -> 257 / 9; `S/models/deepseek_v2.py:675-678`), per-rank
intermediate 2048/8 = 256, `first_k_dense_replace` 3 (`layer_id >= config.first_k_dense_replace`, `S/models/deepseek_v2.py:2762-2766`;
`M/config.json` `mlp_layer_types`: 3 dense + 75 sparse). Attention and indexer weights are full copies on every DP rank; only the
MoE intermediate dim, the dense MLP and the lm_head vocab are split by TP8. A dense GEMM reads its whole weight once per call
for any row count >= 1; MoE GEMMs read only the experts that received rows (bytes per expert above; all 257: 429.5 MB gemm1,
214.7 MB gemm2).

Per-token caches: KV cache 576 B per token per layer, FP8 ("nope(512 fp8) + rope(64 fp8), without extra per-block scales",
`S/mem_cache/kv_cache_configurator.py:3231-3237`); index-K 132 B per token (128 FP8 + 4 B fp32 scale, `head_dim_with_sf = 132`,
`S/layers/attention/dsa/dsa_indexer.py:1317`) allocated only for non-shared layers (`return not self.skip_topk_layers[local_layer_idx]`,
`S/mem_cache/memory_pool.py:5264-5265`), in 64-token pages (`PAGE_SIZE = 64`, `CACHE_TOK_STRIDE = 132`,
`K/ops/attention/dsa/hip_gfx950/fused_decode.py:26-27`).

## 4. One verify layer in execution order (kernel names from `seq_2p1d_c208_r0.txt`)

R = `4*bs_g` rows on this rank, M = `32*bs_g`. Attention method: `handle_attention_dsa` -> MLA -> `MLA_ROCM`
(`S/models/deepseek_common/attention_backend_handler.py:36,206-217`).

| # | Op | Kernel in the trace | Input x weight -> output | Evidence |
|---|---|---|---|---|
| 0 | Embedding (layer 0 only) + zero allocator | `vectorized_gather_kernel`, `FillFunctor<float>` | R rows of [154880, 6144] | `S/models/deepseek_v2.py:3080,3105-3109` |
| 1 | `input_layernorm` (+ add of previous FFN output; layer 0 without add) | `aiter::add_rmsnorm_quant_kernel` | [R, 6144] | `S/layers/layer_boundary/residual/add_norm.py:187-192`; AR fusion excluded by `and not is_dp_attention_enabled()` (`:182`) |
| 2 | `fused_qkv_a_proj_with_mqa` | `hgemm_bf16_t64x64x64x6_ksd_w2x4x1` | [R, 6144] x [2624, 6144] | `S/models/deepseek_v2.py:2531`; dispatch `tgemm.mm(x, layer.weight, bias, otype=x.dtype)` `S/layers/quantization/unquant.py:589` |
| 3 | `q_a_layernorm` + `kv_a_layernorm` | `aiter::fused_qk_rmsnorm_kernel` | [R, 2048], [R, 512] | `S/models/…/forward_mla_rocm.py:587-595` |
| 4 | `q_b_proj` | `hgemm_bf16_t128x64x64x4_ks1_w4x1x2` | [R, 2048] x [16384, 2048] -> [R, 64, 256] | `forward_mla_rocm.py:642`; `S/models/deepseek_v2.py:2544-2545` |
| 5 | Indexer `wq_b` (full layers) | `hgemm_bf16_t64x32x128x4_ks1_w4x1x1` | [R, 2048] x [4096, 2048] -> [R, 32, 128] | `S/layers/attention/dsa/dsa_indexer.py:922-923` |
| 6 | Indexer `wk` | `hgemm_bf16_t16x32x128x7_ksd_w1x2x1` | [R, 6144] x [128, 6144] | `:930` |
| 7 | Indexer `k_norm` | ck_tile `Layernorm2dFwd` | [R, 128] | `:931` |
| 8 | Indexer RoPE (q, k) | `aiter::kn_entry_2c_sbhd_cached_indirect_inplace` | 64 rope dims | `:936` |
| 9 | Hadamard on q and k | `fast_hadamard_transform_kernel` x2 | | `:964-965`, `:245-247` |
| 10 | `act_quant(query)` -> q_fp8 [R, 32, 128] + scale | `act_quant_kernel__kernel` | | `:2162`; `K/ops/attention/dsa/tilelang_kernel.py:104` |
| 11 | Index-K quant + cache write (132 B per row) | `aiter::indexer_k_quant_and_cache_kernel` | | `:1905-1919` |
| 12 | `weights_proj` + head-gate scale | `hgemm_bf16_t16x32x128x6_ksd_w1x2x1`, `triton_poi_fused_mul_unsqueeze_0` | [R, 6144] x [32, 6144] | `:2241`, `:577-584` |
| 13 | Paged MQA logits | `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` (KF copy, same name as aiter's) | q_fp8 x index-K pages -> fp32 [R, max_ctx] | `:1408-1417`; `OVK/yihou_kf/__init__.py:169-185` |
| 14 | Top-k 2048 + slot transform | `_topk_est`, `_topk_collect`, `_topk_select` (KF) | -> int32 [R, 2048] slots, -1 padded | `:1488`; `OVK/yihou_kf/kf_indexer.py:2142-2151` |
| 15 | W_kc absorb + RoPE + concat + KV write (one kernel) | `_fused_fp8_bmm_rope_cat_and_cache_mla_kernel` | q_nope [R, 64, 192] x w_kc [64, 192, 512] FP8 -> q [R, 64, 576] FP8; writes 576 B per row to the KV cache | `forward_mla_rocm.py:444-454,678-682,839-847,476-492` |
| 16 | Sparse MLA | `_sparse_mla_split_k_kernel`, `_sparse_mla_reduce_dq_kernel` (KF; in-tree reduce is `_sparse_mla_reduce_kernel`) | gathers <= 2048 KV rows of 576 B per query row -> [R, 64, 512] bf16 | `S/layers/attention/dsa_backend.py:2186-2193,2362-2381`; `OVK/yihou_kf/kf_sparse_mla.py:1692,1739` |
| 17 | W_vc absorb | `_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel` | [R, 64, 512] x w_vc [64, 512, 256] FP8 -> [R, 16384] | `forward_mla_rocm.py:281-303,1077` |
| 18 | `o_proj` | `Cijk_Alik_Bljk_BBS_BH_…MT160x96x64…` + `Cijk_SB_BiasS_HAS_ScaleAlphaVec_PostGSU8_VW4` | [R, 16384] x [6144, 16384] | `forward_mla_rocm.py:1091` (second kernel being part of the same hipBLASLt call: UNCONFIRMED, library source not in tree) |
| 19 | `post_attention_layernorm` (+ add of attention output), on local rows before the gather | `aiter::add_rmsnorm_quant_kernel` | [R, 6144] | `S/layers/layer_boundary/prepare.py:183-225` (`read.update_and_read` then `dp_gather`) |
| 20 | DP all-gather | `aiter::allgather_vec` | [R, 6144] -> [M, 6144] bf16 | `S/layers/dp_attention.py:696`; `S/distributed/parallel_state.py:1297-1323` |
| 21 | Router gate | `hgemm_bf16_t64x64x64x6_ksd_w4x2x1` | [M, 6144] x [256, 6144] | `S/models/deepseek_v2.py:1303-1309,579-580` |
| 22 | Top-k 8 of 256 | `aiter::topk_reg_kernel` | | `S/models/deepseek_v2.py:1328-1334`; `S/layers/moe/topk.py:233,1141` |
| 23 | Append shared expert id 256 -> topk 9 | `_fused_append_shared_experts_kernel` | | `K/ops/moe/fused_moe_triton_kernels.py:1284,1364` |
| 24 | Sort (aiter path) | `opus_moe_sorting` P0_v2 + P23 | 9*M pairs | `OVK/yihou_kf/kf_moe.py:14-21` (kernel list of the aiter path) |
| 25 | Activation MXFP4 quant + scale scatter | `aiter::dynamic_per_group_scaled_quant_kernel<bf16, fp4>`, `aiter::mxfp4_moe_sort_kernel` | [M, 6144] | same |
| 26 | gemm1 (+ SiLU(gate)*up + MXFP4 quant) | `mfma_moe1_silu_mul_afp4_wfp4_fp4_t64x64x256…` | sorted rows x w13 [e, 512, 3072] | same; tuned row in section 6 |
| 27 | gemm2 (x top-k weight) | `gemm2_a4w4_port_…bm64_bn128_bk256_nt_reduce_tk9…` | x w2 [e, 6144, 128] | same |
| 28 | Sum of the 9 outputs (only for "reduce" gemm2 instances) | `moe_reduction_bf16_bf16_t9_n6144…` | -> [M, 6144] | same |
| 29 | Reduce-scatter (completes the TP sum, returns this rank's rows) | `aiter::reduce_scatter_split_first_dim` | [M, 6144] -> [R, 6144] | `S/layers/layer_boundary/exit.py:76-94`; `S/layers/dp_attention.py:1096-1097`; `S/distributed/parallel_state.py:1170-1207` |

Variants of the table:

| Case | Difference | Evidence |
|---|---|---|
| Shared layers (57 of 78) | Steps 5-14 absent: no indexer module, no index-K write, no logits, no top-k; the layer reuses `prev_topk_indices` (`maybe_capture_indexer_topk(self.layer_id, prev_topk_indices)`) and still runs steps 15-16 on its own KV buffer. | `S/models/deepseek_v2.py:2110-2113`; `forward_mla_rocm.py:645-656,1097-1100` |
| Which layers are full | `indexer_types` takes priority over freq/offset: `indexer_types[layer_id] == "shared"`. Full = 0,1,2,6,10,…,74 (21 layers); `index_topk_freq 4` / `index_skip_topk_offset 3` give the same set but are not read. `index_topk 2048`. | `S/configs/model_config.py:414-418,432-442`; `M/config.json` |
| Dense layers 0-2 | Steps 21-28 replaced by `gate_up_proj` GEMM (`Cijk…MT256x144x64` + `PostGSU4`), `act_and_mul_kernel` (SiLU), `down_proj` GEMM (`Cijk…MT192x96x128`); gather and reduce-scatter unchanged. | `S/models/deepseek_v2.py:386,486-487` |
| Rows <= 48 | Steps 5-14 replaced by SGLang's in-tree 4-kernel indexer: `dual_gemv_kernel`, `qk_rope_hadamard_quant_kernel`, `logits_hist_m<8, 12>`, `k_scatter`. | `S/layers/attention/dsa/dsa_indexer.py:2033-2047`; `K/ops/attention/dsa/hip_gfx950/fused_decode.py:237-327` |
| KF MoE (97 <= M <= 455) | Steps 24-28 become `glm52_moe_sort_fill`, `gemm1_a4w4_port_fp4_h6144_i256_ne257_bm16_…_xcd4`, `gemm2_a4w4_port_…bm16_bn128_bk128_atomic…` (names from `peek_koptpp8_c208.txt`). | section 6 |

Tail after layer 77 (names from `seq_megatp_c160_r3.txt`):

| # | Op | Kernel | Evidence |
|---|---|---|---|
| T1 | Final `norm` (+ add of the last FFN output) | `aiter::add_rmsnorm_quant_kernel` | `S/models/deepseek_v2.py:3211-3215` |
| T2 | DP gather of hidden states for the lm_head, done as zero-fill + own rows + all-reduce: `LogitsMetadata` hard-codes `dp_padding_mode=DpPaddingMode.SUM_LEN`; `global_tokens.fill_(0)`, `memcpy(...)`, `tensor_model_parallel_all_reduce(global_tokens)`. Buffer rows = M. | `rocprim … scan` (cumsum), `FillFunctor<BFloat16>`, `memcpy_triton_kernel`, `aiter::cross_device_reduce_2stage` | `S/layers/logits_processor.py:412,417-436,1146`; `S/layers/dp_attention.py:634,650-652,677` |
| T3 | lm_head GEMM `torch.matmul(hidden_states.to(lm_head.weight.dtype), lm_head.weight.T)` | `Cijk…MT192x320x64…` : [M, 6144] x [19360, 6144] | `S/layers/logits_processor.py:1106-1108` |
| T4 | TP all-to-all (default on a decode-only node: `view.disaggregation_mode == "decode" and view.attn_dp_size > 1 and view.tp_size == view.attn_dp_size`) | `ncclDevKernel_Generic_1` | `S/arg_groups/overrides.py:1415-1422`; `S/layers/logits_processor.py:943-948,1218-1225`; `S/distributed/parallel_state.py:1211-1215` |
| T5 | Reassemble + copy to the fp32 logits buffer | `direct_copy` kernels, `bfloat16tofloat32_copy` | `S/layers/logits_processor.py:1476-1494,1267` |

## 5. Context-length scaling

L_r = causal length of row r = request length + 1..4 (`torch.arange(kv_len - qo_len + 1, kv_len + 1)`,
`S/layers/attention/dsa_backend.py:1713-1719`).

| Kernel | Iterates over | Reads per row | Evidence |
|---|---|---|---|
| `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` (KF) | grid `batch_size * next_n * SplitKV`; each program takes one row and a slice of `tl.cdiv(context_length, ChunkK)` 256-token chunks, walking that row's page table in 64-token pages. All cached tokens of the request up to L_r. | `ceil(L_r/64)` pages x 8448 B (= 132 B per token); q 32 x 128 FP8, 32 fp32 head weights; writes 4 B x L_r logits | `OVK/yihou_kf/kf_indexer.py:472-490,1792,1857`; layout comment `:2210-2211` |
| `_gluon_paged_mqa_logits_fused2` (KF) | pairs of consecutive rows (`r0 = g_sel * GROUP + 2 * p_pair`, `r1 = r0 + 1`); one pass over the pages when both rows have the same page ids (`fbi` 0), else two passes; bounded by `Lmax` of the pair. | same pages, once per pair when fused | `OVK/yihou_kf/kf_indexer.py:1282-1283,1307-1320,1828` |
| `logits_hist_m<8, 12>` (in-tree, rows <= 48) | `for (int p = wid; p < npages; p += nwaves)` with `npages_raw = (seqlen + PAGE_TOK - 1) / PAGE_TOK`; grid 48 blocks per row. | `kv + phys * PAGE_BYTES` (8448 B per page); also builds the histogram when `seqlen_eff > 2048` | `K/jit/csrc/dsa_gfx950/paged_mqa_logits.cuh:32-37,114,159-161,193`; `K/ops/attention/dsa/hip_gfx950/loader.py:24` |
| `_topk_est` (KF) | one program per row; fixed sample of 64 probes x 128 = 8192 logits. Row-count only. | 32 KB | `kf_indexer.py:1969-1975,2004-2008,2142` |
| `_topk_collect` (KF) | 2048 persistent programs over `sum_r ceil(L_r / 8192)` chunks: "ONE full pass over the logits". Context-proportional. | 4 B x L_r | `kf_indexer.py:1959-1961,2052-2064,2145` |
| `_topk_select` (KF) | one program per row over <= 8192 candidates; emits `min(L, topk)` slots through `page_table_1`. Row-count only. | 64 KB | `kf_indexer.py:2091-2122,2149` |
| `k_scatter` (in-tree top-k) | G = 64 blocks per row selecting via the histogram from `logits_hist`; loop bounds not analysed here. | UNCONFIRMED | `K/jit/csrc/dsa_gfx950/topk_transform.cuh:1-5`; `loader.py:25` |
| `_sparse_mla_split_k_kernel` (KF) | grid `(seq, n_head_blocks, kv_splits)`; fixed trip count `cdiv(topk, KV_SPLITS * BLOCK_K)` tiles over the 2048 index slots; loads masked by `valid = slot >= 0`. Independent of context once L >= 2048; fewer KV rows are loaded when L < 2048 (index rows are -1 padded), the loop length stays. | `min(L_r, 2048)` x 576 B per row and layer; q [64, 576] FP8 | `OVK/yihou_kf/kf_sparse_mla.py:951-957,1002-1006,1073-1076,1692` |

Logit formula of the in-tree kernel: per head `fmaf(fmaxf(a0[i], 0.f), w0[i], acc)` over 32 heads, then `T * ks` (per-token K scale)
(`paged_mqa_logits.cuh:179,192`). The logits buffer has the static width of the captured page table (`max_seq_len = block_tables.shape[1] * page_size`,
`S/layers/attention/dsa/dsa_indexer.py:1244`; page table `[max_num_tokens, max_ctx_len]`, `S/layers/attention/dsa_backend.py:1540-1551`); only columns below L_r are computed.

Cost shape: logits and `_topk_collect` scale with `sum over rows of L_r` (x 21 full layers); `_topk_est`, `_topk_select`, sparse MLA and
every other attention op scale with rows only.

## 6. KF overlay gating (`OVK/yihou_kf/`; identical in `OV/` except the three noted lines)

| Path | Hooked symbol | Gate (verbatim fragments) | Evidence |
|---|---|---|---|
| Sparse MLA | `triton_sparse_mla.triton_sparse_mla_fwd` (`YIHOU_KF_SPARSE=1`) | `_verify("sparse") and rows <= SPARSE_MAX_ROWS and heads == 64 and d_v == 512 and topk_length is None`; `SPARSE_MAX_ROWS` = env `YIHOU_KF_SPARSE_MAX_ROWS`, default 512. Verify-only: mode recorded by wrapping `forward_extend` / `forward_decode`. | `__init__.py:117,128,95,105-107` |
| Sparse plan | inside KF | `decode_fp8_64h and H == 64` -> `splits = min(num_cu // max(seq, 1), …)`, used if `>= 2` -> split-K with `block_h=H`, `decode_tuned=True`. i32 page offsets only if `num_pages * kv_dim <= _I32_MAX`. | `kf_sparse_mla.py:1852-1870,331-336,345-346,1602` |
| Indexer <= 48 rows | none (in-tree `Gfx950FusedIndexer`, default on) | `MAX_ROWS = 48`; `if not (1 <= rows <= MAX_ROWS)` -> standard path; only decode / idle / target-verify, not `force_unfused_topk`. The KF replacement `kf_fused_indexer.py` needs `YIHOU_KF_FUSED_INDEXER=1` (not set) and hooks module `sglang.srt.layers.attention.dsa.gfx950_fused_indexer`, which does not exist in `S/`. | `K/ops/attention/dsa/hip_gfx950/fused_decode.py:31`; `S/layers/attention/dsa/dsa_indexer.py:464-469,722-726,734,778`; `__init__.py:408-409,516-517` |
| Indexer logits 49..128 rows | `dsa_indexer.aiter_paged_mqa_logits` (`YIHOU_KF_INDEXER=1`) | `_verify("indexer") and preshuffle and kv_block_size == 64 and rows <= INDEXER_MAX_ROWS and q_fp8.dim() == 3 and block_tables.shape[0] == rows and seq_lens.numel() == rows`; `INDEXER_MAX_ROWS = 128`. | `__init__.py:151,172-173` |
| fused2 vs preshuffle | inside KF logits | fused2 iff `_FUSE and next_n == 1 and heads == 32 and hidden_dim == 128 and KVBlockSize == 64 and ChunkK == 256 and get_cdna_version() == 4 and nx > 1 and _FUSE_SPLITKV % nx == 0 and batch_size % _LPT_GROUP_ROWS == 0 and batch_size // _LPT_GROUP_ROWS <= _LPT_NG2`. Defaults `DSA_FUSE=1`, `DSA_LPT_GROUP_ROWS=6`, `DSA_LPT_NG2=16` (hard-coded 6 / 16 in `OV/…/kf_indexer.py:1757-1758`). So rows must be a multiple of 6 and <= 96. | `kf_indexer.py:1758-1759,1762-1763,1811-1823` |
| Indexer top-k | `DSATopKBackend.topk_transform` | `envs.SGLANG_DSA_FUSE_TOPK.get() and not force_unfused_topk and self.is_sgl_kernel() and not self.should_use_topk_v2() and topk_transform_method == PAGED and … topk == 2048 and rows <= INDEXER_MAX_ROWS … and _verify("indexer")`. `should_use_topk_v2` is False because of `SGLANG_OPT_USE_TOPK_V2=false`. | `__init__.py:218-224`; `S/layers/attention/dsa/dsa_topk_backend.py:55-56` |
| MoE | `AiterRunnerCore.run` (`YIHOU_KF_MOE=1`) | `MOE_M_LO <= M <= MOE_M_HI and h.shape[1] == 6144 … and quant_info.w13_weight.shape[0] == 257 and runner_input.topk_ids.shape[1] == 9`; `MOE_M_LO = 97`, `MOE_M_HI` = env `YIHOU_KF_MOE_M_HI` default 455 (`OV/`: `MOE_M_LO, MOE_M_HI = 97, 455`). | `__init__.py:251-252,272-283` |
| `_HIP_SORT_MAXP` | prologue `glm52_moe_sort_fill` | `_HIP_SORT_NT` = 512 threads, `_HIP_SORT_MAXP = 4096 // _HIP_SORT_NT` = 8 pairs per thread; used only if `0 < topk_ids.shape[0] * topk <= _HIP_SORT_NT * _HIP_SORT_MAXP` (M*9 <= 4096 -> M <= 455). | `kf_moe.py:163-164,494` |
| MoE kernel choice | tuned-table key | `get_padded_M` returns `_DECODE_KEY_M` = 128 for `_DECODE_M_LO <= M <= _DECODE_M_HI` (97..512) while KF is active, else aiter's `nextPow2(M)`. | `kf_moe.py:79-93`; `A/fused_moe.py:1864-1879` |
| gemm1 at intermediate 512 (TP4) | KF launcher `…_glm52v{variant}` | `_g1_engages` requires `2 * kw["D_INTER"] == 128 * 8`. | `kf_moe.py:1037-1049,885-886` |
| gemm1 at intermediate 256 (TP8) | **aiter's own launcher**: the gate above is false (2*256 != 1024), `return _orig(**kw)`. With the token=128 row for (6144, 256, 257): `flydsl_mxmoe_g1_a4w4_16x128x256_f16in_hpf_nt_xcd4` + `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_sbm16`. | `kf_moe.py:1062-1063`; `A/configs/model_configs/glm5_fp4_tuned_fmoe.csv` (row `6144,256,257`, token 128) |
| gemm2 override | only when the tuned stage-2 name is `…t16x128x128_atomic_nt_sbm16` (the intermediate-512 row) -> `…t16x128x256_atomic_nt_sbm16`. Not matched at intermediate 256 (name has no `_nt_`). | `kf_moe.py:1118-1120,1136` |

Production PP8 decode (`CFG/arm-C-1p1d-pp8.env`: `OV/` overlay, shipped fmoe table), per graph batch:

| bs_g | rows R | M | Sparse MLA | Indexer on the 21 full layers | MoE on the 75 sparse layers |
|---|---|---|---|---|---|
| 4 | 16 | 128 | KF split-K | in-tree 4-kernel fused (R <= 48) | KF: `glm52_moe_sort_fill`, bm16 gemm1 (aiter launcher), bm16 atomic gemm2 |
| 8 | 32 | 256 | KF | in-tree fused | KF, same kernels |
| 12 | 48 | 384 | KF | in-tree fused (R = 48 = MAX_ROWS) | KF, same kernels |
| 16 | 64 | 512 | KF | standard chain + KF preshuffle logits (64 % 6 != 0) + KF top-k | aiter: token-512 row `flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w4_kw2_fp4` + `…t32x128x256_atomic_nt_sbm32` |
| 20 | 80 | 640 | KF | standard chain + KF preshuffle logits (80 % 6 != 0) + KF top-k | aiter: token-1024 row `…t64x64x256_w3_fp4` + `…t64x128x256_reduce_nt_sbm64` + `moe_reduction` |
| 24 | 96 | 768 | KF | standard chain + KF **fused2** logits (96/6 = 16 <= 16) + KF top-k | aiter: token-1024 row, as bs 20 |

Notes: with the `kopt` arms the token-1024 row is replaced by the token-512 kernels
(`/perf_apps/liyingli/bench_agentx/kopt-20261109/configs/glm5_fp4_tuned_fmoe.csv`, `CFG/arm-C-1p1d-pp8-kopt.env:2-3`).
The only other bucket taking fused2 is bs 18 (72 rows). With K=3 the fused2 pair (rows 2p, 2p+1) is still two rows of one request,
but the 6-row "group" used for ordering spans 1.5 requests (`kf_indexer.py:1757` comment: "6 at MTP K5, 4 at K3").
How aiter picks `glm5_fp4_tuned_fmoe.csv` among its tables was not traced (UNCONFIRMED); the kernel names in the traces match the rows quoted.

## 7. MegaMoeTP hook (`OVK/kopt_megatp/__init__.py`, `KOPT_MEGATP=1`)

| Item | What the code does | Evidence |
|---|---|---|
| Scope | Layers whose `mlp` is `DeepseekV2MoE` and not `is_nextn`, only under `get_is_capture_mode()`, mode decode or target-verify, `0 < m <= _MAX_LOCAL` with `m = forward_batch.input_ids.shape[0]` (local padded rows 4*bs_g) and `_MAX_LOCAL` = env `KOPT_MEGATP_MAX_LOCAL` default 128. Dense layers 0-2 and the NextN layer keep the original path. | `:39-50,22` |
| Replaced | `dp_gather` returns the local rows (`if _State.gather_off: return hidden_states`); `DeepseekV2MoE.forward` -> `_mega_moe_forward`; FFN exit -> `_MegaExit` (no reduce-scatter). So: all-gather, sort/quant, gemm1, activation, gemm2, reduce-scatter are inside one kernel. The post-attention norm + residual add stays outside. | `:157-160,169-172,183-195,116-141` |
| Router / top-k | Run before the kernel on the **local m rows**: `mod._hip_moe.forward_gate(moe, hidden_states, None, fused_gate=True)` then `moe.topk(hidden_states, router_logits, num_token_non_padded=None, …)` (trace: `hgemm_bf16_t32x32x64x7…`, `topk_reg_kernel`, `_fused_append_shared_experts_kernel`). | `:99-106` |
| Token count | The kernel gets this rank's `[m, 6144]` rows, `[m, 9]` weights and ids; `comm_mode="ag_rs"`, `comm_dtype=_COMM` (bf16), `max_local_tokens=_MAX_LOCAL`. Contract: "every rank calls it the same number of times, in the same order, with the same local token count". Global tokens = 8*m = M. One instance for all layers (`set_weights`). | `:80-89,112-113`; `A/ops/flydsl/mega_moe_tp.py:12-13` |
| Cross-rank waiting | Yes, by construction: "The kernel is persistent (one CTA per CU, CTAs and ranks spin on each other): it needs the whole GPU."; "A wait that times out (2 s) sets a sticky bit"; `poll_errors`: "Nonzero if a wait inside the kernel gave up (a peer never arrived)". Device code polls peer epoch flags (`poll_zero`, `spin_sys_ge`, `ag_wait_meta`, `yag_wait`). The kernel's duration therefore includes the time until every peer has pushed its input rows and its partial sums for this rank's rows. | `A/ops/flydsl/mega_moe_tp.py:18-21,219`; `A/ops/flydsl/kernels/mega_moe_tp/communication.py:240-251,701-704,1206-1209` |
| Kernel name | `mega_moe_tp_fused_h6144_i256_e257_k9_mt4_t1024_silu_ag1_tp8_nab3_npp2_cb16_lb_q2_mtskip_pf`: H 6144, per-rank intermediate 256, 257 experts, top-k 9, row tile 16*4, max 1024 global tokens, 8 ranks, `npp2`/`q2` from `AITER_MEGAMOE_TP_LB_NPP=2` / `LB_Q=2`. | `CFG/arm-C-1p1d-pp8-megatp-mt4.env:6`; `A/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_kernel.py:59-121` |

## 8. lm_head on decode with DP attention

| Item | Answer | Evidence |
|---|---|---|
| Rows | All M = 32*bs_g global rows on every rank (hidden states gathered across DP ranks first, section 4 T2). | `S/layers/logits_processor.py:477-479,1126-1147` |
| Weight slice | Vocab 154880 split over the TP8 group: [19360, 6144] BF16 per rank (`use_attn_tp_group` False because `enable_dp_lm_head` is False). | `S/models/deepseek_v2.py:3264-3270` |
| Logits exchange | Not all-gathered: `all_to_all_single` sends each destination DP rank only its own row block; result on a rank = [R, 154880], then copied to the fp32 buffer. Falls back to TP all-gather + `dp_scatter` only if `_can_use_tp_lm_head_all_to_all` is false. | `S/layers/logits_processor.py:943-966,1179-1225` |
| Draft heads | `draft` and `draft_extend` graphs run the NextN `lm_head` (same class and flag). Their gather path was not traced line by line (UNCONFIRMED). | `S/models/deepseek_nextn.py:293-299` |

## 9. Not confirmed from the provided trees

| # | Item | Why |
|---|---|---|
| 1 | `SGLANG_USE_AITER=1` on the decode. Every aiter / KF path above requires it; the default is `EnvBool(False)` (`S/environ.py:903`) and no launch file sets it. | Must come from the image environment; the trace kernel names imply it is on. |
| 2 | Last-wins handling of the duplicated flags (`--speculative-*`, `--mem-fraction-static`, `--json-model-override-args`) and an unchanged `--max-running-requests 256`. | The engine is started through `python3 -m infera.engine.sglang` (`W:139`), which is not in the trees. The 32-request bucket cap in section 2 depends on it. |
| 3 | `CFG/site.local.env` contents. | Not read (blocked by the session's credential guard). |
| 4 | Attribution of `Cijk_SB_BiasS_HAS_ScaleAlphaVec_PostGSU8/4_VW4` to the preceding `o_proj` / dense `gate_up_proj` GEMM. | hipBLASLt source not in the trees; by position and name only. |
| 5 | Bodies of `aiter::allgather_vec`, `aiter::reduce_scatter_split_first_dim`, `aiter::cross_device_reduce_2stage`. | Only the python wrapper (`A/dist/device_communicators/custom_all_reduce.py`) is present; mapping is by call site and name. |
| 6 | Which tuned-table row yields which `hgemm_bf16_*` / `Cijk_*` instance, and how aiter selects `glm5_fp4_tuned_fmoe.csv`. | `tgemm` / config lookup not traced; op-to-kernel mapping for BF16 GEMMs is by execution order. |
| 7 | Loop bounds of the in-tree top-k (`k_scatter`). | `topk_transform.cuh` only skimmed. |
| 8 | CU count 256 for the KF sparse-MLA split plan (`splits = min(num_cu // max(seq, 1), …)` must be >= 2). | `_cu_count` reads the device; value taken from the `cu_num` column of the aiter tables. |
| 9 | Kernel-level op list, DP gather and lm_head path of the `draft` and `draft_extend` graphs. | Only forward counts, modes and row widths were traced. |
| 10 | The first `direct_copy` kernel of the tail, and whether `global_tokens[:] = …` after the all-reduce launches a copy. | Not attributed. |
| 11 | That a rank with zero requests replays the same bucket with pad rows only. | From the comment at `FIX/…/decode_cuda_graph_runner.py:1352-1354`; scheduler path not traced. |
| 12 | Whether `fused_qkv_a` goes through `linear_with_fused_a_gemm` or the plain module call. | `fused_a_gemm_weight_eligible` not read; both are one BF16 GEMM on the same weight. |
| 13 | Resolution of the DSA page size to 64 on the decode. | Implied by the paths that ran (KF gate `kv_block_size == 64`, in-tree `PAGE_SIZE = 64`), not traced in the arg hooks. |
