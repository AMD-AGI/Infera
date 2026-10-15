# Operator shapes — results/p3_bs16_isl110k_off_triton/shapes_rank_0_yihou.jsonl

meta: `{"event": "meta", "verify_iteration": 12, "seq_lens_before": [110047, 110047, 110047, 110047], "accept_lens": [4, 4, 4, 4], "rank": 0, "local_batch_size": 4, "graph_disabled": true}`

## Forwards in the captured iteration

| runner | mode | batch | tokens | extend_seq_lens |
|---|---|---|---|---|
| draft | DECODE | 4 | 4 | None |
| draft | DECODE | 4 | 4 | None |
| draft | DECODE | 4 | 4 | None |
| draft | DECODE | 4 | 4 | None |
| target | TARGET_VERIFY | 4 | 24 | None |
| draft | DRAFT_EXTEND_V2 | 4 | 24 | [6, 6, 6, 6] |

## DECODE

| module | class | inputs | kwargs | output | calls | layers |
|---|---|---|---|---|---|---|
| `model.embed_tokens` | VocabParallelEmbedding | int64[4] |  | bfloat16[4, 6144] | 4 | - |
| `model.enorm` | RMSNorm | bfloat16[4, 6144] |  | bfloat16[4, 6144] | 4 | - |
| `model.hnorm` | RMSNorm | bfloat16[4, 6144] |  | bfloat16[4, 6144] | 4 | - |
| `model.eh_proj` | ReplicatedLinear | bfloat16[4, 12288] |  | bfloat16[4, 6144] | 4 | - |
| `model.decoder.input_layernorm` | RMSNorm | bfloat16[4, 6144] |  | bfloat16[4, 6144] | 4 | - |
| `model.decoder.self_attn` | DeepseekV2AttentionMLA |  | positions=int64[4], hidden_states=bfloat16[4, 6144], prev_topk_indices=int32[4, 2048] | bfloat16[4, 6144], int32[4, 2048] | 4 | - |
| `model.decoder.self_attn.fused_qkv_a_proj_with_mqa` | ReplicatedLinear | bfloat16[4, 6144] |  | bfloat16[4, 2624] | 4 | - |
| `model.decoder.self_attn.q_b_proj` | ColumnParallelLinear | bfloat16[4, 2048] |  | bfloat16[4, 16384] | 4 | - |
| `model.decoder.self_attn.attn_mqa` | RadixAttention | float8_e4m3fn[4, 64, 576] | k_rope=bfloat16[4, 1, 64], topk_indices=int32[4, 2048] | bfloat16[1, 4, 64, 512] | 4 | - |
| `model.decoder.self_attn.o_proj` | RowParallelLinear | bfloat16[4, 16384] |  | bfloat16[4, 6144] | 4 | - |
| `model.decoder.post_attention_layernorm` | RMSNorm | bfloat16[4, 6144], bfloat16[4, 6144] |  | bfloat16[4, 6144], bfloat16[4, 6144] | 4 | - |
| `model.decoder.mlp` | DeepseekV2MoE | bfloat16[16, 6144] |  | bfloat16[16, 6144] | 4 | - |
| `model.decoder.mlp.gate` | MoEGate | bfloat16[16, 6144] |  | bfloat16[16, 256] | 4 | - |
| `model.decoder.mlp.topk` | TopK | bfloat16[16, 6144], bfloat16[16, 256] |  | float32[16, 9], int32[16, 9], bfloat16[16, 256] | 4 | - |
| `model.decoder.mlp.experts` | FusedMoE | bfloat16[16, 6144], float32[16, 9], int32[16, 9], bfloat16[16, 256] |  | bfloat16[16, 6144] | 4 | - |
| `model.shared_head.norm` | RMSNorm | bfloat16[4, 6144], bfloat16[4, 6144] |  | bfloat16[4, 6144], bfloat16[4, 6144] | 4 | - |
| `logits_processor` | LogitsProcessor | int64[4], bfloat16[4, 6144] |  |  | 4 | - |

## TARGET_VERIFY

| module | class | inputs | kwargs | output | calls | layers |
|---|---|---|---|---|---|---|
| `model.embed_tokens` | VocabParallelEmbedding | int64[24] |  | bfloat16[24, 6144] | 1 | - |
| `model.layers.*.input_layernorm` | RMSNorm | bfloat16[24, 6144] |  | bfloat16[24, 6144] | 1 | 0 |
| `model.layers.*.self_attn` | DeepseekV2AttentionMLA |  | positions=int64[24], hidden_states=bfloat16[24, 6144] | bfloat16[24, 6144] | 2 | 0-1 |
| `model.layers.*.self_attn.fused_qkv_a_proj_with_mqa` | ReplicatedLinear | bfloat16[24, 6144] |  | bfloat16[24, 2624] | 78 | 0-77 |
| `model.layers.*.self_attn.q_b_proj` | ColumnParallelLinear | bfloat16[24, 2048] |  | bfloat16[24, 16384] | 78 | 0-77 |
| `model.layers.*.self_attn.indexer` | Indexer |  | x=bfloat16[24, 6144], q_lora=bfloat16[24, 2048], positions=int64[24] | int32[24, 2048] | 21 | 0-2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74 |
| `model.layers.*.self_attn.indexer.wq_b` | ReplicatedLinear | bfloat16[24, 2048] |  | bfloat16[24, 4096] | 21 | 0-2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74 |
| `model.layers.*.self_attn.indexer.wk` | ReplicatedLinear | bfloat16[24, 6144] |  | bfloat16[24, 128] | 21 | 0-2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74 |
| `model.layers.*.self_attn.indexer.k_norm` | LayerNorm | bfloat16[24, 128] |  | bfloat16[24, 128] | 21 | 0-2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74 |
| `model.layers.*.self_attn.indexer.rotary_emb` | RotaryEmbedding | int64[24], bfloat16[24, 32, 64], bfloat16[24, 64] |  | bfloat16[24, 32, 64], bfloat16[24, 64] | 21 | 0 |
| `model.layers.*.self_attn.attn_mqa` | RadixAttention | float8_e4m3fn[24, 64, 576] | k_rope=bfloat16[24, 1, 64], topk_indices=int32[24, 2048] | bfloat16[1, 24, 64, 512] | 78 | 0-77 |
| `<attn_backend DeepseekSparseAttnBackend.forward_extend>` | DeepseekSparseAttnBackend | float8_e4m3fn[24, 64, 576] | k_rope=bfloat16[24, 1, 64], topk_indices=int32[24, 2048] |  | 78 | - |
| `model.layers.*.self_attn.o_proj` | RowParallelLinear | bfloat16[24, 16384] |  | bfloat16[24, 6144] | 78 | 0-77 |
| `model.layers.*.post_attention_layernorm` | RMSNorm | bfloat16[24, 6144], bfloat16[24, 6144] |  | bfloat16[24, 6144], bfloat16[24, 6144] | 78 | 0-77 |
| `model.layers.*.mlp` | DeepseekV2MLP | bfloat16[96, 6144] |  | bfloat16[96, 6144] | 3 | 0-2 |
| `model.layers.*.mlp.gate_up_proj` | MergedColumnParallelLinear | bfloat16[96, 6144] |  | bfloat16[96, 6144] | 3 | 0-2 |
| `model.layers.*.mlp.act_fn` | SiluAndMul | bfloat16[96, 6144] |  | bfloat16[96, 3072] | 3 | 0-2 |
| `model.layers.*.mlp.down_proj` | RowParallelLinear | bfloat16[96, 3072] |  | bfloat16[96, 6144] | 3 | 0-2 |
| `model.layers.*.input_layernorm` | RMSNorm | bfloat16[24, 6144], bfloat16[24, 6144] |  | bfloat16[24, 6144], bfloat16[24, 6144] | 77 | 1-77 |
| `model.layers.*.self_attn` | DeepseekV2AttentionMLA |  | positions=int64[24], hidden_states=bfloat16[24, 6144] | bfloat16[24, 6144], int32[24, 2048] | 19 | 2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74 |
| `model.layers.*.self_attn` | DeepseekV2AttentionMLA |  | positions=int64[24], hidden_states=bfloat16[24, 6144], prev_topk_indices=int32[24, 2048] | bfloat16[24, 6144], int32[24, 2048] | 38 | 3-4,7-8,11-12,15-16,19-20,23-24,27-28,31-32,35-36,39-40,43-44,47-48,51-52,55-56,59-60,63-64,67-68,71-72,75-76 |
| `model.layers.*.mlp` | DeepseekV2MoE | bfloat16[96, 6144] |  | bfloat16[96, 6144] | 75 | 3-77 |
| `model.layers.*.mlp.gate` | MoEGate | bfloat16[96, 6144] |  | bfloat16[96, 256] | 75 | 3-77 |
| `model.layers.*.mlp.topk` | TopK | bfloat16[96, 6144], bfloat16[96, 256] |  | float32[96, 9], int32[96, 9], bfloat16[96, 256] | 75 | 3-77 |
| `model.layers.*.mlp.experts` | FusedMoE | bfloat16[96, 6144], float32[96, 9], int32[96, 9], bfloat16[96, 256] |  | bfloat16[96, 6144] | 75 | 3-77 |
| `model.layers.*.self_attn` | DeepseekV2AttentionMLA |  | positions=int64[24], hidden_states=bfloat16[24, 6144], prev_topk_indices=int32[24, 2048] | bfloat16[24, 6144] | 19 | 5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,69,73,77 |
| `model.norm` | RMSNorm | bfloat16[24, 6144], bfloat16[24, 6144] |  | bfloat16[24, 6144], bfloat16[24, 6144] | 1 | - |
| `logits_processor` | LogitsProcessor | int64[24], bfloat16[24, 6144] |  |  | 1 | - |

## DRAFT_EXTEND_V2

| module | class | inputs | kwargs | output | calls | layers |
|---|---|---|---|---|---|---|
| `model.embed_tokens` | VocabParallelEmbedding | int64[24] |  | bfloat16[24, 6144] | 1 | - |
| `model.enorm` | RMSNorm | bfloat16[24, 6144] |  | bfloat16[24, 6144] | 1 | - |
| `model.hnorm` | RMSNorm | bfloat16[24, 6144] |  | bfloat16[24, 6144] | 1 | - |
| `model.eh_proj` | ReplicatedLinear | bfloat16[24, 12288] |  | bfloat16[24, 6144] | 1 | - |
| `model.decoder.input_layernorm` | RMSNorm | bfloat16[24, 6144] |  | bfloat16[24, 6144] | 1 | - |
| `model.decoder.self_attn` | DeepseekV2AttentionMLA |  | positions=int64[24], hidden_states=bfloat16[24, 6144] | bfloat16[24, 6144], int32[24, 2048] | 1 | - |
| `model.decoder.self_attn.fused_qkv_a_proj_with_mqa` | ReplicatedLinear | bfloat16[24, 6144] |  | bfloat16[24, 2624] | 1 | - |
| `model.decoder.self_attn.q_b_proj` | ColumnParallelLinear | bfloat16[24, 2048] |  | bfloat16[24, 16384] | 1 | - |
| `model.decoder.self_attn.indexer` | Indexer |  | x=bfloat16[24, 6144], q_lora=bfloat16[24, 2048], positions=int64[24] | int32[24, 2048] | 1 | - |
| `model.decoder.self_attn.indexer.wq_b` | ReplicatedLinear | bfloat16[24, 2048] |  | bfloat16[24, 4096] | 1 | - |
| `model.decoder.self_attn.indexer.wk` | ReplicatedLinear | bfloat16[24, 6144] |  | bfloat16[24, 128] | 1 | - |
| `model.decoder.self_attn.indexer.k_norm` | LayerNorm | bfloat16[24, 128] |  | bfloat16[24, 128] | 1 | - |
| `model.layers.*.self_attn.indexer.rotary_emb` | RotaryEmbedding | int64[24], bfloat16[24, 32, 64], bfloat16[24, 64] |  | bfloat16[24, 32, 64], bfloat16[24, 64] | 1 | 0 |
| `model.decoder.self_attn.attn_mqa` | RadixAttention | float8_e4m3fn[24, 64, 512] | q_rope=float8_e4m3fn[24, 64, 64], k_rope=bfloat16[24, 1, 64], topk_indices=int32[24, 2048] | bfloat16[1, 24, 64, 512] | 1 | - |
| `<attn_backend DeepseekSparseAttnBackend.forward_extend>` | DeepseekSparseAttnBackend | float8_e4m3fn[24, 64, 512] | q_rope=float8_e4m3fn[24, 64, 64], k_rope=bfloat16[24, 1, 64], topk_indices=int32[24, 2048] |  | 1 | - |
| `model.decoder.self_attn.o_proj` | RowParallelLinear | bfloat16[24, 16384] |  | bfloat16[24, 6144] | 1 | - |
| `model.decoder.post_attention_layernorm` | RMSNorm | bfloat16[24, 6144], bfloat16[24, 6144] |  | bfloat16[24, 6144], bfloat16[24, 6144] | 1 | - |
| `model.decoder.mlp` | DeepseekV2MoE | bfloat16[96, 6144] |  | bfloat16[96, 6144] | 1 | - |
| `model.decoder.mlp.gate` | MoEGate | bfloat16[96, 6144] |  | bfloat16[96, 256] | 1 | - |
| `model.decoder.mlp.topk` | TopK | bfloat16[96, 6144], bfloat16[96, 256] |  | float32[96, 9], int32[96, 9], bfloat16[96, 256] | 1 | - |
| `model.decoder.mlp.experts` | FusedMoE | bfloat16[96, 6144], float32[96, 9], int32[96, 9], bfloat16[96, 256] |  | bfloat16[96, 6144] | 1 | - |
| `model.shared_head.norm` | RMSNorm | bfloat16[24, 6144], bfloat16[24, 6144] |  | bfloat16[24, 6144], bfloat16[24, 6144] | 1 | - |
| `logits_processor` | LogitsProcessor | int64[24], bfloat16[24, 6144] |  |  | 1 | - |

## Parameters (direct, per module pattern)

| module | params |
|---|---|
| `model.embed_tokens` | weight=bfloat16[154880, 6144] |
| `model.enorm` | weight=bfloat16[6144] |
| `model.hnorm` | weight=bfloat16[6144] |
| `model.eh_proj` | weight=bfloat16[6144, 12288] |
| `model.decoder.input_layernorm` | weight=bfloat16[6144] |
| `model.decoder.self_attn.fused_qkv_a_proj_with_mqa` | weight=bfloat16[2624, 6144] |
| `model.decoder.self_attn.q_b_proj` | weight=bfloat16[16384, 2048] |
| `model.decoder.self_attn.attn_mqa` | k_scale=float32[], v_scale=float32[] |
| `model.decoder.self_attn.o_proj` | weight=bfloat16[6144, 16384] |
| `model.decoder.post_attention_layernorm` | weight=bfloat16[6144] |
| `model.decoder.mlp.gate` | weight=bfloat16[256, 6144], e_score_correction_bias=float32[256] |
| `model.decoder.mlp.experts` | w13_weight=bfloat16[257, 1024, 6144], w2_weight=bfloat16[257, 6144, 512] |
| `model.shared_head.norm` | weight=bfloat16[6144] |
| `model.layers.*.input_layernorm` | weight=bfloat16[6144] |
| `model.layers.*.self_attn.fused_qkv_a_proj_with_mqa` | weight=bfloat16[2624, 6144] |
| `model.layers.*.self_attn.q_b_proj` | weight=bfloat16[16384, 2048] |
| `model.layers.*.self_attn.indexer.wq_b` | weight=bfloat16[4096, 2048] |
| `model.layers.*.self_attn.indexer.wk` | weight=bfloat16[128, 6144] |
| `model.layers.*.self_attn.indexer.k_norm` | bias=bfloat16[128], weight=bfloat16[128] |
| `model.layers.*.self_attn.indexer.rotary_emb` | cos_cache=bfloat16[1048576, 1, 1, 32], sin_cache=bfloat16[1048576, 1, 1, 32] |
| `model.layers.*.self_attn.attn_mqa` | k_scale=float32[], v_scale=float32[] |
| `model.layers.*.self_attn.o_proj` | weight=bfloat16[6144, 16384] |
| `model.layers.*.post_attention_layernorm` | weight=bfloat16[6144] |
| `model.layers.*.mlp.gate_up_proj` | weight=bfloat16[6144, 6144] |
| `model.layers.*.mlp.down_proj` | weight=bfloat16[6144, 3072] |
| `model.layers.*.mlp.gate` | weight=bfloat16[256, 6144], e_score_correction_bias=float32[256] |
| `model.layers.*.mlp.experts` | w13_weight=uint8[257, 1024, 3072], w2_weight=uint8[257, 6144, 256], w13_weight_scale=uint8[257, 1024, 192], w2_weight_scale=uint8[257, 6144, 16] |
| `model.norm` | weight=bfloat16[6144] |
| `model.decoder.self_attn.indexer.wq_b` | weight=bfloat16[4096, 2048] |
| `model.decoder.self_attn.indexer.wk` | weight=bfloat16[128, 6144] |
| `model.decoder.self_attn.indexer.k_norm` | bias=bfloat16[128], weight=bfloat16[128] |
