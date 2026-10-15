# Decode-side "prefill" — key operator shapes, GLM-5.2-MXFP4 decode leg (TP4 / attn-DP4 / EP1)

All shapes are **per rank** (rank 0 = DP0/TP0), first-hand from the runs listed at the end.
Notation: **B** = requests per attention-DP rank, **T = 6B** (draft 6 tokens/request in
verify and draft-extend), **G = 4T** (MLP/MoE input after the DP all-gather over 4 attention-DP
ranks), **C** = context length padded to page 64 (`max_seq_len`).
Main point: **B = 4 (global 16), ISL 110,000 → T = 24, G = 96, C = 110,080**.

## 1. What "decode-side prefill" is here

The live decode leg resolved `speculative_attention_mode=prefill`. In sglang `a9fb1c3238`,
`deepseek_v2.py:dispatch_attn_forward_method` then sends **TARGET_VERIFY** and
**DRAFT_EXTEND_V2** through the **prefill** attention backend (`dsa_prefill_backend=triton`).
These are the decode leg's extend-shaped forwards (6 tokens per request). PD `PREBUILT` runs no
model forward. Plain draft DECODE (1 token/request, 4 times per iteration) is included only for
completeness (section 5).

One speculative iteration, captured (rank 0, B = 4):

| runner | mode | requests | tokens |
|---|---|---|---|
| draft | DECODE x4 | 4 | 4 each |
| target | **TARGET_VERIFY** | 4 | **24** |
| draft | **DRAFT_EXTEND_V2** | 4 | **24** (extend 6 each) |

## 2. Deployment and workload (from the reference packup)

| item | value | source |
|---|---|---|
| decode leg | TP4, attn-DP4, EP1, `moe_a2a=none`, GPUs 4-7 | `server-info.29257.json` |
| spec | EAGLE steps 5 / draft 6 / topk 1, simulated accept 3.61 | same, `runtime.env` |
| attention | dsa, triton prefill+decode, topk sgl-kernel, `index_share_for_mtp_iteration=true` | same |
| KV | fp8_e4m3, page 64 | same |
| comms | `--disable-custom-all-reduce`, aiter AR fusion | same |
| ISL | p50 110,498 / mean 145,279 / p90 275,315 / p95 337,078 | `agentx_conc40.json` |
| OSL (actual) | p50 390 / mean 987 / p90 2,345 | same |
| running req per DP rank (decode, profiling window) | mode 3, **p50 4**, p90 6, max 15 | `decode-0.log`, 7,356 lines |
| context per running request | p50 ~237K (time-weighted) | same |

Chosen: B = 4 (p50), ISL 110,000 (request p50). Variants: ISL 240,000 (P2, context
dependence) and B = 6 (P4, the p90 batch).

## 3. TARGET_VERIFY — key operators and shapes (B = 4, T = 24, G = 96, C = 110,080)

Graph-ON time is rank 0, deployment mode (CUDA graph replay), 5 profiled iterations
(`p1g2`). TARGET_VERIFY kernels = **33.1 ms / iteration**. Shapes are from the graph-OFF run of
the same point (graph captured at exactly bs = [4], so identical tensors).

| # | operator (kernel) | % verify | calls | shapes (activation x weight → out) |
|---|---|---|---|---|
| 1 | **MoE gemm1** (aiter `fused_moe_`, MXFP4 a4w4, `gemm1_a4w4_..._ne257_bm16`) | 20.6 | 75 | x bf16[G=96, 6144]; w13 fp4x2[257, 1024, 3072] (+e8m0 scale u8[257, 1024, 192]); topk ids/w [96, 9] |
| 2 | **all-gather / reduce-scatter** (RCCL; DP-attn gather before MLP, scatter after) | 13.4 | 158 | gather bf16[24, 6144] → [96, 6144]; scatter bf16[96, 6144] → [24, 6144]; lm-head gather [96, 38720] |
| 3 | **sparse MLA attention** (`_sparse_mla_split_k_kernel`, triton) | 12.4 | 78 | q_nope fp8[24, 64, 512], q_rope fp8[24, 64, 64], KV fp8[pool, 1, 576], topk idx i32[24, 2048] → partial bf16[24, 2, 64, 512]; grid [24, 4, 2], KV_SPLITS 2 |
| 4 | **MoE gemm2** (aiter, MXFP4) | 10.0 | 75 | inter [96·9, 512]; w2 fp4x2[257, 6144, 256] (+scale u8[257, 6144, 16]) → bf16[96, 6144] |
| 5 | **o_proj** (hipBLASLt via `aiter::gemm_a16w16`) | 8.4 (+1.6 GSU tail) | 78 | bf16[24, 16384] x bf16[16384 → 6144] |
| 6 | **q_b_proj** (triton `_gemm_a16_w16`) | 4.1 | 78 | bf16[24, 2048] x bf16[2048 → 16384] |
| 7 | **indexer logits** (`_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle`) | 3.3 | 21 | q fp8[24, 32, 128]; index-K cache u8[pages, 64, 1, 132]; weights f32[24, 32]; block table i32[24, 1720]; seqlens i32[24] → **logits f32[24, C=110080]** |
| 8 | **MLA absorb bmm** x2 (triton `_batched_gemm_a8w8_..._prequant`) | 3.2 | 156 | W_UK: bf16[24, 64, 192] x fp8[64, 192, 512] → [64, 24, 512]; W_UV: bf16[24, 64, 512] x fp8[64, 512, 256] → [24, 64, 256] |
| 9 | **fused_qkv_a_proj_with_mqa** (`aiter::gemm_a16w16` hgemm) | 2.5 | 78 | bf16[24, 6144] x bf16[6144 → 2624] (q_lora 2048 + kv_lora 512 + rope 64) |
| 10 | add_rmsnorm (+quant) | 2.3 | 156 | bf16[24, 6144] |
| 11 | sparse MLA reduce | 1.7 | 78 | [24, 2, 64, 512] → bf16[24, 64, 512]; grid [24, 64, 8] |
| 12 | MoE router gate | 1.5 | 75 | bf16[96, 6144] x bf16[6144 → 256] |
| 13 | mxfp4 MoE sort+quant | 1.2 | 75 | ids/w [96, 9] → sorted 4720 slots, x → fp4 |
| 14 | grouped topk (biased) | 1.1 | 75 | logits bf16[96, 256] → [96, 8]; + shared expert appended → [96, 9] |
| 15 | fused qk-rope + cat + KV write | 1.1 | 78 | q_nope bf16[24, 64, 512], q_pe [24, 64, 64], k_nope [24, 1, 512], k_pe [24, 1, 64] → q fp8[24, 64, 576], KV write 24 slots |
| 16 | fused q/kv-a RMSNorm | 1.0 | 78 | bf16[24, 2048] and [24, 512] |
| 17 | indexer topk (`sgl_kernel::fast_topk_transform_fused`, 4 kernels) | 2.4 total | 21 | logits f32[24, 110080] → i32[24, 2048] |
| 18 | indexer weights proj | 0.7 | 21 | bf16[24, 6144] x bf16[6144 → 32] |
| 19 | indexer wq_b / wk / k_norm / rope / Hadamard / K-quant+cache | <0.7 each | 21 | wq_b [24, 2048]→[24, 4096]; wk [24, 6144]→[24, 128]; k_norm [24, 128]; rope q [24, 32, 64], k [24, 64]; K cache write [24, 128] |

Structure facts read off the capture:

- **Indexer runs on 21 of 78 layers**: 0, 1, 2, 6, 10, ..., 74 (every 4th after 2). The other
  57 layers take `prev_topk_indices` from the last indexer layer.
- **MoE sees G = 4T** tokens: attention runs on the local 24, the MLP on all four DP ranks'
  96 (all-gather before, reduce-scatter after). **topk = 9** = 8 routed + 1 fused shared expert
  over **257** experts. MoE TP = 4 → per-rank intermediate **512** (2048 / 4).
- Dense MLP (layers 0-2): gate_up bf16[96, 6144] x [6144 → 6144], down [96, 3072] x [3072 → 6144].
- **Weights**: target routed experts MXFP4 (u8-packed + e8m0 scales); every attention linear,
  the indexer, router, dense MLP and o_proj are **bf16**. MLA absorb matrices fp8.
- Verify hands the attention q as one fp8[T, 64, 576] tensor; the kernel reads it as
  q_nope [.., 512] + q_rope [.., 64] views.

## 4. DRAFT_EXTEND_V2 (MTP layer, 1 decoder layer) — 1.52 ms / iteration graph-ON

| operator | % | shapes |
|---|---|---|
| **MoE (bf16 CK 2-stage)** stage1 + stage2 | 25.1 + 12.1 | x bf16[96, 6144]; w13 **bf16**[257, 1024, 6144]; w2 **bf16**[257, 6144, 512]; topk [96, 9] |
| all-gather / reduce-scatter | 8.8 | as verify, 1 layer |
| argmax / softmax over vocab | 6.1 + 3.3 | f32[4, 154880] |
| lm-head mm | 5.9 | [96, 6144] x [6144 → 38720] (vocab/4 per rank) |
| sparse MLA | 3.5 | q_nope fp8[24, 64, 512] + q_rope fp8[24, 64, 64] (passed **separately** here), idx [24, 2048] |
| indexer logits | 3.4 | same as verify, f32[24, 110080] |
| eh_proj | 2.3 | bf16[24, 12288] x [12288 → 6144] |
| q_b / o_proj / qkv_a | 1.2 / 2.4 / 0.8 | as verify, T = 24 |

The **draft model's MoE is bf16, not MXFP4**: CK `ck_moe_stage1/2` instead of the a4w4 kernels.

## 5. Draft DECODE x4 (for completeness; not "prefill")

T = B = 4 per step, G = 16 at the MoE (bf16 CK), sparse MLA decode kernels
(`_sparse_mla_decode_split/reduce`). **Graph-ON traces show none of the draft-decode graph's
kernels** (its `hipGraphLaunch` correlates to 0 kernels), so its time is only available
graph-OFF. Not investigated further — out of scope.

## 6. What scales with what (first-hand, three points)

| dimension | B=4, C=110,080 (P1/P3/P5) | B=4, C=240,064 (P2) | B=6, C=110,080 (P4) |
|---|---|---|---|
| verify / extend tokens T | 24 | 24 | **36** |
| MoE tokens G | 96 | 96 | **144** |
| indexer logits | f32[24, 110080] | f32[24, **240064**] | not captured (P4 predates the function recorder); indexer out i32[**36**, 2048] |
| index block table | i32[24, 1720] | i32[24, **3751**] | not captured |
| everything else | — | **identical to P1** | scales with T / G only |

Only the indexer logits, its block table and the topk input depend on context; all GEMMs,
attention (fixed 2048 selected tokens) and comms depend on T / G only.

## 7. Caveats

1. **Synthetic KV and simulated acceptance.** Shapes are exact; values (and thus MoE routing
   balance) are not the deployment's. Same as the tool's published caveat.
2. **The ranking is from one profiled graph-ON run** (n = 1), rank 0 only, with the profiler on
   (`is_performance_measurement: false`). Use it for "which ops matter", not as latency.
3. **Graph-OFF runs P1/P2 ran concurrently** on GPUs 4-7 / 0-3; nothing timed is taken from them.
4. **Image ID differs** (`291f148a` vs the deployment's `fd7220a5`) because `docker load` on this
   node reports a manifest digest; all 72 RootFS layer DiffIDs are identical.
5. The tool needed three fixes for this sglang (published-context initialisers,
   `is_startup_weight_load_overlap`, and the `can_run_decode_cuda_graph` replay gate). Without
   the third, "graph ON" silently ran eager — caught by `graph_execution_counts = 0`.
6. `act_quant` (indexer q fp8 quant) was not captured by name; its output q fp8[24, 32, 128]
   is the logits kernel's recorded input.

## Runs

| run | B (global) | ISL | graph | purpose |
|---|---|---|---|---|
| `smoke1` | 16 | 110,000 | off | port check, first shapes |
| `p1_bs16_isl110k_off` | 16 | 110,000 | off | module shapes + aten/aiter op dims |
| `p2_bs16_isl240k_off` | 16 | 240,000 | off | context dependence |
| `p1g_..._NOGRAPH` | 16 | 110,000 | "on" (did not replay) | kept as evidence of the gate bug |
| **`p1g2_bs16_isl110k_on`** | 16 | 110,000 | **on** (30/30 replays) | **time ranking** |
| `p3_bs16_isl110k_off_triton` | 16 | 110,000 | off | Triton launch args + weight shapes |
| `p4_bs24_isl110k_off_triton` | 24 | 110,000 | off | batch dependence (p90 batch) |
| `p5_bs16_isl110k_off_fn` | 16 | 110,000 | off | indexer helper-function shapes |

Generated tables: `analysis/hot_ops_joined_p1.yihou.md` (graph-ON ranking x graph-OFF shapes),
`analysis/modules_*.yihou.md` (every module call), `analysis/kernels_*.yihou.md` (every kernel).
