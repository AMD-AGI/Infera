# DSv4.1-Flash on MI355X (vLLM 9c1d45c2, Infera): hot-op distribution per role, and where mHC ranks

Date 2026-11-22. Evidence marks: [F] first-hand (trace / log / code read), [A] arithmetic on [F], [I] inference.

## 0. Answer

- **Is mHC hot?** No. It is not a top kernel in either role. Measured as a share of GPU kernel time [F]:

  | role | mHC share | mHC share, NCCL excluded | bucket rank | first mHC kernel rank |
  |---|---|---|---|---|
  | prefill (PD prefill engine) | 4.5% | 6.4% | 7 of 10 | #12 |
  | decode (colocated proxy, decode steps only) | 2.4% | 3.6% | 6 of 10 | #11 |

  - Prefill: 1,657.7 kernel-ms in 80 steps.
  - Decode: 899 kernel-ms in 571 steps, about 1.4 ms per step out of a 42–48 ms GPU step span.
- **What is hot instead** [F]:
  - Prefill: NCCL Generic 29%, which includes waiting for the peer DP rank; MoE 15.4%; sparse-MLA prefill attention 15.3%; MX quant 13.6%; GEMM 11.1%.
  - Decode: NCCL 33.5%, which includes the peer wait; MoE 22.0%; GEMM 16.3%; sparse decode attention 13.1%.
- The mission still requires mHC to be optimised on both sides. The upper bound on end-to-end gain is about the mHC share above [A].

## 1. What was profiled

| profile | run | deployment | graph mode | load | step token stats |
|---|---|---|---|---|---|
| prefill | kfp-a2-…-11220504 | Infera+vLLM PD 2P2D: prefill DP2 ×2 (TP1/rank, MoE TP over 2 ranks, Engram on host); Mooncake TCP | FULL graphs captured; 16k-token prefill steps run eager (dispatches 10,648 of 38,808 graph-replayed) | AgentX c768, cold warmup phase, prefill queues 50+ | MIXED steps: 2 req × 8192 = 16,384 tokens/step (long-prefill 8192) [F] |
| decode | kfp-b-agg4-…-11220926 | colocated proxy: 4 × DP2 engines with the PD decode attention path (fp8_ds_mla KV, AITER sparse-MLA fast path off), user choice 11-22 | FULL + PIECEWISE (breakable) graphs; decode dispatches 72–94% graph-replayed | AgentX c512, +1200 s, 30–58 running per rank | DECODE: 29–64 req/rank × 6 = 174–384 tokens/step; golden acceptance 3.52 measured [F] |

Why decode is a proxy [F]:
- On crusoe, same-node PD decode moved no KV.
- With Mooncake TCP, kfp-a2 and kfp-a3 logged 280–498 "transfer engine returned -1" errors per decode engine, with 0 decode steps carrying requests.
- The RDMA container setup crashes at warmup with Triton Code 209 (leader E7). Leader table: `../dsv41_vllm_pd_20261119/research/mooncake_samenode_transport.md`.
- The proxy runs the same engine shape and kernel path as a PD decode engine. Its batch per rank (≤64) may be lower than a saturated PD decode [I].

Method:
- Profiler: Muse calibrated profiler, i.e. the rocprofiler-sdk fork (jiejingzhangamd/rocm-systems v35 @ae7c95f4), built in this image.
- Window hook ported to vLLM V1 `GPUModelRunner`: `scripts/calib/hook/sitecustomize.py`.
- Every kernel is attributed to its step through the launching HIP call. In every rank and window: 0 unattributed, 0 causality violations, 0 missing symbols.
- Tables: `scripts/calib/analysis/vllm_table.py`, `stage_rollup.py`.
- Full tables: `results/kfp-a2-prefill-table.md`, `results/kfp-b-agg-table.md`, plus `*-rollup.md`.

## 2. Prefill (kfp-a2, 4 ranks × 20 MIXED steps, 16,384 tokens/step) [F]

| # | kernel | share | bucket |
|---|---|---|---|
| 1 | ncclDevKernel_Generic (MoE-TP reduce-scatter / all-gather, incl. peer wait) | 29.05% | comm |
| 2 | opus_mla_v4_prefill_a16w16 | 15.09% | attention |
| 3 | mfma_moe1_silu_mul_afp8_wfp4 t128x128x256 | 14.20% | moe |
| 4 | gemm2_a4w4 (MoE down) | 7.88% | gemm |
| 5 | _inverse_rope_mxfp8_quant | 3.87% | quant |
| 6 | flydsl_mxpsh 128x256x128 | 2.96% | gemm (FlyDSL MXFP8) |
| 7 | bmm_a8w8_mxscale | 2.28% | gemm |
| 8 | flydsl_bpreshuffle_mxfp8 k8192 | 2.13% | gemm |
| 9 | _pa_mqa_logits_mxfp4 (indexer) | 1.94% | indexer |
| 10 | _engram_lookup | 1.80% | misc |
| 12–13 | aiter mega_mhc_kernel<5120,11,…> (two variants) | 1.76% + 1.70% | **mhc** |
| 20 | mhc_pre_big_fuse_tilelang_kernel | 0.73% | **mhc** |

mHC kernels per prefill step [F]:
- About 40 big mega_mhc launches: ~400–435 µs each, grid 262,144.
- About 40 small launches: ~10–15 µs each.
- 1 TileLang `mhc_pre_big_fuse`: 0.8–4.4 ms, rank-dependent.
- 2 `mhc_post_kernel`.

## 3. Decode (kfp-b proxy, 8 ranks, 571 DECODE steps + their sampling/draft) [F]

| # | kernel | share | calls/step | bucket |
|---|---|---|---|---|
| 1 | ncclDevKernel_Generic | 33.51% | 86 | comm |
| 2 | mfma_moe1 … t32x128x256 | 16.82% | 41.5 | moe |
| 3 | _sparse_attn_decode_gfx950_partial | 11.62% | 43 | attention |
| 4 | gemm2_a4w4 bm32 (MoE down) | 9.10% | 41.5 | gemm |
| 5 | mfma_moe1 t128 | 2.49% | 1 | moe |
| 7 | _sparse_attn_decode_reduce | 1.28% | 43 | attention |
| 11–12 | aiter mega_mhc_kernel<5120,7,…> (two variants) | 1.04% + 0.96% | 38 + 35 | **mhc** |

mHC in decode is about 1.4 ms per step on every rank, against a 42–48 ms step [F].

## 4. mHC shapes from the live run (input to the workset) [F: hook shape recorder, `mhc_shapes-<pid>.json`]

- Entry points called:
  - `rocm_aiter_ops.mhc_mega_post_pre_delayed_rms_norm`, with bf16 output (FFN seam).
  - `rocm_aiter_ops.mhc_mega_post_pre_delayed_rms_norm_mxfp8`, with MXFP8 output (attention seam).
  - Each takes a `[T, 4, 5120]` bf16 residual.
  - 20 calls of each per step (40 in total; the kernel-level count is about 80).
- Decode (eager calls inside the window): T = 174 … 384 (= batch × 6).
- Prefill: T = 16,384 (16k chunks), plus the tail-chunk Ts in the window list.
- Graph capture sizes: T = 1 … 1,024 (vLLM buckets).
- Op dispatch, code read at 9c1d45c2: `research/mhc_dispatch_9c1d45c2.md`.

## 5. Caveats

- NCCL time contains rank-skew waiting. That is why the NCCL-excluded shares are also shown.
- Decode is a colocated proxy, not a PD decode engine.
- Prefill is cold warmup traffic: early turns, short prefixes. Prefix-cache-heavy prefill was not measured.
- One window per profile. No replicate.
