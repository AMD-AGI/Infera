> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# Program: GLM-5.2-MXFP4 decode DSA top-k indexer (gluon logits + top-2048) on MI355X

**GPU**: gfx950 (AMD Instinct MI355X, 256 CUs, 256 MB MALL). **Backend**: Triton/Gluon (logits)
plus a compiled sgl-kernel HIP top-k (replaceable).

## Objective

Make `dsa_indexer_topk_paged` in `dsa_indexer_kernel.py` faster while staying correct. It is
the computation inside sglang v0.5.21 `Indexer._get_topk_paged` for the TARGET_VERIFY rows of
the GLM-5.2 decode leg: paged MQA logits over the fp8 index-K cache, then top-2048 selection with
page transform. It runs 21× per TARGET_VERIFY (layers 0,1,2,6,10,…,74; the other layers reuse
the top-k via `index_share_for_mtp_iteration`) plus once per DRAFT_EXTEND, and it is **12.4 %**
of the distribution-weighted decode iteration — the only decode op whose cost grows with context.

## Production call path (image `historical-sglang-image-d`)

sglang 11bfd868, aiter e7d2453, triton 3.7.0. Deployment: SGLANG_USE_AITER=1, `--dsa-topk-backend
sgl-kernel`, SGLANG_DSA_FUSE_TOPK default (true), **SGLANG_OPT_USE_TOPK_V2=false** (set by the
GLM-5.2 PD recipe `bench/glm5p2_pd/config.sh`; see "Assumptions"), EAGLE 6 draft tokens, page 64.

1. `srt/layers/attention/dsa/dsa_indexer.py:1878` `forward_cuda` → `:1957` gfx950 fused decode
   indexer gate → declines for > 48 rows (`hip_gfx950/fused_decode.py:31` `MAX_ROWS = 48`,
   `dsa_indexer.py:761`) → standard path → `:2163ff` → `_get_topk_paged` (`:1191`).
2. `dsa_indexer.py:1352` `aiter_paged_mqa_logits(q_fp8, kv_cache_fp8, weights, seqlens_expanded,
   real_page_table, max_seq_len, preshuffle=True, kv_block_size=64)`
   (`kernels/ops/attention/dsa/paged_mqa_logits.py:95`) →
   `aiter/ops/triton/attention/pa_mqa_logits.py:465` `deepgemm_fp8_paged_mqa_logits(Preshuffle=True,
   KVBlockSize=64)` → gluon kernel `aiter/ops/triton/gluon/pa_mqa_logits.py:412`
   `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` (grid rows × SplitKV, SplitKV = 10 here).
3. `dsa_indexer.py:1421` `_mask_init_and_local_tokens` — no-op for GLM-5.2 (no init/local tokens).
4. `dsa_indexer.py:1422` `metadata.topk_transform` → `dsa_topk_backend.py:96` →
   `:204` `sgl_kernel.fast_topk_transform_fused(score=logits, lengths=seqlens_expanded,
   page_table_size_1=page_table_1, cu_seqlens_q=arange(rows+1), topk=2048)`
   (kernel `sgl::dsa_topk::coop_topk_kernel<2048,12,4096,1024>`).

The kernel file holds verbatim copies of the gluon kernel and its host wrapper (reduced to this
path) and of the sglang glue. Pristine copy = production, verified first-hand: logits bitwise equal
to the image's own `aiter_paged_mqa_logits` over every row's [0, L), top-k index sets equal to the
image's `fast_topk_transform_fused` (bs10, bs12; `VALIDATION.md`).

**bs8 note**: 48 rows ≤ `MAX_ROWS`, so in production bs8 TARGET_VERIFY takes the gfx950 fused
four-kernel indexer (GEMV + rope/Hadamard/quant/cache-write + logits_hist + topk), a different op
boundary (code read; the GLM-5.2-MXFP4 indexer weights wk/weights_proj/wq_b are bf16 — excluded from
Quark quantization in the checkpoint config — so the weight gate passes). This driver times bs8
through the standard path; bs8 is unscored.

## Shapes (per attention-DP rank; driver cases)

rows = 6 × bs_graph, bs_graph ∈ {8, 10, 12, 14}, P = 2.0 / 42.3 / 48.7 / 6.9 %.
- `q_fp8` float8_e4m3fn [rows, 32, 128]; `weights` fp32 [rows, 32]
- `kv_cache_fp8` uint8 [34,787 pages, 64 × 132]: per page 64×128 fp8 K in the aiter MFMA-16×16
  **preshuffled** layout, then 64 fp32 ue8m0 scales — written in the driver by the production writer
  `aiter.ops.cache.indexer_k_quant_and_cache(k, cache, slots, 128, "ue8m0", preshuffle=True)`.
- `seqlens_expanded` int32 [rows] = L+1 … L+6 per request (workset blobs).
- `real_page_table` int32 [rows, 16,386], `page_table_1` int32 [rows, 1,048,651]: row-expanded, as
  wide as the decode CUDA graph's tables (req_to_token width = context 1,048,576 + 75), so the logits
  buffer is fp32 [rows, 1,048,704] (only [0, L) of each row is written/read).
- output int32 [rows, 2048] physical token slots, -1 padded (contexts here are 27 K–660 K ≫ 2048).

Scored: **T60_bs10, T72_bs12** (91 % of the mass); T48_bs8 / T84_bs14 timed, unscored, must stay
correct. Timing: CUDA-graph replay of 4 calls over 4 distinct index-K pools (294 MB each, > MALL),
per-call ms reported; contexts = workset `ctx0` blobs (heterogeneous lognormal contexts).

## Pristine baseline (this driver, node compute-node-01, GPU 2)

| case | run A ms | run B ms |
|---|---|---|
| T48_bs8 (unscored) | 0.376482 | 0.377443 |
| T60_bs10 | 0.364913 | 0.365972 |
| T72_bs12 | 0.488164 | 0.487903 |
| T84_bs14 (unscored) | 0.471003 | 0.470923 |
| median_ms (P-weighted) | 0.432564 | 0.432900 |

Spread A/B: 0.08 % on median_ms (two earlier runs on a docstring-only-different file: 0.432330,
0.431912). Pristine SNR per case: bs8 137.6/146.2, bs10 153.8/150.3, bs12 154.8/148.9, bs14
100.0/144.0 dB (ctx0/ctx1); min 100.00 dB.

Breakdown (same pools, CUDA graph): bs10 logits 274 µs + top-k 86 µs; bs12 logits 308 µs + top-k
174 µs. Anchors: with *uniform* contexts at the anchors' ISLs (bs8 230K, bs10 210K, bs12 180K, bs14
155K) the same code gives 272 / 342 / 340 / 338 µs vs the in-model 280 / 337 / 334 / 340 µs. The
workset ctx0 contexts are heterogeneous (bs12: mean 181 K, max 433 K) and cost more: bs12 logits
308 vs 246 µs, top-k 174 vs 87 µs. Each row is split into a fixed 10 KV chunks, so the longest row
sets the tail — that is where the work imbalance sits (measured; the top-k doubling mechanism is not
established).

## Correctness

Top-k index sets are not SNR-comparable, so the driver (fib DsaTopkIndexer idea) computes fp32
reference scores `sum_h relu(q·K)·w` from the dequantized cache (workset reference semantics), and per
row requires validity — exactly min(L, 2048) non-negative slots, no duplicates, every slot among the
row's first L tokens — then compares the reference scores at the returned tokens (sorted) with the
reference top-2048 scores (sorted). SNR over all rows of a case; any invalid row → -100 dB. Gate:
min over 8 cases (4 bs × 2 context sets) ≥ **60 dB** (orchestrator: tightened from 30, ≈ recall 0.98). Pristine: 100–155 dB per case (100 = exact),
top-k set recall ≥ 0.99996 (`VALIDATION.md`).

Adaptations of the workset oracle, explicitly: (1) production stores K preshuffled; the oracle
dequantizes the same K written by the same writer with `preshuffle=False` (the workset layout);
(2) scores compared, not index sets, because near-ties make the set order-/rounding-dependent.

Calibration of the metric (bs12, selection perturbed on purpose): Gaussian noise 1e-2·σ on scores →
68 dB / recall 0.990; bf16-rounded scores → 74 dB / 0.995; noise 1e-1·σ → 38 dB / 0.905; dropping the
best 100 tokens → 26 dB; random tokens → -0.2 dB. **A 30 dB gate admits selections with recall ≈ 0.85;**
keep the scores exact (fp32 accumulate, fp8 inputs as given) — that is what production returns.

## Modification rules

1. Keep `dsa_indexer_topk_paged(q_fp8, kv_cache_fp8, weights, seqlens_expanded, real_page_table,
   page_table_1, cu_seqlens_q, index_topk=2048)` and its semantics (exact top-2048 of the logits,
   physical slots via the page tables, -1 padding) — it must drop back into sglang `_get_topk_paged`
   / aiter `deepgemm_fp8_paged_mqa_logits`.
2. Keep the cache layout (aiter preshuffle, page 64, 132 B/token) and dtypes: the writer is shared
   with prefill and the fused paths.
3. The top-k may be replaced (Triton/Gluon in this file, or fused into the logits kernel); the
   `sgl_kernel` C++ source is not in this file. Rows with L ≤ 2048 must still return all L tokens.
4. Do NOT edit `driver.py`, `graph_harness.py` or `data/`. No caching across calls, no dependence on
   the driver's fixed inputs (contexts vary per call in serving; the page-table widths are fixed by
   the CUDA graph).

## Assumptions (labelled)

- SGLANG_OPT_USE_TOPK_V2=false: from the GLM-5.2 PD recipe (`config.sh` default
  `SGLANG_OPT_USE_TOPK_V2="${SGLANG_OPT_USE_TOPK_V2:-false}"`, seen in
  historical-e2e-workspace/registry/glm52.cluster-1p1d-v0521-agentx-c44_20261009.packup_20261012/patches/image-recipe)
  and the workset description; the image's env default is true (`srt/environ.py:1336`), which
  would route to `dsv4.topk.topk_transform_paged_v2` instead.
- context_len = 1,048,576 (model `max_position_embeddings`; no `--context-length` in the v0521
  dryruns), extra 75 from `get_req_to_token_extra_context_len()` (EAGLE 5/1/6, page 64).
- Random K ~ N(0,1) bf16, q ~ N(0,1) clamped to ±4, w ~ N(0, 1/32): not production activations.
