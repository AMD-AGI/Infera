> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# Program: GLM-5.2 gfx950 FUSED decode DSA indexer (TARGET_VERIFY rows 6..48) on MI355X

**GPU**: gfx950 (AMD Instinct MI355X, 256 CUs, 256 MB MALL, HBM ~8 TB/s) — **Backend**: HIP (sglang JIT, tvm-ffi)

## Objective

Make `fused_decode_indexer` in `fused_indexer_kernel.py` faster while staying exact. It is sglang's
`Gfx950FusedIndexer.run` (srt/layers/attention/dsa/gfx950_fused_indexer.py): `Indexer.forward_cuda` routes EVERY
DECODE / TARGET_VERIFY indexer call with 1 <= rows <= 48 (per-rank decode batch of 1..8 requests x 6 EAGLE tokens)
to these four kernels instead of the standard indexer. On the GLM-5.2 DPA decode leg (TP4/DP4, EAGLE 5/1/6,
index_share_for_mtp_iteration=true -> 21 indexer layers per TARGET_VERIFY) such small per-rank batches are 33 % of decode
steps at C80 (contributor 4+4) and 88-93 % at C56 (same-node P4D4). Nobody has optimized this path yet.

Four nodes (all HIP sources are in this file as strings, JIT-built from a content-addressed temp dir with the
production flags; edit them or replace any node):
1. `dual_gemv_bf16`: q_proj = q_lora[rows, 2048] @ wq_b[4096, 2048].T and kw = x[rows, 6144] @ [wk ; weights_proj][160, 6144].T
2. `indexer_qk_rope_hadamard_quant_and_cache`: k LayerNorm, rope (interleaved), Hadamard on q and k, fp8 e4m3 act-quant
   of q (ue8m0), quantize + WRITE this step's index-K into the paged cache, head gate
3. `logits_hist`: paged fp8 MQA logits over the row's whole context (32 heads x 128, page 64, cache row = 64 x 132 B:
   128 fp8 + fp32 scale) + per-row 12-bit histogram
4. `topk_transform`: top-2048 + page-table transform to physical slots, guided by the histogram

## Shapes (driver cases)

rows = 6 x bs, bs in {2, 4, 6, 8} (P = 24 / 31 / 29 / 16 %; all scored). Per request: context L lognormal (median
180k tokens, sigma 0.9, clipped to [4096, 600k]); its 6 rows have lengths L+1..L+6 and share the page table.
Index-K pool: 34,787 pages per rank (production writer, preshuffled). Weights bf16 (random, fixed seed).

Baseline (pristine, this driver, compute-cluster compute-node-09): T12 0.0509, T24 0.0869, T36 0.1308, T48 0.1640 ms per call.
At these contexts node 3 (logits over ~180k tokens per row, 6 rows per request reading the SAME pages) is expected
to dominate; the 6 sibling rows of a request share their page table (exploitable: load each K page once).

## Correctness

Compared with the image's own production kernels on the same inputs: every row's top-2048 slot SET must be identical
(order free) AND the bytes of this step's written K slots must be identical. case_snr = 100 dB only then; gate 60 dB.

## Modification rules

- Edit only `fused_indexer_kernel.py`; keep `fused_decode_indexer`'s keyword signature, output ([rows, 2048] int32
  slots, -1 padded) and the K write. `Workspace` may change (it is constructed once per case, outside timing).
- CUDA-graph safe: no host sync, no allocation that depends on data, nothing data-dependent on the host.
- Rows can be any 1..48 in production (not only multiples of 6); keep it correct for any rows / lengths.
