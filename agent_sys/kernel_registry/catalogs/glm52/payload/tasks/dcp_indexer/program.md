> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# Program: GLM-5.2-MXFP4 DCP4 decode split indexer (one DCP rank) on MI355X

**GPU**: gfx950 (AMD Instinct MI355X, 256 CUs, 256 MB MALL, HBM ~8 TB/s) — **Backend**: Triton/Gluon + sgl-kernel

## Objective

Make `dcp_indexer_topk` in `dcp_indexer_kernel.py` faster while staying exact. It is what sglang 93034eba91 (v0.5.22
overlay) runs per indexer layer on each DCP rank of the **DCP4 decode leg** (TP4 EP4 DP1 `--dcp-size 4`,
SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1 + _TRITON=1) of the current best GLM-5.2 PD config (ppdc, historical-e2e-workspace
`registry/glm52.ppdc-pp4-dcp4.packup_20261023`): `Indexer._dcp_split_decode_topk` -> `split_decode_topk`. It runs 21x per
TARGET_VERIFY (index_share_for_mtp_iteration=true: layers 0,1,2,6,10,...,74).

Pipeline today (see the module docstring): aiter gluon paged-MQA logits over this rank's pages (r, r+4, ...) ->
pad -> `sgl_kernel.fast_topk_v2` local top-2048 -> triton pack [2, T, 2048] -> (all_gather: input) -> triton merge of
4 x 2048 candidates to the global top-2048 + position -> KV slot transform.

## Shapes (rank 0; driver cases)

T = 6 x bs_graph rows, bs_graph in {24, 32, 40, 48, 64}, P = 4.6 / 16 / 26 / 22 / 21 %; scored T192..T384 (86 %),
T144 timed unscored. q fp8 [T, 32, 128], weights fp32 [T, 32]; index-K pool replicated on every rank: 147,824 pages x
64 tokens x (128 fp8 + 4 B ue8m0 scale) in aiter's preshuffled layout (production writer `indexer_k_quant_and_cache`).
Contexts lognormal (median 89k, p90 283k tokens); this rank scores ~L/4 tokens per row. `owned_tables` [T, P/4],
`owned_lens` [T], `real_page_table` [T, P], `other_packs` [6, T, 2048] (ranks 1..3).

Baseline (pristine, this driver, compute-node-10 GPU 2): T144 0.231, T192 0.362, T240 0.373, T288 0.409, T384 0.353 ms.

## Correctness

The output must be the exact global top-2048 slot SET of every row (order free): the driver compares against a torch
re-implementation of all 4 ranks (production logits, torch.topk, torch pack/merge). case_snr = 100 dB only if every row
matches; gate 60 dB (recall >= 0.999999). Logits may change bit patterns only if the selected sets stay identical.

## Hints / prior art

- `REDACTED_LOCAL_PATH` is a KF-optimized NON-DCP version of the same
  logits kernel (gluon, preshuffle, SplitKV / LPT changes, fused logits path) plus a 3-kernel triton top-2048 selector
  (est / collect / select) that replaced sgl-kernel's top-k: 2.8-3.7x on rows 48-84 over full contexts. Here rows are
  ~4x more and contexts ~4x shorter per rank, and the top-k is local (columns = owned positions, no page transform).
- The merge reads 4 x 2048 candidates per row; pack + merge are small but per-layer.

## Modification rules

- Edit only `dcp_indexer_kernel.py`; keep `dcp_indexer_topk`'s signature and output contract ([T, 2048] int32 slots,
  -1 padded). You may replace the aiter / sgl-kernel calls with in-file kernels.
- CUDA-graph safe: no host sync, no data-dependent host control flow, shapes from input shapes only.
