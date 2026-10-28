> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# Program: GLM-5.2-MXFP4 DCP4 decode sparse MLA (split-K, owned slots + LSE) on MI355X

**GPU**: gfx950 (AMD Instinct MI355X, 256 CUs, 256 MB MALL, HBM ~8 TB/s) — **Backend**: triton

## Objective

Make `dcp_sparse_mla_decode` in `dcp_sparse_kernel.py` faster while staying correct. It is the kernel call of sglang
93034eba91 (v0.5.22 overlay) `DeepseekSparseAttnBackend._forward_triton_dcp` (dsa_backend.py:3632) = the sparse MLA of
the **DCP4 decode leg** (TP4 EP4 DP1, `--dcp-size 4`, triton DSA) of the current best GLM-5.2 PD config (ppdc,
historical-e2e-workspace `registry/glm52.ppdc-pp4-dcp4.packup_20261023`). It runs once per layer (78) per TARGET_VERIFY.

## Shapes (one DCP rank; driver cases)

- DCP decode is DP1: every rank holds ALL running requests. T = 6 x bs_graph verify rows, bs_graph in
  {24, 32, 40, 48, 64} with P = 4.6 / 16 / 26 / 22 / 21 % (running requests of ppdc C56 + C72 snapped to the capture list).
  Scored: T192_bs32, T240_bs40, T288_bs48, T384_bs64 (86 % of the mass); T144_bs24 timed, unscored.
- q heads are all-gathered over the DCP group (16 local x 4 = **64 heads**): `q_nope` fp8_e4m3 [T, 64, 512],
  `q_rope` fp8_e4m3 [T, 64, 64].
- KV: this rank's shard, fp8_e4m3 [2,365,184, 1, 576] (global slot s lives on rank s % 4 at local slot s // 4).
- `indices` int32 [T, 1, 2048]: the row's global top-2048, localized (owned ones packed first as local slots, then -1);
  `lengths` int32 [T] = owned count, ~512 (binomial 2048 x 1/4). sm_scale 0.0625, d_v 512.
- Output: bf16 [1, T, 64, 512] over the owned slots only + fp32 base-2 LSE [T, 64] (the caller merges ranks by LSE;
  an empty row must return LSE -2**30 and must not be -inf/NaN).

Baseline (this driver, pristine, compute-node-10 GPU 1): T144 0.0461, T192 0.0540, T240 0.0583, T288 0.0745, T384 0.0946 ms.
Per-row work is ~4x smaller than the non-DCP decode (512 instead of 2048 slots) while rows are ~4x more.

## Correctness

fp32 reference over the owned slots: output SNR >= 30 dB (pristine 35.6) AND |LSE - ref| <= 0.05 (log2 units) on every
case; a case failing the LSE bound scores -100 dB. Keep `lengths` semantics: only the first lengths[i] indices of a row
are valid, the rest are -1.

## Modification rules

- Edit only `dcp_sparse_kernel.py`; keep `dcp_sparse_mla_decode`'s signature and return contract.
- Must stay CUDA-graph safe (no host sync, no data-dependent host branching on tensor values, allocation shapes depend
  only on input shapes); the driver times it under graph replay over 8 KV shards.
- The non-DCP functions in the file (triton_sparse_mla_fwd etc.) may be changed only if the DCP path still uses them.
