> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# Program: GLM-5.2-MXFP4 decode sparse MLA (triton) on MI355X

**GPU**: gfx950 (AMD Instinct MI355X, 256 CUs, 256 MB MALL) — **Backend**: triton

## Objective

Make `triton_sparse_mla_fwd` in `sparse_mla_kernel.py` faster for the GLM-5.2 decode leg while
staying correct. It is sglang v0.5.21's production kernel (`dsa_impl == "triton"`), called 78×
per TARGET_VERIFY (one per layer) and once per DRAFT_EXTEND in every speculative iteration; it is
**13.8 %** of the distribution-weighted decode iteration (≈ 8.2 ms of 59.4 ms). Every µs saved per
call is ×79 per iteration.

## Shapes (per attention-DP rank; driver cases)

T = 6 × bs_graph query rows; bs_graph ∈ {8, 10, 12, 14} with P = 2 / 42 / 49 / 7 %.
`q_nope` fp8_e4m3 [T, 64, 512], `q_rope` fp8_e4m3 [T, 64, 64], KV pool fp8_e4m3
[2,226,368, 1, 576] (latent 512 = also V, rope 64), `indices` int32 [T, 1, 2048] global token
slots (contexts ≫ 2048, so all 2048 valid), `sm_scale` 0.0625, `d_v` 512,
`topk_length=None`, `max_topk_length=2048`. Output bf16 [1, T, 64, 512].

Scored cases: **T60_bs10 and T72_bs12** (91 % of the mass). T48_bs8 and T84_bs14 are timed and
must stay correct but are unscored. The driver times each case over 8 distinct KV pools per
graph replay (cold-ish cache, like 78 layers) and reports per-call ms.

Baseline (pristine, this driver, compute-node-12): bs10 0.0898 ms, bs12 0.1162 ms, bs8 0.0908 ms,
bs14 0.1163 ms per call — matches the in-model profile (≈ 89 / 114 µs per call).

## What the code does today (read it; measure everything)

`base_ctas = T × ceil(64/16)`; with 256 CUs, bs8/bs10 (192/240 CTAs) take the split-K path
(`_kv_splits_heuristic` → `_gfx950_sparse_mla_kv_splits`, partials + `_sparse_mla_reduce_kernel`),
bs12/bs14 (288/336 CTAs) take the gfx950 fp8 `kv_splits=1` path. The 2048 gathered KV rows per
query row (1.18 MB) are shared by all 64 heads.

## Correctness

The gate is SNR ≥ **25 dB** vs an fp32 reference (min over all cases and two index sets).
Note the pristine kernel itself scores only **26.5–27 dB on bs8/bs10 (split-K path)** and
36.5 dB on bs12/bs14 — the split-K partial/reduce precision is the weak spot. Improving that
precision is welcome; going below 25 dB is not allowed.

## Modification rules

1. Keep `triton_sparse_mla_fwd(q_nope, q_rope, kv, indices, sm_scale, d_v=512, topk_length=None,
   max_topk_length=None)` and its semantics/return shape — sglang calls it directly.
2. Stay in Triton (it must drop back into sglang's `kernels/ops/attention/dsa/triton_sparse_mla.py`).
   Other shapes/dtypes of the original dispatch (bf16 KV, other head counts, CUDA) may keep their
   current paths, but must not break.
3. Do NOT edit `driver.py`, `graph_harness.py` or `data/`.
4. No caching of results across calls, no dependence on the driver's fixed inputs.
