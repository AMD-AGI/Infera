# Notes — 06 PD mooncake RDMA + MTP

## pd-unified MTP is SIMPLER than rc6 MTP — one fix, not two

On the **rc6** image (experiments 04/05), enabling GLM-5.2 MTP needed TWO fixes:
1. a 1-line nextn `eh_proj` patch, and
2. env `SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA=0` (to dodge a gfx950-incompatible CUDA
   `fused_metadata_copy` JIT kernel that otherwise hangs decode).

On the **pd-unified** image, only fix #1 is needed. Why fix #2 is unnecessary here:
- pd-unified's `dsa_backend.py` wraps the multi-backend CUDA kernel in `try: ... except (ImportError,
  Exception): <fallback to a plain per-backend loop>`. So even if the CUDA kernel fails to compile on
  gfx950, it's caught and the loop runs — no hang.
- And for `speculative_num_steps <= 3` it takes an `else` branch that uses the plain loop directly,
  never touching the CUDA kernel. We use **EAGLE steps=3**, so the kernel isn't even reached.
- The env `SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA` does not exist in this image anyway (the outer
  `if envs.…get()` gate that rc6 had is gone).

## Fix #1 — the nextn patch is image-specific

pd-unified's `deepseek_nextn.py` is **423 lines** (rc6's was 365) and has the same bare-prefix quark-
exclude bug but at **line 363** (rc6: line 305). The fix is identical in spirit — append `.eh_proj`
so the submodule-level quark exclude (`model.layers.78.eh_proj`) matches, so `eh_proj` is built bf16
instead of MXFP4-packed:

    - ckpt_prefix = f"model.layers.{config.num_hidden_layers}"
    + ckpt_prefix = f"model.layers.{config.num_hidden_layers}.eh_proj"

But because the surrounding file differs from rc6, you must patch **this image's** stock file, not
reuse the rc6 patch (`patches/deepseek_nextn.unified.diff` is the pd-unified one). Reusing a
cross-version nextn file is exactly what broke the very first MTP attempt (see 04/notes §Bug 1).

## Decode-leg MTP tuning (PD KV-pool stability)

Same as 05: EAGLE **steps=3** (not 5), `num-draft-tokens 4`, `num-reserved-decode-tokens 256`,
decode `mem-fraction 0.80`. Keeps the draft-extend KV allocation inside the PD decode pool under
conc=64 (no crash). Env-overridable in `pd_leg_mtp.sh`: `SPEC_STEPS`, `SPEC_DRAFT`, `RESERVED_TOK`.

## Result vs no-MTP mooncake PD (03)

conc=64, same workload/transport, MTP on decode:
- total throughput 5147 → 5302 tok/s (+3%)
- median TPOT 20.9 → 19.0 ms (spec-dec faster per-token)
- accept len 2.65–2.90 (of 4 drafts), still 256/256.

The MTP gain here is modest (+3%) vs the mori PD-MTP run's +44% (05), mainly because this run used a
shorter warmup and the 1k/1k workload is more prefill-weighted; the spec-dec is genuinely active
(accept len ~2.7). For a larger MTP win, use a longer-output / higher-cache-hit workload.

## Transport

Same as 03 — real mooncake RDMA (prefill log: 8× `rdma_context.cpp HIP dmabuf disabled`), dmabuf
OFF, zero transfer errors. MTP works across the RDMA PD boundary: the draft head loads and accepts
tokens on the decode node while KV arrives over mooncake RDMA from prefill.
