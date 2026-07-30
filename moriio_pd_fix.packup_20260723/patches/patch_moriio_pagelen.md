# patch_moriio_pagelen.py — what / why / how / context

**File:** `patch_moriio_pagelen.py` (copied verbatim from
`deploy/docker/patches/vllm/patch_moriio_pagelen.py` at git 29021d3).
**Target:** `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py`,
the `get_layer_transfer_geometry` MLA (3-dim) branch.

## What
Change the MLA branch to use the authoritative page size instead of shape-derived
bytes:
```
block_len    = spec.page_size_bytes                     # was block_size * latent_dim * element_size
block_stride = spec.page_size_bytes // element_size     # was stride[0]
```

## Why
vLLM v0.25.1's MoRIIO connector derives the MLA per-block transfer SIZE and STRIDE
from the KV tensor's `.shape`/`.stride()`, not from `spec.page_size_bytes`. For
block-scaled fp8 MLA caches this is wrong in two independent ways — one bug, two
symptoms:

- **DeepSeek-V4-Pro (`fp8_ds_mla`, padded, ratio 1):** the physical per-block page
  is 576-byte-aligned, so `page_size_bytes == stride[0]*es > block_size*latent*es`.
  The dropped tail carries the UE8M0 per-block scale → decode dequantizes with a
  stale scale → RIGHT-STRUCTURE / WRONG-FACT output ("France"→"a good idea").

- **GLM-5.1-FP8 (per-kernel-block=1, ratio 16):** the cache is laid out per kernel
  block of size 1 (`shape[1]==1`, ~1.25M blocks) while the scheduler pages at
  `spec.block_size=16`. So `page_size_bytes == 16*slot` but `stride[0]*es == 1*slot`
  — the shape-derived length is **16× too small** → only 1/16 of each block's KV is
  moved → TOTAL garbage ("is is is").

Mooncake is correct on the same nodes/model because it already transfers
`cache.stride(0)*es` and uses `layer_spec.page_size_bytes` for MLA. This patch
mirrors Mooncake.

## How
Idempotent, self-locating, anchor-matched string replace (same pattern as sibling
`patch_moriio_*.py`). The anchor is the 3-line MLA unpack + the `LayerTransferGeometry`
return preamble — unique to the MLA branch (the bare `block_len = ...` line recurs in
the K/V branches, which are intentionally NOT touched). Applied by the Dockerfile.vllm
`for f in /tmp/vllm-patches/*.py` loop; also runnable manually:
`docker exec <ctr> python3 patch_moriio_pagelen.py`. Verified: anchor matches pristine
v0.25.1 exactly once; py_compile passes; 2nd run prints "already patched".

## Context (what symptom it cured)
- Necessary AND sufficient for BOTH models. No-op for contiguous matched-block
  fp16/bf16 K/V (Qwen, Kimi): there `page_size_bytes == stride[0]*es == shape bytes`,
  so block_len/stride are byte-for-byte unchanged.
- This is a genuine upstream vLLM v0.25.1 MoRIIO bug, NOT the infera router protocol
  (`infera/router/disagg_protocols/vllm_moriio.py`). The forged request_id wire-shape
  matches v0.25.1's `_PREFILL/_DECODE_ZMQ_RE` and was correct throughout.
- The three older GLM MoRIIO patches (`patch_moriio_dsa_write`, `patch_moriio_hetero`,
  `patch_vllm_moriio_blocksize`) are no-op on v0.25.1 and did NOT cover this — their
  concerns (layer coverage, per-layer geometry) are handled natively in v0.25.1, but
  v0.25.1 introduced this new page-size bug.

## Provenance
An earlier draft used `block_len = stride[0]*element_size`. That fixes DSv4 (padded)
but is still 16× short for GLM (contiguous per-token stride). The shipped version uses
`spec.page_size_bytes`, which is correct for both. See `../notes.md` for the full
debug narrative.
