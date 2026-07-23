###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Apply the MoRIIO MLA per-block transfer-geometry (page size) fix.

Baked into the engine image by Dockerfile.vllm (the `for f in vllm/*.py` loop).
Idempotent and self-locating, so a vLLM version bump degrades gracefully.

Symptom this fixes: DeepSeek-V4 / GLM-5.1 (block-scaled fp8 MLA + DSA lightning
indexer, kv-cache-dtype fp8) over MoRIIO PD produce WRONG output while prefill
queried directly is correct. DSv4 shows RIGHT-STRUCTURE / WRONG-FACT ("The capital
of France is" -> "a good idea..."); GLM shows total garbage ("is is is is..."). READ
and WRITE modes fail identically. Diagnosed 2026-07-22 by differential vs Mooncake
(correct on the same nodes/model) + live per-layer geometry instrumentation.

Root cause: `get_layer_transfer_geometry` (moriio_layout.py) MLA (3-dim) branch
derives the per-block transfer SIZE and STRIDE from the tensor *shape*:
    slot_size_bytes = latent_dim * element_size
    block_len       = block_size * slot_size_bytes      # shape[1] * inner bytes
    block_stride    = stride[0]                          # kernel-block stride
The authoritative per-scheduler-block page is `spec.page_size_bytes`. The shape-derived
values disagree with it in two independent ways, both of which corrupt the transfer:

  1. DeepSeek-V4 `fp8_ds_mla` (UE8M0 block-scaled, 576-byte-aligned): the page is
     PADDED, so `page_size_bytes == stride[0]*element_size` > `block_size*inner*es`.
     The dropped tail carries the per-block scale/alignment -> decode dequantizes with
     a stale scale -> facts garble, structure survives.
  2. GLM-5.1: the cache is laid out per KERNEL block of size 1 (`shape[1] == 1`,
     ~1.25M blocks) while the scheduler pages at `spec.block_size` (16). Here
     `page_size_bytes == 16 * slot` but `stride[0]*es == 1 * slot`, so the shape-derived
     block_len/stride are 16x TOO SMALL -> only 1/16 of each block's KV is moved (and to
     the wrong offset) -> total garbage.

Both are the same defect: the MLA branch must use `spec.page_size_bytes` (which is
alignment- AND scheduler/kernel-block-ratio aware) instead of shape-derived bytes.
`block_stride` is in ELEMENTS, so it becomes `page_size_bytes // element_size`.
Mooncake already does exactly this (it registers `stride(0)*es` and uses
`layer_spec.page_size_bytes` for MLA), which is why Mooncake PD is correct.

No-op for contiguous, matched-block caches (fp16/bf16 K/V: Qwen, Kimi): there
`page_size_bytes == stride[0]*es == block_size*inner*es`, so block_len and block_stride
are byte-for-byte unchanged. Only the padded (DSv4) / per-kernel-block (GLM) MLA caches
change. The K/V (5-dim) branches are left untouched (they already handle the
kernel/logical ratio via `kernel_blocks_per_block`).

Verified 2026-07-22, TP4 2-node MoRIIO PD, temp=0:
  DSv4-Pro: France->Paris, China->Beijing, PD == prefill-direct.
  GLM-5.1-FP8: (see working log / verification run).

Run inside a container with vLLM installed:
    docker exec <ctr> python3 patch_moriio_pagelen.py
Exits 0 (no-op) if the MoRIIO layout module isn't present.
"""

import os
import py_compile
import sys

try:
    import vllm
except Exception:
    print("moriio-pagelen: vLLM not importable — skipping")
    sys.exit(0)

f = os.path.join(
    os.path.dirname(vllm.__file__),
    "distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py",
)
if not os.path.exists(f):
    print(f"moriio-pagelen: {f} not found (no moriio layout module) — skipping")
    sys.exit(0)

src = open(f).read()
MARKER = "MoRIIO-MLA-PAGELEN-FIX"
if MARKER in src:
    print("moriio-pagelen: already patched")
    sys.exit(0)

# Anchor: the MLA (3-dim) branch. Replace its block_len line + the block_stride in the
# returned geometry with spec.page_size_bytes-based values. The three-line unpack + the
# return preamble uniquely identify the MLA branch (the bare block_len line recurs in the
# K/V branches, which we intentionally do NOT touch).
old = (
    "        num_blocks, block_size, latent_dim = shape\n"
    "        slot_size_bytes = latent_dim * element_size\n"
    "        block_len = block_size * slot_size_bytes\n"
    "        return LayerTransferGeometry(\n"
    "            num_blocks=num_blocks,\n"
    "            block_size=block_size,\n"
    "            block_len=block_len,\n"
    "            slot_size_bytes=slot_size_bytes,\n"
    "            block_stride=stride[0],\n"
)
new = (
    "        num_blocks, block_size, latent_dim = shape\n"
    "        slot_size_bytes = latent_dim * element_size\n"
    "        # MoRIIO-MLA-PAGELEN-FIX: use spec.page_size_bytes (authoritative:\n"
    "        # alignment- AND scheduler/kernel-block-ratio aware) for the transfer size\n"
    "        # and stride, not the shape-derived block_size*latent*es. DSv4 fp8_ds_mla:\n"
    "        # page is 576-aligned/padded so page > shape bytes (dropped tail = UE8M0\n"
    "        # per-block scale). GLM-5.1: cache is per-kernel-block of size 1 while the\n"
    "        # scheduler pages at spec.block_size(16), so page == 16*slot while shape\n"
    "        # bytes == 1*slot (16x short). Both corrupt the KV. Mirror Mooncake, which\n"
    "        # uses page_size_bytes. block_stride is in ELEMENTS -> page // element_size.\n"
    "        # No-op for contiguous matched-block K/V (page == stride[0]*es == shape bytes).\n"
    "        _mla_spec = layer_to_spec[layer_name]\n"
    "        _page_bytes = getattr(_mla_spec, 'page_size_bytes', None)\n"
    "        if _page_bytes is None:\n"
    "            _page_bytes = stride[0] * element_size\n"
    "        block_len = _page_bytes\n"
    "        mla_block_stride = _page_bytes // element_size\n"
    "        return LayerTransferGeometry(\n"
    "            num_blocks=num_blocks,\n"
    "            block_size=block_size,\n"
    "            block_len=block_len,\n"
    "            slot_size_bytes=slot_size_bytes,\n"
    "            block_stride=mla_block_stride,\n"
)

n = src.count(old)
if n != 1:
    print(
        f"moriio-pagelen: expected 1 match of the MLA geometry anchor, found {n} — "
        "moriio_layout changed? aborting"
    )
    sys.exit(1)

src = src.replace(old, new)
open(f, "w").write(src)
try:
    py_compile.compile(f, doraise=True)
except py_compile.PyCompileError as e:
    print(f"moriio-pagelen: py_compile FAILED after patch: {e}")
    sys.exit(1)
print("moriio-pagelen: patched", f)
