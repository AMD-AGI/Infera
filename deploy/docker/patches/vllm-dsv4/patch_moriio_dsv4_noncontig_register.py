###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Register non-contiguous DSv4 KV caches with MoRI via the low-level
``register_memory`` (data_ptr + storage bytes) instead of
``register_torch_tensor`` (which rejects non-contiguous tensors).

WHAT: for a non-contiguous tensor, register the storage span via
  register_memory(data_ptr, storage_nbytes - storage_offset_bytes, dev, loc)
  (same base as _compute_block_transfer_offsets); contiguous keeps the old path.
WHY: DSv4 fp8_ds_mla KV cache is a 576B-aligned non-contiguous view;
  register_torch_tensor forces contiguous and raises, and .contiguous() would copy
  to a new address, detaching from the buffer the model forward writes into.
DEPS: baked by Dockerfile.vllm; idempotent, no-op if anchor absent.
  Mirrors sglang's mori conn. VERSION: verified vllm 0.23.x ROCm, DSv4-Pro 1P1D.
"""

import os
import sys

try:
    import vllm
except Exception:
    print("moriio-dsv4-noncontig: vLLM not importable — skipping")
    sys.exit(0)

f = os.path.join(
    os.path.dirname(vllm.__file__),
    "distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py",
)
if not os.path.exists(f):
    print(f"moriio-dsv4-noncontig: {f} not found — skipping")
    sys.exit(0)

src = open(f).read()
MARKER = "# MORIIO-DSV4-NONCONTIG-PATCH"
if MARKER in src:
    print("moriio-dsv4-noncontig: already patched")
    sys.exit(0)

OLD = (
    "        try:\n"
    "            self.local_memory_metadata = self.moriio_engine.register_torch_tensor(\n"
    "                tensor\n"
    "            )\n"
)
NEW = (
    "        try:\n"
    "            # MORIIO-DSV4-NONCONTIG-PATCH: DSv4 fp8_ds_mla KV is a 576B-aligned\n"
    "            # non-contiguous view; register_torch_tensor rejects it. Register the\n"
    "            # storage span by ptr+len (same base as offsets), like sglang mori.\n"
    "            if not tensor.is_contiguous():\n"
    "                from mori.io.engine import TORCH_DEVICE_TYPE_MAP\n"
    "                _dev = tensor.device.index\n"
    "                _dev = -1 if _dev is None else _dev\n"
    "                _so_bytes = tensor.storage_offset() * tensor.element_size()\n"
    "                _ptr = tensor.data_ptr()\n"
    "                _nbytes = tensor.untyped_storage().nbytes() - _so_bytes\n"
    "                self.local_memory_metadata = self.moriio_engine.register_memory(\n"
    "                    _ptr, _nbytes, _dev, TORCH_DEVICE_TYPE_MAP[tensor.device.type]\n"
    "                )\n"
    "            else:\n"
    "                self.local_memory_metadata = self.moriio_engine.register_torch_tensor(\n"
    "                    tensor\n"
    "                )\n"
)

if src.count(OLD) != 1:
    print(
        f"moriio-dsv4-noncontig: anchor found {src.count(OLD)}x (expected 1) — "
        "moriio_engine layout drifted, skipping"
    )
    sys.exit(0)

src = src.replace(OLD, NEW, 1)
open(f, "w").write(src)
print(f"moriio-dsv4-noncontig: patched {f}")
