###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Mooncake connector: accept ROCm's padded KV backing as block-major.

Baked into the engine image by Dockerfile.vllm.dsv41. Idempotent; exits non-zero
when its anchor is missing.

Why: on ROCm, vLLM pads the shared KV cache backing up to a 4 KiB multiple
(v1/worker/utils.py _allocate_kv_cache_buffer, whose comment warns that readers
of storage.nbytes() must tolerate the tail). The Mooncake connector decides
whether a layer is a packed, block-major region with
``tensor_blocks * block_stride == storage.nbytes()``, so the answer depends on
whether that engine's KV size happens to be 4 KiB-aligned — i.e. on its block
count. A prefill and a decode with different memory utilization then register
the same layers differently (one packed with shared group views, one not), and
every pull fails with "Mooncake registered group index mismatch ...
producer=0, consumer=-1". Seen on DeepSeek-V4.1-Flash P/D with prefill at
--gpu-memory-utilization 0.82 and decode at 0.9.

The fix applies the rule the connector already uses for a flat buffer: block-
major when the storage holds the blocks plus a tail shorter than one block.
Not upstream as of 2026-11-20.
"""

import os
import sys

import vllm

f = os.path.join(
    os.path.dirname(vllm.__file__),
    "distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py",
)
src = open(f).read()
MARKER = "storage.nbytes() - tensor_blocks * block_stride < block_stride"
if MARKER in src:
    print("mooncake-packed-rocm-tail: already patched")
    sys.exit(0)

old = (
    "                storage_is_block_major = (\n"
    "                    tensor_blocks > 0\n"
    "                    and tensor_blocks * block_stride == storage.nbytes()\n"
    "                )\n"
)
new = (
    "                # ROCm pads the KV backing to 4 KiB; a tail shorter than one\n"
    "                # block still leaves it block-major, as for a flat buffer above.\n"
    "                storage_is_block_major = (\n"
    "                    tensor_blocks > 0\n"
    "                    and 0 <= storage.nbytes() - tensor_blocks * block_stride < block_stride\n"
    "                )\n"
)
n = src.count(old)
if n != 1:
    print(f"mooncake-packed-rocm-tail: expected 1 anchor in {f}, found {n}")
    sys.exit(1)
open(f, "w").write(src.replace(old, new))
print("mooncake-packed-rocm-tail: patched", f)
