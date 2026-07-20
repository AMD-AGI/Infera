###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Apply the vLLM Mooncake KV-connector logical/kernel block-size ratio fix.

Baked into the engine image by Dockerfile.vllm (the `for f in vllm/*.py`
loop). Idempotent and self-locating, so a vLLM version bump degrades gracefully.

Why: some attention backends (ROCm Aiter MLA, and the DeepSeek-V3.2 / GLM-5.1
DSA "lightning indexer") force a *kernel* block size of 1 even when the user
pins --block-size 16. The KV tensor is then allocated at kernel (per-token)
granularity -- cache.shape[0] == num_logical_pages * ratio -- while the
scheduler still hands the connector *logical* page block_ids. The stock
connector registers num_blocks = cache.shape[0] and block_len =
stride(0)*elemsize, so base + page_id*block_len lands `ratio`x too early, RDMA
transfers empty rows, and the decode engine attends over all-zero prompt KV and
emits garbage from the first token (observed with GLM-5.1 fp8 mooncake PD; the
transfer also overflows -> "destination transfer region exceeds remote KV block
size").

The fix mirrors vLLM's NIXL connector: compute
ratio = logical_block_size // kernel_block_size in _sync_block_size_with_kernel
and register at logical-page granularity (num_blocks //= ratio,
block_len *= ratio). ratio == 1 for dense / matched-block models, so this is a
byte-for-byte no-op there.

Run inside a container with vLLM installed:
    docker exec <ctr> python3 patch_vllm_mooncake_blocksize.py
"""

import os
import sys

try:
    import vllm
except Exception:
    print("mooncake-blocksize: vLLM not importable — skipping")
    sys.exit(0)

f = os.path.join(
    os.path.dirname(vllm.__file__),
    "distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py",
)
if not os.path.exists(f):
    print(f"mooncake-blocksize: {f} not found (no mooncake connector in this image) — skipping")
    sys.exit(0)

src = open(f).read()
MARKER = "_physical_blocks_per_logical_kv_block"
if MARKER in src:
    print("mooncake-blocksize: already patched")
    sys.exit(0)

# Each edit: (old, new). All must match exactly once or the connector layout has
# drifted from the version this patch was generated against -> fail loudly.
reps = [
    # 1) init the ratio attribute (default 1) before block-size sync.
    (
        "        self.use_mla = self.model_config.use_mla\n"
        "        self._sync_block_size_with_kernel()\n",
        "        self.use_mla = self.model_config.use_mla\n"
        "        # Ratio of physical (kernel) blocks per logical (scheduler) block.\n"
        "        # Stays 1 unless the kernel block size is smaller than the scheduler\n"
        "        # block size (computed in _sync_block_size_with_kernel).\n"
        "        self._physical_blocks_per_logical_kv_block = 1\n"
        "        self._sync_block_size_with_kernel()\n",
    ),
    # 2) record the ratio when the kernel block size is smaller than logical.
    (
        "            assert self.block_size > kernel_block_size\n"
        "            self.block_size = kernel_block_size\n",
        "            assert self.block_size > kernel_block_size\n"
        "            self._physical_blocks_per_logical_kv_block = (\n"
        "                self.block_size // kernel_block_size\n"
        "            )\n"
        "            self.block_size = kernel_block_size\n",
    ),
    # 3) register at logical-page granularity (num_blocks //= ratio).
    (
        "                if tensor_size_bytes is None:\n"
        "                    tensor_size_bytes = cache.nbytes\n"
        "                    self.num_blocks = cache.shape[0]\n"
        "                assert cache.shape[0] == self.num_blocks, (\n",
        "                ratio = self._physical_blocks_per_logical_kv_block\n"
        "                if tensor_size_bytes is None:\n"
        "                    tensor_size_bytes = cache.nbytes\n"
        "                    self.num_blocks = cache.shape[0] // ratio\n"
        "                assert cache.shape[0] == self.num_blocks * ratio, (\n",
    ),
    # 4) one block_len spans a full logical page.
    (
        "                block_len = cache.stride(0) * cache.element_size()\n",
        "                block_len = cache.stride(0) * cache.element_size() * ratio\n",
    ),
]

for i, (old, new) in enumerate(reps, 1):
    n = src.count(old)
    if n != 1:
        print(
            f"mooncake-blocksize: edit {i} expected 1 occurrence, found {n} — "
            "mooncake_connector layout changed?"
        )
        sys.exit(1)
    src = src.replace(old, new)

open(f, "w").write(src)
print("mooncake-blocksize: patched", f)
