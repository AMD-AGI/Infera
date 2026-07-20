###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Apply the vLLM MoRIIO KV-connector logical/kernel block-size ratio fix.

Baked into the engine image by Dockerfile.vllm (the `for f in vllm/*.py`
loop). Idempotent and self-locating, so a vLLM version bump degrades gracefully.

Why: some attention backends (ROCm Aiter MLA, the DeepSeek-V3.2 / GLM-5.1 DSA
"lightning indexer") force a kernel block size of 1 even when the user pins
--block-size 16. The KV tensor is then laid out at *kernel* (per-token)
granularity -- for MLA the shape is (num_tokens, 1, head_dim) -- while the
scheduler still hands the connector *logical* page block_ids (each page = 16
tokens). The stock MoRIIO connector:
  1) hard-asserts `block_size == self.block_size` in register_kv_caches, where
     block_size is read from shape[-2] (== 1 here) -> 1 != 16 -> AssertionError
     crashes every worker at bring-up; and
  2) computes byte offsets as `block_id * block_stride` with a one-kernel-block
     transfer size, so even without the assert it would address token row
     `page_id` (instead of `page_id * 16`) and move a single token -> the decode
     side receives empty/garbage KV (same class of bug fixed for Mooncake).

The fix respects MoRIIO's own offset model: a logical page spans
ratio = scheduler_block_size // kernel_block_size kernel blocks, so it records
`self.block_size_ratio` in register_kv_caches (instead of asserting) and, in
_compute_block_transfer_offsets, expands the block index by `ratio` and scales
the transfer size by `ratio` to move whole logical pages. ratio == 1 for dense /
matched-block models, so this is a byte-for-byte no-op there.

Run inside a container with vLLM installed:
    docker exec <ctr> python3 patch_vllm_moriio_blocksize.py
"""

import os
import sys

try:
    import vllm
except Exception:
    print("moriio-blocksize: vLLM not importable — skipping")
    sys.exit(0)

f = os.path.join(
    os.path.dirname(vllm.__file__),
    "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py",
)
if not os.path.exists(f):
    print(f"moriio-blocksize: {f} not found (no moriio connector in this image) — skipping")
    sys.exit(0)

src = open(f).read()
MARKER = "block_size_ratio"
if MARKER in src:
    print("moriio-blocksize: already patched")
    sys.exit(0)

# Each edit: (old, new). All must match exactly once or the connector layout has
# drifted from the version this patch was generated against -> fail loudly.
reps = [
    # 1) record the logical/kernel ratio instead of asserting they're equal.
    (
        "        assert block_size == self.block_size\n"
        "        # TODO(tms): self.block_len needs to be per-layer for sliding window,\n",
        "        # `block_size` above is read from the KV tensor shape, i.e. the *kernel*\n"
        "        # block size. For most backends it equals the scheduler block size\n"
        "        # (cache_config.block_size). But some ROCm MLA / DSA-indexer backends\n"
        "        # (e.g. GLM-5.1 GlmMoeDsa) lay the KV cache out per kernel block of size 1\n"
        "        # (shape[-2] == 1, effectively per-token) while the scheduler still pages\n"
        "        # at cache_config.block_size (16). There one logical page spans\n"
        "        # `block_size_ratio` kernel blocks; record it so the transfer-offset math\n"
        "        # can address whole logical pages instead of asserting here. For\n"
        "        # matched-block models the ratio is 1 and behavior is identical to before.\n"
        "        assert self.block_size % block_size == 0, (\n"
        '            f"scheduler block_size {self.block_size} is not a multiple of "\n'
        '            f"kernel block_size {block_size}"\n'
        "        )\n"
        "        self.block_size_ratio = self.block_size // block_size\n"
        "        # TODO(tms): self.block_len needs to be per-layer for sliding window,\n",
    ),
    # 2) scale the transfer size by ratio (one transfer = one whole logical page).
    (
        "        transfer_size_byte = blksize * hn * hs * sz\n"
        "        per_block = 1 if is_mla else 2\n",
        "        # One logical (scheduler) page spans `ratio` physical/kernel blocks when\n"
        "        # the KV cache is laid out per kernel block (e.g. GLM-5.1 ROCm MLA stores\n"
        "        # it per-token: shape[-2] == 1 but cache_config.block_size == 16). A page\n"
        "        # then covers `ratio` consecutive kernel blocks, so each transfer must be\n"
        "        # `ratio` kernel blocks long and a scheduler block_id must be expanded by\n"
        "        # `ratio` to land on the page start. ratio == 1 (matched-block models)\n"
        "        # reproduces the original offsets byte-for-byte.\n"
        '        ratio = getattr(self, "block_size_ratio", 1)\n'
        "        transfer_size_byte = blksize * hn * hs * sz * ratio\n"
        "        per_block = 1 if is_mla else 2\n",
    ),
    # 3a) K offset: expand logical block_id to its first kernel block (lb * ratio).
    (
        "            # K\n"
        "            offset_local[w] = sz * (lb * block_stride)\n"
        "            offset_remote[w] = sz * (rb * block_stride)\n",
        "            # K. Expand the logical block_id to its first kernel block\n"
        "            # (lb * ratio); transfer_size_byte already spans the whole page.\n"
        "            offset_local[w] = sz * (lb * ratio * block_stride)\n"
        "            offset_remote[w] = sz * (rb * ratio * block_stride)\n",
    ),
    # 3b) V offset: same ratio expansion.
    (
        "                offset_local[w] = sz * (1 * local_ktov_stride + lb * block_stride)\n"
        "                offset_remote[w] = sz * (1 * remote_ktov_stride + rb * block_stride)\n",
        "                offset_local[w] = sz * (\n"
        "                    1 * local_ktov_stride + lb * ratio * block_stride\n"
        "                )\n"
        "                offset_remote[w] = sz * (\n"
        "                    1 * remote_ktov_stride + rb * ratio * block_stride\n"
        "                )\n",
    ),
]

for i, (old, new) in enumerate(reps, 1):
    n = src.count(old)
    if n != 1:
        print(
            f"moriio-blocksize: edit {i} expected 1 occurrence, found {n} — "
            "moriio_connector layout changed?"
        )
        sys.exit(1)
    src = src.replace(old, new)

open(f, "w").write(src)
print("moriio-blocksize: patched", f)
