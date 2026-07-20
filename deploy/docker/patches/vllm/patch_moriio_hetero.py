###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Apply the MoRIIO heterogeneous-KV-shape read-transfer fix.

Baked into the engine image by Dockerfile.vllm (the `for f in vllm/*.py`
loop). Idempotent and self-locating, so a vLLM version bump degrades gracefully.

GLM-5.1 (GlmMoeDsa: MLA + DSA "lightning indexer") registers TWO differently
shaped KV caches: the attention latent (..., 1, 576) and the indexer (..., 1, 132).
They have different head_dim, hence different per-block stride and transfer size.

The stock MoRIIO connector does NOT handle this:
  1) `_compute_block_transfer_offsets` reads block dims from the GLOBAL
     `self.kv_cache_shape` (set to the FIRST layer's shape in register_kv_caches),
     not from the layer it is computing offsets for; and
  2) `_read_blocks` computes the offsets ONCE for `first_layer` and reuses the
     same byte offsets/sizes for EVERY layer's RDMA read.
So whichever layer type differs from the first layer's shape is read with the
wrong stride/size -> the decode side gets corrupted/empty KV -> the model emits
degenerate garbage ("0,0,0,..."). HTTP is 200 (KV "transferred"), output is wrong.

Fix (read path):
  (1) derive block dims from the PER-LAYER shape `self.kv_caches[layer_name].shape`;
  (2) compute `_compute_block_transfer_offsets` inside the per-layer loop.
For homogeneous models every layer shares one shape, so both changes are a
byte-for-byte no-op (the recomputed offsets are identical each iteration).

Run inside a container with vLLM installed:
    docker exec <ctr> python3 patch_moriio_hetero.py
Exits 0 (no-op) if the MoRIIO connector isn't present.
"""

import os
import py_compile
import sys

try:
    import vllm
except Exception:
    print("moriio-hetero: vLLM not importable — skipping")
    sys.exit(0)

f = os.path.join(
    os.path.dirname(vllm.__file__),
    "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py",
)
if not os.path.exists(f):
    print(f"moriio-hetero: {f} not found (no moriio connector in this image) — skipping")
    sys.exit(0)

src = open(f).read()
MARKER = "_layer_shape = tuple(self.kv_caches[layer_name].shape)"
if MARKER in src:
    print("moriio-hetero: already patched")
    sys.exit(0)

# --- Edit 1: per-layer shape in _compute_block_transfer_offsets ---------------
old1 = (
    '        assert self.kv_cache_shape is not None, "KV caches shape not initialized"\n'
    "        is_mla = len(self.kv_cache_shape) == 3\n"
    "        stride = self.kv_caches[layer_name].stride()\n"
    "        sz = self.kv_caches[layer_name].element_size()\n"
    "        if is_mla:\n"
    "            blknum, blksize, hs = self.kv_cache_shape\n"
    "            hn = 1\n"
    "            block_stride = stride[0]\n"
    "        else:\n"
    "            _, blknum, blksize, hn, hs = self.kv_cache_shape\n"
)
new1 = (
    '        assert self.kv_cache_shape is not None, "KV caches shape not initialized"\n'
    "        # Per-layer shape: GLM-5.1 (MLA + DSA) registers heterogeneous KV\n"
    "        # caches (attn latent (...,1,576), indexer (...,1,132)); the global\n"
    "        # self.kv_cache_shape == first layer is wrong for the others, so read\n"
    "        # dims/stride from THIS layer to match its layout.\n"
    "        _layer_shape = tuple(self.kv_caches[layer_name].shape)\n"
    "        is_mla = len(_layer_shape) == 3\n"
    "        stride = self.kv_caches[layer_name].stride()\n"
    "        sz = self.kv_caches[layer_name].element_size()\n"
    "        if is_mla:\n"
    "            blknum, blksize, hs = _layer_shape\n"
    "            hn = 1\n"
    "            block_stride = stride[0]\n"
    "        else:\n"
    "            _, blknum, blksize, hn, hs = _layer_shape\n"
)

# --- Edit 2: per-layer offsets in _read_blocks --------------------------------
old2 = (
    "        first_layer = list(self.layer_name_to_local_kv_cache_metadata.keys())[0]\n"
    "        offs = self._compute_block_transfer_offsets(\n"
    "            first_layer, local_block_ids, remote_block_ids, remote_moriio_meta\n"
    "        )\n"
    "\n"
    "        for layer_name in self.layer_name_to_local_kv_cache_metadata:\n"
    "            sess_idx = list(self.layer_name_to_local_kv_cache_metadata.keys()).index(\n"
    "                layer_name\n"
    "            )\n"
    "            # TODO : apply multi-session batch-read when moriio support it\n"
    "            transfer_status = self.moriio_wrapper.read_remote_data(\n"
    "                offs[2], offs[0], offs[1], sessions[sess_idx]\n"
    "            )\n"
)
new2 = (
    "        # Per-layer offsets: GLM-5.1's attn vs indexer KV caches differ in\n"
    "        # head_dim, so offsets/sizes computed for the first layer don't apply\n"
    "        # to the others. Compute per layer (homogeneous models: identical).\n"
    "        for layer_name in self.layer_name_to_local_kv_cache_metadata:\n"
    "            sess_idx = list(self.layer_name_to_local_kv_cache_metadata.keys()).index(\n"
    "                layer_name\n"
    "            )\n"
    "            offs = self._compute_block_transfer_offsets(\n"
    "                layer_name, local_block_ids, remote_block_ids, remote_moriio_meta\n"
    "            )\n"
    "            # TODO : apply multi-session batch-read when moriio support it\n"
    "            transfer_status = self.moriio_wrapper.read_remote_data(\n"
    "                offs[2], offs[0], offs[1], sessions[sess_idx]\n"
    "            )\n"
)

for tag, old in (("edit1", old1), ("edit2", old2)):
    n = src.count(old)
    if n != 1:
        print(f"moriio-hetero: {tag} expected 1 match, found {n} — moriio layout changed? aborting")
        sys.exit(1)

src = src.replace(old1, new1).replace(old2, new2)

open(f, "w").write(src)
try:
    py_compile.compile(f, doraise=True)
except py_compile.PyCompileError as e:
    print(f"moriio-hetero: py_compile FAILED after patch: {e}")
    sys.exit(1)
print("moriio-hetero: patched", f)
