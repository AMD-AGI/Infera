###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Relax the MoRIIO connector's GLOBAL block_size equality checks so DSv4's
hybrid (per-layer) KV block sizes are accepted.

WHAT: drop the `assert first_geometry.block_size == self.block_size` and demote
  the per-layer mismatch `raise` to a debug log.
WHY: DSv4 registers per-layer KV caches with different block sizes (c128a sparse=2,
  SWA/state=256); offsets already use the per-layer self.block_lens dict, so the
  global equality check is spurious and kills the prefill worker at KV registration.
DEPS: baked by Dockerfile.vllm; idempotent, no-op if patterns absent.
  VERSION: verified vllm 0.23.x ROCm, DeepSeek-V4-Pro MoRIIOConnector 1P1D.
"""

import os
import sys

try:
    import vllm
except Exception:
    print("moriio-dsv4-blocksize: vLLM not importable — skipping")
    sys.exit(0)

f = os.path.join(
    os.path.dirname(vllm.__file__),
    "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py",
)
if not os.path.exists(f):
    print(f"moriio-dsv4-blocksize: {f} not found — skipping")
    sys.exit(0)

src = open(f).read()
MARKER = "# MORIIO-DSV4-HYBRID-BLOCKSIZE-PATCH"
if MARKER in src:
    print("moriio-dsv4-blocksize: already patched")
    sys.exit(0)

edits = 0

# 1) Drop the global assert (first layer's block_size == global).
ASSERT_OLD = "        assert first_geometry.block_size == self.block_size\n"
ASSERT_NEW = (
    "        # MORIIO-DSV4-HYBRID-BLOCKSIZE-PATCH: DSv4 registers per-layer KV\n"
    "        # caches with different block sizes (c128a=2, SWA=256); offsets use\n"
    "        # the per-layer self.block_lens dict, so no global equality needed.\n"
    "        _ = first_geometry.block_size  # (was: assert == self.block_size)\n"
)
if src.count(ASSERT_OLD) == 1:
    src = src.replace(ASSERT_OLD, ASSERT_NEW, 1)
    edits += 1

# 2) Demote the per-layer raise to a debug log.
RAISE_OLD = (
    "            if geometry.block_size != self.block_size:\n"
    "                raise ValueError(\n"
    '                    "MoRIIO KV cache block size mismatch for layer "\n'
    '                    f"{layer_name}: {geometry.block_size} != {self.block_size}"\n'
    "                )\n"
)
RAISE_NEW = (
    "            if geometry.block_size != self.block_size:\n"
    "                # MORIIO-DSV4-HYBRID-BLOCKSIZE-PATCH: hybrid per-layer block\n"
    "                # sizes are expected for DSv4; handled per-layer via block_lens.\n"
    "                logger.debug(\n"
    '                    "MoRIIO per-layer block_size differs for %s: %s != global %s",\n'
    "                    layer_name, geometry.block_size, self.block_size,\n"
    "                )\n"
)
if src.count(RAISE_OLD) == 1:
    src = src.replace(RAISE_OLD, RAISE_NEW, 1)
    edits += 1

if edits == 0:
    print("moriio-dsv4-blocksize: no target patterns found (already handled?) — skipping")
    sys.exit(0)

open(f, "w").write(src)
print(f"moriio-dsv4-blocksize: patched {f} ({edits}/2 edits)")
