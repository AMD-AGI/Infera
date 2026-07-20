###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Make aiter flydsl 2-stage MoE GEMM accept memref-typed pointer args.

WHAT: revert gemm1/gemm2 buffer-resource builds to the memref-friendly buffer_ops
  calls (create_buffer_resource / extract_base_index); leave gemm3 untouched.
WHY: vLLM passes tensors -> flydsl memrefs, but aiter 0.1.16's fx.ptrtoint rejects
  memref -> MLIRError crash. Needed by Kimi-K2.6 int4 W4A16 MoE; no-op for DSv4 MXFP4.
DEPS: baked by Dockerfile.vllm (vllm-dsv4-patches loop); idempotent,
  no-op if anchors absent. VERSION: verified dev748 (aiter 0.1.16.post2, flydsl 0.2.0).
"""

import os
import sys

try:
    import aiter
except Exception:
    print("aiter-flydsl-memref: aiter not importable - skipping")
    sys.exit(0)

f = os.path.join(
    os.path.dirname(aiter.__file__),
    "ops/flydsl/kernels/moe_gemm_2stage.py",
)
if not os.path.exists(f):
    print("aiter-flydsl-memref: target not found - skipping")
    sys.exit(0)

src = open(f).read()
MARKER = "# AITER-FLYDSL-MEMREF-BUFRES"
if MARKER in src:
    print("aiter-flydsl-memref: already patched")
    sys.exit(0)

OLD_HELPER = (
    "            def _ptr_buffer_resource(ptr, num_records_bytes):\n"
    "                addr = fx.ptrtoint(ptr)\n"
    "                addr_i64 = arith.index_cast(T.i64, addr)\n"
    "                return buffer_ops.create_buffer_resource_from_addr(\n"
    "                    addr_i64, num_records_bytes=num_records_bytes\n"
    "                )\n"
)
NEW_HELPER = (
    "            def _ptr_buffer_resource(ptr, num_records_bytes):\n"
    "                # AITER-FLYDSL-MEMREF-BUFRES: vLLM passes tensors -> flydsl\n"
    "                # memrefs; ptrtoint rejects memref. Build the buffer resource\n"
    "                # straight from the memref (aiter 0.1.13 behaviour).\n"
    "                return buffer_ops.create_buffer_resource(\n"
    "                    ptr, max_size=False, num_records_bytes=num_records_bytes\n"
    "                )\n"
)

OLD_OUTBASE = "out_base_idx = arith.index_cast(T.index, fx.ptrtoint(arg_out))"
NEW_OUTBASE = "out_base_idx = buffer_ops.extract_base_index(arg_out)  # AITER-FLYDSL-MEMREF-BUFRES"

n_helper = src.count(OLD_HELPER)
n_outbase = src.count(OLD_OUTBASE)
if n_helper != 2 or n_outbase != 2:
    print(
        f"aiter-flydsl-memref: anchors helper={n_helper} outbase={n_outbase} "
        "(expected 2/2) - aiter layout drifted or fixed upstream, skipping"
    )
    sys.exit(0)

src = src.replace(OLD_HELPER, NEW_HELPER, 2)
src = src.replace(OLD_OUTBASE, NEW_OUTBASE, 2)
open(f, "w").write(src)
print("aiter-flydsl-memref: patched (2 helpers + 2 out-base)")
