#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Let SGLANG_AITER_CAPTURE_COPY_IN cover the aiter custom all-reduce too (custom_all_reduce.py).

WHAT: under CUDA-graph capture, sglang v0.5.22 builds aiter's CustomAllreduce with
`enable_register_for_capturing=True` on ROCm, so the captured all-reduce registers the
graph-pool input tensor with its peers. `parallel_state.py` already documents that this
registered path "makes peers read stale data over IPC on replay (garbage rows / NaN under
DP attention + TP MoE on MI355X)" and offers SGLANG_AITER_CAPTURE_COPY_IN=1, which stages
the custom all-gather / reduce-scatter inputs through the pre-registered IPC pool instead --
but not the all-reduce. With GLM-5.2 aggregated TP4 + DP4 attention under sustained
concurrency (EXTEND steps interleaved with decode-graph replays), the captured all-reduce
alone garbles about a quarter of the answers; staging it the same way removes that while
keeping the custom all-reduce (200 GSM8K prompts at 64 in flight: 50 garbled -> 0, with
and without MTP). A PD decode node runs no EXTEND steps and was clean either way.

The patch makes the existing switch apply to the all-reduce as well; unset, nothing changes.

Self-locating and idempotent; exits non-zero when the anchor drifted.
Marker: `_infera_ar_capture_copy_in`.
"""

import importlib.util
import py_compile
import sys
from pathlib import Path

TAG = "[aiter-ar-capture-copy-in]"
MARKER = "_infera_ar_capture_copy_in"
OLD = (
    "            tms_cudagraph = envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()\n"
    "            return partial(\n"
    "                AiterCustomAllreduce,\n"
    "                enable_register_for_capturing=not tms_cudagraph,\n"
    "            )\n"
)
NEW = (
    "            tms_cudagraph = envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()\n"
    f"            # {MARKER}: SGLANG_AITER_CAPTURE_COPY_IN stages the captured all-reduce\n"
    "            # through the pre-registered IPC pool too, as it does all-gather and\n"
    "            # reduce-scatter; the registered path reads stale peer data on replay.\n"
    "            copy_in = get_bool_env_var(\"SGLANG_AITER_CAPTURE_COPY_IN\", default=\"false\")\n"
    "            return partial(\n"
    "                AiterCustomAllreduce,\n"
    "                enable_register_for_capturing=not (tms_cudagraph or copy_in),\n"
    "            )\n"
)


def main() -> int:
    spec = importlib.util.find_spec("sglang")
    if not spec or not spec.origin:
        print(f"{TAG} cannot locate sglang")
        return 1
    path = (Path(spec.origin).parent / "srt" / "distributed" / "device_communicators"
            / "custom_all_reduce.py")
    source = path.read_text()
    if MARKER in source:
        print(f"{TAG} already present — skipping")
        return 0
    if source.count(OLD) != 1 or "get_bool_env_var" not in source:
        print(f"{TAG} anchor not found exactly once in {path} — re-cut the patch")
        return 1
    path.write_text(source.replace(OLD, NEW, 1))
    for cached in (path.parent / "__pycache__").glob(path.stem + ".*.pyc"):
        cached.unlink()
    py_compile.compile(str(path), doraise=True)
    print(f"{TAG} patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
