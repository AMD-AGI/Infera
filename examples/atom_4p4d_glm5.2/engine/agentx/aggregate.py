#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""InferenceX AgentX aggregation with the right GPU count for single-node PD.

At InferenceX 918524ff, ``_gpu_shape`` uses the prefill/decode worker fields
only when IS_MULTINODE=true, so a single-node 4P+4D run would be counted as 4
GPUs instead of 8. This wraps the calculator to treat DISAGG=true like a
multinode PD run, and leaves the real IS_MULTINODE=false in the metadata.
Request data and metric calculations are unchanged; the InferenceX checkout
is not modified.

Run from the InferenceX checkout with INFMAX_CONTAINER_WORKSPACE set, like
InferenceX's own aggregation entry point. ``--check-topology`` prints the
resolved GPU shape without aggregating.
"""

import json
import os
import sys

sys.path.insert(0, os.environ["INFMAX_CONTAINER_WORKSPACE"])
from utils.agentic.aggregation import process_agentic_result as processor  # noqa: E402

_original_gpu_shape = processor._gpu_shape


def gpu_shape():
    if processor.env_bool("IS_MULTINODE") or not processor.env_bool("DISAGG"):
        return _original_gpu_shape()
    original = os.environ.get("IS_MULTINODE")
    try:
        # Local to the GPU-shape calculation; restored before metadata is written.
        os.environ["IS_MULTINODE"] = "true"
        return _original_gpu_shape()
    finally:
        if original is None:
            os.environ.pop("IS_MULTINODE", None)
        else:
            os.environ["IS_MULTINODE"] = original


processor._gpu_shape = gpu_shape

if __name__ == "__main__":
    if sys.argv[1:] == ["--check-topology"]:
        print(
            json.dumps({"shape": gpu_shape(), "is_multinode": processor.env_bool("IS_MULTINODE")})
        )
    else:
        sys.exit(processor.main())
