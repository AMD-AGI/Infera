#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Exchange the sparse-indexer DCP top-k candidates over aiter's custom all-gather.

In DCP decode every indexer layer gathers each rank's local top-k (score, id)
candidates across the DCP group (21 indexer layers in the verify graph plus one
per draft for GLM-5.2: 24 gathers per step of 1.3 MB per rank at graph size 40).
That call used ``GroupCoordinator.all_gather``, whose default is pynccl, at
~100 GB/s bus bandwidth (~43 us each, ~1 ms per decode step), while ATOM's own
``dcp_all_gather`` -- already used for the LSE and query-head gathers -- picks
aiter's IPC all-gather when the shape qualifies and falls back to pynccl
otherwise. The payload is a bit-exact copy either way (moved as 4-byte words).

Written against ATOM 661ef766 in rocm/atom-dev:nightly_202610131505.
Strict: see anchor_patch.py.
"""

import sys

from anchor_patch import apply

PATCH = {
    "model_ops/dcp_ops.py": (
        "# infera: custom all-gather for the top-k exchange",
        [
            (
                "    gathered_sc = (\n"
                "        get_dcp_group()\n"
                "        .all_gather(send.view(torch.int32), dim=0)\n",
                "    # infera: custom all-gather for the top-k exchange (pynccl otherwise).\n"
                "    gathered_sc = (\n"
                "        dcp_all_gather(get_dcp_group(), send.view(torch.int32), 0)\n",
            ),
        ],
    ),
}

if __name__ == "__main__":
    sys.exit(apply("atom-patch dcp-topk-custom-ag", PATCH, package="atom"))
