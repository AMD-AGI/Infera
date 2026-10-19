#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Merge DCP attention partials over aiter's custom all-to-all.

Each DCP decode layer exchanges its per-head partial outputs and LSEs with one
all_to_all_single on RCCL (81 per GLM-5.2 step, ~3.3 ms). With
patch_aiter_custom_alltoall.py the DCP group's custom all-reduce communicator
offers custom_all_to_all; use it when it applies and keep RCCL otherwise. The
choice depends only on the shape, dtype and communicator, so every rank of the
group makes the same one.

Written against ATOM 661ef766 in rocm/atom-dev:nightly_202610131505.
Strict: see anchor_patch.py.
"""

import sys

from anchor_patch import apply

PATCH = {
    "model_ops/dcp_ops.py": (
        "# infera: custom all-to-all for the DCP merge",
        [
            (
                "class CPTritonContext:\n",
                "def _dcp_custom_all_to_all(cp_group, send):\n"
                '    """aiter\'s custom (IPC) all-to-all over the DCP group, or None for RCCL."""\n'
                '    device_comm = getattr(cp_group, "device_communicator", None)\n'
                '    ca = getattr(device_comm, "ca_comm", None)\n'
                '    if ca is None or ca.disabled or not hasattr(ca, "custom_all_to_all"):\n'
                "        return None\n"
                "    return ca.custom_all_to_all(send)\n"
                "\n"
                "\n"
                "class CPTritonContext:\n",
            ),
            (
                "    recv = torch.empty_like(send)\n"
                "    torch.distributed.all_to_all_single(recv, send, group=cp_group.device_group)\n",
                "    # infera: custom all-to-all for the DCP merge, RCCL when it does not apply.\n"
                "    recv = _dcp_custom_all_to_all(cp_group, send)\n"
                "    if recv is None:\n"
                "        recv = torch.empty_like(send)\n"
                "        torch.distributed.all_to_all_single(recv, send, group=cp_group.device_group)\n",
            ),
        ],
    ),
}

if __name__ == "__main__":
    sys.exit(apply("atom-patch dcp-custom-a2a", PATCH, package="atom"))
