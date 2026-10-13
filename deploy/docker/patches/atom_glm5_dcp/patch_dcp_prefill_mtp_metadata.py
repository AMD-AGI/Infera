#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Publish the DCP per-token block table when MTP drafts follow a prefill.

With DCP > 1 the eagle proposer only refreshed ``dcp_token_block_tables`` when
it was already set, i.e. after a verify step. A prefill has none yet, so the
draft's first decode step reaches the DCP attention ops with ``None`` and
crashes. Publish it whenever DCP is on. Strict: see anchor_patch.py.
"""

import sys

from anchor_patch import apply

IMPORT = "from atom.distributed.dcp_utils import get_dcp_world_size\n"

PATCH = {
    "spec_decode/eagle_proposer.py": (
        "# A prefill has no per-token decode table yet.",
        [
            # The new branch calls get_dcp_world_size; require its import.
            (IMPORT, IMPORT),
            (
                "        if attn_metadata.dcp_token_block_tables is not None:\n"
                "            # One query per sequence, so the per-token table is block_tables\n"
                "            # itself; the verify step's has the right row count, wrong rows.\n",
                "        if get_dcp_world_size() > 1:\n"
                "            # A prefill has no per-token decode table yet. Publish it when\n"
                "            # entering the draft's decode steps too, not only after verify.\n"
                "            # One query per sequence means block_tables is already row-aligned.\n",
            ),
        ],
    ),
}

if __name__ == "__main__":
    sys.exit(apply("atom-patch dcp-prefill-mtp-metadata", PATCH))
