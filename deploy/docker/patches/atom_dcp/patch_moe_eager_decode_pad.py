#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Pad an eager MoE decode step for the DP-attention gather like a prefill.

An eager decode step (--enforce-eager, e.g. an MTP PD producer with DP
attention) reaches the MoE gather unpadded and trips its row-count assert. Pad
it to the running tokens and zero the padding: the same helper pads router
logits, where garbage can select out-of-range experts. Only DP attention plus
eager decode reaches this path. Strict: see anchor_patch.py.
"""

import sys

from anchor_patch import apply

LOCAL_TOKENS = "    local_tokens = ctx.scheduled_tokens if ctx.is_prefill else running_tokens\n"

EAGER_DECODE = """\
    # An eager decode step (--enforce-eager, e.g. an MTP PD producer with DP
    # attention) reaches here unpadded; pad it for the gather like a prefill.
    eager_decode = not ctx.is_prefill and x.shape[0] == ctx.scheduled_tokens
    if eager_decode:
        local_tokens = ctx.scheduled_tokens
"""

DEBUG_ZERO = "    # padded_x[local_tokens:, :].zero_() # keep for debug\n"

ZERO_PADDING = """\
    if eager_decode:
        # The same helper pads router logits; garbage there can select experts
        # out of range.
        padded_x[local_tokens:, :].zero_()
"""

PATCH = {
    "model_ops/moe.py": (
        "eager_decode = not ctx.is_prefill and x.shape[0] == ctx.scheduled_tokens",
        [
            (LOCAL_TOKENS, LOCAL_TOKENS + EAGER_DECODE),
            (DEBUG_ZERO, ZERO_PADDING),
        ],
    ),
}

if __name__ == "__main__":
    sys.exit(apply("atom-patch moe-eager-decode-pad", PATCH))
