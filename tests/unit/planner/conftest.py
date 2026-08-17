###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Shared profiling fixtures for the planner tests.

The numbers are deliberately round so replica arithmetic can be checked by
hand: prefill sustains 10000 tok/s/GPU at every profiled ISL, and decode
sustains 1000 tok/s/GPU everywhere except the most loaded column.
"""

from __future__ import annotations

import pytest

# The planner is an optional extra (`pip install ".[planner]"`), so skip this
# directory rather than fail collection when numpy is not installed.
pytest.importorskip("numpy", reason="the SLA planner needs numpy: pip install '.[planner]'")

from infera.planner.profile_data import parse_profile_data  # noqa: E402


def flat_profile_dict() -> dict:
    """Profiling data with flat throughput, so expected replica counts are exact.

    TTFT rises linearly with ISL (100ms at 1000 tokens, 200ms at 2000). ITL is
    10ms up to 50% KV usage and 40ms at 90%, so an ITL budget between the two
    picks a well-defined operating point.
    """
    return {
        "prefill": {
            "isl": [1000, 2000],
            "ttft_ms": [100.0, 200.0],
            "thpt_per_gpu": [10000.0, 10000.0],
        },
        "decode": {
            "kv_usage": [0.1, 0.5, 0.9],
            "context_length": [1000, 2000],
            "itl_ms": [
                [10.0, 10.0, 40.0],
                [10.0, 10.0, 40.0],
            ],
            "thpt_per_gpu": [
                [1000.0, 1000.0, 1000.0],
                [1000.0, 1000.0, 1000.0],
            ],
            "max_kv_tokens": 100_000,
        },
        "prefill_engine_num_gpu": 1,
        "decode_engine_num_gpu": 1,
    }


@pytest.fixture
def flat_profile():
    return parse_profile_data(flat_profile_dict())
