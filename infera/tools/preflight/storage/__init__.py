###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Storage section: local NVMe inventory + local-NVMe KV read/write throughput."""

from ..finding import Finding
from . import info, perf


def collect() -> list[Finding]:
    return info.collect() + perf.collect()


__all__ = ["collect"]
