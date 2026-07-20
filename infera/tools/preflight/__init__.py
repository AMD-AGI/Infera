###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Infera preflight: cluster diagnostic tool.

Per-node probes (host / gpu / network / firmware / perf / topology) plus
coordinated multi-node checks (RoCE bandwidth, Mooncake / Mori KV transfer),
aggregated into one cluster HTML report.
"""

from .finding import Finding, status_from

__all__ = ["Finding", "status_from"]
