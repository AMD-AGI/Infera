###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Infera end-to-end test suite (real server + real engine workers).

This is a package so the per-scenario, per-engine suites can share the harness
via relative imports (e.g. ``from ...harness import ...`` from a
``pd_mixed/<engine>/`` module); pytest resolves the test modules as
``e2e.pd_mixed.<engine>.test_mixed`` / ``e2e.pd_disag.<engine>.test_disagg``
only when this directory is importable. Layout:

    harness/    engine-agnostic shared components
    pd_mixed/   prefill+decode co-located (sglang/vllm/atom)
    pd_disag/   prefill/decode split across nodes (scaffold; sglang/vllm/atom)
"""
