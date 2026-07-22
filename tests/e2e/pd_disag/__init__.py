###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""PD-disaggregated e2e suites: prefill and decode split across separate nodes.

Scaffold only — the flow is not implemented yet (see
:mod:`tests.e2e.harness.disagg_suite` for what's still needed). Mirrors the
:mod:`tests.e2e.pd_mixed` layout: one per-engine subpackage
(``sglang`` / ``vllm`` / ``atom``), each reusing the shared
:mod:`tests.e2e.harness`.
"""
