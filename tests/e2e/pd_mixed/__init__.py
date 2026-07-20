###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""PD-mixed e2e suites: prefill and decode co-located in one worker (no PD split).

One per-engine subpackage (``sglang`` / ``vllm`` / ``atom``); each spawns a
single mixed worker and verifies liveness + correctness. The engine-agnostic
harness is shared from :mod:`tests.e2e.harness` (one level up). The sibling
:mod:`tests.e2e.pd_disag` package holds the prefill-decode-disaggregated
(across-node) counterpart.
"""
