###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""vLLM e2e — PD-disaggregated prefill/decode split across two nodes.

The test body is engine-agnostic and shared (see
:mod:`tests.e2e.harness.disagg_suite`); only the parametrize list is
engine-specific. Binds to the vLLM ``disagg_stack`` fixture (this dir's conftest).
Self-skips when the SLURM/2-node environment isn't available.
"""

import pytest

from ...harness.disagg_suite import run_disagg_case
from .matrix import vllm_disagg_params


@pytest.mark.slow
@pytest.mark.parametrize("params", vllm_disagg_params())
async def test_disagg(params, disagg_stack):
    await run_disagg_case(params, disagg_stack)
