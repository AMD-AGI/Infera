###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""vLLM e2e — mixed worker (prefill-decode-mix, no PD).

Shares the test body with sglang (see :mod:`tests.e2e.harness.mixed_suite`);
only the parametrize list is engine-specific. All bind to the vLLM ``worker``
fixture (this dir's conftest).
"""

import pytest

from ...harness.mixed_suite import run_mixed_case
from .matrix import vllm_mixed_params


@pytest.mark.slow
@pytest.mark.parametrize("params", vllm_mixed_params())
async def test_mixed(params, infera_server, worker):
    await run_mixed_case(params, infera_server, worker)
