###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""SGLang e2e — mixed worker (prefill-decode-mix, no PD).

The test body is engine-agnostic and shared (see
:mod:`tests.e2e.harness.mixed_suite`); only the parametrize list is
engine-specific. All bind to the SGLang ``worker`` fixture (this dir's conftest).
"""

import pytest

from ...harness.mixed_suite import run_mixed_case
from .matrix import sglang_mixed_params


@pytest.mark.slow
@pytest.mark.parametrize("params", sglang_mixed_params())
async def test_mixed(params, infera_server, worker):
    await run_mixed_case(params, infera_server, worker)
