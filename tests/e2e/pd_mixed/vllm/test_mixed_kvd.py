###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""vLLM e2e — mixed worker WITH kvd L3 offload (aggregated + offload).

The plain mixed suite (``test_mixed.py``) runs a bare worker and never touches
the kvd tiered cache. This one spawns the same aggregated worker but wired to a
live ``infera.kvd`` daemon (the ``kvd_daemon`` fixture), drives a long request
through the router, and asserts the KV was offloaded to L3. kvd is a vLLM-only
feature, so this test lives in the vLLM suite only.
"""

import pytest

from ...harness import resources
from ...harness.kvd_offload import run_mixed_kvd_offload
from ...harness.matrix import QWEN3_8B, resolve_model
from ...harness.params import EngineParams


@pytest.mark.slow
async def test_mixed_kvd(infera_server, worker, kvd_daemon):
    """Aggregated worker + kvd: one long request must offload KV to the L3 tier
    (daemon sets_total>0, misses_total==0)."""
    params = EngineParams(model=resolve_model(QWEN3_8B), tensor_parallel_size=1)
    resources.require_gpus(params)

    server = await infera_server()
    await run_mixed_kvd_offload(server, worker, params, kvd_daemon)
