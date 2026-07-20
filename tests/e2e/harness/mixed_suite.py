###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Shared "mixed" (prefill-decode-mix) e2e building block — engine-agnostic.

:func:`run_mixed_case` is the shared body of the parametrized ``test_mixed``
(guards + server + run_mixed). Each engine's ``test_mixed.py`` wraps it with its
own parametrize list (matrices diverge per engine — see
:mod:`tests.e2e.harness.matrix`).
"""

from __future__ import annotations

from . import resources, scenarios
from .params import EngineParams

__all__ = ["run_mixed_case"]


async def run_mixed_case(params: EngineParams, infera_server, worker) -> None:
    """Shared body: skip unsupported combos, start the server, run the mixed
    scenario (chat + streaming + counting/capital correctness)."""
    resources.require_supported(params)
    resources.require_gpus(params)

    server = await infera_server()
    await scenarios.run_mixed(server, worker, params)
