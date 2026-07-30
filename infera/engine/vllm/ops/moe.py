###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""MoE experts injection seam (issue #40).

vLLM runs MoE through a ``FusedMoE`` layer whose experts kernel is a
``FusedMoEModularKernel`` (``FusedMoEPrepareAndFinalize`` +
``FusedMoEPermuteExpertsUnpermute``) or a ``FusedMoEMethodBase.apply()``.
:func:`install_moe_ops` is where an Infera/HyperLoom experts kernel replaces the
default while keeping vLLM's routing / quant / EP-DP dispatch — the less-invasive
alternative to vLLM-ATOM's whole-model ``register_model`` wrapper.

Today it is a logged pass-through so the seam is wired and observable without
changing numerics; the real kernel lands behind this function.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_INSTALLED = False


def install_moe_ops() -> None:
    """Install the Infera MoE experts kernel (pass-through stub for now)."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    # TODO(#40): register a FusedMoEPermuteExpertsUnpermute / FusedMoEMethodBase
    # implementation dispatching to the Infera experts-GEMM kernel. Until then
    # this is a no-op so behaviour is unchanged while the seam is in place.
    logger.info("infera-vllm-ops: MoE experts seam active (pass-through)")
