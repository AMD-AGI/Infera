###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""MoE experts injection seam (issue #40).

vLLM runs every MoE block through a modular kernel (``FusedMoEKernel`` in vLLM
0.23; its ``apply`` / ``apply_monolithic`` route → experts-GEMM → combine). We
patch those methods from ``register_ops`` (a general plugin, so ``vllm`` is
initialized) with a pass-through wrapper — the single point where an
Infera/HyperLoom experts kernel replaces the default while keeping vLLM's
routing / quant / EP-DP dispatch. This is the less-invasive alternative to
vLLM-ATOM's whole-model ``register_model`` wrapper.

Today it is a delegating pass-through (bitwise identical). ``INFERA_MOE_EXPERTS``
is reserved for selecting a custom experts implementation. The MoE modular-kernel
API moves between vLLM versions, so the patch is defensive: if the class/methods
aren't found it logs and no-ops rather than raising.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_INSTALLED = False

# MoE compute chokepoints to wrap, newest vLLM first. Extend as the API moves.
_KERNEL_CANDIDATES = (("vllm.model_executor.layers.fused_moe.modular_kernel", "FusedMoEKernel"),)
_METHODS = ("apply", "apply_monolithic")


def install_moe_ops() -> None:
    """Wrap the MoE modular-kernel compute methods with the Infera seam (idempotent)."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    import importlib

    kernel = None
    for mod_name, cls_name in _KERNEL_CANDIDATES:
        try:
            kernel = getattr(importlib.import_module(mod_name), cls_name)
            break
        except (ImportError, AttributeError):
            continue
    if kernel is None:
        logger.warning(
            "infera-vllm-ops: MoE seam not wired — no known modular-kernel class on this vLLM"
        )
        return

    experts = os.environ.get("INFERA_MOE_EXPERTS") or None
    wrapped = []
    for name in _METHODS:
        original = getattr(kernel, name, None)
        if original is None or getattr(original, "_infera_wrapped", False):
            continue
        setattr(kernel, name, _make_seam(original))
        wrapped.append(name)

    logger.info(
        "infera-vllm-ops: MoE experts seam on %s.{%s} (experts=%s)",
        kernel.__name__,
        ",".join(wrapped) or "<none>",
        experts or "pass-through",
    )


def _make_seam(original):
    """Delegating pass-through — the point a custom experts kernel is dropped in."""

    def _seam(self, *args, **kwargs):
        # TODO(#40): dispatch to the Infera experts-GEMM kernel when
        # INFERA_MOE_EXPERTS is set. Until then, delegate unchanged.
        return original(self, *args, **kwargs)

    _seam._infera_wrapped = True
    return _seam
