###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Entry point for the Infera vLLM op-injection plugin (issue #40).

vLLM discovers this via ``entry_points`` and calls it once at startup:

  ``vllm.general_plugins`` → :func:`register_ops` installs both injection seams
      by patching the already-resolved platform / MoE layer — a general plugin
      runs *after* ``vllm.platforms`` is initialized and *before* the model is
      built, so it is import-safe (unlike a ``platform_plugins`` subclass, which
      re-enters ``vllm.platforms`` during vLLM's eager ``current_platform``
      resolution and circular-imports).

No-op unless ROCm is present and ``INFERA_VLLM_OPS_DISABLE != 1``. Kept free of
top-level vLLM / torch imports so this module is safe to import anywhere; the
seams import vLLM internals lazily, inside :func:`register_ops`.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _disabled() -> bool:
    return os.environ.get("INFERA_VLLM_OPS_DISABLE", "0") == "1"


def _is_rocm() -> bool:
    """Cheap ROCm probe (no torch import) — the seams target ROCm, so leave
    CUDA/CPU vLLM untouched."""
    if os.environ.get("ROCM_PATH") or os.environ.get("HIP_VISIBLE_DEVICES"):
        return True
    import glob

    return bool(glob.glob("/opt/rocm*"))


def register_ops() -> None:
    """vLLM ``general_plugins`` hook: install the Attention + MoE injection seams."""
    if _disabled():
        logger.info("infera-vllm-ops: disabled (INFERA_VLLM_OPS_DISABLE=1)")
        return
    if not _is_rocm():
        logger.info("infera-vllm-ops: no ROCm detected — seams not installed")
        return

    from infera.engine.vllm.ops.attention import install_attention_ops
    from infera.engine.vllm.ops.moe import install_moe_ops

    install_attention_ops()
    install_moe_ops()
