###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Entry points for the Infera vLLM op-injection plugin (issue #40).

vLLM discovers these via ``entry_points`` and calls them at startup:

  ``vllm.platform_plugins`` → :func:`register_platform` returns the qualname of
      :class:`InferaPlatform` (or ``None`` to skip), whose ``get_attn_backend_cls``
      is the **attention** injection seam.
  ``vllm.general_plugins`` → :func:`register_ops` installs the **MoE** experts
      injection seam.

Both are no-ops when ``INFERA_VLLM_OPS_DISABLE=1``. Kept free of top-level vLLM /
torch imports so this module is safe to import in any environment; the ROCm-only
platform is loaded lazily by vLLM via the qualname string.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_PLATFORM_QUALNAME = "infera.engine.vllm.ops.platform:InferaPlatform"


def _disabled() -> bool:
    return os.environ.get("INFERA_VLLM_OPS_DISABLE", "0") == "1"


def _is_rocm() -> bool:
    """Cheap ROCm probe (no torch import): the platform subclasses RocmPlatform,
    so only activate where ROCm is present — leave CUDA/CPU vLLM untouched."""
    if os.environ.get("ROCM_PATH") or os.environ.get("HIP_VISIBLE_DEVICES"):
        return True
    import glob

    return bool(glob.glob("/opt/rocm*"))


def register_platform() -> str | None:
    """vLLM ``platform_plugins`` hook: return InferaPlatform's qualname, else None.

    Opt-in via ``INFERA_VLLM_OPS_PLATFORM=1``. Default off (returns None) because
    activating a ``RocmPlatform`` subclass here re-enters ``vllm.platforms`` while
    it is still initializing (vLLM resolves ``current_platform`` eagerly during
    ``import vllm``), so importing ``platform.py`` at that moment circular-imports.
    The attention seam will move to an import-safe path (a ``register_ops``
    monkey-patch of ``get_attn_backend_cls``, ATOM's MLA approach) — tracked in #40.
    """
    if _disabled() or not _is_rocm():
        return None
    if os.environ.get("INFERA_VLLM_OPS_PLATFORM", "0") != "1":
        return None
    logger.info("infera-vllm-ops: activating platform %s", _PLATFORM_QUALNAME)
    return _PLATFORM_QUALNAME


def register_ops() -> None:
    """vLLM ``general_plugins`` hook: install the MoE experts injection seam."""
    if _disabled() or not _is_rocm():
        return
    from infera.engine.vllm.ops.moe import install_moe_ops

    install_moe_ops()
