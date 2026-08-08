###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Attention injection seam (issue #40) — import-safe monkey-patch.

Rather than activating a ``RocmPlatform`` *subclass* through ``platform_plugins``
(which circular-imports: vLLM resolves ``current_platform`` eagerly during
``import vllm``, and a platform module that imports ``vllm.platforms.rocm`` at top
level re-enters ``vllm.platforms`` before it finishes), we patch
``get_attn_backend_cls`` on the already-resolved platform from ``register_ops`` —
a ``general_plugins`` hook that runs *after* ``vllm.platforms`` is initialized and
*before* the model (and thus attention-backend selection) is built. Same style as
vLLM-ATOM's MLA ``forward_impl`` patch.

Set ``INFERA_ATTN_BACKEND="module.path:AttentionBackend"`` to substitute a custom
attention backend; unset ⇒ pass-through (delegate to vLLM's default selection).
The custom class must implement vLLM's ``AttentionBackend`` / ``AttentionImpl``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_PATCHED = False


def install_attention_ops() -> None:
    """Patch the current platform's ``get_attn_backend_cls`` (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return

    override = os.environ.get("INFERA_ATTN_BACKEND") or None
    # Safe here: general plugins load after vllm.platforms is initialized.
    from vllm.platforms import current_platform

    platform_cls = type(current_platform)
    original = platform_cls.get_attn_backend_cls  # bound classmethod (pre-patch)

    def _patched(cls, *args, **kwargs):
        if override is not None:
            logger.info("infera-vllm-ops: attention backend → %s", override)
            return override
        # Pass-through: vLLM's default selection, unchanged.
        return original(*args, **kwargs)

    platform_cls.get_attn_backend_cls = classmethod(_patched)
    _PATCHED = True
    logger.info(
        "infera-vllm-ops: attention seam installed on %s (backend=%s)",
        platform_cls.__name__,
        override or "pass-through",
    )
