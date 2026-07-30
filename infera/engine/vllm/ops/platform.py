###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""InferaPlatform — the **attention** injection seam (issue #40).

Subclasses vLLM's ``RocmPlatform`` and overrides ``get_attn_backend_cls``. Today
it is a pass-through (delegates to ``RocmPlatform``, so numerics are bitwise
identical), but it is the single supported point where an Infera/HyperLoom
``AttentionBackend`` is substituted — exactly the seam vLLM-ATOM uses
(``ATOMPlatform`` → ``AiterBackend`` / ``AiterMLABackend``).

Imported by vLLM only via the ``platform_plugins`` qualname string, so the vLLM
import below never runs outside a vLLM process.
"""

from __future__ import annotations

import logging

from vllm.platforms.rocm import RocmPlatform

logger = logging.getLogger(__name__)

# Set to a ``"module.path:AttentionBackend"`` string to inject a custom attention
# backend; ``None`` = pass-through (vLLM's default ROCm backend selection). The
# custom class must implement vLLM's ``AttentionBackend`` / ``AttentionImpl``.
INFERA_ATTN_BACKEND: str | None = None


class InferaPlatform(RocmPlatform):
    """ROCm platform with an attention-backend injection seam."""

    @classmethod
    def get_attn_backend_cls(cls, *args, **kwargs):
        if INFERA_ATTN_BACKEND is not None:
            logger.info("infera-vllm-ops: attention backend → %s", INFERA_ATTN_BACKEND)
            return INFERA_ATTN_BACKEND
        # Pass-through: vLLM's default ROCm attention backend, unchanged.
        return super().get_attn_backend_cls(*args, **kwargs)
