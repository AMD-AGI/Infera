###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""ROCm opt-out for SGLang's CUDA-only DSA ``topk_v2`` JIT kernel.

SGLang >= v0.5.15 defaults ``SGLANG_OPT_USE_TOPK_V2`` ON, but that kernel's
header includes ``<cooperative_groups.h>`` (CUDA-only; ROCm ships only
``hip/hip_cooperative_groups.h``), so hipcc fails at JIT time and the engine
dies in decode CUDA-graph capture. Upstream turns the flag off on HIP only in
the ``DeepseekV4ForCausalLM`` branch of ``server_args.py``, yet the kernel is
shared by every DSA arch — GLM-5.x, DeepseekV32, LongcatFlash, MistralLarge3 —
which therefore still hit it. Default it OFF on ROCm instead; the backend then
takes the legacy transform path ROCm used before v0.5.15.

Set-if-unset, so it self-retires: once upstream guards the kernel, its own value
(or an explicit ``SGLANG_OPT_USE_TOPK_V2=1``) wins and this becomes a no-op.

Verified 2026-07-30, MI355X/gfx950, sglang v0.5.15.post1, GLM-5.1-FP8 tp4:
capture dies at 0/4 without it, completes 4/4 with it.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# env var -> default value (applied only if unset).
_ROCM_DSA_DEFAULTS: dict[str, str] = {
    "SGLANG_OPT_USE_TOPK_V2": "0",  # sglang reads this as EnvBool (default True)
}


def _is_rocm() -> bool:
    """True iff running on ROCm/HIP. Probe ``/dev/kfd`` to avoid importing torch."""
    return os.path.exists("/dev/kfd")


def apply_rocm_dsa_env_defaults() -> dict[str, str]:
    """Set the ROCm DSA env defaults (set-if-unset); no-op elsewhere.

    Call ONCE at engine startup, BEFORE the sglang subprocess is spawned — the
    flag is read in that subprocess, at CUDA-graph capture. Returns the vars
    actually applied (empty if none / not ROCm).
    """
    if not _is_rocm():
        return {}
    applied: dict[str, str] = {}
    for key, value in _ROCM_DSA_DEFAULTS.items():
        if os.environ.get(key) in (None, ""):
            os.environ[key] = value
            applied[key] = value
    if applied:
        logger.info(
            "ROCm DSA env defaults applied (set-if-unset; override via env): %s",
            applied,
        )
    return applied
