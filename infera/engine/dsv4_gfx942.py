###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""MI325X (gfx942 / CDNA3) DeepSeek-V4 support policy.

gfx942 has no native FP4 MoE kernel, so the dsv4 family's runnable configurations
differ by engine. This module is the single place that encodes that contract and
applies the knobs (env + CLI) each supported combination needs — set-if-unset so
an operator/launcher always overrides. It replaces the earlier FP4-only mechanism
and enables NO third-party source patches: unsupported combinations fail fast with
an actionable message instead.

Support matrix (variant x quant x engine) on gfx942:

    variant  quant  vllm                     sglang            atom
    Pro      fp4    native (triton dequant)  UNSUPPORTED       UNSUPPORTED
    Pro      fp8    UNSUPPORTED              native (env+CLI)   native (env)
    Flash    fp4    native                   UNSUPPORTED       UNSUPPORTED
    Flash    fp8    UNSUPPORTED              native (env+CLI+MTP) native (env+MTP)

Why:
  * fp4 -> vLLM only: gfx942 lacks an FP4 MoE kernel; vLLM's ``triton_unfused``
    MoE backend upcasts fp4->bf16 in-kernel with no patch. sglang/atom only ran
    fp4 via source patches (now forbidden) -> fail fast.
  * fp8 -> sglang/atom: run natively (not validated on vLLM).
  * Flash-fp8 needs MTP: the gfx942 dsv4-Flash compressed-MQA *decode* kernel is
    broken (prefill correct, decode -> garbage); routing decode through a
    speculative (EAGLE / MTP) path avoids it. Pro-fp8 is correct without it.

Detection reads ``config.json`` from a LOCAL dir only (never downloads). Variant
is keyed off model dimensions (Pro: hidden 7168 / 61 layers; Flash: 4096 / 43),
never the directory name; quant off ``quantization_config``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Literal

from infera.engine.rocm_rdma_env import is_gfx942  # noqa: F401  (used by Task 2 enforcement)

logger = logging.getLogger(__name__)

Variant = Literal["pro", "flash"]
Quant = Literal["fp4", "fp8"]


class Dsv4UnsupportedError(RuntimeError):
    """Raised when a (engine, variant, quant) combo is unsupported on gfx942."""


@dataclass(frozen=True)
class Dsv4Model:
    """A detected local DeepSeek-V4 checkpoint: which variant, which quant."""

    variant: Variant
    quant: Quant


def detect_dsv4(model_path: str | None) -> Dsv4Model | None:
    """Return the dsv4 variant+quant for a LOCAL checkpoint dir, else None.

    None (leave the native path alone) if: no path, not a local dir, no
    ``config.json``, a read error, not a dsv4 model, or the quant is neither
    fp4 nor fp8. Never downloads (a bare HF repo id returns None).
    """
    if not model_path or not os.path.isdir(model_path):
        return None
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path) as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return None

    model_type = str(cfg.get("model_type", "")).lower()
    is_dsv4 = model_type.startswith("deepseek_v4") or "index_topk" in cfg
    if not is_dsv4:
        return None

    variant = _detect_variant(cfg)
    quant = _detect_quant(cfg)
    if quant is None:
        return None
    return Dsv4Model(variant=variant, quant=quant)


def _detect_variant(cfg: dict) -> Variant:
    """Pro vs Flash by dimensions. Pro: hidden 7168 / 61 layers; Flash: 4096 / 43.

    Threshold (not equality) so a minor config revision doesn't misclassify:
    hidden_size >= 6144 or >= 52 layers -> Pro, else Flash.
    """
    hidden = int(cfg.get("hidden_size", 0) or 0)
    layers = int(cfg.get("num_hidden_layers", 0) or 0)
    if hidden >= 6144 or layers >= 52:
        return "pro"
    return "flash"


def _detect_quant(cfg: dict) -> Quant | None:
    """fp4/fp8 from quantization_config; None if neither (unquantized)."""
    qc = cfg.get("quantization_config") or {}
    blob = json.dumps(qc).lower() if isinstance(qc, dict) else str(qc).lower()
    if "fp4" in blob or "mxfp4" in blob or "e2m1" in blob:
        return "fp4"
    if "fp8" in blob or "e4m3" in blob:
        return "fp8"
    return None
