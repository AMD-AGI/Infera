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

from infera.engine.rocm_rdma_env import is_gfx942

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


# Supported (engine, variant, quant) combos on gfx942. Anything not listed here
# raises Dsv4UnsupportedError. This tuple set IS the enforced contract.
_SUPPORTED: frozenset[tuple[str, Variant, Quant]] = frozenset(
    {
        ("vllm", "pro", "fp4"),
        ("vllm", "flash", "fp4"),
        ("sglang", "pro", "fp8"),
        ("sglang", "flash", "fp8"),
        ("atom", "pro", "fp8"),
        ("atom", "flash", "fp8"),
    }
)

# fp8 env defaults per engine (set-if-unset).
_SGLANG_FP8_ENV: dict[str, str] = {
    "HSA_NO_SCRATCH_RECLAIM": "1",  # gfx942 firmware: dist-init FATALs without it
    "SGLANG_USE_ROCM700A": "0",
    "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",  # default tilelang MLA crashes on gfx942
    "AITER_BF16_FP8_MOE_BOUND": "0",
}
_ATOM_FP8_ENV: dict[str, str] = {
    "HSA_NO_SCRATCH_RECLAIM": "1",
}

# Functional CLI defaults (set-if-unset) for sglang fp8. Pairs of (flag, value);
# value None = bare flag.
_SGLANG_FP8_CLI: list[tuple[str, str | None]] = [
    ("--attention-backend", "dsv4"),
    ("--disable-shared-experts-fusion", None),
]
# Flash-only MTP flags (broken gfx942 decode kernel -> route decode via EAGLE).
_SGLANG_FLASH_MTP_CLI: list[tuple[str, str | None]] = [
    ("--speculative-algorithm", "EAGLE"),
    ("--speculative-num-steps", "3"),
    ("--speculative-eagle-topk", "1"),
    ("--speculative-num-draft-tokens", "4"),
]
_ATOM_FLASH_MTP_CLI: list[tuple[str, str | None]] = [
    ("--method", "mtp"),
    ("--num-speculative-tokens", "3"),
]


def apply_gfx942_dsv4(model_path: str | None, *, engine: str, argv: list[str]) -> list[str]:
    """Enforce the gfx942 dsv4 support matrix and apply its knobs (set-if-unset).

    No-op (returns ``argv`` unchanged, sets no env) if not gfx942 or not a local
    dsv4 checkpoint. Otherwise, on an unsupported ``(engine, variant, quant)``
    raises :class:`Dsv4UnsupportedError`; on a supported one, sets the env
    defaults and returns ``argv`` with any missing functional CLI flags appended.
    Call ONCE at startup BEFORE the engine subprocess is spawned so env is
    inherited and injected CLI reaches the subprocess.
    """
    if not is_gfx942():
        return argv
    model = detect_dsv4(model_path)
    if model is None:
        return argv

    key = (engine, model.variant, model.quant)
    if key not in _SUPPORTED:
        raise Dsv4UnsupportedError(_unsupported_message(engine, model))

    if engine == "vllm":
        # fp4 vllm runs natively (aiter already defaulted elsewhere); nothing to do.
        return argv

    if engine == "sglang":
        _apply_env(_SGLANG_FP8_ENV, engine)
        argv = _append_cli_if_absent(argv, _SGLANG_FP8_CLI)
        if model.variant == "flash":
            argv = _append_cli_if_absent(argv, _SGLANG_FLASH_MTP_CLI)
        return argv

    if engine == "atom":
        _apply_env(_ATOM_FP8_ENV, engine)
        if model.variant == "flash":
            argv = _append_cli_if_absent(argv, _ATOM_FLASH_MTP_CLI)
        return argv

    return argv


def _unsupported_message(engine: str, model: Dsv4Model) -> str:
    """Actionable error naming the engine that DOES support this combo."""
    variant = model.variant.capitalize()
    if model.quant == "fp4":
        return (
            f"DeepSeek-V4-{variant} FP4 is not supported on {engine} on "
            f"gfx942 (MI325X): gfx942 has no native FP4 MoE kernel and infera "
            f"does not patch third-party engines. Use vLLM for FP4 dsv4 (it "
            f"upcasts fp4->bf16 in-kernel), or an FP8 checkpoint on {engine}."
        )
    # fp8 on vllm
    return (
        f"DeepSeek-V4-{variant} FP8 is not supported on {engine} on "
        f"gfx942 (MI325X). Use sglang or atom for FP8 dsv4."
    )


def _apply_env(defaults: dict[str, str], engine: str) -> None:
    """Set each var if unset; log what was applied. Operator/env always wins."""
    applied: dict[str, str] = {}
    for k, v in defaults.items():
        if os.environ.get(k) in (None, ""):
            os.environ[k] = v
            applied[k] = v
    if applied:
        logger.info(
            "gfx942 DSv4-FP8 env defaults applied for %s (set-if-unset; override via env): %s",
            engine,
            applied,
        )


def _append_cli_if_absent(argv: list[str], flags: list[tuple[str, str | None]]) -> list[str]:
    """Append each (flag[, value]) not already present. Returns a new list."""
    out = list(argv)
    appended: list[str] = []
    for flag, value in flags:
        if any(t == flag or t.startswith(flag + "=") for t in out):
            continue  # operator already set it -> leave their value
        out.append(flag)
        appended.append(flag)
        if value is not None:
            out.append(value)
            appended.append(value)
    if appended:
        logger.info(
            "gfx942 DSv4-FP8 CLI defaults appended (set-if-unset): %s",
            " ".join(appended),
        )
    return out
