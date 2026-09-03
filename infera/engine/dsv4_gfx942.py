###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""gfx942 (MI300X / MI325X, CDNA3) DeepSeek-V4 support policy.

gfx942 has no FP4 MFMA, so the dsv4 family's runnable configurations differ by
engine. This module is the single place that encodes that contract and applies
the knobs (env + CLI) each supported combination needs — set-if-unset so an
operator/launcher always overrides. It enables NO third-party source patches:
unsupported combinations fail fast with an actionable message instead.

Support matrix (variant x quant x engine) on gfx942:

    variant  quant  vllm                     sglang                atom
    Pro      fp4    native (triton unfused)  UNSUPPORTED           UNSUPPORTED
    Pro      fp8    UNSUPPORTED              native (env+CLI)      native (env)
    Flash    fp4    native (triton unfused)  UNSUPPORTED           UNSUPPORTED
    Flash    fp8    UNSUPPORTED              native (env+CLI+MTP)  native (env+MTP)

Why:
  * fp4 -> vLLM. Its ``triton_unfused`` MoE upcasts fp4->bf16 *in-kernel*, so the
    experts stay packed in VRAM and no FP4 MFMA is ever asked for. Measured green
    for Pro on both e2e tiers.
  * fp4 -> NOT SGLang. Its packed-FP4 expert kernels all gate on NVIDIA SM
    versions, and the one fallback, ``SGLANG_DSV4_FP4_DEQUANT``, is dead code
    here: the dequant sits behind an ``else`` that gfx942 never reaches, FP8
    being e4m3**fnuz** on CDNA3. Both sides of that branch measured.
  * fp4 -> NOT ATOM. All three of its MXFP4 MoE paths fail, the last of them on
    hardware — aiter's CK MX-FP4 MoE has no gfx942 device image, CDNA3 having no
    FP4 MFMA to build one for.
  * fp8 -> sglang/atom: run natively. Not validated on vLLM, so it raises there.

Both fp4 exclusions are reachable by unpacking the experts to block-FP8 at load,
and that is deliberately NOT what this module does. It takes a third-party source
patch either way (SGLang needs the dequant hoisted ahead of the fnuz arm; ATOM
needs ``quant_method`` swapped to ``Fp8MoEMethod``), and for Pro it does not pay
even when it works: 195.8 GiB a card at tp8 against 191.98 usable, measured
twice, short before a byte of KV cache. Upstream's already-unpacked re-packaging
``sgl-project/DeepSeek-V4-Pro-FP8`` lands in the same place.

**Flash is what makes these engines useful on these cards**, and the difference
is size rather than support: same architecture, 43 layers against Pro's 61, 4096
hidden against 7168, which fits at tp8 or tp4 in either quant. So SGLang and ATOM
reach a dsv4 checkpoint here without anyone patching them, which is the whole
reason the Flash rows exist in the e2e matrix.

Detection reads ``config.json`` from a LOCAL dir only (never downloads). Variant
is keyed off model dimensions (Pro: hidden 7168 / 61 layers; Flash: 4096 / 43),
never the directory name. Quant is the ROUTED EXPERTS' dtype, not the
checkpoint's: dsv4 ships hybrid (FP8 attention, MXFP4 experts) and it is the
expert half that gfx942 has no kernel for. See :func:`_detect_quant`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Literal

from infera.common.arch import is_gfx942

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
    architectures = [str(a).lower() for a in cfg.get("architectures") or []]
    if not _is_dsv4(model_type, architectures, cfg):
        return None

    variant = _detect_variant(cfg)
    quant = _detect_quant(cfg, model_path)
    if quant is None:
        return None
    return Dsv4Model(variant=variant, quant=quant)


def _is_dsv4(model_type: str, architectures: list[str], cfg: dict) -> bool:
    """True iff this config is a DeepSeek-V4 checkpoint.

    ``index_topk`` on its own only says "sparse attention": other families ship a
    DSA variant too (GLM-5.2 is ``glm_moe_dsa`` with ``index_topk`` 2048), and
    matching on it alone handed them the dsv4 gfx942 knobs — dsv4 attention
    backend, no shared-experts fusion, the FlashMLA hack — that are not theirs. So
    trust whatever the checkpoint calls itself, and fall back to ``index_topk``
    only when it names no architecture at all (a re-exported dsv4 checkpoint).
    """
    if model_type.startswith("deepseek_v4"):
        return True
    if architectures:
        return any(a.startswith("deepseekv4") for a in architectures)
    return "index_topk" in cfg


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


def _detect_quant(cfg: dict, model_path: str) -> Quant | None:
    """The dtype of the ROUTED EXPERT weights; None if unquantized.

    This is deliberately not "the checkpoint's quantization", because a dsv4
    checkpoint does not have one. ``deepseek-ai/DeepSeek-V4-Pro`` is *hybrid*:
    attention is FP8 (``F8_E4M3``, 21.6 GiB) while the routed experts are MXFP4
    (``I8``, two values per byte, 732.4 GiB). Since what gfx942 lacks is an FP4
    *MoE* kernel, the experts are the half that decides which engine can serve
    the checkpoint.

    ``quantization_config`` describes only the attention half — it reads
    ``quant_method: fp8`` on a checkpoint whose experts are FP4 — so consulting
    it first, as this function used to, reported fp8 and routed the model to the
    two engines that cannot serve it while rejecting the one that can.

    Three sources, most authoritative first:

    1. ``expert_dtype``, dsv4's own top-level field. vLLM reads this same key
       (defaulting to fp4 when absent).
    2. The routed experts' on-disk dtype, from the safetensors headers. Both
       SGLang (``configs/deepseek_v4.py::try_detect_fp4_experts``) and ATOM
       probe this rather than trusting the config, which is worth copying: a
       re-exported or converted checkpoint keeps its weights honest long after
       an inherited config field has gone stale.
    3. ``quantization_config``, for a checkpoint that is genuinely uniform.
    """
    declared = str(cfg.get("expert_dtype", "") or "").lower()
    if "fp4" in declared:
        return "fp4"
    if "fp8" in declared:
        return "fp8"

    probed = _probe_routed_expert_dtype(model_path)
    if probed is not None:
        return probed

    qc = cfg.get("quantization_config") or {}
    blob = json.dumps(qc).lower() if isinstance(qc, dict) else str(qc).lower()
    if "fp4" in blob or "mxfp4" in blob or "e2m1" in blob:
        return "fp4"
    if "fp8" in blob or "e4m3" in blob:
        return "fp8"
    return None


# safetensors dtype strings, as they appear in a shard header, for the two
# layouts a dsv4 routed expert ships in. FP4 has no safetensors dtype of its
# own, so an MXFP4 checkpoint stores two values per byte under an integer tag.
_FP4_ONDISK_DTYPES = frozenset({"U8", "I8", "F4", "F4_E2M1", "E2M1"})
_FP8_ONDISK_DTYPES = frozenset({"F8_E4M3", "F8_E5M2"})


def _probe_routed_expert_dtype(model_path: str) -> Quant | None:
    """Routed-expert dtype read from the safetensors headers; None if unreadable.

    Reads the header of ONE shard. A safetensors header is a JSON blob whose
    byte length is the file's first 8 bytes, so this touches kilobytes and never
    the hundreds of GiB behind them. Shared experts are excluded: they stay FP8
    even in an MXFP4 checkpoint, so matching one would invert the answer.
    """
    index = os.path.join(model_path, "model.safetensors.index.json")
    try:
        with open(index) as fh:
            weight_map = json.load(fh).get("weight_map") or {}
        name = next(
            k
            for k in weight_map
            if ".experts." in k and k.endswith(".weight") and "shared" not in k
        )
        shard = os.path.join(model_path, str(weight_map[name]))
        with open(shard, "rb") as fh:
            header_len = int.from_bytes(fh.read(8), "little")
            if not 0 < header_len <= 100 * 1024 * 1024:
                return None
            header = json.loads(fh.read(header_len))
    except (OSError, ValueError, TypeError, AttributeError, StopIteration):
        return None

    dtype = str((header.get(name) or {}).get("dtype", "")).upper()
    if dtype in _FP4_ONDISK_DTYPES:
        return "fp4"
    if dtype in _FP8_ONDISK_DTYPES:
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

    # engine == "atom" (the only remaining supported engine at this point;
    # any other engine was already rejected by the _SUPPORTED check above).
    _apply_env(_ATOM_FP8_ENV, engine)
    if model.variant == "flash":
        argv = _append_cli_if_absent(argv, _ATOM_FLASH_MTP_CLI)
    return argv


def _unsupported_message(engine: str, model: Dsv4Model) -> str:
    """Actionable error naming the engine that DOES support this combo."""
    variant = model.variant.capitalize()
    if model.quant == "fp4" and engine == "sglang":
        return (
            f"DeepSeek-V4-{variant} has MXFP4 routed experts, and this SGLang "
            f"build cannot serve them on gfx942. It has no packed-FP4 expert "
            f"kernel here (Marlin and FlashInfer gate on is_sm90/sm120; "
            f"`humming` is not a distribution that exists), so the only route is "
            f"SGLANG_DSV4_FP4_DEQUANT — and that env is dead code on this "
            f"hardware: the dequant sits in the `else` of "
            f"process_weights_after_loading_block_quant's `_is_fp8_fnuz / "
            f"_use_aiter / _is_cpu / else` chain, and _is_fp8_fnuz is True on "
            f"gfx942 because MI300X FP8 is e4m3fnuz. No flag reaches it: aiter "
            f"on leaves the experts packed until decode graph capture fails, "
            f"aiter off trips the fnuz normalisation's dtype assert. TWO WAYS "
            f"FORWARD, neither of them a flag: hand SGLang an FP8 checkpoint "
            f"instead (sgl-project/DeepSeek-V4-Flash-FP8 is block-FP8 and native "
            f"here), or serve the FP4 weights on vLLM, whose triton_unfused MoE "
            f"upcasts fp4->bf16 in-kernel and never unpacks them."
        )
    if model.quant == "fp4" and engine == "atom":
        return (
            f"DeepSeek-V4-{variant} has MXFP4 routed experts, and all three of "
            f"ATOM's MXFP4 MoE paths are dead on gfx942 — measured, in this "
            f"order: (1) the triton path, which gfx94x takes by default, imports "
            f"`triton_kernels.routing`, absent from the image's ROCm fork of "
            f"triton_kernels along with every symbol it wants; (2) "
            f"ATOM_USE_TRITON_MOE=0 reaches aiter, which selects an untuned "
            f"FlyDSL fp4 kernel that fails to compile ('LLVM ERROR: Do not know "
            f"how to expand this operator's operand!'); (3) plus "
            f"AITER_FLYDSL_FORCE=0 reaches aiter's CK MX-FP4 MoE, whose module "
            f"loads but whose kernel has no gfx942 device image ('Cannot find "
            f"Symbol ck::kernel_moe_mxgemm_2lds<...f4x2_pk_t,e8m0_bexp_t...>'). "
            f"(3) is CDNA3 having no FP4 MFMA, so no flag or rebuild reaches it. "
            f"Hand ATOM an FP8 checkpoint instead "
            f"(sgl-project/DeepSeek-V4-Flash-FP8 is block-FP8 and native here), "
            f"or serve the FP4 weights on vLLM, whose triton_unfused MoE upcasts "
            f"fp4->bf16 in-kernel so no FP4 MFMA is ever asked for."
        )
    if model.quant == "fp4":
        return (
            f"DeepSeek-V4-{variant} has MXFP4 routed experts, which are not "
            f"supported on {engine} on gfx942 (MI300X/MI325X). Use vLLM: its "
            f"triton_unfused MoE upcasts fp4->bf16 in-kernel, so the experts "
            f"stay packed in VRAM."
        )
    # fp8 on vllm
    return (
        f"DeepSeek-V4-{variant} FP8 is not supported on {engine} on gfx942 "
        f"(MI300X/MI325X). Use sglang or atom for FP8 dsv4."
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
