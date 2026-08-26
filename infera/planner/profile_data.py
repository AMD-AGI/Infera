###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Loading and validation of offline profiling results.

The planner cannot invent a performance model: it needs to know, for this
model on this GPU, how TTFT grows with prompt length and how ITL degrades as
KV cache fills. That comes from the sweep in :mod:`infera.planner.profile`,
handed to the planner as a single JSON file::

    {
      "prefill": {
        "isl":          [512, 1024, 2048, 4096],
        "ttft_ms":      [45.0, 82.0, 165.0, 340.0],
        "thpt_per_gpu": [9800.0, 11200.0, 11900.0, 12100.0]
      },
      "decode": {
        "kv_usage":       [0.1, 0.3, 0.5, 0.7, 0.9],
        "context_length": [1024, 4096, 16384],
        "itl_ms":         [[...5 values...], [...], [...]],
        "thpt_per_gpu":   [[...5 values...], [...], [...]],
        "max_kv_tokens":  262144
      },
      "prefill_engine_num_gpu": 1,
      "decode_engine_num_gpu": 1
    }

``prefill`` is three parallel 1-D series indexed by prompt length: the TTFT of
an unqueued request at that length, and the prefill throughput one GPU sustains
there (tokens/s/GPU).

``decode`` is a **regular grid**: ``itl_ms[i][j]`` and ``thpt_per_gpu[i][j]``
are measured at ``context_length[i]`` and ``kv_usage[j]``. Requiring a full
grid rather than scattered samples is what lets the planner interpolate with
numpy alone -- no scipy. ``max_kv_tokens`` is the engine's total KV capacity in
tokens, used to convert a concurrency into a KV utilisation fraction.

The GPU counts are per *engine replica* (i.e. the engine's tensor/pipeline
parallel width), and are what the planner divides by to turn a per-GPU
throughput into a replica count. They may be overridden on the command line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MISSING_PROFILE_HELP = (
    "The SLA planner needs offline profiling results. Produce them with "
    "`python -m infera.planner.profile`; see the SLA planner guide in manual/."
)


class ProfileDataError(ValueError):
    """The profiling file is missing keys, or its arrays don't line up."""


@dataclass(frozen=True)
class PrefillProfile:
    """Prefill sweep: TTFT and per-GPU throughput as a function of ISL."""

    isl: np.ndarray
    ttft_ms: np.ndarray
    thpt_per_gpu: np.ndarray


@dataclass(frozen=True)
class DecodeProfile:
    """Decode sweep on a ``context_length x kv_usage`` grid."""

    kv_usage: np.ndarray
    context_length: np.ndarray
    itl_ms: np.ndarray  # shape (len(context_length), len(kv_usage))
    thpt_per_gpu: np.ndarray  # same shape
    max_kv_tokens: int


@dataclass(frozen=True)
class ProfileData:
    prefill: PrefillProfile
    decode: DecodeProfile
    prefill_engine_num_gpu: int = 1
    decode_engine_num_gpu: int = 1


def _series(block: dict, section: str, key: str) -> np.ndarray:
    if key not in block:
        raise ProfileDataError(f"profiling data: {section!r} is missing {key!r}")
    arr = np.asarray(block[key], dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ProfileDataError(f"profiling data: {section}.{key} must be a non-empty 1-D list")
    if not np.all(np.isfinite(arr)):
        raise ProfileDataError(f"profiling data: {section}.{key} contains NaN or infinity")
    return arr


def _grid(block: dict, section: str, key: str, shape: tuple[int, int]) -> np.ndarray:
    if key not in block:
        raise ProfileDataError(f"profiling data: {section!r} is missing {key!r}")
    arr = np.asarray(block[key], dtype=float)
    if arr.shape != shape:
        raise ProfileDataError(
            f"profiling data: {section}.{key} has shape {arr.shape}, expected {shape} "
            f"(rows = context_length, columns = kv_usage)"
        )
    if not np.all(np.isfinite(arr)):
        raise ProfileDataError(f"profiling data: {section}.{key} contains NaN or infinity")
    return arr


def _ascending(arr: np.ndarray, section: str, key: str) -> None:
    if np.any(np.diff(arr) <= 0):
        raise ProfileDataError(
            f"profiling data: {section}.{key} must be strictly ascending, got {arr.tolist()}"
        )


def parse_profile_data(raw: dict) -> ProfileData:
    """Validate a decoded profiling document and build a :class:`ProfileData`.

    Raises :class:`ProfileDataError` with a specific message rather than
    letting a shape mismatch surface later as a confusing scaling decision.
    """
    for section in ("prefill", "decode"):
        if not isinstance(raw.get(section), dict):
            raise ProfileDataError(f"profiling data: missing {section!r} section")

    p = raw["prefill"]
    isl = _series(p, "prefill", "isl")
    _ascending(isl, "prefill", "isl")
    ttft_ms = _series(p, "prefill", "ttft_ms")
    thpt = _series(p, "prefill", "thpt_per_gpu")
    if not (isl.size == ttft_ms.size == thpt.size):
        raise ProfileDataError(
            "profiling data: prefill.isl, prefill.ttft_ms and prefill.thpt_per_gpu "
            f"must be the same length, got {isl.size}, {ttft_ms.size}, {thpt.size}"
        )
    if np.any(isl <= 0) or np.any(ttft_ms <= 0):
        raise ProfileDataError("profiling data: prefill.isl and prefill.ttft_ms must be positive")
    if np.any(thpt <= 0):
        raise ProfileDataError("profiling data: prefill.thpt_per_gpu must be positive")

    d = raw["decode"]
    kv_usage = _series(d, "decode", "kv_usage")
    _ascending(kv_usage, "decode", "kv_usage")
    if np.any(kv_usage <= 0) or np.any(kv_usage > 1):
        raise ProfileDataError("profiling data: decode.kv_usage must be in (0, 1]")
    context_length = _series(d, "decode", "context_length")
    _ascending(context_length, "decode", "context_length")
    if np.any(context_length <= 0):
        raise ProfileDataError("profiling data: decode.context_length must be positive")
    shape = (context_length.size, kv_usage.size)
    itl_ms = _grid(d, "decode", "itl_ms", shape)
    d_thpt = _grid(d, "decode", "thpt_per_gpu", shape)
    if np.any(itl_ms <= 0):
        raise ProfileDataError("profiling data: decode.itl_ms must be positive")
    if np.any(d_thpt <= 0):
        raise ProfileDataError("profiling data: decode.thpt_per_gpu must be positive")
    max_kv_tokens = int(d.get("max_kv_tokens", 0))
    if max_kv_tokens <= 0:
        raise ProfileDataError("profiling data: decode.max_kv_tokens must be a positive integer")
    prefill_num_gpu = int(raw.get("prefill_engine_num_gpu", 1))
    decode_num_gpu = int(raw.get("decode_engine_num_gpu", 1))
    if prefill_num_gpu <= 0 or decode_num_gpu <= 0:
        raise ProfileDataError("profiling data: engine GPU counts must be positive integers")

    return ProfileData(
        prefill=PrefillProfile(isl=isl, ttft_ms=ttft_ms, thpt_per_gpu=thpt),
        decode=DecodeProfile(
            kv_usage=kv_usage,
            context_length=context_length,
            itl_ms=itl_ms,
            thpt_per_gpu=d_thpt,
            max_kv_tokens=max_kv_tokens,
        ),
        prefill_engine_num_gpu=prefill_num_gpu,
        decode_engine_num_gpu=decode_num_gpu,
    )


def load_profile_data(path: str | Path) -> ProfileData:
    """Read and validate the profiling JSON at ``path``."""
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProfileDataError(f"profiling data not found at {p}. {MISSING_PROFILE_HELP}") from exc
    except ValueError as exc:
        raise ProfileDataError(f"profiling data at {p} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProfileDataError(f"profiling data at {p} must be a JSON object")
    return parse_profile_data(raw)
