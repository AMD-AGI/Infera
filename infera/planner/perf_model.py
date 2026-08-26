###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Performance model built from pre-deployment profiling results.

Two questions the planner needs answered, both from offline measurements:

  * **Prefill** -- at this prompt length, what TTFT does an unqueued request
    see, and how many prefill tokens/s does one GPU sustain?
  * **Decode** -- at this context length and this KV utilisation, what ITL do
    requests see, and how many decode tokens/s does one GPU sustain? And
    inverted: what is the highest per-GPU throughput still within an ITL
    budget?

Interpolation is numpy-only. Prefill is a 1-D linear interpolation over the
profiled ISL points; decode is bilinear over the profiled
``context_length x kv_usage`` grid, which is what insisting on a full
rectangular grid buys -- interpolating scattered samples instead would mean
pulling in scipy. Queries outside the profiled range are clamped to the nearest
edge rather than extrapolated -- a linear extrapolation of a saturating
throughput curve invents capacity that isn't there.

The model is stateless and cheap to query; runtime deviation from these
predictions is handled by the correction factors in
:mod:`infera.planner.core`, not here.
"""

from __future__ import annotations

import numpy as np

from infera.planner.profile_data import DecodeProfile, PrefillProfile, ProfileData

# How finely to resample the KV-usage axis when searching for the operating
# point that just meets an ITL budget. The profiled grid is typically only a
# handful of columns wide, so the search would otherwise snap to coarse steps.
_KV_SEARCH_RESOLUTION = 100


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))


class PrefillPerfModel:
    """TTFT and per-GPU prefill throughput as a function of prompt length."""

    def __init__(self, profile: PrefillProfile) -> None:
        self._isl = profile.isl
        self._ttft_ms = profile.ttft_ms
        self._thpt = profile.thpt_per_gpu
        self.min_isl = float(self._isl[0])
        self.max_isl = float(self._isl[-1])

    def interpolate_ttft(self, isl: float) -> float:
        """TTFT in milliseconds for an unqueued request of length ``isl``."""
        return float(np.interp(_clamp(isl, self.min_isl, self.max_isl), self._isl, self._ttft_ms))

    def interpolate_thpt_per_gpu(self, isl: float) -> float:
        """Prefill throughput in tokens/s/GPU at length ``isl``."""
        return float(np.interp(_clamp(isl, self.min_isl, self.max_isl), self._isl, self._thpt))


class DecodePerfModel:
    """ITL and per-GPU decode throughput over the profiled decode grid."""

    def __init__(self, profile: DecodeProfile, *, resolution: int = _KV_SEARCH_RESOLUTION) -> None:
        self._kv_usage = profile.kv_usage
        self._context_length = profile.context_length
        self._itl_ms = profile.itl_ms
        self._thpt = profile.thpt_per_gpu
        self.max_kv_tokens = profile.max_kv_tokens
        self.min_kv_usage = float(self._kv_usage[0])
        self.max_kv_usage = float(self._kv_usage[-1])
        self.min_context_length = float(self._context_length[0])
        self.max_context_length = float(self._context_length[-1])
        # Fine KV-usage axis used by find_best_throughput_per_gpu.
        self._kv_search = np.linspace(self.min_kv_usage, self.max_kv_usage, max(2, resolution))

    def kv_usage_for(self, concurrency: float, context_length: float) -> float:
        """Fraction of the engine's KV cache held by ``concurrency`` requests
        of ``context_length`` tokens each, clamped to the profiled range."""
        usage = max(0.0, concurrency) * max(0.0, context_length) / self.max_kv_tokens
        return _clamp(usage, self.min_kv_usage, self.max_kv_usage)

    def _row_at(self, grid: np.ndarray, context_length: float) -> np.ndarray:
        """Slice ``grid`` at ``context_length``, one value per profiled KV usage.

        Half of the bilinear interpolation: linear along the context-length
        axis, leaving a 1-D curve over KV usage for the caller to index or
        search.
        """
        cl = _clamp(context_length, self.min_context_length, self.max_context_length)
        return np.array(
            [np.interp(cl, self._context_length, grid[:, j]) for j in range(self._kv_usage.size)]
        )

    def interpolate_itl(self, concurrency: float, context_length: float) -> float:
        """ITL in milliseconds at the given concurrency and context length."""
        row = self._row_at(self._itl_ms, context_length)
        return float(np.interp(self.kv_usage_for(concurrency, context_length), self._kv_usage, row))

    def interpolate_thpt_per_gpu(self, concurrency: float, context_length: float) -> float:
        """Decode throughput in tokens/s/GPU at the given operating point."""
        row = self._row_at(self._thpt, context_length)
        return float(np.interp(self.kv_usage_for(concurrency, context_length), self._kv_usage, row))

    def find_best_throughput_per_gpu(
        self, itl_ms: float, context_length: float
    ) -> tuple[float, float, float]:
        """Highest per-GPU throughput whose ITL still fits ``itl_ms``.

        Returns ``(thpt_per_gpu, itl_ms, kv_usage)`` at the chosen operating
        point. The whole KV-usage axis is scanned rather than bisected: neither
        curve is guaranteed monotonic in KV usage -- decode throughput typically
        peaks partway up and falls off as the cache fills -- so taking the most
        loaded compliant point would settle for less throughput than a lighter
        compliant one offers, and size the pool for capacity the hardware has.

        When even the lightest profiled load misses the budget, the lightest
        point is returned; the caller can tell from the returned ITL exceeding
        its target that the SLA is unreachable on this hardware.
        """
        itl_curve = np.interp(
            self._kv_search, self._kv_usage, self._row_at(self._itl_ms, context_length)
        )
        thpt_curve = np.interp(
            self._kv_search, self._kv_usage, self._row_at(self._thpt, context_length)
        )
        within = np.flatnonzero(itl_curve <= itl_ms)
        best = int(within[np.argmax(thpt_curve[within])]) if within.size else 0
        return float(thpt_curve[best]), float(itl_curve[best]), float(self._kv_search[best])


class PerfModel:
    """The prefill and decode models plus the per-replica GPU counts."""

    def __init__(self, data: ProfileData) -> None:
        self.prefill = PrefillPerfModel(data.prefill)
        self.decode = DecodePerfModel(data.decode)
        self.prefill_engine_num_gpu = data.prefill_engine_num_gpu
        self.decode_engine_num_gpu = data.decode_engine_num_gpu
