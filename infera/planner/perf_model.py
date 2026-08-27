###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Read-only capacity envelope assembled from an offline sweep.

The sweep records two surfaces. Prompt processing is a one-dimensional curve
indexed by prompt length. Token generation is a rectangular surface indexed by
mean context length and occupied KV fraction. This module exposes points on
those surfaces; policy and fleet state deliberately live elsewhere.

Values outside the measured domain use the nearest boundary. Extrapolating a
saturated engine would claim capacity that was never observed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from infera.planner.profile_data import DecodeProfile, PrefillProfile, ProfileData

_SURFACE_SAMPLES = 100


def _bounded(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))


@dataclass(frozen=True)
class PrefillPoint:
    """Measured-envelope values for one prompt length."""

    latency_ms: float
    tokens_per_second_per_gpu: float


@dataclass(frozen=True)
class DecodePoint:
    """One feasible generation operating point."""

    latency_ms: float
    tokens_per_second_per_gpu: float
    kv_fraction: float


class PrefillCurve:
    """Prompt-processing capacity indexed by prompt tokens."""

    def __init__(self, profile: PrefillProfile) -> None:
        self._isl = profile.isl
        self._ttft_ms = profile.ttft_ms
        self._thpt = profile.thpt_per_gpu
        self.min_isl = float(self._isl[0])
        self.max_isl = float(self._isl[-1])

    def point(self, prompt_tokens: float) -> PrefillPoint:
        prompt_tokens = _bounded(prompt_tokens, self.min_isl, self.max_isl)
        return PrefillPoint(
            latency_ms=float(np.interp(prompt_tokens, self._isl, self._ttft_ms)),
            tokens_per_second_per_gpu=float(np.interp(prompt_tokens, self._isl, self._thpt)),
        )


class DecodeSurface:
    """Generation capacity over context length and occupied KV fraction."""

    def __init__(self, profile: DecodeProfile, *, samples: int = _SURFACE_SAMPLES) -> None:
        self._kv_usage = profile.kv_usage
        self._context_length = profile.context_length
        self._itl_ms = profile.itl_ms
        self._thpt = profile.thpt_per_gpu
        self.max_kv_tokens = profile.max_kv_tokens
        self.min_kv_usage = float(self._kv_usage[0])
        self.max_kv_usage = float(self._kv_usage[-1])
        self.min_context_length = float(self._context_length[0])
        self.max_context_length = float(self._context_length[-1])
        self._candidates = np.linspace(
            self.min_kv_usage,
            self.max_kv_usage,
            max(2, samples),
        )

    def _occupied_fraction(self, in_flight: float, context_tokens: float) -> float:
        occupied = max(0.0, in_flight) * max(0.0, context_tokens) / self.max_kv_tokens
        return _bounded(occupied, self.min_kv_usage, self.max_kv_usage)

    def _context_slice(self, grid: np.ndarray, context_tokens: float) -> np.ndarray:
        context_tokens = _bounded(
            context_tokens,
            self.min_context_length,
            self.max_context_length,
        )
        return np.array(
            [
                np.interp(context_tokens, self._context_length, grid[:, column])
                for column in range(self._kv_usage.size)
            ]
        )

    def point(self, in_flight: float, context_tokens: float) -> DecodePoint:
        fraction = self._occupied_fraction(in_flight, context_tokens)
        latency = self._context_slice(self._itl_ms, context_tokens)
        throughput = self._context_slice(self._thpt, context_tokens)
        return DecodePoint(
            latency_ms=float(np.interp(fraction, self._kv_usage, latency)),
            tokens_per_second_per_gpu=float(np.interp(fraction, self._kv_usage, throughput)),
            kv_fraction=fraction,
        )

    def capacity_within(self, latency_budget_ms: float, context_tokens: float) -> DecodePoint:
        """Return the fastest sampled point allowed by the latency budget.

        Measurements need not improve or degrade in a single direction as KV
        occupancy rises, so every candidate is evaluated. If none qualifies,
        the least-occupied measured point is returned and the caller can report
        that the requested budget is below the measured floor.
        """
        itl_curve = np.interp(
            self._candidates,
            self._kv_usage,
            self._context_slice(self._itl_ms, context_tokens),
        )
        thpt_curve = np.interp(
            self._candidates,
            self._kv_usage,
            self._context_slice(self._thpt, context_tokens),
        )
        allowed = np.flatnonzero(itl_curve <= latency_budget_ms)
        selected = int(allowed[np.argmax(thpt_curve[allowed])]) if allowed.size else 0
        return DecodePoint(
            latency_ms=float(itl_curve[selected]),
            tokens_per_second_per_gpu=float(thpt_curve[selected]),
            kv_fraction=float(self._candidates[selected]),
        )


class PerfModel:
    """Facade over both profiled capacity surfaces."""

    def __init__(self, data: ProfileData) -> None:
        self._prefill = PrefillCurve(data.prefill)
        self._decode = DecodeSurface(data.decode)
        self.prefill_engine_num_gpu = data.prefill_engine_num_gpu
        self.decode_engine_num_gpu = data.decode_engine_num_gpu

    def prompt_capacity(self, prompt_tokens: float) -> PrefillPoint:
        return self._prefill.point(prompt_tokens)

    def generation_point(self, in_flight: float, context_tokens: float) -> DecodePoint:
        return self._decode.point(in_flight, context_tokens)

    def generation_capacity(
        self,
        latency_budget_ms: float,
        context_tokens: float,
    ) -> DecodePoint:
        return self._decode.capacity_within(latency_budget_ms, context_tokens)
