###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The reviewable result emitted by one capacity-planning window."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScalingDecision:
    """Target pool sizes and the measurements that support them."""

    num_prefill: int
    num_decode: int

    # What the fleet looked like when the decision was made.
    observed_prefill: int = 0
    observed_decode: int = 0

    # Observed/profiled latency at the sampled operating point.
    prefill_latency_ratio: float = 1.0
    decode_latency_ratio: float = 1.0

    num_req: float = 0.0
    isl: float = 0.0
    osl: float = 0.0

    @property
    def changes_anything(self) -> bool:
        return self.num_prefill != self.observed_prefill or self.num_decode != self.observed_decode

    def summary(self) -> str:
        return (
            f"prefill {self.observed_prefill}->{self.num_prefill}, "
            f"decode {self.observed_decode}->{self.num_decode} "
            f"(latency ratios p={self.prefill_latency_ratio:.3f} "
            f"d={self.decode_latency_ratio:.3f}; "
            f"load req={self.num_req:.1f} isl={self.isl:.0f} osl={self.osl:.0f})"
        )
