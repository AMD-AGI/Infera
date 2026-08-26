###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The sizing result produced by the planner core."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScalingDecision:
    """Target replica counts for one adjustment interval, plus how we got there.

    The diagnostic fields are what make a decision reviewable after the fact:
    a correction factor far from 1.0 says the profiling data no longer describes
    the deployment, which is a different problem from a genuine traffic change.
    """

    num_prefill: int
    num_decode: int

    # What the fleet looked like when the decision was made.
    observed_prefill: int = 0
    observed_decode: int = 0

    # actual / profiled latency over the observed window. Prefill is normally
    # above 1.0 (queueing adds to TTFT); decode should sit near 1.0.
    prefill_correction: float = 1.0
    decode_correction: float = 1.0

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
            f"(corrections p={self.prefill_correction:.3f} d={self.decode_correction:.3f}; "
            f"load req={self.num_req:.1f} isl={self.isl:.0f} osl={self.osl:.0f})"
        )
