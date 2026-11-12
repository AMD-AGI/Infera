###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The device a non-autoregressive step is priced on.

Peaks come from the same tables the autoregressive projector uses, so the two
simulators cannot disagree about what an MI355X is: the HBM bandwidth and dense
BF16 rate from the Origami roofline profiles, the compute unit count from the
same profile, and the links from the collective model's defaults. Nothing here
is a new measurement.

The dLLM-Sim proposal quotes 5.03 PFLOP/s FP8, 8.0 TB/s and 1,194.8 GB/s
Infinity Fabric and divides by them directly. The profile's 2.5 PFLOP/s dense
BF16 doubles to the same FP8 peak, and its 8.0 TB/s is the same HBM figure; what
differs is that a kernel is charged the achievable fraction of each peak rather
than the peak, and the links are the collective model's 1,024 GB/s node and
50 GB/s NIC rates with that model's own efficiencies applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from infera.projection.core.projection.module_profilers.collective_args import CollectiveArgs
from infera.projection.core.projection.simulation_backends.origami_backend import (
    _KNOWN_PROFILES,
    _OPERAND_BYTES,
    _PEAK_MULTIPLE,
    _ROOFLINE_COMPUTE_EFF,
    _ROOFLINE_MEM_EFF,
)

_HBM_GB = {"mi300x": 192.0, "gfx942": 192.0, "mi325x": 256.0, "mi355x": 288.0, "gfx950": 288.0}
GEMM_BACKENDS = ("auto", "origami", "roofline")


class OrigamiGemmPricer:
    """InferaSim's Origami GEMM backend, memoised per problem.

    The backend already caps Origami's small-M over-prediction at the same
    efficiency-adjusted roofline the ops use, so a problem it prices is never
    slower than the roofline would have said.
    """

    def __init__(self, arch: str) -> None:
        from infera.projection.core.projection.simulation_backends.origami_backend import (
            OrigamiGEMMBackend,
        )

        self._backend = OrigamiGEMMBackend(gpu_arch=arch)
        self._cache: dict[tuple, float] = {}

    @property
    def available(self) -> bool:
        return self._backend.is_available()

    def time_ms(self, gemm: tuple[int, int, int, int], dtype: str) -> float:
        key = (gemm, dtype)
        hit = self._cache.get(key)
        if hit is None:
            m, n, k, batch = (max(1, int(x)) for x in gemm)
            hit = self._backend.simulate_gemm(m, n, k, dtype=dtype, batch=batch).forward_time_ms
            self._cache[key] = hit
        return hit


def dtype_bytes(dtype: str) -> float:
    """Bytes one element occupies, block scales included (mxfp4 is 0.53125)."""
    return _OPERAND_BYTES.get(str(dtype or "bf16").lower().strip(), 2.0)


def dtype_peak_multiple(dtype: str) -> float:
    """Matrix-core rate relative to BF16 for a GEMM whose operands are ``dtype``."""
    return _PEAK_MULTIPLE.get(str(dtype or "bf16").lower().strip(), 1.0)


@dataclass
class DLLMHardware:
    """One GPU and the two links it reaches its peers over."""

    name: str = "mi355x"
    peak_tflops_bf16: float = 2500.0
    hbm_gbps: float = 8000.0
    hbm_gb: float = 288.0
    compute_units: int = 256
    gpus_per_node: int = 8
    # Achievable fractions of the two peaks, shared with the GEMM roofline.
    compute_eff: float = _ROOFLINE_COMPUTE_EFF
    mem_eff: float = _ROOFLINE_MEM_EFF
    # Host-side cost of issuing one kernel when the step is not graph-captured.
    kernel_launch_us: float = CollectiveArgs.kernel_launch_latency
    # Raw per-GPU link rates (GB/s). The collective model applies its own
    # efficiencies; these are what it is handed.
    node_bw_gbps: float = CollectiveArgs.node_bw
    pod_bw_gbps: float = CollectiveArgs.pod_bw
    # Anything else the collective model accepts (``node_topology``,
    # ``nics_per_node``, ...), passed through untouched.
    collective_overrides: dict[str, Any] = field(default_factory=dict)
    # Prices a GEMM op's ``(M, N, K, batch)`` problem; None = the roofline.
    gemm_pricer: Any = field(default=None, repr=False, compare=False)

    def with_gemm_backend(self, backend: str = "auto") -> DLLMHardware:
        """``origami``: the tile model (raises if unavailable); ``roofline``:
        none; ``auto``: Origami when it is installed."""
        backend = str(backend or "auto").lower().strip()
        if backend not in GEMM_BACKENDS:
            raise ValueError(f"[dllm-sim] gemm_backend must be one of {GEMM_BACKENDS}")
        if backend == "roofline":
            return replace(self, gemm_pricer=None)
        pricer = OrigamiGemmPricer(self.name)
        if not pricer.available:
            if backend == "origami":
                raise RuntimeError("[dllm-sim] gemm_backend=origami but Origami is not installed")
            return replace(self, gemm_pricer=None)
        return replace(self, gemm_pricer=pricer)

    @classmethod
    def from_arch(cls, arch: str = "mi355x", **overrides: Any) -> DLLMHardware:
        key = str(arch).lower().strip()
        prof = _KNOWN_PROFILES.get(key)
        if prof is None:
            raise ValueError(f"[dllm-sim] unknown GPU architecture {arch!r}")
        hw = cls(
            name=key,
            peak_tflops_bf16=float(prof.peak_tflops_bf16),
            hbm_gbps=float(prof.hbm_bandwidth_gbps),
            hbm_gb=_HBM_GB.get(key, 192.0),
            compute_units=int(prof.n_cu),
        )
        unknown = [k for k in overrides if not hasattr(hw, k)]
        if unknown:
            raise ValueError(f"[dllm-sim] unknown hardware field(s): {', '.join(unknown)}")
        return replace(hw, **overrides)

    def peak_flops(self, dtype: str) -> float:
        """Achievable matrix FLOP/s for operands of ``dtype``."""
        return self.peak_tflops_bf16 * 1e12 * dtype_peak_multiple(dtype) * self.compute_eff

    def mem_bytes_per_s(self) -> float:
        """Achievable HBM bytes/s."""
        return self.hbm_gbps * 1e9 * self.mem_eff

    def collective_config(self) -> dict[str, Any]:
        """The ``hardware_config`` the collective model is built from."""
        return {
            "node_bw": self.node_bw_gbps,
            "pod_bw": self.pod_bw_gbps,
            "kernel_launch_latency": self.kernel_launch_us,
            **self.collective_overrides,
        }
