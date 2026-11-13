###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The device a refinement step is priced on.

Kernels are priced by the same models the autoregressive projector uses, and
with the same rule about what happens without them: GEMMs by Origami's tile
model, attention by the FAv3 tile model built on Origami, and a missing Origami
is an error rather than a quiet change of model. ``backend="roofline"`` is the
explicit opt-out, for a host without Origami or a first-order estimate; it
prices every kernel at the larger of its compute and memory bounds on the
achievable peaks those same backends cap themselves at.

The peaks are the Origami profiles' (HBM bandwidth, dense BF16 rate, compute
units), and the interconnect is whatever the collective model resolves on its
own -- this class hands it only what the user overrides.
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
BACKENDS = ("origami", "roofline")


def dtype_bytes(dtype: str) -> float:
    """Bytes one element occupies, block scales included (mxfp4 is 0.53125)."""
    return _OPERAND_BYTES.get(str(dtype or "bf16").lower().strip(), 2.0)


def dtype_peak_multiple(dtype: str) -> float:
    """Matrix-core rate relative to BF16 for a GEMM whose operands are ``dtype``."""
    return _PEAK_MULTIPLE.get(str(dtype or "bf16").lower().strip(), 1.0)


class OrigamiPricer:
    """InferaSim's GEMM and SDPA simulation backends, memoised per problem.

    Built through the same factory the projector uses, so it raises the same
    error, with the same install hint, when Origami is not installed. The GEMM
    backend caps Origami's small-M over-prediction at the efficiency-adjusted
    roofline, so a GEMM it prices is never slower than the roofline says.
    """

    def __init__(self, arch: str, gpu_clock_mhz: int | None = None) -> None:
        from infera.projection.core.projection.simulation_backends.factory import (
            get_gemm_simulation_backend,
            get_sdpa_simulation_backend,
        )

        self._gemm = get_gemm_simulation_backend(
            gpu_arch=arch, gpu_clock_mhz=gpu_clock_mhz, require_simulation=True
        )
        self._sdpa = get_sdpa_simulation_backend(gpu_arch=arch, gpu_clock_mhz=gpu_clock_mhz)
        self._cache: dict[tuple, float] = {}

    def gemm_ms(self, gemm: tuple[int, int, int, int], dtype: str) -> float:
        key = ("gemm", gemm, dtype)
        hit = self._cache.get(key)
        if hit is None:
            m, n, k, batch = (max(1, int(x)) for x in gemm)
            hit = self._gemm.simulate_gemm(m, n, k, dtype=dtype, batch=batch).forward_time_ms
            self._cache[key] = hit
        return hit

    def attention_ms(self, attn: tuple, dtype: str, kv_bytes_per_token: float | None) -> float:
        """``attn`` is ``(batch, heads, q, kv, head_dim, kv_heads, causal)``."""
        key = ("sdpa", attn, dtype, kv_bytes_per_token)
        hit = self._cache.get(key)
        if hit is None:
            batch, heads, q, kv, hd, kv_heads, causal = attn
            hit = self._sdpa.simulate_sdpa(
                batch_size=int(batch),
                num_heads=int(heads),
                seq_len=int(q),
                head_dim=int(hd),
                causal=bool(causal),
                dtype="bf16" if dtype not in ("bf16", "fp16", "fp8", "fp32") else dtype,
                seq_len_kv=int(kv),
                num_heads_kv=int(kv_heads),
                kv_bytes_per_token=kv_bytes_per_token,
            ).forward_time_ms
            self._cache[key] = hit
        return hit


@dataclass
class DLLMHardware:
    """One GPU, its kernel models, and any interconnect overrides."""

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
    # The collective model's ``hardware_config`` (``node_bw``, ``pod_bw``,
    # ``node_topology``, ...). Empty = the defaults the projector uses.
    collective_overrides: dict[str, Any] = field(default_factory=dict)
    # Compute clock Origami prices at; None = its profile's (``INFERASIM_GPU_CLOCK_MHZ``
    # if set). A power-capped part sustains less under GEMM load: MI325X at its
    # 1000 W cap runs ~1510 MHz, not the 2100 MHz peak.
    gpu_clock_mhz: int | None = None
    # The clock ``peak_tflops_bf16`` is quoted at.
    peak_clock_mhz: float = 2100.0
    # Origami's GEMM and attention models; None = the roofline.
    pricer: Any = field(default=None, repr=False, compare=False)

    @property
    def backend(self) -> str:
        return "roofline" if self.pricer is None else "origami"

    def with_backend(self, backend: str = "origami") -> DLLMHardware:
        backend = str(backend or "origami").lower().strip()
        if backend not in BACKENDS:
            raise ValueError(f"[dllm-sim] backend must be one of {BACKENDS}")
        if backend == "roofline":
            return replace(self, pricer=None)
        return replace(self, pricer=OrigamiPricer(self.name, self.gpu_clock_mhz))

    @classmethod
    def from_arch(
        cls, arch: str = "mi355x", *, backend: str = "origami", **overrides: Any
    ) -> DLLMHardware:
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
            peak_clock_mhz=float(prof.compute_clock_khz) / 1000.0,
        )
        unknown = [k for k in overrides if k == "pricer" or not hasattr(hw, k)]
        if unknown:
            raise ValueError(f"[dllm-sim] unknown hardware field(s): {', '.join(unknown)}")
        return replace(hw, **overrides).with_backend(backend)

    def peak_flops(self, dtype: str) -> float:
        """Achievable matrix FLOP/s for operands of ``dtype``."""
        clock = self.gpu_clock_mhz / self.peak_clock_mhz if self.gpu_clock_mhz else 1.0
        return self.peak_tflops_bf16 * 1e12 * dtype_peak_multiple(dtype) * self.compute_eff * clock

    def mem_bytes_per_s(self) -> float:
        """Achievable HBM bytes/s."""
        return self.hbm_gbps * 1e9 * self.mem_eff

    def collective_config(self) -> dict[str, Any] | None:
        """The ``hardware_config`` handed to the collective model, if any."""
        return dict(self.collective_overrides) or None
