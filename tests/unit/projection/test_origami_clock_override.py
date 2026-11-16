"""The GPU clock override must reach the GEMM roofline cap, not only the tile model."""

import pytest

from infera.projection.core.projection.simulation_backends.origami_backend import (
    OrigamiGEMMBackend,
)


def test_peak_tflops_defaults_to_profile_clock(monkeypatch):
    monkeypatch.delenv("INFERASIM_GPU_CLOCK_MHZ", raising=False)
    assert OrigamiGEMMBackend(gpu_arch="mi325x").peak_tflops_bf16 == pytest.approx(1307.0)


@pytest.mark.parametrize("clock_mhz", [1500, 1709, 2100])
def test_peak_tflops_scales_with_cli_clock(monkeypatch, clock_mhz):
    monkeypatch.delenv("INFERASIM_GPU_CLOCK_MHZ", raising=False)
    backend = OrigamiGEMMBackend(gpu_arch="mi325x", gpu_clock_mhz=clock_mhz)
    assert backend.peak_tflops_bf16 == pytest.approx(1307.0 * clock_mhz / 2100)


def test_peak_tflops_scales_with_env_clock(monkeypatch):
    monkeypatch.setenv("INFERASIM_GPU_CLOCK_MHZ", "1524")
    backend = OrigamiGEMMBackend(gpu_arch="mi325x")
    assert backend.peak_tflops_bf16 == pytest.approx(1307.0 * 1524 / 2100)


def test_roofline_ceiling_follows_clock(monkeypatch):
    monkeypatch.delenv("INFERASIM_GPU_CLOCK_MHZ", raising=False)
    shape = dict(m=8192, n=8192, k=8192, batch=1, sim_dtype="bf16", native_fp8=False)
    fast = OrigamiGEMMBackend(gpu_arch="mi325x")._roofline_ceiling_ms(**shape)
    slow = OrigamiGEMMBackend(gpu_arch="mi325x", gpu_clock_mhz=1050)._roofline_ceiling_ms(**shape)
    assert slow == pytest.approx(2 * fast)
