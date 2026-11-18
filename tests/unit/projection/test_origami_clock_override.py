"""The GPU clock override must reach the GEMM roofline cap, not only the tile model."""

import pytest

from infera.projection.core.projection.inference_projection import pareto
from infera.projection.core.projection.simulation_backends.origami_backend import (
    OrigamiGEMMBackend,
    profile_peaks,
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


def test_profile_peaks_follow_arch_and_clock(monkeypatch):
    monkeypatch.delenv("INFERASIM_GPU_CLOCK_MHZ", raising=False)
    assert profile_peaks("mi355x") == pytest.approx((2500.0, 8000.0))
    assert profile_peaks("mi325x", 1524) == pytest.approx((1307.0 * 1524 / 2100, 6000.0))
    monkeypatch.setenv("INFERASIM_GPU_CLOCK_MHZ", "1050")
    assert profile_peaks("mi355x")[0] == pytest.approx(1250.0)


def test_profile_peaks_without_a_profile_use_mi300x(monkeypatch):
    monkeypatch.delenv("INFERASIM_GPU_CLOCK_MHZ", raising=False)
    assert profile_peaks("gfx9999") == profile_peaks("mi300x")


def test_pareto_passes_the_clock_to_every_projection():
    ap = pareto.build_parser()
    a = ap.parse_args(["--model", "x", "--gpu-clock-mhz", "1524", "--", "--kv-offload"])
    assert pareto.projection_extra(a) == ["--kv-offload", "--gpu-clock-mhz", "1524"]
    assert "--gpu-clock-mhz" not in pareto.projection_extra(ap.parse_args(["--model", "x"]))
