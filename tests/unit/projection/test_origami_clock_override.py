"""Per-datatype peaks and the clocks derived from them, an override for every datatype, and both reaching the roofline cap."""

import pytest

from infera.projection.core.projection.inference_projection import pareto
from infera.projection.core.projection.simulation_backends.origami_backend import (
    _FLOPS_PER_CU_CLOCK,
    _KNOWN_PROFILES,
    _PEAK_TFLOPS,
    OrigamiGEMMBackend,
    dtype_clock_mhz,
    dtype_peak_tflops,
    profile_peaks,
)
from infera.projection.core.projection.simulation_backends.sdpa_simulator import (
    _get_hardware_spec,
)


@pytest.fixture(autouse=True)
def _no_env_clock(monkeypatch):
    monkeypatch.delenv("INFERASIM_GPU_CLOCK_MHZ", raising=False)


@pytest.mark.parametrize(
    "arch, dtype, mhz",
    [
        ("mi300x", "bf16", 930),
        ("mi300x", "fp8", 1006),
        ("mi300x", "fp32", 1413),
        ("mi325x", "bf16", 1328),  # 304 CUs
        ("mi325x", "fp8", 1309),
        ("mi350x", "bf16", 1378),
        ("mi350x", "fp8", 1444),
        ("mi350x", "mxfp4", 1104),
        ("mi350x", "fp6", 865),
        ("mi350x", "fp32", 1984),
        ("mi355x", "bf16", 1645),
        ("mi355x", "fp8", 1669),
        ("mi355x", "mxfp8", 1392),
        ("mi355x", "mxfp6", 1090),
        ("mi355x", "mxfp4", 1349),
        ("mi355x", "fp6", 974),
        ("mi355x", "fp32", 2090),
        ("gfx942", "fp8", 1006),
        ("gfx950", "bf16", 1645),
    ],
)
def test_clock_is_derived_from_the_datatype_peak(arch, dtype, mhz):
    assert dtype_clock_mhz(arch, dtype) == pytest.approx(mhz, abs=1)


@pytest.mark.parametrize("arch", sorted(_PEAK_TFLOPS))
def test_cus_at_the_derived_clock_reproduce_every_peak(arch):
    profile = _KNOWN_PROFILES[arch]
    for dtype, peak in _PEAK_TFLOPS[arch].items():
        per_cu = _FLOPS_PER_CU_CLOCK[profile.arch_enum_name][dtype]
        assert profile.n_cu * per_cu * dtype_clock_mhz(arch, dtype) / 1e6 == pytest.approx(peak)


@pytest.mark.parametrize(
    "arch, dtype, runs_as",
    [
        ("mi325x", "mxfp4", "bf16"),  # upcast to BF16 on gfx942
        ("mi300x", "mxfp4", "bf16"),
        ("mi325x", "fp32", "bf16"),
        ("mi355x", "fp16", "bf16"),
        ("mi355x", "int8", "fp8"),
        ("mi350x", "mxfp8", "fp8"),
        ("mi350x", "mxfp6", "fp6"),
    ],
)
def test_datatype_without_a_peak_runs_as_its_fallback(arch, dtype, runs_as):
    assert dtype_peak_tflops(arch, dtype) == _PEAK_TFLOPS[arch][runs_as]
    assert dtype_clock_mhz(arch, dtype) == dtype_clock_mhz(arch, runs_as)


def test_arch_without_a_profile_uses_mi300x():
    assert dtype_peak_tflops("gfx9999", "fp8") == dtype_peak_tflops("mi300x", "fp8")
    assert dtype_clock_mhz("gfx9999") == dtype_clock_mhz("mi300x")


def test_override_applies_to_every_datatype(monkeypatch):
    assert {dtype_clock_mhz("mi355x", d, 1500) for d in ("bf16", "fp8", "mxfp4", "fp32")} == {1500}
    assert dtype_peak_tflops("mi355x", "bf16", 2000) == pytest.approx(256 * 4096 * 2000 / 1e6)
    assert dtype_peak_tflops("mi355x", "mxfp4", 2000) == pytest.approx(256 * 16384 * 2000 / 1e6)
    monkeypatch.setenv("INFERASIM_GPU_CLOCK_MHZ", "1200")
    assert dtype_peak_tflops("mi325x", "fp8") == pytest.approx(304 * 4096 * 1200 / 1e6)


def test_backend_peak_is_the_bf16_peak(monkeypatch):
    assert OrigamiGEMMBackend(gpu_arch="mi325x").peak_tflops_bf16 == pytest.approx(827.0)
    assert profile_peaks("mi325x") == pytest.approx((827.0, 6000.0))
    backend = OrigamiGEMMBackend(gpu_arch="mi325x", gpu_clock_mhz=1709)
    assert backend.peak_tflops_bf16 == pytest.approx(304 * 2048 * 1709 / 1e6)


def test_roofline_ceiling_follows_the_datatype_peak():
    shape = dict(m=8192, n=8192, k=8192, batch=1, native_fp8=False)
    b = OrigamiGEMMBackend(gpu_arch="mi355x")
    bf16 = b._roofline_ceiling_ms(sim_dtype="bf16", **shape)
    assert b._roofline_ceiling_ms(sim_dtype="fp32", **shape) == pytest.approx(bf16 * 1725 / 137)
    fast = OrigamiGEMMBackend(gpu_arch="mi355x", gpu_clock_mhz=2000)
    slow = OrigamiGEMMBackend(gpu_arch="mi355x", gpu_clock_mhz=1000)
    assert slow._roofline_ceiling_ms(sim_dtype="bf16", **shape) == pytest.approx(
        2 * fast._roofline_ceiling_ms(sim_dtype="bf16", **shape)
    )


def test_attention_peaks_are_the_datatype_peaks():
    hw = _get_hardware_spec("mi355x")
    assert (hw.peak_tflops_bf16, hw.peak_tflops_fp16, hw.peak_tflops_fp8) == (1725, 1725, 3500)
    assert _get_hardware_spec("mi325x", 2100).peak_tflops_bf16 == pytest.approx(
        304 * 2048 * 2100 / 1e6
    )


def test_pareto_passes_the_clock_to_every_projection():
    ap = pareto.build_parser()
    a = ap.parse_args(["--model", "x", "--gpu-clock-mhz", "1524", "--", "--kv-offload"])
    assert pareto.projection_extra(a) == ["--kv-offload", "--gpu-clock-mhz", "1524"]
    assert "--gpu-clock-mhz" not in pareto.projection_extra(ap.parse_args(["--model", "x"]))
