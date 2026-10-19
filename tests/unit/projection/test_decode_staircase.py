"""Where the measured decode curve is read as a CUDA-graph staircase."""

from infera.projection.core.projection.inference_projection.performance import (
    InferencePerformanceProjector,
)

_PTS = [(1, 10.0), (8, 12.0), (16, 14.0), (32, 18.0), (64, 26.0)]


def _projector(pad=True, capture_max=0):
    p = InferencePerformanceProjector.__new__(InferencePerformanceProjector)
    p._meas_whole = {"decode": list(_PTS)}
    p._decode_pad_to_capture = pad
    p._decode_capture_max = capture_max
    p._decode_kv_slope_ms = 0.0
    p._decode_ctx_ref = 0.0
    p._decode_ctx_mid = 0.0
    return p


def test_a_batch_between_captured_sizes_pays_for_the_next_one():
    assert _projector()._measured_decode_step_ms(17) == 18.0


def test_eager_anchors_are_read_as_a_smooth_curve():
    p = _projector(pad=False)
    assert 14.0 < p._measured_decode_step_ms(17) < 18.0


def test_above_the_largest_captured_size_the_batch_runs_unpadded():
    p = _projector(capture_max=32)
    assert p._measured_decode_step_ms(32) == 18.0
    assert 18.0 < p._measured_decode_step_ms(40) < 26.0


def test_above_the_top_measured_rung_the_curve_is_extrapolated_not_clamped():
    assert _projector()._measured_decode_step_ms(128) > 26.0
