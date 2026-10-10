###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The analytical drafter: what a draft costs and how many drafts survive.

The acceptance side is held to the measurements it is fitted from -- the
golden depth sweeps and SPEED-Bench's per-domain acceptance -- and to the one
invariant a caller relies on: a reference acceptance is reproduced exactly at
its own depth. The cost side is held to the shapes that distinguish drafters:
an autoregressive draft grows with depth, a block draft barely does, and the
best depth falls as the batch turns the step compute-bound.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infera.projection.core.projection.inference_projection import speculative as sp
from infera.projection.core.projection.inference_projection.performance import (
    InferencePerformanceProjector,
    PhaseForwardTimes,
)


@pytest.mark.parametrize("source", sorted(sp._DEPTH_CURVES))
def test_depth_fit_tracks_the_golden_sweep(source):
    a, d = sp._fit_curve(source)
    assert 0.0 < a <= sp.AR_MAX
    assert 0.5 <= d <= 1.0
    for k, al in sp._DEPTH_CURVES[source].items():
        assert sp.acceptance_length(sp.chain_rates(a, d, k)) == pytest.approx(al, abs=0.12)


@pytest.mark.parametrize("method", ["mtp", "eagle3", "draft_model", "ngram"])
def test_class_levels_reproduce_speed_bench(method):
    d = sp.decay_for(method)
    rates = sp._class_rates(method)
    for cls, domains in sp.ENTROPY_CLASSES.items():
        vals = [al for dom in domains for al in sp._SPEED_BENCH_AL[method][dom]]
        mean_al = sum(vals) / len(vals)
        got = sp.acceptance_length(sp.chain_rates(rates[cls], d, sp._SPEED_BENCH_DL))
        assert got == pytest.approx(mean_al, abs=0.05)


@pytest.mark.parametrize("method", ["mtp", "eagle3", "dspark", "draft_model", "ngram"])
def test_entropy_orders_acceptance(method):
    al = {
        e: sp.acceptance_length(sp.acceptance_profile(method, 5, entropy=e))
        for e in ("low", "mixed", "high")
    }
    assert al["low"] > al["mixed"] > al["high"] > 1.0


@pytest.mark.parametrize(
    "method,k,al", [("mtp", 3, 2.49), ("mtp", 5, 3.61), ("dspark", 6, 3.77), ("eagle3", 3, 2.78)]
)
def test_reference_is_reproduced_at_its_depth(method, k, al):
    rates = sp.acceptance_profile(method, k, entropy="mixed", reference=(k, al))
    assert sp.acceptance_length(rates) == pytest.approx(al, abs=1e-6)
    # Deeper drafts commit more, by less each time.
    deeper = sp.acceptance_length(sp.acceptance_profile(method, k + 2, reference=(k, al)))
    assert al < deeper < al + 2 * rates[-1]


def test_legacy_rate_reads_back_as_its_length():
    k, a = 3, 0.7
    ref = sp.reference_from_rate(k, a)
    assert ref == (k, pytest.approx(1 + a + a * a + a**3))
    assert sp.reference_from_rate(0, a) is None


def test_entropy_axis():
    assert sp.resolve_entropy(None) == 0.0
    assert sp.resolve_entropy("low_entropy") == -1.0
    assert sp.resolve_entropy("HIGH") == 1.0
    assert sp.resolve_entropy("0.5") == 0.5
    assert sp.resolve_entropy(4) == 1.0
    lo = sp.entropy_multiplier("mtp", -1)
    half = sp.entropy_multiplier("mtp", -0.5)
    assert lo > half > 1.0
    assert half == pytest.approx(lo**0.5)


def test_drafter_shapes():
    v = 129280
    mtp = sp.drafter_for("deepseek_mtp", target_vocab=v, mtp_layers=1, mtp_window=128)
    assert (mtp.method, mtp.layers, mtp.moe, mtp.parallel, mtp.window) == (
        "mtp",
        1,
        True,
        False,
        128,
    )
    e3 = sp.drafter_for("EAGLE3", target_vocab=v)
    assert not e3.moe and e3.vocab_fraction == pytest.approx(32000 / v)
    ds = sp.drafter_for("dspark", target_vocab=v)
    assert ds.parallel and ds.layers == 3 and ds.window == 128 and ds.serial_head_rank == 256
    assert sp.drafter_for("ngram", target_vocab=v).layers == 0
    with pytest.raises(ValueError):
        sp.drafter_for("draft_model", target_vocab=v)
    assert sp.canonical_method("nope") is None


# ---- projector side -------------------------------------------------------------

_LAYERS = 60


class _Drafting:
    """The projector's drafter methods over a toy step-cost model.

    A layer costs a fixed read plus a term linear in the tokens it carries, so
    the step is memory-bound at small batch and compute-bound at large.
    """

    _drafter = InferencePerformanceProjector._drafter
    _drafter_ms = InferencePerformanceProjector._drafter_ms
    _draft_overhead_ms = InferencePerformanceProjector._draft_overhead_ms
    _spec_k = InferencePerformanceProjector._spec_k
    _spec_rates = InferencePerformanceProjector._spec_rates
    _spec_tokens_per_step = InferencePerformanceProjector._spec_tokens_per_step
    _best_spec_k = InferencePerformanceProjector._best_spec_k
    speculative_schedule = InferencePerformanceProjector.speculative_schedule

    def __init__(self, method, *, k=3, accept=0.7, kmax=0, conc=1):
        self.cfg = SimpleNamespace(
            request_config=SimpleNamespace(
                speculative_method=method,
                speculative_num_tokens=k,
                speculative_acceptance_rate=accept,
                speculative_entropy=None,
                speculative_max_num_tokens=kmax,
                speculative_draft_layers=0,
                speculative_draft_vocab=0,
                speculative_draft_window=None,
                speculative_draft_cost_factor=0.0,
                input_seq_len=8192,
                output_seq_len=1024,
                resolved_max_concurrency=lambda: conc,
            ),
            model_config=SimpleNamespace(
                num_layers=_LAYERS, padded_vocab_size=129280, mtp_num_layers=1, hidden_size=7168
            ),
        )
        self._measured_mode = False
        self._conc = conc

    def _mtp_window(self):
        return 0

    def _decode_occupancy_ms(self):
        return 0.0

    def _effective_concurrency(self):
        return {"concurrency": self._conc}

    def _forward_times(self, batch, q_len, phase, kv_len):
        layer = 0.1 + 0.0005 * batch * q_len
        return PhaseForwardTimes(
            layers_ms=_LAYERS * layer,
            embedding_ms=0.0,
            final_norm_ms=0.0,
            output_ms=0.2 + 0.001 * batch * q_len,
            dense_layer_ms=layer,
            moe_layer_ms=layer,
            sampling_ms=0.01 * batch * q_len,
        )

    def _analytic_decode_step_ms(self, batch, kv_len, q_len=1):
        ft = self._forward_times(batch, q_len, "decode", kv_len)
        return ft.total_ms + self._drafter_ms(batch, kv_len, q_len - 1)


def test_reference_rate_keeps_the_legacy_tokens_per_step():
    p = _Drafting("mtp", k=3, accept=0.7)
    assert p._spec_tokens_per_step() == pytest.approx(1 + 0.7 + 0.49 + 0.343)


def test_autoregressive_draft_grows_with_depth_block_draft_does_not():
    mtp = _Drafting("mtp")
    ds = _Drafting("dspark")
    m3, m7 = mtp._drafter_ms(8, 8192, 3), mtp._drafter_ms(8, 8192, 7)
    d3, d7 = ds._drafter_ms(8, 8192, 3), ds._drafter_ms(8, 8192, 7)
    assert m7 > 1.8 * m3
    assert 1.0 < d7 / d3 < m7 / m3
    assert _Drafting("ngram")._drafter_ms(8, 8192, 3) == 0.0


def test_drafter_is_off_in_benchmark_mode_and_without_a_method():
    p = _Drafting("mtp")
    p._measured_mode = True
    assert p._drafter() is None
    assert _Drafting(None)._drafter() is None


def test_best_depth_falls_as_the_batch_grows():
    small = _Drafting("mtp", k=3, accept=0.8, kmax=7, conc=1)
    large = _Drafting("mtp", k=3, accept=0.8, kmax=7, conc=512)
    assert small._spec_k() > large._spec_k()
    k, rates = small.speculative_schedule()
    assert k == small._spec_k() and len(rates) == k
