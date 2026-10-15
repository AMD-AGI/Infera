###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The analytical drafter: what a draft costs and how many drafts survive.

The acceptance side is held to the files it is read from: a measured depth
reproduces its golden acceptance length exactly, a stated acceptance is
reproduced at its own depth, and entropy is relative to coding, where the
curves were measured. The cost side is held to the shapes that distinguish
drafters: an autoregressive draft grows with depth, a block draft barely
does, the best depth falls as the batch turns the step compute-bound, and a
drafter that cannot be priced falls back to the flat factor.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infera.projection.core.projection.inference_projection import golden_al as ga
from infera.projection.core.projection.inference_projection import speculative as sp
from infera.projection.core.projection.inference_projection.performance import (
    InferencePerformanceProjector,
    PhaseForwardTimes,
)

_GLM = ga.load_curve("glm5.2_mtp.yaml").points("on")[0]


def _glm():
    return ga.Acceptance(_GLM, "test")


@pytest.mark.parametrize("k", sorted(_GLM))
def test_profile_reproduces_the_golden_length_at_every_measured_depth(k):
    rates = sp.acceptance_profile(_glm(), k, method="mtp")
    assert len(rates) == k
    assert sp.acceptance_length(rates) == pytest.approx(_GLM[k], abs=1e-6)
    assert all(a >= b for a, b in zip(rates, rates[1:]))


@pytest.mark.parametrize("k,al", [(3, 2.49), (5, 3.61), (1, 1.5)])
def test_stated_acceptance_is_reproduced_at_its_depth(k, al):
    rates = sp.acceptance_profile(_glm(), k, method="mtp", reference=(k, al))
    assert sp.acceptance_length(rates) == pytest.approx(al, abs=1e-6)
    deeper = sp.acceptance_length(sp.acceptance_profile(_glm(), k + 2, reference=(k, al)))
    assert al < deeper < k + 3


def test_stated_acceptance_keeps_the_curve_shape_between_depths():
    # Scaling the gain above the bonus token through (3, 2.49) keeps the
    # file's ratio between depths 3 and 5.
    rates = sp.acceptance_profile(_glm(), 5, method="mtp", reference=(3, 2.49))
    expect = 1.0 + (_GLM[5] - 1.0) * (2.49 - 1.0) / (_GLM[3] - 1.0)
    assert sp.acceptance_length(rates) == pytest.approx(expect, abs=1e-6)


def test_legacy_rate_reads_back_as_its_length():
    k, a = 3, 0.7
    ref = sp.reference_from_rate(k, a)
    assert ref == (k, pytest.approx(1 + a + a * a + a**3))
    assert sp.reference_from_rate(0, a) is None


@pytest.mark.parametrize("method", ["mtp", "eagle3", "dspark", "draft_model", "ngram"])
def test_entropy_is_off_by_default_and_orders_acceptance(method):
    acc = ga.resolve(method)
    base = sp.acceptance_length(sp.acceptance_profile(acc, 5, method=method))
    assert base == pytest.approx(acc.at(5), abs=1e-6)
    al = {
        e: sp.acceptance_length(sp.acceptance_profile(acc, 5, method=method, entropy=e))
        for e in ("low", "mixed", "high")
    }
    assert al["low"] > al["mixed"] > al["high"] > 1.0
    assert al["high"] < base


def test_entropy_axis():
    assert sp.resolve_entropy(None) is None
    assert sp.resolve_entropy("coding") is None
    assert sp.resolve_entropy("low_entropy") == -1.0
    assert sp.resolve_entropy("HIGH") == 1.0
    assert sp.resolve_entropy("0.5") == 0.5
    assert sp.resolve_entropy(4) == 1.0
    assert sp.entropy_multiplier("mtp", None) == 1.0
    mixed = sp.entropy_multiplier("mtp", 0.0)
    high = sp.entropy_multiplier("mtp", 1.0)
    half = sp.entropy_multiplier("mtp", 0.5)
    assert high < half < mixed
    assert half == pytest.approx((mixed * high) ** 0.5)


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
    assert sp.drafter_for("mtp", target_vocab=v, mtp_layers=2).layers == 2
    e3 = sp.drafter_for("EAGLE3", target_vocab=v)
    assert not e3.moe and e3.layers == 1 and e3.vocab_fraction == pytest.approx(32000 / v)
    ds = sp.drafter_for("dspark", target_vocab=v)
    assert ds.parallel and ds.layers == 3 and ds.window == 128 and ds.serial_head_rank == 256
    assert sp.drafter_for("dspark", target_vocab=v, layers=5).layers == 5
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

    for _name in (
        "_drafter",
        "_drafter_ms",
        "_drafter_failed",
        "_draft_overhead_ms",
        "_flat_draft_factor",
        "_spec_method",
        "_spec_k",
        "_spec_acceptance",
        "_spec_reference",
        "_spec_acceptance_source",
        "_spec_rates",
        "_spec_tokens_per_step",
        "_spec_extras",
        "_best_spec_k",
        "speculative_schedule",
    ):
        locals()[_name] = getattr(InferencePerformanceProjector, _name)
    del _name

    def __init__(self, method, *, k=3, accept=0.0, kmax=0, conc=1, model=None, **req):
        fields = dict(
            speculative_method=method,
            speculative_num_tokens=k,
            speculative_acceptance_rate=accept,
            speculative_acceptance_length=0.0,
            speculative_acceptance_curve=None,
            speculative_thinking="on",
            speculative_entropy=None,
            speculative_max_num_tokens=kmax,
            speculative_draft_layers=0,
            speculative_draft_vocab=0,
            speculative_draft_window=None,
            speculative_draft_cost_model="analytic",
            speculative_draft_cost_factor=0.0,
            serving_engine=None,
            input_seq_len=8192,
            output_seq_len=1024,
            resolved_max_concurrency=lambda: conc,
        )
        fields.update(req)
        self.cfg = SimpleNamespace(
            request_config=SimpleNamespace(**fields),
            model_config=model
            or SimpleNamespace(
                num_layers=_LAYERS, padded_vocab_size=129280, mtp_num_layers=1, hidden_size=7168
            ),
        )
        self._measured_mode = False
        self._bench_spec_k = 0
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
        return ft.total_ms + self._draft_overhead_ms(
            ft.total_ms / q_len, batch=batch, kv_len=kv_len, spec_k=q_len - 1
        )

    def _measured_decode_step_ms(self, batch, kv_len):
        return 2.0 * self._analytic_decode_step_ms(batch, kv_len, 1)


def test_acceptance_is_read_from_the_default_curve_without_a_measurement():
    p = _Drafting("mtp", k=3)
    assert p._spec_tokens_per_step() == pytest.approx(ga.default_curve("mtp").points()[0][3])
    assert p._spec_acceptance_source().startswith("default:mtp.yaml")


def test_stated_rate_keeps_the_legacy_tokens_per_step():
    p = _Drafting("mtp", k=3, accept=0.7)
    assert p._spec_tokens_per_step() == pytest.approx(1 + 0.7 + 0.49 + 0.343)
    assert p._spec_acceptance_source().startswith("stated AL")


def test_stated_length_wins_over_the_rate():
    p = _Drafting("mtp", k=3, accept=0.7, speculative_acceptance_length=2.2)
    assert p._spec_tokens_per_step() == pytest.approx(2.2)


def test_no_speculation_commits_one_token():
    p = _Drafting("mtp", k=0)
    assert p._spec_method() is None
    assert p._spec_tokens_per_step() == 1.0
    assert p.speculative_schedule() is None
    assert p._draft_overhead_ms(5.0, batch=8, kv_len=1024) == 0.0


def test_method_defaults_to_the_models_own_drafter():
    with_head = SimpleNamespace(
        num_layers=_LAYERS, padded_vocab_size=129280, mtp_num_layers=1, hidden_size=7168
    )
    assert _Drafting(None, model=with_head)._spec_method() == "mtp"
    no_head = SimpleNamespace(
        num_layers=_LAYERS, padded_vocab_size=129280, mtp_num_layers=0, hidden_size=7168
    )
    assert _Drafting(None, model=no_head)._spec_method() == "eagle3"
    declared = SimpleNamespace(
        num_layers=_LAYERS,
        padded_vocab_size=129280,
        mtp_num_layers=1,
        hidden_size=7168,
        speculative=SimpleNamespace(default="dspark"),
    )
    assert _Drafting(None, model=declared)._spec_method() == "dspark"


def test_drafter_layers_come_from_the_model_then_the_request():
    model = SimpleNamespace(
        num_layers=_LAYERS,
        padded_vocab_size=129280,
        mtp_num_layers=2,
        hidden_size=7168,
        speculative=SimpleNamespace(dspark=SimpleNamespace(layers=4)),
    )
    assert _Drafting("mtp", model=model)._drafter().layers == 2
    assert _Drafting("dspark", model=model)._drafter().layers == 4
    assert _Drafting("dspark", model=model, speculative_draft_layers=1)._drafter().layers == 1


def test_autoregressive_draft_grows_with_depth_block_draft_does_not():
    mtp = _Drafting("mtp")
    ds = _Drafting("dspark")
    m3, m7 = mtp._drafter_ms(8, 8192, 3), mtp._drafter_ms(8, 8192, 7)
    d3, d7 = ds._drafter_ms(8, 8192, 3), ds._drafter_ms(8, 8192, 7)
    assert m7 > 1.8 * m3
    assert 1.0 < d7 / d3 < m7 / m3
    assert _Drafting("ngram")._drafter_ms(8, 8192, 3) == 0.0
    assert _Drafting("ngram")._draft_overhead_ms(5.0, batch=8, kv_len=8192) == 0.0


def test_draft_is_analytic_by_default():
    p = _Drafting("mtp")
    assert p._draft_overhead_ms(5.0, batch=8, kv_len=8192) == pytest.approx(
        p._drafter_ms(8, 8192, 3)
    )
    assert p._spec_extras(8, 8192)["speculative_draft_cost"].startswith("analytic")


def test_flat_cost_is_the_fallback():
    flat = _Drafting("mtp", speculative_draft_cost_model="flat")
    assert flat._drafter() is None
    assert flat._draft_overhead_ms(5.0, batch=8, kv_len=8192) == pytest.approx(
        sp.FLAT_DRAFT_COST["mtp"] * 3 * 5.0
    )
    stated = _Drafting(
        "mtp", speculative_draft_cost_model="flat", speculative_draft_cost_factor=0.2
    )
    assert stated._draft_overhead_ms(5.0, batch=8, kv_len=8192) == pytest.approx(0.2 * 3 * 5.0)
    # A draft model has no published depth: priced flat until one is stated.
    unsized = _Drafting("draft_model")
    assert unsized._drafter() is None
    assert unsized._draft_overhead_ms(5.0, batch=8, kv_len=8192) == pytest.approx(
        sp.FLAT_DRAFT_COST["draft_model"] * 3 * 5.0
    )
    assert "flat" in unsized._spec_extras(8, 8192)["speculative_draft_cost"]
    # An unknown method keeps the model's acceptance and prices flat.
    unknown = _Drafting("suffix")
    assert unknown._spec_method() == "mtp"
    assert unknown._drafter() is None


def test_flat_cost_covers_a_simulator_that_cannot_price_the_drafter():
    p = _Drafting("mtp")

    def broken(*a, **k):
        raise RuntimeError("no kernel profile")

    p._drafter_ms = broken
    assert p._draft_overhead_ms(5.0, batch=8, kv_len=8192) == pytest.approx(
        sp.FLAT_DRAFT_COST["mtp"] * 3 * 5.0
    )
    assert p._drafter() is None


def test_benchmark_mode_scales_the_drafter_to_the_anchor():
    p = _Drafting("mtp")
    sim = p._draft_overhead_ms(1.0, batch=8, kv_len=8192)
    p._measured_mode = True
    one = p._analytic_decode_step_ms(8, 8192, 1)
    measured = 2.0 * one
    assert p._draft_overhead_ms(measured, batch=8, kv_len=8192) == pytest.approx(2.0 * sim)
    # A speculating anchor already ran its draft.
    p._bench_spec_k = 3
    assert p._drafter() is None
    assert p._draft_overhead_ms(measured, batch=8, kv_len=8192) == 0.0
    assert p._spec_tokens_per_step() > 1.0


def test_best_depth_falls_as_the_batch_grows():
    small = _Drafting("mtp", k=3, kmax=7, conc=1)
    large = _Drafting("mtp", k=3, kmax=7, conc=512)
    assert small._spec_k() > large._spec_k()
    k, rates = small.speculative_schedule()
    assert k == small._spec_k() and len(rates) == k
