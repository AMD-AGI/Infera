###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""How MoE tokens reach their experts decides which collective a layer pays.

Under ``a2a`` dispatch each token is sent to its experts' ranks and combined
back, so an EP==TP MoE layer pays an all-to-all and only the attention
all-reduce. SGLang's ``--ep-size`` without ``--moe-a2a-backend`` keeps attention
tensor-parallel: every rank already holds every token, runs its own experts, and
the TP all-reduce that follows sums their outputs. That layer pays both
all-reduces and no all-to-all.
"""

from __future__ import annotations

import pytest


def _model(monkeypatch, dispatch, tp=8, ep=8):
    from infera.projection.core.projection.inference_projection.collectives import (
        InferenceCollectiveModel,
    )
    from infera.projection.core.projection.training_config import (
        InferenceCollectiveConfig,
    )

    if dispatch is None:
        monkeypatch.delenv("INFERASIM_EP_DISPATCH", raising=False)
    else:
        monkeypatch.setenv("INFERASIM_EP_DISPATCH", dispatch)

    class _Model:
        hidden_size = 6144
        moe_router_topk = 8

    class _Parallel:
        tensor_model_parallel_size = tp
        pipeline_model_parallel_size = 1
        expert_model_parallel_size = ep
        context_model_parallel_size = 1
        attention_data_parallel_size = 1

    return InferenceCollectiveModel(_Model(), _Parallel(), InferenceCollectiveConfig(enabled=True))


def test_default_dispatch_is_the_all_to_all(monkeypatch):
    default = _model(monkeypatch, None).layer_comm_ms(16, 1, is_moe=True)
    a2a = _model(monkeypatch, "a2a").layer_comm_ms(16, 1, is_moe=True)
    assert default.ep_a2a_ms > 0.0
    assert default.ep_a2a_ms == pytest.approx(a2a.ep_a2a_ms)
    assert default.tp_allreduce_ms == pytest.approx(a2a.tp_allreduce_ms)


def test_allreduce_dispatch_pays_both_all_reduces_and_no_all_to_all(monkeypatch):
    a2a = _model(monkeypatch, "a2a").layer_comm_ms(16, 1, is_moe=True)
    ar = _model(monkeypatch, "allreduce").layer_comm_ms(16, 1, is_moe=True)
    assert ar.ep_a2a_ms == 0.0
    assert ar.tp_allreduce_ms == pytest.approx(2.0 * a2a.tp_allreduce_ms)


def test_dispatch_does_not_touch_dense_layers_or_ep1(monkeypatch):
    for dispatch in ("a2a", "allreduce"):
        dense = _model(monkeypatch, dispatch).layer_comm_ms(16, 1, is_moe=False)
        assert dense.ep_a2a_ms == 0.0
    ep1_a2a = _model(monkeypatch, "a2a", ep=1).layer_comm_ms(16, 1, is_moe=True)
    ep1_ar = _model(monkeypatch, "allreduce", ep=1).layer_comm_ms(16, 1, is_moe=True)
    assert ep1_a2a.tp_allreduce_ms == pytest.approx(ep1_ar.tp_allreduce_ms)
    assert ep1_a2a.ep_a2a_ms == ep1_ar.ep_a2a_ms == 0.0


def test_cli_accepts_the_dispatch_choices():
    pytest.importorskip("yaml")
    from infera.projection.cli import build_parser

    base = ["inference", "--config", "unused.yaml"]
    for choice in ("a2a", "allreduce"):
        args = build_parser().parse_args([*base, "--ep-dispatch", choice])
        assert args.ep_dispatch == choice
    assert build_parser().parse_args(base).ep_dispatch is None
    with pytest.raises(SystemExit):
        build_parser().parse_args([*base, "--ep-dispatch", "deepep"])
