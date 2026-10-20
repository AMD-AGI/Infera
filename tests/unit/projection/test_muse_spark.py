###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Muse Spark: the DFlash drafter, its window interleave and its mixed precision."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infera.projection.core.projection.inference_projection import golden_al
from infera.projection.core.projection.inference_projection.memory import _weight_bytes
from infera.projection.core.projection.inference_projection.speculative import (
    DFLASH_LAYERS,
    canonical_method,
    drafter_for,
)
from infera.projection.core.projection.training_config import InferenceRequestConfig

from .conftest import project_spec

_MUSE = dict(
    model="muse_spark",
    tp=4,
    ep=4,
    concurrency=1,
    input_len=8192,
    output_len=256,
    weight_dtype="bf16",
    moe_expert_dtype="mxfp4",
    linear_weight_dtype="bf16",
    kv_cache_dtype="fp8",
)


def test_dflash_is_a_block_drafter_of_its_own_shape():
    assert canonical_method("DFLASH") == canonical_method("dflash2") == "dflash"
    shape = {"ffn": 28672, "heads": 64, "kv_heads": 8, "head_dim": 128, "fused_layers": 5}
    d = drafter_for("dflash", target_vocab=202048, window=2048, shape=shape)
    assert (d.layers, d.parallel, d.moe, d.window) == (DFLASH_LAYERS, True, False, 2048)
    own = (d.ffn_hidden_size, d.heads, d.kv_heads, d.head_dim, d.fused_layers)
    assert own == (28672, 64, 8, 128, 5)
    # Without a stated shape it drafts with the target's own layers.
    assert drafter_for("dflash", target_vocab=202048).ffn_hidden_size == 0


def test_dflash_reads_its_own_golden_curve():
    assert golden_al.method_of_file("musespark_dflash.yaml") == "dflash"
    pts, _ = golden_al.default_curve("dflash").points("off")
    assert sorted(pts) == list(range(1, 9))
    assert pts[7] == pytest.approx(4.05)


def test_a_model_window_fraction_sits_between_request_and_interleave():
    req = InferenceRequestConfig()
    assert req.resolved_sliding_window_fraction(False, 46 / 62) == pytest.approx(46 / 62)
    assert req.resolved_sliding_window_fraction(True) == 0.5
    assert req.resolved_sliding_window_fraction(False) == 1.0
    req.sliding_window_layer_fraction = 0.25
    assert req.resolved_sliding_window_fraction(False, 46 / 62) == 0.25
    # 46 windowed layers read 2048 tokens, 16 full ones the whole context.
    want = round((46 * 2048 + 16 * 10000) / 62)
    got = InferenceRequestConfig().effective_attn_kv(10000, 2048, model_fraction=46 / 62)
    assert got == pytest.approx(want, abs=1)


def _cfg(**req):
    mc = SimpleNamespace(
        num_experts=4,
        moe_pattern=[1, 1, 1, 1],
        moe_ffn_hidden_size=8,
        ffn_hidden_size=16,
        swiglu=True,
        hidden_size=10,
        moe_unquantized_expert_layers=req.pop("unquantized", 0),
    )
    rc = SimpleNamespace(weight_dtype="bf16", moe_expert_dtype=None, linear_weight_dtype=None)
    for k, v in req.items():
        setattr(rc, k, v)
    return SimpleNamespace(model_config=mc, request_config=rc)


def test_weights_cost_their_own_widths():
    lm = SimpleNamespace(layers=[0, 1, 2, 3])
    experts = 4 * 3 * 10 * 8 * 4  # experts x projections x hidden x ffn x layers
    total = experts + 1000
    assert _weight_bytes(_cfg(), lm, total, 1) == 2 * total
    mixed = _weight_bytes(_cfg(moe_expert_dtype="mxfp4"), lm, total, 1)
    assert mixed == int(experts * 0.53125 + 1000 * 2)
    # Experts the checkpoint left unquantized stay at the linear width.
    kept = _weight_bytes(_cfg(moe_expert_dtype="mxfp4", unquantized=2), lm, total, 1)
    assert kept == int(experts / 2 * 0.53125 + (experts / 2 + 1000) * 2)


def test_muse_spark_fits_tp4_with_mxfp4_experts():
    r = project_spec(**_MUSE)
    weights = r["memory_gb"] - r["kv_cache_gb"]
    # ~451 GB of checkpoint over four GPUs, the BF16 parts and the two
    # unquantized expert layers included.
    assert 105.0 < weights < 120.0


def test_muse_spark_decode_reads_only_the_window():
    """SGLang windows Muse's decode, so long context grows only its 16 full layers."""
    long = dict(_MUSE, input_len=131072, output_len=64, concurrency=8)
    capped = project_spec(**long)
    full = project_spec(**long, model_overrides=["sink_window_decode_capped=false"])
    assert capped["decode_step_ms"] < full["decode_step_ms"]
    short = dict(_MUSE, input_len=1024, output_len=64, concurrency=8)
    a = project_spec(**short)["decode_step_ms"]
    b = project_spec(**short, model_overrides=["sink_window_decode_capped=false"])
    # Inside the window the cap changes nothing.
    assert a == pytest.approx(b["decode_step_ms"], rel=1e-6)


def test_dflash_drafts_from_its_own_layers():
    plain = project_spec(**_MUSE)
    spec = project_spec(**_MUSE, speculative_num_tokens=7)
    assert "charging the flat draft cost" not in spec["report"]
    # 4.05 tokens a verify at k=7, against a draft and a wider verify.
    assert 1.5 < plain["tpot_ms"] / spec["tpot_ms"] < 4.05
