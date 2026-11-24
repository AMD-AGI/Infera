###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Pluggable inference op models: registry, coverage, integration, validator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from infera.projection.core.projection.op_models import coverage, integration, registry
from infera.projection.core.projection.op_models.base import (
    SLOT_ATTENTION,
    SLOT_ATTENTION_EXTRA,
    OpContext,
    OpCost,
    OpModel,
)
from infera.projection.core.projection.op_models.validate import Workload, validate_op_model

from .conftest import project_spec

_ROOT = Path(__file__).resolve().parents[3]
_LFM2 = _ROOT / "infera/projection/examples/workloads/lfm2_8B_A1B-BF16-serving.yaml"


def _config(num_layers=8, tp=1, **extra):
    mc = SimpleNamespace(
        hidden_size=1024,
        num_attention_heads=8,
        kv_channels=128,
        num_layers=num_layers,
        fp8=None,
        extra_args=dict(extra),
    )
    mp = SimpleNamespace(tensor_model_parallel_size=tp, context_model_parallel_size=1)
    return SimpleNamespace(model_config=mc, model_parallel_config=mp)


class _Toy(OpModel):
    """A column projection, a cache read in decode, and an elementwise stream; counted exactly."""

    name = "toy_op"
    slot = SLOT_ATTENTION
    claims = {"toy_variant": ("toy",)}
    consumes = ("toy_width",)
    references = ("unit test",)
    frac = 0.5

    def layer_fraction(self, model_config):
        return self.frac

    def cost(self, ctx: OpContext) -> OpCost:
        w, h = int(ctx.value("toy_width", 512)), ctx.mc.hidden_size
        proj = ctx.linear(w, h)
        nbytes = ctx.linear_bytes(w, h) + 3 * ctx.tokens * w * 2
        if ctx.is_decode:
            nbytes += ctx.batch_per_rank * ctx.kv_len * w * 2
        elem = ctx.stream_ms(nbytes)
        return OpCost(
            ms=proj + elem,
            flops=2.0 * ctx.tokens * w * h,
            bytes=nbytes,
            breakdown={"proj": proj, "elem": elem},
            kv_bytes_per_token=float(w * 2),
            state_bytes_per_seq=64.0,
        )

    def reference_flops(self, ctx):
        return 2.0 * ctx.tokens * int(ctx.value("toy_width", 512)) * ctx.mc.hidden_size


@pytest.fixture
def toy():
    registry.register_op_model(_Toy)
    yield _Toy
    registry.unregister_op_model(_Toy.name)


# ---------------------------------------------------------------- registry / coverage


def test_kill_switch_disables_active_ops(toy, monkeypatch):
    mc = _config(toy_variant="toy").model_config
    assert [o.name for o in registry.active_op_models(mc)] == ["toy_op"]
    monkeypatch.setenv("INFERASIM_OP_MODELS", "0")
    assert registry.active_op_models(mc) == []


def test_shipped_generated_ops_discover_cleanly():
    assert "lfm2_short_conv" in {c.name for c in registry.registered_op_models()}
    assert not {k: v for k, v in registry.discovery_errors().items() if "generated" in k}


def test_lfm2_conv_layers_are_a_gap_until_the_op_is_registered(monkeypatch):
    cfg = {"lfm_layer_types": ["conv", "full_attention", "conv", "conv"], "conv_L_cache": 3}
    assert coverage.detect_op_gaps(cfg, baseline={}).blocking == []
    monkeypatch.setenv("INFERASIM_OP_MODELS", "0")
    assert [g.op_id for g in coverage.detect_op_gaps(cfg, baseline={}).blocking] == ["lfm_conv"]


def test_built_in_linear_attention_is_not_a_gap():
    cfg = {"experimental_attention_variant": "gated_delta_net", "linear_attention_freq": 4}
    assert coverage.detect_op_gaps(cfg, baseline={}).blocking == []


def test_acknowledged_shared_expert_gate_is_covered():
    assert coverage.detect_op_gaps({"moe_shared_expert_gate": True}, baseline={}).blocking == []


def test_gap_markers_round_trip():
    rep = coverage.detect_op_gaps({"experimental_attention_variant": "brand_new_attn"}, baseline={})
    text = "noise\n" + "\n".join(g.marker_line() for g in rep.blocking) + "\nmore noise"
    (g,) = coverage.parse_gap_markers(text)
    assert g.op_id == "brand_new_attn" and "INFERASIM_OP_GAP" in rep.blocking[0].marker_line()


# ---------------------------------------------------------------- integration


def test_attention_blend_replaces_its_share_and_adds_extras(toy):
    class Extra(_Toy):
        name = "toy_extra"
        slot = SLOT_ATTENTION_EXTRA
        frac = 0.25

    registry.register_op_model(Extra)
    try:
        cfg = _config(toy_variant="toy")
        lom = integration.layer_op_models(cfg.model_config)
        ctx = OpContext(cfg, 4, 1, 2048, "decode")
        c = _Toy().cost(ctx).ms
        got = lom.attention_ms(ctx, 10.0, other_share=0.25)
        assert got == pytest.approx(0.25 * 10.0 + 0.5 * c + 0.25 * c)
    finally:
        registry.unregister_op_model(Extra.name)


def test_kv_cache_terms_swap_the_entry_on_the_ops_layers(toy):
    cfg = _config(toy_variant="toy", toy_width=256)
    ic = SimpleNamespace(model_config=cfg.model_config, as_training_config=lambda **kw: cfg)
    d_token, state = integration.kv_cache_op_terms(ic, per_token_per_layer=4096.0, layers_on_rank=8)
    assert d_token == pytest.approx(0.5 * 8 * (256 * 2 - 4096.0))
    assert state == pytest.approx(0.5 * 8 * 64.0)


def test_no_active_op_leaves_the_projection_alone():
    assert integration.layer_op_models(_config().model_config) is None
    ic = SimpleNamespace(model_config=_config().model_config)
    assert integration.kv_cache_op_terms(ic, 4096.0, 8) == (0.0, 0.0)


# ---------------------------------------------------------------- validator


def _wl(**extra):
    return Workload(_config(**extra), None, None, batch=4, input_len=1024, output_len=256)


def test_validator_accepts_a_sound_op():
    rep = validate_op_model(_Toy, _wl(toy_variant="toy"), gap_keys=["toy_variant", "toy_width"])
    assert rep.ok, rep.feedback()
    assert {c["phase"] for c in rep.cases} == {"prefill", "decode"}


def test_validator_rejects_uncovered_keys_and_an_absent_op():
    rep = validate_op_model(
        _Toy, _wl(toy_variant="toy"), gap_keys=["toy_variant", "toy_mystery_key"]
    )
    assert {c.name for c in rep.failures()} == {"gap_keys_covered"}
    assert "matches" in {c.name for c in validate_op_model(_Toy, _wl()).failures()}


def test_validator_rejects_a_decode_faster_than_its_cache_read():
    class NoCacheTime(_Toy):
        def cost(self, ctx):
            c = super().cost(ctx)
            if ctx.is_decode:
                c.ms, c.breakdown = c.breakdown["proj"], {}
            return c

    names = {c.name for c in validate_op_model(NoCacheTime, _wl(toy_variant="toy")).failures()}
    assert "roofline[decode_2x_ctx]" in names


def test_validator_rejects_cost_falling_with_context():
    class Shrinks(_Toy):
        def cost(self, ctx):
            c = super().cost(ctx)
            if ctx.is_decode:
                scale = 1000.0 / ctx.kv_len
                c.ms, c.breakdown = c.ms * scale, {}
                c.bytes = c.bytes * scale * 0.5
            return c

    names = {c.name for c in validate_op_model(Shrinks, _wl(toy_variant="toy")).failures()}
    assert "monotonic[decode_2x_ctx]" in names


def test_validator_cross_checks_closed_form_flops():
    class Miscounted(_Toy):
        def reference_flops(self, ctx):
            return 2 * super().reference_flops(ctx)

    names = {c.name for c in validate_op_model(Miscounted, _wl(toy_variant="toy")).failures()}
    assert "reference_flops[prefill]" in names


# ---------------------------------------------------------------- end to end


def test_lfm2_short_conv_drops_the_kv_of_conv_layers_and_speeds_up_decode(monkeypatch):
    spec = dict(
        model="lfm2_8B_A1B",
        tp=1,
        concurrency=256,
        input_len=1024,
        output_len=128,
        weight_dtype="bf16",
    )
    monkeypatch.setenv("INFERASIM_OP_MODELS", "0")
    off = project_spec(**spec)
    monkeypatch.setenv("INFERASIM_OP_MODELS", "1")
    on = project_spec(**spec)
    assert "lfm2_short_conv" in on["report"] and "INFERASIM_OP_GAP" in off["report"]
    # 18 of 24 layers are conv: they keep a 2-entry conv window, not a KV cache.
    assert on["kv_cache_gb"] == pytest.approx(off["kv_cache_gb"] * 6 / 24, rel=0.02)
    assert on["decode_step_ms"] < off["decode_step_ms"]
