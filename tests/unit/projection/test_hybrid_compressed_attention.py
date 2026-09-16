###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A layer that compresses its KV should not be billed for context it never reads.

DeepSeek-V4 records a per-layer schedule saying which attention branch each
layer runs: a local window only, a heavily-compressed pool with full visibility
over it, or a compressed pool that an indexer selects a fixed top-k from. The
checkpoint ships it, and ``deepseek_v4_base.yaml`` ships the geometry that sizes
it, but nothing read either -- the whole 61-layer stack was priced from one
top-k scale floored at 0.15.

A floor cannot say what that schedule says. Attention under selection stops
growing once the pool exceeds the top-k, so what keeps growing with context is
the indexer scoring the pool, which is 9% of a CSA layer at 8k and 62% at 130k.
Flooring the scale instead makes the non-shrinking part grow like dense
attention over the whole prompt: at 130k the schedule costs 0.0157 of dense
against the floor's 0.15, a 9.6x over-read, and the resulting prefill chunk was
1,905 ms where the model's own calibrated anchor says 311 ms.

These assert the shape of the schedule rather than exact milliseconds: that the
cost falls as context grows, that the indexer is the term which does not, and
that a schedule this cost model has not been taught is refused instead of
guessed at.
"""

from __future__ import annotations

import pytest

from infera.projection.core.projection.training_config import (
    _SPARSE_ATTENTION_FLOOR,
    hybrid_attention_scale,
    parse_compress_ratios,
)

from .conftest import project_spec

# DeepSeek-V4-Pro, verbatim from its yaml: 2 HCA to bootstrap, then CSA/HCA
# interleaved, then a trailing CSA and one dense windowed layer. That is 62
# entries against 61 declared layers, because the last one is the MTP head.
PRO_SCHEDULE = "[128, 128" + ", 4, 128" * 29 + ", 4, 0]"


class _Model:
    """Only the fields the scale reads, at DeepSeek-V4-Pro's values."""

    def __init__(self, **over):
        self.compress_ratios = PRO_SCHEDULE
        self.num_layers = 61
        self.mtp_num_layers = 1
        self.num_attention_heads = 128
        self.kv_channels = 512
        self.qk_pos_emb_head_dim = 64
        self.index_n_heads = 64
        self.index_head_dim = 128
        self.index_topk = 1024
        self.attn_sliding_window = 128
        for k, v in over.items():
            setattr(self, k, v)


def _only(schedule: list, **over) -> _Model:
    """A model whose whole stack is ``schedule``, for isolating one branch."""
    return _Model(compress_ratios=schedule, num_layers=len(schedule), mtp_num_layers=0, **over)


def test_the_schedule_the_checkpoint_ships_is_read_as_written():
    sched = parse_compress_ratios(_Model())
    assert len(sched) == 62, "61 layers plus the MTP head the schedule also names"
    assert sched.count(128) == 31
    assert sched.count(4) == 30
    assert sched.count(0) == 1


def test_the_list_form_and_the_string_form_agree():
    as_list = parse_compress_ratios(_Model(compress_ratios=[128, 4, 0]))
    as_text = parse_compress_ratios(_Model(compress_ratios="[128, 4, 0]"))
    assert as_list == as_text == [128, 4, 0]


def test_the_speculative_head_is_not_averaged_into_the_stack():
    """The schedule's last entry is the MTP head, which this forward doesn't run.

    It is also the one dense windowed layer in V4-Pro, and the cheapest at long
    context, so averaging it in would quietly make the stack look cheaper than
    the 61 layers that actually run.
    """
    with_head = hybrid_attention_scale(_Model(num_layers=62, mtp_num_layers=0), 130_000)
    stack_only = hybrid_attention_scale(_Model(), 130_000)
    assert stack_only > with_head, "dropping a near-free layer must raise the mean"
    # 61 of 62 layers, so the two are close; the point is which one is used.
    assert stack_only == pytest.approx(with_head, rel=0.05)


@pytest.mark.parametrize(
    "context,ceiling",
    [(8192, 0.10), (32768, 0.04), (130000, 0.02), (262144, 0.02)],
)
def test_a_compressed_stack_costs_less_of_dense_the_longer_the_prompt(context, ceiling):
    """The pools grow far slower than the prompt, so the fraction has to fall."""
    assert hybrid_attention_scale(_Model(), context) < ceiling


def test_the_long_context_cost_is_nothing_like_the_floor_it_replaces():
    """At 130k the floor charges 9.6x what the schedule does."""
    scale = hybrid_attention_scale(_Model(), 130_000)
    assert scale < _SPARSE_ATTENTION_FLOOR / 5, (
        f"schedule says {scale:.4f}; the floor charged {_SPARSE_ATTENTION_FLOOR}"
    )


def test_the_floor_was_set_where_it_barely_showed():
    """At 8k the two are close, which is why an 8k sweep never caught this."""
    assert hybrid_attention_scale(_Model(), 8192) > _SPARSE_ATTENTION_FLOOR / 3


def _indexer_share(ctx: int, schedule=None) -> float:
    build = _Model if schedule is None else (lambda **kw: _only(schedule, **kw))
    with_idx = hybrid_attention_scale(build(), ctx)
    without = hybrid_attention_scale(build(index_n_heads=0), ctx)
    return (with_idx - without) / with_idx


def test_the_indexer_is_the_term_that_keeps_growing_with_context():
    """Selected attention saturates; scoring the pool to select it does not.

    Measured on a CSA layer alone, because only 29 of the 61 layers carry an
    indexer at all and averaging over the HCA layers that don't would dilute
    the very thing being asserted.
    """
    short, long = _indexer_share(8192, [4]), _indexer_share(130_000, [4])
    assert short < 0.15, f"indexer is a small part of an 8k step, got {short:.2f}"
    assert long > 0.5, f"indexer dominates a 130k step, got {long:.2f}"
    assert long > 5 * short


def test_over_the_whole_stack_the_indexer_grows_from_a_rounding_error_to_half():
    """Diluted by the 31 HCA layers that have no indexer, but the trend holds."""
    assert _indexer_share(8192) < 0.12
    assert _indexer_share(262_144) > 0.45


def test_selected_attention_stops_growing_once_the_pool_exceeds_the_top_k():
    """Past the crossover a CSA layer reads ``top_k`` entries at any context.

    With the indexer removed the only context term left is the pool it would
    have scored, so the cost per layer goes flat and the fraction falls as
    exactly 1/context.
    """
    csa_only = _only([4] * 8, index_n_heads=0)
    # Crossover is at index_topk * compress_rate = 4096 tokens.
    a = hybrid_attention_scale(csa_only, 32768) * 32768
    b = hybrid_attention_scale(csa_only, 262144) * 262144
    assert a == pytest.approx(b, rel=0.01)


def test_a_schedule_this_cost_model_has_not_been_taught_is_refused():
    """An unknown branch means the caller keeps its old behaviour, not a guess."""
    assert hybrid_attention_scale(_only([128, 7, 0]), 130_000) is None


def test_a_model_without_a_schedule_is_left_alone():
    assert hybrid_attention_scale(_Model(compress_ratios=None), 130_000) is None
    assert hybrid_attention_scale(_Model(compress_ratios="[]"), 130_000) is None


def test_a_dense_model_is_never_costed_from_a_schedule():
    """gpt-oss declares no schedule, so nothing here can change its attention."""
    assert parse_compress_ratios(_Model(compress_ratios=None)) is None


def _ttft(input_len: int) -> float:
    return project_spec(
        model="deepseek_v4_pro",
        tp=8,
        ep=1,
        pp=1,
        attn_dp=8,
        input_len=input_len,
        output_len=900,
        concurrency=64,
        weight_dtype="fp4",
        kv_cache_dtype="fp8",
        prefix_cache_hit_rate=0.0,
    )["ttft_ms"]


def test_prefill_grows_with_the_prompt_and_not_with_the_square_of_it():
    """End to end, through the real yaml, at the config the fleet serves.

    TTFT is *supposed* to grow with the prompt -- the projections and the MLP
    are linear in tokens and dominate here -- so the thing to check is not that
    it stays flat but that nothing grows faster than the prompt does. Attention
    is the only term that could, and under the schedule its pools grow 128x and
    4x slower than the prompt, so the excess over linear should be slight.

    Measured over a 15.87x longer prompt: the schedule comes out 1.06x
    superlinear, and the floor it replaces came out 10.84x (183,810 ms of TTFT
    at 130k, against 3,471 ms here).
    """
    length_ratio = 130_000 / 8192
    short, long = _ttft(8192), _ttft(130_000)
    assert long > short, "a longer prompt still has to cost more"
    superlinear = (long / short) / length_ratio
    assert superlinear < 1.3, (
        f"TTFT grew {superlinear:.1f}x faster than the prompt did, which is the "
        f"dense-attention shape the schedule is supposed to remove"
    )
