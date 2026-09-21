###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A model that selects on every layer still has to pay for choosing.

DeepSeek-V4 records a per-layer compression schedule and is costed from it.
GLM-5.2 and MiniMax-M3 do not: they keep one window on every layer and differ
only in how wide the indexer is and how many layers compute one. Having no
schedule, both fell through to a top-k floored at 0.15, which on the agentic
corpus -- served between 136k and 365k tokens -- is a constant: nine to
twenty-seven times the selection it names, and nothing at all for the
indexing it stands in for.

The two terms are the point. Selection reads ``topk`` entries however long the
prompt is, so its share falls away like ``topk / context``. Indexing scores
every token in the pool on the layers that carry an indexer, so it grows with
context exactly as dense attention does and its share is flat. A single
multiplicative floor can be either of those and is neither.

These assert the shape -- that the total falls with context, that what it
falls towards is the indexing term rather than zero, and that the two models
are separated by geometry the floor could not see -- rather than exact
milliseconds.
"""

from __future__ import annotations

import pytest

from infera.projection.core.projection.training_config import (
    _SPARSE_ATTENTION_FLOOR,
    sparse_indexer_decode_overhead,
    uniform_sparse_attention_scale,
)


class _GLM:
    """Only the fields the scale reads, at GLM-5.2's published values."""

    def __init__(self, **over):
        self.num_layers = 78
        self.num_attention_heads = 64
        self.multi_latent_attention = True
        self.kv_lora_rank = 512
        self.qk_pos_emb_head_dim = 64
        self.sparse_index_head_dim = 128
        self.sparse_index_n_heads = 32
        # IndexShare: 21 of the 78 layers compute an index, the rest reuse it.
        self.sparse_index_layers = 21
        self.__dict__.update(over)


class _M3:
    """MiniMax-M3: GQA rather than MLA, and a four-head indexer."""

    def __init__(self, **over):
        self.num_layers = 60
        self.num_attention_heads = 64
        self.multi_latent_attention = False
        self.kv_channels = 128
        self.sparse_index_head_dim = 128
        self.sparse_index_n_heads = 4
        self.sparse_index_layers = 57
        self.__dict__.update(over)


def test_selection_falls_away_as_the_prompt_grows():
    """Reading 2048 of 8k is a quarter of the work; 2048 of 365k is not."""
    near = uniform_sparse_attention_scale(_GLM(), 8192, 2048)
    far = uniform_sparse_attention_scale(_GLM(), 364672, 2048)
    assert far < near
    # The floor it replaces cannot do this: past 14k it is the same at both.
    assert max(_SPARSE_ATTENTION_FLOOR, 2048 / 32768) == _SPARSE_ATTENTION_FLOOR
    assert max(_SPARSE_ATTENTION_FLOOR, 2048 / 364672) == _SPARSE_ATTENTION_FLOOR


def test_what_it_falls_towards_is_the_indexer_not_zero():
    """Selection vanishes at long context; the cost that remains is indexing."""
    scale = uniform_sparse_attention_scale(_GLM(), 4_000_000, 2048)
    assert scale > 0.0
    # Far past the window, selection is negligible and what is left is the
    # indexer's own share of a layer -- flat, because it grows with context
    # exactly as the dense attention it is measured against does.
    wider = uniform_sparse_attention_scale(_GLM(), 16_000_000, 2048)
    assert wider == pytest.approx(scale, rel=0.05)


def test_the_floor_over_reads_the_corpus_this_model_serves():
    """At the contexts GLM-5.2 was served, the floor is the wrong constant."""
    for ctx in (131072, 262144, 364672):
        priced = uniform_sparse_attention_scale(_GLM(), ctx, 2048)
        assert priced < _SPARSE_ATTENTION_FLOOR
    # And the gap widens with context, which is what a constant cannot track.
    near = _SPARSE_ATTENTION_FLOOR / uniform_sparse_attention_scale(_GLM(), 131072, 2048)
    far = _SPARSE_ATTENTION_FLOOR / uniform_sparse_attention_scale(_GLM(), 364672, 2048)
    assert far > near


def test_indexer_width_separates_two_models_the_floor_could_not():
    """Thirty-two heads against four is a difference; 0.15 against 0.15 is not."""
    glm = uniform_sparse_attention_scale(_GLM(), 364672, 2048)
    m3 = uniform_sparse_attention_scale(_M3(), 364672, 2048)
    assert glm != pytest.approx(m3, rel=0.05)


def test_index_share_is_cheaper_than_indexing_every_layer():
    """Reusing one layer's index across the next three is 21 layers, not 78."""
    shared = uniform_sparse_attention_scale(_GLM(), 131072, 2048)
    every = uniform_sparse_attention_scale(_GLM(sparse_index_layers=78), 131072, 2048)
    assert shared < every


def test_a_model_with_no_indexer_geometry_is_left_alone():
    """Declaring a window but no indexer is not enough to be priced here."""
    assert uniform_sparse_attention_scale(_GLM(sparse_index_n_heads=0), 131072, 2048) is None
    assert uniform_sparse_attention_scale(_GLM(sparse_index_head_dim=0), 131072, 2048) is None
    assert uniform_sparse_attention_scale(_GLM(), 131072, 0) is None


def test_a_fused_selection_leaves_the_calibrated_decode_step_alone():
    """Decode's dense charge is measured; the fused path must not disturb it."""
    assert sparse_indexer_decode_overhead(_GLM(), 2048, 1.0) == 0.0


def test_an_unfused_selection_costs_decode_a_bounded_amount():
    """A slower top-k kernel is a percentage of the step, not a multiple."""
    over = sparse_indexer_decode_overhead(_GLM(), 2048, 3.3)
    assert 0.0 < over < 0.10
    assert sparse_indexer_decode_overhead(_GLM(), 2048, 8.0) > over
