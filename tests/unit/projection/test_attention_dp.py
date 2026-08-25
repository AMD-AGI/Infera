###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Data-parallel attention: the axis MLA serving is actually deployed on.

Tensor parallelism shards a GQA model's KV cache because it shards KV heads.
MLA has no KV heads to shard -- it caches one compressed latent that every head
reads -- so tensor parallelism replicates it, and at TP=8 a DeepSeek-style model
stores the same cache eight times over. Data-parallel attention is what fixes
that: a rank owns a subset of the running requests and holds their whole cache,
so the replica stores each sequence once.

Without this axis the projector cannot answer the capacity question for the way
these models are really served, and would answer it eight times too pessimistically.
"""

from __future__ import annotations

import pytest

from .conftest import project_spec

MLA = {"model": "deepseek_v3", "tp": 8, "ep": 8, "concurrency": 64}


def test_tensor_parallelism_replicates_the_mla_cache_and_dp_attention_does_not():
    tp_only = project_spec(**MLA)
    dp_attn = project_spec(**MLA, attn_dp=8)

    # The same 64 sequences, stored once across the replica instead of once per rank.
    assert dp_attn["kv_cache_gb"] == pytest.approx(tp_only["kv_cache_gb"] / 8.0, rel=1e-6)


def test_the_cache_a_rank_holds_is_its_own_share_of_the_requests():
    """Sequences split across ranks, not sliced -- so the split is by request."""
    whole = project_spec(**MLA, attn_dp=8)
    half = project_spec(**{**MLA, "concurrency": 32}, attn_dp=8)

    assert whole["kv_cache_gb"] == pytest.approx(2.0 * half["kv_cache_gb"], rel=1e-6)


def test_capacity_is_the_question_the_axis_exists_to_answer():
    """Freed cache is concurrency: the long-context / agentic capacity number."""
    tp_only = project_spec(**MLA)
    dp_attn = project_spec(**MLA, attn_dp=8)

    assert dp_attn["max_concurrent_sequences"] > tp_only["max_concurrent_sequences"]


def test_a_gqa_model_gains_nothing_because_tp_already_sharded_its_heads():
    """The axis moves the split from heads to requests; for GQA that is a wash.

    Attention DP holds whole KV heads for 1/dp of the requests where TP held
    1/tp of the heads for all of them. At dp == tp those are the same bytes, and
    a model that claimed a win here would be double-counting the head shard.
    """
    gqa = {"model": "gpt_oss_120B", "tp": 8, "concurrency": 64}
    assert project_spec(**gqa, attn_dp=8)["kv_cache_gb"] == pytest.approx(
        project_spec(**gqa)["kv_cache_gb"], rel=1e-6
    )


def test_attention_that_is_not_tensor_parallel_has_nothing_to_all_reduce():
    """A rank's attention output is whole, so the per-layer AR count drops."""
    tp_only = project_spec(**MLA)
    dp_attn = project_spec(**MLA, attn_dp=8)

    assert dp_attn["comm_decode_tp_allreduce_ms"] < tp_only["comm_decode_tp_allreduce_ms"]


def test_the_capacity_is_bought_with_per_rank_batch_efficiency():
    """The trade the axis makes, which a memory-only model would hide.

    A rank runs attention over 1/dp of the batch at dp times the heads. The work
    is the same but the shape is narrower, and narrow GEMMs are worse GEMMs -- so
    the decode step gets slower even as the cache it can hold grows eightfold.
    A projector that reported only the capacity would be recommending the axis
    without its cost.
    """
    tp_only = project_spec(**MLA)
    dp_attn = project_spec(**MLA, attn_dp=8)

    assert dp_attn["decode_step_ms"] > tp_only["decode_step_ms"]
    # Small next to what it buys: the step pays single-digit percent for 8x cache.
    assert dp_attn["decode_step_ms"] < 1.3 * tp_only["decode_step_ms"]


def test_the_axis_subdivides_the_tensor_parallel_group_rather_than_adding_gpus():
    dp_attn = project_spec(**MLA, attn_dp=8)
    assert dp_attn["replica_gpus"] == project_spec(**MLA)["replica_gpus"]

    # And a split the group cannot be divided into is a configuration error, not
    # a silently rounded one.
    with pytest.raises(ValueError, match="must divide"):
        project_spec(**{**MLA, "tp": 4}, attn_dp=8)
