###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import pytest

from infera.kv.types import (
    DEFAULT_TIER_CREDIT,
    AttentionGroup,
    AttentionRole,
    BlockKey,
    Execution,
    MmRun,
    OverlapBlocks,
    Tier,
)


def test_tier_round_trip() -> None:
    assert Tier("device") == Tier.DEVICE
    assert Tier("host") == Tier.HOST
    assert Tier("disk") == Tier.DISK
    assert Tier("fabric") == Tier.FABRIC


def test_attention_role_indexable_set() -> None:
    assert AttentionRole.INDEXABLE.is_indexable()
    assert AttentionRole.SLIDING.is_indexable()
    assert not AttentionRole.RECURRENT.is_indexable()
    assert not AttentionRole.ENCODER_ONLY.is_indexable()
    assert not AttentionRole.CROSS.is_indexable()
    assert not AttentionRole.UNKNOWN.is_indexable()


def test_attention_role_emits_events() -> None:
    # SLIDING emits events (we use them for option-C window attenuation);
    # RECURRENT does not.
    assert AttentionRole.INDEXABLE.emits_events_to_main_index()
    assert AttentionRole.SLIDING.emits_events_to_main_index()
    assert not AttentionRole.RECURRENT.emits_events_to_main_index()


def test_attention_group_sliding_requires_window() -> None:
    with pytest.raises(ValueError, match="window required"):
        AttentionGroup(family="gpt-oss", role=AttentionRole.SLIDING)


def test_attention_group_indexable_no_window_required() -> None:
    # Should succeed without window.
    g = AttentionGroup(family="qwen3", role=AttentionRole.INDEXABLE)
    assert g.window is None


def test_default_tier_credits_match_design() -> None:
    # Hard-coded — if these change, design docs in 13-tier-and-pool.md
    # and 04-routing.md must also change.
    assert DEFAULT_TIER_CREDIT[Tier.DEVICE] == 1.0
    assert DEFAULT_TIER_CREDIT[Tier.HOST] == 0.75
    assert DEFAULT_TIER_CREDIT[Tier.DISK] == 0.25
    assert DEFAULT_TIER_CREDIT[Tier.FABRIC] == 0.15


def test_overlap_blocks_total() -> None:
    o = OverlapBlocks(device=10, host=20, disk=30, fabric=5)
    assert o.total() == 65


def test_overlap_blocks_default_zero() -> None:
    o = OverlapBlocks()
    assert o.total() == 0


def test_mm_run_validates_ordering() -> None:
    with pytest.raises(ValueError, match="must be >"):
        MmRun(mm_hash=0xCAFE, start=100, end=100)
    with pytest.raises(ValueError, match="must be >"):
        MmRun(mm_hash=0xCAFE, start=100, end=99)


def test_mm_run_len() -> None:
    assert len(MmRun(mm_hash=0xCAFE, start=10, end=20)) == 10


def test_execution_constructors_set_kind() -> None:
    assert Execution.mixed("w1").kind == "mixed"
    assert Execution.disagg("p", "d").kind == "disagg"
    assert Execution.local("d").kind == "local"


def test_execution_predicates_mutually_exclusive() -> None:
    e = Execution.mixed("w1")
    assert e.is_mixed() and not e.is_disagg() and not e.is_local()
    e = Execution.disagg("p", "d")
    assert e.is_disagg() and not e.is_mixed() and not e.is_local()
    e = Execution.local("d")
    assert e.is_local() and not e.is_mixed() and not e.is_disagg()


def test_execution_disagg_fields() -> None:
    e = Execution.disagg("p1", "d1")
    assert e.p_worker_id == "p1"
    assert e.d_worker_id == "d1"
    assert e.worker_id is None


def test_execution_local_sets_both_worker_fields() -> None:
    # local uses the decode worker — populate both for downstream convenience.
    e = Execution.local("d1")
    assert e.worker_id == "d1"
    assert e.d_worker_id == "d1"


def test_block_key_immutable() -> None:
    bk = BlockKey(sequence_hash=1, block_hash=2, parent_sequence_hash=None)
    with pytest.raises((AttributeError, Exception)):
        bk.sequence_hash = 99  # type: ignore[misc]
