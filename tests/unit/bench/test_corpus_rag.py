###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for the RAG-shaped corpus generator.

The whole point of this corpus is **shared bytes across sessions** —
N users hit the same long prefix with their own queries. The tests
pin that invariant + replay.py compatibility (user-led, no system
role mixing) so a future refactor doesn't silently drop the cache-
relevant property.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

_BENCH_DIR = Path(__file__).resolve().parents[3] / "bench" / "kvcache"
sys.path.insert(0, str(_BENCH_DIR))
mod = importlib.import_module("deterministic_l3.corpus_rag")
sys.path.remove(str(_BENCH_DIR))


# ----------------------------------------------------------------------
# Shared-prefix invariant
# ----------------------------------------------------------------------


def test_shared_prefix_bit_identical_across_sessions():
    """Every session's first message MUST start with the same bytes.
    This is the property kvd / L2 use to recognize cross-session
    cache hits. If a future refactor accidentally seeds the system
    prompt per-session, hit rate goes to 0 and the bench becomes
    meaningless."""
    corpus, _ = mod.build_corpus(
        num_sessions=4, system_prompt_tokens=500, query_tokens=50, seed=1337
    )
    first_contents = [s["messages"][0]["content"] for s in corpus["sessions"]]
    # All sessions share a common prefix of at least system_prompt_chars.
    shared_prefix = corpus["config"]["prefix_chars_per_tenant"]
    leaders = [c[:shared_prefix] for c in first_contents]
    assert all(leader == leaders[0] for leader in leaders[1:]), (
        "shared prefix must be byte-identical across all sessions"
    )


def test_query_unique_per_session():
    """The trailing per-session content must differ — otherwise the
    full request is duplicated and the bench doesn't actually
    exercise different user queries."""
    corpus, _ = mod.build_corpus(
        num_sessions=4, system_prompt_tokens=500, query_tokens=200, seed=1337
    )
    contents = [s["messages"][0]["content"] for s in corpus["sessions"]]
    # No two full contents identical.
    assert len(set(contents)) == len(contents), "queries must be unique per session"


def test_deterministic_across_runs():
    """Same inputs → byte-identical output. This is the property
    that lets us replay the SAME corpus against multiple
    configurations (sync vs async, L2 on vs off) and compare apples
    to apples."""
    c1, _ = mod.build_corpus(num_sessions=8, system_prompt_tokens=1000, query_tokens=100, seed=42)
    c2, _ = mod.build_corpus(num_sessions=8, system_prompt_tokens=1000, query_tokens=100, seed=42)
    assert c1 == c2


def test_seed_changes_shared_prefix():
    """Changing `--seed` changes the prefix bytes — operators can
    generate distinct prefixes for multi-tenant simulation."""
    c1, _ = mod.build_corpus(num_sessions=2, system_prompt_tokens=200, query_tokens=50, seed=1)
    c2, _ = mod.build_corpus(num_sessions=2, system_prompt_tokens=200, query_tokens=50, seed=2)
    assert (
        c1["sessions"][0]["messages"][0]["content"] != c2["sessions"][0]["messages"][0]["content"]
    )


def test_query_seeding_independent_of_num_sessions():
    """Generating 4 vs 8 sessions must give the first 4 sessions
    identical queries. Otherwise scaling the bench shifts every
    session's content and previous warm-up writes don't line up."""
    c1, _ = mod.build_corpus(num_sessions=4, system_prompt_tokens=200, query_tokens=50, seed=7)
    c2, _ = mod.build_corpus(num_sessions=8, system_prompt_tokens=200, query_tokens=50, seed=7)
    for i in range(4):
        assert c1["sessions"][i]["messages"] == c2["sessions"][i]["messages"]


# ----------------------------------------------------------------------
# replay.py compatibility
# ----------------------------------------------------------------------


def test_each_session_has_exactly_one_user_message():
    """`replay.py` walks `messages[:turn_idx*2+1]` per user turn.
    For RAG shape, each session is ONE request — one user message
    means n_user_turns=1 and replay sends the whole message as a
    single completion."""
    corpus, _ = mod.build_corpus(num_sessions=4, system_prompt_tokens=300, query_tokens=50, seed=0)
    for sess in corpus["sessions"]:
        msgs = sess["messages"]
        assert len(msgs) == 1, "RAG sessions are single-turn"
        assert msgs[0]["role"] == "user", "user-led for replay.py compatibility"


def test_request_count_matches_session_count():
    """Stats report the number of user turns vLLM will see, which is
    `replay.py`'s loop count. For RAG shape that's exactly
    num-sessions."""
    _, stats = mod.build_corpus(num_sessions=32, system_prompt_tokens=500, query_tokens=50, seed=0)
    assert stats["num_user_turns"] == 32
    assert stats["num_sessions"] == 32


# ----------------------------------------------------------------------
# Sizing
# ----------------------------------------------------------------------


def test_shared_prefix_size_grows_with_tokens_arg():
    """A 1000-token target should produce visibly more chars than a
    100-token target. Approximate, not strict — the word generator
    is rng-bounded."""
    small, _ = mod.build_corpus(num_sessions=1, system_prompt_tokens=100, query_tokens=10, seed=0)
    large, _ = mod.build_corpus(num_sessions=1, system_prompt_tokens=1000, query_tokens=10, seed=0)
    assert (
        large["config"]["prefix_chars_per_tenant"] > small["config"]["prefix_chars_per_tenant"] * 5
    )


def test_total_bytes_includes_shared_prefix_n_times():
    """Each session embeds the full shared prefix, so total_bytes
    scales linearly with `num_sessions`. This is the wire-level
    redundancy that L1 / L2 / kvd are meant to deduplicate."""
    c8, s8 = mod.build_corpus(num_sessions=8, system_prompt_tokens=500, query_tokens=50, seed=0)
    c16, s16 = mod.build_corpus(num_sessions=16, system_prompt_tokens=500, query_tokens=50, seed=0)
    # Roughly 2x for 2x sessions (queries are small enough to ignore).
    ratio = s16["total_content_bytes"] / s8["total_content_bytes"]
    assert 1.9 <= ratio <= 2.1, f"expected ~2x bytes scaling, got {ratio:.3f}"


# ----------------------------------------------------------------------
# JSON round-trip — important because replay.py reads via json.load
# ----------------------------------------------------------------------


def test_corpus_json_roundtrip(tmp_path):
    corpus, _ = mod.build_corpus(num_sessions=2, system_prompt_tokens=100, query_tokens=10, seed=0)
    p = tmp_path / "rag.json"
    p.write_text(json.dumps(corpus))
    loaded = json.loads(p.read_text())
    assert loaded == corpus


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------


def test_zero_sessions_returns_empty_corpus():
    corpus, stats = mod.build_corpus(
        num_sessions=0, system_prompt_tokens=100, query_tokens=10, seed=0
    )
    assert corpus["sessions"] == []
    assert stats["num_sessions"] == 0


def test_tiny_system_prompt_still_works():
    """`target=1` token should produce >0 chars; the floor in
    `_fixed_system_prompt` is `max(1, ...)`."""
    corpus, _ = mod.build_corpus(num_sessions=1, system_prompt_tokens=1, query_tokens=1, seed=0)
    content = corpus["sessions"][0]["messages"][0]["content"]
    assert len(content) > 0


# ----------------------------------------------------------------------
# Multi-tenant variant — the L2-friendly workload
# ----------------------------------------------------------------------


def test_multi_tenant_produces_distinct_prefixes():
    """`num_tenants=M` must produce M distinct prefix byte sequences,
    one per tenant. If they accidentally collapse, the bench reduces
    to single-tenant and L2 has no work to absorb."""
    corpus, stats = mod.build_corpus(
        num_sessions=8,
        system_prompt_tokens=300,
        query_tokens=50,
        seed=1337,
        num_tenants=4,
    )
    # Collect each session's prefix (everything before "---").
    by_tenant: dict[int, set[str]] = {}
    for sess in corpus["sessions"]:
        prefix = sess["messages"][0]["content"].split("---")[0]
        by_tenant.setdefault(sess["tenant_id"], set()).add(prefix)
    # 4 distinct tenants → 4 distinct prefix strings.
    distinct_prefixes = {next(iter(v)) for v in by_tenant.values()}
    assert len(distinct_prefixes) == 4
    assert stats["num_tenants"] == 4


def test_multi_tenant_round_robin_interleaves_tenants():
    """Sessions must be dispatched in round-robin tenant order:
    s0=t0, s1=t1, s2=t2, s3=t3, s4=t0, ... — so consecutive engine
    requests see distinct tenants, forcing L1 to evict the prior
    tenant's prefix."""
    corpus, _ = mod.build_corpus(
        num_sessions=8,
        system_prompt_tokens=200,
        query_tokens=20,
        seed=0,
        num_tenants=4,
    )
    tenant_ids = [s["tenant_id"] for s in corpus["sessions"]]
    assert tenant_ids == [0, 1, 2, 3, 0, 1, 2, 3]


def test_multi_tenant_same_tenant_sessions_share_prefix():
    """Within a tenant, the prefix must be byte-identical across
    that tenant's sessions — that's what makes the SECOND access
    by the same tenant a candidate for L2 / kvd hit (the prefix
    was evicted from L1 between tenant_0's session_0 and
    session_1, but kvd still has it)."""
    corpus, _ = mod.build_corpus(
        num_sessions=8,
        system_prompt_tokens=300,
        query_tokens=20,
        seed=0,
        num_tenants=4,
    )
    # Sessions 0 and 4 are both tenant 0 (round-robin).
    s0_prefix = corpus["sessions"][0]["messages"][0]["content"].split("---")[0]
    s4_prefix = corpus["sessions"][4]["messages"][0]["content"].split("---")[0]
    assert s0_prefix == s4_prefix


def test_multi_tenant_distinct_prefix_bytes_scales_with_tenants():
    """Stats must accurately report total distinct-prefix bytes —
    operators use this to size the workload against L1/L2 capacity.
    M tenants × per-tenant bytes ≈ distinct_prefix_bytes."""
    _, s1 = mod.build_corpus(
        num_sessions=8,
        system_prompt_tokens=500,
        query_tokens=20,
        seed=0,
        num_tenants=1,
    )
    _, s4 = mod.build_corpus(
        num_sessions=8,
        system_prompt_tokens=500,
        query_tokens=20,
        seed=0,
        num_tenants=4,
    )
    # 4× tenants → ~4× distinct prefix bytes (slight slack: tenant
    # tokens have a "tNN_" prefix one char longer than "sys" used in
    # single-tenant mode, so per-tenant prefix is marginally larger).
    ratio = s4["distinct_prefix_bytes"] / s1["distinct_prefix_bytes"]
    assert 3.8 <= ratio <= 4.6, f"expected ~4x scaling, got {ratio:.3f}"


def test_multi_tenant_rejects_more_tenants_than_sessions():
    with pytest.raises(ValueError):
        mod.build_corpus(
            num_sessions=2,
            system_prompt_tokens=100,
            query_tokens=10,
            seed=0,
            num_tenants=4,
        )


def test_multi_tenant_rejects_zero_tenants():
    with pytest.raises(ValueError):
        mod.build_corpus(
            num_sessions=2,
            system_prompt_tokens=100,
            query_tokens=10,
            seed=0,
            num_tenants=0,
        )


def test_multi_tenant_default_is_single_tenant():
    """The default num_tenants=1 must reproduce the original
    single-tenant corpus byte-for-byte — otherwise existing bench
    artifacts get invalidated silently."""
    c_default, _ = mod.build_corpus(
        num_sessions=4, system_prompt_tokens=200, query_tokens=50, seed=1337
    )
    c_explicit, _ = mod.build_corpus(
        num_sessions=4,
        system_prompt_tokens=200,
        query_tokens=50,
        seed=1337,
        num_tenants=1,
    )
    # Strip the new tenant_id field which is always 0 in single-tenant mode.
    for c in (c_default, c_explicit):
        for s in c["sessions"]:
            s.pop("tenant_id", None)
            s.pop("session_within_tenant", None)
    assert c_default["sessions"] == c_explicit["sessions"]


def test_query_seed_independent_of_shared_prefix_seed():
    """Bumping `system_prompt_tokens` must NOT change the per-session
    query bytes — the two RNG streams are independent (rag-system-
    prompt vs rag-query). This lets operators tune prefix size
    without re-burning kvd content for previously-cached queries."""
    c1, _ = mod.build_corpus(num_sessions=2, system_prompt_tokens=500, query_tokens=100, seed=0)
    c2, _ = mod.build_corpus(num_sessions=2, system_prompt_tokens=5000, query_tokens=100, seed=0)
    # Different prefix length, but query suffix at same session_id
    # must match. Easiest check: extract after the "---" delimiter.
    q1 = c1["sessions"][0]["messages"][0]["content"].split("---")[-1]
    q2 = c2["sessions"][0]["messages"][0]["content"].split("---")[-1]
    assert q1 == q2
