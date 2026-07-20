###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for infera/router/policy/kv_event_aware.py.

The policy minimises:
    cost(w) = overlap_weight * (request_blocks - hits(w)) + active_blocks(w)

`hits(w)` is the longest *prefix* of the request's chained-hash list
that's in the worker's cache_view (chains break on the first miss).
`active_blocks(w)` is a refcounted set of distinct in-flight block
hashes — K concurrent requests sharing one prefix add one to the
load term, not K, which is the bug a prior refactor fixed.

We use stubs for `KvEventClient` and `BlockHasher` so the tests are
pure unit tests on the policy logic.
"""

from __future__ import annotations

from infera.common.worker_pool import (
    DisaggMode,
    EngineType,
    WorkerInfo,
    WorkerStatus,
)
from infera.router.policy.kv_event_aware import KvEventAwarePolicy

# ----------------------------------------------------------------------
# Stubs
# ----------------------------------------------------------------------


class _StubKvClient:
    """In-memory stand-in for KvEventClient: takes a dict
    {worker_id: set-of-cached-hashes} at construction."""

    def __init__(self, views: dict[str, set[int]] | None = None) -> None:
        self._views = views or {}
        self.added: list[WorkerInfo] = []
        self.removed: list[str] = []

    def cache_view(self, worker_id: str, dp_rank: int | None = None) -> set[int]:
        return self._views.get(worker_id, set())

    def set_view(self, worker_id: str, view: set[int]) -> None:
        self._views[worker_id] = view

    def on_worker_added(self, w: WorkerInfo) -> None:
        self.added.append(w)

    def on_worker_removed(self, worker_id: str) -> None:
        self.removed.append(worker_id)

    async def aclose(self) -> None:
        pass


class _StubHasher:
    """Returns a pre-canned hash list for a given request, indifferent
    to block_size (tests pass the same hashes regardless)."""

    def __init__(self, hashes: list[int]) -> None:
        self._hashes = hashes

    def hash_for(self, body: dict, *, block_size: int, engine=None) -> list[int]:
        return list(self._hashes)


def _worker(worker_id: str, *, kv_block_size: int = 4) -> WorkerInfo:
    return WorkerInfo(
        worker_id=worker_id,
        url=f"http://{worker_id}",
        model_name="test/m",
        engine=EngineType.SGLANG,
        status=WorkerStatus.ACTIVE,
        disagg_mode=DisaggMode.MIXED,
        kv_events_endpoint=f"tcp://{worker_id}:5557",
        kv_block_size=kv_block_size,
    )


# ----------------------------------------------------------------------
# pick() — cache-locality term
# ----------------------------------------------------------------------


def test_pick_prefers_worker_with_full_prefix_cached():
    """3-block request; w1 has all 3, w2 has none → w1 picked."""
    hashes = [10, 20, 30]
    client = _StubKvClient({"w1": {10, 20, 30}, "w2": set()})
    policy = KvEventAwarePolicy(client, _StubHasher(hashes))  # type: ignore[arg-type]

    picked, blocks = policy.pick([_worker("w1"), _worker("w2")], {"model": "m"})
    assert picked.worker.worker_id == "w1"
    assert blocks == hashes


def test_pick_returns_empty_blocks_for_request_that_doesnt_hash():
    """Stateless / unknown request: hasher returns []; pick still returns
    a worker, but `blocks` is []."""
    client = _StubKvClient({"w1": set(), "w2": set()})
    policy = KvEventAwarePolicy(client, _StubHasher([]))  # type: ignore[arg-type]
    picked, blocks = policy.pick([_worker("w1"), _worker("w2")], {})
    assert picked.worker.worker_id in {"w1", "w2"}
    assert blocks == []


def test_pick_hits_count_is_longest_prefix_not_membership():
    """Chains break on first miss — w1 has blocks 0 and 2 but not 1;
    that's a prefix of length 1, not 2."""
    hashes = [10, 20, 30]
    client = _StubKvClient(
        {
            "w1": {10, 30},  # missing middle
            "w2": {10, 20},  # contiguous prefix len 2
        }
    )
    policy = KvEventAwarePolicy(client, _StubHasher(hashes))  # type: ignore[arg-type]

    picked, _ = policy.pick([_worker("w1"), _worker("w2")], {"model": "m"})
    assert picked.worker.worker_id == "w2"


def test_pick_tie_breaks_by_lower_active_blocks():
    """Both workers have equal cost — tie-break goes to the less loaded one."""
    hashes = [10, 20]
    client = _StubKvClient({"w1": {10, 20}, "w2": {10, 20}})
    policy = KvEventAwarePolicy(client, _StubHasher(hashes))  # type: ignore[arg-type]

    # Pre-load w1 with one active block so it's "busier".
    policy.on_request_started("w1", [99])

    picked, _ = policy.pick([_worker("w1"), _worker("w2")], {"model": "m"})
    assert picked.worker.worker_id == "w2"


def test_pick_falls_back_to_least_loaded_on_zero_overlap_workload():
    """Request blocks don't match any worker's cache — cost reduces to
    `overlap_weight * total + active_blocks`. With equal totals,
    `active` decides."""
    hashes = [10, 20, 30]
    client = _StubKvClient({"w1": set(), "w2": set()})
    policy = KvEventAwarePolicy(client, _StubHasher(hashes))  # type: ignore[arg-type]

    policy.on_request_started("w1", [1, 2, 3, 4, 5])  # 5 distinct in-flight blocks
    policy.on_request_started("w2", [1])

    picked, _ = policy.pick([_worker("w1"), _worker("w2")], {"model": "m"})
    assert picked.worker.worker_id == "w2"


def test_overlap_weight_high_value_overrides_load_penalty():
    """At overlap_weight=100, a 1-block reduction in misses is worth
    100 active-block points — enough to override a moderate load gap."""
    hashes = [10, 20]
    client = _StubKvClient({"w1": {10, 20}, "w2": set()})
    policy = KvEventAwarePolicy(client, _StubHasher(hashes), overlap_weight=100.0)  # type: ignore[arg-type]

    # Give w1 a heavy load but w2 a light one.
    for i in range(50):
        policy.on_request_started("w1", [i + 1000])

    picked, _ = policy.pick([_worker("w1"), _worker("w2")], {"model": "m"})
    assert picked.worker.worker_id == "w1"


# ----------------------------------------------------------------------
# active_blocks refcounting
# ----------------------------------------------------------------------


def test_on_request_started_finished_balanced_yields_zero_active():
    client = _StubKvClient({})
    policy = KvEventAwarePolicy(client, _StubHasher([]))  # type: ignore[arg-type]

    policy.on_request_started("w1", [10, 20, 30])
    policy.on_request_finished("w1", [10, 20, 30])

    # Empty dict left behind → len = 0.
    assert policy._active_block_refs["w1"] == {}


def test_two_requests_sharing_prefix_dedupe_active_blocks():
    """The whole point of refcounting: K shared-prefix requests
    contribute 1 to len(active), not K."""
    client = _StubKvClient({})
    policy = KvEventAwarePolicy(client, _StubHasher([]))  # type: ignore[arg-type]

    policy.on_request_started("w1", [10, 20, 30])
    policy.on_request_started("w1", [10, 20, 99])  # shares 2 blocks with above

    # len of distinct hashes = 4: {10, 20, 30, 99}
    assert len(policy._active_block_refs["w1"]) == 4


def test_finished_removes_one_refcount_per_block_not_the_whole_set():
    """Two requests on the same prefix; finishing one must NOT evict
    the shared blocks — the other request is still using them."""
    client = _StubKvClient({})
    policy = KvEventAwarePolicy(client, _StubHasher([]))  # type: ignore[arg-type]

    policy.on_request_started("w1", [10, 20])
    policy.on_request_started("w1", [10, 20])
    policy.on_request_finished("w1", [10, 20])

    # Refcounts went 0 → 1 → 2 → 1 each. Still present.
    assert policy._active_block_refs["w1"] == {10: 1, 20: 1}

    policy.on_request_finished("w1", [10, 20])
    assert policy._active_block_refs["w1"] == {}


def test_finished_with_unknown_blocks_is_safe():
    """Out-of-order or duplicate finished call: must not crash, must
    not push a refcount negative."""
    client = _StubKvClient({})
    policy = KvEventAwarePolicy(client, _StubHasher([]))  # type: ignore[arg-type]

    policy.on_request_started("w1", [10])
    # Finish reports an extra block that we never started.
    policy.on_request_finished("w1", [10, 999])
    assert 10 not in policy._active_block_refs["w1"]


def test_finished_for_worker_with_no_state_is_noop():
    client = _StubKvClient({})
    policy = KvEventAwarePolicy(client, _StubHasher([]))  # type: ignore[arg-type]
    policy.on_request_finished("w-never-seen", [10])  # must not raise


def test_started_finished_with_none_blocks_is_noop():
    """Pass None (stateless policy contract); refcount untouched."""
    client = _StubKvClient({})
    policy = KvEventAwarePolicy(client, _StubHasher([]))  # type: ignore[arg-type]

    policy.on_request_started("w1", None)
    policy.on_request_finished("w1", None)
    assert "w1" not in policy._active_block_refs


# ----------------------------------------------------------------------
# Worker lifecycle delegated to KvEventClient
# ----------------------------------------------------------------------


def test_on_worker_added_forwards_to_kv_client():
    client = _StubKvClient({})
    policy = KvEventAwarePolicy(client, _StubHasher([]))  # type: ignore[arg-type]
    w = _worker("w1")
    policy.on_worker_added(w)
    assert client.added == [w]


def test_on_worker_removed_clears_active_state_and_forwards():
    client = _StubKvClient({})
    policy = KvEventAwarePolicy(client, _StubHasher([]))  # type: ignore[arg-type]
    policy.on_request_started("w1", [10, 20])

    policy.on_worker_removed("w1")
    assert client.removed == ["w1"]
    assert "w1" not in policy._active_block_refs


# ----------------------------------------------------------------------
# Multi-block-size handling
# ----------------------------------------------------------------------


def test_pick_uses_per_block_size_hashes_when_workers_differ():
    """Real fleets can have mixed block sizes (ROCm AITER worker with
    page_size=1, classic CUDA worker with 16). The policy hashes the
    request once per distinct block size and uses the right list per
    worker — verified by passing two stub hashers via a small wrapper."""

    class _PerSizeHasher:
        def __init__(self, by_size: dict[int, list[int]]) -> None:
            self._by_size = by_size

        def hash_for(self, body: dict, *, block_size: int, engine=None) -> list[int]:
            return list(self._by_size.get(block_size, []))

    hasher = _PerSizeHasher({4: [10, 20], 16: [555]})
    client = _StubKvClient({"w1": {10, 20}, "w2": {555}})
    policy = KvEventAwarePolicy(client, hasher)  # type: ignore[arg-type]

    w_small = _worker("w1", kv_block_size=4)
    w_big = _worker("w2", kv_block_size=16)

    # Both fully hit. Both have 0 active. Same cost — picker breaks tie
    # deterministically (min over identical key returns first). Just
    # assert blocks returned matches picked worker's block_size.
    picked, blocks = policy.pick([w_small, w_big], {"model": "m"})
    if picked.worker.worker_id == "w1":
        assert blocks == [10, 20]
    else:
        assert blocks == [555]


def test_pick_does_not_crash_on_worker_without_kv_block_size():
    """Workers that didn't register kv_block_size must not crash the
    cost function.

    Edge note: a non-KV worker currently looks "cost 0" (zero misses,
    zero load) to the policy, so it can be preferred over a KV worker
    that has any miss. Operators mixing KV-aware and non-KV workers in
    one fleet should either enable kv-events on all of them or split
    the fleets. This test only locks in non-crash; it deliberately
    does NOT assert which worker is picked, because that ranking is
    a separate policy decision that may evolve.
    """
    hashes = [10, 20]
    client = _StubKvClient({"w1": set(), "w2": {10, 20}})
    policy = KvEventAwarePolicy(client, _StubHasher(hashes))  # type: ignore[arg-type]

    w1 = WorkerInfo(
        worker_id="w1",
        url="http://w1",
        model_name="m",
        engine=EngineType.SGLANG,
        status=WorkerStatus.ACTIVE,
        disagg_mode=DisaggMode.MIXED,
        kv_events_endpoint=None,
        kv_block_size=None,
    )
    w2 = _worker("w2", kv_block_size=4)

    picked, _ = policy.pick([w1, w2], {"model": "m"})
    assert picked.worker.worker_id in {"w1", "w2"}


# ----------------------------------------------------------------------
# PD-aware: role_hint applies the right overlap weight
# ----------------------------------------------------------------------


def test_pick_prefill_role_uses_prefill_weight():
    """role_hint='prefill' should use prefill_overlap_weight. With a high
    prefill weight, even a heavily-loaded worker is preferred when it has
    the full prefix cached.
    """
    hashes = [10, 20]
    # w1 has full prefix cached; w2 is empty.
    client = _StubKvClient({"w1": {10, 20}, "w2": set()})
    policy = KvEventAwarePolicy(
        client,  # type: ignore[arg-type]
        _StubHasher(hashes),
        overlap_weight=1.0,
        prefill_overlap_weight=50.0,  # very aggressive on cache
        decode_overlap_weight=1.0,
    )

    # Load w1 hard (50 active blocks) — with overlap_weight=1, w2 would win;
    # with overlap_weight=50, w1 still wins because the cache hit is huge.
    for i in range(50):
        policy.on_request_started("w1", [i + 1000])

    picked, _ = policy.pick([_worker("w1"), _worker("w2")], {"model": "m"}, role_hint="prefill")
    assert picked.worker.worker_id == "w1"


def test_pick_decode_role_uses_decode_weight():
    """role_hint='decode' should use decode_overlap_weight. With a low
    decode weight, load balance dominates even when one worker has the
    full prefix cached — exactly the behavior we want for decode workers
    (memory-bound, not compute-bound on prefill).
    """
    hashes = [10, 20]
    # w1 has full prefix; w2 has nothing.
    client = _StubKvClient({"w1": {10, 20}, "w2": set()})
    policy = KvEventAwarePolicy(
        client,  # type: ignore[arg-type]
        _StubHasher(hashes),
        overlap_weight=20.0,  # would normally heavily favour w1
        prefill_overlap_weight=20.0,
        decode_overlap_weight=1.0,  # for decode, treat cache as cheap
    )

    # Pre-load w1 with 5 active blocks so it looks busy.
    for i in range(5):
        policy.on_request_started("w1", [i + 1000])

    picked, _ = policy.pick([_worker("w1"), _worker("w2")], {"model": "m"}, role_hint="decode")
    # With decode_overlap_weight=1.0:
    #   cost(w1) = 1*(2-2) + 5 = 5
    #   cost(w2) = 1*(2-0) + 0 = 2
    # → w2 wins on load.
    assert picked.worker.worker_id == "w2"


def test_pick_role_hint_none_uses_default_overlap_weight():
    """Backward compat: pick() without role_hint must keep using the
    original overlap_weight, so mixed-pool routing is unchanged."""
    hashes = [10, 20]
    client = _StubKvClient({"w1": {10, 20}, "w2": set()})
    policy = KvEventAwarePolicy(
        client,  # type: ignore[arg-type]
        _StubHasher(hashes),
        overlap_weight=100.0,
        prefill_overlap_weight=0.0,  # purposefully crazy values
        decode_overlap_weight=0.0,
    )

    picked, _ = policy.pick([_worker("w1"), _worker("w2")], {"model": "m"})
    # No role_hint → uses overlap_weight=100. w1 has the cache; cost(w1)=0,
    # cost(w2)=200. w1 must win.
    assert picked.worker.worker_id == "w1"


def test_pick_prefill_and_decode_weights_default_to_overlap_weight():
    """If prefill_overlap_weight / decode_overlap_weight are None, both
    roles inherit overlap_weight — i.e. the PD knobs default to the
    legacy behavior."""
    hashes = [10, 20]
    client = _StubKvClient({"w1": {10, 20}, "w2": set()})
    policy = KvEventAwarePolicy(
        client,  # type: ignore[arg-type]
        _StubHasher(hashes),
        overlap_weight=50.0,
        # prefill_overlap_weight / decode_overlap_weight left unset
    )

    p_pick, _ = policy.pick([_worker("w1"), _worker("w2")], {"model": "m"}, role_hint="prefill")
    d_pick, _ = policy.pick([_worker("w1"), _worker("w2")], {"model": "m"}, role_hint="decode")
    # Both should behave like overlap_weight=50 → w1 wins for both.
    assert p_pick.worker.worker_id == "w1"
    assert d_pick.worker.worker_id == "w1"


def test_pick_returns_blocks_aligned_with_picked_worker():
    """Even with role_hint, the returned blocks list must match the
    picked worker's kv_block_size — same contract as before."""
    hashes = [10, 20]
    client = _StubKvClient({"w1": {10, 20}, "w2": set()})
    policy = KvEventAwarePolicy(
        client,  # type: ignore[arg-type]
        _StubHasher(hashes),
        overlap_weight=1.0,
        prefill_overlap_weight=50.0,
        decode_overlap_weight=1.0,
    )

    picked, blocks = policy.pick(
        [_worker("w1"), _worker("w2")], {"model": "m"}, role_hint="prefill"
    )
    assert picked.worker.worker_id == "w1"
    assert blocks == hashes


# ----------------------------------------------------------------------
# Retention-aware: cache_control hint amplifies/dampens overlap weight
# ----------------------------------------------------------------------


def _request_with_retention(retention_str: str, *, hashes_hint: list[int] | None = None) -> dict:
    """Build a request body that yields the requested retention via
    parse_cache_hints. Note: `explicit_none` produces explicit_hint_seen=True
    + retention=NONE; `none` (no hint at all) produces explicit_hint_seen=False."""
    body: dict = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    if retention_str == "long":
        body["system"] = [
            {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
        ]
    elif retention_str == "short":
        body["system"] = [
            {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}}
        ]
    elif retention_str == "explicit_none":
        # Client opted OUT of caching explicitly (OpenAI-style).
        body["prompt_cache_retention"] = "none"
    elif retention_str == "none":
        # No hint at all — implicit default, neutral amplifier.
        pass
    return body


def test_long_retention_amplifies_overlap_weight():
    """A request marked retention=long should treat cache locality as
    more important. With cache hit and small base weight, long should
    still pick the warm worker even when the load gap looks bad."""
    hashes = [10, 20]
    client = _StubKvClient({"w1": {10, 20}, "w2": set()})
    # Small base weight (1.0) — without retention amplification, a tiny
    # load advantage on w2 would flip the choice. With long-retention
    # amplification (2.0×), w1 wins decisively.
    policy = KvEventAwarePolicy(client, _StubHasher(hashes), overlap_weight=1.0)  # type: ignore[arg-type]
    # Give w1 just enough load that base weight alone wouldn't win:
    #   cost(w1) = 1*(2-2) + 3 = 3
    #   cost(w2) = 1*(2-0) + 0 = 2  → w2 wins on base
    for i in range(3):
        policy.on_request_started("w1", [i + 1000])

    body = _request_with_retention("long")
    picked, _ = policy.pick([_worker("w1"), _worker("w2")], body)
    # With amplifier=2.0:
    #   cost(w1) = 2*(2-2) + 3 = 3
    #   cost(w2) = 2*(2-0) + 0 = 4  → w1 wins
    assert picked.worker.worker_id == "w1"


def test_none_retention_lets_load_balance_win_on_partial_overlap():
    """With retention=none + partial overlap, dampened weight means
    load balance dominates."""
    hashes = [10, 20]
    # w1 has block 10 only (partial); w2 has nothing.
    client = _StubKvClient({"w1": {10}, "w2": set()})
    policy = KvEventAwarePolicy(client, _StubHasher(hashes), overlap_weight=10.0)  # type: ignore[arg-type]
    # Give w1 some load.
    policy.on_request_started("w1", [99, 88])  # active=2

    body = _request_with_retention("explicit_none")
    picked, _ = policy.pick([_worker("w1"), _worker("w2")], body)
    # With dampener=0.1:
    #   cost(w1) = 1*(2-1) + 2 = 3
    #   cost(w2) = 1*(2-0) + 0 = 2  → w2 wins (cache doesn't matter much)
    # Without dampener (weight=10):
    #   cost(w1) = 10*(2-1) + 2 = 12
    #   cost(w2) = 10*(2-0) + 0 = 20  → w1 wins
    assert picked.worker.worker_id == "w2"


def test_short_retention_uses_neutral_amplifier():
    """retention=short should behave like the original policy
    (amplifier=1.0)."""
    hashes = [10, 20]
    client = _StubKvClient({"w1": {10, 20}, "w2": set()})
    policy = KvEventAwarePolicy(client, _StubHasher(hashes), overlap_weight=5.0)  # type: ignore[arg-type]

    body = _request_with_retention("short")
    picked, _ = policy.pick([_worker("w1"), _worker("w2")], body)
    # Same as no hint at all.
    assert picked.worker.worker_id == "w1"


def test_implicit_none_uses_neutral_amplifier_not_dampener():
    """A request body with NO cache_control at all is parsed as
    retention=NONE + explicit_hint_seen=False. This is the most
    common case (legacy clients, no Anthropic SDK) — must behave
    like the original policy (amplifier=1.0), NOT dampen the
    overlap weight to ~0."""
    hashes = [10, 20]
    client = _StubKvClient({"w1": {10, 20}, "w2": set()})
    policy = KvEventAwarePolicy(client, _StubHasher(hashes), overlap_weight=100.0)  # type: ignore[arg-type]

    # Heavy base weight should still favour cache hit on implicit-NONE.
    body = _request_with_retention("none")  # no cache_control → implicit
    picked, _ = policy.pick([_worker("w1"), _worker("w2")], body)
    assert picked.worker.worker_id == "w1"  # full hit + neutral amplifier = w1 wins


# ----------------------------------------------------------------------
# Multimodal silent-corruption guard
#
# The router-side hasher is text-only. For vision/audio requests the
# placeholder token id is the same regardless of which image, so two
# requests with the same surrounding tokens but different images
# produce the same chain hash. If the policy trusted cache locality on
# such a request, it would route to a worker that has the OTHER
# image's KV cached and serve wrong KV — silent corruption.
#
# The fix: when has_multimodal_content=True, the policy forces
# overlap_weight=0 so cost() = active(w), pure load balance.
# ----------------------------------------------------------------------


def _mm_request(text: str = "describe this image") -> dict:
    return {
        "model": "test/m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgo...",
                        },
                    },
                ],
            }
        ],
    }


def test_mm_request_forces_pure_load_balance_even_with_full_cache_hit():
    """A vision request with the same text as a previously-cached prompt
    must NOT route to the warm worker — that worker might have cached
    a DIFFERENT image's KV under the same hash."""
    hashes = [10, 20]
    # w1 has the entire chain cached (would normally win decisively).
    client = _StubKvClient({"w1": {10, 20}, "w2": set()})
    policy = KvEventAwarePolicy(client, _StubHasher(hashes), overlap_weight=100.0)  # type: ignore[arg-type]

    # w1 has slightly more load — without MM guard, cache locality
    # would dominate (100*(2-2)+1 vs 100*(2-0)+0 → w1 wins on hit).
    policy.on_request_started("w1", [999])  # w1 active=1

    body = _mm_request()
    picked, _ = policy.pick([_worker("w1"), _worker("w2")], body)
    # With MM guard: w_overlap=0, so cost(w1)=1, cost(w2)=0 → w2 wins
    # on load. The "cache hit" on w1 is treated as untrustworthy.
    assert picked.worker.worker_id == "w2"


def test_mm_request_still_returns_block_hashes_for_refcounting():
    """MM guard zeroes the WEIGHT, but the picked_blocks list must
    still come back — `on_request_started/finished` use it to
    refcount active blocks. Routing decode requests later relies on
    this being non-empty."""
    hashes = [10, 20, 30]
    client = _StubKvClient({"w1": set(), "w2": set()})
    policy = KvEventAwarePolicy(client, _StubHasher(hashes), overlap_weight=10.0)  # type: ignore[arg-type]

    body = _mm_request()
    _, blocks = policy.pick([_worker("w1"), _worker("w2")], body)
    assert blocks == [10, 20, 30]


def test_mm_request_does_not_affect_text_only_request_routing():
    """The MM guard is scoped to ONE request — a subsequent text-only
    request still gets full cache-locality treatment."""
    hashes = [10, 20]
    client = _StubKvClient({"w1": {10, 20}, "w2": set()})
    policy = KvEventAwarePolicy(client, _StubHasher(hashes), overlap_weight=10.0)  # type: ignore[arg-type]
    # Pre-load w1 so the load-balance term actually distinguishes the
    # two workers when MM forces w_overlap=0.
    policy.on_request_started("w1", [777])  # w1 active=1, w2 active=0

    # First: MM request goes to w2 (load balance, since w_overlap=0
    # collapses cost to active(w): w1=1, w2=0 → w2 wins).
    mm_body = _mm_request()
    picked_mm, _ = policy.pick([_worker("w1"), _worker("w2")], mm_body)
    assert picked_mm.worker.worker_id == "w2"

    # Second: text-only request gets cache-locality treatment.
    #   cost(w1) = 10*(2-2) + 1 = 1
    #   cost(w2) = 10*(2-0) + 0 = 20 → w1 wins on cache hit
    text_body = {"model": "test/m", "messages": [{"role": "user", "content": "hello"}]}
    picked_text, _ = policy.pick([_worker("w1"), _worker("w2")], text_body)
    assert picked_text.worker.worker_id == "w1"  # cache hit dominates


def test_mm_request_with_long_retention_still_skips_cache_locality():
    """Even if the client tagged the request with cache_control=long,
    the MM safety guard takes precedence — silent corruption beats
    "honoring the client's hint" every time."""
    hashes = [10, 20]
    client = _StubKvClient({"w1": {10, 20}, "w2": set()})
    policy = KvEventAwarePolicy(client, _StubHasher(hashes), overlap_weight=10.0)  # type: ignore[arg-type]
    policy.on_request_started("w1", [999])  # w1 active=1

    body = _mm_request()
    # Add a long-retention cache_control to the image-bearing message.
    body["system"] = [
        {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    ]
    picked, _ = policy.pick([_worker("w1"), _worker("w2")], body)
    # MM guard wins: w_overlap=0 (NOT 20× from long amplifier), so
    # cost(w1)=1, cost(w2)=0 → w2 wins on load.
    assert picked.worker.worker_id == "w2"


def test_mm_request_metric_increment():
    """The cache_locality_skipped_total counter should record the
    multimodal downgrade so ops can monitor how often it fires."""
    from infera.server import metrics

    hashes = [10, 20]
    client = _StubKvClient({"w1": set(), "w2": set()})
    policy = KvEventAwarePolicy(client, _StubHasher(hashes))  # type: ignore[arg-type]

    before = metrics.cache_locality_skipped_total.labels(reason="multimodal")._value.get()
    policy.pick([_worker("w1"), _worker("w2")], _mm_request())
    after = metrics.cache_locality_skipped_total.labels(reason="multimodal")._value.get()
    assert after - before == 1


def test_text_only_request_does_NOT_increment_skipped_metric():
    """Sanity: the metric only fires on MM requests, not every pick()."""
    from infera.server import metrics

    hashes = [10, 20]
    client = _StubKvClient({"w1": set(), "w2": set()})
    policy = KvEventAwarePolicy(client, _StubHasher(hashes))  # type: ignore[arg-type]

    text_body = {"model": "test/m", "messages": [{"role": "user", "content": "hi"}]}
    before = metrics.cache_locality_skipped_total.labels(reason="multimodal")._value.get()
    policy.pick([_worker("w1"), _worker("w2")], text_body)
    after = metrics.cache_locality_skipped_total.labels(reason="multimodal")._value.get()
    assert after == before


def test_retention_hint_works_via_parse_cache_hints():
    """When `_infera_cache_hints` is NOT pre-attached, the policy
    should parse it from the body inline. Same outcome as the
    pre-attached case."""
    from infera.router.cache_control import parse_cache_hints

    hashes = [10, 20]
    client = _StubKvClient({"w1": {10, 20}, "w2": set()})
    policy = KvEventAwarePolicy(client, _StubHasher(hashes), overlap_weight=1.0)  # type: ignore[arg-type]
    for i in range(3):
        policy.on_request_started("w1", [i + 1000])

    body = _request_with_retention("long")
    # Verify a fresh body has no pre-attached hints
    assert "_infera_cache_hints" not in body
    picked_inline, _ = policy.pick([_worker("w1"), _worker("w2")], body)

    # Now attach hints explicitly (what the server does) — same result.
    body2 = _request_with_retention("long")
    body2["_infera_cache_hints"] = parse_cache_hints(body2)
    # Need a fresh policy so active_blocks state is identical
    policy2 = KvEventAwarePolicy(client, _StubHasher(hashes), overlap_weight=1.0)  # type: ignore[arg-type]
    for i in range(3):
        policy2.on_request_started("w1", [i + 1000])
    picked_attached, _ = policy2.pick([_worker("w1"), _worker("w2")], body2)

    assert picked_inline.worker.worker_id == picked_attached.worker.worker_id == "w1"
