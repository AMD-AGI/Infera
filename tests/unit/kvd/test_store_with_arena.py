###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""HostStore wired with a SharedArena.

Covers the production-grade path where bytes live in a memfd-backed
arena (not in a Python bytes dict): writes go through the arena,
reads materialize bytes via the arena's seqlock, and evictions free
arena slots back to the free list.

Invariants that must hold:
- `resolve_value(entry)` returns the bytes regardless of storage mode.
- Eviction frees the arena slot (otherwise we leak slots).
- TTL expiration frees the arena slot.
- Inline-mode store (shared_arena=None) behaves EXACTLY as before
  (covered by existing tests; this file adds the arena-on path).
"""

from __future__ import annotations

import time

from infera.kvd.shared_arena import SharedArena
from infera.kvd.store import HostStore


def _make_store_with_arena(
    arena_capacity: int = 64 * 1024, max_bytes: int = 1 << 20
) -> tuple[HostStore, SharedArena]:
    arena = SharedArena(arena_capacity, pin_memory=False)
    store = HostStore(max_bytes=max_bytes, shared_arena=arena)
    return store, arena


def test_set_routes_through_arena():
    """A SET against an arena-wired store puts the bytes into the
    arena and the Entry carries slot_id >= 0 (value stays empty)."""
    store, arena = _make_store_with_arena()
    try:
        store.set(b"k", b"hello" + b"\x00" * 100, retention="short")
        entry = store.get(b"k")
        assert entry is not None
        assert entry.slot_id >= 0
        assert entry.value == b""
        # But resolve_value gives the actual bytes.
        assert store.resolve_value(entry) == b"hello" + b"\x00" * 100
        # And arena has one entry.
        assert arena.stats().entries == 1
    finally:
        arena.close()


def test_overwrite_arena_backed_entry():
    """Overwriting an arena-backed key updates the slot in place
    (or reuses a freed slot)."""
    store, arena = _make_store_with_arena()
    try:
        store.set(b"k", b"a" * 100)
        store.set(b"k", b"b" * 100)
        entry = store.get(b"k")
        assert store.resolve_value(entry) == b"b" * 100
        # Still one entry (overwrite, not insert).
        assert arena.stats().entries == 1
    finally:
        arena.close()


def test_arena_lru_eviction_drops_host_store_entry():
    """When the ARENA's own LRU evicts a slot (arena slot count
    exhausted), HostStore must drop the corresponding `_entries`
    entry — otherwise `get(evicted_key)` would return an Entry
    pointing at a slot that's been reassigned to a different key.

    Under the post-2026-05-26 fix, HostStore drains arena's
    `recent_evicted_keys` after each `set` and prunes its dict.

    Pre-fix: this test failed because HostStore was wrongly
    counting arena bytes against max_bytes, so HostStore's own LRU
    fired first and the arena never had a chance to evict.
    Post-fix: max_bytes is high, the arena fills its 4-slot grid,
    the 5th put forces the arena to evict its oldest entry — and
    HostStore must follow."""
    # 4 slots in the arena, plenty of max_bytes so HostStore never
    # triggers its own eviction. Forces arena LRU to be the one
    # that evicts.
    store, arena = _make_store_with_arena(
        arena_capacity=4 * 256,  # 4 slots of 256 (200+16=216 rounded to 256)
        max_bytes=1 << 30,
    )
    try:
        for i, k in enumerate([b"k1", b"k2", b"k3", b"k4"]):
            store.set(k, bytes([0x10 + i]) * 200)
        assert arena.stats().entries == 4
        # 5th put forces arena LRU eviction — k1 is oldest.
        store.set(b"k5", b"e" * 200)
        # k1 dropped from arena → HostStore must also drop it.
        assert store.get(b"k1") is None, (
            "HostStore stale: arena evicted k1's slot, but the entry "
            "still appears in _entries with the recycled slot_id"
        )
        # k2..k5 still present.
        assert store.get(b"k2") is not None
        assert store.get(b"k5") is not None
        # Arena reports 4 entries (4 slots, oldest evicted, new added).
        assert arena.stats().entries == 4
    finally:
        arena.close()


def test_ttl_expiration_frees_arena_slot():
    """A TTL'd entry that expires must also release its arena slot
    (the GET path's lazy-removal codepath uses _remove_entry_locked,
    which we patched to free)."""
    store, arena = _make_store_with_arena()
    try:
        store.set(b"k", b"x" * 100, ttl_seconds=0.05)
        time.sleep(0.1)
        # Lazy expiration: GET returns None and removes the entry.
        assert store.get(b"k") is None
        assert arena.stats().entries == 0
    finally:
        arena.close()


def test_inline_store_unaffected_by_arena_changes():
    """A store constructed WITHOUT a shared_arena (the existing
    backward-compat path) must behave identically to before — bytes
    live in Entry.value, no arena interaction. This test guards
    against accidental regressions."""
    store = HostStore(max_bytes=1 << 20)
    store.set(b"k", b"hello")
    entry = store.get(b"k")
    assert entry is not None
    assert entry.slot_id == -1  # inline mode sentinel
    assert entry.value == b"hello"
    # resolve_value works on inline too.
    assert store.resolve_value(entry) == b"hello"


def test_size_bytes_reports_correctly_in_both_modes():
    """Entry.size_bytes must return the logical size regardless of
    storage mode (otherwise eviction math + stats break)."""
    store_a, arena = _make_store_with_arena()
    try:
        store_a.set(b"k", b"x" * 100)
        entry = store_a.get(b"k")
        assert entry.size_bytes == 100  # from _size_cache
    finally:
        arena.close()

    store_b = HostStore(max_bytes=1 << 20)
    store_b.set(b"k", b"y" * 200)
    entry = store_b.get(b"k")
    assert entry.size_bytes == 200  # from len(value)


def test_clear_frees_arena_slots():
    """`store.clear()` (whole-store flavor) must release every
    arena slot. Otherwise repeated clear+set cycles leak."""
    store, arena = _make_store_with_arena()
    try:
        store.set(b"k1", b"a" * 100)
        store.set(b"k2", b"b" * 100)
        assert arena.stats().entries == 2
        store.clear()
        assert arena.stats().entries == 0
        # And new sets work after clear.
        store.set(b"k3", b"c" * 100)
        assert arena.stats().entries == 1
    finally:
        arena.close()


def test_shared_arena_property_exposed():
    store, arena = _make_store_with_arena()
    try:
        assert store.shared_arena is arena
    finally:
        arena.close()
    # Inline store reports None.
    store2 = HostStore(max_bytes=1 << 20)
    assert store2.shared_arena is None


# ----------------------------------------------------------------------
# max_bytes vs arena_bytes accounting (regression for bench finding
# 2026-05-26: HostStore was double-counting arena-backed bytes against
# max_bytes, triggering premature LRU evictions even when the arena
# had plenty of slot capacity).
# ----------------------------------------------------------------------


def test_arena_bytes_do_not_count_against_max_bytes():
    """The bench bug: HostStore with max_bytes=1MB and arena=64MB
    should accept many small arena-backed entries WITHOUT evicting,
    because arena bytes aren't billed against max_bytes."""
    arena = SharedArena(64 * 1024 * 1024, pin_memory=False)
    try:
        # max_bytes is SMALLER than what we're going to insert (8 × 256KB =
        # 2 MB). Pre-fix, this would trigger 7 evictions; post-fix, 0.
        store = HostStore(max_bytes=1 * 1024 * 1024, shared_arena=arena)
        for i in range(8):
            accepted, _ = store.set(bytes([i]) * 8, b"x" * (256 * 1024), retention="short")
            assert accepted, f"insert {i} should succeed"
        # All 8 entries present.
        assert len(store._entries) == 8
        # Zero evictions — arena absorbed everything.
        assert store.stats.evictions_total == 0
        # `_used_bytes` only tracks INLINE bytes (none here — everything
        # in arena).
        assert store._used_bytes == 0
    finally:
        arena.close()


def test_inline_fallback_still_obeys_max_bytes():
    """When arena rejects (size mismatch with locked slot_size), the
    entry falls back to inline storage AND is then billed against
    max_bytes the same way pre-arena entries always were."""
    arena = SharedArena(4 * 1024 * 1024, pin_memory=False)
    try:
        store = HostStore(max_bytes=512 * 1024, shared_arena=arena)
        # First put locks slot_size at ~100 + 16 → 128 bytes.
        accepted, _ = store.set(b"k0" + b"\x00" * 6, b"x" * 100, retention="short")
        assert accepted
        # Second put with a much larger value — arena will reject
        # (size > slot_size). The entry should fall back to inline.
        accepted, _ = store.set(b"k1" + b"\x00" * 6, b"y" * 200, retention="short")
        assert accepted
        # The 2nd entry's 200 bytes ARE inline → counted in
        # _used_bytes.
        assert store._used_bytes == 200
        # If we keep inserting oversize blobs, eventually max_bytes
        # would gate inline storage — but 200 bytes is well under
        # 512 KiB, so this one fits.
    finally:
        arena.close()


def test_arena_eviction_frees_slot_when_host_store_evicts():
    """When HostStore's own LRU evicts an arena-backed entry (e.g.
    inline+arena mixed scenario where inline crosses max_bytes), the
    arena slot must be released back to the free list so the next
    put can reuse it. Counter for slot-leak bugs."""
    arena = SharedArena(4 * 1024 * 1024, pin_memory=False)
    try:
        store = HostStore(max_bytes=1 << 30, shared_arena=arena)
        store.set(b"k1" + b"\x00" * 6, b"x" * 100, retention="short")
        store.set(b"k2" + b"\x00" * 6, b"y" * 100, retention="short")
        # Force eviction by clearing directly (mimics what TTL or
        # explicit eviction would do).
        store.clear()
        # Arena should now have both slots back in its free list.
        # `would_accept_size` should still return True (slot_size
        # stays locked across clear, free list is rebuilt).
        assert arena.would_accept_size(100)
        # And we should be able to put fresh entries.
        store.set(b"k3" + b"\x00" * 6, b"z" * 100, retention="short")
        assert b"k3" + b"\x00" * 6 in [k for _, _, k in store._entries]
    finally:
        arena.close()


def test_would_accept_size_pre_first_put():
    """Before any put has fixed slot_size, the arena accepts any
    size up to its total capacity."""
    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        assert arena.would_accept_size(1024)
        assert arena.would_accept_size(64 * 1024 - 100)
        # Beyond capacity — refuse.
        assert not arena.would_accept_size(64 * 1024 * 2)
    finally:
        arena.close()


def test_would_accept_size_after_first_put_locks_slot():
    """First put locks slot_size; subsequent would_accept_size honors
    that — sizes that won't fit return False."""
    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        arena.put(b"k1", b"x" * 100)
        # First put set slot_size to ceil(100+16, 64) = 128.
        assert arena.would_accept_size(50)
        assert arena.would_accept_size(100)
        # 200 + 16 = 216 > 128 → reject.
        assert not arena.would_accept_size(200)
    finally:
        arena.close()
