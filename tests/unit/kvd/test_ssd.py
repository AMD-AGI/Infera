###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for infera/kvd/ssd.py — SpilloverRegion + LongStorageRegion.

These cover the region in isolation (without HostStore). HostStore
integration tests live in test_store_ssd.py.

## On-disk layout assertions

Since 2026-06 each block lives at:

    {root}/{hash[:2]}/{hash[2:4]}/<urlencoded(composite)>.kvcache
    {root}/{hash[:2]}/{hash[2:4]}/<urlencoded(composite)>.kvcache.metadata

with composite = ``f"{model}|{compat_key}|{b64url(key)}"``. The 4 KiB
``.kvcache.metadata`` sidecar carries per-entry metadata; on startup
the long region rebuilds its index by parallel-scanning sidecars.
The legacy ``blocks/<path_hex>.kv`` layout + ``manifest.json`` is
gone — see ssd.py module docstring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from infera.kvd.ssd import (
    _DATA_EXT,
    _METADATA_EXT,
    LongStorageRegion,
    SpilloverRegion,
    _composite_hash,
    _decode_composite,
    _decode_sidecar,
    _encode_composite,
    _encode_sidecar,
    _filename_for_composite,
)


def _key(s: str) -> bytes:
    return s.encode("ascii").ljust(8, b"\x00")


def _all_data_files(region_root: Path) -> list[Path]:
    """All ``.kvcache`` files anywhere under the region root."""
    if not region_root.exists():
        return []
    return list(region_root.rglob(f"*{_DATA_EXT}"))


def _all_metadata_files(region_root: Path) -> list[Path]:
    if not region_root.exists():
        return []
    return list(region_root.rglob(f"*{_METADATA_EXT}"))


# ----------------------------------------------------------------------
# Composite encoding helpers
# ----------------------------------------------------------------------


class TestCompositeEncoding:
    def test_round_trip_simple(self):
        composite = _encode_composite("model/x", "fp16", b"abc")
        model, ck, key = _decode_composite(composite)
        assert model == "model/x"
        assert ck == "fp16"
        assert key == b"abc"

    def test_round_trip_binary_key(self):
        raw = bytes(range(256))
        composite = _encode_composite("m", "ck", raw)
        _, _, key = _decode_composite(composite)
        assert key == raw

    def test_round_trip_empty_fields(self):
        composite = _encode_composite("", "", b"")
        model, ck, key = _decode_composite(composite)
        assert model == ""
        assert ck == ""
        assert key == b""

    def test_filename_is_posix_safe(self):
        # b64url can produce '-' and '_' which are already safe; the
        # urlencode is mostly about the '|' separator and any model
        # name oddities.
        composite = _encode_composite("ns/m", "ck=1", b"\x00\x01\x02")
        fname = _filename_for_composite(composite)
        assert "/" not in fname
        assert "|" not in fname
        # And the encoded name is a single flat string usable as a
        # filename on POSIX.
        assert all(c not in fname for c in "/\0")

    def test_decode_rejects_malformed(self):
        with pytest.raises(ValueError):
            _decode_composite("not-composite-shape")


# ----------------------------------------------------------------------
# Sidecar encode/decode
# ----------------------------------------------------------------------


class TestSidecar:
    def test_sidecar_is_exactly_4kib(self):
        blob = _encode_sidecar({"version": 1, "size_bytes": 16, "retention": "long"})
        assert len(blob) == 4096

    def test_round_trip(self):
        payload = {
            "version": 1,
            "retention": "long",
            "size_bytes": 12345,
            "model": "test/m",
            "compat_key": "ck",
            "extra": [1, 2, "three"],
        }
        blob = _encode_sidecar(payload)
        decoded = _decode_sidecar(blob)
        assert decoded == payload

    def test_payload_too_large_raises(self):
        from infera.kvd.ssd import SidecarError

        huge = {"bigfield": "x" * 5000}
        with pytest.raises(SidecarError):
            _encode_sidecar(huge)

    def test_decode_rejects_wrong_size(self):
        from infera.kvd.ssd import SidecarError

        with pytest.raises(SidecarError):
            _decode_sidecar(b"too short")


# ----------------------------------------------------------------------
# Spillover region — short retention, lazy, wiped on restart
# ----------------------------------------------------------------------


class TestSpilloverRegion:
    def test_put_get_round_trip(self, tmp_path: Path):
        region = SpilloverRegion(tmp_path / "spillover", max_bytes=1024)
        region.start()

        accepted, reason = region.put(_key("a"), b"hello", retention="short")
        assert accepted is True
        assert reason is None
        assert region.get_bytes(_key("a")) == b"hello"

    def test_get_miss_returns_none(self, tmp_path: Path):
        region = SpilloverRegion(tmp_path / "spillover", max_bytes=1024)
        region.start()
        assert region.get_bytes(_key("nope")) is None

    def test_rejects_long_retention(self, tmp_path: Path):
        region = SpilloverRegion(tmp_path / "spillover", max_bytes=1024)
        region.start()
        accepted, reason = region.put(_key("a"), b"x", retention="long")
        assert accepted is False
        assert reason == "spillover_only_accepts_short_retention"

    def test_value_larger_than_region_rejected(self, tmp_path: Path):
        region = SpilloverRegion(tmp_path / "spillover", max_bytes=10)
        region.start()
        accepted, reason = region.put(_key("a"), b"x" * 100, retention="short")
        assert accepted is False
        assert reason == "value_larger_than_region"

    def test_lru_evicts_oldest_when_full(self, tmp_path: Path):
        """Plain LRU within the region."""
        import time

        region = SpilloverRegion(tmp_path / "spillover", max_bytes=20)
        region.start()
        region.put(_key("a"), b"x" * 10, retention="short")
        time.sleep(0.01)
        region.put(_key("b"), b"y" * 10, retention="short")
        # Now full. Adding c should evict a (oldest).
        accepted, _ = region.put(_key("c"), b"z" * 10, retention="short")
        assert accepted
        assert region.get_bytes(_key("a")) is None
        assert region.get_bytes(_key("b")) == b"y" * 10
        assert region.get_bytes(_key("c")) == b"z" * 10

    def test_get_refreshes_last_access(self, tmp_path: Path):
        """Hitting a key promotes it past newer entries in LRU order."""
        import time

        region = SpilloverRegion(tmp_path / "spillover", max_bytes=20)
        region.start()
        region.put(_key("a"), b"x" * 10, retention="short")
        time.sleep(0.01)
        region.put(_key("b"), b"y" * 10, retention="short")
        # Touch a — now b is older.
        region.get_bytes(_key("a"))
        # Insert c: b should be evicted (now the oldest).
        region.put(_key("c"), b"z" * 10, retention="short")
        assert region.get_bytes(_key("a")) == b"x" * 10
        assert region.get_bytes(_key("b")) is None
        assert region.get_bytes(_key("c")) == b"z" * 10

    def test_remove(self, tmp_path: Path):
        region = SpilloverRegion(tmp_path / "spillover", max_bytes=100)
        region.start()
        region.put(_key("a"), b"hello", retention="short")
        assert region.remove(_key("a")) is True
        assert region.remove(_key("a")) is False  # idempotent
        assert region.get_bytes(_key("a")) is None

    def test_clear(self, tmp_path: Path):
        region = SpilloverRegion(tmp_path / "spillover", max_bytes=100)
        region.start()
        region.put(_key("a"), b"x", retention="short")
        region.put(_key("b"), b"y", retention="short")
        assert region.clear() == 2
        assert region.entries_count == 0
        assert region.used_bytes == 0

    def test_restart_wipes_state(self, tmp_path: Path):
        """Restart contract: nothing survives a SpilloverRegion start()."""
        region1 = SpilloverRegion(tmp_path / "spillover", max_bytes=100)
        region1.start()
        region1.put(_key("a"), b"x", retention="short")
        assert region1.get_bytes(_key("a")) == b"x"

        # New region pointing at the same directory.
        region2 = SpilloverRegion(tmp_path / "spillover", max_bytes=100)
        region2.start()
        # Wiped.
        assert region2.get_bytes(_key("a")) is None
        assert region2.entries_count == 0
        # And the on-disk files are gone too.
        assert _all_data_files(tmp_path / "spillover") == []

    def test_namespace_isolation(self, tmp_path: Path):
        """Same key under different (model, compat_key) shouldn't collide."""
        region = SpilloverRegion(tmp_path / "spillover", max_bytes=200)
        region.start()
        region.put(_key("k"), b"v1", retention="short", model="m1", compat_key="ck1")
        region.put(_key("k"), b"v2", retention="short", model="m2", compat_key="ck2")
        assert region.get_bytes(_key("k"), model="m1", compat_key="ck1") == b"v1"
        assert region.get_bytes(_key("k"), model="m2", compat_key="ck2") == b"v2"


# ----------------------------------------------------------------------
# On-disk path shape assertions
# ----------------------------------------------------------------------


class TestOnDiskPathShape:
    """The whole point of the 2026-06 refactor: 2/2-sharded directory
    tree with sidecars."""

    def test_data_and_sidecar_pair_exist(self, tmp_path: Path):
        region = LongStorageRegion(tmp_path / "long", max_bytes=1024)
        region.start()
        region.put(_key("a"), b"hello", retention="long", model="m", compat_key="ck")

        datas = _all_data_files(tmp_path / "long")
        metas = _all_metadata_files(tmp_path / "long")
        assert len(datas) == 1
        assert len(metas) == 1
        # Same stem.
        assert datas[0].name[: -len(_DATA_EXT)] == metas[0].name[: -len(_METADATA_EXT)]

    def test_sidecar_is_exactly_4kib_on_disk(self, tmp_path: Path):
        region = LongStorageRegion(tmp_path / "long", max_bytes=1024)
        region.start()
        region.put(_key("a"), b"hello", retention="long", model="m", compat_key="ck")
        metas = _all_metadata_files(tmp_path / "long")
        assert metas[0].stat().st_size == 4096

    def test_path_uses_22_sharded_dirs(self, tmp_path: Path):
        region = LongStorageRegion(tmp_path / "long", max_bytes=1024)
        region.start()
        region.put(_key("a"), b"hello", retention="long", model="m", compat_key="ck")
        datas = _all_data_files(tmp_path / "long")
        # Path under the region root: {XX}/{YY}/<file>.kvcache
        rel = datas[0].relative_to(tmp_path / "long")
        parts = rel.parts
        assert len(parts) == 3, f"expected 3 parts (XX/YY/file), got {parts}"
        l1, l2, _ = parts
        assert len(l1) == 2 and all(c in "0123456789abcdef" for c in l1)
        assert len(l2) == 2 and all(c in "0123456789abcdef" for c in l2)

    def test_shard_path_matches_composite_hash(self, tmp_path: Path):
        region = LongStorageRegion(tmp_path / "long", max_bytes=1024)
        region.start()
        region.put(_key("a"), b"hello", retention="long", model="m", compat_key="ck")
        composite = _encode_composite("m", "ck", _key("a"))
        h = _composite_hash(composite)
        expected = tmp_path / "long" / h[:2] / h[2:4]
        # The file lives in the expected leaf dir.
        leaf = _all_data_files(tmp_path / "long")[0].parent
        assert leaf == expected


# ----------------------------------------------------------------------
# Long-storage region — long retention, write_through, persistent
# ----------------------------------------------------------------------


class TestLongStorageRegion:
    def test_put_get_round_trip(self, tmp_path: Path):
        region = LongStorageRegion(tmp_path / "long", max_bytes=1024)
        region.start()
        accepted, _ = region.put(_key("a"), b"hello", retention="long")
        assert accepted
        assert region.get_bytes(_key("a")) == b"hello"

    def test_rejects_short_retention(self, tmp_path: Path):
        region = LongStorageRegion(tmp_path / "long", max_bytes=1024)
        region.start()
        accepted, reason = region.put(_key("a"), b"x", retention="short")
        assert accepted is False
        assert reason == "long_region_only_accepts_long_retention"

    def test_state_survives_restart(self, tmp_path: Path):
        """The whole point of the long region: bytes AND index recover."""
        region1 = LongStorageRegion(tmp_path / "long", max_bytes=1024)
        region1.start()
        region1.put(
            _key("a"), b"hello-from-region1", retention="long", model="test/m", compat_key="ck1"
        )
        region1.put(_key("b"), b"another", retention="long", model="test/m", compat_key="ck1")

        # New region pointing at the same directory — sidecar recovery.
        region2 = LongStorageRegion(tmp_path / "long", max_bytes=1024)
        region2.start()

        # After restart, GETs should hit (file read happens on demand).
        assert (
            region2.get_bytes(_key("a"), model="test/m", compat_key="ck1") == b"hello-from-region1"
        )
        assert region2.get_bytes(_key("b"), model="test/m", compat_key="ck1") == b"another"
        assert region2.entries_count == 2

    def test_restart_recovers_many_entries_parallel(self, tmp_path: Path):
        """Bulk recovery: write N entries, restart, assert all come
        back. Exercises the parallel scandir codepath (multiple shard
        dirs)."""
        N = 64
        region1 = LongStorageRegion(tmp_path / "long", max_bytes=N * 1024)
        region1.start()
        for i in range(N):
            region1.put(
                f"key-{i:04d}".encode(),
                f"value-{i:04d}".encode() * 32,  # ~256 bytes each
                retention="long",
                model="test/m",
                compat_key="ck",
            )
        assert region1.entries_count == N

        region2 = LongStorageRegion(tmp_path / "long", max_bytes=N * 1024)
        region2.start()
        assert region2.entries_count == N
        # Spot-check a few keys round-trip.
        for i in (0, N // 2, N - 1):
            v = region2.get_bytes(f"key-{i:04d}".encode(), model="test/m", compat_key="ck")
            assert v == f"value-{i:04d}".encode() * 32

    def test_restart_unlinks_orphan_data_files(self, tmp_path: Path):
        """A `.kvcache` with no matching `.kvcache.metadata` is the
        signature of a crash between the data and sidecar writes. The
        next startup must unlink it rather than serving phantom bytes."""
        region1 = LongStorageRegion(tmp_path / "long", max_bytes=1024)
        region1.start()
        region1.put(_key("a"), b"x" * 64, retention="long")

        # Delete just the sidecar — simulates crash after data rename
        # but before sidecar rename.
        metas = _all_metadata_files(tmp_path / "long")
        for m in metas:
            m.unlink()

        region2 = LongStorageRegion(tmp_path / "long", max_bytes=1024)
        region2.start()
        assert region2.entries_count == 0
        # And the orphan data file was reclaimed.
        assert _all_data_files(tmp_path / "long") == []

    def test_restart_unlinks_sidecar_with_no_data(self, tmp_path: Path):
        """Inverse: dangling sidecar with no data file. The startup
        scan must drop the sidecar (no bytes to serve)."""
        region1 = LongStorageRegion(tmp_path / "long", max_bytes=1024)
        region1.start()
        region1.put(_key("a"), b"x" * 64, retention="long")

        for d in _all_data_files(tmp_path / "long"):
            d.unlink()

        region2 = LongStorageRegion(tmp_path / "long", max_bytes=1024)
        region2.start()
        assert region2.entries_count == 0
        assert _all_metadata_files(tmp_path / "long") == []

    def test_restart_drops_size_mismatch(self, tmp_path: Path):
        """If the data file's size doesn't match the sidecar's
        size_bytes, the loader should drop both files (corruption /
        truncation)."""
        region1 = LongStorageRegion(tmp_path / "long", max_bytes=1024)
        region1.start()
        region1.put(_key("a"), b"x" * 64, retention="long")

        # Truncate the data file.
        for f in _all_data_files(tmp_path / "long"):
            f.write_bytes(b"x" * 16)  # smaller than sidecar claims

        region2 = LongStorageRegion(tmp_path / "long", max_bytes=1024)
        region2.start()
        assert region2.get_bytes(_key("a")) is None
        assert region2.entries_count == 0

    def test_restart_drops_corrupt_sidecar(self, tmp_path: Path, caplog):
        """A sidecar whose bytes are garbage should be reclaimed (both
        files unlinked) and the operator warned."""
        import logging

        region1 = LongStorageRegion(tmp_path / "long", max_bytes=1024)
        region1.start()
        region1.put(_key("a"), b"hello", retention="long")

        # Corrupt the sidecar.
        for m in _all_metadata_files(tmp_path / "long"):
            m.write_bytes(b"GARBAGE" * 512 + b"x" * (4096 - 7 * 512))

        region2 = LongStorageRegion(tmp_path / "long", max_bytes=1024)
        with caplog.at_level(logging.WARNING, logger="infera.kvd.ssd"):
            region2.start()
        assert region2.entries_count == 0
        assert any("unreadable" in r.message or "dropping" in r.message for r in caplog.records)

    def test_lru_when_full(self, tmp_path: Path):
        import time

        region = LongStorageRegion(tmp_path / "long", max_bytes=20)
        region.start()
        region.put(_key("a"), b"x" * 10, retention="long")
        time.sleep(0.01)
        region.put(_key("b"), b"y" * 10, retention="long")
        # Full; adding c evicts a.
        accepted, _ = region.put(_key("c"), b"z" * 10, retention="long")
        assert accepted
        assert region.get_bytes(_key("a")) is None
        assert region.get_bytes(_key("b")) == b"y" * 10
        assert region.get_bytes(_key("c")) == b"z" * 10

    def test_namespace_isolation(self, tmp_path: Path):
        region = LongStorageRegion(tmp_path / "long", max_bytes=200)
        region.start()
        region.put(_key("k"), b"v1", retention="long", model="m1", compat_key="ck")
        region.put(_key("k"), b"v2", retention="long", model="m2", compat_key="ck")
        assert region.get_bytes(_key("k"), model="m1", compat_key="ck") == b"v1"
        assert region.get_bytes(_key("k"), model="m2", compat_key="ck") == b"v2"


# ----------------------------------------------------------------------
# Legacy / migration handling
# ----------------------------------------------------------------------


class TestLegacyMigration:
    def test_warn_on_old_blocks_dir_only_no_crash(self, tmp_path: Path, caplog):
        """If the region root contains only the legacy 'blocks/' dir
        (no new shard dirs), we WARN and proceed with empty state."""
        import logging

        root = tmp_path / "long"
        (root / "blocks").mkdir(parents=True)
        # Drop a fake legacy block file to make it look real.
        (root / "blocks" / "deadbeef.kv").write_bytes(b"legacy data")

        region = LongStorageRegion(root, max_bytes=1024)
        with caplog.at_level(logging.WARNING, logger="infera.kvd.ssd"):
            region.start()
        # No crash; empty index.
        assert region.entries_count == 0
        # Operator warning fired.
        assert any("legacy" in r.message.lower() for r in caplog.records)
        # Legacy file is left alone (we don't auto-delete).
        assert (root / "blocks" / "deadbeef.kv").exists()

    def test_assert_on_ambiguous_old_plus_new(self, tmp_path: Path):
        """If BOTH the legacy 'blocks/' AND the new sharded layout are
        present, we have no way to know which represents reality. Fail
        fast rather than silently dropping data."""
        root = tmp_path / "long"
        # Bootstrap a real new-layout entry.
        region1 = LongStorageRegion(root, max_bytes=1024)
        region1.start()
        region1.put(_key("a"), b"x", retention="long")
        # Now add a fake legacy dir alongside it.
        (root / "blocks").mkdir()
        (root / "blocks" / "fake.kv").write_bytes(b"legacy")

        region2 = LongStorageRegion(root, max_bytes=1024)
        with pytest.raises(RuntimeError, match="ambiguous"):
            region2.start()

    def test_spillover_also_handles_legacy(self, tmp_path: Path, caplog):
        """Spillover applies the same legacy check so a stale 'blocks/'
        from a pre-refactor process doesn't silently mix layouts."""
        import logging

        root = tmp_path / "spillover"
        (root / "blocks").mkdir(parents=True)

        region = SpilloverRegion(root, max_bytes=1024)
        with caplog.at_level(logging.WARNING, logger="infera.kvd.ssd"):
            region.start()
        assert region.entries_count == 0
        assert any("legacy" in r.message.lower() for r in caplog.records)


# insert_metadata_only — hipfile-tier metadata index. kvd never opens
# or reads the engine-owned file; this
# method records (path, size, retention) so LookupTier can answer
# without disk IO.
# ----------------------------------------------------------------------


class TestInsertMetadataOnly:
    def test_long_region_indexes_path_without_writing(self, tmp_path: Path):
        region = LongStorageRegion(tmp_path / "long", max_bytes=1 << 20)
        region.start()
        # An arbitrary fictitious path — kvd should NOT stat or open it.
        path = "/srv/engine-hipfile/abc.kv"
        accepted, reason = region.insert_metadata_only(
            _key("a"),
            path=path,
            size=4096,
            retention="long",
            model="m",
            compat_key="ck",
            version=3,
        )
        assert accepted is True
        assert reason is None

        entry = region.get_entry(_key("a"), model="m", compat_key="ck")
        assert entry is not None
        assert entry.size_bytes == 4096
        assert entry.retention == "long"
        assert entry.metadata.get("path") == path
        assert entry.metadata.get("version") == 3
        # The kvd-internal block file is NOT created — metadata-only.
        blocks_dir = tmp_path / "long" / "blocks"
        assert not list(blocks_dir.glob("*.kv"))

    def test_spillover_region_indexes_short_retention(self, tmp_path: Path):
        region = SpilloverRegion(tmp_path / "spill", max_bytes=1024)
        region.start()
        accepted, reason = region.insert_metadata_only(
            _key("a"),
            path="/srv/hipfile/x.kv",
            size=1024,
            retention="short",
            model="m",
            compat_key="ck",
            version=0,
        )
        assert accepted is True
        assert reason is None
        assert region.entries_count == 1
        assert region.used_bytes == 1024

    def test_zero_size_rejected(self, tmp_path: Path):
        region = LongStorageRegion(tmp_path / "long", max_bytes=1024)
        region.start()
        accepted, reason = region.insert_metadata_only(
            _key("a"), path="/x", size=0, retention="long"
        )
        assert accepted is False
        assert reason == "bad_size"

    def test_negative_size_rejected(self, tmp_path: Path):
        region = LongStorageRegion(tmp_path / "long", max_bytes=1024)
        region.start()
        accepted, reason = region.insert_metadata_only(
            _key("a"), path="/x", size=-1, retention="long"
        )
        assert accepted is False

    def test_full_region_evicts_lru_to_make_room(self, tmp_path: Path):
        import time

        region = LongStorageRegion(tmp_path / "long", max_bytes=20)
        region.start()
        region.insert_metadata_only(_key("a"), path="/a", size=10, retention="long")
        time.sleep(0.01)
        region.insert_metadata_only(_key("b"), path="/b", size=10, retention="long")
        # Full — third entry should evict a (oldest).
        accepted, _ = region.insert_metadata_only(_key("c"), path="/c", size=10, retention="long")
        assert accepted
        assert region.get_entry(_key("a")) is None
        assert region.get_entry(_key("b")) is not None
        assert region.get_entry(_key("c")) is not None

    def test_update_in_place(self, tmp_path: Path):
        """Re-registering an existing key updates size + path + version
        without bumping entries_count."""
        region = LongStorageRegion(tmp_path / "long", max_bytes=1024)
        region.start()
        region.insert_metadata_only(_key("a"), path="/v1", size=10, retention="long", version=1)
        region.insert_metadata_only(_key("a"), path="/v2", size=20, retention="long", version=2)
        assert region.entries_count == 1
        assert region.used_bytes == 20
        entry = region.get_entry(_key("a"))
        assert entry.metadata.get("path") == "/v2"
        assert entry.metadata.get("version") == 2
        assert entry.size_bytes == 20

    def test_eviction_does_not_fail_on_missing_engine_file(self, tmp_path: Path):
        """When a metadata-only entry is evicted, the kvd-internal
        block file doesn't exist (the bytes live in the engine's
        hipfile). Eviction must swallow the FileNotFoundError."""
        region = LongStorageRegion(tmp_path / "long", max_bytes=10)
        region.start()
        region.insert_metadata_only(_key("a"), path="/a", size=10, retention="long")
        # This eviction would unlink <dir>/blocks/<path_hex>.kv which
        # was never created; the region's _delete_block_file_locked
        # already swallows FileNotFoundError. We just need to ensure
        # the call doesn't raise.
        accepted, _ = region.insert_metadata_only(_key("b"), path="/b", size=10, retention="long")
        assert accepted

    def test_bad_retention_value_rejected(self, tmp_path: Path):
        region = LongStorageRegion(tmp_path / "long", max_bytes=1024)
        region.start()
        import pytest

        with pytest.raises(ValueError, match="unknown retention"):
            region.insert_metadata_only(_key("a"), path="/x", size=10, retention="forever")
