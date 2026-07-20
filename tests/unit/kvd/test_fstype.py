###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for the kvd filesystem-type detection helper.

`get_fstype()` + `hipfile_friendly_fstype()` let the
engine-side hipFile call site auto-disable GDS on
tmpfs / overlay where the direct DMA path silently falls back to a
CPU bounce buffer. Same pattern as LMCache's `gds_backend.py`.

Detection has to:
  * resolve symlinks BEFORE matching /proc/mounts (bind-mounts, etc.);
  * pick the longest mount-point prefix (since `/` always matches);
  * raise loudly when no mount matches (shouldn't happen on Linux,
    but be explicit instead of falling through to a wrong fstype).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from infera.kvd import ssd as ssd_mod
from infera.kvd.ssd import (
    SpilloverRegion,
    get_fstype,
    hipfile_friendly_fstype,
)

# ----------------------------------------------------------------------
# get_fstype()
# ----------------------------------------------------------------------


class TestGetFstype:
    def test_get_fstype_returns_tmpfs_for_dev_shm(self):
        """/dev/shm is tmpfs on every Linux distro we'd ever deploy on."""
        assert get_fstype("/dev/shm") == "tmpfs"

    def test_get_fstype_returns_a_real_fs_for_tmp(self):
        """/tmp varies by host (tmpfs on systemd defaults, ext4 on
        bare-metal). Assert only that the result is in the known set
        of fstypes we'd reasonably encounter — not which specific one."""
        fstype = get_fstype("/tmp")
        # Don't hard-pin to one — just "is it a plausible Linux fs".
        assert fstype in {
            "tmpfs",
            "ext4",
            "xfs",
            "btrfs",
            "overlay",
            "overlayfs",
            "nfs",
            "nfs4",
            "wekafs",
            "zfs",
            "f2fs",
        }, f"unexpected fstype for /tmp: {fstype!r}"

    def test_get_fstype_resolves_symlinks(self, tmp_path: Path):
        """Symlink → mkdir under /dev/shm; the symlink's fstype should
        come from /dev/shm (tmpfs) NOT from where the symlink itself
        lives. This is the whole point of the .resolve() call."""
        real_dir = Path("/dev/shm") / f"kvd_fstype_test_{id(self)}"
        real_dir.mkdir(exist_ok=True)
        try:
            link = tmp_path / "linked"
            link.symlink_to(real_dir)
            assert get_fstype(link) == "tmpfs"
        finally:
            try:
                real_dir.rmdir()
            except OSError:
                pass

    def test_get_fstype_raises_on_unmounted_path(self, monkeypatch):
        """Mock /proc/mounts to contain only a single non-prefix mount,
        then look up an unrelated path — no prefix can match, so we
        must raise. (Faking the file is the only portable way to
        synthesize this — Linux always has `/` mounted.)"""
        fake_mounts = "tmpfs /some/other/mount tmpfs rw 0 0\n"

        class _FakeFile:
            def __init__(self, content: str) -> None:
                self._content = content

            def __enter__(self):
                import io

                return io.StringIO(self._content)

            def __exit__(self, *args):
                return False

        def fake_open(path, *args, **kwargs):  # noqa: ARG001 — match builtin signature
            assert path == "/proc/mounts"
            return _FakeFile(fake_mounts)

        monkeypatch.setattr(ssd_mod, "open", fake_open, raising=False)
        with pytest.raises(RuntimeError, match="Unable to detect fstype"):
            get_fstype("/var/lib/whatever")

    def test_longest_prefix_wins(self, monkeypatch):
        """Both `/` and `/mnt/foo` are prefixes of `/mnt/foo/bar`; the
        longer one (the actual mount) must win."""
        fake_mounts = "/dev/sda1 / ext4 rw 0 0\ntmpfs /mnt/foo tmpfs rw 0 0\n"

        def fake_read():
            return [("/", "ext4"), ("/mnt/foo", "tmpfs")]

        monkeypatch.setattr(ssd_mod, "_read_mounts", fake_read)
        # Use a path under /mnt/foo that doesn't have to exist —
        # resolve() will collapse it to /mnt/foo/bar.
        # Use Path("/mnt/foo/bar"); resolve may collapse but no symlinks.
        from pathlib import Path as _P

        assert get_fstype(_P("/mnt/foo/bar")) == "tmpfs"
        assert get_fstype(_P("/var/lib/x")) == "ext4"
        _ = fake_mounts  # silence unused (kept for documentation)

    def test_prefix_match_respects_path_boundaries(self, monkeypatch):
        """`/var` must NOT match `/var-old/whatever` — naive
        `startswith` would; we use a proper path-segment boundary."""

        def fake_read():
            return [("/", "ext4"), ("/var", "xfs")]

        monkeypatch.setattr(ssd_mod, "_read_mounts", fake_read)
        # /var-old is NOT under /var; should fall back to /.
        assert get_fstype(Path("/var-old")) == "ext4"
        assert get_fstype(Path("/var")) == "xfs"
        assert get_fstype(Path("/var/lib")) == "xfs"

    def test_decodes_octal_escapes_in_mount_field(self):
        """/proc/mounts encodes whitespace in mount points as `\\040`.
        Make sure the parser undoes that so prefix matching works."""
        assert ssd_mod._decode_mount_field("plain") == "plain"
        assert ssd_mod._decode_mount_field("/mnt/has\\040space") == "/mnt/has space"
        assert ssd_mod._decode_mount_field("/a\\011b") == "/a\tb"


# ----------------------------------------------------------------------
# hipfile_friendly_fstype()
# ----------------------------------------------------------------------


class TestHipfileFriendlyFstype:
    @pytest.mark.parametrize("fstype", ["tmpfs", "overlay", "overlayfs"])
    def test_known_compat_fallbacks_are_unfriendly(self, fstype: str):
        assert hipfile_friendly_fstype(fstype) is False

    @pytest.mark.parametrize(
        "fstype",
        ["ext4", "xfs", "btrfs", "nfs", "nfs4", "wekafs"],
    )
    def test_real_filesystems_are_friendly(self, fstype: str):
        assert hipfile_friendly_fstype(fstype) is True

    def test_unknown_fstype_defaults_to_friendly(self):
        """Safe default: we don't maintain an allow-list, hipFile's own
        runtime will decide if the hardware can DMA on this mount."""
        assert hipfile_friendly_fstype("some-future-fs") is True


# ----------------------------------------------------------------------
# SsdRegion start() — informational log on tmpfs/overlay
# ----------------------------------------------------------------------


class TestSsdRegionFstypeLog:
    def test_ssd_region_logs_compat_warning_on_tmpfs(self, tmp_path: Path, caplog, monkeypatch):
        """Operator-facing INFO line fires when the region root is on
        tmpfs/overlay. Use SpilloverRegion (simplest start() path) and
        point it at a tmp_path subdir — we monkeypatch the detector to
        unconditionally return tmpfs so the test is portable across
        hosts where /tmp may actually be ext4."""

        def fake_get_fstype(path):  # noqa: ARG001
            return "tmpfs"

        monkeypatch.setattr(ssd_mod, "get_fstype", fake_get_fstype)

        region = SpilloverRegion(tmp_path / "spillover", max_bytes=1024)
        with caplog.at_level(logging.INFO, logger="infera.kvd.ssd"):
            region.start()

        compat_records = [
            r
            for r in caplog.records
            if "compat-fallback" in r.message and r.levelno == logging.INFO
        ]
        assert len(compat_records) == 1, (
            f"expected exactly one compat-fallback INFO, got {len(compat_records)}: "
            f"{[r.message for r in caplog.records]}"
        )
        assert "spillover" in compat_records[0].message
        assert "tmpfs" in compat_records[0].message

    def test_ssd_region_silent_on_friendly_fstype(self, tmp_path: Path, caplog, monkeypatch):
        """No compat-fallback log on ext4 etc."""

        def fake_get_fstype(path):  # noqa: ARG001
            return "ext4"

        monkeypatch.setattr(ssd_mod, "get_fstype", fake_get_fstype)

        region = SpilloverRegion(tmp_path / "spillover", max_bytes=1024)
        with caplog.at_level(logging.INFO, logger="infera.kvd.ssd"):
            region.start()

        assert not any("compat-fallback" in r.message for r in caplog.records)

    def test_ssd_region_swallows_detection_errors(self, tmp_path: Path, caplog, monkeypatch):
        """A blown-up /proc/mounts must not break region startup —
        we degrade to a debug log and keep going."""

        def fake_get_fstype(path):  # noqa: ARG001
            raise RuntimeError("synthetic")

        monkeypatch.setattr(ssd_mod, "get_fstype", fake_get_fstype)

        region = SpilloverRegion(tmp_path / "spillover", max_bytes=1024)
        # Should NOT raise.
        region.start()
        # No INFO compat-fallback (the detector failed, not "tmpfs").
        assert not any("compat-fallback" in r.message for r in caplog.records)
