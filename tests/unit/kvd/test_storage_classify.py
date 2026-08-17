###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unit tests for infera/kvd/storage_classify.py.

All tests mock the external subprocess calls (``findmnt`` / ``lsblk``)
and the /proc/mounts read — none of them touch the real system. This
keeps the test suite portable (runs in CI containers that may not have
util-linux) and deterministic across kernels.

Each test sets up a fake transport stack via monkeypatching the module's
``_run`` shim (so we don't have to fake ``subprocess.run`` itself), then
asserts on the ``pick_io_mode`` decision + the ``StorageInfo`` payload.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from infera.kvd import storage_classify
from infera.kvd.storage_classify import (
    DeviceInfo,
    StorageInfo,
    _parse_lsblk_devices,
    classify_storage,
    format_workers_decision,
    pick_io_mode,
    pick_workers_per_shard,
)

# ----------------------------------------------------------------------
# Fake _run() — drives findmnt + lsblk responses from a per-test table.
# ----------------------------------------------------------------------


class _FakeRun:
    """Drop-in for ``storage_classify._run``. Configure with a dict
    keyed by the first argv element (``findmnt`` / ``lsblk``); each
    value is either a string (stdout, rc=0) or a tuple ``(stdout, rc)``.
    Returns None for any unconfigured command."""

    def __init__(self, responses: dict[str, str | tuple[str, int]]):
        self.responses = responses
        self.calls: list[list[str]] = []

    def __call__(self, cmd, timeout=2.0):
        self.calls.append(list(cmd))
        if not cmd:
            return None
        head = cmd[0]
        if head not in self.responses:
            return None
        v = self.responses[head]
        if isinstance(v, tuple):
            stdout, rc = v
        else:
            stdout, rc = v, 0
        return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr="")


@pytest.fixture
def fake_run(monkeypatch):
    """Returns a constructor — tests do ``fake_run({...})`` per case.

    Also pins ``os.cpu_count`` high so ``pick_workers_per_shard``'s CPU
    guardrail (``cap = max(2, cpu // n_shards)``) never clamps the
    storage-derived worker count these tests assert. Without this the
    suite fails on low-core CI runners (e.g. a 2-core GitHub runner caps
    every "picks 8/4" case to 2 → ``assert 2 == 8``)."""

    monkeypatch.setattr("os.cpu_count", lambda: 64)

    def make(responses: dict[str, str | tuple[str, int]]) -> _FakeRun:
        fr = _FakeRun(responses)
        monkeypatch.setattr(storage_classify, "_run", fr)
        return fr

    return make


@pytest.fixture
def fake_proc_mounts(monkeypatch, tmp_path):
    """Patch /proc/mounts reading inside _nconnect_for_nfs by writing a
    test file and pointing the open() builtin at it via monkeypatching.
    We override the ``_nconnect_for_nfs`` import target instead — the
    function calls ``open("/proc/mounts")`` directly."""

    def write(contents: str) -> Path:
        f = tmp_path / "proc_mounts"
        f.write_text(contents)
        # Patch the global `open` used inside _nconnect_for_nfs.
        real_open = open

        def fake_open(path, *args, **kwargs):
            if path == "/proc/mounts":
                return real_open(f, *args, **kwargs)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)
        return f

    return write


# ----------------------------------------------------------------------
# _parse_lsblk_devices — direct unit tests on the column parser.
# ----------------------------------------------------------------------


def test_parse_lsblk_devices_single_nvme():
    """lsblk -no NAME,TRAN,ROTA /dev/nvme0n1 with no partitions."""
    out = "nvme0n1 nvme 0\n"
    leaves = _parse_lsblk_devices(out)
    assert leaves == [("nvme0n1", "nvme", False)]


def test_parse_lsblk_devices_disk_with_partitions():
    """Partitions inherit the disk's transport (lsblk leaves TRAN
    blank on partition rows). The walker must propagate the parent's
    transport to the leaf partition."""
    # Mimic lsblk tree output. Parent disk row has TRAN=sata; child
    # partition row has empty TRAN but ROTA inherited from disk.
    out = "sda sata 0\n└─sda1   0\n"
    leaves = _parse_lsblk_devices(out)
    assert leaves == [("sda1", "sata", False)]


def test_parse_lsblk_devices_mdraid_two_sata_ssds():
    """md0 over two SATA SSDs — leaves are the two member partitions,
    each carrying the parent disk's transport."""
    out = "md0  0\n├─sda sata 0\n│ └─sda2  0\n└─sdb sata 0\n  └─sdb2  0\n"
    leaves = _parse_lsblk_devices(out)
    assert leaves == [("sda2", "sata", False), ("sdb2", "sata", False)]


# ----------------------------------------------------------------------
# pick_io_mode end-to-end via mocked findmnt + lsblk.
# ----------------------------------------------------------------------


def test_nvme_direct_picks_o_direct(fake_run):
    fake_run(
        {
            "findmnt": "/dev/nvme0n1p1 ext4\n",
            "lsblk": "nvme0n1 nvme 0\n",
        }
    )
    o_direct, rationale = pick_io_mode(Path("/mnt/nvme0-bench"))
    assert o_direct is True
    assert "nvme-ssd" in rationale
    assert "nvme0n1" in rationale


def test_sata_ssd_picks_buffered(fake_run):
    """A SATA SSD (rotational=0, TRAN=sata) gets buffered IO for the
    cold-read readahead win."""
    fake_run(
        {
            "findmnt": "/dev/sda1 ext4\n",
            "lsblk": "sda sata 0\n",
        }
    )
    o_direct, rationale = pick_io_mode(Path("/tmp"))
    assert o_direct is False
    assert "sata-ssd" in rationale
    assert "readahead" in rationale


def test_mdraid_two_sata_ssds_picks_buffered(fake_run):
    """mdraid over 2 SATA SSDs — buffered, mdraid walk identifies
    the underlying SATA transport on each member."""
    fake_run(
        {
            "findmnt": "/dev/md0 ext4\n",
            "lsblk": ("md0  0\n├─sda sata 0\n│ └─sda2  0\n└─sdb sata 0\n  └─sdb2  0\n"),
        }
    )
    o_direct, rationale = pick_io_mode(Path("/var/lib/kvd-long"))
    assert o_direct is False
    assert "sata-ssd" in rationale


def test_mdraid_mixed_nvme_and_sata_picks_buffered(fake_run):
    """Mixed-device array: a single SATA member pulls the whole array
    into buffered (worst-case wins)."""
    fake_run(
        {
            "findmnt": "/dev/md0 ext4\n",
            "lsblk": ("md0  0\n├─nvme0n1 nvme 0\n│ └─nvme0n1p1  0\n└─sdb sata 0\n  └─sdb2  0\n"),
        }
    )
    o_direct, rationale = pick_io_mode(Path("/var/lib/kvd-mixed"))
    assert o_direct is False
    # Worst-case rationale should mention the SATA member, not NVMe.
    assert "sata-ssd" in rationale


def test_nfs_high_nconnect_picks_buffered(fake_run, fake_proc_mounts):
    fake_run({"findmnt": "nfsserver:/export nfs4\n"})
    fake_proc_mounts("nfsserver:/export /mnt/nfs32 nfs4 rw,vers=4.1,nconnect=32,timeo=600 0 0\n")
    info = classify_storage(Path("/mnt/nfs32"))
    assert info.fs_type == "nfs4"
    assert info.nconnect == 32
    o_direct, rationale = pick_io_mode(Path("/mnt/nfs32"))
    assert o_direct is False
    assert "nconnect=32" in rationale


def test_nfs_low_nconnect_picks_buffered_and_reports_nconnect(fake_run, fake_proc_mounts):
    fake_run({"findmnt": "nfsserver:/export nfs4\n"})
    fake_proc_mounts("nfsserver:/export /mnt/nfs nfs4 rw,vers=4.1,nconnect=8,timeo=600 0 0\n")
    info = classify_storage(Path("/mnt/nfs"))
    assert info.nconnect == 8
    o_direct, rationale = pick_io_mode(Path("/mnt/nfs"))
    assert o_direct is False
    assert "nconnect=8" in rationale


# ----------------------------------------------------------------------
# rsize/wsize parsing + low-wsize warning
# ----------------------------------------------------------------------


def test_nfs_rsize_wsize_parsed(fake_run, fake_proc_mounts):
    """1 MB wsize is the kernel default and is what we want to warn about."""
    fake_run({"findmnt": "nfsserver:/export nfs\n"})
    fake_proc_mounts(
        "nfsserver:/export /mnt/nfs nfs rw,vers=3,rsize=1048576,wsize=1048576,"
        "nconnect=8,timeo=600 0 0\n"
    )
    info = classify_storage(Path("/mnt/nfs"))
    assert info.rsize_bytes == 1048576
    assert info.wsize_bytes == 1048576


def test_nfs_low_wsize_emits_warning(fake_run, fake_proc_mounts):
    """The wsize<8MB warning must fire on the kernel-default 1 MB mount."""
    fake_run({"findmnt": "nfsserver:/export nfs\n"})
    fake_proc_mounts(
        "nfsserver:/export /mnt/nfs nfs rw,vers=3,rsize=1048576,wsize=1048576,"
        "nconnect=8,timeo=600 0 0\n"
    )
    info = classify_storage(Path("/mnt/nfs"))
    matching = [w for w in info.warnings if "wsize=1024KB" in w]
    assert matching, f"expected wsize warning, got: {info.warnings}"


def test_nfs_recommended_wsize_no_warning(fake_run, fake_proc_mounts):
    """At wsize=16MB (the recommended value) the warning does NOT fire."""
    fake_run({"findmnt": "nfsserver:/export nfs\n"})
    fake_proc_mounts(
        "nfsserver:/export /mnt/nfs nfs rw,vers=3,rsize=16777216,wsize=16777216,"
        "nconnect=32,timeo=600 0 0\n"
    )
    info = classify_storage(Path("/mnt/nfs"))
    assert info.rsize_bytes == 16777216
    assert info.wsize_bytes == 16777216
    no_wsize_warns = [w for w in info.warnings if "wsize" in w]
    assert not no_wsize_warns, f"wsize warning should not fire, got: {info.warnings}"


def test_nfs_threshold_boundary(fake_run, fake_proc_mounts):
    """At exactly the threshold (8MB), no warning. Below (4MB), warning fires."""
    # At threshold
    fake_run({"findmnt": "nfsserver:/export nfs\n"})
    fake_proc_mounts(
        "nfsserver:/export /mnt/nfs nfs rw,vers=3,rsize=8388608,wsize=8388608,"
        "nconnect=8,timeo=600 0 0\n"
    )
    info_at = classify_storage(Path("/mnt/nfs"))
    assert not [w for w in info_at.warnings if "wsize" in w]

    # Below threshold
    fake_proc_mounts(
        "nfsserver:/export /mnt/nfs nfs rw,vers=3,rsize=4194304,wsize=4194304,"
        "nconnect=8,timeo=600 0 0\n"
    )
    info_below = classify_storage(Path("/mnt/nfs"))
    assert [w for w in info_below.warnings if "wsize" in w]


def test_nfs_no_rsize_wsize_in_opts(fake_run, fake_proc_mounts):
    """When rsize/wsize aren't in the mount opts (kernel uses default),
    parser returns None and no warning fires — we can't claim what the
    actual rsize is without seeing it."""
    fake_run({"findmnt": "nfsserver:/export nfs\n"})
    fake_proc_mounts("nfsserver:/export /mnt/nfs nfs rw,vers=3,nconnect=8,timeo=600 0 0\n")
    info = classify_storage(Path("/mnt/nfs"))
    assert info.rsize_bytes is None
    assert info.wsize_bytes is None
    assert not [w for w in info.warnings if "wsize" in w]


def test_tmpfs_picks_buffered_rationale_unsupported(fake_run):
    fake_run({"findmnt": "tmpfs tmpfs\n"})
    info = classify_storage(Path("/dev/shm/whatever"))
    assert info.fs_type == "tmpfs"
    o_direct, rationale = pick_io_mode(Path("/dev/shm/whatever"))
    assert o_direct is False
    assert "O_DIRECT unsupported" in rationale


def test_rotational_hdd_picks_buffered_mentions_readahead(fake_run):
    fake_run(
        {
            "findmnt": "/dev/sdc1 ext4\n",
            "lsblk": "sdc sata 1\n",
        }
    )
    o_direct, rationale = pick_io_mode(Path("/mnt/spinning-rust"))
    assert o_direct is False
    assert "rotational" in rationale.lower() or "hdd" in rationale.lower()
    assert "readahead" in rationale.lower()


def test_iscsi_picks_buffered(fake_run):
    fake_run(
        {
            "findmnt": "/dev/sdd1 ext4\n",
            "lsblk": "sdd iscsi 0\n",
        }
    )
    o_direct, rationale = pick_io_mode(Path("/mnt/iscsi-vol"))
    assert o_direct is False
    assert "iscsi" in rationale.lower() or "san" in rationale.lower()


def test_unknown_transport_picks_buffered_with_warn(fake_run, caplog):
    fake_run(
        {
            "findmnt": "/dev/loop0 ext4\n",
            "lsblk": "loop0  0\n",  # blank TRAN
        }
    )
    with caplog.at_level("WARNING", logger="infera.kvd.storage_classify"):
        o_direct, rationale = pick_io_mode(Path("/mnt/loop"))
    assert o_direct is False
    assert "unknown transport" in rationale.lower()
    # A WARN was logged.
    assert any("unknown transport" in r.message.lower() for r in caplog.records)


def test_sas_ssd_picks_o_direct(fake_run):
    """Enterprise SAS SSD — treated like NVMe at the kernel block layer."""
    fake_run(
        {
            "findmnt": "/dev/sde1 xfs\n",
            "lsblk": "sde sas 0\n",
        }
    )
    o_direct, rationale = pick_io_mode(Path("/mnt/enterprise"))
    assert o_direct is True
    assert "sas-ssd" in rationale


def test_bind_mount_source_strips_subpath(monkeypatch):
    """A bind mount makes findmnt print SOURCE as ``/dev/md0[/sub/path]``.

    That bracketed suffix is not part of the device name. Passing the raw
    string to lsblk gets ``not a block device``, so the probe silently
    degrades to buffered even on hardware that qualifies for O_DIRECT.

    Observed on a real deployment: kvd's L3 bind-mounted into a container
    reported ``devices = [(none)]``, ``rationale: unknown device,
    conservative buffered``, on an ext4-on-md0 mount.

    This test models the *real* lsblk contract — it errors on a bracketed
    argument — rather than a fake that accepts anything, which is what
    lets the bug through.
    """

    def run(cmd, timeout=2.0):
        class R:
            returncode = 0
            stdout = ""

        r = R()
        if cmd[0] == "findmnt":
            r.stdout = "/dev/nvme0n1p1[/kvd-long] ext4\n"
        elif cmd[0] == "lsblk":
            target = cmd[-1]
            if "[" in target:  # what real lsblk does
                r.returncode = 32
                r.stdout = ""
            else:
                r.stdout = "nvme0n1 nvme 0\n"
        return r

    monkeypatch.setattr(storage_classify, "_run", run)
    o_direct, rationale = pick_io_mode(Path("/kvd-long"))
    assert o_direct is True, f"bind mount misclassified: {rationale}"
    assert "nvme-ssd" in rationale


def test_bind_mount_classify_reports_clean_source(fake_run):
    """The bracketed subpath must not leak into the reported mount source
    either — operators read this field to sanity-check the probe."""
    fake_run(
        {
            "findmnt": "/dev/sda1[/some/where] ext4\n",
            "lsblk": "sda sata 0\n",
        }
    )
    info = classify_storage(Path("/some/where"))
    assert info.mount_source == "/dev/sda1"
    assert info.devices, "bind-mounted device should still be resolved"


def test_findmnt_missing_falls_back_to_buffered(monkeypatch, caplog):
    """When findmnt isn't installed (minimal containers), we don't
    raise — we just default to buffered and log a WARN. The classifier
    is best-effort."""

    def always_none(cmd, timeout=2.0):
        return None

    monkeypatch.setattr(storage_classify, "_run", always_none)
    with caplog.at_level("WARNING", logger="infera.kvd.storage_classify"):
        info = classify_storage(Path("/anything"))
    assert info.fs_type == "unknown"
    o_direct, rationale = pick_io_mode(Path("/anything"))
    assert o_direct is False
    assert "unknown" in rationale.lower() or "conservative" in rationale.lower()


def test_lsblk_missing_falls_back_to_buffered(fake_run, caplog):
    """findmnt succeeds (ext4 mount) but lsblk is missing → buffered + WARN.
    Conservative — without device data we can't justify O_DIRECT."""
    fake_run({"findmnt": "/dev/sda1 ext4\n"})  # lsblk omitted → _run returns None
    with caplog.at_level("WARNING", logger="infera.kvd.storage_classify"):
        o_direct, rationale = pick_io_mode(Path("/anything"))
    assert o_direct is False
    assert "unknown" in rationale.lower() or "conservative" in rationale.lower()


def test_pick_io_mode_never_raises(monkeypatch):
    """Defensive contract: pick_io_mode MUST NOT raise. Even if
    classify_storage blows up internally (it shouldn't), the caller
    gets (False, <something>) back."""

    def boom(path):
        raise RuntimeError("simulated probe failure")

    monkeypatch.setattr(storage_classify, "classify_storage", boom)
    o_direct, rationale = pick_io_mode(Path("/whatever"))
    assert o_direct is False
    assert "error" in rationale.lower() or "buffered" in rationale.lower()


# ----------------------------------------------------------------------
# format_decision — sanity-check the human-readable rendering.
# ----------------------------------------------------------------------


def test_format_decision_contains_key_fields(fake_run):
    fake_run(
        {
            "findmnt": "/dev/nvme0n1p1 ext4\n",
            "lsblk": "nvme0n1 nvme 0\n",
        }
    )
    out = storage_classify.format_decision(Path("/mnt/nvme0-bench"))
    assert "DIRECT" in out
    assert "/mnt/nvme0-bench" in out
    assert "nvme0n1" in out
    assert "rationale" in out
    assert "override" in out


def test_format_decision_nfs_includes_nconnect(fake_run, fake_proc_mounts):
    fake_run({"findmnt": "nfsserver:/export nfs4\n"})
    fake_proc_mounts("nfsserver:/export /mnt/nfs nfs4 rw,vers=4.1,nconnect=8 0 0\n")
    out = storage_classify.format_decision(Path("/mnt/nfs"))
    assert "BUFFERED" in out
    assert "nconnect" in out
    assert "8" in out


# ----------------------------------------------------------------------
# DeviceInfo / StorageInfo dataclass sanity.
# ----------------------------------------------------------------------


def test_storage_info_dataclass_defaults():
    info = StorageInfo(path=Path("/x"), mount_source="/dev/sda1", fs_type="ext4", nconnect=None)
    assert info.devices == []
    assert info.warnings == []


def test_device_info_fields():
    d = DeviceInfo(dev="nvme0n1", transport="nvme", rotational=False)
    assert d.dev == "nvme0n1"
    assert d.transport == "nvme"
    assert d.rotational is False


# ----------------------------------------------------------------------
# pick_workers_per_shard — companion of pick_io_mode.
# ----------------------------------------------------------------------
#
# All cases use n_shards=1 unless otherwise stated, so the CPU guardrail
# (max(2, cpu_count() // n_shards)) won't trip on any realistic CI box
# — single-shard cap is the entire cpu count, which dwarfs 8.


def test_workers_nvme_picks_8(fake_run):
    fake_run(
        {
            "findmnt": "/dev/nvme0n1p1 ext4\n",
            "lsblk": "nvme0n1 nvme 0\n",
        }
    )
    workers, rationale = pick_workers_per_shard(Path("/mnt/nvme0-bench"))
    assert workers == 8
    assert "nvme-ssd" in rationale
    assert "queue depth" in rationale


def test_workers_sas_ssd_picks_8(fake_run):
    fake_run(
        {
            "findmnt": "/dev/sde1 xfs\n",
            "lsblk": "sde sas 0\n",
        }
    )
    workers, rationale = pick_workers_per_shard(Path("/mnt/enterprise"))
    assert workers == 8
    assert "sas-ssd" in rationale


def test_workers_sata_ssd_picks_4(fake_run):
    fake_run(
        {
            "findmnt": "/dev/sda1 ext4\n",
            "lsblk": "sda sata 0\n",
        }
    )
    workers, rationale = pick_workers_per_shard(Path("/tmp"))
    assert workers == 4
    assert "sata-ssd" in rationale


def test_workers_sata_raid1_picks_4(fake_run):
    """mdraid over 2 SATA SSDs — still SATA-dominated, picks 4."""
    fake_run(
        {
            "findmnt": "/dev/md0 ext4\n",
            "lsblk": ("md0  0\n├─sda sata 0\n│ └─sda2  0\n└─sdb sata 0\n  └─sdb2  0\n"),
        }
    )
    workers, rationale = pick_workers_per_shard(Path("/var/lib/kvd-long"))
    assert workers == 4
    assert "sata-ssd" in rationale


def test_workers_nfs_high_nconnect_picks_8(fake_run, fake_proc_mounts, monkeypatch):
    """nconnect=32 → min(8, 32//4)=8. Pin a generous cpu_count so the
    CPU guardrail doesn't clamp it on tiny CI boxes."""
    monkeypatch.setattr(storage_classify.os, "cpu_count", lambda: 64)
    fake_run({"findmnt": "nfsserver:/export nfs4\n"})
    fake_proc_mounts("nfsserver:/export /mnt/nfs32 nfs4 rw,vers=4.1,nconnect=32 0 0\n")
    workers, rationale = pick_workers_per_shard(Path("/mnt/nfs32"))
    assert workers == 8
    assert "nconnect=32" in rationale


def test_workers_nfs_mid_nconnect_picks_4(fake_run, fake_proc_mounts):
    fake_run({"findmnt": "nfsserver:/export nfs4\n"})
    fake_proc_mounts("nfsserver:/export /mnt/nfs16 nfs4 rw,vers=4.1,nconnect=16 0 0\n")
    workers, rationale = pick_workers_per_shard(Path("/mnt/nfs16"))
    assert workers == 4
    assert "nconnect=16" in rationale
    assert "mid-tier" in rationale


def test_workers_nfs_low_nconnect_picks_2(fake_run, fake_proc_mounts):
    fake_run({"findmnt": "nfsserver:/export nfs4\n"})
    fake_proc_mounts("nfsserver:/export /mnt/nfs nfs4 rw,vers=4.1,nconnect=8 0 0\n")
    workers, rationale = pick_workers_per_shard(Path("/mnt/nfs"))
    assert workers == 2
    assert "nconnect=8" in rationale
    assert "low" in rationale


def test_workers_hdd_picks_2(fake_run):
    fake_run(
        {
            "findmnt": "/dev/sdc1 ext4\n",
            "lsblk": "sdc sata 1\n",
        }
    )
    workers, rationale = pick_workers_per_shard(Path("/mnt/spinning-rust"))
    assert workers == 2
    assert "rotational" in rationale.lower() or "hdd" in rationale.lower()
    assert "seek" in rationale.lower()


def test_workers_tmpfs_picks_8(fake_run):
    fake_run({"findmnt": "tmpfs tmpfs\n"})
    workers, rationale = pick_workers_per_shard(Path("/dev/shm/whatever"))
    assert workers == 8
    assert "tmpfs" in rationale.lower() or "ram" in rationale.lower()


def test_workers_iscsi_picks_4(fake_run):
    fake_run(
        {
            "findmnt": "/dev/sdd1 ext4\n",
            "lsblk": "sdd iscsi 0\n",
        }
    )
    workers, rationale = pick_workers_per_shard(Path("/mnt/iscsi-vol"))
    assert workers == 4
    assert "iscsi" in rationale.lower() or "san" in rationale.lower()


def test_workers_unknown_transport_picks_4_and_warns(fake_run, caplog):
    fake_run(
        {
            "findmnt": "/dev/loop0 ext4\n",
            "lsblk": "loop0  0\n",  # blank TRAN
        }
    )
    with caplog.at_level("WARNING", logger="infera.kvd.storage_classify"):
        workers, rationale = pick_workers_per_shard(Path("/mnt/loop"))
    assert workers == 4
    assert "unknown transport" in rationale.lower() or "conservative" in rationale.lower()
    # A WARN was emitted from the picker itself.
    assert any(
        "workers_per_shard" in r.message or "unknown transport" in r.message for r in caplog.records
    )


def test_workers_no_devices_picks_4(fake_run):
    """findmnt OK but lsblk returns nothing (e.g. exotic source) →
    conservative 4 + no crash."""
    fake_run({"findmnt": "/dev/sda1 ext4\n"})  # lsblk omitted → None
    workers, rationale = pick_workers_per_shard(Path("/anything"))
    assert workers == 4
    assert "conservative" in rationale.lower()


def test_workers_cpu_guardrail_clamps_high_pick(fake_run, monkeypatch):
    """8-shard config on a 4-core box. NVMe would normally pick 8;
    guardrail caps it at max(2, 4//8) = 2."""
    monkeypatch.setattr(storage_classify.os, "cpu_count", lambda: 4)
    fake_run(
        {
            "findmnt": "/dev/nvme0n1p1 ext4\n",
            "lsblk": "nvme0n1 nvme 0\n",
        }
    )
    workers, rationale = pick_workers_per_shard(Path("/mnt/nvme0-bench"), n_shards=8)
    assert workers == 2
    assert "capped" in rationale
    assert "n_shards=8" in rationale


def test_workers_cpu_guardrail_does_not_floor_below_2(fake_run, monkeypatch):
    """Even on a 1-core box with many shards, the floor is 2 — we
    never collapse to 1 worker (which defeats the fan-out)."""
    monkeypatch.setattr(storage_classify.os, "cpu_count", lambda: 1)
    fake_run(
        {
            "findmnt": "/dev/nvme0n1p1 ext4\n",
            "lsblk": "nvme0n1 nvme 0\n",
        }
    )
    workers, _ = pick_workers_per_shard(Path("/mnt/nvme0-bench"), n_shards=16)
    assert workers == 2


def test_workers_cpu_count_none_falls_back_to_4(fake_run, monkeypatch):
    """os.cpu_count() can return None on exotic platforms. The picker
    must still produce a positive int."""
    monkeypatch.setattr(storage_classify.os, "cpu_count", lambda: None)
    fake_run(
        {
            "findmnt": "/dev/nvme0n1p1 ext4\n",
            "lsblk": "nvme0n1 nvme 0\n",
        }
    )
    workers, _ = pick_workers_per_shard(Path("/mnt/nvme0-bench"), n_shards=2)
    # cpu fallback 4, n_shards=2 → cap=max(2,2)=2; pick was 8 → 2.
    assert workers == 2


def test_workers_never_raises(monkeypatch):
    """Defensive contract: pick_workers_per_shard MUST NOT raise. Even
    if classify_storage blows up, caller gets a positive int back."""

    def boom(path):
        raise RuntimeError("simulated probe failure")

    monkeypatch.setattr(storage_classify, "classify_storage", boom)
    workers, rationale = pick_workers_per_shard(Path("/whatever"))
    assert workers > 0
    assert "error" in rationale.lower() or "conservative" in rationale.lower()


def test_workers_findmnt_missing_falls_back_to_4(monkeypatch):
    """When findmnt isn't installed (minimal containers), pick_workers
    defaults to 4 — same conservative posture as pick_io_mode."""

    def always_none(cmd, timeout=2.0):
        return None

    monkeypatch.setattr(storage_classify, "_run", always_none)
    # Pin cpu_count high so the CPU guardrail doesn't clamp the
    # fallback-4 on low-core CI runners (this test doesn't use fake_run).
    monkeypatch.setattr("os.cpu_count", lambda: 64)
    workers, rationale = pick_workers_per_shard(Path("/anything"))
    assert workers == 4
    assert "conservative" in rationale.lower() or "no device" in rationale.lower()


# ----------------------------------------------------------------------
# format_workers_decision — sanity-check the human-readable rendering.
# ----------------------------------------------------------------------


def test_format_workers_decision_contains_key_fields(fake_run):
    fake_run(
        {
            "findmnt": "/dev/nvme0n1p1 ext4\n",
            "lsblk": "nvme0n1 nvme 0\n",
        }
    )
    info = classify_storage(Path("/mnt/nvme0-bench"))
    workers, rationale = pick_workers_per_shard(Path("/mnt/nvme0-bench"))
    out = format_workers_decision(info, workers, rationale, n_shards=1)
    assert "workers_per_shard" in out
    assert "/mnt/nvme0-bench" in out
    assert "nvme0n1" in out
    assert "rationale" in out
    assert "override" in out
    assert "INFERA_KVD_WORKERS_PER_SHARD" in out


def test_format_workers_decision_nfs_includes_nconnect(fake_run, fake_proc_mounts):
    fake_run({"findmnt": "nfsserver:/export nfs4\n"})
    fake_proc_mounts("nfsserver:/export /mnt/nfs nfs4 rw,vers=4.1,nconnect=8 0 0\n")
    info = classify_storage(Path("/mnt/nfs"))
    workers, rationale = pick_workers_per_shard(Path("/mnt/nfs"))
    out = format_workers_decision(info, workers, rationale)
    assert "nconnect" in out
    assert "8" in out


# ----------------------------------------------------------------------
# sysfs device walk — the path that covers what lsblk cannot resolve.
#
# Every case here builds a real directory tree shaped like sysfs and points
# the module at it, so the walk is exercised end to end (symlinks, slaves/
# recursion, partition→disk resolution) without depending on whatever disks
# the test machine happens to have.
# ----------------------------------------------------------------------


@pytest.fixture
def fake_sysfs(monkeypatch, tmp_path):
    """Build a sysfs fixture tree and aim the module's walk at it.

    ``devices`` maps each block device name to its shape:

        rotational   "0" / "1"     ``queue/rotational``; whole disks only,
                                   since that is where the kernel puts it
        slaves       [names]       the devices this one is built on
        subsystem    "nvme"/...    the bus its backing device sits on
        devpath      "a/b/c"       where under ``/sys/devices`` that lands —
                                   this is what carries ``/hostN/`` and ``/usb``
        sas_address  True          the attribute that marks a SAS disk
        parent_disk  "sda"         makes this device a partition of that disk

    ``scsi_hosts`` maps a host name to its ``proc_name``, the attribute
    util-linux reads to tell iSCSI and SATA apart.

    Returns the path to probe: ``tmp_path`` itself, registered in the fake
    ``/sys/dev/block`` under its own real ``st_dev`` so ``stat()`` lands on
    the fixture topology on any machine.
    """

    def build(devices: dict, target: str, scsi_hosts: dict | None = None) -> Path:
        root = tmp_path / "sys"
        blk = root / "class" / "block"
        blk.mkdir(parents=True)
        (root / "dev" / "block").mkdir(parents=True)
        devices_dir = root / "devices"

        def node_dir(name: str, spec: dict) -> Path:
            parent = spec.get("parent_disk")
            if not parent:
                d = blk / name
                d.mkdir(parents=True, exist_ok=True)
                return d
            # A partition lives inside its disk's directory, and /sys/class
            # only links to it — which is exactly what the walk relies on to
            # find the disk that carries the queue/ and device/ attributes.
            d = devices_dir / "block" / parent / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "partition").write_text("1\n")
            if not (blk / name).exists():
                (blk / name).symlink_to(d)
            return d

        # Whole disks first: a partition's entry has to be able to point into
        # its parent's directory.
        ordered = sorted(devices.items(), key=lambda kv: bool(kv[1].get("parent_disk")))
        for name, spec in ordered:
            d = node_dir(name, spec)
            if "rotational" in spec:
                (d / "queue").mkdir(exist_ok=True)
                (d / "queue" / "rotational").write_text(spec["rotational"] + "\n")
            for child in spec.get("slaves", []):
                (d / "slaves").mkdir(exist_ok=True)
                (d / "slaves" / child).mkdir(exist_ok=True)
            subsystem = spec.get("subsystem")
            if subsystem:
                phys = devices_dir / spec.get("devpath", f"pci0000:00/{name}")
                phys.mkdir(parents=True, exist_ok=True)
                bus = root / "bus" / subsystem
                bus.mkdir(parents=True, exist_ok=True)
                if not (phys / "subsystem").exists():
                    (phys / "subsystem").symlink_to(bus)
                if spec.get("sas_address"):
                    (phys / "sas_address").write_text("0x5000c500a1b2c3d4\n")
                if not (d / "device").exists():
                    (d / "device").symlink_to(phys)

        for host, proc_name in (scsi_hosts or {}).items():
            hd = root / "class" / "scsi_host" / host
            hd.mkdir(parents=True, exist_ok=True)
            (hd / "proc_name").write_text(proc_name + "\n")

        probe = tmp_path / "probe"
        probe.mkdir(exist_ok=True)
        st = probe.stat()
        devno = f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}"
        (root / "dev" / "block" / devno).symlink_to(blk / target)

        monkeypatch.setattr(storage_classify, "_SYSFS_ROOT", str(root))
        return probe

    return build


def test_sysfs_walks_lvm_stack_down_to_nvme_members(fake_sysfs):
    """The case lsblk cannot do: the logical volume's name is not a path,
    so only a major:minor lookup reaches the members."""
    probe = fake_sysfs(
        {
            "dm-8": {"slaves": ["dm-0", "dm-1"], "rotational": "0"},
            "dm-0": {"slaves": ["nvme0n1"], "rotational": "0"},
            "dm-1": {"slaves": ["nvme1n1"], "rotational": "0"},
            "nvme0n1": {"subsystem": "nvme", "rotational": "0"},
            "nvme1n1": {"subsystem": "nvme", "rotational": "0"},
        },
        target="dm-8",
    )
    devices = storage_classify._sysfs_for_path(probe)
    assert [d.dev for d in devices] == ["nvme0n1", "nvme1n1"]
    assert all(d.transport == "nvme" and not d.rotational for d in devices)


def test_sysfs_walks_md_stack(fake_sysfs):
    probe = fake_sysfs(
        {
            "md0": {"slaves": ["sda", "sdb"], "rotational": "0"},
            "sda": {"subsystem": "scsi", "devpath": "pci0000:00/ata1/host1/sda", "rotational": "0"},
            "sdb": {"subsystem": "scsi", "devpath": "pci0000:00/ata2/host2/sdb", "rotational": "0"},
        },
        target="md0",
        scsi_hosts={"host1": "ahci", "host2": "ahci"},
    )
    devices = storage_classify._sysfs_for_path(probe)
    assert [d.dev for d in devices] == ["sda", "sdb"]
    assert all(d.transport == "sata" for d in devices)


def test_sysfs_dedupes_a_leaf_reached_through_two_parents(fake_sysfs):
    """An LVM RAID mirrors each leg through its own rimage, so the same
    physical disk can be reachable more than once."""
    probe = fake_sysfs(
        {
            "dm-9": {"slaves": ["dm-2", "dm-3"], "rotational": "0"},
            "dm-2": {"slaves": ["nvme0n1"], "rotational": "0"},
            "dm-3": {"slaves": ["nvme0n1"], "rotational": "0"},
            "nvme0n1": {"subsystem": "nvme", "rotational": "0"},
        },
        target="dm-9",
    )
    assert [d.dev for d in storage_classify._sysfs_for_path(probe)] == ["nvme0n1"]


def test_sysfs_resolves_a_partition_to_its_parent_disk(fake_sysfs):
    """Partitions carry neither queue/ nor device/ — both belong to the disk."""
    probe = fake_sysfs(
        {
            "sda": {
                "subsystem": "scsi",
                "devpath": "platform/host0/session1/target0:0:0/0:0:0:1",
                "rotational": "1",
            },
            "sda1": {"parent_disk": "sda"},
        },
        target="sda1",
        scsi_hosts={"host0": "iscsi_tcp"},
    )
    devices = storage_classify._sysfs_for_path(probe)
    assert [(d.dev, d.transport, d.rotational) for d in devices] == [("sda1", "iscsi", True)]


def test_sysfs_reads_sas_from_the_device_attribute(fake_sysfs):
    probe = fake_sysfs(
        {
            "sdc": {
                "subsystem": "scsi",
                "devpath": "pci0000:00/host3/sdc",
                "sas_address": True,
                "rotational": "0",
            }
        },
        target="sdc",
        scsi_hosts={"host3": "mpt3sas"},
    )
    assert storage_classify._sysfs_for_path(probe)[0].transport == "sas"


def test_sysfs_does_not_name_a_bus_from_machine_wide_state(fake_sysfs, tmp_path):
    """A RAID card presenting a logical volume supplies no per-device signal,
    so it has to stay unknown. Answering from something machine-wide instead —
    "this box has an fc_host, so call it fc" — labels a local disk a SAN.
    """
    probe = fake_sysfs(
        {
            "sdf": {
                "subsystem": "scsi",
                "devpath": "pci0000:00/host2/target2:2:0/2:2:0:0",
                "rotational": "0",
            }
        },
        target="sdf",
        scsi_hosts={"host2": "megaraid_sas"},
    )
    # An FC HBA elsewhere in the box, on a host this disk has nothing to do with.
    (tmp_path / "sys" / "class" / "fc_host" / "host9").mkdir(parents=True)
    assert storage_classify._sysfs_for_path(probe)[0].transport == ""


def test_sysfs_names_fc_when_the_hba_is_this_disk_s_own_host(fake_sysfs, tmp_path):
    """The other half of the pair: a real SAN is worth naming, because
    'SAN fc → buffered' reads a great deal better in the startup log than an
    unknown transport, which also logs a WARN."""
    probe = fake_sysfs(
        {
            "sdg": {
                "subsystem": "scsi",
                "devpath": "pci0000:00/host4/rport-4:0-0/target4:0:0/4:0:0:0",
                "rotational": "0",
            }
        },
        target="sdg",
        scsi_hosts={"host4": "lpfc"},
    )
    (tmp_path / "sys" / "class" / "fc_host" / "host4").mkdir(parents=True)
    assert storage_classify._sysfs_for_path(probe)[0].transport == "fc"


def test_sysfs_returns_nothing_when_the_devno_is_not_registered(fake_sysfs, tmp_path):
    """A filesystem with no block device behind it — NFS reports a synthetic
    st_dev that has no /sys/dev/block entry."""
    fake_sysfs({"nvme0n1": {"subsystem": "nvme", "rotational": "0"}}, target="nvme0n1")
    monkey_root = Path(str(tmp_path / "sys" / "dev" / "block"))
    for entry in monkey_root.iterdir():
        entry.unlink()
    assert storage_classify._sysfs_for_path(tmp_path / "probe") == []


def test_sysfs_leaves_transport_blank_for_an_unrecognised_bus(fake_sysfs):
    """Unknown is still a useful answer — it keeps the conservative default
    rather than guessing O_DIRECT onto something that would hate it."""
    probe = fake_sysfs(
        {"xvda": {"subsystem": "xen", "rotational": "0"}},
        target="xvda",
    )
    devices = storage_classify._sysfs_for_path(probe)
    assert [(d.dev, d.transport) for d in devices] == [("xvda", "")]


# --- and the same thing end to end, through pick_io_mode ---------------


def _lvm_on_nvme(fake_sysfs):
    return fake_sysfs(
        {
            "dm-8": {"slaves": ["nvme0n1"], "rotational": "0"},
            "nvme0n1": {"subsystem": "nvme", "rotational": "0"},
        },
        target="dm-8",
    )


def test_lsblk_failing_outright_is_recovered_by_sysfs(fake_run, fake_sysfs):
    """Inside a container /dev holds only what --device put there, so lsblk
    cannot open the node at all. Before the sysfs fallback this mount was
    classified 'unknown device' and ran buffered on NVMe."""
    probe = _lvm_on_nvme(fake_sysfs)
    fake_run(
        {
            "findmnt": "/dev/mapper/nvme_vg-nvme_lv xfs\n",
            "lsblk": ("lsblk: not a block device\n", 32),
        }
    )
    o_direct, rationale = pick_io_mode(probe)
    assert o_direct is True
    assert "nvme" in rationale


def test_lsblk_answering_without_a_transport_is_recovered_by_sysfs(fake_run, fake_sysfs):
    """On the host lsblk does resolve the LV, but the --inverse walk that
    would supply the transport is handed /dev/<vg>-<lv>, which is not a
    path that exists. The row comes back bare."""
    probe = _lvm_on_nvme(fake_sysfs)
    fake_run(
        {
            "findmnt": "/dev/mapper/nvme_vg-nvme_lv xfs\n",
            "lsblk": "nvme_vg-nvme_lv  0\n",  # blank TRAN, and no inverse answer
        }
    )
    info = classify_storage(probe)
    assert [d.dev for d in info.devices] == ["nvme0n1"]
    assert pick_io_mode(probe)[0] is True


def test_a_complete_lsblk_answer_is_not_second_guessed(fake_run, fake_sysfs):
    """sysfs is a fallback, not an override: when lsblk names a transport it
    wins, even where the fixture topology would say something else."""
    probe = _lvm_on_nvme(fake_sysfs)  # sysfs here would say nvme → O_DIRECT
    fake_run(
        {
            "findmnt": "/dev/sdb1 ext4\n",
            "lsblk": "sdb sata 0\n",
        }
    )
    o_direct, rationale = pick_io_mode(probe)
    assert o_direct is False
    assert "sata" in rationale


def test_both_probes_failing_still_lands_on_conservative_buffered(fake_run, monkeypatch, tmp_path):
    """No lsblk answer and no sysfs either — the posture that protects an
    HDD or a SAN from having O_DIRECT guessed onto it."""
    fake_run(
        {
            "findmnt": "/dev/mapper/vg-lv xfs\n",
            "lsblk": ("", 32),
        }
    )
    monkeypatch.setattr(storage_classify, "_SYSFS_ROOT", str(tmp_path / "no-sysfs-here"))
    info = classify_storage(tmp_path)
    assert info.devices == []
    assert any("sysfs" in w for w in info.warnings)
    assert pick_io_mode(tmp_path) == (False, "unknown device, conservative buffered")
