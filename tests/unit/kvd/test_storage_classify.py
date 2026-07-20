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
