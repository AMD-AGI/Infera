# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Criteria 15 and 16 — sync runs once, is scoped to the task, and skips the
playground."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from env_mgr.fs.zone import Zone
from env_mgr.sync import Direction, conflicts, sync
from task_graph.ids import TaskId

pytestmark = pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync is not installed")


@pytest.fixture
def sides(tmp_path: Path) -> tuple[str, str, Zone]:
    """A local root with two tasks, and an empty remote root.

    Two tasks, because the scoping claim is only meaningful when there is
    something the sync must **not** touch.
    """
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    zone = local / "task.a"
    (zone / "handoffs").mkdir(parents=True)
    (zone / "handoffs" / "in.txt").write_text("input")
    (zone / "playground").mkdir()
    (zone / "playground" / "scratch.txt").write_text("local scratch")
    other = local / "task.b"
    other.mkdir()
    (other / "theirs.txt").write_text("another task's material")
    remote.mkdir()
    return str(local), str(remote), Zone(TaskId.new(), 0, str(zone.resolve()))


# ---------------------------------------------------------- criterion 15


def test_sync_once_at_start(sides: tuple[str, str, Zone]) -> None:
    """A one-time job, not a reconciliation loop: nothing syncs afterwards
    without a call."""
    local, remote, zone = sides
    sync(zone, {local: remote}, direction=Direction.LOCAL_TO_REMOTE)
    mirrored = Path(remote) / "task.a" / "handoffs" / "in.txt"
    assert mirrored.read_text() == "input"

    (Path(zone.root) / "handoffs" / "later.txt").write_text("written after")
    assert not (Path(remote) / "task.a" / "handoffs" / "later.txt").exists()


def test_destination_matches_source(sides: tuple[str, str, Zone]) -> None:
    """*"Identical"* has a direction and the spec does not say which.

    ``rsync -a --delete`` makes two trees equal by **destroying** everything the
    destination had that the source did not, and there is no symmetric mode. So
    the destination is made identical to the source, and the caller names which
    is which — `Direction` is a required argument.
    """
    local, remote, zone = sides
    stale = Path(remote) / "task.a" / "handoffs" / "stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("remote-only, and about to be destroyed")
    sync(zone, {local: remote}, direction=Direction.LOCAL_TO_REMOTE)
    assert not stale.exists()


def test_direction_is_required() -> None:
    with pytest.raises(TypeError):
        sync(Zone(TaskId.new(), 0, "/tmp"), {})  # type: ignore[call-arg]


def test_scoped_to_task_not_root(sides: tuple[str, str, Zone]) -> None:
    """Syncing a root would move material belonging to tasks that have nothing
    to do with this one. Measured, the *time* argument is only 2× at 20 tasks ×
    50 files, because rsync's fixed startup dominates — **the argument that
    holds at this size is correctness.**"""
    local, remote, zone = sides
    sync(zone, {local: remote}, direction=Direction.LOCAL_TO_REMOTE)
    assert not (Path(remote) / "task.b").exists()


def test_a_zone_with_no_mapping_is_an_error(tmp_path: Path) -> None:
    zone = Zone(TaskId.new(), 0, str(tmp_path))
    with pytest.raises(KeyError, match="no mapping covers"):
        sync(zone, {"/elsewhere": "/remote"}, direction=Direction.LOCAL_TO_REMOTE)


# ---------------------------------------------------------- criterion 16


def test_playground_not_synced(sides: tuple[str, str, Zone]) -> None:
    """Their contents can be completely different, and that is correct: each
    side's scratch is about the work happening on that side."""
    local, remote, zone = sides
    sync(zone, {local: remote}, direction=Direction.LOCAL_TO_REMOTE)
    assert not (Path(remote) / "task.a" / "playground" / "scratch.txt").exists()
    assert (Path(zone.root) / "playground" / "scratch.txt").read_text() == "local scratch"


def test_playground_dir_created_empty(sides: tuple[str, str, Zone]) -> None:
    """``--exclude playground/`` omits the *contents* but still creates the
    directory on the far side — consistent with the remote having its own
    playground, and it is what the assertion must actually say."""
    local, remote, zone = sides
    sync(zone, {local: remote}, direction=Direction.LOCAL_TO_REMOTE)
    far = Path(remote) / "task.a" / "playground"
    assert far.is_dir()
    assert list(far.iterdir()) == []


# --------------------------------------------- conflict detection (design §9.3)


def test_conflict_detected_before_anything_is_written(sides: tuple[str, str, Zone]) -> None:
    """No ``rsync`` flag reports that both sides changed: ``-a`` and
    ``--checksum`` silently discard one, and ``--update`` guesses by mtime.
    Detection is a pre-pass, and refusing converts silent data loss into a
    stopped task."""
    local, remote, zone = sides
    far = Path(remote) / "task.a" / "handoffs"
    far.mkdir(parents=True)
    (far / "in.txt").write_text("REMOTE edit")

    assert conflicts(zone.root, str(far.parent)) == ("handoffs/in.txt",)
    report = sync(zone, {local: remote}, direction=Direction.LOCAL_TO_REMOTE)
    assert report.conflicts == ("handoffs/in.txt",)


def test_a_playground_difference_is_not_a_conflict(sides: tuple[str, str, Zone]) -> None:
    """It is never synced, so the two sides differing there is the design."""
    local, remote, zone = sides
    far = Path(remote) / "task.a" / "playground"
    far.mkdir(parents=True)
    (far / "scratch.txt").write_text("remote scratch")
    assert conflicts(zone.root, str(far.parent)) == ()


# ------------------------------------------------- crossing a host boundary (R1a)
#
# `sync`'s copy semantics stay in `sync`: `--delete` is the whole of spec §5.3's
# "made identical" and `--exclude=playground/**` is criterion 16. `Connection.push`
# has neither, so routing the copy through it would have dropped both silently and
# every test above would still have passed. What a transport supplies is only how
# to *reach* the far side, which is why the seam is named `rsync_spec`.


class FakeTransport:
    """A `SyncTransport` that reaches a far side which is not there.

    Structural, not a subclass: `SyncTransport` is a `Protocol`, and a stub that
    had to inherit would not be testing the same thing the production classes
    satisfy.
    """

    def __init__(self, *, far_exists: bool) -> None:
        self.far_exists = far_exists
        self.commands: list[list[str]] = []

    def run(self, argv, *, cwd=None, timeout=None):  # noqa: ANN001, ANN201
        import subprocess

        self.commands.append(list(argv))
        code = 0 if (argv[0] == "test" and self.far_exists) else (1 if argv[0] == "test" else 0)
        return subprocess.CompletedProcess(list(argv), code, "", "")

    def push(self, local: str, remote: str):  # noqa: ANN201
        raise AssertionError("sync must not route the copy through push: it loses --delete")

    def pull(self, remote: str, local: str):  # noqa: ANN201
        raise AssertionError("sync must not route the copy through pull: it loses --delete")

    def rsync_spec(self):  # noqa: ANN201
        return ("ssh", "-o", "BatchMode=yes"), "somehost:"


def test_a_transport_puts_the_rsh_and_the_prefix_on_the_command(
    sides: tuple[str, str, Zone], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The far side is addressed, and the copy's own flags are still `sync`'s.

    Nothing is executed: the rsync argv is captured. Its control is
    `test_no_transport_addresses_neither_host` below, which must produce two
    plain local paths and no `-e`.
    """
    import subprocess

    import env_mgr.sync as sync_mod

    local, remote, zone = sides
    seen: list[list[str]] = []

    def fake_run(argv, **kwargs):  # noqa: ANN001, ANN202
        seen.append(list(argv))
        return subprocess.CompletedProcess(list(argv), 0, "", "")

    monkeypatch.setattr(sync_mod.subprocess, "run", fake_run)
    conn = FakeTransport(far_exists=False)
    sync(
        zone,
        {local: remote},
        direction=Direction.LOCAL_TO_REMOTE,
        transports={local: conn},
    )
    rsync_argv = next(a for a in seen if any("rsync" in part for part in a[:1]))
    assert "-e" in rsync_argv
    assert rsync_argv[rsync_argv.index("-e") + 1] == "ssh -o BatchMode=yes"
    assert rsync_argv[-1].startswith("somehost:"), rsync_argv[-1]
    assert not rsync_argv[-2].startswith("somehost:"), "the source is the local end"
    # The semantics did not move into the transport.
    assert "--delete" in rsync_argv
    assert any(part.startswith("--exclude=playground") for part in rsync_argv)
    # The far side's directories are made over the connection, not with os.makedirs.
    assert ["mkdir", "-p"] == conn.commands[-1][:2]


def test_no_transport_addresses_neither_host(
    sides: tuple[str, str, Zone], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control, and the configuration every run before R1 used."""
    import subprocess

    import env_mgr.sync as sync_mod

    local, remote, zone = sides
    seen: list[list[str]] = []
    monkeypatch.setattr(
        sync_mod.subprocess,
        "run",
        lambda argv, **kw: (seen.append(list(argv)), subprocess.CompletedProcess(argv, 0, "", ""))[
            1
        ],
    )
    sync(zone, {local: remote}, direction=Direction.LOCAL_TO_REMOTE)
    assert "-e" not in seen[0]
    assert ":" not in Path(seen[0][-1]).name


def test_an_unreadable_far_side_refuses_rather_than_copying(sides: tuple[str, str, Zone]) -> None:
    """**Fail closed.** `conflicts` is `filecmp` over two local trees and has no
    cross-host equivalent yet, and `PrepareRefused` exists precisely because
    rsync cannot report that both sides changed. A pre-pass that cannot run is a
    refusal, not a pass — proceeding would turn the one guard against silent data
    loss into a comment.
    """
    from env_mgr.protocols import PrepareRefused

    local, remote, zone = sides
    with pytest.raises(PrepareRefused, match="conflict pre-pass cannot read"):
        sync(
            zone,
            {local: remote},
            direction=Direction.LOCAL_TO_REMOTE,
            transports={local: FakeTransport(far_exists=True)},
        )


def test_a_far_side_that_does_not_exist_yet_has_nothing_to_conflict_with(
    sides: tuple[str, str, Zone], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control for the refusal above, and the ordinary case.

    A zone is named per attempt, so the far side is fresh nearly every time. If
    this failed, the rule would refuse every remote sync and step 3 of the probe
    would prove nothing.
    """
    import subprocess

    import env_mgr.sync as sync_mod

    local, remote, zone = sides
    monkeypatch.setattr(
        sync_mod.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "", "")
    )
    report = sync(
        zone,
        {local: remote},
        direction=Direction.LOCAL_TO_REMOTE,
        transports={local: FakeTransport(far_exists=False)},
    )
    assert report.conflicts == ()
