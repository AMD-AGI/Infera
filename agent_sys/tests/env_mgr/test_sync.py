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
