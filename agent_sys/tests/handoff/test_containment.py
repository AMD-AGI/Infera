"""Criterion 16: permission is containment, asserted against the real layout.

`test_prefix_sibling_denied` is named separately because it is the
`/a/b`-versus-`/a/bc` case, which the other three would all pass — and it is
the live one: three CVEs in 2026, one of whose fix commits is titled *"is
relative to is better than starts with"* and is a single line.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from handoff import check_contained, version_dir
from handoff.errors import NotContained
from task_graph.ids import HandoffId


def test_own_subtree_and_subtasks_reachable(tmp_path: Path) -> None:
    """A task reaches its own subtree and its subtasks': because a subtask's
    storage sits inside its parent's, the decision is containment and a hook
    answers it without consulting a registry."""
    parent = tmp_path / "task-parent"
    child = parent / "task-child"
    child.mkdir(parents=True)
    (child / "out").write_text("x")

    check_contained(parent, parent)  # its own zone
    check_contained(child, parent)  # a subtask's
    check_contained(child / "out", parent)


def test_sibling_denied(tmp_path: Path) -> None:
    a, b = tmp_path / "task-a", tmp_path / "task-b"
    a.mkdir()
    b.mkdir()
    with pytest.raises(NotContained, match="outside"):
        check_contained(b, a)


def test_prefix_sibling_denied(tmp_path: Path) -> None:
    """`str.startswith(zone)` says True here and is wrong."""
    zone = tmp_path / "a" / "b"
    zone.mkdir(parents=True)
    intruder = tmp_path / "a" / "bc" / "x"
    intruder.mkdir(parents=True)

    assert str(intruder).startswith(str(zone)), "the naive check would pass; that is the point"
    with pytest.raises(NotContained):
        check_contained(intruder, zone)


def test_dotdot_denied(tmp_path: Path) -> None:
    """`is_relative_to` is purely lexical and says True for `…/a/b/../bc/x`, so
    `..` is rejected by policy *before* resolving — and the message quotes the
    path as written rather than as resolved."""
    zone = tmp_path / "a" / "b"
    (tmp_path / "a" / "bc").mkdir(parents=True)
    zone.mkdir(parents=True)
    escape = zone / ".." / "bc" / "x"

    assert Path(escape).is_relative_to(zone), "the lexical check would pass"
    with pytest.raises(NotContained, match=r"\.\."):
        check_contained(escape, zone)


def test_a_symlink_out_of_the_zone_is_denied(tmp_path: Path) -> None:
    """`is_relative_to` says True for an in-zone symlink pointing out; only a
    resolve catches it."""
    zone = tmp_path / "zone"
    zone.mkdir()
    (tmp_path / "elsewhere").mkdir()
    os.symlink(tmp_path / "elsewhere", zone / "link")

    with pytest.raises(NotContained):
        check_contained(zone / "link", zone)


def test_a_path_that_does_not_exist_yet_is_still_checkable(tmp_path: Path) -> None:
    """One of the three things the string check buys that a kernel layer
    cannot: `openat2` cannot check a path that does not exist."""
    zone = tmp_path / "zone"
    zone.mkdir()
    check_contained(zone / "not" / "created" / "yet", zone)
    with pytest.raises(NotContained):
        check_contained(tmp_path / "other" / "not-created", zone)


def test_the_store_layout_is_what_containment_is_asserted_against(tmp_path: Path) -> None:
    """Against the real layout, not a synthetic one: `<root>/<hid>/v<N>/`."""
    root = tmp_path / "handoffs"
    mine, theirs = HandoffId.new(), HandoffId.new()
    check_contained(version_dir(root, mine, 0), root / str(mine))
    with pytest.raises(NotContained):
        check_contained(version_dir(root, theirs, 0), root / str(mine))
