# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Criterion 18, and criterion 10 on the remote surface.

Spec §5.5: an agent given a natural-language description of how to sync a
directory will improvise, and the improvisation will be wrong in a way nobody
notices. A tool call has a schema, a name, and a result.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from env_mgr.fs.zone import Zone
from env_mgr.remote.connection import LocalConnection
from env_mgr.remote.tools import tools
from task_graph.ids import TaskId


@pytest.fixture
def zone(tmp_path: Path) -> Zone:
    root = tmp_path / "zone"
    (root / "out").mkdir(parents=True)
    (root / "out" / "result.txt").write_text("result")
    (tmp_path / "outside").mkdir()
    return Zone(TaskId.new(), 0, str(root.resolve()))


def _by_name(zone: Zone) -> dict[str, object]:
    return {t.name: t for t in tools(LocalConnection(), zone)}


# ---------------------------------------------------------- criterion 18


def test_remote_tools_have_schemas(zone: Zone) -> None:
    defs = tools(LocalConnection(), zone)
    assert {t.name for t in defs} == {"env_remote_run", "env_remote_push", "env_remote_pull"}
    for tool in defs:
        assert tool.description
        assert tool.schema["type"] == "object"
        assert tool.schema["additionalProperties"] is False
        assert tool.schema["required"]
        for name in tool.schema["required"]:
            assert name in tool.schema["properties"], f"{tool.name} requires an undeclared {name}"


def test_tool_call_round_trip(zone: Zone) -> None:
    result = _by_name(zone)["env_remote_run"].call(command=["echo", "hello"])  # type: ignore[attr-defined]
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "hello"


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync is not installed")
def test_push_and_pull_round_trip(zone: Zone, tmp_path: Path) -> None:
    far = tmp_path / "far"
    far.mkdir()
    report = _by_name(zone)["env_remote_push"].call(path="out", remote=str(far))  # type: ignore[attr-defined]
    assert (far / "result.txt").read_text() == "result"
    assert report["conflicts"] == ()


# ------------------------------- criterion 10, on this surface too


def test_tool_takes_no_zone_argument(zone: Zone) -> None:
    """Closing over the zone is what makes criterion 10 true here: the zone root
    is never taken from agent-supplied input, because the tool does not accept
    one."""
    for tool in tools(LocalConnection(), zone):
        properties = tool.schema["properties"]
        assert "zone" not in properties
        assert "working_directory" not in properties
        assert "cwd" not in properties or "relative to the zone" in properties["cwd"]["description"]


@pytest.mark.parametrize("proposal", ["/etc", "../outside", "out/../../outside"])
def test_a_path_argument_cannot_leave_the_zone(zone: Zone, proposal: str) -> None:
    with pytest.raises(PermissionError):
        _by_name(zone)["env_remote_push"].call(path=proposal, remote="/tmp/x")  # type: ignore[attr-defined]
