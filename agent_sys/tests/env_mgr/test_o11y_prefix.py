# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The `~/.infera_agent_sys` prefix: where it is, and what names it publishes."""

from __future__ import annotations

from pathlib import Path

from env_mgr.o11y.prefix import Prefix


def test_default_root_is_infera_agent_sys_under_home(tmp_path: Path) -> None:
    p = Prefix.resolve({"HOME": str(tmp_path)})
    assert p.root == tmp_path / ".infera_agent_sys"


def test_env_var_overrides_home(tmp_path: Path) -> None:
    p = Prefix.resolve({"HOME": "/nowhere", "AGENT_SYS_HOME": str(tmp_path / "elsewhere")})
    assert p.root == tmp_path / "elsewhere"


def test_the_layout_is_local_shaped(tmp_path: Path) -> None:
    p = Prefix.resolve({"HOME": str(tmp_path)})
    assert p.bin == p.root / "bin"
    assert p.share == p.root / "share"
    assert p.state == p.root / "state"
    assert p.run == p.root / "run"
    assert p.claude_home == p.state / "claude"
    assert p.agentsview_data == p.state / "agentsview"


def test_environment_names_every_directory(tmp_path: Path) -> None:
    p = Prefix.resolve({"HOME": str(tmp_path)})
    env = p.environment()
    assert env["AGENT_SYS_HOME"] == str(p.root)
    assert env["AGENT_SYS_BIN"] == str(p.bin)
    assert env["AGENT_SYS_SHARE"] == str(p.share)
    assert env["AGENT_SYS_STATE"] == str(p.state)
    assert env["AGENT_SYS_RUN"] == str(p.run)
    assert env["AGENT_SYS_CLAUDE_HOME"] == str(p.claude_home)
    assert env["AGENTSVIEW_DATA_DIR"] == str(p.agentsview_data)
    assert env["CLAUDE_PROJECTS_DIR"] == str(p.claude_home / "projects")


def test_create_is_idempotent(tmp_path: Path) -> None:
    p = Prefix.resolve({"HOME": str(tmp_path)})
    p.create()
    p.create()
    for d in (p.bin, p.share, p.state, p.run, p.claude_home, p.agentsview_data):
        assert d.is_dir()


def test_resolve_does_not_read_the_ambient_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_SYS_HOME", "/should/be/ignored")
    p = Prefix.resolve({"HOME": str(tmp_path)})
    assert p.root == tmp_path / ".infera_agent_sys"


def test_prefix_names_are_reachable_from_the_paths_family() -> None:
    """`paths` is where a reader looks for an `AGENT_SYS_*` name. All of them."""
    from env_mgr import paths

    assert paths.HOME_ENV_VAR == "AGENT_SYS_HOME"
    assert paths.BIN_ENV_VAR == "AGENT_SYS_BIN"
    assert paths.CLAUDE_HOME_ENV_VAR == "AGENT_SYS_CLAUDE_HOME"
    for name in ("HOME_ENV_VAR", "BIN_ENV_VAR", "CLAUDE_HOME_ENV_VAR"):
        assert name in paths.__all__


def test_agent_environment_carries_claude_config_dir(tmp_path: Path) -> None:
    from env_mgr.o11y.prefix import Prefix, agent_environment

    p = Prefix.resolve({"HOME": str(tmp_path)})
    env = agent_environment(p, base={"PATH": "/usr/bin"})
    assert env["CLAUDE_CONFIG_DIR"] == str(p.claude_home)
    assert env["PATH"].startswith(str(p.bin) + ":")
    assert "/usr/bin" in env["PATH"]


def test_agent_environment_does_not_touch_this_process(tmp_path: Path) -> None:
    """The guard on 'the user's own Claude Code is unaffected'."""
    import os

    from env_mgr.o11y.prefix import Prefix, agent_environment

    before = dict(os.environ)
    agent_environment(Prefix.resolve({"HOME": str(tmp_path)}), base={"PATH": "/usr/bin"})
    assert dict(os.environ) == before
    assert "CLAUDE_CONFIG_DIR" not in os.environ
