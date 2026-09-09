# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`_probe_environment` relocates `CLAUDE_CONFIG_DIR`, so it must carry the block.

The same defect `material.deploy` already fixed one module over, at a second
site: moving the config directory also moves away the ``env`` block holding the
endpoint and credentials, and the CLI then answers ``Not logged in`` and blames
itself. `preflight_credentials`' own docstring tabulates it — *relocated config
dir, no injection → rc=1*.

**It was invisible on a developer's host** and only surfaced under `--docker`:
`_probe_environment` copies `os.environ`, and an operator's shell exports
`ANTHROPIC_*` already, so the missing injection was masked. Measured inside the
container, where nothing exports them: host 7 such variables, container 0, and
the preflight failed. These tests set the variable *absent* for that reason.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from cli.environment import _probe_environment
from env_mgr.prefix import CLAUDE_CONFIG_ENV_VAR, Prefix

#: A key the operator's settings file declares. Real name, so a reader sees the
#: case that actually failed rather than a placeholder.
KEY = "ANTHROPIC_BASE_URL"

FROM_SETTINGS = "https://from-settings.example"


def _settings(home: Path, block: dict[str, str]) -> None:
    """Write the operator's settings file where `harness.settings_path` looks."""
    config = home / ".claude"
    config.mkdir(parents=True, exist_ok=True)
    (config / "settings.json").write_text(json.dumps({"env": block}))


def _isolate(monkeypatch: Any, home: Path) -> None:
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("AGENT_SYS_HOME", raising=False)
    monkeypatch.delenv(CLAUDE_CONFIG_ENV_VAR, raising=False)
    monkeypatch.delenv(KEY, raising=False)


def test_the_block_reaches_the_probe_when_the_environment_does_not_carry_it(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """The container's case: nothing exports the key, so the file must supply it."""
    _isolate(monkeypatch, tmp_path)
    _settings(tmp_path, {KEY: FROM_SETTINGS})

    assert _probe_environment().get(KEY) == FROM_SETTINGS


def test_the_relocation_this_function_decided_is_not_displaced(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """`harness_env`'s reserved set, asserted here rather than trusted.

    A settings file may name `CLAUDE_CONFIG_DIR`, and carrying it across would
    undo the very relocation that made carrying anything necessary. `PATH` is
    reserved for the same reason one level down.
    """
    _isolate(monkeypatch, tmp_path)
    before = os.environ.get("PATH")
    _settings(tmp_path, {CLAUDE_CONFIG_ENV_VAR: "/not/the/prefix", "PATH": "/not/the/path"})

    env = _probe_environment()

    assert env[CLAUDE_CONFIG_ENV_VAR] == str(Prefix.resolve(os.environ).claude_home)
    assert env.get("PATH") == before


def test_a_live_value_still_outranks_the_file(monkeypatch: Any, tmp_path: Path) -> None:
    """The host's case, which must not change: an exported override wins.

    `harness_env` resolves this per key, so the injection cannot overwrite what
    the operator is actually running under — which is why adding it here is
    safe on every machine where the probe already passes.
    """
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv(KEY, "https://from-the-shell.example")
    _settings(tmp_path, {KEY: FROM_SETTINGS})

    assert _probe_environment().get(KEY) == "https://from-the-shell.example"


def test_no_settings_file_is_not_a_failure(monkeypatch: Any, tmp_path: Path) -> None:
    """A machine with no Claude Code runs non-AI tasks; the probe still builds
    an environment rather than raising."""
    _isolate(monkeypatch, tmp_path)

    env = _probe_environment()

    assert env[CLAUDE_CONFIG_ENV_VAR] == str(Prefix.resolve(os.environ).claude_home)
    assert KEY not in env
