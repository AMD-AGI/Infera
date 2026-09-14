# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`env_mgr.harness` — what the operator's installation configured.

The behaviour under test was measured before it was written:
`scratch/impl-2026-08/env_mgr/p7_relocated_config_loses_auth.py` runs one prompt
three ways and finds that a relocated ``CLAUDE_CONFIG_DIR`` alone answers
``Not logged in · Please run /login`` (rc=1), while the same relocation with the
settings `env` block injected answers ``OK`` (rc=0).

Nothing here reaches the network or reads the operator's real settings file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from env_mgr.harness import harness_env, settings_path


def _settings(tmp_path: Path, env: dict[str, object]) -> str:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"env": env, "theme": "dark"}), encoding="utf-8")
    return str(path)


# ------------------------------------------------- resolving the settings file


def test_settings_path_prefers_the_supervisors_config_dir(tmp_path: Path) -> None:
    """An operator who set `CLAUDE_CONFIG_DIR` meant it.

    Read from the **supervisor's** environment and nowhere else: by the time a
    task runs, `material.deploy` has already rewritten that variable to point
    into the zone, so the answer only exists before confinement.
    """
    assert settings_path({"CLAUDE_CONFIG_DIR": str(tmp_path)}) == str(tmp_path / "settings.json")


def test_settings_path_falls_back_to_the_home_directory() -> None:
    assert settings_path({}) == os.path.join(os.path.expanduser("~"), ".claude", "settings.json")


# ------------------------------------------------------------ the `env` block


def test_the_block_is_carried_across(tmp_path: Path) -> None:
    """The measured repair: what the relocated config would have provided.

    Arm C of the probe. Without this the agent authenticates against nothing and
    reports a login problem, which names the wrong cause.
    """
    path = _settings(tmp_path, {"ANTHROPIC_BASE_URL": "https://example.invalid/x"})
    assert harness_env(path, {}) == {"ANTHROPIC_BASE_URL": "https://example.invalid/x"}


def test_a_live_value_beats_the_file(tmp_path: Path) -> None:
    """Two sources that disagree, and the disagreement matters.

    The file is what the operator *wrote*; the environment is what this
    supervisor is *actually running under*. An export for this session would be
    silently undone by the file, so the live value wins.
    """
    path = _settings(tmp_path, {"ANTHROPIC_BASE_URL": "https://from-the-file.invalid"})
    got = harness_env(path, {"ANTHROPIC_BASE_URL": "https://exported.invalid"})
    assert got == {"ANTHROPIC_BASE_URL": "https://exported.invalid"}


def test_a_key_the_block_does_not_name_is_not_forwarded(tmp_path: Path) -> None:
    """**This is what keeps it an allow-list.**

    Forwarding every supervisor variable that happens to be set would put the
    operator's whole environment inside the sandbox. The allow-list is the
    operator's own configuration rather than a guess of ours, so a secret they
    did not put in the block does not travel.
    """
    path = _settings(tmp_path, {"ANTHROPIC_BASE_URL": "https://example.invalid/x"})
    got = harness_env(path, {"SOME_OTHER_SECRET": "s3cret", "ANTHROPIC_BASE_URL": "x"})
    assert "SOME_OTHER_SECRET" not in got


@pytest.mark.parametrize("key", ["CLAUDE_CONFIG_DIR", "CLAUDE_CODE_TMPDIR", "TMPDIR", "PATH"])
def test_a_reserved_key_is_never_forwarded(tmp_path: Path, key: str) -> None:
    """Carrying the operator's configuration must not undo the relocation that
    made carrying it necessary — and `PATH` is derived from the granted set
    precisely so that it cannot name a directory the kernel will refuse.

    An agent spec's **declared** `env` may still set any of them; that update is
    applied after this module, and an author saying so outranks a default.
    """
    path = _settings(tmp_path, {key: "/somewhere/else", "ANTHROPIC_BASE_URL": "x"})
    assert key not in harness_env(path, {})


def test_no_settings_file_is_a_configuration_not_a_degradation(tmp_path: Path) -> None:
    """A machine with no Claude Code installed runs `program` tasks perfectly
    well. There is nothing to report."""
    assert harness_env(str(tmp_path / "absent.json"), {}) == {}


def test_a_settings_file_that_does_not_parse_raises_and_names_the_path(tmp_path: Path) -> None:
    """The one loud case, and the reason it is loud.

    Swallowing it gives the agent `Not logged in` — the wrong cause, one
    character wide, and the exact failure this module exists to remove.
    """
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="could not be read"):
        harness_env(str(path), {})


def test_a_block_that_is_not_a_mapping_yields_nothing(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"env": ["not", "a", "mapping"]}), encoding="utf-8")
    assert harness_env(str(path), {}) == {}


def test_no_value_from_the_block_reaches_the_exception(tmp_path: Path) -> None:
    """**The block routinely holds a subscription key.**

    Every message here names keys. A malformed file is the only raise, and it
    carries the path and the parser's complaint — never a value.
    """
    path = tmp_path / "settings.json"
    path.write_text(
        '{"env": {"ANTHROPIC_CUSTOM_HEADERS": "Ocp-Key: sup3rsecret"} ', encoding="utf-8"
    )
    with pytest.raises(ValueError) as caught:
        harness_env(str(path), {})
    assert "sup3rsecret" not in str(caught.value)
