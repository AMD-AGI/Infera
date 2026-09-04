# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`material.deploy` — the zone's `config/`, and the one hole in it.

The zone-local `CLAUDE_CONFIG_DIR` is deliberate and stays (see `material.py`'s
own comment). But it also relocated the *transcripts*, and the o11y panel reads
one fixed directory. Measured on demo2: nine agent transcripts landed under
`<zone>/config/projects/` and none reached the prefix, so the panel showed
nothing. These tests hold the seam that fixes it — `projects` alone is shared —
and the surrounding behaviour it must not disturb.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from env_mgr import material
from env_mgr.fs.zone import Zone
from env_mgr.prefix import Prefix

from .stubs import AgentSpec


@pytest.fixture
def prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Prefix:
    """A prefix of our own, so no test touches the operator's `~`."""
    root = tmp_path / "prefix"
    monkeypatch.setenv("AGENT_SYS_HOME", str(root))
    return Prefix.resolve(os.environ)


def zone_at(tmp_path: Path, name: str) -> Zone:
    root = tmp_path / "zones" / name
    root.mkdir(parents=True)
    return Zone(task_id=name, attempt=0, root=str(root))


def test_config_projects_links_to_the_prefix(tmp_path: Path, prefix: Prefix) -> None:
    """Gate 1 of the design's four, in the pipeline rather than only in a probe."""
    zone = zone_at(tmp_path, "a")

    env = material.deploy(AgentSpec(), zone)

    link = Path(env["CLAUDE_CONFIG_DIR"]) / "projects"
    assert link.is_symlink(), "a real directory here is the bug this fixes"
    assert link.resolve() == (prefix.claude_home / "projects").resolve()


def test_the_prefix_target_is_created_when_absent(tmp_path: Path, prefix: Prefix) -> None:
    """`env_mgr`'s deploy path creates it, but a task may run without one."""
    assert not (prefix.claude_home / "projects").exists()

    material.deploy(AgentSpec(), zone_at(tmp_path, "a"))

    assert (prefix.claude_home / "projects").is_dir()


def test_two_zones_write_under_the_prefix_in_different_slugs(
    tmp_path: Path, prefix: Prefix
) -> None:
    """Sharing one physical `projects/` cannot collide: Claude Code names each
    subdirectory after the slugified cwd, and every attempt has its own zone."""
    envs = [material.deploy(AgentSpec(), zone_at(tmp_path, n)) for n in ("a", "b")]

    for i, env in enumerate(envs):
        slug = Path(env["CLAUDE_CONFIG_DIR"]) / "projects" / f"-slug-{i}"
        slug.mkdir()
        (slug / "session.jsonl").write_text("{}\n")

    landed = sorted(p.parent.name for p in (prefix.claude_home / "projects").glob("*/*.jsonl"))
    assert landed == ["-slug-0", "-slug-1"]


def test_a_wrong_link_is_repaired(tmp_path: Path, prefix: Prefix) -> None:
    """Idempotence has to cover the stale case, not just the correct one."""
    zone = zone_at(tmp_path, "a")
    config = Path(zone.root) / material.CONFIG_DIR
    config.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (config / "projects").symlink_to(elsewhere)

    material.deploy(AgentSpec(), zone)

    assert (config / "projects").resolve() == (prefix.claude_home / "projects").resolve()


def test_deploy_is_idempotent_over_a_correct_link(tmp_path: Path, prefix: Prefix) -> None:
    zone = zone_at(tmp_path, "a")
    material.deploy(AgentSpec(), zone)
    material.deploy(AgentSpec(), zone)

    link = Path(zone.root) / material.CONFIG_DIR / "projects"
    assert link.is_symlink()
    assert link.resolve() == (prefix.claude_home / "projects").resolve()


def test_an_empty_directory_there_is_replaced(tmp_path: Path, prefix: Prefix) -> None:
    """`rmdir` refuses a non-empty directory, so this branch cannot lose data."""
    zone = zone_at(tmp_path, "a")
    config = Path(zone.root) / material.CONFIG_DIR
    (config / "projects").mkdir(parents=True)

    material.deploy(AgentSpec(), zone)

    assert (config / "projects").is_symlink()


def test_a_populated_directory_there_is_left_alone_with_a_warning(
    tmp_path: Path, prefix: Prefix, caplog: pytest.LogCaptureFixture
) -> None:
    """Transcripts already written are somebody's evidence. The panel losing a
    zone is cheaper than deleting one, so this warns and proceeds."""
    zone = zone_at(tmp_path, "a")
    config = Path(zone.root) / material.CONFIG_DIR
    (config / "projects").mkdir(parents=True)
    (config / "projects" / "kept.jsonl").write_text("{}\n")

    with caplog.at_level(logging.WARNING):
        material.deploy(AgentSpec(), zone)

    assert (config / "projects" / "kept.jsonl").is_file()
    assert not (config / "projects").is_symlink()
    assert any("projects" in r.message for r in caplog.records)


def test_a_failed_link_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """**The law of this whole feature**: a broken panel never breaks a run.

    The unreachable target is a prefix rooted *inside a regular file*, so
    `target.mkdir(parents=True)` raises `NotADirectoryError`. This used to
    delete `$HOME` and `AGENT_SYS_HOME` instead, because `Prefix.resolve` then
    raised `KeyError` — but that was the bug, not the fixture: two other call
    sites did not catch it and a run with no `$HOME` died on a feature it never
    asked for. `resolve` is total now, so the failure has to come from the
    filesystem, which is where a real one would come from anyway.
    """
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    monkeypatch.setenv("AGENT_SYS_HOME", str(blocker / "prefix"))

    with caplog.at_level(logging.WARNING):
        env = material.deploy(AgentSpec(), zone_at(tmp_path, "a"))

    assert env["CLAUDE_CONFIG_DIR"]
    assert any("projects" in r.message for r in caplog.records)


def test_the_rest_of_the_config_directory_is_untouched(
    tmp_path: Path, prefix: Prefix
) -> None:
    """The credential/settings mechanism is why the zone-local config exists.
    Linking one subdirectory must not disturb it."""
    rule = tmp_path / "rules" / "style.md"
    rule.parent.mkdir()
    rule.write_text("# house style\n")
    zone = zone_at(tmp_path, "a")

    env = material.deploy(AgentSpec(rules=(str(rule),)), zone)

    config = Path(zone.root) / material.CONFIG_DIR
    assert env["CLAUDE_CONFIG_DIR"] == str(config)
    assert (config / "rules" / "style.md").read_text() == "# house style\n"
    assert env["CLAUDE_CODE_TMPDIR"] == os.path.join(zone.root, "tmp")
    assert env["TMPDIR"] == os.path.join(zone.root, "tmp")


def test_deploy_does_not_mutate_this_process_environment(
    tmp_path: Path, prefix: Prefix
) -> None:
    before = dict(os.environ)

    material.deploy(AgentSpec(), zone_at(tmp_path, "a"))

    assert dict(os.environ) == before
