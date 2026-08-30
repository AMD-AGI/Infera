# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Criteria 11 and 20 — the policy is not writable, and the workspace shares an
object store without a worktree.

Criterion 20's wording asks for a worktree. **Deviation D1**: measured, a
worktree's index lives in the main repository, so the only configuration in
which an agent can commit lets it write ``<main>/.git/hooks`` — where the hook
then runs. That is CVE-2026-26268, CVSS 9.9, no user interaction, and criterion
11 exists to prevent exactly it. So the tests assert §6.1's stated *purpose* —
a shared object store and an unmodified main checkout — rather than its
mechanism.
"""

from __future__ import annotations

import errno
import os
import subprocess
from pathlib import Path

import pytest

from env_mgr.fs.zone import Zone
from env_mgr.isolation.policy import Granted, Mode
from env_mgr.workspace import PRECIOUS, collect, cut, ensure_precious, is_precious
from task_graph.ids import TaskId

from .conftest import attempt, base_policy, run_confined


def _git(*args: str, cwd: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True
    )


@pytest.fixture
def main_repo(tmp_path: Path) -> str:
    repo = tmp_path / "main"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=str(repo))
    _git("config", "user.email", "t@example.com", cwd=str(repo))
    _git("config", "user.name", "t", cwd=str(repo))
    (repo / "f.txt").write_text("base\n")
    _git("add", "f.txt", cwd=str(repo))
    _git("commit", "-qm", "base", cwd=str(repo))
    (repo / ".git" / "hooks").mkdir(exist_ok=True)
    return str(repo)


@pytest.fixture
def zone(tmp_path: Path) -> Zone:
    root = tmp_path / "zone"
    root.mkdir()
    return Zone(task_id=TaskId.new(), attempt=0, root=str(root.resolve()))


# ---------------------------------------------------------- criterion 20


def test_workspace_shares_object_store(main_repo: str, zone: Zone) -> None:
    """Objects are **read** from the main repository through
    ``.git/objects/info/alternates`` and **written** locally, so spec §6.1's
    stated purpose — several agents sharing one object store — holds exactly."""
    ensure_precious(main_repo)
    ws = cut(main_repo, zone, branch="task/x")
    alternates = Path(ws.path) / ".git" / "objects" / "info" / "alternates"
    assert alternates.exists()
    assert os.path.realpath(main_repo) in alternates.read_text()
    # A blob from the shared store is readable without having been copied.
    head = _git("rev-parse", "HEAD:f.txt", cwd=ws.path).stdout.strip()
    assert _git("cat-file", "-p", head, cwd=ws.path).stdout == "base\n"


def test_main_checkout_unmodified(main_repo: str, zone: Zone) -> None:
    ensure_precious(main_repo)
    before = _git("rev-parse", "HEAD", cwd=main_repo).stdout
    ws = cut(main_repo, zone, branch="task/x")
    (Path(ws.path) / "f.txt").write_text("changed by the agent\n")
    _git("config", "user.email", "a@example.com", cwd=ws.path)
    _git("config", "user.name", "a", cwd=ws.path)
    _git("commit", "-aqm", "agent work", cwd=ws.path)

    assert _git("rev-parse", "HEAD", cwd=main_repo).stdout == before
    assert (Path(main_repo) / "f.txt").read_text() == "base\n"
    assert _git("status", "--porcelain", cwd=main_repo).stdout == ""


def test_the_agent_can_commit(main_repo: str, zone: Zone) -> None:
    """The property a read-only worktree could not have: staging is a write into
    the main repository's index, and a clone's index is its own."""
    ensure_precious(main_repo)
    ws = cut(main_repo, zone, branch="task/x")
    _git("config", "user.email", "a@example.com", cwd=ws.path)
    _git("config", "user.name", "a", cwd=ws.path)
    (Path(ws.path) / "new.txt").write_text("agent\n")
    _git("add", "new.txt", cwd=ws.path)
    _git("commit", "-qm", "agent work", cwd=ws.path)
    assert "agent work" in _git("log", "--oneline", cwd=ws.path).stdout


def test_collect_returns_work_by_a_supervisor_side_fetch(main_repo: str, zone: Zone) -> None:
    """Spec §6.1's second reason for a worktree — a branch visible without a
    fetch — comes back as one supervisor-side fetch. The write happens on the
    main side, performed by the process that already holds write access, and
    the write rule holds unmodified."""
    ensure_precious(main_repo)
    ws = cut(main_repo, zone, branch="task/x")
    _git("config", "user.email", "a@example.com", cwd=ws.path)
    _git("config", "user.name", "a", cwd=ws.path)
    (Path(ws.path) / "new.txt").write_text("agent\n")
    _git("add", "new.txt", cwd=ws.path)
    _git("commit", "-qm", "agent work", cwd=ws.path)

    assert collect(ws, main_repo) == "task/x"
    assert "agent work" in _git("log", "--oneline", "task/x", cwd=main_repo).stdout


# ----------------------------------- the gc hazard, and its built-in mitigation


def test_cut_refuses_a_main_repository_without_precious_objects(main_repo: str, zone: Zone) -> None:
    """``man git-clone`` warns that a borrower "will become corrupt" and names
    ordinary ``git commit`` in the *source* as the trigger. Measured, it is
    total rather than degraded: ``fatal: bad object HEAD``. Refusing is the only
    alternative to an agent's work destroyed by someone else's housekeeping."""
    assert is_precious(main_repo) is False
    with pytest.raises(RuntimeError, match=PRECIOUS):
        cut(main_repo, zone, branch="task/x")


def test_precious_objects_blocks_the_prune(main_repo: str, zone: Zone) -> None:
    """The mitigation is built into git, and this asserts the operation that
    actually removes objects a borrower may be pointing at.

    Measured here on git 2.34.1: ``repack -a -d`` is refused with *"cannot
    delete packs in a precious-objects repo"*, while ``gc --prune=now`` exits 0
    — it declines the destructive half silently rather than erroring, and the
    design's recorded *"cannot prune in a precious-objects repo"* is a different
    git's wording. The refusal is asserted against the operation, not against a
    message, because the message is the part that varies.
    """
    ensure_precious(main_repo)
    assert is_precious(main_repo) is True
    repacked = subprocess.run(
        ["git", "repack", "-a", "-d"], cwd=main_repo, capture_output=True, text=True
    )
    assert repacked.returncode != 0
    assert "precious" in (repacked.stderr + repacked.stdout)


# ---------------------------------------------------------- criterion 11


def test_main_git_hooks_denied(main_repo: str, zone: Zone) -> None:
    """The main repository is granted **read-only**, and that is what makes the
    clone-with-alternates admissible where a worktree was not."""
    ensure_precious(main_repo)
    ws = cut(main_repo, zone, branch="task/x")
    hook = os.path.join(main_repo, ".git", "hooks", "pre-commit")
    Path(hook).write_text("#!/bin/sh\n")
    policy = base_policy(Granted(zone.root, Mode.READ_WRITE), Granted(main_repo, Mode.READ_EXEC))

    denied, control = run_confined(
        policy, lambda: (attempt(hook, "w"), attempt(os.path.join(ws.path, "f.txt"), "r"))
    )
    assert control == 0, "the positive control failed: the workspace was unreadable"
    assert denied == errno.EACCES


def test_a_read_write_grant_would_permit_the_hook_write(main_repo: str, zone: Zone) -> None:
    """The **negative control** for the test above, and it is also design §7.1's
    measured finding stated as a test.

    A worktree's index lives in the main repository, so the only configuration
    in which an agent can commit is one granting it **read-write** — and under
    exactly that grant, writing `<main>/.git/hooks/pre-commit` succeeds and the
    hook then runs. That is CVE-2026-26268, CVSS 9.9, no user interaction.

    Asserting it here does two things: it proves the read-only grant is what
    denies the write above, rather than something incidental about the path, and
    it pins the reason deviation D1 exists. If this ever stops succeeding, the
    clone-with-alternates was never necessary and D1 should be revisited.
    """
    ensure_precious(main_repo)
    cut(main_repo, zone, branch="task/x")
    hook = os.path.join(main_repo, ".git", "hooks", "pre-commit")
    Path(hook).write_text("#!/bin/sh\n")
    permissive = base_policy(
        Granted(zone.root, Mode.READ_WRITE), Granted(main_repo, Mode.READ_WRITE)
    )
    assert run_confined(permissive, lambda: attempt(hook, "w")) == 0, (
        "the forbidden grant did NOT permit the write, so the test above proves "
        "nothing about the grant — the denial must come from somewhere else"
    )


def test_main_git_config_denied(main_repo: str, zone: Zone) -> None:
    ensure_precious(main_repo)
    cut(main_repo, zone, branch="task/x")
    config = os.path.join(main_repo, ".git", "config")
    policy = base_policy(Granted(zone.root, Mode.READ_WRITE), Granted(main_repo, Mode.READ_EXEC))
    denied, control = run_confined(policy, lambda: (attempt(config, "w"), attempt(config, "r")))
    assert control == 0, "the positive control failed: the config was unreadable"
    assert denied == errno.EACCES


def test_shell_rc_denied(zone: Zone, tmp_path: Path) -> None:
    """Spec §4.4's other named target. A home directory is not in the default
    granted set at all, so this is denied by the allow-list rather than by a
    rule about shell files."""
    rc = tmp_path / "home" / ".bashrc"
    rc.parent.mkdir()
    rc.write_text("export PATH=$PATH\n")
    policy = base_policy(Granted(zone.root, Mode.READ_WRITE))
    assert run_confined(policy, lambda: attempt(str(rc), "w")) == errno.EACCES


def test_policy_not_writable_by_agent(zone: Zone, tmp_path: Path) -> None:
    """If an agent can write the file that grants its own permissions, there is
    no boundary. Precedent: CVE-2026-48124, hooks executed from a
    workspace-local settings file."""
    from env_mgr import meta

    policy_file = tmp_path / "policy" / "meta.json"
    meta.save(
        meta.Meta(domains=(("store", str(tmp_path / "root"), "handoff_storage"),)), str(policy_file)
    )
    granted = base_policy(Granted(zone.root, Mode.READ_WRITE))

    denied, control = run_confined(
        granted,
        lambda: (
            attempt(str(policy_file), "w"),
            attempt(os.path.join(zone.root, "own.txt"), "w"),
        ),
    )
    assert control == 0, "the positive control failed: the zone was not writable"
    assert denied == errno.EACCES


def test_the_agents_own_hooks_are_its_own(main_repo: str, zone: Zone) -> None:
    """The residual, stated because it is bounded and real: the agent can write
    ``<zone>/workspace/.git/hooks``. That is inside its zone, it is not an
    escape, and an agent that can run code in its own zone can do that anyway.
    Criterion 11 is about the policy file and about *shared* repository state."""
    ensure_precious(main_repo)
    ws = cut(main_repo, zone, branch="task/x")
    own_hook = os.path.join(ws.path, ".git", "hooks", "pre-commit")
    policy = base_policy(Granted(zone.root, Mode.READ_WRITE), Granted(main_repo, Mode.READ_EXEC))
    assert run_confined(policy, lambda: attempt(own_hook, "w")) == 0
