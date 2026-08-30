# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The workspace: a clone with alternates. Design §7. **Deviation D1.**

Spec §6.1 asks for a *worktree*; spec §4.5 says a task's executor may not write
outside its zones, no exception. Measured, those cannot both hold: a worktree's
``.git`` is a file pointing at ``<main>/.git/worktrees/<name>``, its index lives
in the main repository, and so staging is a write outside the zone. The only
configuration in which an agent can commit is one where it can write the shared
repository — and confined with exactly that grant, writing
``<main>/.git/hooks/pre-commit`` **succeeded** and the hook then **ran**. That is
CVE-2026-26268, CVSS 9.9, no user interaction, which is the reason criterion 11
exists.

``git clone --shared`` satisfies both: objects are **read** from the main
repository through ``.git/objects/info/alternates`` and **written** locally, so
spec §6.1's stated purpose — several agents sharing one object store — is
preserved exactly while the main repository stays read-only. Its mechanism is
what changes.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from typing import NamedTuple

from env_mgr.fs.domain import DomainKind, subdir_for
from env_mgr.fs.zone import Zone

__all__ = [
    "NEUTRALISED_ENV",
    "PRECIOUS",
    "Workspace",
    "collect",
    "cut",
    "ensure_precious",
    "is_precious",
]

#: An allow-list makes an ungranted file look **broken**, not absent. Measured:
#: ``GIT_CONFIG_GLOBAL`` at a path that does not exist gives rc=0, and at a path
#: that exists but is not granted gives rc=128 and *"fatal: unknown error
#: occurred while reading the configuration files"*. Real programs treat an
#: optional file's absence as normal and a permission error on it as fatal, so
#: every path a tool merely *probes* becomes a new hard failure under §5.1.
#:
#: The fix is per tool, not per sandbox. This list is the shape of the problem
#: and not the end of it: nothing enumerates what the next tool probes.
NEUTRALISED_ENV = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
}

PRECIOUS = "extensions.preciousObjects"


class Workspace(NamedTuple):
    path: str
    branch: str
    main_repo: str
    repos: tuple[str, ...] = ()


def _git(*args: str, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(NEUTRALISED_ENV)
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def is_precious(main_repo: str) -> bool:
    try:
        got = _git("config", "--get", PRECIOUS, cwd=main_repo)
    except subprocess.CalledProcessError:
        return False
    return got.stdout.strip().lower() == "true"


def ensure_precious(main_repo: str) -> None:
    """Set ``extensions.preciousObjects`` in the **main** repository.

    ``man git-clone`` warns that a borrower "will become corrupt" and names
    ordinary ``git commit`` in the *source* as the trigger, via the automatic
    ``git maintenance run --auto``. Reproduced, and it is total rather than
    degraded: ``fatal: bad object HEAD``, and ``fsck`` reporting a missing blob.
    The agent's entire workspace history becomes unreadable.

    The mitigation is built into git and verified: ``cannot prune in a
    precious-objects repo``. `cut` refuses without it, because the alternative
    is an agent's work destroyed by someone else's routine housekeeping.
    """
    _git("config", PRECIOUS, "true", cwd=main_repo)


def cut(
    main_repo: str,
    zone: Zone,
    *,
    branch: str,
    repos: Sequence[str] = (),
    repo_locations: dict[str, str] | None = None,
) -> Workspace:
    """``git clone --shared --no-hardlinks`` into ``<zone>/workspace``.

    Each declared dependency repository is cut the same way (design §7.1.1): the
    main repository is what the *system* is working on and is global, while the
    dependencies are what *this task* needs, so a task that needs none does not
    pay for cloning three. Nothing here resolves a name to a location — a
    declared ``repos`` entry is a key into the run configuration, exactly as a
    resource pool name is.
    """
    if not is_precious(main_repo):
        raise RuntimeError(
            f"{main_repo} does not set {PRECIOUS}; refusing to cut a workspace that "
            f"a `git gc` in the main repository would silently corrupt"
        )
    path = os.path.join(zone.root, subdir_for(DomainKind.WORKSPACE))
    os.makedirs(path, exist_ok=True)
    if not os.path.isdir(os.path.join(path, ".git")):
        _git("clone", "--shared", "--no-hardlinks", main_repo, path)
    _git("checkout", "-B", branch, cwd=path)

    locations = repo_locations or {}
    cloned: list[str] = []
    for name in repos:
        source = locations.get(name)
        if source is None:
            raise KeyError(
                f"task declares repo {name!r}, which the run configuration does not "
                f"locate (have {sorted(locations)})"
            )
        if not is_precious(source):
            raise RuntimeError(f"{source} does not set {PRECIOUS}; refusing to clone it")
        dst = os.path.join(path, name)
        if not os.path.isdir(os.path.join(dst, ".git")):
            _git("clone", "--shared", "--no-hardlinks", source, dst)
        cloned.append(dst)
    return Workspace(path=path, branch=branch, main_repo=main_repo, repos=tuple(cloned))


def collect(ws: Workspace, main_repo: str) -> str:
    """Supervisor-side. Fetch the agent's branch **into** the main repository.

    Spec §6.1's second reason for a worktree was that a branch is visible to the
    operator without a fetch, and a clone gives that up. Measured, of the three
    ways work can come back only this one is admissible: the agent pushing needs
    the main repository writable, which is the grant §7.1 forbids. So the write
    happens on the main side, performed by the process that already holds write
    access, **outside the agent's confinement** — and the property becomes one
    supervisor-side fetch instead of a property of the layout.
    """
    _git("fetch", ws.path, f"{ws.branch}:{ws.branch}", cwd=main_repo)
    return ws.branch
