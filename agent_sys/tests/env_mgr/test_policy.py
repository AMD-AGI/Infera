# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Criterion 10 — the zone root is never taken from agent-supplied input.

Plus the two things the design's measurements added to spec §4.5.1's default
set, both of which are the difference between a working agent and one that dies
before reaching any question of its own.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from env_mgr.fs.zone import Zone
from env_mgr.isolation.policy import (
    BIN_DIRS,
    DEFAULT_SYSTEM_SET,
    Granted,
    Mode,
    Policy,
    anchor_zone_root,
    executable_path,
    interpreter_grants,
)
from task_graph.ids import TaskId


@pytest.fixture
def zone(tmp_path: Path) -> Zone:
    root = tmp_path / "zone"
    (root / "inner").mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    return Zone(task_id=TaskId.new(), attempt=0, root=str(root.resolve()))


# ---------------------------------------------------------- criterion 10


def test_zone_root_not_from_agent_input(zone: Zone, tmp_path: Path) -> None:
    """A model-controlled ``cwd`` was CVE-2025-59532 in Codex and
    CVE-2026-50548 in Cursor, both CVSS 9.8. The answer is anchored to what the
    harness started with."""
    with pytest.raises(PermissionError, match="outside zone"):
        anchor_zone_root(str(tmp_path / "outside"), zone)


def test_a_proposal_inside_the_subtree_is_honoured(zone: Zone) -> None:
    inner = os.path.join(zone.root, "inner")
    assert anchor_zone_root(inner, zone) == inner
    assert anchor_zone_root(None, zone) == zone.root


def test_a_traversal_proposal_is_denied(zone: Zone) -> None:
    """The proposal is checked by canonical containment, not by inspection: a
    string that reads like it is inside is not."""
    with pytest.raises(PermissionError):
        anchor_zone_root(os.path.join(zone.root, "inner", "..", "..", "outside"), zone)


# ------------------------------------------------- the default set, corrected


def test_dev_null_is_writable() -> None:
    """Measured: without it git dies before reaching any repository question —
    ``fatal: could not open '/dev/null' for reading and writing``."""
    entry = next(g for g in DEFAULT_SYSTEM_SET if g.path == "/dev/null")
    assert entry.mode is Mode.READ_WRITE
    assert entry.optional is False


def test_the_three_grants_nobody_guesses_are_present() -> None:
    """Each was found by something breaking with a symptom that named the wrong
    cause.

    ``/dev/urandom``: the backend is a standalone Bun binary that aborts in 3 ms
    without it and hands the operator a crash-report URL for the wrong project.
    ``/run/systemd/resolve/stub-resolv.conf``: ``/etc/resolv.conf`` is a symlink
    out of ``/etc`` and Landlock rules apply to the **resolved** path, so
    granting ``/etc`` does not give you DNS — one file, and the symptoms were an
    immediate clean failure in one tool and a three-minute hang in another. The
    readable temp directory is `material.py`'s, inside the zone.
    """
    paths = {g.path for g in DEFAULT_SYSTEM_SET}
    assert "/dev/urandom" in paths
    assert "/run/systemd/resolve/stub-resolv.conf" in paths


def test_no_home_directory_in_the_default_set() -> None:
    """It is where credentials, SSH keys, and other tasks' scratch live.
    Granting it by default would make the zone boundary decorative."""
    home = os.path.expanduser("~")
    for entry in DEFAULT_SYSTEM_SET:
        assert not entry.path.startswith(home), entry.path


def test_interpreter_grants_cover_this_interpreter() -> None:
    """The default set is never sufficient alone, which the spec does not say.

    Every ordinary Python install — conda, pyenv, uv, venv — is under ``$HOME``,
    and the failure is raised by ``subprocess`` in the **parent**, naming the
    interpreter rather than the sandbox.
    """
    import sys

    grants = interpreter_grants()
    assert any(sys.executable.startswith(g.path + os.sep) for g in grants), (
        f"{sys.executable} is not covered by {[g.path for g in grants]}"
    )


def test_optional_is_per_entry() -> None:
    """Neither mechanism's default is right, because the two cases are
    different: ``/lib64`` on a merged-``/usr`` distro is expected, and a path a
    task's permissions named is an error."""
    by_path = {g.path: g for g in DEFAULT_SYSTEM_SET}
    assert by_path["/lib64"].optional is True
    assert by_path["/usr"].optional is False


def test_policy_with_appends() -> None:
    policy = Policy((Granted("/usr", Mode.READ_EXEC),))
    wider = policy.with_(Granted("/zone", Mode.READ_WRITE))
    assert len(policy.granted) == 1, "with_ must not mutate"
    assert wider.granted[-1].path == "/zone"


# ------------------------------------------ PATH, projected from the granted set


def test_path_is_not_a_boundary(tmp_path: Path) -> None:
    """The measurement that decides what `executable_path` is *for*.

    `validator` found that a body handed an environment block with no `PATH`
    still reaches `python3` and `git`, because POSIX `sh` substitutes a built-in
    default, and asked whether that is an isolation hole. Measured
    (`scratch/impl-2026-08/env_mgr/p2_path_is_not_a_boundary.py`) it is not:

    | cell | result |
    |---|---|
    | `/usr` granted, `PATH=""` | `git --version` **rc=0** |
    | `/usr` ungranted, `PATH` naming it | exec **EACCES** |

    So reachability is the allow-list's, in the kernel, and removing an entry
    from `PATH` buys nothing. This test is the second cell, which is the one
    that would matter if it ever stopped holding.
    """
    import errno
    import subprocess

    from .conftest import run_confined

    zone = tmp_path / "zone"
    zone.mkdir()
    policy = Policy(
        (
            Granted("/etc", Mode.READ_EXEC),
            Granted("/proc", Mode.READ_EXEC),
            Granted("/dev/null", Mode.READ_WRITE),
            Granted(str(zone), Mode.READ_WRITE),
        )
    )

    def body() -> int:
        try:
            subprocess.run(
                ["/usr/bin/env", "true"],
                capture_output=True,
                env={"PATH": ":".join(BIN_DIRS)},
            )
        except OSError as e:
            return e.errno or 0
        return 0

    assert run_confined(policy, body) == errno.EACCES


def test_executable_path_names_only_granted_directories() -> None:
    """The invariant that is worth more than the determinism: `PATH` can never
    name a directory the kernel will refuse.

    §5.5 measured what the alternative costs — an ungranted-but-existing path
    makes a tool report *itself* broken rather than report the file as absent,
    and the symptom names the wrong cause.
    """
    from env_mgr.fs.path import contained

    policy = Policy((Granted("/usr", Mode.READ_EXEC),))
    dirs = executable_path(policy).split(os.pathsep)
    assert dirs, "the projection produced nothing at all"
    for d in dirs:
        assert contained(d, "/usr"), d

    # And the membership test is `contained`, not a string prefix, which is why
    # `/sbin` appears above: on a merged-`/usr` system it is a symlink to
    # `/usr/sbin`, so the kernel — which also resolves — really will allow it.
    # A prefix check would have dropped a directory that works.
    if os.path.realpath("/sbin").startswith("/usr/"):
        assert "/sbin" in dirs


def test_executable_path_is_empty_when_nothing_executable_is_granted() -> None:
    """Not a failure — a policy granting no bin directory is a policy under
    which nothing can be run, and saying so with an empty string is more honest
    than substituting a default the kernel will refuse."""
    assert executable_path(Policy((Granted("/dev/null", Mode.READ_WRITE),))) == ""


def test_executable_path_keeps_the_conventional_order() -> None:
    """Order matters and is not ours to reinvent: it is the order every POSIX
    shell writes, so `/usr/bin/python3` wins over `/bin/python3` exactly as it
    does unconfined."""
    dirs = executable_path(Policy(DEFAULT_SYSTEM_SET)).split(os.pathsep)
    present = [d for d in BIN_DIRS if d in dirs]
    assert [d for d in dirs if d in BIN_DIRS] == present


def test_executable_path_includes_a_granted_prefixs_bin() -> None:
    """The conda / pyenv / uv / venv case. The interpreter is under `$HOME`,
    the default set excludes it, and `interpreter_grants` is what puts it back —
    so its `bin/` has to be findable or the body must spell out an absolute
    path to its own interpreter."""
    import sys

    policy = Policy(DEFAULT_SYSTEM_SET + interpreter_grants())
    dirs = executable_path(policy).split(os.pathsep)
    if not sys.executable.startswith(("/usr/", "/bin/")):
        assert os.path.dirname(sys.executable) in dirs, (
            f"{sys.executable} is not findable through {dirs}"
        )


# ------------------------ the task package: **no grant here**, §4.16

# `package_grants` is gone. F19's third position stages a copy into the zone
# instead, so the tests for *what the package looks like to a task* live with
# the staging primitive — `tests/env_mgr/test_layout.py`, `stage_package`.
#
# The grant it replaced is not merely unused: a granted package root is what
# opened criterion 13's second route, because `validators/` lives in the
# package. See `test_a_staged_package_still_carries_the_validators`.
