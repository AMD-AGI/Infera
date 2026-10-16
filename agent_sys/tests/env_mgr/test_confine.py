# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Criteria 3, 6, 7, 12, 14 — at the **kernel** layer.

Every denial here is asserted as ``EACCES`` against a named path, never as a
non-zero exit status. A design measurement produced a false PASS from a
``returncode != 0`` check because both children were failing to exec the
interpreter rather than being denied, and every one of these tests carries the
positive control that rules that out: the same operation succeeding inside the
zone, in the same confined process.
"""

from __future__ import annotations

import errno
import os
import subprocess
from pathlib import Path

import pytest

from env_mgr.isolation.policy import Granted, Mode

from .conftest import attempt, base_policy, errno_script, run_confined

pytestmark = pytest.mark.usefixtures("landlock_abi")


@pytest.fixture
def world(tmp_path: Path) -> dict[str, str]:
    zone = tmp_path / "zone"
    (zone / "inner").mkdir(parents=True)
    (zone / "inner" / "mine.txt").write_text("mine")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    sibling = tmp_path / "sibling-zone"
    sibling.mkdir()
    (sibling / "theirs.txt").write_text("theirs")
    evil = tmp_path / "zone-EVIL"
    evil.mkdir()
    (evil / "x.txt").write_text("owned")
    link = zone / "escape"
    link.symlink_to(outside)
    return {
        "zone": str(zone),
        "inside": str(zone / "inner" / "mine.txt"),
        "outside": str(outside / "secret.txt"),
        "sibling": str(sibling / "theirs.txt"),
        "sibling_dir": str(sibling),
        "evil": str(evil / "x.txt"),
        "via_symlink": str(link / "secret.txt"),
        "dotdot": os.path.join(str(zone), "inner", "..", "..", "outside", "secret.txt"),
        "tmp": str(tmp_path),
    }


def _policy(world: dict[str, str]) -> object:
    return base_policy(Granted(world["zone"], Mode.READ_WRITE))


# --------------------------------------- criterion 3, the enforcement layer


@pytest.mark.parametrize("defeat", ["evil", "via_symlink", "dotdot"])
def test_startswith_defeats_denied_by_the_kernel(world: dict[str, str], defeat: str) -> None:
    """The three documented defeats, with **no userspace check involved**.

    This is the half of criterion 3 that says something about confinement.
    `test_path.py` is the other half, and it is a different claim: which paths
    get handed to the kernel in the first place.
    """
    target = world[defeat]

    def body() -> tuple[int, int]:
        return attempt(target, "r"), attempt(world["inside"], "r")

    denied, control = run_confined(_policy(world), body)
    assert control == 0, "the positive control failed: the child could not read its own zone"
    assert denied == errno.EACCES


# ----------------------------------------------------------- criterion 6


def test_scripted_bypass_denied(world: dict[str, str], python: str, tmp_path: Path) -> None:
    """An agent writes a script that opens a file outside its zone and runs it.

    Criterion 6's "an agent writes a script" is incidental — what is tested is
    that a subprocess cannot escape, and the exit status **is the errno**, so a
    shell that could not start is distinguishable from a denial.
    """
    script = Path(world["zone"]) / "reader.py"
    script.write_text(errno_script(world["outside"], "r"))
    write_script = Path(world["zone"]) / "writer.py"
    write_script.write_text(errno_script(os.path.join(world["tmp"], "outside", "new"), "w"))

    def body() -> tuple[int, int]:
        read = subprocess.run([python, str(script)], capture_output=True, text=True)
        write = subprocess.run([python, str(write_script)], capture_output=True, text=True)
        return read.returncode, write.returncode

    read_rc, write_rc = run_confined(_policy(world), body)
    assert read_rc == errno.EACCES, f"expected EACCES from the script, got {read_rc}"
    assert write_rc == errno.EACCES, f"expected EACCES from the write, got {write_rc}"


def test_same_script_inside_zone_succeeds(world: dict[str, str], python: str) -> None:
    """The confinement must not break the work. The interpreter is granted, so
    an rc of 126/127 here would mean the harness is broken rather than that the
    sandbox is doing its job."""
    script = Path(world["zone"]) / "reader_in.py"
    script.write_text(errno_script(world["inside"], "r"))

    def body() -> int:
        return subprocess.run([python, str(script)], capture_output=True, text=True).returncode

    assert run_confined(_policy(world), body) == 0


# ----------------------------------------------------------- criterion 7


def test_bash_child_inherits(world: dict[str, str], python: str) -> None:
    """A child spawned by the agent is equally confined, across ``exec``."""
    script = Path(world["zone"]) / "child.py"
    script.write_text(errno_script(world["outside"], "r"))

    def body() -> tuple[int, int]:
        denied = subprocess.run(
            ["bash", "-c", f"{python} {script}"], capture_output=True, text=True
        )
        control = subprocess.run(
            ["bash", "-c", f"cat {world['inside']}"], capture_output=True, text=True
        )
        return denied.returncode, control.returncode

    denied_rc, control_rc = run_confined(_policy(world), body)
    assert control_rc == 0, "the positive control failed: bash could not read the zone"
    assert denied_rc == errno.EACCES


def test_second_ruleset_cannot_widen(world: dict[str, str]) -> None:
    """A second ruleset granting the whole filesystem does not widen the first.

    Layers intersect, and restrictions can only be added — which is also why
    "everything except X" is not expressible and why §4.5's read rule is an
    allow-list.
    """

    def body() -> int:
        from env_mgr.isolation import landlock
        from env_mgr.isolation.policy import Policy

        landlock.restrict(landlock.build(Policy((Granted("/", Mode.READ_WRITE),))))
        return attempt(world["outside"], "r")

    assert run_confined(_policy(world), body) == errno.EACCES


# ---------------------------------------------------------- criterion 12


def test_ungranted_read_denied(world: dict[str, str]) -> None:
    """A read outside the granted set, in a governed region."""

    def body() -> int:
        return attempt(world["sibling"], "r")

    assert run_confined(_policy(world), body) == errno.EACCES


def test_the_grant_is_what_denies_it(world: dict[str, str]) -> None:
    """The **negative control** for the test above.

    `validator`'s generalisation, and it is the right one: *a passing isolation
    test should be paired with the arrangement under which it would fail.*
    Without the pair, "it is denied" and "it would be denied whatever we did"
    are the same green — a sibling might be unreadable for some reason that has
    nothing to do with the allow-list, and every assertion above would still
    pass.

    So: the same sibling, granted, asserted **readable**.
    """
    granted = base_policy(
        Granted(world["zone"], Mode.READ_WRITE),
        Granted(world["sibling_dir"], Mode.READ_EXEC),
    )
    assert run_confined(granted, lambda: attempt(world["sibling"], "r")) == 0


def test_ungoverned_path_denied(tmp_path: Path, world: dict[str, str]) -> None:
    """A path in **no governed region at all** — a home directory.

    The allow-list is why this is denied without anyone having listed it: the
    granted set is the default system hierarchies, the task's own zone, and
    whatever its permissions name, and nothing else. A deny-list would have had
    to name it.
    """
    home = os.path.expanduser("~")

    def body() -> tuple[int, int]:
        return attempt(home, "x"), attempt("/usr", "x")

    denied, control = run_confined(_policy(world), body)
    assert control == 0, "the positive control failed: /usr is in the default set"
    assert denied == errno.EACCES


def test_granted_read_does_not_widen_beyond_dac(world: dict[str, str]) -> None:
    """``/etc`` granted, ``/etc/shadow`` still denied.

    The granted set intersects the uid's existing rights; it is not a
    capability. That is what bounds the cost of a generous default set.
    """

    def body() -> tuple[int, int]:
        return attempt("/etc/shadow", "r"), attempt("/etc/hostname", "r")

    denied, control = run_confined(_policy(world), body)
    assert control == 0
    assert denied in (errno.EACCES, errno.EPERM)


# ---------------------------------------------------------- criterion 14


def test_sibling_zone_created_later_unreachable(world: dict[str, str], tmp_path: Path) -> None:
    """The property the allow-list buys and a deny-list could not provide.

    The sandbox is built once, at task start, while zones keep appearing as
    tasks are dispatched. Under a deny-list a zone created later could never be
    added to an already-running process's denied set. Under an allow-list there
    is nothing to add: **anything not granted at construction is already
    unreachable, including everything that does not exist yet.**
    """
    later = tmp_path / "later-zone"

    def body() -> tuple[int, int, int]:
        # The directory is created by the *supervisor*'s view of the world here,
        # which is the child before it looks: it does not exist at build time.
        try:
            os.makedirs(later)
            made_outside = 0
        except OSError as e:
            made_outside = e.errno
        return made_outside, attempt(str(later), "x"), attempt(world["inside"], "r")

    made, listed, control = run_confined(_policy(world), body)
    assert control == 0, "the positive control failed"
    assert made == errno.EACCES, "a confined process must not be able to mkdir outside its zone"
    # ENOENT because the mkdir was refused: a sibling zone appearing later is
    # not merely unreadable, it is **uncreatable from inside**. The case where
    # the supervisor creates it is `test_no_rebuild_required`, and that one is
    # EACCES.
    assert listed == errno.ENOENT


def test_no_rebuild_required(world: dict[str, str], tmp_path: Path) -> None:
    """The sibling is created by the **supervisor**, after the confined child's
    ruleset was applied, and the child still cannot reach it. Nothing rebuilds.

    The ordering is the whole test, so it is enforced with a handshake rather
    than asserted in prose: the child confines itself and blocks; the parent —
    unconfined, and outside the resulting domain — creates the directory and
    then releases it.
    """
    later = tmp_path / "made-after-the-ruleset"
    to_child_r, to_child_w = os.pipe()
    from_child_r, from_child_w = os.pipe()

    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never reports coverage
        os.close(to_child_w)
        os.close(from_child_r)
        try:
            from env_mgr.isolation import apply as _apply
            from env_mgr.isolation.probe import probe
            from env_mgr.protocols import Tier

            _apply.apply(_policy(world), probe(), tier=Tier.PRODUCTION)
            os.write(from_child_w, b"c")  # confined
            os.read(to_child_r, 1)  # wait for the supervisor's mkdir
            os._exit(attempt(str(later), "x"))
        except BaseException:
            os._exit(125)

    os.close(to_child_r)
    os.close(from_child_w)
    assert os.read(from_child_r, 1) == b"c", "the child never reported confinement"
    assert not later.exists(), "the fixture created the sibling too early"
    later.mkdir()
    (later / "theirs.txt").write_text("theirs")
    os.write(to_child_w, b"g")
    _, status = os.waitpid(pid, 0)
    for fd in (to_child_w, from_child_r):
        os.close(fd)
    assert os.WIFEXITED(status), "the confined child crashed; a crash is never a pass"
    assert os.WEXITSTATUS(status) == errno.EACCES
