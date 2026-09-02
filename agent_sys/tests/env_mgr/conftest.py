# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The sandbox test harness. Design §14.2 and §14.3.

Three things here, and each exists because of a measured failure:

**The harness forks; tests do not.** Landlock restriction is irreversible and
inherited, so a test that sandboxes the pytest process poisons every later test.
The kernel's answer is that isolation belongs to the runner — ``TEST_F_FORK`` was
deprecated into an alias for ``TEST_F`` because the harness forks everything.
pytest does not fork, so this supplies it once. A crash in the child is never
scored a pass, and the child ``os._exit()``s so handlers inherited from pytest do
not run inside the sandbox.

**Denials are asserted by errno against a named path.** A ``returncode != 0``
check produced a false PASS during the design measurements, because both children
were failing to exec the interpreter rather than being denied. The kernel's own
suite has zero occurrences of ``ASSERT_NE(0, …)`` on an access check.

**The mechanism is a declared input, not a discovered condition.** Spec §10: when
no sandbox mechanism is available the suite **fails**, it does not skip. Nobody
achieves that by probing at run time, so CI declares the mechanism and the ABI
and one session-scoped fixture asserts the machine matches. The gate lives in one
place because bubblewrap's equivalent leaks: its variable appears in one file and
the Python half of its own suite skips unconditionally, passing green in the CI
job that sets it.
"""

from __future__ import annotations

import errno
import os
import pickle
import struct
import sys
import textwrap
from collections.abc import Callable
from typing import Any

import pytest

from env_mgr.isolation import landlock
from env_mgr.isolation.policy import DEFAULT_SYSTEM_SET, Granted, Policy, interpreter_grants
from env_mgr.isolation.probe import Availability, probe

#: Set by CI. Absent — a developer's machine — the harness auto-detects and runs.
MECHANISM_VAR = "ENV_MGR_TEST_MECHANISM"
ABI_VAR = "ENV_MGR_TEST_ABI"

_CRASHED = "the confined child crashed; a crash is never a pass"


@pytest.fixture(scope="session")
def availability() -> Availability:
    """What this machine actually has. Fails when it has nothing.

    This is the single gate every confinement test traverses.
    """
    av = probe()
    if not av.bwrap and not av.landlock_abi:
        pytest.fail(
            "no confinement mechanism on this machine: bwrap is absent and Landlock "
            "is unavailable. The suite fails rather than skipping — these tests exist "
            "precisely to catch the case where confinement is not working (spec §10)."
        )
    return av


@pytest.fixture(scope="session")
def landlock_abi(availability: Availability) -> int:
    if not availability.landlock_abi:
        pytest.skip(
            "this machine has bwrap but no Landlock; the in-process binding cannot "
            "be exercised. Skipping environmental variation, never the property."
        )
    return availability.landlock_abi


def base_policy(*extra: Granted) -> Policy:
    """The default system set plus this interpreter, plus whatever the test adds.

    The interpreter matters: it is under ``$HOME`` on every conda / pyenv / uv /
    venv install, which the default set deliberately excludes, and without it
    ``subprocess`` fails in the *parent* naming the interpreter rather than the
    sandbox.
    """
    return Policy(tuple(DEFAULT_SYSTEM_SET) + interpreter_grants() + tuple(extra))


def attempt(path: str, mode: str) -> int:
    """0 on success, else errno. **The only way this suite observes an access.**

    `mode` is ``r``, ``w`` or ``x`` (list a directory).
    """
    try:
        if mode == "r":
            with open(path, "rb") as fh:
                fh.read(1)
        elif mode == "w":
            with open(path, "ab") as fh:
                fh.write(b"")
        elif mode == "x":
            os.listdir(path)
        else:  # pragma: no cover - a typo in a test, not a condition
            raise ValueError(f"unknown mode {mode!r}")
    except OSError as e:
        return e.errno or errno.EIO
    return 0


def run_confined(policy: Policy, body: Callable[[], Any], *, tier: Any = None) -> Any:
    """Apply `policy` in a forked child, run `body` there, return what it returns.

    The parent never restricts itself. A `pytest.fail` in the parent is the only
    outcome if the child dies on a signal.
    """
    from env_mgr.isolation import apply as _apply
    from env_mgr.protocols import Tier

    # `pickle` here is this process talking to its own fork over an anonymous
    # pipe created a line below. There is no untrusted source and no third
    # party can reach the fd; the alternative, JSON, would not carry the
    # exception and tuple shapes the tests marshal back.
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(read_fd)
        payload: tuple[str, Any]
        try:
            _apply.apply(policy, probe(), tier=tier or Tier.PRODUCTION)
            payload = ("ok", body())
        except BaseException as e:  # noqa: BLE001 - marshalled back, not swallowed
            payload = ("error", f"{type(e).__name__}: {e}")
        try:
            blob = pickle.dumps(payload)
        except Exception:  # pragma: no cover - a test returned something exotic
            blob = pickle.dumps(("error", f"unpicklable result from {body!r}"))
        os.write(write_fd, struct.pack("!I", len(blob)))
        os.write(write_fd, blob)
        os.close(write_fd)
        # `os._exit`, so pytest's inherited atexit and capture handlers do not
        # run inside the sandbox.
        os._exit(0)

    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as fh:
        header = fh.read(4)
        blob = fh.read(struct.unpack("!I", header)[0]) if len(header) == 4 else b""
    _, status = os.waitpid(pid, 0)
    if not blob:
        pytest.fail(f"{_CRASHED} (wait status {status})")
    if os.WIFSIGNALED(status):
        pytest.fail(f"{_CRASHED} (signal {os.WTERMSIG(status)})")
    kind, value = pickle.loads(blob)
    if kind == "error":
        pytest.fail(f"the confined child raised: {value}")
    return value


def errno_script(path: str, mode: str) -> str:
    """A helper script whose **exit status is the errno**. Design §14.2.

    Used for the cross-``exec`` cases, where the parent cannot observe the
    child's exception. 126/127 stay free to mean "the harness itself is broken",
    which is what makes a shell that could not start distinguishable from a
    denial.
    """
    return textwrap.dedent(
        f"""
        # Self-contained on purpose: importing the test package would need the
        # repository granted, and a failed import exits 1 — which is EPERM, and
        # would be read as a denial. That is the false-PASS trap one step over.
        import os, sys
        try:
            if {mode!r} == "r":
                open({path!r}, "rb").read(1)
            elif {mode!r} == "w":
                open({path!r}, "ab").write(b"")
            else:
                os.listdir({path!r})
        except OSError as e:
            sys.exit(e.errno or 5)
        sys.exit(0)
        """
    ).strip()


@pytest.fixture
def sandboxed(availability: Availability) -> Callable[..., Any]:
    """`run_confined`, with the session gate already traversed."""
    return run_confined


@pytest.fixture
def python() -> str:
    return sys.executable


def landlock_or_fail() -> int:
    abi = landlock.abi_version()
    if abi is None:  # pragma: no cover - covered by the session fixture
        pytest.fail("Landlock is unavailable")
    return abi


@pytest.fixture(autouse=True)
def _switch_off_the_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """**This directory asserts the enforcing mode, and now it has to say so.**

    Measured, before the default flipped: with `AGENT_SYS_NO_PERMISSIONS=1`
    exported, fourteen tests here fail — correctly, because they assert the
    enforcement the switch turns off. But the failure names the assertion rather
    than the variable, which is this module's characteristic defect (*the
    symptom names the wrong cause*) pointed at its own suite.

    So the fixture used to `delenv`, and that was enough while **unset meant
    enforced**. Since 2026-08-30 unset means *off* (`interfaces.md` §4.22f), and
    a `delenv` would hand every test here the mode it does not assert. Pinning
    the value states the mode instead of relying on a default that has now moved
    once — which is the point of the flip's test rule: a test that asserts a
    denial says which mode it asserts under.

    The deeper reason is unchanged and is `demo`'s: a run whose result changes
    with the reviewer's dotfiles is not reproducible. A test suite is a run.

    A test that wants the switch sets it with `monkeypatch.setenv`, which is
    applied after this and wins.
    """
    monkeypatch.setenv("AGENT_SYS_NO_PERMISSIONS", "0")
