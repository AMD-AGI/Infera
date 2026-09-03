# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The granted set, mechanism-independent. Design §5.

One `Policy` builds either a bubblewrap argv or a Landlock ruleset, and nothing
above ``isolation/`` knows which will consume it.
"""

from __future__ import annotations

import os
import sys
from typing import Any, NamedTuple

from env_mgr.fs.path import contained, resolve_strict
from env_mgr.fs.zone import Zone

# `Mode` and `Granted` are inert and are declared once, in `protocols.py`.
# `Policy` is not: `with_` has behaviour, and a declaration whose body is `...`
# would return `None`.
from env_mgr.protocols import Granted, Mode

__all__ = [
    "BIN_DIRS",
    "DEFAULT_SYSTEM_SET",
    "Granted",
    "Mode",
    "Policy",
    "agent_cli_grants",
    "anchor_zone_root",
    "component_grants",
    "executable_path",
    "interpreter_grants",
]


class Policy(NamedTuple):
    """`optional` is per entry rather than global.

    bubblewrap exposes exactly this choice as two flags, ``--ro-bind`` versus
    ``--ro-bind-try``, and uses both side by side; Landlock's ecosystem helper
    makes the fail-open choice silently for every path, so a typo in an
    allow-list evaporates. Neither default is right, because the two cases are
    genuinely different — `/lib64` on a merged-`/usr` distro is expected, and a
    path a task's permissions named is an error.
    """

    #: **No default, matching `protocols.Policy`.** It defaulted to `()` here and
    #: not there, which no caller used — every construction in the repository
    #: passes a granted set — and which quietly made *grant nothing* the easiest
    #: thing to write in the module whose central rule is the opposite:
    #: `UnresolvedGrant` exists because a grant that resolves to nothing is
    #: *"raised rather than resolving to an empty granted set"* (`protocols.py:66`).
    #: A convenience that makes the refused state the default state is not one.
    granted: tuple[Granted, ...]

    def with_(self, *more: Granted) -> Policy:
        return Policy(self.granted + tuple(more))


#: Spec §4.5.1, corrected by measurement (design §5.2, deviation D2).
#:
#: Without ``/dev/null`` **writable**, git dies before reaching any repository
#: question: ``fatal: could not open '/dev/null' for reading and writing``.
#: ``/dev/urandom`` is the one nobody guesses — the Claude backend here is a
#: standalone Bun binary that aborts in 3 ms without it and blames itself, with
#: a crash-report URL for the wrong project. ``/run/systemd/resolve`` is the
#: second: ``/etc/resolv.conf`` is a symlink out of ``/etc`` and Landlock rules
#: apply to the **resolved** path, so granting ``/etc`` does not give you DNS.
#:
#: A default **in configuration, not a constant in code** (spec §4.5.1): a site
#: may narrow or widen it. A home directory is deliberately absent.
DEFAULT_SYSTEM_SET: tuple[Granted, ...] = (
    Granted("/usr", Mode.READ_EXEC),
    Granted("/lib", Mode.READ_EXEC, optional=True),
    Granted("/lib64", Mode.READ_EXEC, optional=True),
    Granted("/bin", Mode.READ_EXEC),
    Granted("/sbin", Mode.READ_EXEC, optional=True),
    Granted("/etc", Mode.READ_EXEC),
    Granted("/proc", Mode.READ_EXEC),
    Granted("/dev/null", Mode.READ_WRITE),
    Granted("/dev/zero", Mode.READ_EXEC),
    Granted("/dev/urandom", Mode.READ_EXEC),
    Granted("/dev/random", Mode.READ_EXEC, optional=True),
    Granted("/dev/full", Mode.READ_EXEC, optional=True),
    Granted("/dev/tty", Mode.READ_WRITE, optional=True),
    Granted("/run/systemd/resolve/stub-resolv.conf", Mode.READ_EXEC, optional=True),
)


def interpreter_grants(executable: str | None = None) -> tuple[Granted, ...]:
    """The interpreter's own prefix, read-execute.

    Measured: with the interpreter under ``$HOME`` — conda, pyenv, uv, venv, so
    every ordinary install — ``subprocess`` fails in the **parent** with
    ``PermissionError`` naming the interpreter rather than the sandbox. The
    default set can never be sufficient alone, which the spec does not say.

    Resolved at prepare time so the failure mode is a missing declaration
    there rather than an ``EACCES`` at exec time.
    """
    exe = executable or sys.executable
    prefixes = {sys.base_prefix, sys.prefix, os.path.dirname(os.path.dirname(exe))}
    return tuple(
        Granted(p, Mode.READ_EXEC, optional=True)
        for p in sorted(prefixes)
        if p and p not in ("/", "/usr")
    )


#: The conventional executable directories, in the order every POSIX shell
#: writes them. Order matters and is not ours to reinvent.
BIN_DIRS = (
    "/usr/local/sbin",
    "/usr/local/bin",
    "/usr/sbin",
    "/usr/bin",
    "/sbin",
    "/bin",
)


def executable_path(policy: Policy) -> str:
    """``PATH``, **projected from the granted set** rather than chosen.

    **`PATH` is not a boundary and this does not pretend it is one.** Measured
    (`scratch/impl-2026-08/env_mgr/p2_path_is_not_a_boundary.py`): with ``/usr``
    granted and ``PATH=""`` a confined process runs ``/usr/bin/git`` happily,
    and with ``/usr`` **un**granted but named in ``PATH`` the exec is ``EACCES``.
    What a process may execute is decided by the allow-list, in the kernel;
    removing an entry from ``PATH`` buys exactly nothing.

    What it is for is **determinism**. A process handed an environment block
    with no ``PATH`` does not get an empty one: POSIX ``sh`` substitutes a
    built-in default, and that value is a property of the shell binary — ``dash``
    here answers ``/usr/local/sbin:…:/bin`` while ``confstr(_CS_PATH)`` answers
    ``/bin:/usr/bin``. So two hosts resolve ``python3`` differently and nothing
    in the run record says which. Reported by `validator`, who measured it and
    correctly declined to pick a value.

    Deriving it from the policy makes one invariant structural: **`PATH` can
    never name a directory the kernel will refuse.** That is worth more than the
    determinism, because §5.5 measured what the alternative costs — an
    ungranted-but-existing path makes a tool report itself broken rather than
    report the file as absent, and the symptom names the wrong cause.

    A granted prefix's own ``bin/`` is included, which is the conda / pyenv /
    uv / venv case: the interpreter is under ``$HOME``, the default set excludes
    it, and `interpreter_grants` is what puts it back.
    """
    granted = tuple(g.path for g in policy.granted)
    dirs: list[str] = []
    for candidate in BIN_DIRS + tuple(os.path.join(g, "bin") for g in granted):
        if candidate in dirs or not os.path.isdir(candidate):
            continue
        if any(contained(candidate, root) for root in granted):
            dirs.append(candidate)
    return os.pathsep.join(dirs)


def agent_cli_grants(agent_cli: str | None) -> tuple[Granted, ...]:
    """The agent backend's own CLI, read-execute. **Derived, never hand-written.**

    The install directory rather than the binary, and the depth is adopted from
    `demo`'s measurement rather than invented: it lives under ``$HOME`` on every
    ordinary install, ``$HOME`` is granted nowhere, and this is the narrowest
    grant they found that keeps that true. Narrowing it further would be
    guessing at what a self-contained backend reads beside itself, which is
    design O3's *"nothing enumerates what the next tool probes"*.

    Derived here for the reason `stub-resolv` taught: a grant every run needs,
    written once per caller, is a grant one caller forgets.
    """
    if agent_cli is None:
        return ()
    resolved = resolve_strict(agent_cli)
    if resolved is None:
        raise ValueError(
            f"the declared agent CLI {agent_cli!r} does not resolve. It is declared "
            f"rather than discovered on purpose — see `Context.agent_cli`."
        )
    return (Granted(os.path.dirname(os.path.dirname(resolved)), Mode.READ_EXEC),)


def component_grants(agent_spec: Any) -> tuple[Granted, ...]:
    """`agent_sys/components/`, read-only, **iff this agent declares any.**

    `agent_cli_grants`' shape and its reason, one row down: *a grant every run
    needs, written once per caller, is a grant one caller forgets.* This one is
    needed by every run whose agent names an L2 component, and by no other.

    **Conditional, and the condition is the same one that exports the name.**
    `agent_assets.install` emits ``AGENT_SYS_COMPONENTS_ROOT`` under
    ``if _sequence(agent_spec, "components")`` and this emits the grant under the
    identical test, so `paths.py`'s rule — exported and granted agree by
    construction — cannot be broken by one of the two becoming unconditional. A
    run declaring no component gets neither, which is every package in the tree
    but one.

    **`READ_EXEC`, not `READ_WRITE`, and read is the ceiling on purpose.** A
    component is *read* from here and copied into the zone before anything runs
    it: skills are copied to ``<zone>/config/skills``, a marketplace is copied
    before it is registered (probe F), and a bundled MCP server is launched by
    absolute path under the supervisor's own interpreter. If something ever has
    to execute out of this directory the answer is to copy that component into
    the zone as well, not to widen this.

    Returns `()` for a spec that is `None` or declares nothing, so the caller
    composes it unconditionally and never branches.
    """
    if agent_spec is None:
        return ()
    declared = agent_spec.get("components") if isinstance(agent_spec, dict) else None
    if declared is None:
        declared = getattr(agent_spec, "components", None)
    if not declared:
        return ()
    # Imported at use time rather than at module scope: `env_mgr.agent_assets`
    # is above this package in the graph and importing it here would be an edge
    # from `isolation/` to a module that reads agent specs. The name is the
    # dependency (`engineer_principle.md` §1's last row).
    from env_mgr.agent_assets import COMPONENTS_ROOT  # noqa: PLC0415

    return (Granted(COMPONENTS_ROOT, Mode.READ_EXEC, optional=True),)


def anchor_zone_root(proposed: str | None, zone: Zone) -> str:
    """Criterion 10. **Never derive the zone root from model-supplied input.**

    A model-controlled ``cwd`` / ``working_directory`` was CVE-2025-59532 in
    Codex and CVE-2026-50548 in Cursor, both CVSS 9.8. A proposal inside the
    task's own subtree is honoured; anything else is refused, and the answer is
    anchored to what the harness started with either way.
    """
    if proposed is None:
        return zone.root
    if not contained(proposed, zone.root):
        raise PermissionError(
            f"working directory {proposed!r} is outside zone {zone.root!r}; "
            f"the zone root is never taken from agent-supplied input"
        )
    return zone.root if proposed == zone.root else os.path.realpath(proposed)
