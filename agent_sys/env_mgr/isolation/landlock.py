# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Landlock ctypes binding. Design §4.4.

**Why this is written by hand.** There is no maintained Python binding for
Landlock; the canonical one is `rust-landlock`, and the three syscalls have no
libc wrapper at all, so *any* Python binding is ``syscall(2)`` by number
regardless of who writes it. This is promoted from the measuring instrument at
``scratch/design/probes-envmgr/landlock.py`` that took every measurement the
design cites.

Three details that are not obvious and each cost a measurement:

**(a) The rights mask depends on what the target is.** A directory-only right
(``MAKE_REG``, ``READ_DIR``, ``MAKE_DIR``, …) for a non-directory target is
``EINVAL``, not an ignored bit — so an implementation that grants uniformly dies
on the first ``/dev/null`` it is handed, with no indication which path caused it.
`rust-landlock` does the same ``fstat``-and-mask, with the comment
``// Linux would return EINVAL.`` two lines above (``src/fs.rs:316``).

**(b) The rights mask also depends on the ABI.** Handing a bit the kernel does
not know is ``EINVAL``.

**(c) ``restrict_self()`` restricts only the calling thread below ABI 8.**
``all_threads()`` arrives at ABI 8; at ABI 3 it does not exist, and a
best-effort implementation silently leaves sibling threads unrestricted **while
the status still reports enforced**. So confinement is applied before any thread
is started, and a test asserts it.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from enum import Enum
from typing import NamedTuple

from env_mgr.isolation.policy import Granted, Mode, Policy

__all__ = [
    "Enforced",
    "LayerLimitReached",
    "MAX_LAYERS",
    "RestrictionStatus",
    "Ruleset",
    "WANTED_ABI",
    "abi_version",
    "available",
    "build",
    "restrict",
]

# x86-64 syscall numbers. There is no libc wrapper for any of the three.
_NR_CREATE_RULESET = 444
_NR_ADD_RULE = 445
_NR_RESTRICT_SELF = 446
_NR_PRCTL = 157

_PR_SET_NO_NEW_PRIVS = 38
_CREATE_RULESET_VERSION = 1 << 0
_RULE_PATH_BENEATH = 1

A_EXECUTE = 1 << 0
A_WRITE_FILE = 1 << 1
A_READ_FILE = 1 << 2
A_READ_DIR = 1 << 3
A_REMOVE_DIR = 1 << 4
A_REMOVE_FILE = 1 << 5
A_MAKE_CHAR = 1 << 6
A_MAKE_DIR = 1 << 7
A_MAKE_REG = 1 << 8
A_MAKE_SOCK = 1 << 9
A_MAKE_FIFO = 1 << 10
A_MAKE_BLOCK = 1 << 11
A_MAKE_SYM = 1 << 12
A_REFER = 1 << 13  # ABI 2
A_TRUNCATE = 1 << 14  # ABI 3

#: What a given ABI level accepts in ``handled_access_fs``. (b) above.
_BY_ABI = {1: (1 << 13) - 1, 2: (1 << 14) - 1, 3: (1 << 15) - 1}

#: The highest ABI whose rights this module uses. Running below it means some
#: requested right could not be handled, which is `Enforced.PARTIALLY`.
WANTED_ABI = 3

#: Measured on Linux 6.5.0-45: sixteen ``restrict_self`` calls stack and the
#: seventeenth is ``E2BIG``, matching the kernel's ``LANDLOCK_MAX_NUM_LAYERS``.
#: The man page's 64 is a later kernel's raised limit, not this one's — so the
#: number is a property of the running kernel and not of the API. Recorded
#: because design §8.4's architecture choice turns on it: the supervisor spawns
#: each executor directly, so every executor carries exactly one layer.
MAX_LAYERS = 16

_READ_EXEC = A_EXECUTE | A_READ_FILE | A_READ_DIR
#: The only rights that may be granted when the target is not a directory. (a).
_FILE_RIGHTS = A_EXECUTE | A_WRITE_FILE | A_READ_FILE | A_TRUNCATE
_FULL_WRITE = (
    A_WRITE_FILE
    | A_REMOVE_DIR
    | A_REMOVE_FILE
    | A_MAKE_CHAR
    | A_MAKE_DIR
    | A_MAKE_REG
    | A_MAKE_SOCK
    | A_MAKE_FIFO
    | A_MAKE_BLOCK
    | A_MAKE_SYM
    | A_TRUNCATE
)

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.syscall.restype = ctypes.c_long


class _RulesetAttr(ctypes.Structure):
    # ABI 1-3 knows only the first field; passing size=8 selects that layout.
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1  # the kernel struct is __packed__: 12 bytes, not 16
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class LayerLimitReached(RuntimeError):
    """``E2BIG``: this thread already carries `MAX_LAYERS` Landlock domains."""


def _call(nr: int, *args: object) -> int:
    _libc.syscall.argtypes = [ctypes.c_long] + [ctypes.c_void_p] * len(args)
    ctypes.set_errno(0)
    rc = _libc.syscall(nr, *args)
    if rc < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"landlock syscall {nr}: {os.strerror(err)}")
    return int(rc)


def abi_version() -> int | None:
    """The running kernel's Landlock ABI, or ``None`` when unavailable."""
    try:
        return _call(_NR_CREATE_RULESET, None, 0, _CREATE_RULESET_VERSION)
    except OSError:
        return None


def available() -> bool:
    return abi_version() is not None


def _rights(mode: Mode, mask: int) -> int:
    bits = _READ_EXEC
    if mode & Mode.READ_WRITE:
        bits |= _FULL_WRITE
    return bits & mask


class Ruleset(NamedTuple):
    """A built, not-yet-applied ruleset. `fd` is owned by the caller."""

    fd: int
    abi: int
    added: tuple[str, ...]
    dropped: tuple[str, ...]


def build(policy: Policy) -> Ruleset:
    """Every entry, or an error — deviation from the ecosystem's default.

    `rust-landlock`'s ``path_beneath_rules`` does ``Err(_) => None`` on a path it
    cannot open: no rule, and no error. Under deny-by-default a vanished grant
    is not an escalation, but an allow-list typo silently evaporates, and spec
    principle 3 is *cannot canonicalise a path, cannot obtain a sandbox, cannot
    decide — deny*. So a non-`optional` entry that cannot be opened raises, and
    `optional` decides that per entry rather than per implementation accident.
    """
    abi = abi_version()
    if abi is None:
        raise OSError("landlock is not available on this kernel")
    mask = _BY_ABI.get(min(abi, WANTED_ABI), _BY_ABI[WANTED_ABI])
    attr = _RulesetAttr(handled_access_fs=mask)
    fd = _call(_NR_CREATE_RULESET, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    added: list[str] = []
    dropped: list[str] = []
    try:
        for entry in policy.granted:
            if _add(fd, entry, mask):
                added.append(entry.path)
            else:
                dropped.append(entry.path)
    except BaseException:
        os.close(fd)
        raise
    return Ruleset(fd=fd, abi=abi, added=tuple(added), dropped=tuple(dropped))


def _add(fd: int, entry: Granted, mask: int) -> bool:
    try:
        pfd = os.open(entry.path, os.O_PATH | os.O_CLOEXEC)
    except OSError as e:
        if entry.optional:
            return False
        raise OSError(e.errno, f"granted path {entry.path!r} cannot be opened: {e.strerror}") from e
    try:
        access = _rights(entry.mode, mask)
        if not os.path.isdir(entry.path):
            access &= _FILE_RIGHTS
        rule = _PathBeneathAttr(allowed_access=access, parent_fd=pfd)
        _call(_NR_ADD_RULE, fd, _RULE_PATH_BENEATH, ctypes.byref(rule), 0)
    finally:
        os.close(pfd)
    return True


class Enforced(str, Enum):
    NOTHING = "nothing"
    PARTIALLY = "partially"
    FULLY = "fully"


class RestrictionStatus(NamedTuple):
    enforced: Enforced
    abi: int
    dropped: tuple[str, ...]


def restrict(ruleset: Ruleset) -> RestrictionStatus:
    """Apply `ruleset` to this thread and every descendant. **Irreversible.**

    Closes the ruleset fd. ``PR_SET_NO_NEW_PRIVS`` first, because
    ``restrict_self`` requires it without ``CAP_SYS_ADMIN``.
    """
    try:
        _call(_NR_PRCTL, _PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
        try:
            _call(_NR_RESTRICT_SELF, ruleset.fd, 0)
        except OSError as e:
            if e.errno == 7:  # E2BIG
                raise LayerLimitReached(
                    f"this thread already carries {MAX_LAYERS} Landlock layers"
                ) from e
            raise
    finally:
        os.close(ruleset.fd)
    if not ruleset.added:
        return RestrictionStatus(Enforced.NOTHING, ruleset.abi, ruleset.dropped)
    if ruleset.dropped or ruleset.abi < WANTED_ABI:
        return RestrictionStatus(Enforced.PARTIALLY, ruleset.abi, ruleset.dropped)
    return RestrictionStatus(Enforced.FULLY, ruleset.abi, ruleset.dropped)
