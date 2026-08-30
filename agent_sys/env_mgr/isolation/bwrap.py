# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The bubblewrap argument builder. Design §4.3.

There is no binding to write: the mechanism is a process, and what it takes is
an argument vector. ``bwrap`` is **absent on the development machine**, so every
test over this module is a test of the argv it produces.
"""

from __future__ import annotations

from collections.abc import Sequence

from env_mgr.isolation.policy import Mode, Policy

__all__ = ["argv"]


def argv(policy: Policy, *, bwrap: str, command: Sequence[str] = ()) -> list[str]:
    """``--ro-bind`` / ``--bind`` from a `Policy`, plus the two unshares.

    ``--unshare-net`` and ``--unshare-pid`` are where rung 1's extra properties
    come from: Landlock at ABI 3 has neither, so the chain degrades in
    *properties* and not only in preference, and `Confinement` reports which.

    The ``-try`` suffix is taken from bubblewrap's own tests, which use
    ``--ro-bind-try`` for exactly the paths that legitimately do not exist on
    some hosts. Here it follows `Granted.optional`, so both rungs agree about
    which entries may be missing.
    """
    out = [bwrap, "--unshare-net", "--unshare-pid", "--die-with-parent", "--proc", "/proc"]
    for entry in policy.granted:
        if entry.path == "/proc":
            continue  # --proc mounts a fresh one; binding it too is a conflict
        write = bool(entry.mode & Mode.READ_WRITE)
        flag = "--bind" if write else "--ro-bind"
        if entry.optional:
            flag += "-try"
        out += [flag, entry.path, entry.path]
    if command:
        out += ["--", *command]
    return out
