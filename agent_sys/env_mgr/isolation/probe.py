# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which mechanism is available, and which one to use. Design §4.2."""

from __future__ import annotations

import shutil
from typing import Literal, NamedTuple

from env_mgr.isolation import landlock
from env_mgr.protocols import NoConfinement

__all__ = ["Availability", "probe", "select"]


class Availability(NamedTuple):
    bwrap: str | None  # path to the binary, or None
    landlock_abi: int | None  # or None


def probe() -> Availability:
    """The real one. Called by `apply`'s caller, never by `select`."""
    return Availability(bwrap=shutil.which("bwrap"), landlock_abi=landlock.abi_version())


def select(av: Availability) -> Literal["bwrap", "landlock"]:
    """bubblewrap when present, Landlock otherwise, **refuse** when neither.

    Taking an `Availability` rather than calling `probe()` is the whole trick,
    and it exists because of a measurement: **no machine runs all three
    branches** (criterion 9). ``bwrap`` is absent here, so rung 1 cannot be
    exercised; and there is no ordinary way to make a Landlock-capable kernel
    look incapable, so rung 3 cannot be exercised wherever rung 2 works. With
    the input injected the three branches are three one-line unit tests, and one
    end-to-end test runs against whatever the machine actually has.
    """
    if av.bwrap:
        return "bwrap"
    if av.landlock_abi:
        return "landlock"
    raise NoConfinement(
        "no confinement mechanism: bwrap is absent and Landlock is unavailable. "
        "An agent started without confinement runs with the operator's full "
        "privileges while the system reports it is sandboxed, so the task does "
        "not start."
    )
