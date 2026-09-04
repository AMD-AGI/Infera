###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import asyncio

import pytest

from infera.engine.base import EngineDeath, watch_engine_death


class _DeadEngine:
    """Provides the process contract consumed by the death watcher."""

    _proc = object()

    def __init__(self, returncode: int | None) -> None:
        self.returncode = returncode

    async def wait(self) -> int | None:
        return self.returncode


@pytest.mark.parametrize(
    ("returncode", "exit_status"),
    [(-9, 137), (-15, 143), (7, 7), (0, 1), (None, 1)],
)
def test_engine_death_maps_subprocess_returncode(returncode, exit_status):
    """Signal exits retain their conventional POSIX shell status."""
    death = EngineDeath(observed=True, returncode=returncode)
    assert death.exit_status == exit_status


def test_unobserved_engine_death_has_no_exit_status():
    """An operator shutdown remains a successful launcher exit."""
    assert EngineDeath().exit_status is None


@pytest.mark.asyncio
async def test_watcher_records_death_before_requesting_shutdown():
    """The launcher can distinguish engine death from an operator signal."""
    stop = asyncio.Event()
    death = EngineDeath()

    task = watch_engine_death(_DeadEngine(-9), stop, death)
    await task

    assert stop.is_set()
    assert death.returncode == -9
    assert death.exit_status == 137
