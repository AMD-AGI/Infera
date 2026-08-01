###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""``INFERA_SGLANG_READY_TIMEOUT`` overrides the readiness deadline.

The hardcoded 30 min covers weight load plus cuda-graph capture for the models
we ship recipes for, but not a very large checkpoint read twice -- MTP re-reads
the weights to extract the draft layer -- and least of all when both PD legs
load off one filesystem at once. That leg is killed while it is still making
progress and the whole bring-up restarts, so the deadline has to be raisable
without editing a launcher.

The resolved value is observable without waiting for it: ``_wait_ready`` puts it
in the ``TimeoutError`` it raises. These tests drive the timeout path with a
clock that has already jumped past the deadline, so a 1800 s default is asserted
in milliseconds.

Two deliberate choices about how this runs:

* ``importorskip("sglang")`` -- ``infera.engine.sglang.worker`` imports
  ``sglang.srt.server_args`` at module load, so this runs in the SGLang
  container and skips on a bare dev box. Same pattern as
  ``test_decode_radix_vs_speculative.py``.
* ``asyncio.run`` in sync test functions rather than ``@pytest.mark.asyncio``
  -- the engine image ships pytest without ``pytest-asyncio``, where that marker
  is silently unknown and every async test is skipped-as-passed. These have to
  actually execute in the image they guard.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("sglang")

from infera.engine.sglang import worker as worker_mod  # noqa: E402
from infera.engine.sglang.worker import SglangEngine  # noqa: E402


class _JumpedClock:
    """A loop clock that is already past any deadline after the first read.

    ``_wait_ready`` reads ``loop.time()`` once to set the deadline and again in
    the ``while`` guard. Returning 0.0 then a huge value skips the poll loop
    entirely, so the timeout branch is reached with no real waiting.
    """

    def __init__(self) -> None:
        self._reads = 0

    def time(self) -> float:
        self._reads += 1
        return 0.0 if self._reads == 1 else 1e9


@pytest.fixture
def jumped_clock(monkeypatch):
    monkeypatch.setattr(worker_mod.asyncio, "get_running_loop", lambda: _JumpedClock())
    yield


def _engine() -> SglangEngine:
    """An engine shell carrying only what ``_wait_ready`` reads."""
    eng = SglangEngine.__new__(SglangEngine)
    eng.server_args = SimpleNamespace(host="127.0.0.1", port=1)
    eng._proc = None
    return eng


@pytest.mark.parametrize(
    "env, expected",
    [
        (None, 1800.0),  # unset -> the shipped default is unchanged
        ("5400", 5400.0),  # raised for a big checkpoint
        ("", 1800.0),  # empty means unset, not "0 seconds"
        ("not-a-number", None),  # a typo must not be silently treated as 0
    ],
)
def test_timeout_comes_from_the_env(monkeypatch, jumped_clock, env, expected):
    if env is None:
        monkeypatch.delenv("INFERA_SGLANG_READY_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("INFERA_SGLANG_READY_TIMEOUT", env)

    eng = _engine()
    if expected is None:
        with pytest.raises(ValueError):
            asyncio.run(eng._wait_ready())
        return
    with pytest.raises(TimeoutError, match=rf"not ready after {expected}s"):
        asyncio.run(eng._wait_ready())


def test_explicit_argument_beats_the_env(monkeypatch, jumped_clock):
    """Flag > env > default, the same precedence the rest of the CLI uses."""
    monkeypatch.setenv("INFERA_SGLANG_READY_TIMEOUT", "5400")
    eng = _engine()
    with pytest.raises(TimeoutError, match=r"not ready after 7\.0s"):
        asyncio.run(eng._wait_ready(7.0))
