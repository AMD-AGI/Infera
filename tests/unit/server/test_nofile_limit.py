###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The router lifts its open-file soft limit to the hard limit at start-up.

A container's default soft limit (1024) is below the connection count of a few
hundred concurrent PD requests (client + prefill + decode legs each).
"""

from __future__ import annotations

import resource

import pytest

from infera.server.__main__ import raise_nofile_limit


def _fake_limits(monkeypatch: pytest.MonkeyPatch, soft: int, hard: int) -> list[tuple[int, int]]:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(resource, "getrlimit", lambda _: (soft, hard))
    monkeypatch.setattr(resource, "setrlimit", lambda _, lim: calls.append(lim))
    return calls


def test_soft_limit_goes_up_to_hard(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_limits(monkeypatch, 1024, 1048576)
    raise_nofile_limit()
    assert calls == [(1048576, 1048576)]


def test_unlimited_hard_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_limits(monkeypatch, 1024, resource.RLIM_INFINITY)
    raise_nofile_limit()
    assert calls == [(1 << 20, resource.RLIM_INFINITY)]


def test_already_at_hard_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_limits(monkeypatch, 65536, 65536)
    raise_nofile_limit()
    assert calls == []


def test_refused_raise_does_not_abort_start_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resource, "getrlimit", lambda _: (1024, 4096))

    def refuse(*_):
        raise ValueError("not allowed")

    monkeypatch.setattr(resource, "setrlimit", refuse)
    raise_nofile_limit()
