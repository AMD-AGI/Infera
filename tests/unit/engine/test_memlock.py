###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
import resource

from infera.engine import memlock


def _limits(monkeypatch, soft, hard, *, fail=False):
    calls = []

    def setrlimit(which, limits):
        if fail:
            raise ValueError("not allowed")
        calls.append((which, limits))

    monkeypatch.setattr(memlock.resource, "getrlimit", lambda which: (soft, hard))
    monkeypatch.setattr(memlock.resource, "setrlimit", setrlimit)
    return calls


def test_raises_the_soft_limit_to_an_unlimited_hard_limit(monkeypatch):
    calls = _limits(monkeypatch, 8 << 30, resource.RLIM_INFINITY)

    assert memlock.raise_memlock_soft_limit() == resource.RLIM_INFINITY
    assert calls == [(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))]


def test_raises_the_soft_limit_to_a_finite_hard_limit(monkeypatch):
    calls = _limits(monkeypatch, 64 << 10, 16 << 30)

    assert memlock.raise_memlock_soft_limit() == 16 << 30
    assert calls == [(resource.RLIMIT_MEMLOCK, (16 << 30, 16 << 30))]


def test_leaves_equal_limits_alone(monkeypatch):
    calls = _limits(monkeypatch, resource.RLIM_INFINITY, resource.RLIM_INFINITY)

    assert memlock.raise_memlock_soft_limit() is None
    assert calls == []


def test_a_refused_raise_does_not_stop_the_engine(monkeypatch):
    _limits(monkeypatch, 8 << 30, resource.RLIM_INFINITY, fail=True)

    assert memlock.raise_memlock_soft_limit() is None


def test_the_real_soft_limit_ends_at_the_hard_limit():
    memlock.raise_memlock_soft_limit()

    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    assert soft == hard
