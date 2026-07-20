###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unit tests for `warn_if_hicache_prefetch_disabled`.

The function lives in `infera.engine.sglang.args` and exists to flag
SGLang configs where `--hicache-ratio` is small enough that
`prefetch_capacity_limit` rounds to 0 and silently disables L3
prefetch. Background: see PD design §5.4 + the 2026-05-23 TP=1
debugging session.

These tests don't depend on SGLang being installed — they pass a
SimpleNamespace masquerading as `ServerArgs`.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace


def _fake_server_args(**overrides):
    """Stand-in for SGLang's ServerArgs. Only the attributes the
    warning function reads need defaults."""
    return SimpleNamespace(
        enable_hierarchical_cache=overrides.pop("enable_hierarchical_cache", True),
        hicache_storage_backend=overrides.pop("hicache_storage_backend", "dynamic"),
        hicache_ratio=overrides.pop("hicache_ratio", 2.0),
        hicache_size=overrides.pop("hicache_size", 0),
        **overrides,
    )


# ======================================================================
# Default config — no warning
# ======================================================================


def test_no_warning_when_hicache_off(caplog):
    """hicache_off → warning is irrelevant, don't fire."""
    from infera.engine.sglang.hicache_validate import warn_if_hicache_prefetch_disabled

    args = _fake_server_args(enable_hierarchical_cache=False)
    with caplog.at_level(logging.CRITICAL):
        warn_if_hicache_prefetch_disabled(args)
    assert not any("infera:" in r.message for r in caplog.records)


def test_no_warning_when_no_storage_backend(caplog):
    """hicache_on but no storage backend → only L1+L2 in play, no L3
    prefetch path to worry about → no warning."""
    from infera.engine.sglang.hicache_validate import warn_if_hicache_prefetch_disabled

    args = _fake_server_args(hicache_storage_backend=None)
    with caplog.at_level(logging.CRITICAL):
        warn_if_hicache_prefetch_disabled(args)
    assert not any("infera:" in r.message for r in caplog.records)


def test_no_warning_at_default_ratio(caplog):
    """ratio=2.0 (SGLang default) is fine; capacity_limit = 0.8 * 1.0 *
    device_pool which is plenty of prefetch budget."""
    from infera.engine.sglang.hicache_validate import warn_if_hicache_prefetch_disabled

    args = _fake_server_args(hicache_ratio=2.0)
    with caplog.at_level(logging.CRITICAL):
        warn_if_hicache_prefetch_disabled(args)
    assert not any("infera:" in r.message for r in caplog.records)


def test_no_warning_at_large_ratio(caplog):
    """Higher ratios are explicitly safe."""
    from infera.engine.sglang.hicache_validate import warn_if_hicache_prefetch_disabled

    for ratio in (2.0, 3.0, 5.0, 10.0):
        caplog.clear()
        args = _fake_server_args(hicache_ratio=ratio)
        with caplog.at_level(logging.CRITICAL):
            warn_if_hicache_prefetch_disabled(args)
        assert not any("infera:" in r.message for r in caplog.records), ratio


# ======================================================================
# Dangerous config — warning fires
# ======================================================================


def test_warns_at_ratio_1_0(caplog):
    """The empirically-confirmed broken case: ratio=1.0 → host == device
    → capacity_limit = 0 → prefetch dies. Must surface a CRITICAL."""
    from infera.engine.sglang.hicache_validate import warn_if_hicache_prefetch_disabled

    args = _fake_server_args(hicache_ratio=1.0)
    with caplog.at_level(logging.CRITICAL):
        warn_if_hicache_prefetch_disabled(args)

    crits = [r for r in caplog.records if r.levelname == "CRITICAL"]
    assert len(crits) == 1
    msg = crits[0].message
    # The warning must (a) name the flag, (b) explain the effect, and
    # (c) point operators at a fix.
    assert "--hicache-ratio" in msg
    assert "DISABLE" in msg or "disable" in msg
    assert "prefetch" in msg
    assert "2.0" in msg  # recommended value


def test_warns_at_ratio_below_threshold(caplog):
    """Threshold is currently 1.5 — ratios in [1.0, 1.5) all warn."""
    from infera.engine.sglang.hicache_validate import warn_if_hicache_prefetch_disabled

    for ratio in (1.0, 1.1, 1.25, 1.49):
        caplog.clear()
        args = _fake_server_args(hicache_ratio=ratio)
        with caplog.at_level(logging.CRITICAL):
            warn_if_hicache_prefetch_disabled(args)
        crits = [r for r in caplog.records if r.levelname == "CRITICAL"]
        assert len(crits) == 1, f"ratio={ratio} should warn"


def test_warning_message_names_backend(caplog):
    """When the backend is set, the warning should name it so the
    operator can grep their logs for 'infera.*dynamic'."""
    from infera.engine.sglang.hicache_validate import warn_if_hicache_prefetch_disabled

    args = _fake_server_args(hicache_ratio=1.0, hicache_storage_backend="dynamic")
    with caplog.at_level(logging.CRITICAL):
        warn_if_hicache_prefetch_disabled(args)
    crits = [r for r in caplog.records if r.levelname == "CRITICAL"]
    assert any("dynamic" in r.message for r in crits)


# ======================================================================
# hicache_size override path
# ======================================================================


def test_no_warning_when_hicache_size_overrides_ratio(caplog):
    """When --hicache-size is set, SGLang uses absolute GB rather than
    the ratio formula. We can't evaluate it at arg-parse time without
    knowing GPU mem, so we bail rather than warn falsely."""
    from infera.engine.sglang.hicache_validate import warn_if_hicache_prefetch_disabled

    args = _fake_server_args(hicache_ratio=1.0, hicache_size=32)  # 32 GB explicit
    with caplog.at_level(logging.CRITICAL):
        warn_if_hicache_prefetch_disabled(args)
    assert not any("infera:" in r.message for r in caplog.records)
