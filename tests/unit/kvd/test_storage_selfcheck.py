###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for the kvd L3 storage self-check (infera.kvd.storage_selfcheck).

Covers the daemon-path (auto-resolved) call and the connector-path call where
the caller passes its OWN resolved write/load worker counts (e.g. load clamped
to 1 without P2PDMA). The probe writes/reads under a tmp dir so it runs on any
filesystem without GPU/NFS.
"""

from __future__ import annotations

import tempfile

from infera.kvd.storage_selfcheck import run_storage_selfcheck


def test_selfcheck_default_resolves_config():
    with tempfile.TemporaryDirectory() as d:
        res = run_storage_selfcheck(d, size_gb=0.05, force=True)
    assert res is not None
    assert res["write_gbps"] > 0 and res["read_gbps"] > 0
    # default path resolves a single (symmetric) worker count
    assert res["write_workers"] == res["read_workers"]
    assert res["label"] == "L3"


def test_selfcheck_disabled_by_env(monkeypatch):
    monkeypatch.setenv("INFERA_KVD_STORAGE_SELFCHECK", "0")
    with tempfile.TemporaryDirectory() as d:
        # force=False honors the disable env → None, no I/O
        assert run_storage_selfcheck(d, size_gb=0.05) is None
        # force=True overrides the disable env (e.g. an explicit probe)
        assert run_storage_selfcheck(d, size_gb=0.05, force=True) is not None


def test_selfcheck_connector_overrides_asymmetric_workers():
    """Connector path: write and load worker counts differ (load clamped to 1
    without P2PDMA). The probe must honor both and label them separately."""
    with tempfile.TemporaryDirectory() as d:
        res = run_storage_selfcheck(
            d,
            size_gb=0.05,
            force=True,
            write_workers=8,
            read_workers=1,
            o_direct=False,
            label="L3 connector",
            extra="gpu_direct=off p2pdma=no (load clamped to 1: no P2PDMA)",
        )
    assert res is not None
    assert res["write_workers"] == 8
    assert res["read_workers"] == 1
    assert res["label"] == "L3 connector"
    assert res["write_gbps"] > 0 and res["read_gbps"] > 0
