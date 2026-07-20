###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unit tests for the auto I/O worker resolution and the SAVE-bounce flag
added in the kvd NFS-I/O PR.

`INFERA_KVD_LOAD_WORKERS` / `INFERA_KVD_SAVE_WORKERS` default to ``auto``
→ the L3 mount's ``nconnect`` (fallback 16); an explicit int overrides. These
exercise `_resolve_io_workers` / `_detect_l3_nconnect` directly on a bare
instance (no daemon / GPU needed).
"""

from __future__ import annotations

import builtins
import io

from infera.engine.vllm.kvd_connector import InferaKvdConnector as K

ENV = "INFERA_KVD_LOAD_WORKERS"


def _bare() -> K:
    # _resolve_io_workers only touches os.environ + self._detect_l3_nconnect
    return K.__new__(K)


def test_explicit_int_env_wins(monkeypatch):
    monkeypatch.setenv(ENV, "4")
    monkeypatch.setattr(K, "_detect_l3_nconnect", staticmethod(lambda: 16))
    assert _bare()._resolve_io_workers(ENV) == 4


def test_auto_uses_mount_nconnect(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(K, "_detect_l3_nconnect", staticmethod(lambda: 16))
    assert _bare()._resolve_io_workers(ENV) == 16


def test_auto_literal_uses_nconnect(monkeypatch):
    monkeypatch.setenv(ENV, "auto")
    monkeypatch.setattr(K, "_detect_l3_nconnect", staticmethod(lambda: 8))
    assert _bare()._resolve_io_workers(ENV) == 8


def test_fallback_16_when_no_nconnect(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(K, "_detect_l3_nconnect", staticmethod(lambda: None))
    assert _bare()._resolve_io_workers(ENV) == 16


def test_invalid_env_falls_back_to_nconnect(monkeypatch):
    monkeypatch.setenv(ENV, "notanint")
    monkeypatch.setattr(K, "_detect_l3_nconnect", staticmethod(lambda: 12))
    assert _bare()._resolve_io_workers(ENV) == 12


def test_detect_nconnect_parses_and_strips_whitespace(monkeypatch, tmp_path):
    # env with a space after the comma — the path must be stripped or the
    # mount lookup fails.
    monkeypatch.setenv("INFERA_KVD_HIPFILE_ROOTS", f"long={tmp_path}, short=/other")
    # minimal /proc/self/mountinfo line for tmp_path with nconnect=16
    line = f"36 35 0:1 / {tmp_path} rw - nfs4 srv:/export rw,vers=4.2,nconnect=16,proto=rdma\n"
    real_open = builtins.open

    def fake_open(p, *a, **k):
        if str(p) == "/proc/self/mountinfo":
            return io.StringIO(line)
        return real_open(p, *a, **k)

    monkeypatch.setattr(builtins, "open", fake_open)
    assert K._detect_l3_nconnect() == 16


class _Spec:
    def __init__(self, block_size):
        self.block_size = block_size


class _Group:
    def __init__(self, block_size):
        self.kv_cache_spec = _Spec(block_size)


class _Cfg:
    def __init__(self, block_size):
        self.block_size = block_size


def test_logical_block_size_prefers_spec():
    # MLA: spec carries the logical block size; tensor shape dim (1) is
    # irrelevant here — the helper never sees the tensor.
    assert K._logical_block_size(_Group(16), _Cfg(0)) == 16


def test_logical_block_size_falls_back_to_cfg():
    # spec missing block_size -> use cfg.block_size (same chain the
    # scheduler bootstrap uses, so the two sides stay in sync).
    assert K._logical_block_size(_Group(0), _Cfg(16)) == 16


def test_logical_block_size_default_16():
    assert K._logical_block_size(_Group(0), _Cfg(0)) == 16


def test_logical_block_size_handles_missing_spec():
    class _NoSpec:
        kv_cache_spec = None

    assert K._logical_block_size(_NoSpec(), _Cfg(64)) == 64


def test_detect_nconnect_none_for_non_nfs(monkeypatch, tmp_path):
    monkeypatch.setenv("INFERA_KVD_HIPFILE_ROOTS", f"long={tmp_path}")
    line = f"36 35 0:1 / {tmp_path} rw - ext4 /dev/nvme0n1p1 rw\n"
    real_open = builtins.open

    def fake_open(p, *a, **k):
        if str(p) == "/proc/self/mountinfo":
            return io.StringIO(line)
        return real_open(p, *a, **k)

    monkeypatch.setattr(builtins, "open", fake_open)
    assert K._detect_l3_nconnect() is None
