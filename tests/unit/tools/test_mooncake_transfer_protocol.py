###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Exercise both endpoints with real CPU buffers and an in-memory transport."""

from __future__ import annotations

import ctypes
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from infera.tools.preflight.network import mooncakeperf as probe


@pytest.fixture
def local_transport(monkeypatch):
    class Engine:
        corrupt = False
        fail = False

        def register_memory(self, ptr, size):
            return 0

        def get_rpc_port(self):
            return 19000

        def transfer(self, sources, destinations, sizes):
            if self.fail:
                return -1
            for src, dst, size in zip(sources, destinations, sizes):
                ctypes.memmove(dst, src, size)
            if self.corrupt:
                ctypes.memset(destinations[-1], 0, 1)
            return 0

        def batch_transfer_sync_read(self, host, local, remote, sizes):
            return self.transfer(remote, local, sizes)

        def batch_transfer_sync_write(self, host, local, remote, sizes):
            return self.transfer(local, remote, sizes)

    engine = Engine()
    monkeypatch.setattr(probe, "_engine", lambda *args: engine)
    monkeypatch.setattr(probe, "_mgmt_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(probe, "_geom", lambda loc: (256, 64, 4))
    monkeypatch.setattr(probe, "_MIN_SECONDS", 0.01)
    monkeypatch.setattr(probe, "_TARGET_TIMEOUT", 3)
    monkeypatch.setattr(probe, "_DONE_TIMEOUT", 3)
    return engine


def transfer_pair(path, operation):
    with ThreadPoolExecutor(max_workers=1) as pool:
        target = pool.submit(
            probe._target, str(path), "127.0.0.1:19000", "nodeB", "tcp", operation, "cpu", -1
        )
        result = probe._initiator(str(path), "tcp", operation, "cpu", -1)
        target.result(timeout=5)
    result.update(label="tcp", target="nodeB")
    return result


@pytest.mark.parametrize("operation", ["read", "write"])
@pytest.mark.parametrize("outcome", ["success", "corrupt", "fail"])
def test_transfer_pair_verifies_received_bytes(tmp_path, local_transport, operation, outcome):
    local_transport.corrupt = outcome == "corrupt"
    local_transport.fail = outcome == "fail"
    result = transfer_pair(tmp_path, operation)
    finding = probe._finding(result, "nodeA")
    if outcome == "success":
        assert result["verified"] is True
        assert result["gb_s"] is not None
        assert finding.level != "fail"
    else:
        assert finding.level == "fail"
        if outcome == "corrupt":
            assert result["verified"] is False
        else:
            assert result["reason"] == "transfer_failed"
            assert result["gb_s"] is None
    assert finding.message.startswith(
        "nodeA -> nodeB" if operation == "write" else "nodeB -> nodeA"
    )
    assert (tmp_path / "done").exists()
    if operation == "write":
        assert (tmp_path / "verify.json").exists()
    else:
        assert not (tmp_path / "verify.json").exists()


@pytest.mark.parametrize("payload", [None, "{", "null", "[]", "{}", '{"verified": 1}'])
def test_missing_or_invalid_write_verification_fails(
    tmp_path, local_transport, monkeypatch, payload
):
    # A target that accepts the write but never supplies valid verification.
    buf = probe._Buf("cpu", -1)
    (tmp_path / "target.json").write_text(
        json.dumps({"ok": True, "hostname": "peer", "addr": buf.ptr})
    )
    if payload is not None:
        (tmp_path / "verify.json").write_text(payload)
    monkeypatch.setattr(probe, "_wait_file", lambda path, timeout: Path(path).exists())
    result = probe._initiator(str(tmp_path), "tcp", "write", "cpu", -1)
    assert result["gb_s"] is not None
    assert result["verified"] is None
    assert result["reason"] == (
        "target_verification_timeout" if payload is None else "target_verification_invalid"
    )
    result.update(label="tcp", target="nodeB")
    assert probe._finding(result, "nodeA").level == "fail"


@pytest.mark.parametrize("run_id", [".", "..", "../old", "a/b", "a b"])
def test_run_id_cannot_escape_its_directory(tmp_path, monkeypatch, run_id):
    monkeypatch.setenv("INFERA_PREFLIGHT_RUN_ID", run_id)
    assert probe._run_root(str(tmp_path)) is None


def test_run_id_overrides_job_id(tmp_path, monkeypatch):
    monkeypatch.delenv("INFERA_PREFLIGHT_RUN_ID", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert probe._run_root(str(tmp_path)) == str(tmp_path / "mooncakeperf" / "123")
    monkeypatch.setenv("INFERA_PREFLIGHT_RUN_ID", "retry-2")
    assert probe._run_root(str(tmp_path)) == str(tmp_path / "mooncakeperf" / "retry-2")


@pytest.mark.parametrize("operation", ["", "invalid"])
def test_invalid_operation_is_rejected(tmp_path, operation):
    agreed, values = probe._agree_operation(str(tmp_path), 0, 1, operation)
    assert agreed is None
    assert values == [operation]


def test_missing_rank_fails_agreement(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "_TARGET_TIMEOUT", 0)
    assert probe._agree_operation(str(tmp_path), 0, 2, "write") == (None, [])
