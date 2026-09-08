###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""KV-transfer probe diagnostics (Mooncake / Mori).

A cross-node KV transfer can "succeed" (return a bandwidth) yet fail byte
verification. On the ionic stack the usual cause is a SILENT registration
failure: ``register_memory`` returns 0 even though libibverbs could not pin the
host-DRAM buffer (``Cannot allocate memory [12]`` / ENOMEM), so the buffer's
tail never transfers. These tests lock in that the errno is scraped from the two
libibverbs log formats and folded into the "data mismatch" finding, so the
report names the real cause instead of a bare mismatch.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from infera.tools.preflight.network import mooncakeperf, moriperf
from infera.tools.preflight.network.netperf import _parse_rdma_errno

_MOONCAKE_ENOMEM = (
    "E0717 rdma_context.cpp:243] Failed to register memory 0x72439ffff010: "
    "Cannot allocate memory [12]"
)
_MORI_EFAULT = "RegisterRdmaMemoryRegion failed! errno:14"


def test_parse_errno_keyword_form():
    r = _parse_rdma_errno(_MORI_EFAULT)
    assert r is not None
    assert "errno 14" in r and "EFAULT" in r


def test_parse_errno_bracket_form():
    # mooncake's format: bracketed errno, no "errno" keyword.
    r = _parse_rdma_errno(_MOONCAKE_ENOMEM)
    assert r is not None
    assert "errno 12" in r and "ENOMEM" in r


def test_parse_errno_ignores_unrelated_bracket():
    # A bracketed number with no registration keyword must not be misattributed.
    assert _parse_rdma_errno("Topology discovery complete. Found 1 HCAs [12]") is None


def test_parse_errno_none_when_absent():
    assert _parse_rdma_errno("") is None
    assert _parse_rdma_errno("nothing to see here") is None


def _mismatch_rec(label: str, reg_error):
    return {
        "label": label,
        "target": "nodeB",
        "gb_s": 40.0,
        "gib": 3.0,
        "loc": "cpu",
        "gpu": -1,
        "verified": False,
        "reg_error": reg_error,
    }


def test_mooncake_finding_surfaces_hidden_register_enomem():
    reg = _parse_rdma_errno(_MOONCAKE_ENOMEM)
    f = mooncakeperf._finding(_mismatch_rec("rdma", reg), "nodeA")
    assert f.level == "fail"
    assert "data mismatch" in f.details["reason"]
    assert "ENOMEM" in f.details["reason"]
    assert "could not pin the buffer" in f.details["reason"]


def test_mori_finding_surfaces_hidden_register_enomem():
    reg = _parse_rdma_errno(_MOONCAKE_ENOMEM)
    f = moriperf._finding(_mismatch_rec("cpu", reg), "nodeA")
    assert f.level == "fail"
    assert "ENOMEM" in f.details["reason"]


def test_finding_plain_mismatch_when_no_register_error():
    # No scraped errno -> the reason stays the plain mismatch (no fabricated cause).
    rec = _mismatch_rec("rdma", None)
    f = mooncakeperf._finding(rec, "nodeA")
    assert f.details["reason"] == "data mismatch after transfer"


# The CPU (host-DRAM) baseline must mirror the aux/metadata buffers PD actually
# registers -- tens of MiB moved in <=16 KiB items -- NOT a 1 GiB MR that
# exceeds ulimit -l and silently ENOMEMs. GPU stays the 1 GiB KV-cache size.
def test_cpu_geometry_matches_production_aux_not_1gib():
    for mod in (mooncakeperf, moriperf):
        size, chunk, _ = mod._geom("cpu")
        assert chunk <= 16 << 10, f"{mod.__name__} cpu chunk should be a real aux item"
        assert size <= 64 << 20, f"{mod.__name__} cpu region must not be a 1 GiB MR"


def test_gpu_geometry_stays_kv_cache_sized():
    for mod in (mooncakeperf, moriperf):
        size, chunk, nchunk = mod._geom("gpu")
        assert size == 1 << 30  # 1 GiB VRAM, the real KV path
        assert chunk * nchunk == size


def test_verify_respects_passed_geometry():
    # A buffer stamped with the per-segment pattern for chunk/nchunk verifies;
    # a different geometry (wrong chunk boundaries) does not.
    chunk, nchunk = 16 << 10, 4
    arr = np.empty(chunk * nchunk, dtype=np.uint8)
    for i in range(nchunk):
        arr[i * chunk : (i + 1) * chunk] = mooncakeperf._chunk_byte(-1, i)
    assert mooncakeperf._verify(arr, -1, chunk, nchunk) is True
    assert mooncakeperf._verify(arr, -1, chunk * 2, nchunk) is False


def test_mooncake_spawn_replaces_non_utf8_native_logs(monkeypatch, tmp_path):
    from types import SimpleNamespace

    run_kwargs = {}

    def fake_run(*args, **kwargs):
        run_kwargs.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="native log: \ufffd")

    monkeypatch.setattr(mooncakeperf.subprocess, "run", fake_run)

    rc, out = mooncakeperf._spawn(
        "initiator",
        str(tmp_path),
        "127.0.0.1:19001",
        "node-a",
        "tcp",
        "tcp",
        "cpu",
        -1,
        "",
    )

    assert rc == 0
    assert out.endswith("\ufffd")
    assert run_kwargs["text"] is True
    assert run_kwargs["errors"] == "replace"


def test_mooncake_selected_gid_overrides_nic_probe(monkeypatch):
    monkeypatch.setattr(mooncakeperf, "_GID_CACHE", None)
    monkeypatch.setattr(
        mooncakeperf, "_nics", lambda: (_ for _ in ()).throw(AssertionError("unexpected"))
    )
    monkeypatch.setenv("INFERA_PREFLIGHT_GID_INDEX", "3")

    assert mooncakeperf._ref_gid() == 3


def test_mooncake_invalid_selected_gid_warns_and_uses_detected_value(monkeypatch, capsys):
    monkeypatch.setattr(mooncakeperf, "_GID_CACHE", None)
    monkeypatch.setattr(mooncakeperf, "_nics", lambda: ["mlx5_0"])
    monkeypatch.setattr(mooncakeperf, "_gid_index", lambda _: 7)
    monkeypatch.setenv("INFERA_PREFLIGHT_GID_INDEX", "not-an-integer")

    assert mooncakeperf._ref_gid() == 7
    assert "invalid INFERA_PREFLIGHT_GID_INDEX" in capsys.readouterr().err


def test_mooncake_selected_device_pins_gpu_variants(monkeypatch):
    monkeypatch.setattr(mooncakeperf, "_nics", lambda: ["mlx5_0"])
    monkeypatch.setenv("INFERA_PREFLIGHT_RDMA_DEVICE", "mlx5_0")

    variants = mooncakeperf._variants(2)

    assert variants[-2:] == [
        ("rdma-gpu0", "rdma", "gid", "gpu", 0, "mlx5_0"),
        ("rdma-gpu1", "rdma", "gid", "gpu", 1, "mlx5_0"),
    ]


def test_mooncake_operation_defaults_to_read(monkeypatch):
    monkeypatch.delenv("INFERA_PREFLIGHT_MOONCAKE_OPCODE", raising=False)
    assert mooncakeperf._operation() == "read"


def test_mooncake_write_dispatches_producer_push(monkeypatch):
    calls = []

    class Engine:
        def batch_transfer_sync_write(self, *args):
            calls.append(args)
            return 0

    monkeypatch.setenv("INFERA_PREFLIGHT_MOONCAKE_OPCODE", "write")
    operation = mooncakeperf._operation()

    assert mooncakeperf._batch_transfer(Engine(), "peer", 100, 200, 8, 2, operation)
    assert calls == [("peer", [100, 108], [200, 208], [8, 8])]


def test_mooncake_operation_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("INFERA_PREFLIGHT_MOONCAKE_OPCODE", "copy")
    with pytest.raises(ValueError, match="invalid INFERA_PREFLIGHT_MOONCAKE_OPCODE"):
        mooncakeperf._operation()


def test_mooncake_write_finding_reports_source_direction(monkeypatch):
    monkeypatch.setattr(mooncakeperf, "_ref_gid", lambda: 1)
    finding = mooncakeperf._finding(
        {
            "label": "rdma-gpu0",
            "target": "decode",
            "operation": "write",
            "gb_s": 42.0,
            "gib": 10.0,
            "loc": "gpu",
            "gpu": 0,
            "verified": True,
            "dev": "ionic_0",
        },
        "prefill",
    )

    assert finding.level == "info"
    assert finding.message == "prefill -> decode rdma-gpu0"


def test_mooncake_write_is_verified_on_target(monkeypatch, tmp_path):
    chunk, nchunk, gpu_id = 4, 2, 3

    class Buffer:
        ptr = 100
        size = chunk * nchunk

        def __init__(self):
            self.chunk = chunk
            self.nchunk = nchunk
            self.arr = np.empty(self.size, dtype=np.uint8)

        def fill_inverse_pattern(self):
            for index in range(self.nchunk):
                self.arr[index * self.chunk : (index + 1) * self.chunk] = (
                    mooncakeperf._chunk_byte(gpu_id, index) ^ 0xFF
                )

        def host_bytes(self):
            return self.arr

    class Engine:
        @staticmethod
        def get_rpc_port():
            return 19001

    buf = Buffer()

    def complete_write(_path, _timeout):
        for index in range(buf.nchunk):
            buf.arr[index * buf.chunk : (index + 1) * buf.chunk] = mooncakeperf._chunk_byte(
                gpu_id, index
            )
        return True

    monkeypatch.setenv("INFERA_PREFLIGHT_MOONCAKE_OPCODE", "write")
    monkeypatch.setattr(mooncakeperf, "_engine", lambda *_args: Engine())
    monkeypatch.setattr(mooncakeperf, "_make_buffer", lambda *_args: buf)
    monkeypatch.setattr(mooncakeperf, "_register", lambda *_args: True)
    monkeypatch.setattr(mooncakeperf, "_wait_file", complete_write)

    mooncakeperf._target(str(tmp_path), "10.0.0.1:17000", "decode", "rdma", "gpu", gpu_id)

    result = json.loads((tmp_path / "target-verify.json").read_text())
    assert result == {"verified": True}
