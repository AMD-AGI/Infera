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
    import numpy as np

    # A buffer stamped with the per-segment pattern for chunk/nchunk verifies;
    # a different geometry (wrong chunk boundaries) does not.
    chunk, nchunk = 16 << 10, 4
    arr = np.empty(chunk * nchunk, dtype=np.uint8)
    for i in range(nchunk):
        arr[i * chunk : (i + 1) * chunk] = mooncakeperf._chunk_byte(-1, i)
    assert mooncakeperf._verify(arr, -1, chunk, nchunk) is True
    assert mooncakeperf._verify(arr, -1, chunk * 2, nchunk) is False
