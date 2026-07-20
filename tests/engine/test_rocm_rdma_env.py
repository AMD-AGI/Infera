###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unit tests for infera.engine.rocm_rdma_env host-IP auto-pin."""

from __future__ import annotations

import infera.engine.rocm_rdma_env as rre


def _clear_host_ip_env(monkeypatch):
    for v in rre._KV_HOST_IP_VARS:
        monkeypatch.delenv(v, raising=False)


def test_gid_to_ipv4_mapped():
    # IPv4-mapped GID: last 32 bits = 172.30.27.145
    gid = "0000:0000:0000:0000:0000:ffff:ac1e:1b91"
    assert rre._gid_to_ipv4(gid) == "172.30.27.145"


def test_gid_to_ipv4_rejects_ipv6():
    assert rre._gid_to_ipv4("fe80:0000:0000:0000:0690:81ff:fe36:57f0") is None
    assert rre._gid_to_ipv4("fd93:16d3:59b6:08e6:0690:81ff:fe36:57f0") is None


def test_is_private_ipv4():
    assert rre._is_private_ipv4("10.0.0.38")
    assert rre._is_private_ipv4("192.168.1.5")
    assert rre._is_private_ipv4("172.30.0.1")
    assert not rre._is_private_ipv4("172.17.0.1")  # docker0 excluded
    assert not rre._is_private_ipv4("144.202.62.158")  # public


def test_host_ip_prefers_roce_gid(monkeypatch):
    _clear_host_ip_env(monkeypatch)
    monkeypatch.setattr(rre, "_is_rocm", lambda: True)
    monkeypatch.setenv("MC_GID_INDEX", "3")
    monkeypatch.setattr(
        rre, "_active_rdma_nics", lambda gi: [("ionic_0", "10.9.9.9", "10.9.9.0/24")]
    )
    # fallback must NOT be consulted when a RoCE IPv4 GID exists
    monkeypatch.setattr(rre, "_private_rail_ipv4", lambda: "10.0.0.1")
    assert rre.apply_kv_host_ip_default() == "10.9.9.9"
    import os

    assert os.environ["VLLM_HOST_IP"] == "10.9.9.9"
    assert os.environ["ATOM_HOST_IP"] == "10.9.9.9"


def test_host_ip_falls_back_to_private_nic(monkeypatch):
    """IPv6-only RoCE GIDs (e.g. our ionic fleet) -> private-NIC fallback."""
    _clear_host_ip_env(monkeypatch)
    monkeypatch.setattr(rre, "_is_rocm", lambda: True)
    monkeypatch.setattr(rre, "_active_rdma_nics", lambda gi: [])  # no IPv4 GID
    monkeypatch.setattr(rre, "_private_rail_ipv4", lambda: "10.0.0.38")
    assert rre.apply_kv_host_ip_default() == "10.0.0.38"


def test_host_ip_operator_override_wins(monkeypatch):
    _clear_host_ip_env(monkeypatch)
    monkeypatch.setattr(rre, "_is_rocm", lambda: True)
    monkeypatch.setenv("VLLM_HOST_IP", "1.2.3.4")
    monkeypatch.setattr(rre, "_active_rdma_nics", lambda gi: [("ionic_0", "10.9.9.9", "x")])
    assert rre.apply_kv_host_ip_default() is None  # no-op, respect operator


def test_host_ip_noop_off_rocm(monkeypatch):
    _clear_host_ip_env(monkeypatch)
    monkeypatch.setattr(rre, "_is_rocm", lambda: False)
    assert rre.apply_kv_host_ip_default() is None


def test_aiter_default_on(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_USE_AITER", raising=False)
    monkeypatch.setattr(rre, "_is_rocm", lambda: True)
    assert rre.apply_vllm_aiter_default() == "1"
    import os

    assert os.environ["VLLM_ROCM_USE_AITER"] == "1"


def test_aiter_respects_operator_override(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
    monkeypatch.setattr(rre, "_is_rocm", lambda: True)
    assert rre.apply_vllm_aiter_default() is None  # no-op
    import os

    assert os.environ["VLLM_ROCM_USE_AITER"] == "0"


def test_aiter_noop_off_rocm(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_USE_AITER", raising=False)
    monkeypatch.setattr(rre, "_is_rocm", lambda: False)
    assert rre.apply_vllm_aiter_default() is None
