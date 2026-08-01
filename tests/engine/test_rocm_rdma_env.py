###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unit tests for infera.engine.rocm_rdma_env host-IP auto-pin."""

from __future__ import annotations

import pathlib

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


# --- Mooncake HIP-transport gate: the two halves must agree ------------------
#
# Regression guard for a defect that shipped: rocm_rdma_env set
# MC_DISABLE_HIP_TRANSPORT=1 while the C++ gate we patch into Mooncake read only
# MC_ENABLE_HIP_TRANSPORT, so NOTHING consumed the variable infera was setting
# and the "disable" default was a silent no-op. Cross-node PD then died in KV
# transfer with hipIpcOpenMemHandle 201 (a hipIpc handle is host-local, so a peer
# node can never open it).
#
# These tests pin the Python default and the C++ patch to the same spelling, so
# renaming either half without the other fails here instead of on a cluster.

_MC_PATCH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "deploy"
    / "docker"
    / "patches"
    / "mooncake_cpp"
    / "transfer_engine_impl.diff"
)


def test_hip_transport_disabled_by_default():
    """infera must ship the HIP transport OFF, via the name the gate reads."""
    assert rre._ROCM_RDMA_DEFAULTS["MC_DISABLE_HIP_TRANSPORT"] == "1"
    # It must NOT hand out an "enable" default -- that would re-break cross-node PD.
    assert "MC_ENABLE_HIP_TRANSPORT" not in rre._ROCM_RDMA_DEFAULTS


def test_mooncake_patch_consumes_the_var_infera_sets():
    """The C++ gate must read every MC_*_HIP_TRANSPORT var infera sets."""
    patch = _MC_PATCH.read_text()
    added = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+"))
    for var in rre._ROCM_RDMA_DEFAULTS:
        if "HIP_TRANSPORT" in var:
            assert f'getenv("{var}")' in added, (
                f"{var} is set by rocm_rdma_env but never read by the Mooncake "
                f"gate patch -- it would be a silent no-op"
            )


def test_mooncake_patch_defaults_off_without_any_env():
    """The gate must not require an env var to be safe.

    The installTransport("hip") call must sit in the `else` of a check that is
    false when nothing is set, so an operator who exports nothing still gets RDMA.
    """
    patch = _MC_PATCH.read_text()
    added = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+"))
    # Enabling requires MC_ENABLE_HIP_TRANSPORT to be present AND non-"0" ...
    assert 'mc_hip_enable && strcmp(mc_hip_enable, "0") != 0' in added
    # ... and MC_DISABLE_HIP_TRANSPORT vetoes it outright.
    assert 'mc_hip_disable && strcmp(mc_hip_disable, "0") != 0' in added
    assert "if (!mc_hip_wanted || mc_hip_vetoed) {" in added
    # The unconditional upstream block must be gone.
    removed = [line[1:] for line in patch.splitlines() if line.startswith("-")]
    assert any(line.strip() == "{" for line in removed)
