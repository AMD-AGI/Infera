###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unit tests for infera.engine.rocm_rdma_env host-IP auto-pin."""

from __future__ import annotations

import os

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
    assert os.environ["VLLM_ROCM_USE_AITER"] == "1"


def test_aiter_respects_operator_override(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
    monkeypatch.setattr(rre, "_is_rocm", lambda: True)
    assert rre.apply_vllm_aiter_default() is None  # no-op
    assert os.environ["VLLM_ROCM_USE_AITER"] == "0"


def test_aiter_noop_off_rocm(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_USE_AITER", raising=False)
    monkeypatch.setattr(rre, "_is_rocm", lambda: False)
    assert rre.apply_vllm_aiter_default() is None


# --- destination-device affinity: default ON, opt-out must actually work -----
#
# Mooncake enables the policy on the mere PRESENCE of the env var, so a "0" from
# an operator would silently still enable it. And setting it alongside
# MC_ENABLE_HCA_PEER_AFFINITY makes Mooncake disable BOTH policies, which is
# worse than either alone. Both hazards are handled here, not in the C++.


def _on_rocm_without_affinity_env(monkeypatch):
    for v in (rre._MC_DEST_AFFINITY, rre._MC_HCA_PEER_AFFINITY):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(rre, "_is_rocm", lambda: True)


def test_dest_affinity_defaults_on(monkeypatch):
    _on_rocm_without_affinity_env(monkeypatch)
    assert rre.apply_rocm_rdma_env_defaults()[rre._MC_DEST_AFFINITY] == "1"
    assert os.environ[rre._MC_DEST_AFFINITY] == "1"


def test_dest_affinity_opt_out_unsets_the_var(monkeypatch):
    """A false-y value must DISABLE it, not just be left in the environment."""
    _on_rocm_without_affinity_env(monkeypatch)
    for value in ("0", "False"):
        monkeypatch.setenv(rre._MC_DEST_AFFINITY, value)
        assert rre._MC_DEST_AFFINITY not in rre.apply_rocm_rdma_env_defaults()
        assert rre._MC_DEST_AFFINITY not in os.environ


def test_dest_affinity_defers_to_explicit_hca_peer_affinity(monkeypatch):
    _on_rocm_without_affinity_env(monkeypatch)
    monkeypatch.setenv(rre._MC_HCA_PEER_AFFINITY, "true")
    assert rre._MC_DEST_AFFINITY not in rre.apply_rocm_rdma_env_defaults()
    assert rre._MC_DEST_AFFINITY not in os.environ


def test_dest_affinity_applies_when_hca_peer_affinity_is_off(monkeypatch):
    _on_rocm_without_affinity_env(monkeypatch)
    monkeypatch.setenv(rre._MC_HCA_PEER_AFFINITY, "0")
    assert rre.apply_rocm_rdma_env_defaults()[rre._MC_DEST_AFFINITY] == "1"


# --- GPU MR path: follow the host, not a fleet-wide guess --------------------
#
# Both paths are compiled into the images. Bare ibv_reg_mr needs a peer-memory
# module and cannot register a device pointer without one; dma-buf works there but
# on a peer-mem host it burns a KFD resource under load (HIP-209). So the choice
# has to be made per host, at startup.


def _on_rocm_without_mr_env(monkeypatch, *, peermem: bool):
    _on_rocm_without_affinity_env(monkeypatch)
    monkeypatch.delenv(rre._MC_DISABLE_DMABUF, raising=False)
    monkeypatch.setattr(rre, "_peermem_loaded", lambda: peermem)


def test_no_peermem_leaves_dmabuf_selected(monkeypatch):
    _on_rocm_without_mr_env(monkeypatch, peermem=False)
    assert rre._MC_DISABLE_DMABUF not in rre.apply_rocm_rdma_env_defaults()
    assert rre._MC_DISABLE_DMABUF not in os.environ


def test_peermem_pins_the_bare_reg_mr_path(monkeypatch):
    _on_rocm_without_mr_env(monkeypatch, peermem=True)
    assert rre.apply_rocm_rdma_env_defaults()[rre._MC_DISABLE_DMABUF] == "1"
    assert os.environ[rre._MC_DISABLE_DMABUF] == "1"


def test_operator_mr_path_override_wins(monkeypatch):
    _on_rocm_without_mr_env(monkeypatch, peermem=True)
    monkeypatch.setenv(rre._MC_DISABLE_DMABUF, "0")
    assert rre._MC_DISABLE_DMABUF not in rre.apply_rocm_rdma_env_defaults()
    assert os.environ[rre._MC_DISABLE_DMABUF] == "0"


def _fake_proc_modules(monkeypatch, tmp_path, body):
    mods = tmp_path / "modules"
    mods.write_text(body)
    monkeypatch.setattr(rre, "_PROC_MODULES", str(mods))


def test_peermem_probe_reads_proc_modules(monkeypatch, tmp_path):
    _fake_proc_modules(monkeypatch, tmp_path, "amdgpu 1 0 - Live 0x0\nib_peer_mem 1 1 - Live 0x0\n")
    assert rre._peermem_loaded() is True


def test_peermem_probe_ignores_unrelated_modules(monkeypatch, tmp_path):
    _fake_proc_modules(monkeypatch, tmp_path, "amdgpu 1 0 - Live 0x0\n\n")
    assert rre._peermem_loaded() is False


def test_peermem_probe_reports_absent_when_unreadable(monkeypatch, tmp_path):
    monkeypatch.setattr(rre, "_PROC_MODULES", str(tmp_path / "missing"))
    assert rre._peermem_loaded() is False
