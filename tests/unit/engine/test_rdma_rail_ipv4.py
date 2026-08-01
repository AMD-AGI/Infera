###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The RDMA-rail IPv4 scan must not pick a container/CNI bridge.

On a Kubernetes node the CNI bridge is private, RFC1918 and sorts FIRST, so a
"first private IPv4" scan returns it. Observed on k3s/flannel: PD prefill pinned
its KV host IP to 10.42.0.1 and decode to 10.42.1.1 — each the other node's
unreachable bridge gateway — while the routable rail was 10.2.122.x.
"""

from __future__ import annotations

import pytest

from infera.engine import rocm_rdma_env as env


@pytest.fixture
def fake_host(monkeypatch):
    """A node laid out the way k3s leaves one: cni0 first, real NIC later."""
    addrs = {
        "cni0": "10.42.0.1",  # flannel bridge — private, and sorts first
        "docker0": "172.17.0.1",
        "flannel.1": "10.42.0.0",
        "enp105s0": "10.2.122.22",  # the routable private rail
        "veth9f2a1b": "10.42.0.7",
        "lo": "127.0.0.1",
    }
    monkeypatch.setattr(env.os, "listdir", lambda _p: sorted(addrs))
    monkeypatch.setattr(env, "_ifaddr_ipv4", lambda n: addrs.get(n))
    return addrs


def test_skips_cni_bridge_and_picks_the_real_nic(fake_host) -> None:
    assert env._private_rail_ipv4() == "10.2.122.22"


@pytest.mark.parametrize(
    "name", ["cni0", "docker0", "flannel.1", "veth9f2a1b", "br-abc123", "cali12ab", "kube-ipvs0"]
)
def test_virtual_interfaces_are_excluded(name: str) -> None:
    assert name.startswith(env._VIRTUAL_IF_PREFIXES), f"{name} would be scanned as a rail"


def test_no_private_nic_yields_none(monkeypatch) -> None:
    """A host with only a public address must return None, not a public IP."""
    addrs = {"lo": "127.0.0.1", "enp1s0": "149.28.124.225"}
    monkeypatch.setattr(env.os, "listdir", lambda _p: sorted(addrs))
    monkeypatch.setattr(env, "_ifaddr_ipv4", lambda n: addrs.get(n))
    assert env._private_rail_ipv4() is None
