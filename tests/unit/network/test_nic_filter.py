###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unit tests for the PD-disagg harness RDMA NIC-filter *logic*
(``tests/e2e/harness/cluster.py``): ``compute_nic_filter`` and the GID helpers.

Cluster-independent: exercises the PURE functions with hardcoded synthetic GIDs
(no srun / no ``/sys`` / no hardware), so it passes on any machine or cluster.
``cluster.py`` is loaded by file path so this stays a plain unit test and does
NOT import the e2e harness package (which pulls httpx/uvicorn)."""

from __future__ import annotations

import importlib.util
import os


def _load_cluster():
    """Locate + import tests/e2e/harness/cluster.py by path (robust to where this
    test file lives — walk up to the repo root that contains it)."""
    root = os.path.abspath(__file__)
    for _ in range(8):
        root = os.path.dirname(root)
        cand = os.path.join(root, "tests", "e2e", "harness", "cluster.py")
        if os.path.isfile(cand):
            spec = importlib.util.spec_from_file_location("_pd_cluster", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise RuntimeError("could not locate tests/e2e/harness/cluster.py from " + __file__)


cl = _load_cluster()


# 8 ionic data rails, each on its OWN /64 subnet (rail-optimized), IPv6-ULA GID
# at the index; plus a mgmt ConnectX whose GID there is link-local.
_IONIC = [(f"ionic_{i}", f"fc01:0{i}00:a{i}0d:2d5e:0690:81ff:fe44:0791") for i in range(8)]
_MLX5_MGMT = ("mlx5_0", "fe80:0000:0000:0000:4c10:2fff:fe0f:3052")
# A flat fabric: two NICs sharing ONE /24 (IPv4-mapped RoCEv2 GIDs).
_FLAT = [
    ("mlx5_0", "0000:0000:0000:0000:0000:ffff:0a0a:0101"),  # 10.10.1.1
    ("mlx5_1", "0000:0000:0000:0000:0000:ffff:0a0a:0102"),  # 10.10.1.2
]


def test_gid_routable():
    assert cl._gid_routable("fc01:0800:500d:2d26:0690:81ff:fe42:1399")
    assert cl._gid_routable("0000:0000:0000:0000:0000:ffff:0af5:97ba")
    assert not cl._gid_routable("fe80:0000:0000:0000:4c10:2fff:fe0f:3052")
    assert not cl._gid_routable("0000:0000:0000:0000:0000:0000:0000:0000")
    assert not cl._gid_routable("")


def test_gid_subnet():
    assert cl._gid_subnet("0000:0000:0000:0000:0000:ffff:0a0a:0101") == "10.10.1"
    assert cl._gid_subnet("fc01:0800:500d:2d26:0690:81ff:fe42:1399") == "fc01:0800:500d:2d26"


def test_filter_rail_optimized_pins_single_rail():
    # ionic (8 distinct subnets) + mgmt -> a single deterministic ionic rail.
    assert cl.compute_nic_filter(_IONIC + [_MLX5_MGMT]) == "ionic_0"


def test_filter_flat_fabric_keeps_all_data_rails():
    foreign = ("eth_ll", "fe80:0000:0000:0000:4c10:2fff:fe0f:9999")
    assert cl.compute_nic_filter(_FLAT + [foreign]) == "mlx5_0,mlx5_1"


def test_filter_noop_flat_no_foreign():
    assert cl.compute_nic_filter(_FLAT) is None


def test_filter_noop_when_no_routable():
    assert cl.compute_nic_filter([_MLX5_MGMT]) is None


def test_filter_noop_empty():
    assert cl.compute_nic_filter([]) is None
