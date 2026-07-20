###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import pytest

from infera.common.disagg_preflight import (
    DisaggPreflightError,
    is_routable_host,
    validate_advertise_host,
    validate_sglang_transport,
    validate_vllm_transport,
)


@pytest.mark.parametrize(
    "host",
    ["", "0.0.0.0", "127.0.0.1", "localhost", "::", "::1", "  0.0.0.0 ", "LOCALHOST"],
)
def test_non_routable_hosts(host):
    assert is_routable_host(host) is False


@pytest.mark.parametrize("host", ["10.0.0.5", "192.168.1.10", "node-a.cluster", "fd00::1"])
def test_routable_hosts(host):
    assert is_routable_host(host) is True


def test_is_routable_host_none():
    assert is_routable_host(None) is False


# --- advertise host ---------------------------------------------------


def test_advertise_host_mixed_skips_validation():
    # Mixed workers may bind/advertise loopback; no error.
    validate_advertise_host("0.0.0.0", is_disagg=False)


def test_advertise_host_disagg_rejects_non_routable():
    with pytest.raises(DisaggPreflightError, match="routable host"):
        validate_advertise_host("0.0.0.0", is_disagg=True)


def test_advertise_host_disagg_accepts_routable():
    validate_advertise_host("10.0.0.5", is_disagg=True)


# --- sglang transport -------------------------------------------------


def test_sglang_transport_mixed_skips():
    validate_sglang_transport(None, is_disagg=False, allow_tcp=False)


@pytest.mark.parametrize("backend", ["mooncake", "mori", "nixl", "ascend", "MoRI"])
def test_sglang_transport_accepts_rdma_backends(backend):
    validate_sglang_transport(backend, is_disagg=True, allow_tcp=False)


def test_sglang_transport_rejects_empty_backend():
    with pytest.raises(DisaggPreflightError, match="explicit"):
        validate_sglang_transport(None, is_disagg=True, allow_tcp=False)


def test_sglang_transport_rejects_non_rdma_backend():
    with pytest.raises(DisaggPreflightError, match="non-RDMA"):
        validate_sglang_transport("tcp", is_disagg=True, allow_tcp=False)


def test_sglang_transport_allow_tcp_overrides():
    # Escape hatch: anything passes.
    validate_sglang_transport(None, is_disagg=True, allow_tcp=True)
    validate_sglang_transport("tcp", is_disagg=True, allow_tcp=True)


# --- vllm transport ---------------------------------------------------


def test_vllm_transport_mixed_skips():
    validate_vllm_transport({}, is_disagg=False, allow_tcp=False)


def test_vllm_transport_accepts_known_protocol():
    validate_vllm_transport(
        {"protocol": "vllm-mooncake", "params": {}}, is_disagg=True, allow_tcp=False
    )


@pytest.mark.parametrize("meta", [None, {}, {"params": {}}, {"protocol": None}])
def test_vllm_transport_rejects_missing_protocol(meta):
    with pytest.raises(DisaggPreflightError, match="RDMA KV connector"):
        validate_vllm_transport(meta, is_disagg=True, allow_tcp=False)


def test_vllm_transport_allow_tcp_overrides():
    validate_vllm_transport({}, is_disagg=True, allow_tcp=True)
