###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""``_compute_disagg_meta`` — connector → protocol mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from infera.engine.vllm.args import _compute_disagg_meta, _pin_engine_id_in_argv


@dataclass
class _Cfg:
    kv_connector: str | None = None
    kv_role: str | None = None
    engine_id: str | None = "eng"
    kv_ip: str = "10.0.0.1"
    kv_port: int = 14579
    kv_connector_extra_config: dict[str, Any] = field(default_factory=dict)


def test_mixed_returns_empty() -> None:
    assert _compute_disagg_meta(None) == {}
    assert _compute_disagg_meta(_Cfg(kv_role="kv_both")) == {}


def test_mooncake_maps_to_protocol() -> None:
    meta = _compute_disagg_meta(
        _Cfg(
            kv_connector="MooncakeConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={"foo": "bar"},
        )
    )
    assert meta["protocol"] == "vllm-mooncake"
    assert meta["params"]["engine_id"] == "eng"
    assert meta["params"]["kv_connector_extra_config"] == {"foo": "bar"}


def test_moriio_env_flips_to_read(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _Cfg(kv_connector="MoRIIOConnector", kv_role="kv_producer")
    monkeypatch.delenv("VLLM_MORIIO_CONNECTOR_READ_MODE", raising=False)
    assert _compute_disagg_meta(cfg)["protocol"] == "vllm-mori-write"
    monkeypatch.setenv("VLLM_MORIIO_CONNECTOR_READ_MODE", "1")
    assert _compute_disagg_meta(cfg)["protocol"] == "vllm-mori-read"


def test_unknown_connector_returns_empty() -> None:
    assert _compute_disagg_meta(_Cfg(kv_connector="P2pNcclConnector", kv_role="kv_producer")) == {}


def test_multiconnector_surfaces_nested_transport() -> None:
    """kvd L3 + Mooncake are combined via vLLM's MultiConnector. The disagg
    transport is nested under kv_connector_extra_config["connectors"]; the
    launcher must descend into it (else no protocol/bootstrap_addr is
    registered and the disagg preflight aborts the worker)."""
    cfg = _Cfg(
        kv_connector="MultiConnector",
        kv_role="kv_producer",
        kv_connector_extra_config={
            "connectors": [
                {"kv_connector": "InferaKvdConnector", "kv_connector_extra_config": {}},
                {
                    "kv_connector": "MooncakeConnector",
                    "kv_connector_extra_config": {"mooncake_protocol": "rdma"},
                },
            ]
        },
    )
    meta = _compute_disagg_meta(cfg, advertise_host="10.0.0.22")
    assert meta["protocol"] == "vllm-mooncake"
    assert meta["params"]["bootstrap_addr"] == "http://10.0.0.22:8998"
    # params carry the *transport child's* extra_config, not the wrapper's.
    assert meta["params"]["kv_connector_extra_config"] == {"mooncake_protocol": "rdma"}
    assert meta["params"]["engine_id"] == "eng"


def test_multiconnector_without_transport_returns_empty() -> None:
    """A MultiConnector with no recognized RDMA transport child has no
    protocol — must still return {} (no false positive)."""
    cfg = _Cfg(
        kv_connector="MultiConnector",
        kv_role="kv_producer",
        kv_connector_extra_config={
            "connectors": [
                {"kv_connector": "InferaKvdConnector", "kv_connector_extra_config": {}},
            ]
        },
    )
    assert _compute_disagg_meta(cfg) == {}


def test_extra_config_is_copied() -> None:
    extra = {"x": 1}
    meta = _compute_disagg_meta(
        _Cfg(
            kv_connector="MooncakeConnector", kv_role="kv_producer", kv_connector_extra_config=extra
        )
    )
    assert meta["params"]["kv_connector_extra_config"] is not extra


def test_mooncake_bootstrap_addr_uses_advertise_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", raising=False)
    meta = _compute_disagg_meta(
        _Cfg(kv_connector="MooncakeConnector", kv_role="kv_producer"),
        advertise_host="10.0.0.22",
    )
    assert meta["params"]["bootstrap_addr"] == "http://10.0.0.22:8998"


def test_mooncake_bootstrap_port_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", "9001")
    meta = _compute_disagg_meta(
        _Cfg(kv_connector="MooncakeConnector", kv_role="kv_producer"),
        advertise_host="10.0.0.1",
    )
    assert meta["params"]["bootstrap_addr"] == "http://10.0.0.1:9001"


# ---- _pin_engine_id_in_argv -----------------------------------------------


def test_pin_engine_id_rewrites_kv_transfer_config() -> None:
    """vLLM's subprocess re-parses the JSON and randomises engine_id by
    default; without this rewrite, etcd and Mooncake's bootstrap server
    disagree on engine_id and every D-leg POST 4xx's."""
    import json

    argv = [
        "--kv-transfer-config",
        '{"kv_connector":"MooncakeConnector","engine_id":"old"}',
    ]
    out = _pin_engine_id_in_argv(argv, "new")
    assert json.loads(out[1])["engine_id"] == "new"
    # engine_id=None (MIXED) must leave argv untouched.
    assert _pin_engine_id_in_argv(argv, None) == argv


def test_pin_engine_id_does_not_touch_multiconnector_children() -> None:
    """vLLM forwards the top-level engine_id to each MultiConnector child
    itself; if we ALSO set it in the nested dicts, the child
    KVTransferConfig() gets engine_id twice -> TypeError at engine init.
    Only the top-level must be pinned."""
    import json

    blob = json.dumps(
        {
            "kv_connector": "MultiConnector",
            "kv_connector_extra_config": {
                "connectors": [
                    {
                        "kv_connector": "MooncakeConnector",
                        "kv_connector_extra_config": {"mooncake_protocol": "rdma"},
                    },
                    {"kv_connector": "InferaKvdConnector", "kv_connector_extra_config": {}},
                ]
            },
        }
    )
    out = _pin_engine_id_in_argv(["--kv-transfer-config", blob], "new")
    parsed = json.loads(out[1])
    assert parsed["engine_id"] == "new"
    assert all("engine_id" not in c for c in parsed["kv_connector_extra_config"]["connectors"])
