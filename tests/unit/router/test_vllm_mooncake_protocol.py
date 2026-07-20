###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""VllmMooncakeProtocol — wire-shape sanity."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infera.router.disagg_protocols import _PROTOCOLS
from infera.router.disagg_protocols.vllm_mooncake import VllmMooncakeProtocol


def _worker(worker_id: str, *, with_bootstrap: bool = True) -> SimpleNamespace:
    params = {"engine_id": f"eng-{worker_id}"}
    if with_bootstrap:
        params["bootstrap_addr"] = "http://10.0.0.1:8998"
    return SimpleNamespace(
        worker_id=worker_id,
        url="http://10.0.0.1:30000",
        dp_size=1,
        disagg_meta={"protocol": "vllm-mooncake", "params": params},
    )


def test_registered_concurrent_no_request_id() -> None:
    assert "vllm-mooncake" in _PROTOCOLS
    proto = VllmMooncakeProtocol()
    assert proto.topology == "concurrent"
    assert proto.request_id_for(_worker("p"), _worker("d"), room_id=1) is None


def test_prefill_body_shape() -> None:
    proto = VllmMooncakeProtocol()
    out = proto.annotate_prefill(
        {"model": "x", "stream_options": {"include_usage": True}},
        _worker("p"),
        _worker("d"),
        room_id=1,
    )
    kv = out["kv_transfer_params"]
    assert kv["do_remote_decode"] is True
    assert kv["do_remote_prefill"] is False
    assert kv["transfer_id"].startswith("xfer-")
    assert out["max_tokens"] == 1
    assert out["stream"] is False
    assert "stream_options" not in out


def test_decode_carries_bootstrap_and_engine_id() -> None:
    proto = VllmMooncakeProtocol()
    out = proto.annotate_decode(
        {"max_tokens": 50},
        _worker("p"),
        _worker("d"),
        room_id=1,
        prefill_handoff=None,
    )
    kv = out["kv_transfer_params"]
    assert kv["do_remote_prefill"] is True
    assert kv["remote_bootstrap_addr"] == "http://10.0.0.1:8998"
    assert kv["remote_engine_id"] == "eng-p"
    assert out["max_tokens"] == 49  # P emitted 1


def test_p_and_d_share_transfer_id() -> None:
    proto = VllmMooncakeProtocol()
    p, d = _worker("p"), _worker("d")
    p_body = proto.annotate_prefill({}, p, d, room_id=99)
    d_body = proto.annotate_decode({}, p, d, room_id=99, prefill_handoff=None)
    assert (
        p_body["kv_transfer_params"]["transfer_id"] == d_body["kv_transfer_params"]["transfer_id"]
    )


def test_decode_missing_bootstrap_addr_raises() -> None:
    proto = VllmMooncakeProtocol()
    with pytest.raises(ValueError, match="bootstrap_addr"):
        proto.annotate_decode(
            {},
            _worker("p", with_bootstrap=False),
            _worker("d"),
            room_id=1,
            prefill_handoff=None,
        )
