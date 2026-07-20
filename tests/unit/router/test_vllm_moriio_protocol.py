###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""VllmMoRIIOReadProtocol — wire-shape sanity + vLLM regex lock."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from infera.router.disagg_protocols import _PROTOCOLS, resolve_protocol
from infera.router.disagg_protocols.vllm_moriio import (
    VllmMoRIIOReadProtocol,
    VllmMoRIIOWriteProtocol,
)


def _worker(worker_id: str, url: str = "http://10.0.0.1:30000") -> SimpleNamespace:
    return SimpleNamespace(
        worker_id=worker_id,
        url=url,
        dp_size=1,
        disagg_meta={
            "protocol": "vllm-mori-read",
            "params": {
                "engine_id": f"eng-{worker_id}",
                "kv_connector_extra_config": {
                    "handshake_port": 6301,
                    "notify_port": 61005,
                    "tp_size": 8,
                },
            },
        },
    )


def test_registered_and_resolves() -> None:
    assert "vllm-mori-read" in _PROTOCOLS
    proto = resolve_protocol(_worker("p"), _worker("d"))
    assert proto.name == "vllm-mori-read"
    assert proto.topology == "serial-pull"


def test_annotate_prefill_shape() -> None:
    proto = VllmMoRIIOReadProtocol()
    body = {"model": "qwen3", "stream_options": {"include_usage": True}}
    out = proto.annotate_prefill(body, _worker("p"), _worker("d"), room_id=1)
    kv = out["kv_transfer_params"]
    assert kv["do_remote_decode"] is True
    assert kv["do_remote_prefill"] is False
    assert kv["remote_engine_id"] is None
    assert kv["remote_block_ids"] is None
    assert kv["transfer_id"].startswith("tx-")
    assert out["max_tokens"] == 1
    assert out["stream"] is False
    assert "stream_options" not in out
    assert body == {"model": "qwen3", "stream_options": {"include_usage": True}}


def test_annotate_decode_propagates_handoff() -> None:
    proto = VllmMoRIIOReadProtocol()
    handoff = {
        "remote_block_ids": [1, 2, 3],
        "remote_engine_id": "p-eng",
        "transfer_id": "tx-abc",
    }
    out = proto.annotate_decode(
        {"max_tokens": 50},
        _worker("p"),
        _worker("d"),
        room_id=1,
        prefill_handoff=handoff,
    )
    kv = out["kv_transfer_params"]
    assert kv["transfer_id"] == "tx-abc"
    assert kv["remote_engine_id"] == "p-eng"
    assert kv["remote_block_ids"] == [1, 2, 3]
    assert out["max_tokens"] == 49  # P emitted 1


def test_extract_handoff_missing_raises() -> None:
    with pytest.raises(KeyError, match="kv_transfer_params"):
        VllmMoRIIOReadProtocol().extract_handoff({"choices": []})


def test_injects_canonical_tp_size_key() -> None:
    """Regression: connector ReqMeta keys peer TP on kv_transfer_params["tp_size"]
    (moriio_common.py), defaulting to 1 if absent -> every decode rank reads prefill
    rank 0's shard -> garbage. All P/D legs must carry ``tp_size`` = peer's real TP."""
    read = VllmMoRIIOReadProtocol()
    p, d = _worker("p"), _worker("d")

    p_body = read.annotate_prefill({}, p, d, room_id=1)
    assert p_body["kv_transfer_params"]["tp_size"] == 8

    handoff = {"remote_block_ids": [1], "remote_engine_id": "p", "transfer_id": "tx-1"}
    r_dec = read.annotate_decode({"max_tokens": 5}, p, d, room_id=1, prefill_handoff=handoff)
    assert r_dec["kv_transfer_params"]["tp_size"] == 8

    w_dec = VllmMoRIIOWriteProtocol().annotate_decode(
        {"max_tokens": 5}, p, d, room_id=1, prefill_handoff=None
    )
    assert w_dec["kv_transfer_params"]["tp_size"] == 8


def test_forged_request_id_matches_vllm_regexes() -> None:
    """Locks the wire format against vLLM's MoRIIO regexes in
    ``moriio_common.py``. Drift here means engine fails to parse peer
    addressing and KV transfer hangs."""
    proto = VllmMoRIIOReadProtocol()
    p = _worker("p", "http://10.0.0.22:30501")
    d = _worker("d", "http://10.0.0.40:30502")
    req_id = proto.request_id_for(p, d, room_id=1)

    prefill_re = re.compile(r"___prefill_addr_(.+?)___decode_addr_")
    decode_re = re.compile(r"___decode_addr_(.+)_[0-9a-f]{32}(?:-.*)?$")
    assert prefill_re.search(req_id).group(1) == "host:10.0.0.22,handshake:6301,notify:61005"
    assert decode_re.search(req_id).group(1) == "host:10.0.0.40,handshake:6301,notify:61005"


def test_request_id_missing_port_raises() -> None:
    proto = VllmMoRIIOReadProtocol()
    p = SimpleNamespace(
        worker_id="p",
        url="http://x:1",
        disagg_meta={"params": {"kv_connector_extra_config": {"notify_port": 1}}},
    )
    with pytest.raises(ValueError, match="handshake_port"):
        proto.request_id_for(p, _worker("d"), room_id=1)


# ---- WRITE protocol -----------------------------------------------


def test_write_registered_and_concurrent() -> None:
    assert "vllm-mori-write" in _PROTOCOLS
    proto = VllmMoRIIOWriteProtocol()
    assert proto.topology == "concurrent"


def test_write_p_and_d_share_transfer_id() -> None:
    """Concurrent dispatch: P and D bodies built independently must
    agree on transfer_id (vLLM correlates the two legs on it)."""
    proto = VllmMoRIIOWriteProtocol()
    p, d = _worker("p"), _worker("d")
    p_body = proto.annotate_prefill({}, p, d, room_id=42)
    d_body = proto.annotate_decode({}, p, d, room_id=42, prefill_handoff=None)
    assert p_body["kv_transfer_params"]["transfer_id"].startswith("tx-")
    assert (
        p_body["kv_transfer_params"]["transfer_id"] == d_body["kv_transfer_params"]["transfer_id"]
    )


def test_write_decode_omits_handoff_fields() -> None:
    """WRITE doesn't use remote_engine_id / remote_block_ids — P
    pushes via zmq notify rather than D pulling."""
    proto = VllmMoRIIOWriteProtocol()
    out = proto.annotate_decode({}, _worker("p"), _worker("d"), room_id=1, prefill_handoff=None)
    kv = out["kv_transfer_params"]
    assert kv["do_remote_prefill"] is True
    assert kv["remote_engine_id"] is None
    assert kv["remote_block_ids"] is None


def test_write_forges_same_request_id_format() -> None:
    """Both READ and WRITE need the same sandwich; engine-side regex
    doesn't know the difference."""
    proto = VllmMoRIIOWriteProtocol()
    req_id = proto.request_id_for(
        _worker("p", "http://10.0.0.22:30501"),
        _worker("d", "http://10.0.0.40:30502"),
        room_id=1,
    )
    assert "___prefill_addr_host:10.0.0.22" in req_id
    assert "___decode_addr_host:10.0.0.40" in req_id
