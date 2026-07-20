###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unit tests for ``infera.router.disagg_protocols``."""

from __future__ import annotations

import pytest

from infera.common.worker_pool import DisaggMode, EngineType, WorkerInfo
from infera.router.disagg_protocols import (
    ProtocolMismatch,
    UnknownProtocol,
    resolve_protocol,
)
from infera.router.disagg_protocols.sglang_bootstrap import SglangBootstrapProtocol


def _worker(
    wid: str,
    mode: DisaggMode,
    *,
    engine: EngineType = EngineType.SGLANG,
    protocol: str | None = "sglang-bootstrap",
    params: dict | None = None,
) -> WorkerInfo:
    meta: dict = {}
    if protocol is not None:
        meta["protocol"] = protocol
        meta["params"] = params if params is not None else {}
    return WorkerInfo(
        worker_id=wid,
        url=f"http://{wid}",
        model_name="test-model",
        engine=engine,
        disagg_mode=mode,
        disagg_meta=meta,
    )


class TestResolveProtocol:
    def test_matching_protocol(self):
        p = _worker("p:1", DisaggMode.PREFILL, params={"bootstrap_addr": "h:1"})
        d = _worker("d:1", DisaggMode.DECODE)
        proto = resolve_protocol(p, d)
        assert proto.name == "sglang-bootstrap"
        assert proto.topology == "concurrent"

    def test_p_missing_protocol_tag_raises(self):
        p = _worker("p:1", DisaggMode.PREFILL, protocol=None)
        d = _worker("d:1", DisaggMode.DECODE)
        with pytest.raises(ProtocolMismatch, match="missing"):
            resolve_protocol(p, d)

    def test_mismatched_protocols_raise(self):
        p = _worker("p:1", DisaggMode.PREFILL, protocol="sglang-bootstrap")
        d = _worker("d:1", DisaggMode.DECODE, protocol="vllm-mooncake")
        with pytest.raises(ProtocolMismatch, match="mismatch"):
            resolve_protocol(p, d)

    def test_unknown_protocol_raises(self):
        p = _worker("p:1", DisaggMode.PREFILL, protocol="vllm-nixl")
        d = _worker("d:1", DisaggMode.DECODE, protocol="vllm-nixl")
        with pytest.raises(UnknownProtocol, match="vllm-nixl"):
            resolve_protocol(p, d)


class TestSglangBootstrapProtocol:
    def setup_method(self):
        self.proto = SglangBootstrapProtocol()
        self.p = _worker("p:1", DisaggMode.PREFILL, params={"bootstrap_addr": "10.0.0.5:8998"})
        self.d = _worker("d:1", DisaggMode.DECODE)

    def test_annotate_prefill_injects_bootstrap_triple(self):
        body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
        out = self.proto.annotate_prefill(body, self.p, self.d, room_id=42)
        assert out["bootstrap_host"] == "10.0.0.5"
        assert out["bootstrap_port"] == 8998
        assert out["bootstrap_room"] == 42
        # Original body fields are preserved.
        assert out["model"] == "x"
        assert out["messages"] == body["messages"]

    def test_annotate_decode_matches_prefill(self):
        body = {"model": "x", "prompt": "hi"}
        p_body = self.proto.annotate_prefill(body, self.p, self.d, room_id=7)
        d_body = self.proto.annotate_decode(body, self.p, self.d, room_id=7, prefill_handoff=None)
        assert p_body == d_body

    def test_annotate_does_not_mutate_input(self):
        body = {"model": "x"}
        out = self.proto.annotate_prefill(body, self.p, self.d, room_id=1)
        assert "bootstrap_host" not in body  # original untouched
        assert "bootstrap_host" in out

    def test_missing_bootstrap_addr_raises(self):
        p = _worker("p:1", DisaggMode.PREFILL, params={})
        with pytest.raises(ValueError, match="bootstrap_addr"):
            self.proto.annotate_prefill({"model": "x"}, p, self.d, room_id=1)

    def test_missing_params_block_raises(self):
        # protocol tag present but no params dict at all.
        p = WorkerInfo(
            worker_id="p:1",
            url="http://p:1",
            model_name="test-model",
            disagg_mode=DisaggMode.PREFILL,
            disagg_meta={"protocol": "sglang-bootstrap"},  # no "params"
        )
        with pytest.raises(ValueError, match="bootstrap_addr"):
            self.proto.annotate_prefill({"model": "x"}, p, self.d, room_id=1)

    def test_ipv6_bootstrap_addr_parses(self):
        # bootstrap_addr uses rsplit(":", 1) so IPv6 host (which contains colons) survives.
        p = _worker(
            "p:1",
            DisaggMode.PREFILL,
            params={"bootstrap_addr": "fd00::1:8998"},
        )
        out = self.proto.annotate_prefill({"model": "x"}, p, self.d, room_id=1)
        assert out["bootstrap_host"] == "fd00::1"
        assert out["bootstrap_port"] == 8998

    def test_extract_handoff_returns_empty(self):
        # Concurrent topology — handoff extraction is never called by the router,
        # but the method must exist and return a safe value.
        assert self.proto.extract_handoff({"any": "payload"}) == {}
