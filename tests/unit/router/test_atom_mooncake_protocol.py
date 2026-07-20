###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unit tests for the ATOM Mooncake PD protocol."""

from __future__ import annotations

import pytest

from infera.common.worker_pool import DisaggMode, EngineType, WorkerInfo
from infera.router.disagg_protocols import resolve_protocol
from infera.router.disagg_protocols.atom_mooncake import AtomMooncakeProtocol


def _worker(wid: str, mode: DisaggMode, params: dict | None = None) -> WorkerInfo:
    host, port = wid.rsplit(":", 1)
    meta: dict = {"protocol": "atom-mooncake"}
    meta["params"] = (
        params
        if params is not None
        else {
            "host": host,
            "http_port": int(port),
            "handshake_port": 6301,
            "tp_size": 1,
            "dp_size": 1,
        }
    )
    return WorkerInfo(
        worker_id=wid,
        url=f"http://{wid}",
        model_name="test-model",
        engine=EngineType.ATOM,
        disagg_mode=mode,
        disagg_meta=meta,
    )


class TestAtomMooncakeProtocol:
    def setup_method(self):
        self.proto = AtomMooncakeProtocol()
        self.p = _worker("10.0.0.1:30001", DisaggMode.PREFILL)
        self.d = _worker("10.0.0.2:30002", DisaggMode.DECODE)

    def test_resolves_and_is_serial_pull(self):
        proto = resolve_protocol(self.p, self.d)
        assert proto.name == "atom-mooncake"
        assert proto.topology == "serial-pull"

    def test_annotate_prefill_single_token_and_targets_decode(self):
        body = {"model": "x", "prompt": "hi", "max_tokens": 50, "stream": True}
        out = self.proto.annotate_prefill(body, self.p, self.d, room_id=1)
        assert out["max_tokens"] == 1
        assert out["stream"] is False
        ktp = out["kv_transfer_params"]
        assert ktp["do_remote_decode"] is True
        assert ktp["do_remote_prefill"] is False
        # Prefill is told where the decode peer lives.
        assert ktp["remote_host"] == "10.0.0.2"
        assert ktp["remote_port"] == 30002
        assert ktp["remote_handshake_port"] == 6301
        assert ktp["remote_engine_id"] is None
        assert ktp["remote_block_ids"] is None
        # Original body untouched.
        assert body["max_tokens"] == 50

    # A realistic prefill-response kv_transfer_params, as ATOM's
    # MooncakeConnectorScheduler.request_finished emits it (the producer fills
    # its own real addressing + the rank that actually stored the KV).
    @staticmethod
    def _prefill_response_kv(dp_rank: int = 0) -> dict:
        return {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_block_ids": [1, 2, 3],
            "remote_engine_id": "10.0.0.1:5555",
            "remote_host": "10.0.0.1",
            "remote_port": 48123,
            "remote_handshake_port": 6301,
            "tp_size": 1,
            "dp_rank": dp_rank,
            "transfer_id": 123,
            "first_token_id": 7,
            "local_slot_index": -1,
        }

    def test_extract_handoff_returns_full_kv(self):
        kv = self._prefill_response_kv()
        payload = {"kv_transfer_params": kv}
        handoff = self.proto.extract_handoff(payload)
        # Full passthrough (shallow copy), so the decode leg can reuse it.
        assert handoff == kv
        assert handoff is not kv

    def test_extract_handoff_missing_kv_raises(self):
        with pytest.raises(ValueError, match="kv_transfer_params"):
            self.proto.extract_handoff({"choices": []})

    def test_extract_handoff_missing_transfer_id_raises(self):
        with pytest.raises(KeyError, match="transfer_id"):
            self.proto.extract_handoff({"kv_transfer_params": {"first_token_id": 7}})

    def test_annotate_decode_reuses_response_and_enriches(self):
        body = {"model": "x", "prompt": "hi", "max_tokens": 50}
        handoff = self.proto.extract_handoff({"kv_transfer_params": self._prefill_response_kv()})
        out = self.proto.annotate_decode(body, self.p, self.d, room_id=1, prefill_handoff=handoff)
        # Decode keeps the original max_tokens (ATOM re-predicts T0).
        assert out["max_tokens"] == 50
        ktp = out["kv_transfer_params"]
        # Verbatim reuse of the producer's own response fields.
        assert ktp["do_remote_prefill"] is True
        assert ktp["do_remote_decode"] is False
        assert ktp["remote_host"] == "10.0.0.1"
        assert ktp["remote_port"] == 48123
        assert ktp["remote_handshake_port"] == 6301
        assert ktp["remote_block_ids"] == [1, 2, 3]
        assert ktp["transfer_id"] == 123
        assert ktp["first_token_id"] == 7
        # enrich_decode_kv: producer parallelism + the rank that holds the KV.
        assert ktp["remote_dp_size"] == 1
        assert ktp["remote_tp_size"] == 1
        assert ktp["tp_size"] == 1
        assert ktp["remote_dp_rank"] == 0
        # Original body untouched.
        assert "kv_transfer_params" not in body

    def test_annotate_decode_dpa_maps_dp_rank_to_remote_dp_rank(self):
        # DP-attention producer: dp_size=8, request landed on rank 3.
        p = _worker(
            "10.0.0.1:30001",
            DisaggMode.PREFILL,
            params={
                "host": "10.0.0.1",
                "http_port": 30001,
                "handshake_port": 6301,
                "tp_size": 8,
                "dp_size": 8,
            },
        )
        handoff = self.proto.extract_handoff(
            {"kv_transfer_params": self._prefill_response_kv(dp_rank=3)}
        )
        out = self.proto.annotate_decode(
            {"model": "x", "prompt": "hi", "max_tokens": 50},
            p,
            self.d,
            room_id=1,
            prefill_handoff=handoff,
        )
        ktp = out["kv_transfer_params"]
        # Consumer dials remote_handshake_port + remote_dp_rank*tp_size + tp_rank.
        assert ktp["remote_dp_rank"] == 3
        assert ktp["remote_dp_size"] == 8
        assert ktp["remote_tp_size"] == 8

    def test_annotate_decode_non_numeric_dp_rank_left_unset(self):
        # Missing/null dp_rank -> leave remote_dp_rank unset so the consumer
        # ReqMeta defaults it to 0 (single-rank behaviour).
        kv = self._prefill_response_kv()
        kv.pop("dp_rank")
        handoff = self.proto.extract_handoff({"kv_transfer_params": kv})
        out = self.proto.annotate_decode(
            {"model": "x", "prompt": "hi"}, self.p, self.d, room_id=1, prefill_handoff=handoff
        )
        assert "remote_dp_rank" not in out["kv_transfer_params"]

    def test_annotate_decode_without_transfer_id_raises(self):
        with pytest.raises(ValueError, match="transfer_id"):
            self.proto.annotate_decode(
                {"model": "x", "prompt": "hi"}, self.p, self.d, room_id=1, prefill_handoff={}
            )

    def test_missing_params_raises(self):
        d = _worker("10.0.0.2:30002", DisaggMode.DECODE, params={})
        with pytest.raises(ValueError, match="params"):
            self.proto.annotate_prefill({"model": "x"}, self.p, d, room_id=1)
