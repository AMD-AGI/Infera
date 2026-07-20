###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Regression: vLLM worker must not register engine_block_size=None.

vLLM's ``--block-size`` defaults to ``None`` (vLLM auto-selects at runtime).
The launcher passed that straight into the kv registration block, so the
server's ``KvRegistrationMetadata.from_dict`` did ``int(None)`` and dropped
the whole ``WorkerInfo`` -- the worker became invisible even for plain
round-robin routing. The launcher must fall back to a real block size
(vLLM's ROCm default is 16), mirroring the SGLang launcher's ``page_size or 1``.
"""

from __future__ import annotations

import pytest

# Importing the launcher module pulls infera runtime deps (httpx, etc.);
# this test runs in the vLLM engine container, skips on bare dev boxes.
vmain = pytest.importorskip("infera.engine.vllm.__main__")

from infera.common.worker_pool import (  # noqa: E402
    DisaggMode,
    KvRegistrationMetadata,
)
from infera.engine.sglang import kv_wiring  # noqa: E402


def _args(block_size):
    """Minimal VllmWorkerArgs with kv_events=auto (so kv metadata is built)."""
    from infera.engine.vllm.args import VllmWorkerArgs

    return VllmWorkerArgs(
        model="Qwen/Qwen3-0.6B",
        served_model_name=None,
        host="0.0.0.0",
        port=30010,
        block_size=block_size,
        etcd_endpoint="127.0.0.1:2479",
        etcd_prefix="/infera/vllm/",
        advertise_host=None,
        disaggregation_allow_tcp=False,
        enable_kv_events=False,
        kv_events="auto",
        index_block_size=64,
        disagg_mode=DisaggMode.MIXED,
        # Required since the PD-disaggregation work added discovery/transport
        # fields to VllmWorkerArgs; defaults mirror the etcd/http single-node path.
        discovery_backend="etcd",
        k8s_namespace=None,
        request_transport="http",
        kv_event_transport="zmq",
        nats_server=None,
    )


def _stub_compute(captured):
    """Replacement for compute_kv_metadata that records the block size and
    returns a valid metadata object (no tokenizer / model files needed)."""

    def fake(
        *,
        model_id,
        engine_block_size,
        index_block_size,
        events_endpoint,
        snapshot_endpoint,
        trust_remote_code=False,
        **_,
    ):
        captured["engine_block_size"] = engine_block_size
        return KvRegistrationMetadata(
            engine_block_size=engine_block_size,
            index_block_size=index_block_size,
            tokenizer=model_id,
            tokenizer_digest="deadbeef",
            tokenizer_canary=[1, 2, 3],
            events_endpoint=events_endpoint or None,
            snapshot_endpoint=snapshot_endpoint,
        )

    return fake


def test_block_size_none_falls_back_to_real_int(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(kv_wiring, "compute_kv_metadata", _stub_compute(captured))

    md = vmain._compute_kv_metadata(
        _args(block_size=None), model_name="Qwen/Qwen3-0.6B", events_endpoint=None
    )

    assert md is not None
    # The bug: None flowed through here, later crashing from_dict server-side.
    assert isinstance(captured["engine_block_size"], int)
    assert captured["engine_block_size"] > 0
    # The actual crash site: server-side deserialization must round-trip.
    KvRegistrationMetadata.from_dict(md.to_dict())


def test_explicit_block_size_is_preserved(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(kv_wiring, "compute_kv_metadata", _stub_compute(captured))

    vmain._compute_kv_metadata(
        _args(block_size=32), model_name="Qwen/Qwen3-0.6B", events_endpoint=None
    )

    assert captured["engine_block_size"] == 32
