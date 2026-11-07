###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The KV-event port a vLLM rank binds must be the one its worker advertises.

vLLM's ZmqEventPublisher binds the configured port plus its data-parallel rank
(``offset_endpoint_port``). The router subscribes to the advertised endpoint of
a rank-pinned worker as-is, so for an external-LB rank the configured port has
to leave room for that offset.
"""

from __future__ import annotations

import json

import pytest

from infera.common.worker_pool import DisaggMode
from infera.engine.vllm import __main__ as vllm_main
from infera.engine.vllm.args import VllmWorkerArgs


def _args(dp_rank: int | None, dp_size: int | None) -> VllmWorkerArgs:
    return VllmWorkerArgs(
        model="m",
        served_model_name=None,
        host="0.0.0.0",
        port=8000,
        block_size=16,
        discovery_backend="etcd",
        etcd_endpoint="127.0.0.1:2379",
        etcd_prefix="/infera",
        k8s_namespace=None,
        request_transport="http",
        advertise_host="10.0.0.5",
        disaggregation_allow_tcp=False,
        enable_kv_events=True,
        kv_events="auto",
        index_block_size=16,
        kv_event_transport="zmq",
        nats_server=None,
        disagg_mode=DisaggMode.MIXED,
        dp_rank=dp_rank,
        dp_size=dp_size,
    )


@pytest.mark.parametrize("dp_rank, dp_size", [(None, None), (0, 2), (1, 2), (3, 4)])
def test_rank_binds_the_advertised_port(monkeypatch, dp_rank, dp_size):
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "0")
    args = _args(dp_rank, dp_size)

    advertised = vllm_main._inject_kv_events_config(args)

    host, advertised_port = advertised.rsplit(":", 1)
    cfg = json.loads(args.vllm_argv[args.vllm_argv.index("--kv-events-config") + 1])
    configured = int(cfg["endpoint"].rsplit(":", 1)[1])
    assert host == "tcp://10.0.0.5"
    # What vLLM's offset_endpoint_port will actually bind for this rank.
    assert configured + (dp_rank or 0) == int(advertised_port)
