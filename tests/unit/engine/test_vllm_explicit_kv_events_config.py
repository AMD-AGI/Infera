###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The vLLM worker advertises a caller-supplied --kv-events-config port and adds no config."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from infera.engine.vllm import __main__ as vllm_main


def _args(argv: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        enable_kv_events=True, advertise_host="10.0.0.1", host="0.0.0.0", vllm_argv=argv
    )


@pytest.fixture(autouse=True)
def _restore_env(monkeypatch):
    monkeypatch.delenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", raising=False)


def test_the_callers_port_is_advertised_and_not_reallocated(monkeypatch):
    monkeypatch.setattr(vllm_main, "free_tcp_port_block", lambda n: pytest.fail("allocated"))
    config = json.dumps(
        {
            "enable_kv_cache_events": True,
            "publisher": "zmq",
            "endpoint": "tcp://*:5560",
            "topic": "kv-events",
        }
    )
    args = _args(["--kv-events-config", config])
    assert vllm_main._inject_kv_events_config(args) == "tcp://10.0.0.1:5560"
    assert args.vllm_argv.count("--kv-events-config") == 1
    assert vllm_main.os.environ["VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES"] == "1"


def test_without_a_config_the_worker_still_allocates_its_own_port(monkeypatch):
    monkeypatch.setattr(vllm_main, "free_tcp_port_block", lambda n: 7000)
    args = _args(["--tensor-parallel-size", "8"])
    assert vllm_main._inject_kv_events_config(args) == "tcp://10.0.0.1:7000"
    i = args.vllm_argv.index("--kv-events-config")
    assert json.loads(args.vllm_argv[i + 1])["endpoint"] == "tcp://*:7000"
