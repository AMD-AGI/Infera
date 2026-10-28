###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A caller-supplied --kv-events-config keeps its port: the worker advertises it and adds none."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from importlib import import_module
from types import ModuleType, SimpleNamespace

import pytest


def _load_worker():
    # The unit-test extra does not install SGLang; the worker only needs ServerArgs as a type.
    if importlib.util.find_spec("sglang") is not None:
        return import_module("infera.engine.sglang.worker")
    sglang = ModuleType("sglang")
    sglang.__path__ = []
    srt = ModuleType("sglang.srt")
    srt.__path__ = []
    server_args = ModuleType("sglang.srt.server_args")
    server_args.ServerArgs = object
    fake_modules = {"sglang": sglang, "sglang.srt": srt, "sglang.srt.server_args": server_args}
    touched = [*fake_modules, "infera.engine.sglang.worker"]
    previous = {name: sys.modules.get(name) for name in touched}
    try:
        sys.modules.update(fake_modules)
        return import_module("infera.engine.sglang.worker")
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


worker = _load_worker()


def _config(port: int) -> str:
    return json.dumps({"publisher": "zmq", "endpoint": f"tcp://*:{port}", "topic": "kv-events"})


@pytest.mark.parametrize(
    ("argv", "port"),
    [
        (["--tp-size", "4"], None),
        (["--kv-events-config", _config(5560), "--tp-size", "4"], 5560),
        ([f"--kv-events-config={_config(5570)}"], 5570),
    ],
)
def test_explicit_port_is_read_from_either_flag_form(argv, port):
    assert worker._explicit_kv_events_port(argv) == port


def test_a_config_without_a_port_is_rejected():
    with pytest.raises(ValueError, match="endpoint with a port"):
        worker._explicit_kv_events_port(["--kv-events-config", json.dumps({"publisher": "zmq"})])


def _start(monkeypatch, argv: list[str], allocate=None):
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(poll=lambda: None)

    async def ready(self, timeout=0):
        return None

    async def page_size(self):
        return 64

    monkeypatch.setattr(worker.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(worker.SglangEngine, "_wait_ready", ready)
    monkeypatch.setattr(worker.SglangEngine, "_resolve_page_size", page_size)
    monkeypatch.setattr(
        worker, "free_tcp_port_block", allocate or (lambda n: pytest.fail("allocated a port"))
    )
    server_args = SimpleNamespace(
        dp_size=4,
        host="0.0.0.0",
        port=6100,
        disaggregation_mode="prefill",
        disaggregation_bootstrap_port=30001,
        served_model_name="m",
        model_path="/model",
        page_size=None,
        speculative_algorithm=None,
        speculative_eagle_topk=None,
    )
    engine = worker.SglangEngine(
        server_args, argv, advertise_host="10.0.0.1", enable_kv_events=True
    )
    config = asyncio.run(engine.start())
    return captured["cmd"], config


def test_the_callers_port_is_advertised_and_not_reallocated(monkeypatch):
    cmd, config = _start(monkeypatch, ["--kv-events-config", _config(5560)])
    assert cmd.count("--kv-events-config") == 1
    assert config.kv_events_endpoint == "tcp://10.0.0.1:5560"
    assert config.kv_block_size == 64
    assert config.dp_size == 4


def test_without_a_config_the_worker_still_allocates_one_port_per_dp_rank(monkeypatch):
    blocks = []

    def allocate(n):
        blocks.append(n)
        return 7000

    cmd, config = _start(monkeypatch, ["--tp-size", "4"], allocate=allocate)
    assert blocks == [4]
    i = cmd.index("--kv-events-config")
    assert json.loads(cmd[i + 1])["endpoint"] == "tcp://*:7000"
    assert config.kv_events_endpoint == "tcp://10.0.0.1:7000"
