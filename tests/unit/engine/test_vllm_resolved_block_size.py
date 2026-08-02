###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The block size a vLLM worker registers must be the one the engine resolved,
not the one that was on the command line.

``--block-size`` defaults to None and the platform picks the real value after
the model loads. Kimi-K3 is the loud case: a hybrid Mamba model, so vLLM logs
"Setting attention block size to 768 tokens to ensure that attention page size
is >= mamba page size" while the parsed namespace still says None. Registering
that None left the router with no block size at all, and kv-aware routing
reported cache_hits=0 on every decision.

The metrics line below is copied verbatim from a running Kimi-K3 worker, so the
parser is pinned against the real format rather than an idealised one.
"""

from __future__ import annotations

import pytest

from infera.engine.vllm.worker import VllmEngine

# Verbatim from `curl localhost:30000/metrics` on kimi-k3-opt-pd-prefill.
REAL_METRICS = (
    "# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig\n"
    "# TYPE vllm:cache_config_info gauge\n"
    'vllm:cache_config_info{_block_size_resolved="True",block_size="768",'
    'cache_dtype="auto",calculate_kv_scales="False",enable_prefix_caching="True",'
    'engine="0",gpu_memory_utilization="0.88",is_attention_free="False",'
    'mamba_block_size="16",num_gpu_blocks="2169",'
    'user_specified_block_size="False"} 1.0\n'
)


def _engine(cli_block_size):
    return VllmEngine(
        vllm_argv=[],
        model_name="m",
        host="0.0.0.0",
        port=30000,
        kv_block_size=cli_block_size,
    )


class _Resp:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code


def _patch_get(monkeypatch, result):
    """Stub httpx.AsyncClient.get; ``result`` is a _Resp or an exception."""

    async def fake_get(self, url, **kw):
        if isinstance(result, Exception):
            raise result
        return result

    import httpx

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)


async def test_reads_the_resolved_value_not_the_cli_one(monkeypatch):
    _patch_get(monkeypatch, _Resp(REAL_METRICS))
    # None on the command line is exactly the Kimi-K3 case.
    assert await _engine(None)._resolve_block_size() == 768


async def test_resolved_value_overrides_an_explicit_one(monkeypatch):
    """vLLM may override what the user asked for; the router has to page the way
    the engine actually pages, not the way it was requested."""
    _patch_get(monkeypatch, _Resp(REAL_METRICS))
    assert await _engine(16)._resolve_block_size() == 768


@pytest.mark.parametrize(
    "result",
    [
        _Resp("", 500),
        _Resp("vllm:num_requests_running 0.0\n"),  # metric absent (older vLLM)
        RuntimeError("connection refused"),
    ],
)
async def test_falls_back_to_the_cli_value(monkeypatch, result):
    """Wrong beats absent: absent disables kv-aware routing for the worker."""
    _patch_get(monkeypatch, result)
    assert await _engine(16)._resolve_block_size() == 16


async def test_fallback_of_none_stays_none(monkeypatch):
    """No metric and no CLI value means we genuinely do not know — registering a
    made-up number would be worse, so None propagates and the router refuses to
    subscribe rather than guessing 1."""
    _patch_get(monkeypatch, RuntimeError("boom"))
    assert await _engine(None)._resolve_block_size() is None


async def test_start_registers_the_resolved_value(monkeypatch):
    """Pins the wiring, not just the parser.

    Testing ``_resolve_block_size`` alone would still pass if ``start()`` went
    back to putting ``self.kv_block_size`` in the EngineConfig — which is the
    original bug, and a one-word edit away.
    """
    _patch_get(monkeypatch, _Resp(REAL_METRICS))

    class _Proc:
        def poll(self):
            return None

    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: _Proc())

    async def noop_ready(self, timeout=None):
        return None

    monkeypatch.setattr(VllmEngine, "_wait_ready", noop_ready)

    cfg = await _engine(None).start()
    assert cfg.kv_block_size == 768, "start() must register what the engine resolved"
