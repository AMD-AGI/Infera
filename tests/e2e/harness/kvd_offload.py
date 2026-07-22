###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Aggregated (mixed) + kvd L3 offload e2e scenario — vLLM only.

CI's mixed suite runs a bare mixed worker; it never exercises the kvd tiered
KV cache. This spawns the SAME mixed worker but with the ``InferaKvdConnector``
wired to a live ``infera.kvd`` daemon, drives a long shared-prefix request
through the router, and asserts the produced KV was offloaded to L3 (daemon
``sets_total`` > 0, ``misses_total`` == 0, ``long_bytes`` > 0).

bf16 KV (``INFERA_DEFAULT_KV_FP8=0``, overriding infera's fp8 product default)
so the offload is an unconditional plain passthrough — independent of the fp8
packed-KV path, which the connector's own unit + gather/scatter roundtrip tests
gate. See ``manual/features/kv_cache_offload.md``.
"""

from __future__ import annotations

import asyncio
import dataclasses

from . import client
from .adapter import emit_reporter_line
from .params import EngineParams

_KV_CONNECTOR_CFG = (
    '{"kv_connector":"InferaKvdConnector","kv_role":"kv_both",'
    '"kv_connector_module_path":"infera.engine.vllm.kvd_connector"}'
)

# A long shared prefix — many tokens over INFERA_KVD_CHUNK_TOKENS below — so at
# least one full chunk is produced and offloaded on a single request.
_PARA = (
    "The kvd tiered KV cache keeps warm prefixes across three storage tiers "
    "instead of recomputing them: GPU HBM is the fastest and smallest, host RAM "
    "is larger but slower, and local NVMe storage is the largest, the slowest, "
    "and durable across engine restarts. As HBM fills, blocks are demoted "
    "downward through the tiers, and on a cache hit they are promoted back up. "
)
_PROMPT = _PARA * 8 + " Question: name the three tiers, fastest to slowest. Answer:"


def kvd_params(base: EngineParams, socket: str) -> EngineParams:
    """Layer the kvd connector + a bf16-KV env onto ``base``."""
    return dataclasses.replace(
        base,
        extra_args=(
            *base.extra_args,
            "--enable-prefix-caching",
            "--kv-transfer-config",
            _KV_CONNECTOR_CFG,
        ),
        extra_env=(
            *base.extra_env,
            ("INFERA_KVD_SOCKET", socket),
            # Override infera's fp8 KV product default -> true bf16 KV, which the
            # connector offloads unconditionally (a plain passthrough, no packed
            # scale to reason about).
            ("INFERA_DEFAULT_KV_FP8", "0"),
            # Small fixed chunk grain so the request's prefix forms >=1 chunk.
            ("INFERA_KVD_CHUNK_TOKENS", "256"),
            # Stable block hashes so scheduler + worker derive the same chunk
            # keys (else misses_total > 0).
            ("PYTHONHASHSEED", "0"),
        ),
        server_ready_timeout=max(base.server_ready_timeout, 600),
    )


async def run_mixed_kvd_offload(server: dict, spawn, base_params: EngineParams, kvd: dict) -> None:
    """Spawn a kvd-enabled mixed worker, drive one long request through the
    router, and assert the KV was offloaded to the L3 tier."""
    params = kvd_params(base_params, kvd["socket"])
    await spawn(server, params)

    text = await client.completion_text(
        server["url"], params.model, _PROMPT, max_tokens=32, temperature=0.0
    )
    assert text, "empty completion from the kvd-enabled mixed worker"

    # The connector flushes chunk saves asynchronously after the step; poll the
    # daemon counters until the offload lands (or time out).
    stats: dict = {}
    for _ in range(40):
        stats = kvd["stats"]()
        if stats.get("sets_total", 0) > 0:
            break
        await asyncio.sleep(1.0)

    emit_reporter_line(f"[e2e kvd] daemon stats after request: {stats}")
    assert stats.get("sets_total", 0) > 0, (
        f"kvd offloaded nothing (sets_total=0); stats={stats}. Expected the mixed "
        "worker's InferaKvdConnector to save KV chunk(s) to L3."
    )
    assert stats.get("misses_total", 0) == 0, (
        f"kvd chunk-key mismatch (misses_total>0): {stats} — scheduler and worker "
        "derived different keys (PYTHONHASHSEED / INFERA_KVD_CHUNK_TOKENS skew?)."
    )
    assert stats.get("long_bytes", 0) > 0, f"nothing written to the L3 (long) tier: {stats}"
