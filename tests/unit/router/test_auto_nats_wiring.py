###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Regression: AutoRouter must pass the NATS request client to BOTH inner
routers, so PD (DisaggRouter) dispatch uses the per-instance NATS transport
instead of silently falling back to HTTP."""

from __future__ import annotations

import pytest

from infera.common.worker_pool import WorkerPool
from infera.router.auto import AutoRouter


class _FakePolicy:
    def pick(self, candidates, body):
        raise AssertionError("not exercised")

    def on_request_started(self, *a):
        pass

    def on_request_finished(self, *a):
        pass


class _SentinelNats:
    async def admit(self, worker_id):
        return True


@pytest.mark.asyncio
async def test_auto_router_wires_nats_into_disagg_and_mixed():
    nats = _SentinelNats()
    router = AutoRouter(WorkerPool(), _FakePolicy(), nats_client=nats)
    try:
        # The bug was: DisaggRouter got constructed without nats_client, so PD
        # always used HTTP. Both inner routers must share the same client.
        assert router._mixed.nats_client is nats
        assert router._disagg.nats_client is nats
    finally:
        await router.aclose()


@pytest.mark.asyncio
async def test_auto_router_none_nats_is_http_for_both():
    router = AutoRouter(WorkerPool(), _FakePolicy(), nats_client=None)
    try:
        assert router._mixed.nats_client is None
        assert router._disagg.nats_client is None
    finally:
        await router.aclose()
