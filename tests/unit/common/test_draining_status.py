###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Announcing DRAINING, and what it is actually for.

``WorkerStatus.DRAINING`` has been in the enum and filtered out of
``list_active`` since the beginning, and until now nothing ever set it. The
value it adds over simply deleting the record is not routing -- both stop new
work -- it is that the worker stays *visible* while it drains. A worker that
vanishes looks identical to one that crashed; one that reports DRAINING tells an
operator a rollout is proceeding and roughly how far along it is.
"""

from __future__ import annotations

import json

import pytest

from infera.common.discovery import worker_info_from_json
from infera.common.registration import build_worker_payload
from infera.common.worker_pool import EngineType, WorkerPool, WorkerStatus
from infera.engine.base import EngineConfig


def _cfg():
    return EngineConfig(model_name="m", host="10.0.0.1", port=8080, engine=EngineType.SGLANG)


def test_healthy_payload_omits_status_entirely():
    """Keeps the record byte-identical to what older workers wrote, so the
    parser's ACTIVE default stands and the field's presence means something."""
    assert "status" not in build_worker_payload(_cfg())
    assert "status" not in build_worker_payload(_cfg(), status=WorkerStatus.ACTIVE)


def test_draining_payload_carries_the_status():
    payload = build_worker_payload(_cfg(), status=WorkerStatus.DRAINING)
    assert payload["status"] == "draining"


def test_round_trip_through_discovery():
    """The wire record has to survive the same parse every backend uses."""
    payload = build_worker_payload(_cfg(), status=WorkerStatus.DRAINING)
    info = worker_info_from_json(json.loads(json.dumps(payload)))
    assert info.status is WorkerStatus.DRAINING


def test_draining_worker_is_excluded_but_still_visible():
    """The whole point: out of rotation, still in the fleet listing."""
    pool = WorkerPool()
    pool.add(worker_info_from_json(build_worker_payload(_cfg())))
    assert [w.worker_id for w in pool.list_active()] == ["10.0.0.1:8080"]

    pool.add(worker_info_from_json(build_worker_payload(_cfg(), status=WorkerStatus.DRAINING)))
    assert pool.list_active() == [], "a draining worker must not be routed to"
    assert pool.get("10.0.0.1:8080") is not None, "but it must still be observable"


# --- the etcd client ----------------------------------------------------------


class _FakeHttp:
    def __init__(self):
        self.puts: list[dict] = []
        self.fail = False

    async def post(self, path, json=None):  # noqa: A002 - mirrors httpx
        if self.fail:
            raise RuntimeError("etcd unreachable")
        self.puts.append({"path": path, "json": json})

        class R:
            @staticmethod
            def raise_for_status():
                pass

        return R()


@pytest.mark.asyncio
async def test_etcd_announce_writes_draining_on_the_same_lease():
    from infera.common.registration import RegistrationClient

    c = RegistrationClient("http://etcd:2379")
    c._http = _FakeHttp()
    c._lease_id, c._key, c._worker_id, c._config = 42, "/infera/workers/w", "w", _cfg()

    assert await c.announce_draining() is True
    (put,) = c._http.puts
    assert put["path"] == "/v3/kv/put"
    assert put["json"]["lease"] == 42, "must keep the lease, not orphan the key"

    import base64

    value = json.loads(base64.b64decode(put["json"]["value"]))
    assert value["status"] == "draining"


@pytest.mark.asyncio
async def test_announce_never_raises_on_the_shutdown_path():
    """It runs immediately before the drain; raising here would skip it."""
    from infera.common.registration import RegistrationClient

    c = RegistrationClient("http://etcd:2379")
    c._http = _FakeHttp()
    c._http.fail = True
    c._lease_id, c._key, c._worker_id, c._config = 1, "/k", "w", _cfg()
    assert await c.announce_draining() is False


@pytest.mark.asyncio
async def test_announce_before_register_is_a_no_op():
    from infera.common.registration import RegistrationClient

    c = RegistrationClient("http://etcd:2379")
    c._http = _FakeHttp()
    assert await c.announce_draining() is False
    assert c._http.puts == []
