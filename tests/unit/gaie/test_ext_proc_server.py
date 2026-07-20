###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import json

import pytest

from infera.common.worker_pool import DisaggMode, WorkerInfo, WorkerPool
from infera.gaie.endpoint_picker import EndpointPicker, _endpoint_of
from infera.gaie.ext_proc_server import (
    DEST_HEADER,
    PREFILL_HEADER,
    WORKER_HEADER,
    ExtProcServicer,
)
from infera.gaie.proto import ext_proc_pb2 as pb
from infera.router.policy.round_robin import RoundRobinPolicy


def _mixed_pool() -> WorkerPool:
    pool = WorkerPool()
    pool.add(WorkerInfo(worker_id="w1", url="http://10.0.0.1:8000", model_name="m"))
    pool.add(WorkerInfo(worker_id="w2", url="http://10.0.0.2:8000", model_name="m"))
    return pool


def _pd_pool() -> WorkerPool:
    pool = WorkerPool()
    pool.add(
        WorkerInfo(
            worker_id="p1",
            url="http://10.0.1.1:8000",
            model_name="m",
            disagg_mode=DisaggMode.PREFILL,
        )
    )
    pool.add(
        WorkerInfo(
            worker_id="d1",
            url="http://10.0.2.1:8000",
            model_name="m",
            disagg_mode=DisaggMode.DECODE,
        )
    )
    return pool


async def _drive(servicer: ExtProcServicer, requests: list[pb.ProcessingRequest]):
    async def gen():
        for r in requests:
            yield r

    out = []
    async for resp in servicer.Process(gen(), context=None):
        out.append(resp)
    return out


def _headers_map(common: pb.CommonResponse) -> dict[str, str]:
    return {
        opt.header.key: opt.header.raw_value.decode() for opt in common.header_mutation.set_headers
    }


def test_endpoint_of_variants():
    assert _endpoint_of("http://10.0.0.1:8000") == "10.0.0.1:8000"
    assert _endpoint_of("10.0.0.1:8000") == "10.0.0.1:8000"
    assert _endpoint_of("http://host") == "host"


@pytest.mark.asyncio
async def test_mixed_routes_on_body():
    picker = EndpointPicker(_mixed_pool(), RoundRobinPolicy())
    servicer = ExtProcServicer(picker)
    reqs = [
        pb.ProcessingRequest(request_headers=pb.HttpHeaders(end_of_stream=False)),
        pb.ProcessingRequest(
            request_body=pb.HttpBody(body=json.dumps({"model": "m"}).encode(), end_of_stream=True)
        ),
    ]
    out = await _drive(servicer, reqs)
    assert len(out) == 2
    # First (headers) is a plain CONTINUE with no mutation.
    assert out[0].WhichOneof("response") == "request_headers"
    assert not out[0].request_headers.response.header_mutation.set_headers
    # Second (body) carries the routing decision.
    assert out[1].WhichOneof("response") == "request_body"
    hdrs = _headers_map(out[1].request_body.response)
    assert hdrs[DEST_HEADER] in ("10.0.0.1:8000", "10.0.0.2:8000")
    assert hdrs[WORKER_HEADER] in ("w1", "w2")
    assert PREFILL_HEADER not in hdrs
    assert out[1].request_body.response.clear_route_cache is True


@pytest.mark.asyncio
async def test_disagg_sets_prefill_header():
    picker = EndpointPicker(_pd_pool(), RoundRobinPolicy())
    servicer = ExtProcServicer(picker)
    reqs = [
        pb.ProcessingRequest(
            request_body=pb.HttpBody(body=json.dumps({"model": "m"}).encode(), end_of_stream=True)
        ),
    ]
    out = await _drive(servicer, reqs)
    hdrs = _headers_map(out[0].request_body.response)
    # Primary endpoint is the decode worker; prefill is tagged separately.
    assert hdrs[DEST_HEADER] == "10.0.2.1:8000"
    assert hdrs[WORKER_HEADER] == "d1"
    assert hdrs[PREFILL_HEADER] == "p1"


@pytest.mark.asyncio
async def test_no_worker_for_model_sets_no_dest_header():
    picker = EndpointPicker(WorkerPool(), RoundRobinPolicy())
    servicer = ExtProcServicer(picker)
    reqs = [
        pb.ProcessingRequest(
            request_body=pb.HttpBody(
                body=json.dumps({"model": "absent"}).encode(), end_of_stream=True
            )
        ),
    ]
    out = await _drive(servicer, reqs)
    hdrs = _headers_map(out[0].request_body.response)
    assert DEST_HEADER not in hdrs


@pytest.mark.asyncio
async def test_release_decrements_inflight():
    pool = _mixed_pool()
    policy = RoundRobinPolicy()
    picker = EndpointPicker(pool, policy)
    servicer = ExtProcServicer(picker)
    reqs = [
        pb.ProcessingRequest(
            request_body=pb.HttpBody(body=json.dumps({"model": "m"}).encode(), end_of_stream=True)
        ),
    ]
    # Round-robin returns empty blocks, so this mainly asserts the stream's
    # finally-release path runs without error and the picker stays usable.
    await _drive(servicer, reqs)
    out2 = await _drive(servicer, reqs)
    assert out2[0].request_body.response.header_mutation.set_headers
