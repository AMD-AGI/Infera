###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""POST /v1/responses/input_tokens (LiteLLM CountTokens)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from infera.server import app as app_module
from infera.server.app import init_app


class _Policy:
    def __init__(self, n: int | None) -> None:
        self._n = n

    def count_input_tokens(self, request: dict) -> int | None:
        assert request.get("input") is not None
        return self._n


class _FakeRouter:
    def __init__(self, n: int | None) -> None:
        self.policy = _Policy(n)

    async def aclose(self) -> None:
        return


def _client(n: int | None = 12) -> TestClient:
    init_app(reg=None, rtr=_FakeRouter(n), kv=None)  # type: ignore[arg-type]
    return TestClient(app_module.app)


def test_counts_when_policy_can_tokenize():
    r = _client(7).post(
        "/v1/responses/input_tokens",
        json={"model": "m", "input": "hello"},
    )
    assert r.status_code == 200
    assert r.json() == {"object": "response.input_tokens", "input_tokens": 7}


def test_refuses_previous_response_id():
    r = _client().post(
        "/v1/responses/input_tokens",
        json={"model": "m", "input": "hello", "previous_response_id": "resp_1"},
    )
    assert r.status_code == 400


def test_requires_input():
    r = _client().post("/v1/responses/input_tokens", json={"model": "m"})
    assert r.status_code == 400


def test_unavailable_when_policy_cannot_count():
    r = _client(None).post(
        "/v1/responses/input_tokens",
        json={"model": "m", "input": "hello"},
    )
    assert r.status_code == 503
