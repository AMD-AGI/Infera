###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""``--timeout-keep-alive`` reaches the router's uvicorn config.

uvicorn's own default is 5 s, short enough that an agentic client reusing one
connection across idle turns can resend just as the router closes it.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI

from infera.server.__main__ import uvicorn_config
from infera.server.args import parse_server_args


def _keep_alive(argv: list[str]) -> int:
    args = parse_server_args(["--router-tokenizer-path", "m", *argv])
    return uvicorn_config(FastAPI(), args).timeout_keep_alive


def test_default_outlives_idle_agentic_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INFERA_TIMEOUT_KEEP_ALIVE", raising=False)
    assert _keep_alive([]) == 900


def test_flag_sets_uvicorn_timeout() -> None:
    assert _keep_alive(["--timeout-keep-alive", "30"]) == 30


def test_env_sets_default_and_flag_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERA_TIMEOUT_KEEP_ALIVE", "120")
    assert _keep_alive([]) == 120
    assert _keep_alive(["--timeout-keep-alive", "60"]) == 60
