###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for infera/kvd/statctl.py — the ops CLI that dumps kvd stats.

We spin up a real kvd daemon in-process and invoke the CLI via the
helper functions (not subprocess — that would slow the suite down and
mostly retest argparse).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import pytest

from infera.kvd.server import KvdServer
from infera.kvd.statctl import _print_stats


@pytest.fixture
async def kvd_daemon(tmp_path: Path):
    socket = tmp_path / f"kvd-statctl-{uuid.uuid4().hex[:8]}.sock"
    server = KvdServer(socket_path=socket, max_bytes=1 << 20)
    await server.start()
    serve_task = asyncio.create_task(server.serve_forever(), name="kvd-statctl-test")
    await asyncio.sleep(0)
    yield server, str(socket)
    server.shutdown()
    try:
        await asyncio.wait_for(serve_task, timeout=2.0)
    except asyncio.TimeoutError:
        serve_task.cancel()
        try:
            await serve_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_statctl_prints_json_when_daemon_reachable(kvd_daemon, capsys):
    _, socket = kvd_daemon
    code = await _print_stats(socket, client_id="test")
    assert code == 0

    out = capsys.readouterr().out
    parsed = json.loads(out)
    # Must contain the documented schema — operators may grep for these fields.
    assert set(parsed) == {
        "entries",
        "host_bytes",
        "spillover_bytes",
        "long_bytes",
        "gets_total",
        "sets_total",
        "hits_total",
        "misses_total",
        "evictions_total",
    }
    assert all(isinstance(v, int) for v in parsed.values())


@pytest.mark.asyncio
async def test_statctl_reports_error_on_unreachable_socket(tmp_path, capsys):
    nonexistent = tmp_path / "does-not-exist.sock"
    code = await _print_stats(str(nonexistent), client_id="test")
    assert code == 1

    captured = capsys.readouterr()
    # Error goes to stderr; stdout stays clean so JSON consumers don't
    # try to parse an error message.
    assert captured.out == ""
    assert "failed to connect" in captured.err
