# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The AgentsView side-car: which port, and every way it is allowed to fail."""

from __future__ import annotations

import http.server
import socket
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from env_mgr.o11y import agentsview
from env_mgr.o11y.prefix import Prefix


def test_the_default_port_is_18888() -> None:
    assert agentsview.DEFAULT_PORT == 18888
    assert agentsview.resolve_port(None, {}) == 18888


def test_the_environment_beats_the_default() -> None:
    assert agentsview.resolve_port(None, {"AGENTSVIEW_PORT": "9001"}) == 9001


def test_the_flag_beats_the_environment() -> None:
    assert agentsview.resolve_port(9002, {"AGENTSVIEW_PORT": "9001"}) == 9002


def test_an_unparseable_environment_value_falls_back_to_the_default() -> None:
    assert agentsview.resolve_port(None, {"AGENTSVIEW_PORT": "not-a-port"}) == 18888


def test_port_is_free_says_no_when_something_is_listening() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        taken = s.getsockname()[1]
        assert agentsview.port_is_free(taken) is False


def test_port_is_free_says_yes_when_nothing_is() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    assert agentsview.port_is_free(free) is True


@pytest.fixture()
def prefix(tmp_path: Path) -> Prefix:
    p = Prefix.resolve({"HOME": str(tmp_path)})
    p.create()
    return p


def _fake_binary(prefix: Prefix, body: str) -> None:
    exe = prefix.bin / "agentsview"
    exe.write_text("#!/bin/sh\n" + body)
    exe.chmod(0o755)


def test_a_taken_port_is_one_warning_and_a_skip(prefix, caplog) -> None:
    _fake_binary(prefix, "exit 0\n")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        taken = s.getsockname()[1]
        with caplog.at_level("WARNING"):
            status = agentsview.ensure_running(prefix, port=taken)
    assert status.running is False
    assert "port" in status.reason
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_a_missing_binary_is_a_warning_and_a_skip(prefix, caplog) -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    with caplog.at_level("WARNING"):
        status = agentsview.ensure_running(prefix, port=free)
    assert status.running is False
    assert "not installed" in status.reason
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_a_daemon_that_exits_nonzero_is_a_warning_and_a_skip(prefix, caplog) -> None:
    _fake_binary(prefix, "echo boom >&2\nexit 3\n")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    with caplog.at_level("WARNING"):
        status = agentsview.ensure_running(prefix, port=free)
    assert status.running is False
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_a_launch_that_times_out_is_a_warning_and_a_skip(prefix, caplog, monkeypatch) -> None:
    _fake_binary(prefix, "sleep 30\n")
    monkeypatch.setattr(agentsview, "LAUNCH_TIMEOUT_S", 0.2)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    with caplog.at_level("WARNING"):
        status = agentsview.ensure_running(prefix, port=free)
    assert status.running is False
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_no_failure_mode_raises(prefix, monkeypatch) -> None:
    """The whole point of the module, asserted directly."""
    monkeypatch.setattr(agentsview, "LAUNCH_TIMEOUT_S", 0.2)
    monkeypatch.setattr(agentsview, "HEALTH_TIMEOUT_S", 0.2)
    monkeypatch.setattr(agentsview, "REUSE_PROBE_TIMEOUT_S", 0.2)
    for body in ("exit 3\n", "sleep 30\n"):
        _fake_binary(prefix, body)
        agentsview.ensure_running(prefix, port=1)  # privileged port: bind fails
        agentsview.ensure_running(prefix, port=0)


def test_a_successful_launch_reports_the_url(prefix, monkeypatch) -> None:
    _fake_binary(prefix, "exit 0\n")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    monkeypatch.setattr(agentsview, "_wait_for_health", lambda url, timeout: True)
    status = agentsview.ensure_running(prefix, port=free)
    assert status.running is True
    assert status.url == f"http://127.0.0.1:{free}"


def test_the_child_gets_the_prefix_environment_and_os_environ_is_untouched(
    prefix, monkeypatch
) -> None:
    """`AGENTSVIEW_DATA_DIR` reaches the child; this process never learns it."""
    seen: dict[str, str] = {}

    def spy(cmd, env=None, **kw):  # noqa: ANN001
        seen.update(env or {})
        return subprocess.CompletedProcess(cmd, 0, "", "")

    _fake_binary(prefix, "exit 0\n")
    monkeypatch.setattr(subprocess, "run", spy)
    monkeypatch.setattr(agentsview, "_wait_for_health", lambda url, timeout: True)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    agentsview.ensure_running(prefix, port=free)
    assert seen["AGENTSVIEW_DATA_DIR"] == str(prefix.agentsview_data)
    assert seen["CLAUDE_PROJECTS_DIR"] == str(prefix.claude_home / "projects")
    assert "AGENTSVIEW_DATA_DIR" not in __import__("os").environ


@contextmanager
def _server_on_a_port(body: bytes, content_type: str) -> Iterator[int]:
    """A real HTTP server on a real ephemeral port, answering everything alike.

    Real rather than mocked because the thing under test is a decision about a
    *stranger's* process, and a mock of the stranger is a mock of exactly the
    party we do not control.
    """

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            """pytest's captured output is not a web server access log."""

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def _claim_port(prefix: Prefix, port: int) -> None:
    """Forge the evidence that *we* started the daemon on `port`."""
    (prefix.run / "agentsview.port").write_text(str(port))


def test_a_stranger_answering_http_on_our_port_is_not_adopted(
    prefix, caplog, monkeypatch
) -> None:
    """A 200 is not an identity.

    The port file is written here deliberately, so the *only* thing that can
    reject this server is the identity probe. Without it this test would pass
    on the ownership check alone and prove nothing about `/api/v1/agents`.
    """
    monkeypatch.setattr(agentsview, "REUSE_PROBE_TIMEOUT_S", 0.2)
    with _server_on_a_port(b"<html>some other service</html>", "text/html") as port:
        _claim_port(prefix, port)
        with caplog.at_level("WARNING"):
            status = agentsview.ensure_running(prefix, port=port)
    assert status.running is False
    assert status.url is None
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_someone_elses_agentsview_is_not_adopted(prefix, caplog, monkeypatch) -> None:
    """A genuine AgentsView we did not start shows the user's whole machine.

    Adopting it would satisfy the health check and break the one requirement
    the panel exists for — that it lists agent_sys's sessions and no others.
    """
    monkeypatch.setattr(agentsview, "REUSE_PROBE_TIMEOUT_S", 0.2)
    with _server_on_a_port(b'[{"name":"claude-code"}]', "application/json") as port:
        assert not (prefix.run / "agentsview.port").exists()  # control
        with caplog.at_level("WARNING"):
            status = agentsview.ensure_running(prefix, port=port)
    assert status.running is False
    assert status.url is None
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_our_own_resident_daemon_is_reused(prefix, caplog, monkeypatch) -> None:
    """Both gates pass: it answers as AgentsView, and we recorded starting it."""
    monkeypatch.setattr(agentsview, "REUSE_PROBE_TIMEOUT_S", 0.2)
    with _server_on_a_port(b'[{"name":"claude-code"}]', "application/json") as port:
        _claim_port(prefix, port)
        with caplog.at_level("WARNING"):
            status = agentsview.ensure_running(prefix, port=port)
    assert status.running is True
    assert status.reason == "already running"
    assert status.url == f"http://127.0.0.1:{port}"
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_the_recipe_installs_agentsview_as_an_optional_bin_item() -> None:
    """`suggested`, not `required`: install failure must stay a warning."""
    from env_mgr.recipe import load_recipe

    _target, items = load_recipe("env_mgr/recipes/agentsview.o11y.yaml")
    (item,) = [i for i in items if i.spec.get("name") == "agentsview"]
    assert item.installer == "bin"
    assert item.importance == "suggested"
    assert item.spec["check_cmd"] == "agentsview --version"
    assert "o11y" in item.tags


def test_the_recipe_pins_a_version_so_check_cmd_is_compared_against_something() -> None:
    """`satisfies(actual, None)` accepts anything; a bare `version:` fixes that."""
    from env_mgr.recipe import load_recipe
    from env_mgr.versions import satisfies

    _target, items = load_recipe("env_mgr/recipes/agentsview.o11y.yaml")
    (item,) = [i for i in items if i.spec.get("name") == "agentsview"]
    assert item.version == "0.42.0"
    assert satisfies("0.42.0", item.version) is True
    assert satisfies("0.41.0", item.version) is False


def test_the_recipe_install_command_uses_a_private_tempfile_and_verifies_checksum() -> None:
    """The three amendments: no fixed shared tempfile, and a real checksum gate."""
    from env_mgr.recipe import load_recipe

    _target, items = load_recipe("env_mgr/recipes/agentsview.o11y.yaml")
    (item,) = [i for i in items if i.spec.get("name") == "agentsview"]
    install = item.spec["install"]
    assert "mktemp" in install
    assert "/tmp/av.tgz" not in install
    assert "sha256sum" in install
    assert "SHA256SUMS" in install
