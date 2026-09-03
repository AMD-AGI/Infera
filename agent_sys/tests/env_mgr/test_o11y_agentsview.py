# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The AgentsView side-car: which port, and every way it is allowed to fail."""

from __future__ import annotations

import http.server
import os
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


def test_ensure_running_passes_replace_to_serve(prefix, monkeypatch) -> None:
    """Measured directly (scratch/port_repro, reported to team lead): without
    `--replace`, `serve --background --port N` silently attaches to any
    daemon already running for this `AGENTSVIEW_DATA_DIR` and reports *its*
    port, ignoring `N` entirely -- exit 0, no error, and our own health check
    on `N` then times out (correctly producing a warning, but only after
    burning the full launch+health timeout, and only ever reporting failure,
    never actually landing on the port we asked for). `--replace` is the flag
    that makes our chosen port actually take effect regardless of any stray
    daemon left over from an earlier run.
    """
    seen_cmd: list[str] = []

    def spy(cmd, env=None, **kw):  # noqa: ANN001
        seen_cmd.extend(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    _fake_binary(prefix, "exit 0\n")
    monkeypatch.setattr(subprocess, "run", spy)
    monkeypatch.setattr(agentsview, "_wait_for_health", lambda url, timeout: True)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    agentsview.ensure_running(prefix, port=free)
    assert "--replace" in seen_cmd


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
    assert item.spec["check_cmd"] == "$AGENT_SYS_HOME/bin/agentsview --version"
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


# --- ensure_installed: the recipe item, actually installed ---------------


@pytest.fixture()
def _no_leftover_environ(monkeypatch) -> None:
    """`AGENT_SYS_HOME` must not already be ambient, or a leak would go unseen."""
    monkeypatch.delenv("AGENT_SYS_HOME", raising=False)


def _fake_run_cmd(*, version_rc: int, install_rc: int, seen_home: list[str | None]):
    """Stands in for both `installers.base.subprocess.run` (a shell *string*,
    used by `run_cmd`) and `agentsview.py`'s own `subprocess.run` calls (an
    argv *list*, used by `check_disabled_agents`/`ensure_running`) -- the same
    monkeypatch target (`subprocess.run` is one shared module attribute)
    serves both call shapes, so this fake must accept the kwargs either
    caller passes (`env=`, `timeout=`) even though it only inspects a few.

    Distinguishes calls by content: the `--version` probe (`_satisfied`) vs.
    the recipe's `install:` body vs. a `health` validation probe (always
    reports success here -- there is a dedicated test for
    `check_disabled_agents` itself; this fake exists to test `ensure_installed`
    without that check's outcome contaminating the assertions). Records
    `AGENT_SYS_HOME` as seen in `os.environ` *at call time* for the first two
    -- the only way to observe whether `_patched_environ` actually reached the
    subprocess, since `run_cmd` passes no explicit `env=`.
    """

    def fake(cmd, shell=True, cwd=None, capture_output=True, text=True, env=None, timeout=None):  # noqa: ANN001
        cmd_text = cmd if isinstance(cmd, str) else " ".join(cmd)
        if "doctor sync" in cmd_text:
            return subprocess.CompletedProcess(
                cmd, 0, "Agent roots:\n  cursor: /fake/path (ok, default)\n", ""
            )
        if "health" in cmd_text:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        seen_home.append(os.environ.get("AGENT_SYS_HOME"))
        if "--version" in cmd_text:
            rc = version_rc
            out = "agentsview v0.42.0\n" if rc == 0 else ""
        else:
            rc = install_rc
            out = "" if rc == 0 else "boom\n"
        return subprocess.CompletedProcess(cmd, rc, out, "")

    return fake


def _install_item_for(prefix: Prefix):
    """Builds the callable `ensure_installed` expects, from the real recipe.

    `ensure_installed` takes this as a dependency rather than loading the
    recipe itself, because `env_mgr`'s installer machinery (`recipe`,
    `runner`, `installers/…`) is below spec §9's decoupling wall and `o11y` is
    not allowed to import it — checked structurally by `test_imports.py`, and
    the first draft of `ensure_installed` failed exactly that test by
    importing `recipe`/`runner` directly. Test code is not subject to the
    wall, so it exercises the real `agentsview.o11y.yaml` recipe end to end
    (`subprocess.run` faked aside), the same way a real caller would build
    this closure.
    """
    from env_mgr.recipe import load_recipe
    from env_mgr.runner import Filters, run

    def install_item():
        target, items = load_recipe(agentsview.RECIPE_PATH)
        target.path = str(prefix.root)
        outs, _status = run(target, items, "install", Filters(item="agentsview"))
        return outs

    return install_item


def test_ensure_installed_reports_already_present_without_reinstalling(
    prefix, monkeypatch, _no_leftover_environ
) -> None:
    seen_home: list[str | None] = []
    fake = _fake_run_cmd(version_rc=0, install_rc=1, seen_home=seen_home)
    monkeypatch.setattr(subprocess, "run", fake)

    status = agentsview.ensure_installed(prefix, _install_item_for(prefix))

    assert status.running is True
    assert "already present" in status.reason
    assert len(seen_home) == 1  # only the --version probe; install never ran
    assert seen_home[0] == str(prefix.root)


def test_ensure_installed_runs_the_recipe_when_missing(
    prefix, monkeypatch, _no_leftover_environ
) -> None:
    seen_home: list[str | None] = []
    fake = _fake_run_cmd(version_rc=1, install_rc=0, seen_home=seen_home)
    monkeypatch.setattr(subprocess, "run", fake)

    status = agentsview.ensure_installed(prefix, _install_item_for(prefix))

    assert status.running is True
    assert len(seen_home) == 2  # the failed probe, then the install
    assert seen_home == [str(prefix.root), str(prefix.root)]


def test_ensure_installed_is_one_warning_and_a_skip_when_install_fails(
    prefix, monkeypatch, caplog, _no_leftover_environ
) -> None:
    fake = _fake_run_cmd(version_rc=1, install_rc=3, seen_home=[])
    monkeypatch.setattr(subprocess, "run", fake)

    with caplog.at_level("WARNING"):
        status = agentsview.ensure_installed(prefix, _install_item_for(prefix))

    assert status.running is False
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_ensure_installed_restores_os_environ_even_when_the_installer_raises(
    prefix, monkeypatch, _no_leftover_environ
) -> None:
    """The exception path is the one that must not leak a half-patched environ."""

    def boom(*a, **k):  # noqa: ANN001, ANN002, ANN003
        raise RuntimeError("this must never escape, and must not leak the environ")

    monkeypatch.setattr(subprocess, "run", boom)
    before = dict(os.environ)

    status = agentsview.ensure_installed(prefix, _install_item_for(prefix))

    assert status.running is False
    assert dict(os.environ) == before
    assert "AGENT_SYS_HOME" not in os.environ


def test_ensure_installed_never_raises_regardless_of_outcome(
    prefix, monkeypatch, _no_leftover_environ
) -> None:
    for version_rc, install_rc in ((0, 1), (1, 0), (1, 3)):
        fake = _fake_run_cmd(version_rc=version_rc, install_rc=install_rc, seen_home=[])
        monkeypatch.setattr(subprocess, "run", fake)
        agentsview.ensure_installed(prefix, _install_item_for(prefix))  # must not raise


# --- check_disabled_agents: OTHER_PROVIDERS validated against the real binary,
# --- in both directions ---


def _fake_agentsview_doctor_sync(
    prefix: Prefix, *, rc: int, stdout: str = "", stderr: str = ""
) -> None:
    """`check_disabled_agents` and `discover_providers` both only ever run
    `doctor sync` -- never `health` -- so one fake, answering that one
    subcommand, covers every test for both of them."""
    exe = prefix.bin / "agentsview"
    exe.write_text(f"#!/bin/sh\nprintf '%s' '{stdout}'\nprintf '%s' '{stderr}' >&2\nexit {rc}\n")
    exe.chmod(0o755)


#: One recognized provider ("cursor") already in `OTHER_PROVIDERS`.
_CLEAN_DOCTOR_SYNC_STDOUT = "Agent roots:\n  cursor: /fake/path (ok, default)\n"


def test_check_disabled_agents_reports_nothing_when_both_directions_are_clean(prefix) -> None:
    _fake_agentsview_doctor_sync(prefix, rc=0, stdout=_CLEAN_DOCTOR_SYNC_STDOUT)
    assert agentsview.check_disabled_agents(prefix) == ()


def test_check_disabled_agents_names_a_renamed_or_removed_provider(prefix) -> None:
    """Direction 1: OTHER_PROVIDERS lists something the binary no longer knows."""
    _fake_agentsview_doctor_sync(
        prefix,
        rc=1,
        stderr='fatal: loading config: disabled_agents: unknown session provider "claude-cowork"',
    )
    assert agentsview.check_disabled_agents(prefix) == ("claude-cowork",)


def test_check_disabled_agents_names_a_provider_added_upstream_and_never_listed(
    prefix,
) -> None:
    """Direction 2, the one that leaks: the binary recognizes a provider
    OTHER_PROVIDERS never mentions -- AgentsView would scan its default
    directory and put its sessions on the panel with no error at all."""
    _fake_agentsview_doctor_sync(
        prefix,
        rc=0,
        stdout="Agent roots:\n  cursor: /fake/path (ok, default)\n"
        "  brand-new-provider: /fake/other (ok, default)\n",
    )
    assert agentsview.check_disabled_agents(prefix) == ("brand-new-provider",)


def test_check_disabled_agents_is_empty_not_a_false_accusation_when_the_probe_cannot_run(
    prefix,
) -> None:
    """A missing/broken binary is 'could not confirm', not 'something is wrong'."""
    assert not (prefix.bin / "agentsview").exists()
    assert agentsview.check_disabled_agents(prefix) == ()


def test_check_disabled_agents_is_empty_on_an_unrelated_failure(prefix) -> None:
    """Exit 1 with no recognizable message: still not a provider-name verdict."""
    _fake_agentsview_doctor_sync(prefix, rc=1, stderr="some unrelated crash")
    assert agentsview.check_disabled_agents(prefix) == ()


def test_check_disabled_agents_never_raises_on_a_hanging_binary(prefix, monkeypatch) -> None:
    monkeypatch.setattr(agentsview, "CHECK_DISABLED_AGENTS_TIMEOUT_S", 0.2)
    exe = prefix.bin / "agentsview"
    exe.write_text("#!/bin/sh\nsleep 30\n")
    exe.chmod(0o755)
    assert agentsview.check_disabled_agents(prefix) == ()


def test_check_disabled_agents_never_starts_a_daemon(prefix) -> None:
    """The bug this whole rewrite exists to close: `health` silently
    autostarted a daemon on a port AgentsView picked. `doctor sync` must
    not start anything at all, in either the config-valid or
    config-invalid case -- checked here by making the fake binary record
    every invocation rather than by asserting on a real process, since a
    unit test should not depend on a real daemon lifecycle to prove a
    negative."""
    exe = prefix.bin / "agentsview"
    calls_path = prefix.run / "calls.log"
    exe.write_text(
        "#!/bin/sh\n"
        f"echo \"$@\" >> {calls_path}\n"
        f"printf '%s' '{_CLEAN_DOCTOR_SYNC_STDOUT}'\n"
        "exit 0\n"
    )
    exe.chmod(0o755)
    agentsview.check_disabled_agents(prefix)
    calls = calls_path.read_text().splitlines()
    assert calls == ["doctor sync"]


# --- discover_providers: the enumeration `check_disabled_agents` uses ---


#: A shape modelled directly on a real `agentsview v0.42.0 doctor sync` run
#: (see PHASE0.md §0.3) -- several names repeat across multiple root lines,
#: which is exactly why `discover_providers` dedupes with a `set`.
_REAL_SHAPED_DOCTOR_SYNC_STDOUT = (
    "Sync Diagnostics\n"
    "Version: v0.42.0\n"
    "Agent roots:\n"
    "  claude: /home/x/.claude/projects (ok, configured)\n"
    "  openclaude: /home/x/.openclaude/projects (missing, default)\n"
    "  cowork: /home/x/Library/Application Support/Claude (missing, default)\n"
    "  cowork: /home/x/.config/Claude (missing, default)\n"
    "  cursor: /home/x/.cursor/projects (ok, default)\n"
    "Recent debug.log evidence:\n"
    "  none\n"
)


def test_discover_providers_parses_a_real_shaped_report(prefix) -> None:
    _fake_agentsview_doctor_sync(prefix, rc=0, stdout=_REAL_SHAPED_DOCTOR_SYNC_STDOUT)
    found = agentsview.discover_providers(prefix)
    assert found == ("cowork", "cursor", "openclaude")  # sorted, deduped, no claude


def test_discover_providers_is_none_when_the_binary_is_missing(prefix) -> None:
    assert not (prefix.bin / "agentsview").exists()
    assert agentsview.discover_providers(prefix) is None


def test_discover_providers_is_none_on_a_nonzero_exit(prefix) -> None:
    _fake_agentsview_doctor_sync(prefix, rc=1, stdout="")
    assert agentsview.discover_providers(prefix) is None


def test_discover_providers_is_none_when_the_report_has_no_agent_roots_section(prefix) -> None:
    _fake_agentsview_doctor_sync(prefix, rc=0, stdout="Sync Diagnostics\nVersion: v0.42.0\n")
    assert agentsview.discover_providers(prefix) is None


def test_discover_providers_is_none_rather_than_empty_when_only_claude_is_found(
    prefix,
) -> None:
    """An empty tuple would read as 'nothing else exists', the single most
    permissive way this function could fail `check_disabled_agents`."""
    _fake_agentsview_doctor_sync(
        prefix, rc=0, stdout="Agent roots:\n  claude: /x/.claude/projects (ok, configured)\n"
    )
    assert agentsview.discover_providers(prefix) is None


def test_discover_providers_never_raises_on_a_hanging_binary(prefix, monkeypatch) -> None:
    monkeypatch.setattr(agentsview, "DISCOVER_PROVIDERS_TIMEOUT_S", 0.2)
    exe = prefix.bin / "agentsview"
    exe.write_text("#!/bin/sh\nsleep 30\n")
    exe.chmod(0o755)
    assert agentsview.discover_providers(prefix) is None


def test_write_config_writes_exactly_the_providers_it_is_given(prefix) -> None:
    """A pure writer: no fake binary, no subprocess, just the list in -> the
    list out, in the file `check_disabled_agents`/`serve` will read."""
    agentsview.write_config(prefix, ("foo", "bar"))
    text = (prefix.agentsview_data / "config.toml").read_text()
    assert 'disabled_agents = ["foo", "bar"]' in text


def test_write_config_keeps_the_daemon_alive_indefinitely(prefix) -> None:
    """Design §4 promises the panel persists across runs; AgentsView's own
    default (20m idle exit) contradicts that unless this key overrides it."""
    agentsview.write_config(prefix, ())
    text = (prefix.agentsview_data / "config.toml").read_text()
    assert 'daemon_idle_timeout = "0s"' in text


def test_other_providers_excludes_claude_itself() -> None:
    """The one provider gate 3 must never disable."""
    assert "claude" not in agentsview.OTHER_PROVIDERS


def test_ensure_installed_warns_once_naming_a_renamed_provider(
    prefix, monkeypatch, caplog, _no_leftover_environ
) -> None:
    """The install-time check, wired end to end through `ensure_installed`,
    direction 1: OTHER_PROVIDERS lists something the binary rejects.
    `check_disabled_agents` only ever runs `doctor sync` now (never
    `health` -- see its docstring), so the rejection comes from that call."""
    seen_home: list[str | None] = []

    def fake(cmd, shell=True, cwd=None, capture_output=True, text=True, env=None, timeout=None):  # noqa: ANN001
        cmd_text = cmd if isinstance(cmd, str) else " ".join(cmd)
        if "doctor sync" in cmd_text:
            return subprocess.CompletedProcess(
                cmd, 1, "", 'fatal: unknown session provider "bogus-provider"'
            )
        seen_home.append(os.environ.get("AGENT_SYS_HOME"))
        return subprocess.CompletedProcess(cmd, 0, "agentsview v0.42.0\n", "")

    monkeypatch.setattr(subprocess, "run", fake)

    with caplog.at_level("WARNING"):
        status = agentsview.ensure_installed(prefix, _install_item_for(prefix))

    assert status.running is True  # the binary install itself still succeeded
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "bogus-provider" in warnings[0].getMessage()


def test_ensure_installed_warns_once_naming_a_provider_added_upstream(
    prefix, monkeypatch, caplog, _no_leftover_environ
) -> None:
    """Direction 2, end to end: a provider `doctor sync` reports that
    OTHER_PROVIDERS never listed -- the direction that leaks silently."""

    def fake(cmd, shell=True, cwd=None, capture_output=True, text=True, env=None, timeout=None):  # noqa: ANN001
        cmd_text = cmd if isinstance(cmd, str) else " ".join(cmd)
        if "doctor sync" in cmd_text:
            return subprocess.CompletedProcess(
                cmd,
                0,
                "Agent roots:\n  cursor: /fake/path (ok, default)\n"
                "  brand-new-provider: /fake/other (ok, default)\n",
                "",
            )
        return subprocess.CompletedProcess(cmd, 0, "agentsview v0.42.0\n", "")

    monkeypatch.setattr(subprocess, "run", fake)

    with caplog.at_level("WARNING"):
        status = agentsview.ensure_installed(prefix, _install_item_for(prefix))

    assert status.running is True
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "brand-new-provider" in warnings[0].getMessage()
