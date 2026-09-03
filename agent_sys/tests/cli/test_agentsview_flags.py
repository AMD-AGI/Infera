# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The two flags, and the promise that o11y cannot fail a run."""

from __future__ import annotations

from cli import main as cli_main


def test_the_port_flag_parses() -> None:
    args = cli_main.parser().parse_args(["run", "--package", "pkg", "--agentsview-port", "9001"])
    assert args.agentsview_port == 9001


def test_the_disable_flag_parses() -> None:
    args = cli_main.parser().parse_args(["run", "--package", "pkg", "--no-agentsview"])
    assert args.no_agentsview is True


def test_the_default_is_enabled_and_unset() -> None:
    args = cli_main.parser().parse_args(["run", "--package", "pkg"])
    assert args.no_agentsview is False
    assert args.agentsview_port is None


def test_disabled_makes_no_external_call(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(cli_main, "ensure_running", lambda *a, **k: called.append(1))
    cli_main._start_o11y(port_flag=None, disabled=True)
    assert called == []


def test_a_dry_run_starts_no_daemon(monkeypatch) -> None:
    """`--dry-run` promises *resolve everything, do nothing*.

    Starting a resident daemon, creating `~/.infera_agent_sys` and writing a
    `config.toml` are all side effects, and a dry run that leaves a daemon
    behind has broken its only contract.
    """
    called = []
    monkeypatch.setattr(cli_main, "ensure_running", lambda *a, **k: called.append(1))
    monkeypatch.setattr(cli_main, "_dry_run", lambda args, stream: 0)
    assert cli_main.main(["run", "--package", "pkg", "--dry-run"]) == 0
    assert called == []


def test_clean_starts_no_daemon(monkeypatch) -> None:
    """`--clean` removes every run and exits; a panel for it is pointless."""
    called = []
    monkeypatch.setattr(cli_main, "ensure_running", lambda *a, **k: called.append(1))
    monkeypatch.setattr(cli_main, "_clean", lambda args, stream: 0)
    assert cli_main.main(["run", "--package", "pkg", "--clean"]) == 0
    assert called == []


def test_a_raising_side_car_does_not_reach_the_caller(monkeypatch) -> None:
    """Belt and braces: even a bug inside ensure_running cannot fail a run."""

    def boom(*a, **k):
        raise RuntimeError("this must never escape")

    monkeypatch.setattr(cli_main, "ensure_running", boom)
    assert cli_main._start_o11y(port_flag=None, disabled=False) is None


def test_the_side_car_never_exports_into_this_process(monkeypatch) -> None:
    """`_start_o11y` reads `os.environ`; it must never write to it.

    The panel's whole isolation story is that `CLAUDE_CONFIG_DIR` lives in a
    child's environment dict. A convenience `os.environ[...] = ...` on this
    path would redirect a Claude Code the *user* started, which is the one
    outcome the integration promised to avoid.
    """
    import os

    from env_mgr.o11y.agentsview import Status

    monkeypatch.setattr(
        cli_main,
        "ensure_running",
        lambda prefix, port: Status(True, "started", f"http://127.0.0.1:{port}"),
    )
    before = dict(os.environ)
    assert cli_main._start_o11y(port_flag=9009, disabled=False) == "http://127.0.0.1:9009"
    assert dict(os.environ) == before
