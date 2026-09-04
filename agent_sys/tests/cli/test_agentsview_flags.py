# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The two flags, and the promise that o11y cannot fail a run."""

from __future__ import annotations

import pytest

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


def _installed(reason: str):
    """Stand in for `ensure_installed`, reporting one of its two ok-reasons."""
    from env_mgr.o11y.agentsview import Status

    return lambda prefix, install_item: Status(True, reason)


def test_a_fresh_install_says_so_exactly_once(monkeypatch, caplog) -> None:
    """A 45 MB download nobody asked for must be visible when it happens."""
    from env_mgr.o11y.agentsview import Status

    monkeypatch.setattr(cli_main, "ensure_installed", _installed("installed agentsview"))
    monkeypatch.setattr(cli_main, "ensure_running", lambda prefix, port: Status(False, "x"))
    with caplog.at_level("INFO", logger="demo"):
        cli_main._start_o11y(port_flag=9009, disabled=False)
    notices = [r for r in caplog.records if r.levelname == "INFO" and "fetched" in r.message]
    assert len(notices) == 1


def test_an_install_that_was_already_satisfied_is_silent(monkeypatch, caplog) -> None:
    """The notice fires on the install, not on the 500 runs after it.

    A line printed every time is noise, and noise is how a real warning gets
    scrolled past.
    """
    from env_mgr.o11y.agentsview import Status

    monkeypatch.setattr(
        cli_main, "ensure_installed", _installed("agentsview already present (skip)")
    )
    monkeypatch.setattr(cli_main, "ensure_running", lambda prefix, port: Status(False, "x"))
    with caplog.at_level("INFO", logger="demo"):
        cli_main._start_o11y(port_flag=9009, disabled=False)
    assert [r for r in caplog.records if "fetched" in r.message] == []


def test_a_failed_install_does_not_go_on_to_start_a_daemon(monkeypatch) -> None:
    from env_mgr.o11y.agentsview import Status

    ran = []
    monkeypatch.setattr(cli_main, "ensure_installed", lambda prefix, install_item: Status(False, "no network"))
    monkeypatch.setattr(cli_main, "ensure_running", lambda *a, **k: ran.append(1))
    assert cli_main._start_o11y(port_flag=None, disabled=False) is None
    assert ran == []


def test_the_installers_two_ok_messages_still_discriminate() -> None:
    """A drift guard on the string `_start_o11y` reads.

    `ensure_installed` passes `Outcome.message` through verbatim, so the notice
    can only tell "installed just now" from "already there" by that text. If
    `BinInstaller.install` ever rephrases either branch the notice silently
    stops firing — or starts firing on every run — and nothing else would
    catch it. So both phrasings are taken from the real installer here.
    """
    from env_mgr.installers.bin import BinInstaller
    from env_mgr.recipe import Item, Target

    target = Target(kind="prefix", name="t", path=".")
    # A `check_cmd` that really answers a version, so `_satisfied` is true and
    # `install` takes its skip branch for real rather than being asserted about.
    satisfied = Item(
        "bin",
        "suggested",
        "system",
        spec={"name": "agentsview", "check_cmd": "echo agentsview 0.42.0"},
    )
    fresh = Item(
        "bin", "suggested", "system", spec={"name": "agentsview", "check_cmd": "", "install": ":"}
    )
    # **Both branches are driven for real; neither string is written here.**
    # An earlier version fell back to a literal when the skip branch produced
    # nothing (`... ] or [f"{name} already present (skip)"]`) and then asserted
    # only that the result was truthy — so the test invented the very string it
    # was meant to be guarding. Rephrasing `BinInstaller`'s skip message to
    # "installed {name} (cached)" — the exact change that makes the 45 MB
    # notice fire on *every* run — left it green. `satisfied` therefore gets a
    # `check_cmd` that really succeeds, which is what reaches the skip branch.
    (fresh_msg,) = [o.message for o in BinInstaller().install(fresh, target) if o.level == "ok"]
    (skip_msg,) = [
        o.message for o in BinInstaller().install(satisfied, target) if o.level == "ok"
    ]

    assert cli_main._was_freshly_installed(fresh_msg) is True
    assert cli_main._was_freshly_installed(skip_msg) is False


def test_the_install_closure_runs_nothing_until_it_is_called(monkeypatch, tmp_path) -> None:
    """Laziness is the property, not an implementation detail.

    `ensure_installed` takes a callable so the recipe does not run until it has
    decided the install is wanted. A precomputed `Outcome` list would have
    downloaded 45 MB before the `--dry-run` check could stop it, so "nothing
    happened at construction time" is worth asserting directly.
    """
    import env_mgr.runner
    from env_mgr.prefix import Prefix

    ran = []
    monkeypatch.setattr(
        env_mgr.runner, "run", lambda *a, **k: (ran.append((a, k)), ([], "ok"))[1]
    )
    prefix = Prefix.resolve({"HOME": str(tmp_path)})

    call = cli_main._install_item(prefix)
    assert ran == []  # constructing it must not have run the installer

    call()
    (args, _kw), = ran
    target, _items, stage, filters = args[0], args[1], args[2], args[3]
    assert stage == "install"
    assert filters.item == "agentsview"
    # The checked-in recipe's `target.path` is a placeholder; the caller is
    # what points it at this prefix.
    assert target.path == str(prefix.root)


def test_a_dry_run_installs_nothing_and_starts_nothing(monkeypatch) -> None:
    """`--dry-run` downloading 45 MB would be a worse breach than the daemon."""
    called = []
    monkeypatch.setattr(cli_main, "ensure_installed", lambda *a, **k: called.append("install"))
    monkeypatch.setattr(cli_main, "ensure_running", lambda *a, **k: called.append("run"))
    monkeypatch.setattr(cli_main, "_dry_run", lambda args, stream, panel_url=None: 0)
    assert cli_main.main(["run", "--package", "pkg", "--dry-run"]) == 0
    assert called == []


def test_a_dry_run_starts_no_daemon(monkeypatch) -> None:
    """`--dry-run` promises *resolve everything, do nothing*.

    Starting a resident daemon, creating `~/.infera_agent_sys` and writing a
    `config.toml` are all side effects, and a dry run that leaves a daemon
    behind has broken its only contract.
    """
    called = []
    monkeypatch.setattr(cli_main, "ensure_running", lambda *a, **k: called.append(1))
    monkeypatch.setattr(cli_main, "_dry_run", lambda args, stream, panel_url=None: 0)
    assert cli_main.main(["run", "--package", "pkg", "--dry-run"]) == 0
    assert called == []


def test_clean_starts_no_daemon(monkeypatch) -> None:
    """`--clean` removes every run and exits; a panel for it is pointless."""
    called = []
    monkeypatch.setattr(cli_main, "ensure_running", lambda *a, **k: called.append(1))
    monkeypatch.setattr(cli_main, "_clean", lambda args, stream, panel_url=None: 0)
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


# --------------------------------------------------------------------------- #
# The readiness probe writes a transcript too


def test_the_readiness_probe_writes_into_the_prefix_not_the_users_claude_dir(
    monkeypatch,
) -> None:
    """`preflight_credentials` spawns `claude -p`, so it produces a transcript.

    It is not an agent child, so gate 1 (`assignment.environment`) never
    covered it, and it was therefore writing one JSONL into the user's
    `~/.claude/projects` on **every** run. The promise was that `agent_sys`
    does not write there; the probe is a child like any other and gets the same
    `CLAUDE_CONFIG_DIR`.
    """
    import os
    import subprocess

    from cli import environment as cli_env
    from env_mgr.prefix import Prefix

    seen: dict[str, str] = {}

    def spy(cmd, **kw):
        seen.update(kw.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, "ready", "")

    monkeypatch.setattr(cli_env.shutil, "which", lambda c: "/usr/bin/claude")
    monkeypatch.setattr(cli_env.subprocess, "run", spy)
    before = dict(os.environ)

    assert cli_env.preflight_credentials(cli="claude") == "ready"

    prefix = Prefix.resolve(os.environ)
    assert seen["CLAUDE_CONFIG_DIR"] == str(prefix.claude_home)
    # Built on top of the ambient environment, not instead of it: the child
    # still needs PATH and HOME, and a bare `env={...}` would strip them.
    assert seen["PATH"] == os.environ["PATH"]
    assert seen["HOME"] == os.environ["HOME"]
    # And the same law as everywhere else in this feature.
    assert dict(os.environ) == before
    assert "CLAUDE_CONFIG_DIR" not in os.environ


# --------------------------------------------------------------------------- #
# The wire from `main()` to the panel
#
# Every test above this line drives `_start_o11y` directly, and every test
# below `parser()` drives argparse directly. Neither touches the line that
# joins them -- measured: deleting the `_start_o11y(...)` call from `main()`
# outright left the whole `tests/cli` suite green (179 passed). Acceptance
# criterion 2, "deploying agent_sys starts it automatically", had no test.


def _main_with_o11y_spied(monkeypatch, argv: list[str]) -> dict:
    """Run `main()` for real, with the run itself and the daemon stubbed."""
    from env_mgr.o11y.agentsview import Status

    seen: dict = {}

    def spy_running(prefix, port):
        seen["port"] = port
        return Status(True, "started", f"http://127.0.0.1:{port}")

    monkeypatch.setattr(cli_main, "ensure_installed", _installed("agentsview already present"))
    monkeypatch.setattr(cli_main, "ensure_running", spy_running)
    monkeypatch.setattr(cli_main, "_run", lambda args, stream, panel_url=None: 0)
    seen["exit"] = cli_main.main(argv)
    return seen


def test_a_plain_run_starts_the_panel_on_the_default_port(monkeypatch) -> None:
    seen = _main_with_o11y_spied(monkeypatch, ["run", "--package", "pkg"])
    assert seen["port"] == 18888
    assert seen["exit"] == 0


def test_the_port_flag_reaches_the_daemon(monkeypatch) -> None:
    """`--agentsview-port` is parsed *and* used. Passing `None` here instead of
    `args.agentsview_port` left every other test green."""
    seen = _main_with_o11y_spied(
        monkeypatch, ["run", "--package", "pkg", "--agentsview-port", "9001"]
    )
    assert seen["port"] == 9001


def test_no_agentsview_reaches_the_call_site(monkeypatch) -> None:
    seen = _main_with_o11y_spied(monkeypatch, ["run", "--package", "pkg", "--no-agentsview"])
    assert "port" not in seen


def test_a_dry_run_still_starts_nothing_through_main(monkeypatch) -> None:
    seen = _main_with_o11y_spied(monkeypatch, ["run", "--package", "pkg", "--dry-run"])
    assert "port" not in seen


def test_show_never_reaches_the_panel(monkeypatch) -> None:
    """`show` returns before the call site, so it has no flags to consult."""
    from env_mgr.o11y.agentsview import Status

    ran = []
    monkeypatch.setattr(cli_main, "ensure_running", lambda *a, **k: ran.append(1) or Status(False, "x"))
    monkeypatch.setattr(cli_main, "_show", lambda args, stream, panel_url=None: 0)
    assert cli_main.main(["show", "--package", "pkg"]) == 0
    assert ran == []


# --------------------------------------------------------------------------- #
# The URL has to arrive somewhere the user can see


def test_the_panel_url_reaches_the_user(monkeypatch, capsys) -> None:
    """Read the artefact, not the exit code.

    Both this and the fresh-install notice were `log.info`, and nothing in this
    repository ever configures `logging` -- so the root logger sat at WARNING
    with no handler and both lines were discarded. The failure warnings reached
    stderr through `logging.lastResort`; the successes reached nobody. The
    tests passed only because `caplog.at_level("INFO")` forced the level from
    pytest's side, which is precisely the shape of a test that asserts the
    program's intent rather than its output.
    """
    _main_with_o11y_spied(monkeypatch, ["run", "--package", "pkg", "--agentsview-port", "9001"])
    assert "http://127.0.0.1:9001" in capsys.readouterr().out


def test_the_fresh_install_notice_reaches_the_user(monkeypatch, capsys) -> None:
    from env_mgr.o11y.agentsview import Status

    monkeypatch.setattr(cli_main, "ensure_installed", _installed("installed agentsview"))
    monkeypatch.setattr(cli_main, "ensure_running", lambda prefix, port: Status(False, "x"))
    monkeypatch.setattr(cli_main, "_run", lambda args, stream, panel_url=None: 0)
    cli_main.main(["run", "--package", "pkg"])
    out = capsys.readouterr().out
    assert "agentsview" in out and "kenn-io/agentsview" in out


def test_a_skipped_panel_says_nothing_to_the_user(monkeypatch, capsys) -> None:
    """A warning already went to the log; the event stream is for what *is*."""
    from env_mgr.o11y.agentsview import Status

    monkeypatch.setattr(cli_main, "ensure_installed", _installed("agentsview already present"))
    monkeypatch.setattr(cli_main, "ensure_running", lambda prefix, port: Status(False, "port in use"))
    monkeypatch.setattr(cli_main, "_run", lambda args, stream, panel_url=None: 0)
    cli_main.main(["run", "--package", "pkg"])
    assert "127.0.0.1" not in capsys.readouterr().out


def test_the_readiness_probe_runs_from_a_directory_inside_the_prefix(monkeypatch) -> None:
    """Where the probe runs decides which project its transcript lands in.

    AgentsView names a project after the session's cwd — resolving the git
    *main repository* when there is one. The probe inherited the caller's cwd,
    which is the checkout, so ten identical `Reply with exactly one word:
    ready` sessions piled into the real `infera` project. A directory of its
    own inside the prefix is one argument and no new state.
    """
    import os
    import subprocess

    from cli import environment as cli_env
    from env_mgr.prefix import Prefix

    seen: dict = {}

    def spy(cmd, **kw):
        seen.update(kw)
        return subprocess.CompletedProcess(cmd, 0, "ready", "")

    monkeypatch.setattr(cli_env.shutil, "which", lambda c: "/usr/bin/claude")
    monkeypatch.setattr(cli_env.subprocess, "run", spy)

    assert cli_env.preflight_credentials(cli="claude") == "ready"

    prefix = Prefix.resolve(os.environ)
    assert seen["cwd"] == str(cli_env.probe_cwd(prefix))
    assert str(prefix.root) in seen["cwd"]
    assert os.path.isdir(seen["cwd"]), "the child refuses a cwd that does not exist"


def test_the_probe_still_runs_when_its_directory_cannot_be_made(monkeypatch) -> None:
    """A cwd we could not create is not a reason to refuse the whole run.

    `preflight_credentials` failing aborts everything, which is far worse than
    a transcript filed under the wrong project.
    """
    import subprocess
    from pathlib import Path

    from cli import environment as cli_env

    def boom(*a, **k):
        raise PermissionError("read-only prefix")

    def spy(cmd, **kw):
        assert kw.get("cwd") is None, "no cwd beats a cwd that does not exist"
        return subprocess.CompletedProcess(cmd, 0, "ready", "")

    monkeypatch.setattr(cli_env.shutil, "which", lambda c: "/usr/bin/claude")
    monkeypatch.setattr(Path, "mkdir", boom)
    monkeypatch.setattr(cli_env.subprocess, "run", spy)

    assert cli_env.preflight_credentials(cli="claude") == "ready"


# --------------------------------------------------------------------------- #
# The panel URL has to survive the trip from `_start_o11y` to the mapping call


def test_the_panel_url_reaches_the_run(monkeypatch) -> None:
    """`main` discarded `_start_o11y`'s return value until this feature needed it.

    The mapping call lives in `_real_run` because the run id does not exist when
    the panel starts, so the URL has to be threaded through two frames. Both
    hops are one keyword each and neither is covered by anything else here.
    """
    from env_mgr.o11y.agentsview import Status

    seen: dict = {}

    monkeypatch.setattr(cli_main, "ensure_installed", _installed("agentsview already present"))
    monkeypatch.setattr(
        cli_main, "ensure_running",
        lambda prefix, port: Status(True, "started", f"http://127.0.0.1:{port}"),
    )
    monkeypatch.setattr(
        cli_main, "_real_run",
        lambda args, stream, panel_url=None: seen.setdefault("url", panel_url) and 0 or 0,
    )

    cli_main.main(["run", "--package", "pkg", "--agentsview-port", "9001"])

    assert seen["url"] == "http://127.0.0.1:9001"


def test_a_run_without_a_panel_passes_none_rather_than_failing(monkeypatch) -> None:
    """o11y absent is a `None`, not an exception and not a missing argument."""
    from env_mgr.o11y.agentsview import Status

    seen: dict = {}
    monkeypatch.setattr(cli_main, "ensure_installed", _installed("agentsview already present"))
    monkeypatch.setattr(cli_main, "ensure_running", lambda prefix, port: Status(False, "port in use"))
    monkeypatch.setattr(
        cli_main, "_real_run",
        lambda args, stream, panel_url="MISSING": seen.setdefault("url", panel_url) or 0,
    )

    assert cli_main.main(["run", "--package", "pkg"]) == 0
    assert seen["url"] is None
