"""Process-level ownership of agent executors."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cli import main as cli_main


def _run_harness(monkeypatch: pytest.MonkeyPatch, tmp_path, *, settle_error=None):
    runner = SimpleNamespace(shutdown=Mock())
    task_mgr = SimpleNamespace(all=lambda: ())
    registry = SimpleNamespace(
        get=lambda name: {"runner": runner, "task_mgr": task_mgr}[name]
    )
    monitors = SimpleNamespace(stop=Mock(return_value=[]))
    promises = object()

    monkeypatch.setattr(cli_main, "confinement", lambda: "landlock")
    monkeypatch.setattr(cli_main, "preflight_credentials", lambda: "ready")
    monkeypatch.setattr(cli_main, "preflight_repository", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(cli_main.package, "locate", lambda path: tmp_path)
    monkeypatch.setattr(cli_main.expectations, "for_package", lambda path: promises)
    monkeypatch.setattr(cli_main, "_layout", lambda args: SimpleNamespace(run=tmp_path))
    monkeypatch.setattr(cli_main, "permissions_enforced", lambda: True)
    monkeypatch.setattr(cli_main, "report_dropped", lambda *a: None)
    monkeypatch.setattr(cli_main, "_registry", lambda *a, **k: registry)
    monkeypatch.setattr(cli_main, "start_monitors", lambda r: monitors)
    monkeypatch.setattr(cli_main, "_start", lambda *a: None)

    def settle(*args, **kwargs):
        if settle_error is not None:
            raise settle_error

    monkeypatch.setattr(cli_main, "_settle", settle)
    monkeypatch.setattr(cli_main, "_emit_graph", lambda *a, **k: None)
    monkeypatch.setattr(cli_main, "_describe", lambda *a, **k: None)
    monkeypatch.setattr(cli_main, "_report", lambda *a: 0)

    args = SimpleNamespace(
        package=str(tmp_path),
        allow_repo_config=False,
        resume=False,
        variables={},
    )
    return args, Mock(), runner, monitors


def test_real_run_shuts_down_executors_after_reporting(monkeypatch, tmp_path) -> None:
    args, stream, runner, monitors = _run_harness(monkeypatch, tmp_path)

    assert cli_main._real_run(args, stream) == 0
    monitors.stop.assert_called_once_with(timeout=5.0)
    runner.shutdown.assert_called_once_with()


def test_real_run_shuts_down_executors_when_settle_raises(monkeypatch, tmp_path) -> None:
    failure = RuntimeError("settle failed")
    args, stream, runner, monitors = _run_harness(
        monkeypatch, tmp_path, settle_error=failure
    )

    with pytest.raises(RuntimeError, match="settle failed"):
        cli_main._real_run(args, stream)
    monitors.stop.assert_called_once_with(timeout=5.0)
    runner.shutdown.assert_called_once_with()
