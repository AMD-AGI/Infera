# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Criterion 22 — the shipped machinery is untouched, and the CLI's new
sub-commands do not disturb it.

`cli.py` is the only shipped file this work changes, and 65 tests are pointed at
it. The shipped suite is the assertion that it still behaves; this is the
assertion that the *parse* still behaves, shape by shape, including the one
observable difference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from env_mgr.cli import _parse, main
from env_mgr.runner import STAGES


def test_cli_subcommands_preserve_shipped_shapes(tmp_path: Path) -> None:
    """All six shipped call shapes parse identically and set the same
    attributes."""
    recipe = str(tmp_path / "r.yaml")
    shapes = [
        ["check", recipe],
        ["dry-run", recipe, "--json"],
        ["install", recipe, "--tag", "lsp"],
        ["bootstrap", recipe, "--path", "/p"],
        ["check", recipe, "--workspace", "/ws"],
        ["check", recipe, "--importance", "required"],
    ]
    for shape in shapes:
        args = _parse(shape)
        assert args.stage == shape[0]
        assert args.recipe == recipe
        assert isinstance(args.json, bool)
        assert hasattr(args, "tags")
        assert hasattr(args, "on_conflict")


def test_every_shipped_stage_is_still_a_stage(tmp_path: Path) -> None:
    for stage in STAGES:
        assert _parse([stage, str(tmp_path / "r.yaml")]).stage == stage


def test_an_invalid_stage_still_exits_two() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _parse(["bogus", "r.yaml"])
    assert excinfo.value.code == 2


def test_a_global_flag_before_the_subcommand_no_longer_parses() -> None:
    """**The one observable difference**, recorded rather than discovered.

    No shipped test and no documented invocation places a flag before the
    sub-command, and the design measured that before choosing sub-parsers.
    """
    with pytest.raises(SystemExit):
        _parse(["--json", "check", "r.yaml"])


def test_domain_subcommand(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from env_mgr import meta

    meta_path = tmp_path / "meta.json"
    meta.save(
        meta.Meta(domains=(("store", str(tmp_path / "root"), "handoff_storage"),)), str(meta_path)
    )
    assert main(["domain", "--meta", str(meta_path), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows == [{"name": "store", "kind": "handoff_storage", "root": str(tmp_path / "root")}]


def test_zone_subcommand(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from env_mgr import meta
    from env_mgr.fs import layout

    from .stubs import Task

    root = tmp_path / "root"
    meta_path = tmp_path / "meta.json"
    declared = meta.Meta(domains=(("store", str(root), "handoff_storage"),))
    meta.save(declared, str(meta_path))
    task = Task()
    layout.create(task, task.push_execution(), declared.registry())

    assert main(["zone", "--meta", str(meta_path), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["task"] for r in rows] == [str(task.id)]
    assert rows[0]["attempt"] == "0"


def test_zone_subcommand_filters_by_task(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from env_mgr import meta
    from env_mgr.fs import layout

    from .stubs import Task

    meta_path = tmp_path / "meta.json"
    declared = meta.Meta(domains=(("store", str(tmp_path / "root"), "handoff_storage"),))
    meta.save(declared, str(meta_path))
    reg = declared.registry()
    wanted, other = Task(), Task()
    layout.create(wanted, wanted.push_execution(), reg)
    layout.create(other, other.push_execution(), reg)

    main(["zone", str(wanted.id), "--meta", str(meta_path), "--json"])
    rows = json.loads(capsys.readouterr().out)
    assert [r["task"] for r in rows] == [str(wanted.id)]


def test_the_shipped_modules_are_byte_identical() -> None:
    """Criterion 22's first clause, asserted against the git index rather than
    against a memory of what was changed."""
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    shipped = [
        "env_mgr/recipe.py",
        "env_mgr/layer.py",
        "env_mgr/runner.py",
        "env_mgr/outcome.py",
        "env_mgr/report.py",
        "env_mgr/registry.py",
        "env_mgr/versions.py",
        "env_mgr/installers",
    ]
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--stat", "--", *shipped],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if diff.returncode != 0:  # pragma: no cover - not a git checkout
        pytest.skip("not a git working tree")
    assert diff.stdout.strip() == "", f"the shipped machinery changed:\n{diff.stdout}"
