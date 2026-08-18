# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
import pytest

from env_mgr.installers.base import (
    level_for_missing,
    probe_version,
    run_cmd,
)
from env_mgr.recipe import Item, Target
from env_mgr.registry import REGISTRY, get_installer


def test_run_cmd_success():
    rc, out = run_cmd("echo hello")
    assert rc == 0
    assert "hello" in out


def test_run_cmd_nonzero_does_not_raise():
    rc, _ = run_cmd("exit 3")
    assert rc == 3


def test_probe_version_extracts_token():
    assert probe_version("echo 'tool 1.2.3 (linux)'") == "1.2.3"


def test_probe_version_none_on_failure():
    assert probe_version("false") is None


def test_level_for_missing():
    assert level_for_missing("required") == "fail"
    assert level_for_missing("strongly-suggested") == "warn"
    assert level_for_missing("suggested") == "info"


def _target(tmp_path):
    return Target(kind="repo", name="x", path=str(tmp_path))


def test_registry_has_all_installers():
    for name in ("uv", "apt", "bin", "oneline", "embed", "claude"):
        assert name in REGISTRY
        assert get_installer(name).name == name


def test_get_installer_unknown_raises():
    with pytest.raises(KeyError):
        get_installer("nope")


def test_bin_check_missing_is_fail_for_required(tmp_path):
    item = Item(
        installer="bin",
        importance="required",
        layer="system",
        spec={
            "name": "definitely-not-a-real-bin-xyz",
            "check_cmd": "definitely-not-a-real-bin-xyz --version",
        },
    )
    outs = get_installer("bin").check(item, _target(tmp_path))
    assert any(o.level == "fail" for o in outs)


def test_oneline_check_uses_check_cmd(tmp_path):
    item = Item(
        installer="oneline",
        importance="suggested",
        layer="system",
        spec={"check_cmd": "true", "run": "echo would-install"},
    )
    outs = get_installer("oneline").check(item, _target(tmp_path))
    assert any(o.level == "ok" for o in outs)


def test_apt_plan_prints_command_never_sudos(tmp_path):
    item = Item(
        installer="apt",
        importance="suggested",
        layer="system",
        spec={"packages": ["definitely-not-installed-pkg-xyz"]},
    )
    outs = get_installer("apt").plan(item, _target(tmp_path))
    joined = " ".join(o.message + " " + str(o.details) for o in outs)
    assert "apt-get install" in joined


def test_embed_plan_shows_body_without_running(tmp_path):
    marker = tmp_path / "side_effect"
    item = Item(
        installer="embed",
        importance="required",
        layer="repo",
        spec={"run": f"touch {marker}"},
    )
    outs = get_installer("embed").plan(item, _target(tmp_path))
    assert not marker.exists()  # plan must not run it
    assert outs  # produced at least one Outcome


def test_uv_plan_is_dry_run_and_nonmutating(tmp_path):
    # a ref-form uv item; plan must preview via uv --dry-run without installing
    item = Item(
        installer="uv",
        importance="required",
        layer="repo",
        spec={"name": "uv", "ref": "pyproject.toml"},
    )
    outs = get_installer("uv").plan(item, Target(kind="repo", name="x", path=str(tmp_path)))
    assert outs and all(o.level == "info" for o in outs)
    joined = " ".join(o.message + " " + str(o.details) for o in outs)
    assert "--dry-run" in joined


def test_uv_plan_tool_form_no_dry_run(tmp_path):
    # uv tool install has no --dry-run flag (it errors on it); plan for the
    # tool form must be a static preview, not a command it actually runs.
    item = Item(
        installer="uv",
        importance="required",
        layer="system",
        spec={"tool": "git+https://example.invalid/pkg", "provides": "pkg"},
    )
    outs = get_installer("uv").plan(item, Target(kind="repo", name="x", path=str(tmp_path)))
    assert outs and all(o.level == "info" for o in outs)
    joined = " ".join(o.message + " " + str(o.details) for o in outs)
    assert "--dry-run" not in joined
    assert "uv tool install" in joined


def test_uv_check_warns_when_no_ref_or_tool(tmp_path):
    # a uv item with neither ref nor tool is misconfigured; check must NOT say ok
    item = Item(installer="uv", importance="required", layer="repo", spec={})
    outs = get_installer("uv").check(item, _target(tmp_path))
    assert outs and all(o.level == "warn" for o in outs)


def test_uv_check_ok_with_valid_ref(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    item = Item(installer="uv", importance="required", layer="repo", spec={"ref": "pyproject.toml"})
    outs = get_installer("uv").check(item, _target(tmp_path))
    assert any(o.level == "ok" for o in outs)


def test_claude_present_names_exact_per_line():
    from env_mgr.installers.claude import ClaudeInstaller

    out = "superpowers 1.0\ncode-review 2.1\n"
    names = ClaudeInstaller._present_names(out)
    assert names == {"superpowers", "code-review"}


def test_claude_present_names_no_substring_false_positive():
    from env_mgr.installers.claude import ClaudeInstaller

    # "super" must NOT be considered present just because "superpowers" is
    names = ClaudeInstaller._present_names("superpowers 1.0\n")
    assert "super" not in names


def test_oneline_plan_message_style(tmp_path):
    # oneline previews as a single-line "would run:" message, no "script" wording
    item = Item(
        installer="oneline", importance="suggested", layer="system", spec={"run": "echo hi"}
    )
    outs = get_installer("oneline").plan(item, _target(tmp_path))
    msg = " ".join(o.message for o in outs)
    assert "would run:" in msg
    assert "script" not in msg


def test_embed_plan_message_style(tmp_path):
    # embed previews a multi-line script body, flagged with "script" wording
    item = Item(
        installer="embed", importance="required", layer="repo", spec={"run": "echo hi\necho bye"}
    )
    outs = get_installer("embed").plan(item, _target(tmp_path))
    msg = " ".join(o.message for o in outs)
    assert "script" in msg
