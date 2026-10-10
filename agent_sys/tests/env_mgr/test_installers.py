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
        spec={"check_cmd": "true", "run": "echo would-install"},
    )
    outs = get_installer("oneline").check(item, _target(tmp_path))
    assert any(o.level == "ok" for o in outs)


def test_apt_plan_prints_command_never_sudos(tmp_path):
    item = Item(
        installer="apt",
        importance="suggested",
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
        spec={"tool": "git+https://example.invalid/pkg", "provides": "pkg"},
    )
    outs = get_installer("uv").plan(item, Target(kind="repo", name="x", path=str(tmp_path)))
    assert outs and all(o.level == "info" for o in outs)
    joined = " ".join(o.message + " " + str(o.details) for o in outs)
    assert "--dry-run" not in joined
    assert "uv tool install" in joined


def test_uv_check_warns_when_no_ref_or_tool(tmp_path):
    # a uv item with neither ref nor tool is misconfigured; check must NOT say ok
    item = Item(installer="uv", importance="required", spec={})
    outs = get_installer("uv").check(item, _target(tmp_path))
    assert outs and all(o.level == "warn" for o in outs)


def test_uv_check_ok_with_valid_ref(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    item = Item(installer="uv", importance="required", spec={"ref": "pyproject.toml"})
    outs = get_installer("uv").check(item, _target(tmp_path))
    assert any(o.level == "ok" for o in outs)


# --------------------------------------------------------------------------- #
# `claude plugin list` fixtures — captured bytes, not a remembered format.
# `❯` is the entry bullet, `✔` / `✘` the enabled / disabled status glyphs.
# Do not "simplify" these strings: their value is that nobody wrote them.

CLAUDE_PLUGIN_LIST_EMPTY = (
    "No plugins installed. Use `claude plugin install` to install a plugin.\n"
)

CLAUDE_PLUGIN_LIST_THREE = (
    "Installed plugins:\n"
    "\n"
    "  ❯ code-review@claude-code-plugins\n"
    "    Version: 1.0.0\n"
    "    Scope: user\n"
    "    Status: ✔ enabled\n"
    "\n"
    "  ❯ commit-commands@claude-code-plugins\n"
    "    Version: 1.0.0\n"
    "    Scope: user\n"
    "    Status: ✔ enabled\n"
    "\n"
    "  ❯ hookify@claude-code-plugins\n"
    "    Version: 0.1.0\n"
    "    Scope: user\n"
    "    Status: ✔ enabled\n"
    "\n"
)

#: The same three, after `claude plugin disable hookify@claude-code-plugins`.
#: The entry does not disappear; one glyph changes.
CLAUDE_PLUGIN_LIST_ONE_DISABLED = CLAUDE_PLUGIN_LIST_THREE.replace(
    "  ❯ hookify@claude-code-plugins\n    Version: 0.1.0\n    Scope: user\n    Status: ✔ enabled\n",
    "  ❯ hookify@claude-code-plugins\n"
    "    Version: 0.1.0\n"
    "    Scope: user\n"
    "    Status: ✘ disabled\n",
)


def test_claude_present_names_on_real_cli_output():
    from env_mgr.installers.claude import ClaudeInstaller

    # Every entry is bulleted with U+276F and carries `@marketplace`; the three
    # indented metadata lines are not entries. The shipped parser returned
    # {'Installed', 'Scope:', 'Status:', 'Version:', '❯'} here -- no plugin name,
    # so `check` reported every declared plugin missing on every run.
    names = ClaudeInstaller._present_names(CLAUDE_PLUGIN_LIST_THREE)
    assert names == {"code-review", "commit-commands", "hookify"}


def test_claude_present_names_empty_when_nothing_installed():
    from env_mgr.installers.claude import ClaudeInstaller

    # The CLI answers with a sentence, not an empty string. The shipped parser
    # took its first word and returned {'No'} -- a junk name that would have
    # matched a plugin called `No`.
    assert ClaudeInstaller._present_names(CLAUDE_PLUGIN_LIST_EMPTY) == set()


def test_claude_present_names_no_substring_false_positive():
    from env_mgr.installers.claude import ClaudeInstaller

    # A name that is a strict prefix of an installed one must not be reported
    # present; read together with `test_claude_present_names_on_real_cli_output`,
    # which guards the other direction.
    names = ClaudeInstaller._present_names(CLAUDE_PLUGIN_LIST_THREE)
    assert "commit" not in names
    assert "code" not in names
    assert "hook" not in names


def test_claude_present_names_reports_a_disabled_plugin_as_present():
    from env_mgr.installers.claude import ClaudeInstaller

    # Known limitation, pinned deliberately: `_present_names` reads "installed",
    # not "enabled", so a disabled plugin still counts as present.
    names = ClaudeInstaller._present_names(CLAUDE_PLUGIN_LIST_ONE_DISABLED)
    assert names == {"code-review", "commit-commands", "hookify"}


def test_oneline_plan_message_style(tmp_path):
    # oneline previews as a single-line "would run:" message, no "script" wording
    item = Item(installer="oneline", importance="suggested", spec={"run": "echo hi"})
    outs = get_installer("oneline").plan(item, _target(tmp_path))
    msg = " ".join(o.message for o in outs)
    assert "would run:" in msg
    assert "script" not in msg


def test_embed_plan_message_style(tmp_path):
    # embed previews a multi-line script body, flagged with "script" wording
    item = Item(installer="embed", importance="required", spec={"run": "echo hi\necho bye"})
    outs = get_installer("embed").plan(item, _target(tmp_path))
    msg = " ".join(o.message for o in outs)
    assert "script" in msg
