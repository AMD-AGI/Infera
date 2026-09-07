# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`env_mgr.agent_assets` — installs an agent's declared material (assets,
recipes, MCP servers, plugins) into a zone's config directory.

A fake `claude` CLI goes on `PATH` per test; nothing here runs the real binary.
That lets both a successful install and a `claude plugin install` failure be
tested, which is not observable through a stub that only returns 0.

Key properties covered:
- `claude plugin marketplace add` / `install` respect `CLAUDE_CONFIG_DIR`;
- they **merge** into an existing `settings.json` rather than clobbering it,
  which is why the settings document is written before any install runs;
- a local marketplace needs `.claude-plugin/marketplace.json` carrying
  `{name, owner, plugins: [...]}`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from env_mgr import agent_assets, material
from env_mgr.agent_assets import AgentMaterial, install
from env_mgr.paths import AGENT_ASSETS_ENV_VAR
from env_mgr.protocols import PrepareRefused

#: `.mcp.json`, spelled once. Two literals of a filename in one file is how the
#: second one gets missed by a rename.
MCP_REL = ".mcp.json"


# --------------------------------------------------------------------------- #
# Fixtures: a package with an agent asset directory, and a fake `claude`


class _ZoneAt:
    """The one attribute `material.deploy` reads off a zone. `test_isolation_shown`'s
    double, restated here rather than imported: a test package importing another
    test package's private helper is an edge nobody meant to create."""

    def __init__(self, root: str) -> None:
        self.root = root


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class FakeCli(NamedTuple):
    """A stand-in `claude` binary and the file it records its argv into: the
    path for a pinned absolute-argument call, the log to assert what ran.
    """

    path: str
    log: Path


#: The real `DEFAULT_RECIPE`, captured at import time — **before** the autouse
#: fixture below can replace it. Without this the one test that checks the
#: shipped file would inspect the fixture's fake path and could never see the
#: real one, which is a check that cannot fail in the most literal way: it would
#: be asserting about a string this file wrote.
_SHIPPED_DEFAULT_RECIPE = agent_assets.DEFAULT_RECIPE


@pytest.fixture(autouse=True)
def _no_default_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point `DEFAULT_RECIPE` at nothing for every test here, so the shipped
    default layer adds no extra report entries. Overridden explicitly where
    the default layer is the subject.
    """
    monkeypatch.setattr(agent_assets, "DEFAULT_RECIPE", "/nonexistent/default.env_recipe.yaml")


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeCli:
    """A `claude` stub that records its argv and succeeds, placed both on
    `PATH` and returned by absolute path — so a pinned-path assertion is
    only meaningful if a bare name would also have resolved.
    """
    binroot = tmp_path / "fakebin"
    binroot.mkdir()
    log = tmp_path / "claude.log"
    script = binroot / "claude"
    script.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> {log}\n'
        f'printf "CLAUDE_CONFIG_DIR=%s\\n" "$CLAUDE_CONFIG_DIR" >> {log}\n'
        # Records whether settings.json existed at invocation time. Reading
        # the file after `install` returns cannot distinguish written-before
        # from written-after, and `claude` merges into an existing file
        # rather than replacing it.
        f'printf "settings_existed=%s\\n" '
        f'"$(test -f "$CLAUDE_CONFIG_DIR/settings.json" && echo yes || echo no)" >> {log}\n'
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{binroot}{os.pathsep}{os.environ['PATH']}")
    return FakeCli(str(script), log)


@pytest.fixture
def failing_claude(tmp_path: Path) -> str:
    """A `claude` that exists and exits 1. The case a missing binary cannot cover."""
    binroot = tmp_path / "failbin"
    binroot.mkdir()
    script = binroot / "claude"
    script.write_text("#!/bin/sh\necho 'boom' >&2\nexit 1\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(script)


def _package(root: Path, *, agent: str = "forge") -> Path:
    """A staged package with `assets/<agent>.agent/`, and the directory's path."""
    assets = root / "assets" / f"{agent}.agent"
    assets.mkdir(parents=True)
    return assets


def _spec(**keys: Any) -> dict[str, Any]:
    """An agent spec as a mapping. `agent_assets._get` takes either, and a dict
    keeps a test from depending on `AgentSpec`'s validation for a field it is
    not testing."""
    return {"name": "forge", "kind": "ai", **keys}


def _levels(report: tuple[Any, ...]) -> list[str]:
    return [o.level for o in report]


def _installs(report: tuple[Any, ...]) -> list[str]:
    """Levels, excluding `_place_tree`'s per-member "placed" entries, so a
    count doesn't shift whenever a fixture gains a directory. Filtered by
    message, not level, so a `fail` is never excluded.
    """
    return [o.level for o in report if not o.message.startswith("placed ")]


def _placed(report: tuple[Any, ...]) -> list[str]:
    return [o.message for o in report if o.message.startswith("placed ")]


# --------------------------------------------------------------------------- #
# What a task package carries for one agent — undeclared, auto-detected


def test_a_packages_own_material_is_found_at_the_agent_assets_dot_claude_undeclared(
    tmp_path: Path,
) -> None:
    """An agent's own `.claude/` assets are found and installed with no
    declaration beyond `assets:` pointing at the directory — detection is
    the interface, since restating the origin in YAML would be a second
    writer of one fact.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(assets / ".claude" / "skills" / "packup" / "SKILL.md", "# packup\n")
    config = tmp_path / "zone" / "config"

    got = install(
        _spec(assets="assets/forge.agent"), staged_package=str(pkg), config_dir=str(config)
    )

    assert (config / "skills" / "packup" / "SKILL.md").read_text() == "# packup\n"
    assert got.env[AGENT_ASSETS_ENV_VAR] == str(assets)
    assert _levels(got.report) == ["ok"]


def test_an_agent_that_carries_nothing_installs_nothing_and_does_not_complain(
    tmp_path: Path,
) -> None:
    """Assets declared but the directory empty installs nothing and reports
    nothing — the counterpart to the refusal tests: an agent must not be
    forced to declare an empty directory.
    """
    pkg = tmp_path / "staged"
    _package(pkg)
    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "zone" / "config"),
    )
    assert got == AgentMaterial(
        env={AGENT_ASSETS_ENV_VAR: str(pkg / "assets" / "forge.agent")},
        mcp_servers={},
        settings={},
        report=(),
    )


def test_a_spec_declaring_assets_that_are_not_in_the_staged_copy_refuses(
    tmp_path: Path,
) -> None:
    """`spec_loader` fills `assets` from a directory it found, so a value that no
    longer resolves means the staged copy is not the package the spec was loaded
    from. That is worth stopping for, not shrugging at."""
    pkg = tmp_path / "staged"
    pkg.mkdir()
    with pytest.raises(PrepareRefused, match="declares assets"):
        install(
            _spec(assets="assets/forge.agent"),
            staged_package=str(pkg),
            config_dir=str(tmp_path / "config"),
        )


# --------------------------------------------------------------------------- #
# There is exactly one copy route, and this is the assertion that it is one


def test_the_agents_own_assets_are_the_only_tree_that_is_copied(tmp_path: Path) -> None:
    """The only tree ever copied into the zone is the package's own asset
    directory. Checked through `_addon_trees`'s return value, not the
    absence of a declaration key, so a second route would still be caught.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(assets / ".claude" / "skills" / "chk" / "SKILL.md", "# chk\n")
    config = tmp_path / "config"

    got = install(
        # A spec still carrying the deleted key must not resurrect the route.
        # `_get` reads whatever mapping it is handed, so this is the strongest
        # form available: the key is present in the input and reaches nothing.
        _spec(assets="assets/forge.agent", agent_plugins=["envchk"]),
        staged_package=str(pkg),
        config_dir=str(config),
    )

    origins = [
        origin
        for origin, _ in agent_assets._addon_trees(
            _spec(assets="assets/forge.agent", agent_plugins=["envchk"]),
            staged_package=str(pkg),
        )
    ]
    assert origins == ["the package's own assets"], origins
    # Positive control: the one route that remains really did place a file.
    assert (config / "skills" / "chk" / "SKILL.md").exists()
    assert _levels(got.report) == ["ok"]


def test_nothing_reaches_the_addons_directory_by_declaration(tmp_path: Path) -> None:
    """No declaration key reaches the add-ons directory: `AgentSpec`, the
    JSON schema, and the isolation policy grant all agree `agent_plugins`
    does not exist. Each negative is paired with a positive control.
    """
    import json as _json

    from agent.spec import AgentSpec
    from env_mgr.isolation import policy

    assert "agent_plugins" not in AgentSpec.model_fields
    assert "recipes" in AgentSpec.model_fields

    schema = _json.loads(
        (Path(__file__).parents[2] / "spec_loader" / "schemas" / "agent.schema.json").read_text()
    )
    properties = schema["properties"]
    assert "agent_plugins" not in properties
    assert "recipes" in properties

    assert not hasattr(policy, "addon_grants")
    assert hasattr(policy, "agent_cli_grants")


# --------------------------------------------------------------------------- #
# Recipes


def test_a_package_relative_recipe_runs_and_reports_its_status(tmp_path: Path) -> None:
    """A recipe resolved against the staged package actually runs its item (a
    shell `oneline` writing a marker file), not merely produces a
    manufactured `Outcome`.
    """
    pkg = tmp_path / "staged"
    marker = tmp_path / "ran.txt"
    _write(
        pkg / "recipes" / "tools.yaml",
        "version: 1\n"
        f"target: {{kind: repo, name: t, path: {tmp_path}}}\n"
        "items:\n"
        "  - installer: oneline\n"
        "    importance: suggested\n"
        "    name: marker\n"
        f"    run: touch {marker}\n",
    )
    got = install(
        _spec(recipes=["package:recipes/tools.yaml"]),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
    )
    assert marker.exists()
    assert any("tools.yaml" in o.message for o in got.report)


@pytest.mark.parametrize(
    ("declared", "expect"),
    [
        # **Declared and absent, per root**, and the two are separate cases
        # because before 2026-09-04 neither of them existed: a missing
        # `package:` file silently became a bare-name lookup in the shipped
        # directory, and only *both* missing was an error.
        ("package:absent.yaml", "does not exist"),
        ("agent_sys:absent", "does not exist"),
        # The migration guard. Dated, and its own message, because "names no
        # root" and "is not a recipe root" send a reader to different questions.
        ("serena", "names no root"),
        ("recipes/tools.yaml", "names no root"),
        # An unknown scheme **names both spellings**, so the message teaches the
        # rule instead of citing a document.
        ("envmgr:serena", "is not a recipe root"),
        ("package :r.yaml", "is not a recipe root"),
        # `agent_sys:` is a name and a separator is refused rather than
        # basenamed — the previous code turned `a/b` into `b` in silence.
        ("agent_sys:sub/serena", "never a path"),
        ("agent_sys:", "never a path"),
        # `package:` may not climb out of the staged copy. A recipe is executed.
        ("package:../../escaped.yaml", "stay inside the staged package"),
        ("package:/etc/passwd.yaml", "stay inside the staged package"),
    ],
)
def test_every_way_a_recipe_reference_can_be_refused(
    tmp_path: Path, declared: str, expect: str
) -> None:
    """Each way a recipe reference can be malformed is refused with its own
    distinguishing message: a file missing under either root, an unknown
    scheme, a scheme without a path, and a `package:` path that escapes.
    """
    pkg = tmp_path / "staged"
    pkg.mkdir()
    with pytest.raises(PrepareRefused, match=expect):
        install(
            _spec(recipes=[declared]),
            staged_package=str(pkg),
            config_dir=str(tmp_path / "config"),
        )


def test_the_root_is_the_scheme_and_never_which_file_happens_to_exist(
    tmp_path: Path,
) -> None:
    """The recipe scheme decides the root; a same-named file in the package
    cannot shadow it. `agent_sys:serena` always resolves to the shipped
    recipe, `package:serena` to the staged package's file, even both present.
    """
    pkg = tmp_path / "staged"
    pkg.mkdir()
    shadower = pkg / "serena"
    shadower.write_text("version: 1\n", encoding="utf-8")
    shipped = Path(agent_assets.__file__).parent / "recipes" / "serena.yaml"
    assert shipped.is_file(), (
        "the shipped recipe this test asserts is reached does not exist, so the "
        "assertion below would be about an empty directory"
    )

    # `agent_sys:` reaches ours, with the shadower sitting right there.
    assert agent_assets._recipe_paths(
        _spec(recipes=["agent_sys:serena"]), staged_package=str(pkg)
    ) == [str(shipped)]

    # `package:` reaches theirs — the same name, the other root, no ambiguity.
    assert agent_assets._recipe_paths(
        _spec(recipes=["package:serena"]), staged_package=str(pkg)
    ) == [str(shadower)]


def test_a_package_recipe_with_no_staged_package_says_so(tmp_path: Path) -> None:
    """A `package:` recipe reference with no staged package
    (`staged_package=None`) refuses with a distinct message from "file not
    found", since there is no package to resolve against at all.
    """
    with pytest.raises(PrepareRefused, match="no staged task package"):
        install(
            _spec(recipes=["package:r.yaml"]),
            staged_package=None,
            config_dir=str(tmp_path / "config"),
        )


def test_a_malformed_recipe_is_a_failed_outcome_and_not_a_raise(tmp_path: Path) -> None:
    """A malformed recipe (parses as YAML but fails validation) is a `fail`
    outcome in the report, not a raised exception — the file exists and the
    parser's complaint belongs beside the other install results.
    """
    pkg = tmp_path / "staged"
    _write(pkg / "r.yaml", "items: [not a mapping]\n")
    got = install(
        _spec(recipes=["package:r.yaml"]), staged_package=str(pkg), config_dir=str(tmp_path / "config")
    )
    # `cli.main` already catches `RecipeError` into a `fail` outcome and exits 2,
    # so what arrives is the status line plus the parser's complaint — the child
    # ran and reported, which is the distinction from the case below.
    assert _levels(got.report) == ["info", "fail"]
    assert got.report[0].details["rc"] == 2
    assert "RecipeError" in got.report[1].message


def test_a_child_that_produces_no_report_is_a_failure_and_not_a_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recipe child whose result is unknown — no stdout, e.g. because the
    interpreter could not start — is reported as a `fail`, not treated as
    success.
    """
    monkeypatch.setattr(agent_assets.sys, "executable", str(tmp_path / "no-such-python"))
    pkg = tmp_path / "staged"
    _write(pkg / "r.yaml", "version: 1\ntarget: {kind: repo, path: /tmp}\nitems: []\n")

    got = install(
        _spec(recipes=["package:r.yaml"]), staged_package=str(pkg), config_dir=str(tmp_path / "config")
    )

    assert _levels(got.report) == ["fail"]
    assert "no usable report" in got.report[0].message


# --------------------------------------------------------------------------- #
# Capabilities, one per test


def test_settings_are_written_before_any_install(tmp_path: Path, fake_claude: FakeCli) -> None:
    """`settings.json` exists before the first plugin install runs, since
    `claude plugin marketplace add`/`install` merge into it rather than
    replacing it. The fake `claude` records file-existed-at-invocation-time.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(
        assets / ".claude" / "settings.json",
        json.dumps({"hooks": {"PreToolUse": ["a"]}, "model": "from-the-package"}),
    )
    _write(
        assets / ".claude" / "plugins" / ".claude-plugin" / "marketplace.json",
        json.dumps({"name": "mp", "owner": "us", "plugins": [{"name": "p1"}, {"name": "p2"}]}),
    )
    config = tmp_path / "config"

    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(config),
        agent_cli=fake_claude.path,
    )

    assert got.settings == {"hooks": {"PreToolUse": ["a"]}, "model": "from-the-package"}
    on_disk = json.loads((config / "settings.json").read_text())
    assert on_disk == got.settings

    invoked = fake_claude.log.read_text().splitlines()
    assert invoked[0].startswith("plugin marketplace add ")
    assert f"CLAUDE_CONFIG_DIR={config}" in invoked
    # **The ordering itself, observed from inside the child.** This is the
    # module's most-argued decision and it was untested: with the write moved to
    # after the installs the suite stayed green, because every assertion read
    # the file once `install` had returned.
    assert set(ln for ln in invoked if ln.startswith("settings_existed=")) == {
        "settings_existed=yes"
    }, invoked
    assert [line for line in invoked if line.startswith("plugin install")] == [
        "plugin install p1@mp",
        "plugin install p2@mp",
    ]


def test_a_failing_plugin_install_is_a_named_outcome_and_not_a_silent_skip(
    tmp_path: Path, failing_claude: str
) -> None:
    """rc and output land in the report. An agent whose plugin did not install
    is a run that will behave differently, and the only place that can be seen
    is here."""
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(
        assets / ".claude" / "plugins" / ".claude-plugin" / "marketplace.json",
        json.dumps({"name": "mp", "owner": "us", "plugins": [{"name": "p1"}]}),
    )

    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
        agent_cli=failing_claude,
    )
    failures = [o for o in got.report if o.level == "fail"]
    assert failures, _levels(got.report)
    assert failures[0].details["rc"] == 1
    assert "boom" in failures[0].details["output"]
    # **And the plugin install was not attempted.** One cause, one message: every
    # subsequent install would fail for the same reason and say so once each.
    assert not [o for o in got.report if "plugin p1@mp" in o.message]


def test_external_and_bundled_mcp_servers_arrive_in_one_mapping(tmp_path: Path) -> None:
    """`.mcp.json`'s entries verbatim, plus one generated entry per
    ``tools/*.mcp.py``. One mapping because they are one thing to the model."""
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(
        assets / ".claude" / ".mcp.json",
        json.dumps({"mcpServers": {"weather": {"type": "http", "url": "http://x"}}}),
    )
    _write(assets / ".claude" / "tools" / "envchk.mcp.py", "# a server\n")
    config = tmp_path / "config"

    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(config),
    )
    assert got.mcp_servers["weather"] == {"type": "http", "url": "http://x"}
    assert got.mcp_servers["envchk"]["type"] == "stdio"
    # The PLACED path, not the source: the entry's `args` name the file where
    # it landed in the config directory, not where it was staged from.
    assert got.mcp_servers["envchk"]["args"] == [str(config / "tools" / "envchk.mcp.py")]




def _imported_from(*roots: Path) -> list[tuple[str, str]]:
    """Every `sys.modules` entry whose file lies under one of `roots`, used to
    detect whether a component's tree was imported into the supervisor.
    """
    found = []
    for name, module in list(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if not origin:
            continue
        real = os.path.realpath(origin)
        if any(real.startswith(os.path.realpath(str(root)) + os.sep) for root in roots):
            found.append((name, real))
    return found


def test_no_member_of_a_claude_tree_is_ever_imported_into_the_supervisor(
    tmp_path: Path,
) -> None:
    """Nothing under a component's `.claude/` tree is ever imported into the
    supervisor: it is data this process places, never code it runs. Two
    detectors — a raise-on-import fixture and a `sys.modules` diff — are
    validated against a benign module imported on purpose at the end.
    """
    pkg = tmp_path / "staged"
    config = tmp_path / "config"
    assets = _package(pkg)

    # One member per place a component can put a file, because the narrow test
    # already covers `tools/` and the point here is everywhere else.
    boom = "raise RuntimeError('a .claude/ tree member was imported into the supervisor')\n"
    _write(assets / ".claude" / "at_root.py", boom)
    _write(assets / ".claude" / "tools" / "in_tools.py", boom)
    _write(assets / ".claude" / "hooks" / "in_hooks.py", boom)
    _write(assets / ".claude" / "skills" / "s" / "in_skills.py", boom)
    # Benign on purpose: importing this one would raise nothing, so it is the
    # member only the `sys.modules` detector can catch -- and the one the
    # control at the end uses.
    _write(assets / ".claude" / "tools" / "quiet.py", "VALUE = 1\n")

    before = set(sys.modules)
    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(config),
    )

    # **The subject is present**, at the path each member actually lands at --
    # opened rather than assumed, because a claim about where a file goes is
    # exactly the kind this package requires someone to have looked at.
    for landed in (
        config / "at_root.py",
        config / "tools" / "in_tools.py",
        config / "hooks" / "in_hooks.py",
        config / "skills" / "s" / "in_skills.py",
        config / "tools" / "quiet.py",
    ):
        assert landed.is_file(), (
            f"{landed} was never placed, so this test would pass with an import "
            f"route fully restored"
        )
    assert _installs(got.report) == []

    # **And inert.** Nothing under either the source tree or the placed copy is
    # in `sys.modules`. Both roots, because "load the placed copy, not the
    # source" was the property `env_checker` lost -- importing either is the
    # defect, and naming only one would let the other through.
    leaked = [
        (name, origin)
        for name, origin in _imported_from(pkg, config)
        if name not in before
    ]
    assert leaked == [], f"a component's file was imported into the supervisor: {leaked}"

    # **The control, and it runs every time.** Import a placed member by hand
    # and require the same detector to flag it. Without this the assertion above
    # is indistinguishable from one whose detector never worked.
    control = "_agent_assets_import_detector_control"
    spec = importlib.util.spec_from_file_location(control, config / "tools" / "quiet.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[control] = module
    try:
        spec.loader.exec_module(module)
        assert [n for n, _ in _imported_from(pkg, config) if n not in before] == [control], (
            "the detector did not see a module it was just handed, so its silence "
            "above proves nothing"
        )
    finally:
        del sys.modules[control]


def test_nothing_under_tools_is_ever_imported_into_the_supervisor(
    tmp_path: Path,
) -> None:
    """No file under a component's `tools/` directory is ever imported or
    executed by the supervisor. Both fixture files raise at module scope, so
    execution surfaces as a `RuntimeError`, not a later wrong assertion.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(assets / ".claude" / "tools" / "helper.py", "raise RuntimeError('never run')\n")
    _write(
        assets / ".claude" / "tools" / "t.tooldef.py",
        "raise RuntimeError('the in-process route is deleted; nothing may import this')\n",
    )

    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
    )

    # **The subject is present.** Placed like every other member of `tools/`,
    # which is what makes the inertness below a fact about the loader rather
    # than about an empty directory.
    placed = tmp_path / "config" / "tools"
    assert (placed / "helper.py").is_file()
    assert (placed / "t.tooldef.py").is_file(), (
        "the fixture never reached the zone, so this test would pass with the "
        "import route fully restored"
    )

    # Inert: no install ran, nothing was reported as an in-process tool, and
    # `AgentMaterial` has no field for one to arrive through. Matched on the
    # exact "in-process tool" wording, not the bare word "tool", which the
    # legitimate "placed 'tools' from ..." outcome also contains.
    assert _installs(got.report) == []
    assert not [o for o in got.report if "in-process tool" in o.message], _levels(got.report)
    assert not hasattr(got, "tools")


def test_a_settings_file_that_does_not_parse_refuses(tmp_path: Path) -> None:
    """`harness.harness_env`'s rule: an error one character wide, and continuing
    starts an agent missing its hooks that blames itself."""
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(assets / ".claude" / "settings.json", "{not json")
    with pytest.raises(PrepareRefused, match="could not be read"):
        install(
            _spec(assets="assets/forge.agent"),
            staged_package=str(pkg),
            config_dir=str(tmp_path / "config"),
        )


# --------------------------------------------------------------------------- #
# The seam into `material.deploy`


def test_deploy_copies_the_asset_directory_into_the_workspace_as_a_subdirectory(
    tmp_path: Path,
) -> None:
    """`material.deploy` copies the asset directory into the workspace as a
    subdirectory (`<workspace>/<agent>.agent/`), not spilled into the
    workspace root, avoiding collision with the agent's own working tree.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(assets / "readme.md", "# how to forge\n")
    workspace = tmp_path / "zone" / "workspace"
    workspace.mkdir(parents=True)

    deployed = material.deploy(
        _spec(assets="assets/forge.agent"),
        _ZoneAt(str(tmp_path / "zone")),
        str(pkg),
        str(workspace),
    )

    assert (workspace / "forge.agent" / "readme.md").read_text() == "# how to forge\n"
    assert deployed.environment[AGENT_ASSETS_ENV_VAR] == str(assets)


def test_the_two_argument_call_still_works_and_installs_nothing(tmp_path: Path) -> None:
    """`material.deploy`'s original two-argument call (no staged package, no
    workspace) still works and installs nothing, since there is nothing to
    resolve components against.
    """
    deployed = material.deploy(_spec(), _ZoneAt(str(tmp_path / "zone")))
    assert deployed.environment["CLAUDE_CONFIG_DIR"] == str(tmp_path / "zone" / "config")
    assert deployed.mcp_servers == {}


# --------------------------------------------------------------------------- #
# The subprocess route, and what it pins


def test_the_recipe_child_runs_this_worktree_and_not_an_installed_one() -> None:
    """A recipe child resolves `env_mgr` to this worktree, not any other
    installed copy, run from `cwd="/tmp"` so the tree's own cwd entry can't
    mask a broken `PYTHONPATH` pin.
    """
    ours = Path(agent_assets.__file__).resolve().parents[1]
    assert Path(agent_assets._PACKAGE_ROOT) == ours

    env = agent_assets._child_env(None, "/tmp")
    proc = subprocess.run(
        [sys.executable, "-c", "import env_mgr; print(env_mgr.__file__)"],
        capture_output=True,
        text=True,
        env=env,
        # `cwd` outside the tree: pytest runs from `agent_sys/`, which would
        # put the right tree on the child's `sys.path` via the cwd entry
        # alone. From `/tmp` the only route to `env_mgr` is `PYTHONPATH`.
        cwd="/tmp",
    )
    assert Path(proc.stdout.strip()).resolve().parent == ours / "env_mgr"


def test_nothing_here_mutates_the_supervisors_environment(tmp_path: Path) -> None:
    """`install` never mutates `os.environ`; environment values reach the
    recipe child as an explicit argument, since the supervisor can run
    prepares concurrently across threads.
    """
    before = dict(os.environ)
    pkg = tmp_path / "staged"
    _write(pkg / "r.yaml", "version: 1\ntarget: {kind: repo, path: /tmp}\nitems: []\n")

    install(
        _spec(recipes=["package:r.yaml"]),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
    )

    assert os.environ == before


def test_the_shipped_serena_recipe_resolves_by_bare_name_and_parses() -> None:
    """`recipes: [serena]` resolves to the shipped
    `env_mgr/recipes/serena.yaml` and parses via `dry-run --json`. Asserts
    `status == "OK"`, since a `RecipeError` also prints a well-formed report.
    """
    resolved = agent_assets._recipe_paths(_spec(recipes=["agent_sys:serena"]), staged_package=None)
    assert resolved == [str(Path(agent_assets.__file__).parent / "recipes" / "serena.yaml")]

    proc = subprocess.run(
        [sys.executable, "-m", "env_mgr", "dry-run", resolved[0], "--json", "--path", "/tmp"],
        capture_output=True,
        text=True,
        env=agent_assets._child_env(None, "/tmp"),
    )
    document = json.loads(proc.stdout)
    assert document["status"] == "OK", document


# --------------------------------------------------------------------------- #
# `${VAR}` in a component's `.mcp.json`


def test_a_declared_server_expands_against_the_zone_environment(tmp_path: Path) -> None:
    """`${VAR}` in a declared MCP server's `command`, `args`, and `env` all
    expand against the run's environment mapping.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(
        assets / ".claude" / MCP_REL,
        json.dumps(
            {
                "mcpServers": {
                    "serena": {
                        "type": "stdio",
                        "command": "${UV_TOOL_BIN_DIR}/serena",
                        "args": ["start-mcp-server", "--project", "${AGENT_SYS_MY_WORKSPACE}"],
                        "env": {"HOME": "${AGENT_SYS_MY_PLAYGROUND}"},
                    }
                }
            }
        ),
    )

    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
        environ={
            "UV_TOOL_BIN_DIR": "/zone/uv_bin",
            "AGENT_SYS_MY_WORKSPACE": "/zone/workspace",
            "AGENT_SYS_MY_PLAYGROUND": "/zone/playground",
        },
    )

    entry = got.mcp_servers["serena"]
    assert entry["command"] == "/zone/uv_bin/serena"
    assert entry["args"] == ["start-mcp-server", "--project", "/zone/workspace"]
    assert entry["env"]["HOME"] == "/zone/playground"


def test_an_unresolved_variable_in_a_declared_server_refuses(tmp_path: Path) -> None:
    """An unresolved `${VAR}` in a declared MCP server's fields refuses rather
    than passing through literally, since a literal `${...}` in argv would
    start a server with no tools and no visible error.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(
        assets / ".claude" / MCP_REL,
        json.dumps({"mcpServers": {"x": {"command": "${NOT_SET_ANYWHERE}/x"}}}),
    )

    with pytest.raises(PrepareRefused, match=r"NOT_SET_ANYWHERE"):
        install(
            _spec(assets="assets/forge.agent"),
            staged_package=str(pkg),
            config_dir=str(tmp_path / "config"),
            environ={},
        )


# --------------------------------------------------------------------------- #
# Probe F — the marketplace is registered from inside the zone


def test_the_marketplace_is_copied_into_the_zone_before_it_is_registered(
    tmp_path: Path, fake_claude: FakeCli
) -> None:
    """A component's marketplace is copied into the zone before `claude
    plugin marketplace add` registers it, so the registered source path is
    inside the zone, not the staged package that only installs cleanly.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(
        assets / ".claude" / "plugins" / ".claude-plugin" / "marketplace.json",
        json.dumps({"name": "mp", "owner": "us", "plugins": [{"name": "p1"}]}),
    )
    _write(assets / ".claude" / "plugins" / "p1" / "skills" / "s" / "SKILL.md", "# s")
    config = tmp_path / "config"

    install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(config),
        agent_cli=fake_claude.path,
    )

    registered = config / agent_assets.MARKETPLACES_DIRNAME / "mp"
    assert (registered / "p1" / "skills" / "s" / "SKILL.md").exists()
    (add_line,) = [ln for ln in fake_claude.log.read_text().splitlines() if "marketplace add" in ln]
    assert add_line.endswith(str(registered)), add_line
    # And the source is untouched — the copy is a copy.
    assert (assets / ".claude" / "plugins" / "p1").is_dir()


# --------------------------------------------------------------------------- #
# The install report, and the two exported names


def test_the_install_report_is_written_into_the_zone_and_its_path_exported(
    tmp_path: Path,
) -> None:
    """`install`'s report is written under `logs_dir` and its path exported
    as `AGENT_SYS_INSTALL_REPORT`, which consumers rely on to distinguish an
    honest "unavailable" verdict from a broken install.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(assets / ".claude" / "skills" / "s" / "SKILL.md", "# s")
    logs = tmp_path / "zone" / "logs"

    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
        logs_dir=str(logs),
    )

    path = got.env["AGENT_SYS_INSTALL_REPORT"]
    assert Path(path).parent == logs
    document = json.loads(Path(path).read_text())
    assert [o["level"] for o in document["outcomes"]] == ["ok"]


def test_install_exports_no_path_outside_the_zone(tmp_path: Path) -> None:
    """`install` exports only `AGENT_SYS_AGENT_ASSETS` and
    `AGENT_SYS_INSTALL_REPORT`, both inside the run's own tree. Asserted as
    an exact set, so any new out-of-zone export fails here too.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(assets / ".claude" / "skills" / "s" / "SKILL.md", "# s")
    zone = tmp_path / "zone"

    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(zone / "config"),
        logs_dir=str(zone / "logs"),
    )

    assert set(got.env) == {"AGENT_SYS_AGENT_ASSETS", "AGENT_SYS_INSTALL_REPORT"}, got.env
    for name, value in got.env.items():
        assert Path(value).is_relative_to(tmp_path), f"{name} points outside the run: {value}"




def test_a_recipe_child_that_overruns_is_killed_and_reported(tmp_path: Path) -> None:
    """A recipe child past `recipe_timeout` is killed and reported as a
    `fail`, naming the timeout and that any partial install remains. Driven
    with a real sleeping recipe rather than a patched subprocess.
    """
    pkg = tmp_path / "staged"
    _write(
        pkg / "slow.yaml",
        "version: 1\n"
        f"target: {{kind: repo, name: t, path: {tmp_path}}}\n"
        "items:\n"
        "  - installer: oneline\n"
        "    importance: suggested\n"
        "    name: slow\n"
        "    run: sleep 30\n",
    )

    got = install(
        _spec(recipes=["package:slow.yaml"]),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
        recipe_timeout=0.5,
    )

    assert _levels(got.report) == ["fail"]
    assert "killed after 0.5s" in got.report[0].message
    # **The partial install is named rather than glossed.** The child is dead;
    # whatever it had already done is still done, and a report that implied
    # otherwise would be worse than the hang it replaces.
    assert "still installed" in got.report[0].message


def test_the_recipe_timeout_is_a_parameter_with_a_stated_default() -> None:
    """`RECIPE_TIMEOUT_SECONDS` defaults to 20 minutes — enough headroom for a
    slow install from a cold cache, not a defensively-chosen round number.
    """
    assert agent_assets.RECIPE_TIMEOUT_SECONDS == 20 * 60


# --------------------------------------------------------------------------- #
# The pinned CLI


def test_plugin_installs_run_the_pinned_cli_and_never_the_bare_name(
    tmp_path: Path, fake_claude: FakeCli
) -> None:
    """Plugin installs always run the pinned, absolute `agent_cli` path,
    never a bare `claude` off `PATH`. The fake CLI is placed on `PATH` too,
    so a bare-name fallback would also succeed if one were taken.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(
        assets / ".claude" / "plugins" / ".claude-plugin" / "marketplace.json",
        json.dumps({"name": "mp", "owner": "us", "plugins": [{"name": "p1"}]}),
    )

    install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
        agent_cli=fake_claude.path,
    )

    # The stub records `$*`, so argv[0] is absent from the log — what it proves
    # is that the *pinned* stub ran at all. That it ran by absolute path is what
    # the source assertion below covers, and the two together are the claim.
    lines = fake_claude.log.read_text().splitlines()
    assert any(ln.startswith("plugin marketplace add ") for ln in lines), lines
    assert "plugin install p1@mp" in lines

    source = Path(agent_assets.__file__).read_text()
    assert '"claude", "plugin"' not in source, (
        "a bare `claude` came back into agent_assets.py; under the derived PATH "
        "that reaches a different build from the one the session runs"
    )


def test_an_agent_with_plugins_and_no_pinned_cli_fails_rather_than_guessing(
    tmp_path: Path, fake_claude: FakeCli
) -> None:
    """With plugins declared but no pinned `agent_cli`, `install` reports a
    `fail` rather than a bare-`claude` fallback or a raise. `fake_claude` is
    on `PATH` so a fallback *would* have worked; this refuses to guess.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(
        assets / ".claude" / "plugins" / ".claude-plugin" / "marketplace.json",
        json.dumps({"name": "mp", "owner": "us", "plugins": [{"name": "p1"}]}),
    )

    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
        agent_cli=None,
    )

    assert _levels(got.report) == ["fail"]
    assert "pinned no `claude` CLI" in got.report[0].message
    assert not fake_claude.log.exists(), "it fell back to the CLI on PATH"


def test_an_agent_with_no_plugins_and_no_cli_is_a_working_configuration(
    tmp_path: Path,
) -> None:
    """An agent with no plugins and no `claude` CLI installs successfully —
    a machine with no `claude` still runs non-AI tasks, so absence of a CLI
    must only be fatal to work that actually needs one.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(assets / ".claude" / "skills" / "s" / "SKILL.md", "# s")

    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
        agent_cli=None,
    )

    assert _levels(got.report) == ["ok"]


def test_the_child_gets_the_policy_derived_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`material.deploy`'s `base_env` (e.g. `PATH`) reaches the recipe child
    via `install`'s `environ` argument, and is not echoed back into
    `deployed.environment`.
    """

    class _Zone:
        root = str(tmp_path / "zone")

    # Assert on the mapping `install` was handed, captured at the seam.
    # `test_a_recipe_item_runs_a_binary_reachable_only_through_base_env`
    # proves the same claim end to end; this one localises it, so a
    # failure says which half broke.
    seen: dict[str, Any] = {}
    real = agent_assets.install

    def capture(*args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs.get("environ") or {})
        return real(*args, **kwargs)

    monkeypatch.setattr(material.agent_assets, "install", capture)
    deployed = material.deploy(
        _spec(),
        _Zone(),
        None,
        None,
        base_env={"PATH": "/usr/bin:/bin", "SOMETHING_ELSE": "x"},
    )

    assert seen["PATH"] == "/usr/bin:/bin", "base_env never reached the installer"
    assert seen["SOMETHING_ELSE"] == "x"

    # And **not** echoed back: what `deploy` returns is what `deploy` decided,
    # and `prepare` already holds the rest.
    assert "PATH" not in deployed.environment
    assert "SOMETHING_ELSE" not in deployed.environment

    child = agent_assets._child_env({"PATH": "/usr/bin:/bin"}, str(tmp_path / "cfg"))
    assert child["PATH"] == "/usr/bin:/bin"


# --------------------------------------------------------------------------- #
# Place by default — the blocker, and the general assertion that would have
# caught it


def test_every_member_of_a_claude_tree_is_placed_except_the_named_exceptions(
    tmp_path: Path,
) -> None:
    """Every member of a `.claude/` tree is placed except those named in
    `_NOT_PLACED`. Asserted over the whole directory listing, so a member
    Claude Code adds later fails unless placed or explicitly excepted.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    tree = assets / ".claude"
    _write(tree / "settings.json", json.dumps({"model": "m"}))
    _write(tree / MCP_REL, json.dumps({"mcpServers": {}}))
    _write(tree / "skills" / "s" / "SKILL.md", "# s")
    _write(tree / "hooks" / "on_start.py", "# hook")
    _write(tree / "servers" / "srv.py", "# server")
    _write(tree / "tools" / "t.mcp.py", "# tool")
    _write(tree / "agents" / "sub.md", "# a subagent Claude Code may add later")
    _write(tree / "commands" / "c.md", "# a slash command")
    config = tmp_path / "config"

    install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(config),
    )

    for member in sorted(os.listdir(tree)):
        placed = (config / member).exists()
        if member in agent_assets._NOT_PLACED:
            continue
        assert placed, (
            f"{member!r} is in a .claude/ tree, is not in _NOT_PLACED, and did not "
            f"reach the config directory. Either place it or except it by name"
        )

    assert (config / "hooks" / "on_start.py").read_text() == "# hook"
    assert (config / "servers" / "srv.py").read_text() == "# server"
    assert (config / "agents" / "sub.md").exists()
    assert (config / "commands" / "c.md").exists()

    # The three exceptions, each for the reason `_NOT_PLACED` gives.
    assert json.loads((config / "settings.json").read_text()) == {"model": "m"}, (
        "settings.json is written by the merge, not copied from one level"
    )
    assert not (config / MCP_REL).exists(), (
        ".mcp.json is read and carried as data; a copy in the zone reads as a "
        "configuration the session honours"
    )


def test_every_mcp_server_we_produced_names_a_file_that_exists(tmp_path: Path) -> None:
    """Every MCP server entry `install` produces names files that actually
    exist on disk — both a declared server's `${CLAUDE_CONFIG_DIR}`-relative
    args and a bundled server's placed (not source) path.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(assets / ".claude" / "servers" / "declared.py", "# declared server")
    _write(assets / ".claude" / "tools" / "bundled.mcp.py", "# bundled server")
    _write(
        assets / ".claude" / MCP_REL,
        json.dumps(
            {
                "mcpServers": {
                    "declared": {
                        "type": "stdio",
                        "command": "python3",
                        "args": ["${CLAUDE_CONFIG_DIR}/servers/declared.py"],
                    }
                }
            }
        ),
    )
    config = tmp_path / "config"

    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(config),
        environ={"CLAUDE_CONFIG_DIR": str(config)},
    )

    assert set(got.mcp_servers) == {"declared", "bundled"}
    for name, entry in got.mcp_servers.items():
        for arg in entry.get("args", []):
            if arg.startswith(os.sep):
                assert Path(arg).exists(), (
                    f"MCP server {name!r} names {arg!r} and nothing is there. It "
                    f"would be reported installed, fail to start, and present as a "
                    f"server with no tools"
                )
        command = entry.get("command", "")
        if command.startswith(os.sep):
            assert Path(command).exists(), f"{name!r}'s command {command!r} is absent"


def test_a_marketplace_name_that_climbs_out_of_the_zone_copies_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_claude: FakeCli
) -> None:
    """A marketplace `name` that climbs out of the zone is refused before
    anything is written. Checked on the filesystem: nothing appears at the
    escaped location, and the CLI is never invoked to register it.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    escape = tmp_path / "ESCAPED"
    _write(
        assets / ".claude" / "plugins" / ".claude-plugin" / "marketplace.json",
        json.dumps({"name": "../../../ESCAPED", "owner": "us", "plugins": []}),
    )
    config = tmp_path / "deep" / "zone" / "config"

    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(config),
        agent_cli=fake_claude.path,
    )

    assert "fail" in _levels(got.report)
    assert not escape.exists(), "the tree was written outside the zone before the refusal"
    assert not fake_claude.log.exists(), "it registered a marketplace it had refused"


@pytest.mark.parametrize("declared", ["../../../escaped", "/etc"])
def test_an_assets_path_that_leaves_the_staged_package_refuses(
    tmp_path: Path, declared: str
) -> None:
    """A declared `assets` path that leaves the staged package is refused,
    since `_tooldefs` imports from this directory into the supervisor and
    it must never resolve outside the staged copy.
    """
    pkg = tmp_path / "staged"
    pkg.mkdir()
    with pytest.raises(PrepareRefused, match="stay inside the staged package"):
        install(
            _spec(assets=declared),
            staged_package=str(pkg),
            config_dir=str(tmp_path / "config"),
        )


def test_a_same_tree_mcp_name_collision_is_reported(tmp_path: Path) -> None:
    """A name collision between `.mcp.json` and `tools/*.mcp.py` **within one
    component** is reported as a `warn`, not silently overwritten.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(assets / ".claude" / "tools" / "dup.mcp.py", "# bundled")
    _write(
        assets / ".claude" / MCP_REL,
        json.dumps({"mcpServers": {"dup": {"type": "http", "url": "http://x"}}}),
    )

    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
    )

    assert [o.level for o in got.report if "also ships" in o.message] == ["warn"]
    assert got.mcp_servers["dup"]["type"] == "stdio"


def test_a_components_marketplace_never_lands_on_the_harnesss_own_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_claude: FakeCli
) -> None:
    """A component's marketplace is copied under `MARKETPLACES_DIRNAME`,
    never onto `<config>/plugins/`, which belongs to the CLI's own
    bookkeeping — `plugins/` is relocated rather than placed like any other.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(
        assets / ".claude" / "plugins" / ".claude-plugin" / "marketplace.json",
        json.dumps({"name": "mp", "owner": "us", "plugins": []}),
    )
    _write(assets / ".claude" / "plugins" / "SENTINEL", "the agent's own copy")
    config = tmp_path / "config"

    install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(config),
        agent_cli=fake_claude.path,
    )

    assert (config / agent_assets.MARKETPLACES_DIRNAME / "mp" / "SENTINEL").exists()
    assert not (config / "plugins" / "SENTINEL").exists(), (
        "the agent's marketplace was copied into the directory the CLI owns"
    )



def test_a_recipe_item_runs_a_binary_reachable_only_through_base_env(
    tmp_path: Path,
) -> None:
    """A recipe item can invoke a binary that exists only in a directory
    named by `base_env`'s `PATH`, driven through the real `python -m
    env_mgr` child rather than any one link of that chain in isolation.
    """
    binroot = tmp_path / "onlyhere"
    binroot.mkdir()
    marker = tmp_path / "ran.txt"
    tool = binroot / "envchk-only-here"
    tool.write_text(f"#!/bin/sh\necho ran > {marker}\n", encoding="utf-8")
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    pkg = tmp_path / "staged"
    _write(
        pkg / "r.yaml",
        "version: 1\n"
        f"target: {{kind: repo, name: t, path: {tmp_path}}}\n"
        "items:\n"
        "  - installer: oneline\n"
        "    importance: suggested\n"
        "    name: only-here\n"
        "    run: envchk-only-here\n",
    )

    class _Zone:
        root = str(tmp_path / "zone")

    material.deploy(
        _spec(recipes=["package:r.yaml"]),
        _Zone(),
        str(pkg),
        None,
        base_env={"PATH": str(binroot)},
    )

    assert marker.exists(), (
        "the recipe child could not reach a binary that base_env's PATH names — "
        "this is `uv: not found` reproduced"
    )


@pytest.mark.parametrize("market", ["..", ".", "a/..", "sub/mp", "/abs", ""])
def test_a_marketplace_name_that_is_not_a_single_directory_name_is_refused(
    tmp_path: Path, fake_claude: FakeCli, market: str
) -> None:
    """A marketplace `name` that is not a single directory component (e.g.
    `".."`, `"a/.."`, an absolute path) is refused, even though it never
    escapes the zone — `".."` normalizes to the config root itself.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(
        assets / ".claude" / "plugins" / ".claude-plugin" / "marketplace.json",
        json.dumps({"name": market, "owner": "us", "plugins": [{"name": "p1"}]}),
    )
    _write(assets / ".claude" / "plugins" / "SENTINEL", "the agent's own copy")
    _write(assets / ".claude" / "settings.json", json.dumps({"model": "m"}))
    config = tmp_path / "config"

    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(config),
        agent_cli=fake_claude.path,
    )

    assert "fail" in _levels(got.report), got.report
    # Nothing was copied **anywhere**: not into the config root, not into the
    # harness's own `plugins/`, not into `marketplaces/`.
    assert not (config / "SENTINEL").exists()
    assert not (config / ".claude-plugin").exists()
    assert not (config / "plugins").exists()
    assert not (config / agent_assets.MARKETPLACES_DIRNAME).exists()
    # And nothing was registered — `<config>` itself least of all.
    assert not fake_claude.log.exists(), "a marketplace was registered after the refusal"
    # The rest of the tree still installed; one bad manifest is not fatal.
    assert (config / "settings.json").exists()


def _placing_recipe(pkg: Path, *, relative: str) -> None:
    """A package-layer recipe that writes one file into `$CLAUDE_CONFIG_DIR`,
    used to exercise the case where a recipe and the agent's own `.claude/`
    tree write the same path.
    """
    _write(
        pkg / "assets" / "main.env_recipe.yaml",
        "version: 1\n"
        f"target: {{kind: repo, name: t, path: {pkg}}}\n"
        "items:\n"
        "  - installer: embed\n"
        "    importance: required\n"
        "    name: place\n"
        "    run: |\n"
        f'      mkdir -p "$(dirname "$CLAUDE_CONFIG_DIR/{relative}")"\n'
        f'      printf %s "# from the recipe" > "$CLAUDE_CONFIG_DIR/{relative}"\n',
    )


def test_the_agents_own_tree_replacing_a_recipe_placed_file_is_reported(
    tmp_path: Path,
) -> None:
    """When the agent's own `.claude/` tree overwrites a file a recipe
    already placed, the overwrite is reported as a `warn`, naming the file.
    Precedence is unchanged; only the silence is what changed.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _placing_recipe(pkg, relative="skills/shared/SKILL.md")
    _write(assets / ".claude" / "skills" / "shared" / "SKILL.md", "# from the agent")
    _write(assets / ".claude" / "skills" / "only-mine" / "SKILL.md", "# agent only")
    config = tmp_path / "config"

    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(config),
    )

    (warned,) = [o for o in got.report if o.level == "warn"]
    assert "the package's own assets replaces 1 already-placed file(s)" in warned.message
    assert warned.details["files"] == [os.path.join("shared", "SKILL.md")]
    assert (config / "skills" / "shared" / "SKILL.md").read_text() == "# from the agent"
    assert (config / "skills" / "only-mine" / "SKILL.md").read_text() == "# agent only"


def test_writers_that_share_no_file_do_not_warn(tmp_path: Path) -> None:
    """A recipe and the agent's own tree writing into the same directory
    (e.g. `skills/`) but different files must not warn — only an actual
    same-path collision does.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _placing_recipe(pkg, relative="skills/one/SKILL.md")
    _write(assets / ".claude" / "skills" / "two" / "SKILL.md", "# b")
    config = tmp_path / "config"

    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(config),
    )

    assert (config / "skills" / "one" / "SKILL.md").read_text() == "# from the recipe"
    assert (config / "skills" / "two" / "SKILL.md").exists()
    assert "warn" not in _levels(got.report), got.report


@pytest.mark.parametrize("where", ["toplevel.txt", "nested/inner.txt"])
def test_a_symlink_in_a_claude_tree_is_resolved_at_every_depth(tmp_path: Path, where: str) -> None:
    """A symlink anywhere in a `.claude/` tree — top-level or nested — is
    dereferenced, not preserved. Parametrised over both depths because
    `copy_out`'s underlying behaviour differs by depth.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    target = tmp_path / "outside.txt"
    target.write_text("content from outside the zone\n")
    linked = assets / ".claude" / where
    linked.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, linked)
    config = tmp_path / "config"

    install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(config),
    )

    placed = config / where
    assert placed.exists()
    assert not placed.is_symlink(), (
        f"{where} arrived as a symlink; under confinement it resolves outside "
        f"the zone and the session cannot read it"
    )
    assert placed.read_text() == "content from outside the zone\n"
    from env_mgr.fs.path import contained

    assert contained(str(placed), str(config))


def test_a_claude_tree_of_shapes_nobody_enumerated_is_placed_whole(
    tmp_path: Path,
) -> None:
    """A `.claude/` tree containing shapes beyond named directories — a
    top-level file, an empty directory, a symlink, a nested `.mcp.json` — is
    placed as a whole. Only the top-level `.mcp.json` is read as data.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    tree = assets / ".claude"

    # Directories the code knows nothing about.
    _write(tree / "agents" / "sub.md", "# a subagent")
    _write(tree / "commands" / "c.md", "# a slash command")
    _write(tree / "output-styles" / "terse.md", "# an output style")
    # Directories it does.
    _write(tree / "skills" / "s" / "SKILL.md", "# s")
    _write(tree / "hooks" / "on_start.py", "# hook")
    _write(tree / "servers" / "srv.py", "# server")
    _write(tree / "tools" / "b.mcp.py", "# bundled")
    # Shapes that are not directories at all.
    _write(tree / "CLAUDE.md", "project memory")  # top-level file
    _write(tree / "settings.local.json", '{"model": "m"}')  # sibling of an exception
    statusline = _write(tree / "statusline.sh", "echo hi\n")  # executable
    statusline.chmod(0o755)
    (tree / "empty-dir").mkdir()  # no members to copy
    (tree / "linked.txt").symlink_to(tree / "CLAUDE.md")  # a link, not a file
    _write(tree / "nested" / "deep" / MCP_REL, '{"mcpServers": {}}')  # NOT an interface
    # The exceptions themselves.
    _write(tree / "settings.json", json.dumps({"model": "m"}))
    _write(tree / MCP_REL, json.dumps({"mcpServers": {}}))

    config = tmp_path / "config"
    install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(config),
        environ={"CLAUDE_CONFIG_DIR": str(config)},
    )

    omitted = sorted(
        name
        for name in os.listdir(tree)
        if name not in agent_assets._NOT_PLACED and not (config / name).exists()
    )
    assert not omitted, (
        f"{omitted} are members of a .claude/ tree, are not in _NOT_PLACED, and "
        f"did not reach the config directory. Either place them or except them"
    )

    # The shapes, each asserted for what it is rather than for existing.
    assert (config / "CLAUDE.md").is_file()
    assert (config / "empty-dir").is_dir()
    assert os.access(config / "statusline.sh", os.X_OK), (
        "copy_out uses copy2, so the mode travels; a status line that arrives "
        "without +x is a capability that installs and cannot run"
    )
    assert (config / "nested" / "deep" / MCP_REL).exists(), (
        "only the TOP-LEVEL .mcp.json is an interface document; one below it is "
        "a file the author shipped and is placed like any other"
    )
    # A symlink is dereferenced: `_place_tree` passes `dereference=True`
    # because a preserved link out of the zone fails `contained`. See
    # `test_a_symlink_in_a_claude_tree_is_resolved_at_every_depth` for the
    # out-of-tree case and both depths.
    assert (config / "linked.txt").is_file()
    assert not (config / "linked.txt").is_symlink()


# --------------------------------------------------------------------------- #
# The install report names what it produced, on both MCP routes


def test_both_mcp_routes_record_a_name_in_the_report(tmp_path: Path) -> None:
    """Both MCP routes — external (`.mcp.json`) and bundled
    (`tools/*.mcp.py`) — record a recoverable server name in the install
    report. Every server in `mcp_servers` must be nameable from it alone.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(
        assets / ".claude" / MCP_REL,
        json.dumps(
            {
                "mcpServers": {
                    "weather": {"type": "http", "url": "http://x"},
                    "serena": {"type": "stdio", "command": "/bin/true"},
                }
            }
        ),
    )
    _write(assets / ".claude" / "tools" / "envchk_stdio.mcp.py", "# bundled")

    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
    )

    def details(fragment: str) -> dict[str, Any]:
        (found,) = [o for o in got.report if fragment in o.message]
        return found.details

    # External: the names, because this is the only route whose keys an author
    # chooses in a data file — and both `envchk_baseline` and `serena` are it.
    assert details("external MCP server(s)")["names"] == ["serena", "weather"]

    # Bundled: recorded, not parsed out of the message or the path stem.
    assert details("bundled MCP server")["server"] == "envchk_stdio"

    # Every server the run produced is recoverable from the report alone, which
    # is the property a consumer actually needs.
    recorded = set(details("external MCP server(s)")["names"]) | {
        details("bundled MCP server")["server"]
    }
    assert recorded == set(got.mcp_servers), (
        "a server reached the backend that the install report does not name"
    )



def test_a_bundled_server_entry_states_the_run_environment_rather_than_inheriting_it(
    tmp_path: Path,
) -> None:
    """A bundled MCP server's entry states its `env` explicitly rather than
    relying on inheritance. The whole mapping is asserted, not one key, and
    the entry is a copy — mutating it must not affect the run's own env.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(assets / ".claude" / "tools" / "srv.mcp.py", "# a server")
    config = tmp_path / "config"
    environ = {
        "ENVCHK_NONCE": "deadbeef",
        "PATH": "/usr/bin:/bin",
        "CLAUDE_CONFIG_DIR": str(config),
    }

    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(config),
        environ=environ,
    )

    entry = got.mcp_servers["srv"]
    assert entry["env"]["ENVCHK_NONCE"] == "deadbeef", (
        "the run's declared value is not in the entry, so the server depends on "
        "inheriting it from the CLI child"
    )
    for key, value in environ.items():
        assert entry["env"][key] == value, f"{key} missing; a subset breaks under replace"
    # It is a copy, not the live mapping: one server's entry must not be
    # editable through another's, and `Prepared.mcp_servers` crosses to `agent`.
    assert entry["env"] is not environ


# --------------------------------------------------------------------------- #
# The three recipe layers — default, package, agent. The layer is WHERE THE
# FILE IS, not a field: each test puts a file somewhere and asserts what ran.


def _marker_recipe(path: Path, marker: Path, name: str = "marker") -> None:
    """A recipe whose single item touches `marker`, so "did this layer run" is
    answered by the filesystem rather than by an `Outcome` that could be
    manufactured by a stub."""
    _write(
        path,
        "version: 1\n"
        f"target: {{kind: repo, name: t, path: {path.parent}}}\n"
        "items:\n"
        "  - installer: oneline\n"
        "    importance: suggested\n"
        f"    name: {name}\n"
        f"    run: touch {marker}\n",
    )


def test_the_default_layer_runs_although_nothing_declares_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the default layer: an agent that declares no recipes
    at all still gets it."""
    ran = tmp_path / "default_ran"
    recipe = tmp_path / "default.env_recipe.yaml"
    _marker_recipe(recipe, ran)
    monkeypatch.setattr(agent_assets, "DEFAULT_RECIPE", str(recipe))

    install(_spec(), staged_package=None, config_dir=str(tmp_path / "c"))

    assert ran.exists()


def test_a_missing_default_layer_is_absence_and_not_an_error(tmp_path: Path) -> None:
    """A missing default layer (as this file's autouse fixture arranges) is
    simply absent — no warning, no failure. Only the agent layer can reach
    declared-and-absent, since it is the only layer anything declares.
    """
    got = install(_spec(), staged_package=None, config_dir=str(tmp_path / "c"))

    assert not [o for o in got.report if o.level in ("warn", "fail")]
    # Negative assertion, so it does NOT go red against a tree with no default
    # layer at all — nothing absent can produce a fault. It guards the
    # direction its partner does not; read it with
    # `test_the_default_layer_runs_although_nothing_declares_it`.


def test_the_package_layer_is_auto_detected_in_the_staged_copy(tmp_path: Path) -> None:
    ran = tmp_path / "package_ran"
    pkg = tmp_path / "staged"
    _marker_recipe(pkg / "assets" / "main.env_recipe.yaml", ran)

    install(_spec(), staged_package=str(pkg), config_dir=str(tmp_path / "c"))

    assert ran.exists()


def test_only_one_spelling_of_the_package_recipe_is_admitted(tmp_path: Path) -> None:
    """Only the exact spelling `main.env_recipe.yaml` is recognised as the
    package layer's recipe. Pins that this layer does not reuse the agent
    layer's convention-based lookup, since `env_mgr` doesn't import it.
    """
    ran = tmp_path / "should_not_run"
    pkg = tmp_path / "staged"
    _marker_recipe(pkg / "assets" / "env_recipe.main.yaml", ran)

    install(_spec(), staged_package=str(pkg), config_dir=str(tmp_path / "c"))

    assert not ran.exists()
    # Also negative, and therefore green on a tree where the package layer does
    # not exist at all — it cannot witness the feature, only its boundary. It is
    # meaningful ONLY beside
    # `test_the_package_layer_is_auto_detected_in_the_staged_copy`, which proves
    # the admitted spelling does run.


def test_the_three_layers_run_default_then_package_then_agent(tmp_path: Path) -> None:
    """Order is most general to most specific, and it is observable.

    Each layer appends its own name to one file, so the assertion is the actual
    sequence of executions rather than the order of a list this module built.
    """
    order = tmp_path / "order.txt"
    pkg = tmp_path / "staged"

    def _appender(path: Path, word: str) -> None:
        _write(
            path,
            "version: 1\n"
            f"target: {{kind: repo, name: t, path: {tmp_path}}}\n"
            "items:\n"
            "  - installer: oneline\n"
            "    importance: suggested\n"
            f"    name: {word}\n"
            f"    run: echo {word} >> {order}\n",
        )

    default = tmp_path / "default.env_recipe.yaml"
    _appender(default, "default")
    _appender(pkg / "assets" / "main.env_recipe.yaml", "package")
    _appender(pkg / "recipes" / "agent.yaml", "agent")

    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    mp.setattr(agent_assets, "DEFAULT_RECIPE", str(default))
    try:
        install(
            _spec(recipes=["package:recipes/agent.yaml"]),
            staged_package=str(pkg),
            config_dir=str(tmp_path / "c"),
        )
    finally:
        mp.undo()

    assert order.read_text().split() == ["default", "package", "agent"]


def test_layers_concatenate_rather_than_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A more specific recipe layer adds to a more general one; it does not
    override it — both layers' items with the same name must run.
    """
    from_default = tmp_path / "from_default"
    from_agent = tmp_path / "from_agent"
    pkg = tmp_path / "staged"

    default = tmp_path / "default.env_recipe.yaml"
    # **Deliberately the SAME item name in both layers.** A repeated name is not
    # a conflict — `runner.detect_conflicts` fires only on incompatible version
    # constraints — so both must still run.
    _marker_recipe(default, from_default, name="shared")
    _marker_recipe(pkg / "recipes" / "agent.yaml", from_agent, name="shared")
    monkeypatch.setattr(agent_assets, "DEFAULT_RECIPE", str(default))

    install(
        _spec(recipes=["package:recipes/agent.yaml"]),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "c"),
    )

    assert from_default.exists(), "the agent layer overrode the default instead of adding to it"
    assert from_agent.exists()


def test_the_shipped_default_recipe_exists_and_parses(tmp_path: Path) -> None:
    """The real shipped default recipe file exists at
    `env_mgr/default.env_recipe.yaml` and parses. Asserts the path and that
    it loads, deliberately not its contents, which are free to change.
    """
    from env_mgr.recipe import load_recipe

    assert _SHIPPED_DEFAULT_RECIPE.endswith(os.path.join("env_mgr", "default.env_recipe.yaml")), (
        "the default layer must be in env_mgr/, not in env_mgr/recipes/ — "
        "that placement IS how a reader tells it from a recipe you name"
    )
    assert os.path.isfile(_SHIPPED_DEFAULT_RECIPE)

    target, items = load_recipe(_SHIPPED_DEFAULT_RECIPE)
    assert items, "a default recipe with no items gives the layer no witness"
