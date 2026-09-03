# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`env_mgr.agent_assets` — the three levels of per-agent component.

**Nothing here runs the real `claude` binary.** A fake one goes on `PATH` per
test, which is what makes the *failing* install case testable at all: a machine
with no `claude` and a machine whose `claude plugin install` exits 1 are two
different outcomes and this module reports them differently, so a suite that
could only produce the first would leave the second unmeasured.

The measurements this file encodes were taken 2026-09-03 against Claude Code on
this machine, and are recorded in `agent_assets.py`'s own docstrings. **They are
evidence about `claude` 2.1.246 specifically** — the build `cli/environment.py`
pins — and saying which build is what makes them evidence rather than folklore:
probes B, C and F were first taken against the SDK's *bundled* 2.1.251, because
`_find_cli` prefers its own bundle over `PATH`, and were re-measured as B'/C'/F'
once that was noticed. All six conclusions hold on 2.1.246.

- ``claude plugin marketplace add`` / ``install`` respect ``CLAUDE_CONFIG_DIR``
  fully — each probe's own ``.claude.json`` landed **inside** its relocated
  config directory, and ``~/.claude/plugins/marketplaces`` still held only
  ``claude-plugins-official``. (An earlier version of this note cited
  ``~/.claude.json``'s md5 being unchanged; `PROBES.md` has **withdrawn** that,
  because with no probe running at all the file changed twice in 75 seconds —
  every live Claude Code session on this host rewrites it. The conclusion holds
  on the two facts above; the retracted one is named so nobody re-derives it.)
- they **merge** into an existing ``settings.json`` rather than clobbering it,
  which is why the settings document is written *before* any install runs;
- a local marketplace needs ``.claude-plugin/marketplace.json`` carrying
  ``{name, owner, plugins: [...]}``.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple, get_type_hints

import pytest

from env_mgr import agent_assets, material
from env_mgr.agent_assets import AgentMaterial, install
from env_mgr.paths import AGENT_ASSETS_ENV_VAR
from env_mgr.protocols import Mode, PrepareRefused

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
    """A stand-in `claude`, and the file it records its argv into.

    Both halves are needed by the same tests: the **path** because `install`
    takes the CLI as a pinned absolute argument rather than searching `PATH`,
    and the **log** because two assertions are about *what was invoked* —
    ``marketplace add`` before ``install``, and one ``install`` per plugin in
    the manifest — which a stub that only returned 0 could not tell from no call
    at all.
    """

    path: str
    log: Path


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeCli:
    """A `claude` that records its argv and succeeds.

    Still placed on `PATH` as well as returned by path: a test that asserts the
    **absolute** form is only meaningful if a bare name would also have resolved
    to something. With nothing on `PATH`, `_run_cmd` would answer rc 127 and the
    assertion would pass for the wrong reason.
    """
    binroot = tmp_path / "fakebin"
    binroot.mkdir()
    log = tmp_path / "claude.log"
    script = binroot / "claude"
    script.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> {log}\n'
        f'printf "CLAUDE_CONFIG_DIR=%s\\n" "$CLAUDE_CONFIG_DIR" >> {log}\n'
        # **Records whether settings.json existed AT INVOCATION TIME.** Reading
        # the file after `install` returns cannot distinguish written-before
        # from written-after, so a test that did only that passed with the
        # ordering inverted — verified by mutation, `reviewer` 2026-09-03. The
        # measured behaviour being relied on is that `claude` *merges* into an
        # existing file, and merging is not commutative with creating.
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
    """Levels, minus `_place_tree`'s per-member bookkeeping.

    Placement is reported for every member of a `.claude/` tree, so a test about
    *what an MCP declaration did* would otherwise assert against a count that
    moves whenever a fixture gains a directory. Dropped by message rather than
    by level, so a `fail` is never filtered out — the filter can hide a passing
    detail, never a failing one.
    """
    return [o.level for o in report if not o.message.startswith("placed ")]


def _placed(report: tuple[Any, ...]) -> list[str]:
    return [o.message for o in report if o.message.startswith("placed ")]


# --------------------------------------------------------------------------- #
# L3 — undeclared, auto-detected


def test_l3_is_found_at_the_agent_assets_dot_claude_and_nothing_declares_it(
    tmp_path: Path,
) -> None:
    """The level with no declaration. Detection **is** the interface.

    A second statement of what the directory already says is a second writer of
    one fact, and the two would drift the first time somebody moved the
    directory without editing the YAML.
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
    """Undeclared **and absent** is simply absent, and that is L3's normal shape.

    The counterpart to every refusal below, and it is the case that keeps them
    honest: `material.py:62-86`'s declared-and-absent rule is only defensible
    while the undeclared case stays quiet, because otherwise every agent in the
    tree would have to declare an empty directory to avoid an error.
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
        tools=(),
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
# L2 — declared by name


def test_l2_resolves_a_bare_name_under_the_repositorys_components_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``components: [<name>]`` is a name and never a path, so this can only
    reach a directory we ship."""
    shipped = tmp_path / "components"
    _write(shipped / "envchk" / ".claude" / "skills" / "chk" / "SKILL.md", "# chk\n")
    monkeypatch.setattr(agent_assets, "COMPONENTS_ROOT", str(shipped))
    config = tmp_path / "config"

    got = install(_spec(components=["envchk"]), staged_package=None, config_dir=str(config))

    assert (config / "skills" / "chk" / "SKILL.md").exists()
    assert _levels(got.report) == ["ok"]


def test_a_declared_component_that_does_not_exist_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`material.py`'s declared-and-absent rule, at L2.

    The failure it prevents is the measured one: skip the install silently and
    the agent meets ``Unknown skill`` hours later from inside its own session,
    with nothing in the zone, the events or the logs naming the cause.
    """
    monkeypatch.setattr(agent_assets, "COMPONENTS_ROOT", str(tmp_path / "components"))
    with pytest.raises(PrepareRefused, match="declares component 'nope'"):
        install(_spec(components=["nope"]), staged_package=None, config_dir=str(tmp_path / "c"))


def test_a_component_with_no_dot_claude_directory_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`components/README.md` marks `.claude/` REQUIRED, and a component that
    installs nothing is indistinguishable from one whose contents were
    forgotten."""
    shipped = tmp_path / "components"
    (shipped / "hollow").mkdir(parents=True)
    monkeypatch.setattr(agent_assets, "COMPONENTS_ROOT", str(shipped))
    with pytest.raises(PrepareRefused, match="has no .claude/ directory"):
        install(_spec(components=["hollow"]), staged_package=None, config_dir=str(tmp_path / "c"))


# --------------------------------------------------------------------------- #
# L1 — recipes


def test_l1_runs_a_package_relative_recipe_and_reports_its_status(tmp_path: Path) -> None:
    """A recipe resolved against the **staged** package.

    The item is a shell `oneline` writing a file, so the assertion is that the
    machinery really ran rather than that an `Outcome` was manufactured — this
    is the seam where `agent_assets` crosses the decoupling wall, and a stubbed
    runner would test the stub.
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
        "    layer: system\n"
        "    name: marker\n"
        f"    run: touch {marker}\n",
    )
    got = install(
        _spec(recipes=["recipes/tools.yaml"]),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
    )
    assert marker.exists()
    assert any("tools.yaml" in o.message for o in got.report)


def test_a_declared_recipe_that_resolves_nowhere_refuses(tmp_path: Path) -> None:
    """Both spellings are tried — package-relative, then
    ``env_mgr/recipes/<name>.yaml`` — and the refusal names both candidates so
    the author can see which one they meant to write."""
    pkg = tmp_path / "staged"
    pkg.mkdir()
    with pytest.raises(PrepareRefused, match="declares recipe"):
        install(
            _spec(recipes=["absent"]),
            staged_package=str(pkg),
            config_dir=str(tmp_path / "config"),
        )


def test_a_malformed_recipe_is_a_failed_outcome_and_not_a_raise(tmp_path: Path) -> None:
    """Absent and malformed are different faults with different remedies.

    Absent refuses because the run would proceed with the agent silently short
    of what it declared. Malformed does not, because the file is there, the
    parser's complaint is actionable, and it belongs beside the other install
    results where the author is already looking.
    """
    pkg = tmp_path / "staged"
    _write(pkg / "r.yaml", "items: [not a mapping]\n")
    got = install(
        _spec(recipes=["r.yaml"]), staged_package=str(pkg), config_dir=str(tmp_path / "config")
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
    """The failure mode the JSON round-trip introduces, and the one that must not
    read as success.

    A recipe whose *result is unknown* — the interpreter could not start, the
    child died before `print` — has to be a `fail`. Simulated by pointing the
    child at an interpreter that does not exist, which is the cheapest way to
    get a subprocess with no stdout that is not also a test of the CLI.
    """
    monkeypatch.setattr(agent_assets.sys, "executable", str(tmp_path / "no-such-python"))
    pkg = tmp_path / "staged"
    _write(pkg / "r.yaml", "version: 1\ntarget: {kind: repo, path: /tmp}\nitems: []\n")

    got = install(
        _spec(recipes=["r.yaml"]), staged_package=str(pkg), config_dir=str(tmp_path / "config")
    )

    assert _levels(got.report) == ["fail"]
    assert "no usable report" in got.report[0].message


# --------------------------------------------------------------------------- #
# Capabilities, one per test


def test_settings_are_merged_across_levels_and_written_before_any_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_claude: FakeCli
) -> None:
    """The measured ordering, asserted as an ordering rather than as a file.

    ``claude plugin marketplace add`` *merges* into an existing ``settings.json``
    — a hand-written ``hooks`` key survived a subsequent ``plugin install``,
    which only added ``enabledPlugins`` / ``extraKnownMarketplaces``. Merging is
    not commutative with creating, so the file has to exist first. The fake
    `claude` records what it saw of ``CLAUDE_CONFIG_DIR``, and the settings file
    is asserted to be on disk **by the time it ran**.
    """
    shipped = tmp_path / "components"
    _write(
        shipped / "base" / ".claude" / "settings.json",
        json.dumps({"hooks": {"PreToolUse": ["a"]}, "model": "from-l2"}),
    )
    _write(
        shipped / "base" / ".claude" / "plugins" / ".claude-plugin" / "marketplace.json",
        json.dumps({"name": "mp", "owner": "us", "plugins": [{"name": "p1"}, {"name": "p2"}]}),
    )
    monkeypatch.setattr(agent_assets, "COMPONENTS_ROOT", str(shipped))

    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(assets / ".claude" / "settings.json", json.dumps({"model": "from-l3"}))
    config = tmp_path / "config"

    got = install(
        _spec(components=["base"], assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(config),
        agent_cli=fake_claude.path,
    )

    # L3 wins the collision; L2's untouched key survives the merge.
    assert got.settings == {"hooks": {"PreToolUse": ["a"]}, "model": "from-l3"}
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing_claude: str
) -> None:
    """rc and output land in the report. A component whose plugin did not install
    is a run that will behave differently, and the only place that can be seen
    is here."""
    shipped = tmp_path / "components"
    _write(
        shipped / "base" / ".claude" / "plugins" / ".claude-plugin" / "marketplace.json",
        json.dumps({"name": "mp", "owner": "us", "plugins": [{"name": "p1"}]}),
    )
    monkeypatch.setattr(agent_assets, "COMPONENTS_ROOT", str(shipped))

    got = install(
        _spec(components=["base"]),
        staged_package=None,
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
    # **The PLACED path, not the source.** Registering the source worked for L3
    # only because the staged package happens to be inside the zone; the same
    # file in an L2 component would have named a path under `COMPONENTS_ROOT`,
    # outside every grant — probe F's failure one directory over.
    assert got.mcp_servers["envchk"]["args"] == [str(config / "tools" / "envchk.mcp.py")]


def test_a_tooldef_module_publishes_its_TOOLS_into_the_supervisor(tmp_path: Path) -> None:
    """The one place package-authored code is imported into this process.

    The narrowings are asserted by the tests around this one — only that suffix,
    only under `tools/`, only from the staged package. What this asserts is that
    the mechanism works at all, because an in-process tool that never reaches
    the model is a capability that exists and is unreachable.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(
        assets / ".claude" / "tools" / "probe.tooldef.py",
        "from typing import NamedTuple\n"
        "class T(NamedTuple):\n"
        "    name: str\n"
        "    description: str\n"
        "    schema: dict\n"
        "    call: object\n"
        "TOOLS = [T('probe', 'a probe', {}, lambda: 1)]\n",
    )
    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
    )
    assert [t.name for t in got.tools] == ["probe"]


def test_a_tooldef_that_does_not_import_degrades_and_says_so(tmp_path: Path) -> None:
    """A broken tool module must not take the task down.

    The agent then runs without those tools, which the report names. Raising
    instead would make one bad file in one component fatal to every task that
    uses it — and the component is the thing least likely to be edited by
    whoever hits the failure.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(assets / ".claude" / "tools" / "bad.tooldef.py", "raise RuntimeError('nope')\n")
    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
    )
    assert _installs(got.report) == ["fail"]
    assert "did not import" in got.report[-1].message
    assert got.tools == ()


def test_a_tools_directory_file_with_no_recognised_suffix_is_left_alone(
    tmp_path: Path,
) -> None:
    """The narrowing that matters most, asserted rather than asserted-about.

    Importing package-authored code into the supervisor is done for exactly one
    suffix. A `tools/helper.py` beside a `*.tooldef.py` is a module the author
    expects to be *imported by* their tool, not executed by us.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(assets / ".claude" / "tools" / "helper.py", "raise RuntimeError('never run')\n")
    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
    )
    # The file is *placed* — `tools/` is copied like every other member — and
    # it is not imported, which is the narrowing under test.
    assert _installs(got.report) == []
    assert got.tools == ()


def test_a_later_level_wins_an_mcp_name_collision_and_the_collision_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L1 -> L2 -> L3, so the package's own copy wins — `material.deploy`'s
    precedence, *an author saying so outranks a default*. Reported either way,
    because a silently replaced server is different tools than the ones someone
    wrote."""
    shipped = tmp_path / "components"
    _write(
        shipped / "base" / ".claude" / MCP_REL,
        json.dumps({"mcpServers": {"shared": {"type": "http", "url": "l2"}}}),
    )
    monkeypatch.setattr(agent_assets, "COMPONENTS_ROOT", str(shipped))
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(
        assets / ".claude" / MCP_REL,
        json.dumps({"mcpServers": {"shared": {"type": "http", "url": "l3"}}}),
    )

    got = install(
        _spec(components=["base"], assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
    )
    assert got.mcp_servers["shared"]["url"] == "l3"
    assert [o.level for o in got.report if "redeclared" in o.message] == ["warn"]


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
    """A subdirectory, not the contents.

    The staged copy under `<zone>/package/` is where components are *installed
    from* and sits several directories from the agent's `cwd`; this is the copy
    it can `ls`. Spilling the files into the workspace root would collide with
    whatever `workspace.cut` cloned there, and a collision between an agent's
    material and its working tree is the one nobody would attribute correctly.
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
    """`interfaces.md` §4.6's frozen call, unchanged.

    Omitting the two new parameters is not a degraded mode with a silent cost:
    no staged package means there is nothing to resolve components against,
    which is exactly the state of a run that configured no package.
    """
    deployed = material.deploy(_spec(), _ZoneAt(str(tmp_path / "zone")))
    assert deployed.environment["CLAUDE_CONFIG_DIR"] == str(tmp_path / "zone" / "config")
    assert deployed.mcp_servers == {}
    assert deployed.tools == ()


# --------------------------------------------------------------------------- #
# The subprocess route, and what it pins


def test_the_recipe_child_runs_this_worktree_and_not_an_installed_one() -> None:
    """The hazard `PYTHONPATH` is pinned against, asserted rather than trusted.

    Measured on this host 2026-09-03: the editable install `agent-sys-helper`
    resolves `env_mgr` to `infera.aiopt.all/agent_sys`, a different worktree, so
    a bare ``python3 -m env_mgr`` runs somebody else's checkout — and mostly
    works, which is the failure class `claude_sdk.py` already refuses for the
    `claude` binary.

    Two halves, and the second is the one that fails if the pin is dropped: the
    derived root is the tree this test is running from, and a child started with
    the environment `_child_env` builds imports `env_mgr` from that same tree.
    """
    ours = Path(agent_assets.__file__).resolve().parents[1]
    assert Path(agent_assets._PACKAGE_ROOT) == ours

    env = agent_assets._child_env(None, "/tmp")
    proc = subprocess.run(
        [sys.executable, "-c", "import env_mgr; print(env_mgr.__file__)"],
        capture_output=True,
        text=True,
        env=env,
        # **`cwd` outside the tree, and without it this test cannot fail.**
        # pytest runs from `agent_sys/`, which puts the right tree on the
        # child's `sys.path` through the cwd entry alone — so with the pin
        # deleted the assertion still passed (verified by mutation, `reviewer`
        # 2026-09-03). From `/tmp` the only route to `env_mgr` is `PYTHONPATH`,
        # and the installed one resolves to a different worktree.
        cwd="/tmp",
    )
    assert Path(proc.stdout.strip()).resolve().parent == ours / "env_mgr"


def test_nothing_here_mutates_the_supervisors_environment(tmp_path: Path) -> None:
    """The window the subprocess route closes, pinned so it cannot reopen.

    The earlier in-process design set ``CLAUDE_CONFIG_DIR`` on `os.environ` for
    the duration of the installs, because `installers/base.py::run_cmd` takes no
    `env=`. `agent/runner.py` is threaded by construction, so two concurrent
    prepares would have taken that value from each other. A child takes an
    environment as an argument; this asserts that is the only place it is set.
    """
    before = dict(os.environ)
    pkg = tmp_path / "staged"
    _write(pkg / "r.yaml", "version: 1\ntarget: {kind: repo, path: /tmp}\nitems: []\n")

    install(
        _spec(recipes=["r.yaml"]),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
    )

    assert os.environ == before


def test_the_shipped_serena_recipe_resolves_by_bare_name_and_parses() -> None:
    """`recipes: [serena]` has to reach `env_mgr/recipes/serena.yaml`.

    The name form is the second candidate `_recipe_paths` tries, so a package
    that ships no file of its own gets this one. Parsed through the CLI rather
    than through `load_recipe`, which this module may not import: a recipe that
    resolves and does not parse would be a `fail` at install time on a real run.
    """
    resolved = agent_assets._recipe_paths(_spec(recipes=["serena"]), staged_package=None)
    assert resolved == [str(Path(agent_assets.__file__).parent / "recipes" / "serena.yaml")]

    proc = subprocess.run(
        [sys.executable, "-m", "env_mgr", "dry-run", resolved[0], "--json", "--path", "/tmp"],
        capture_output=True,
        text=True,
        env=agent_assets._child_env(None, "/tmp"),
    )
    document = json.loads(proc.stdout)
    assert [o["level"] for o in document["outcomes"]], document


# --------------------------------------------------------------------------- #
# `${VAR}` in a component's `.mcp.json`


def test_a_declared_server_expands_against_the_zone_environment(tmp_path: Path) -> None:
    """`${UV_TOOL_BIN_DIR}/serena` is the shape probe E measured working, and it
    is the reason expansion exists at all.

    The uv bin directory cannot be on the agent's `PATH`: `executable_path` is
    derived at prepare step 2 and the directory does not exist until step 6b
    installs into it. Naming the binary absolutely is the only route that needs
    no reordering — so the variable has to resolve, everywhere in the entry.
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
    """**Not a pass-through**, which is the whole reason this is not
    `os.path.expandvars`.

    Left in place, the server is started with a literal `${...}` in its argv, it
    does not start, and the symptom the operator sees is *a server with no
    tools* — no error, no cause. That is `interfaces.md` §4.11's family, and
    this package exists to catch that class rather than ship it.
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_claude: FakeCli
) -> None:
    """Probe F, and it is a correctness requirement rather than hygiene.

    Measured 2026-09-03: a session that loaded a plugin's skill reported its base
    directory as the **marketplace source path** — `settings.json` records
    ``extraKnownMarketplaces: {<mp>: {source: {path: <dir>}}}`` and the plugin is
    read from there at run time. Nothing is copied by the install. So registering
    `agent_sys/components/<name>/.claude/plugins` in place would install cleanly,
    report success, and then fail to load under confinement, because that path is
    outside every grant.
    """
    shipped = tmp_path / "components"
    _write(
        shipped / "base" / ".claude" / "plugins" / ".claude-plugin" / "marketplace.json",
        json.dumps({"name": "mp", "owner": "us", "plugins": [{"name": "p1"}]}),
    )
    _write(shipped / "base" / ".claude" / "plugins" / "p1" / "skills" / "s" / "SKILL.md", "# s")
    monkeypatch.setattr(agent_assets, "COMPONENTS_ROOT", str(shipped))
    config = tmp_path / "config"

    install(
        _spec(components=["base"]),
        staged_package=None,
        config_dir=str(config),
        agent_cli=fake_claude.path,
    )

    registered = config / agent_assets.MARKETPLACES_DIRNAME / "mp"
    assert (registered / "p1" / "skills" / "s" / "SKILL.md").exists()
    (add_line,) = [ln for ln in fake_claude.log.read_text().splitlines() if "marketplace add" in ln]
    assert add_line.endswith(str(registered)), add_line
    # And the source is untouched — the copy is a copy.
    assert (shipped / "base" / ".claude" / "plugins" / "p1").is_dir()


# --------------------------------------------------------------------------- #
# The install report, and the two exported names


def test_the_install_report_is_written_into_the_zone_and_its_path_exported(
    tmp_path: Path,
) -> None:
    """`AGENT_SYS_INSTALL_REPORT` — promised, not discoverable.

    `examples/env_checker`'s `check_capabilities_genuine` decides whether an
    ``unavailable`` verdict is honest by reading this file: an ``unavailable``
    beside a clean install report is a failure. A validator that cannot find it
    fails the report for a reason that has nothing to do with the agent.
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


def test_the_components_root_is_exported_only_when_components_are_declared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The export and the grant fire on the identical condition.

    `paths.py` exports only paths that are granted — the four `*_root` names are
    absent from it because they measured `EACCES`. `agent_sys/components/` is
    outside the zone, so exporting it unconditionally would reintroduce exactly
    that: a body failing on our own instruction. `component_grants` reads the
    same key, so the pair cannot fall out of step.
    """
    shipped = tmp_path / "components"
    _write(shipped / "base" / ".claude" / "settings.json", "{}")
    monkeypatch.setattr(agent_assets, "COMPONENTS_ROOT", str(shipped))

    declared = install(
        _spec(components=["base"]), staged_package=None, config_dir=str(tmp_path / "c1")
    )
    assert declared.env["AGENT_SYS_COMPONENTS_ROOT"] == str(shipped)

    plain = install(_spec(), staged_package=None, config_dir=str(tmp_path / "c2"))
    assert "AGENT_SYS_COMPONENTS_ROOT" not in plain.env


def test_the_grant_and_the_export_agree_on_the_same_condition() -> None:
    """The other half of the pair, from `isolation/policy.py`.

    Asserted together with the export above rather than in a policy test of its
    own, because the property is a *relation between two modules* — either one
    alone can be right while the pair is wrong.
    """
    from env_mgr.isolation.policy import component_grants

    assert component_grants(_spec()) == ()
    assert component_grants(None) == ()
    (granted,) = component_grants(_spec(components=["base"]))
    assert granted.path == agent_assets.COMPONENTS_ROOT
    assert granted.mode is Mode.READ_EXEC


def test_a_tooldef_using_a_dataclass_imports(tmp_path: Path) -> None:
    """**The `sys.modules` registration, and this test is the whole point of it.**

    Measured 2026-09-03 on CPython 3.13 by `pkg-author`, against a real artefact:
    loading a `*.tooldef.py` with `module_from_spec` and calling `exec_module`
    **without** registering the module in `sys.modules` first makes any
    ``@dataclass`` in it raise ``AttributeError: 'NoneType' object has no
    attribute '__dict__'``. `dataclasses._is_type` resolves a string annotation
    through ``sys.modules[cls.__module__]`` and finds `None`.

    `from __future__ import annotations` in the fixture is not incidental — it is
    what makes every annotation a string and sends `dataclasses` down that path.
    Without it the bug does not reproduce, so a fixture that omitted it would be
    a test that passes either way.

    The failure this guards is not the exception. It is that a component whose
    tool module fails to import is indistinguishable, from the model's side,
    from a component that was never installed.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(
        assets / ".claude" / "tools" / "dc.tooldef.py",
        "from __future__ import annotations\n"
        "from dataclasses import dataclass, field\n"
        "from typing import Any, NamedTuple, get_type_hints, Callable\n"
        "@dataclass\n"
        "class ToolDef:\n"
        "    name: str\n"
        "    description: str\n"
        "    schema: dict[str, Any] = field(default_factory=dict)\n"
        "    call: Callable[..., Any] | None = None\n"
        "TOOLS = [ToolDef('probe', 'a probe')]\n",
    )

    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
    )

    assert _installs(got.report) == ["ok"], got.report
    assert [t.name for t in got.tools] == ["probe"]


def test_the_tooldef_module_stays_in_sys_modules_after_the_install(tmp_path: Path) -> None:
    """It is registered **and left registered**, which is a decision.

    Popping it would fix the import and reopen the same hole one step later:
    anything that resolves an annotation lazily — `typing.get_type_hints`, a
    pydantic model, a dataclass built at call time — does the same
    ``sys.modules[cls.__module__]`` lookup *after* `install` has returned, when
    nothing is left to catch the `AttributeError`.
    """
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    path = _write(
        assets / ".claude" / "tools" / "late.tooldef.py",
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class T:\n"
        "    name: str\n"
        "TOOLS = [T('probe')]\n",
    )

    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
    )

    (tool,) = got.tools
    assert sys.modules.get(type(tool).__module__) is not None, (
        f"{path} was popped from sys.modules; a later annotation resolution "
        f"against it would raise where nothing catches it"
    )
    # The deferred resolution itself, run here rather than described.
    from typing import get_type_hints

    assert get_type_hints(type(tool)) == {"name": str}


def test_a_recipe_child_that_overruns_is_killed_and_reported(tmp_path: Path) -> None:
    """The bound that only exists because L1 is a child process.

    `installers/base.py::run_cmd` is `subprocess.run(shell=True)` with **no
    timeout**, so before the subprocess route a networked install that went wrong
    hung `prepare` indefinitely — and `agent-sys --timeout` is a ceiling on the
    whole run, not a cure for one stuck child. The bound here applies to the
    `python -m env_mgr` process, which is the thing `run_cmd` cannot bound from
    inside.

    Driven with a recipe that really sleeps and a timeout of a fraction of a
    second, rather than by patching `subprocess`: what is under test is that a
    child which overruns is *killed and reported*, and a patched `run` would
    assert the test's own mock instead.
    """
    pkg = tmp_path / "staged"
    _write(
        pkg / "slow.yaml",
        "version: 1\n"
        f"target: {{kind: repo, name: t, path: {tmp_path}}}\n"
        "items:\n"
        "  - installer: oneline\n"
        "    importance: suggested\n"
        "    layer: system\n"
        "    name: slow\n"
        "    run: sleep 30\n",
    )

    got = install(
        _spec(recipes=["slow.yaml"]),
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
    """A site with a slow mirror is a fact about that site.

    The default is a measurement — probe D's real `uv tool install` of serena
    finished well inside fifteen minutes on this host from a cold cache — plus
    headroom, not a round number picked defensively.
    """
    assert agent_assets.RECIPE_TIMEOUT_SECONDS == 20 * 60


# --------------------------------------------------------------------------- #
# The pinned CLI


def test_plugin_installs_run_the_pinned_cli_and_never_the_bare_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_claude: FakeCli
) -> None:
    """**Name the binary, do not search for it.**

    Measured on this host 2026-09-03: `agent_cli_grants` grants the CLI's
    *install* directory, not the shim directory holding `~/.local/bin/claude`,
    so the shim is not on the policy-derived `PATH` and a bare `claude`
    resolves to `/usr/local/bin/claude` — an npm-owned **2.1.197** from July —
    while `Prepared.agent_cli` is **2.1.246**. One `CLAUDE_CONFIG_DIR` between
    two builds, and probe A's evidence that plugin installs honour that variable
    was taken on 2.1.246 only.

    The fake `claude` is on `PATH` **as well as** pinned, deliberately: with
    nothing on `PATH` a bare name would answer rc 127 and this assertion would
    pass for the wrong reason.
    """
    shipped = tmp_path / "components"
    _write(
        shipped / "base" / ".claude" / "plugins" / ".claude-plugin" / "marketplace.json",
        json.dumps({"name": "mp", "owner": "us", "plugins": [{"name": "p1"}]}),
    )
    monkeypatch.setattr(agent_assets, "COMPONENTS_ROOT", str(shipped))

    install(
        _spec(components=["base"]),
        staged_package=None,
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


def test_a_component_with_plugins_and_no_pinned_cli_fails_rather_than_guessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_claude: FakeCli
) -> None:
    """Absent CLI: **a `fail` outcome, not a fallback and not a raise.**

    Not a fallback, because running whichever build `PATH` happens to reach is
    the defect above. Not a raise, because it is the same event as
    `claude plugin install` exiting non-zero — the plugin did not install and
    the report says so — and raising would make one component's `plugins/`
    directory fatal to every task that names it.

    `fake_claude` is requested so that a bare `claude` *would* have worked: this
    asserts a refusal to guess, not an absence of options.
    """
    shipped = tmp_path / "components"
    _write(
        shipped / "base" / ".claude" / "plugins" / ".claude-plugin" / "marketplace.json",
        json.dumps({"name": "mp", "owner": "us", "plugins": [{"name": "p1"}]}),
    )
    monkeypatch.setattr(agent_assets, "COMPONENTS_ROOT", str(shipped))

    got = install(
        _spec(components=["base"]),
        staged_package=None,
        config_dir=str(tmp_path / "config"),
        agent_cli=None,
    )

    assert _levels(got.report) == ["fail"]
    assert "pinned no `claude` CLI" in got.report[0].message
    assert not fake_claude.log.exists(), "it fell back to the CLI on PATH"


def test_an_agent_with_no_plugins_and_no_cli_is_a_working_configuration(
    tmp_path: Path,
) -> None:
    """The control, and it is what keeps the refusal above proportionate.

    A machine with no `claude` runs non-AI tasks perfectly well —
    `harness.harness_env`'s own words for the same situation — so *no CLI* must
    only be fatal to the thing that actually needs one.
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
    """The second half of the same defect, and it was the more immediate one.

    `material.deploy` builds its environment from scratch — three zone paths
    plus the harness block, whose `_RESERVED` set excludes `PATH` — so before
    `base_env` the child received **no `PATH` at all**. Measured 2026-09-03:
    `sh -c "uv --version"` then answers `uv: not found` (rc 127) while the same
    call under the policy-derived `PATH` answers `uv 0.11.24`. A recipe would
    have failed naming its toolchain rather than naming the cause.
    """

    class _Zone:
        root = str(tmp_path / "zone")

    # **The positive half, and it is the one that was missing.** This test used
    # to assert only that `base_env` is *not* echoed back into the returned
    # mapping — true whether it reaches the child or is discarded on the floor —
    # plus a property of `_child_env`, which was never the broken half. Dropping
    # `**(base_env or {})` from `material.py` is the whole of defect 2 and left
    # 46 tests passing (`reviewer`'s mutation M7).
    #
    # So the assertion is now on **the mapping `install` was handed**, captured
    # at the seam. `test_a_recipe_item_runs_a_binary_reachable_only_through_base_env`
    # is the same claim proved end to end; this one localises it, so a failure
    # says *which* of the two halves broke.
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
    """**The enumeration, asserted as an enumeration.**

    This is the test that was missing. `_install_tree` copied `skills/` and
    `plugins/` and nothing else, so `hooks/` and `servers/` were named by
    consumers and placed by nobody — measured by `reviewer` 2026-09-03 against
    the real package: `settings.json` in the zone named
    `$CLAUDE_CONFIG_DIR/hooks/envchk_session_start.py` with the script absent.

    So the assertion is over the **whole directory listing** rather than over
    the two members somebody remembered. A `.claude/` tree that grows a member
    Claude Code invents next year fails here on the day it is added unless it is
    placed or explicitly excepted, which is what makes `_NOT_PLACED` a closed
    answer rather than a closed-looking list.
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
    """**The general assertion, over all entries rather than today's two.**

    The failure it catches is the one `_expand`'s docstring claims to make
    impossible and does not: the variable resolves, the file is absent, the
    report says `ok  1 external MCP server(s)`, and the symptom the operator
    sees is *a server with no tools* — no error, no cause.

    Both routes are covered. A component's own `.mcp.json` naming
    `${CLAUDE_CONFIG_DIR}/servers/…` is only true if `servers/` was placed; a
    bundled `tools/*.mcp.py` is only true if it is registered at its placed path
    rather than its source.
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
    """**Checked before the copy, not after it.**

    `market` is `manifest["name"]`, an author-controlled JSON string joined
    straight into a path. Measured by `reviewer` 2026-09-03 with
    `"name": "../../../ESCAPED"`: the tree **was written outside the zone** and
    only then did the `contained()` refusal fire. On this host that is not
    hypothetical — the repository's no-delete rule was bought by a real
    out-of-tree write.

    The assertion is therefore about the filesystem, not only about the outcome:
    nothing may appear at the escaped location.
    """
    shipped = tmp_path / "components"
    escape = tmp_path / "ESCAPED"
    _write(
        shipped / "base" / ".claude" / "plugins" / ".claude-plugin" / "marketplace.json",
        json.dumps({"name": "../../../ESCAPED", "owner": "us", "plugins": []}),
    )
    monkeypatch.setattr(agent_assets, "COMPONENTS_ROOT", str(shipped))
    config = tmp_path / "deep" / "zone" / "config"

    got = install(
        _spec(components=["base"]),
        staged_package=None,
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
    """F-D18 applied to the consumer, not only to the loader.

    `spec_loader.AssetIndex.resolve_folder` already records that
    ``Path(staged) / "/abs"`` is ``/abs``; the lesson was applied to what the
    loader *emits* and not to what reads it. It matters most here because
    `_tooldefs` imports from this directory **into the supervisor**, and the
    module docstring's narrowing is that it never comes from outside the staged
    copy.
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
    """The likelier author mistake of the two, and it was the unreported one.

    A cross-*tree* collision already warned; `.mcp.json` declaring `x` beside
    `tools/x.mcp.py` in **one** component silently overwrote, leaving the author
    with a server they did not write and no message.
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
    """`<config>/plugins/` belongs to Claude Code, not to us.

    Measured (probe A, `claude` 2.1.246): `claude plugin install` writes that
    directory itself — `installed_plugins.json`, `known_marketplaces.json`,
    `marketplaces/`, `cache/`. A component's **source** marketplace copied onto
    that name would sit among the CLI's own bookkeeping, put there by us before
    the CLI writes it.

    So this is a namespace collision rather than a naming preference, and it is
    the reason `plugins/` is *relocated* in `_NOT_PLACED` rather than simply
    placed like every other member. The `_NOT_PLACED` table states it; this is
    what keeps it true.
    """
    shipped = tmp_path / "components"
    _write(
        shipped / "base" / ".claude" / "plugins" / ".claude-plugin" / "marketplace.json",
        json.dumps({"name": "mp", "owner": "us", "plugins": []}),
    )
    _write(shipped / "base" / ".claude" / "plugins" / "SENTINEL", "the component's own copy")
    monkeypatch.setattr(agent_assets, "COMPONENTS_ROOT", str(shipped))
    config = tmp_path / "config"

    install(
        _spec(components=["base"]),
        staged_package=None,
        config_dir=str(config),
        agent_cli=fake_claude.path,
    )

    assert (config / agent_assets.MARKETPLACES_DIRNAME / "mp" / "SENTINEL").exists()
    assert not (config / "plugins" / "SENTINEL").exists(), (
        "the component's marketplace was copied into the directory the CLI owns"
    )


def test_a_tooldef_is_imported_from_the_zone_copy_not_the_component_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same rule as the bundled MCP server, one function over.

    For **L2** the source is `agent_sys/components/<name>/.claude/tools/…`, so
    importing it read the repository rather than the copy this attempt was
    pinned to — the bare-`claude` defect's class, two consumers with one of them
    reading a path the run does not own. It also wrote `__pycache__` into the
    repository during `prepare`.

    Asserted through the module's own `__file__`, which is the only thing that
    can tell the two copies apart once the tools are loaded.
    """
    shipped = tmp_path / "components"
    _write(
        shipped / "base" / ".claude" / "tools" / "t.tooldef.py",
        "class T:\n    def __init__(self, name): self.name = name\nTOOLS = [T('probe')]\n",
    )
    monkeypatch.setattr(agent_assets, "COMPONENTS_ROOT", str(shipped))
    config = tmp_path / "config"

    got = install(_spec(components=["base"]), staged_package=None, config_dir=str(config))

    (tool,) = got.tools
    loaded = Path(sys.modules[type(tool).__module__].__file__)
    assert loaded == config / "tools" / "t.tooldef.py", loaded
    assert not (shipped / "base" / ".claude" / "tools" / "__pycache__").exists(), (
        "importing the source wrote __pycache__ into the components registry"
    )


def test_two_components_shipping_tooldefs_do_not_double_register(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Why the enumeration stays on the **source** tree.

    `<config>/tools/` accumulates every level's files as each is placed, so
    listing *it* would re-import the earlier component's module under the later
    component's name and register its tools twice. The source names exactly one
    component's files; the placed path is only where each is read from.
    """
    shipped = tmp_path / "components"
    for name, tool in (("a", "alpha"), ("b", "beta")):
        _write(
            shipped / name / ".claude" / "tools" / f"{name}.tooldef.py",
            f"class T:\n    def __init__(self, name): self.name = name\nTOOLS = [T({tool!r})]\n",
        )
    monkeypatch.setattr(agent_assets, "COMPONENTS_ROOT", str(shipped))

    got = install(
        _spec(components=["a", "b"]),
        staged_package=None,
        config_dir=str(tmp_path / "config"),
    )

    assert sorted(t.name for t in got.tools) == ["alpha", "beta"]


def test_a_recipe_item_runs_a_binary_reachable_only_through_base_env(
    tmp_path: Path,
) -> None:
    """Defect 2, proved end to end rather than at the seam.

    The measured symptom was `sh -c "uv --version"` answering rc 127
    `uv: not found`, because `material.deploy` builds its mapping from scratch
    and `harness._RESERVED` excludes `PATH`, so the recipe child had none. The
    honest test of that is a recipe item invoking a binary that exists **only**
    in a directory named by `base_env`'s `PATH` — if the value does not reach
    the child, the item cannot run, exactly as `uv` could not.

    Driven through the real `python -m env_mgr` child and the real `oneline`
    installer, so what is under test is the whole chain `prepare` → `deploy` →
    `_child_env` → `subprocess.run(env=)` → `run_cmd`'s shell, rather than any
    one link of it.
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
        "    layer: system\n"
        "    name: only-here\n"
        "    run: envchk-only-here\n",
    )

    class _Zone:
        root = str(tmp_path / "zone")

    material.deploy(
        _spec(recipes=["r.yaml"]),
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_claude: FakeCli, market: str
) -> None:
    """**"Is it a single name" is a different question from "does it stay inside".**

    Measured by `reviewer`: with ``name: ".."`` the containment check normalises
    ``marketplaces/..`` to ``.`` and returns `config_dir` **unchanged**, so the
    component's `plugins/` was emptied into the config root and
    ``claude plugin marketplace add <config>`` registered the entire zone
    configuration directory as a marketplace. The trailing `contained()` passed,
    because `config_dir` is contained in itself.

    Nothing escaped the zone — so this is not the earlier defect returning. What
    it defeats is the decision `_NOT_PLACED`'s `plugins` row and
    `MARKETPLACES_DIRNAME` exist to make: keep a component's marketplace off the
    harness's namespace. `".."` landed it one level *above* that namespace, on
    top of `settings.json` and everything `_place_tree` had just written.

    The empty case is covered by the earlier "declares no 'name'" branch and is
    parametrised here so that the two guards cannot both be removed and leave a
    hole between them.
    """
    shipped = tmp_path / "components"
    _write(
        shipped / "base" / ".claude" / "plugins" / ".claude-plugin" / "marketplace.json",
        json.dumps({"name": market, "owner": "us", "plugins": [{"name": "p1"}]}),
    )
    _write(shipped / "base" / ".claude" / "plugins" / "SENTINEL", "component copy")
    _write(shipped / "base" / ".claude" / "settings.json", json.dumps({"model": "m"}))
    monkeypatch.setattr(agent_assets, "COMPONENTS_ROOT", str(shipped))
    config = tmp_path / "config"

    got = install(
        _spec(components=["base"]),
        staged_package=None,
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


def test_two_components_shipping_the_SAME_tooldef_filename_stay_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression `99d3aea` introduced, and the case its sibling cannot see.

    `test_two_components_shipping_tooldefs_do_not_double_register` uses
    **distinct** filenames, so it is blind to this: moving the *load* to the
    placed copy took the module name with it, and the placed path is
    ``<config>/tools/<basename>`` — identical for every component shipping the
    same file name.

    Measured before the fix: one module name twice, the second import replacing
    the first in `sys.modules`, and `get_type_hints` on the **first**
    component's tool raising ``NameError: name 'AlphaArgs' is not defined``
    because its annotations resolved against the other component's namespace.
    Four `ok`s in the report and nothing said anything — exactly the state the
    `sys.modules` registration exists to prevent, one component later.

    `from __future__ import annotations` plus a forward-referenced annotation is
    what makes the corruption observable; without them both modules would look
    fine and the test would pass either way.
    """
    shipped = tmp_path / "components"
    for name, tool in (("a", "alpha"), ("b", "beta")):
        _write(
            shipped / name / ".claude" / "tools" / "util.tooldef.py",
            "from __future__ import annotations\n"
            "from dataclasses import dataclass\n"
            f"class {tool.capitalize()}Args: pass\n"
            "@dataclass\n"
            "class T:\n"
            "    name: str\n"
            f"    args: {tool.capitalize()}Args | None = None\n"
            f"TOOLS = [T({tool!r})]\n",
        )
    monkeypatch.setattr(agent_assets, "COMPONENTS_ROOT", str(shipped))

    got = install(
        _spec(components=["a", "b"]),
        staged_package=None,
        config_dir=str(tmp_path / "config"),
    )

    assert sorted(t.name for t in got.tools) == ["alpha", "beta"]
    modules = [type(t).__module__ for t in got.tools]
    assert len(set(modules)) == 2, f"one module name for two components: {modules}"
    for tool in got.tools:
        # Resolves each class's annotations through `sys.modules[__module__]`,
        # which is the lookup that returned the wrong component's namespace.
        assert list(get_type_hints(type(tool))) == ["name", "args"], tool.name


def test_one_component_replacing_anothers_placed_file_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The overwrite underneath that bug, which was silent.

    Levels install in order and `copy_out` merges, so a member two components
    both ship ends up holding only the later one's bytes — which is how one
    component's artefact can be absent from the zone while its own report says
    `ok`. `_mcp_servers` already warns about this for a server *name*; it was
    unreported for a *file*.

    Precedence is unchanged: later wins, which is L1 → L2 → L3's rule. Only the
    silence changed. And the unit is a **file**: two components both shipping
    `skills/` must not warn unless they ship the same skill.
    """
    shipped = tmp_path / "components"
    _write(shipped / "a" / ".claude" / "skills" / "shared" / "SKILL.md", "# from a")
    _write(shipped / "a" / ".claude" / "skills" / "only-a" / "SKILL.md", "# a only")
    _write(shipped / "b" / ".claude" / "skills" / "shared" / "SKILL.md", "# from b")
    monkeypatch.setattr(agent_assets, "COMPONENTS_ROOT", str(shipped))
    config = tmp_path / "config"

    got = install(_spec(components=["a", "b"]), staged_package=None, config_dir=str(config))

    (warned,) = [o for o in got.report if o.level == "warn"]
    assert "component 'b' replaces 1 already-placed file(s)" in warned.message
    assert warned.details["files"] == [os.path.join("shared", "SKILL.md")]
    assert (config / "skills" / "shared" / "SKILL.md").read_text() == "# from b"
    assert (config / "skills" / "only-a" / "SKILL.md").read_text() == "# a only"


def test_components_that_share_no_file_do_not_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control for the warning above. Two components both shipping
    `skills/` is the ordinary case and must stay quiet — a warning that fires on
    every second component is one nobody reads."""
    shipped = tmp_path / "components"
    _write(shipped / "a" / ".claude" / "skills" / "one" / "SKILL.md", "# a")
    _write(shipped / "b" / ".claude" / "skills" / "two" / "SKILL.md", "# b")
    monkeypatch.setattr(agent_assets, "COMPONENTS_ROOT", str(shipped))

    got = install(
        _spec(components=["a", "b"]),
        staged_package=None,
        config_dir=str(tmp_path / "config"),
    )

    assert [o.level for o in got.report] == ["ok", "ok"], got.report


@pytest.mark.parametrize("where", ["toplevel.txt", "nested/inner.txt"])
def test_a_symlink_in_a_claude_tree_is_resolved_at_every_depth(tmp_path: Path, where: str) -> None:
    """**The same input must not give two answers depending on depth.**

    `copy_out`'s default is asymmetric — measured, a top-level symlink goes
    through `copy2` and is dereferenced, one nested inside a directory goes
    through `copytree(symlinks=True)` and is preserved. `_place_tree` passes
    `dereference=True` so both resolve.

    Resolving is the decision rather than preserving, and the measurement picks
    it: `contained(<zone>/link -> /outside, <zone>)` is **False**, so a preserved
    link is a path the confined session cannot follow while the report says it
    was placed — probe F's installs-cleanly-never-loads shape again. *Copy into
    the zone, do not reference out of it* was already the rule for the
    marketplace, and a preserved link is a reference out of the zone.

    Parametrised over both depths precisely because the bug was that they
    differed; a test at one depth would have passed throughout.
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
    """The closure check, over shapes rather than over remembered names.

    **Written by `reviewer` and adopted verbatim but for the symlink note**,
    because a fixture I build myself can only surprise me as far as my own
    imagination went — and four of these are shapes I did not think of.

    `test_every_member_of_a_claude_tree_is_placed_except_the_named_exceptions`
    asserts the rule with members Claude Code might add. This asserts it with
    members that are not *directories* at all, which is the other axis a
    hand-written fixture misses: a top-level file, an empty directory, a
    symlink, and a `.mcp.json` nested below the top level — the last of which
    must be **placed as data** rather than read, because only the top-level one
    is an interface document.
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
    # A symlink is dereferenced. **`reviewer` wrote this assertion to pin
    # *current* behaviour and flagged that nothing had decided it; it has since
    # been decided** — `_place_tree` passes `dereference=True` because a
    # preserved link out of the zone measurably fails `contained`, so the
    # confined session cannot follow it. See
    # `test_a_symlink_in_a_claude_tree_is_resolved_at_every_depth`, which covers
    # the out-of-tree case and both depths; this one keeps the in-tree link
    # honest inside the closure fixture.
    assert (config / "linked.txt").is_file()
    assert not (config / "linked.txt").is_symlink()


# --------------------------------------------------------------------------- #
# The install report names what it produced, on all three MCP routes


def test_all_three_mcp_routes_record_a_name_in_the_report(tmp_path: Path) -> None:
    """**Four rows of one rule, not one row plus two special cases.**

    `Prepared.mcp_servers` goes supervisor → backend and is written down nowhere
    else, so the install report is the only artefact that can carry evidence of
    *which* servers a run got. Run 1's report carried a count for the external
    route, a path for the bundled one, and for the in-process route nothing at
    all — so a consumer comparing declared capabilities against what installed
    could check one of three, and only by parsing a message string.

    Each route is asserted here for a **recoverable name**, not for a message
    that happens to contain one. Removing any of the three keys fails this.
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
    _write(
        assets / ".claude" / "tools" / "t.tooldef.py",
        "class T:\n"
        "    def __init__(self, name): self.name = name\n"
        "TOOLS = [T('envchk_echo_token')]\n",
    )

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

    # In-process: the TOOL name is the load-bearing half. The server name is a
    # constant every run has, so a check against it alone could not fail.
    inproc = details("in-process tool(s)")
    assert inproc["tools"] == ["envchk_echo_token"]
    assert inproc["server"] == agent_assets.IN_PROCESS_SERVER

    # Every server the run produced is recoverable from the report alone, which
    # is the property a consumer actually needs.
    recorded = set(details("external MCP server(s)")["names"]) | {
        details("bundled MCP server")["server"]
    }
    assert recorded == set(got.mcp_servers), (
        "a server reached the backend that the install report does not name"
    )


def test_a_tool_with_no_name_is_recorded_as_such_rather_than_crashing(
    tmp_path: Path,
) -> None:
    """A `ToolDef` here is duck-typed — package-authored, and this module does
    not define the class. One with no `.name` cannot be addressed by the model
    at all; the report says so instead of raising while building it."""
    pkg = tmp_path / "staged"
    assets = _package(pkg)
    _write(
        assets / ".claude" / "tools" / "t.tooldef.py",
        "class T: pass\nTOOLS = [T()]\n",
    )

    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
    )

    (inproc,) = [o for o in got.report if "in-process tool(s)" in o.message]
    assert inproc.details["tools"] == ["<unnamed>"]
