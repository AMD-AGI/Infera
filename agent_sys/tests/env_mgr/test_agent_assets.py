# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`env_mgr.agent_assets` — the three levels of per-agent component.

**Nothing here runs the real `claude` binary.** A fake one goes on `PATH` per
test, which is what makes the *failing* install case testable at all: a machine
with no `claude` and a machine whose `claude plugin install` exits 1 are two
different outcomes and this module reports them differently, so a suite that
could only produce the first would leave the second unmeasured.

The measurements this file encodes were taken 2026-09-03 against Claude Code on
this machine, and are recorded in `agent_assets.py`'s own docstrings:

- ``claude plugin marketplace add`` / ``install`` respect ``CLAUDE_CONFIG_DIR``
  fully — ``~/.claude.json``'s md5 was unchanged after both;
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
from typing import Any

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


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A `claude` on `PATH` that records its argv and succeeds.

    Records rather than merely succeeds, because two of the assertions below are
    about *what was invoked* — ``marketplace add`` before ``install``, and one
    ``install`` per plugin in the manifest — and a stub that only returned 0
    could not tell a correct call from no call at all.
    """
    binroot = tmp_path / "fakebin"
    binroot.mkdir()
    log = tmp_path / "claude.log"
    script = binroot / "claude"
    script.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> {log}\n'
        f'printf "CLAUDE_CONFIG_DIR=%s\\n" "$CLAUDE_CONFIG_DIR" >> {log}\n'
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{binroot}{os.pathsep}{os.environ['PATH']}")
    return log


@pytest.fixture
def failing_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A `claude` that exists and exits 1. The case a missing binary cannot cover."""
    binroot = tmp_path / "failbin"
    binroot.mkdir()
    script = binroot / "claude"
    script.write_text("#!/bin/sh\necho 'boom' >&2\nexit 1\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{binroot}{os.pathsep}{os.environ['PATH']}")


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_claude: Path
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
    )

    # L3 wins the collision; L2's untouched key survives the merge.
    assert got.settings == {"hooks": {"PreToolUse": ["a"]}, "model": "from-l3"}
    on_disk = json.loads((config / "settings.json").read_text())
    assert on_disk == got.settings

    invoked = fake_claude.read_text().splitlines()
    assert invoked[0].startswith("plugin marketplace add ")
    assert f"CLAUDE_CONFIG_DIR={config}" in invoked
    assert [line for line in invoked if line.startswith("plugin install")] == [
        "plugin install p1@mp",
        "plugin install p2@mp",
    ]


def test_a_failing_plugin_install_is_a_named_outcome_and_not_a_silent_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing_claude: None
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
        _spec(components=["base"]), staged_package=None, config_dir=str(tmp_path / "config")
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
    bundled = _write(assets / ".claude" / "tools" / "envchk.mcp.py", "# a server\n")

    got = install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
    )
    assert got.mcp_servers["weather"] == {"type": "http", "url": "http://x"}
    assert got.mcp_servers["envchk"]["type"] == "stdio"
    assert got.mcp_servers["envchk"]["args"] == [str(bundled)]


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
    assert _levels(got.report) == ["fail"]
    assert "did not import" in got.report[0].message
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
    assert got.report == ()
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


def test_claude_config_dir_is_restored_after_the_installs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cost of reusing `installers/base.py::run_cmd` unchanged.

    It builds subprocesses from `os.environ` and takes no environment argument,
    so the variable is set process-wide for the duration. That is a real window
    in a threaded supervisor and `_claude_config`'s docstring says so; what this
    pins is the half that *is* under this module's control — the value the
    supervisor had is the value it has afterwards, whether it was set or not.
    """
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/operators/own")
    pkg = tmp_path / "staged"
    _package(pkg)
    install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
    )
    assert os.environ["CLAUDE_CONFIG_DIR"] == "/operators/own"

    monkeypatch.delenv("CLAUDE_CONFIG_DIR")
    install(
        _spec(assets="assets/forge.agent"),
        staged_package=str(pkg),
        config_dir=str(tmp_path / "config"),
    )
    assert "CLAUDE_CONFIG_DIR" not in os.environ


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_claude: Path
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

    install(_spec(components=["base"]), staged_package=None, config_dir=str(config))

    registered = config / agent_assets.MARKETPLACES_DIRNAME / "mp"
    assert (registered / "p1" / "skills" / "s" / "SKILL.md").exists()
    (add_line,) = [ln for ln in fake_claude.read_text().splitlines() if "marketplace add" in ln]
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
        "from typing import Any, Callable\n"
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

    assert _levels(got.report) == ["ok"], got.report
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
