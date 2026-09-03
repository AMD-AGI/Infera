# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-agent components, at three levels, into one Claude Code configuration.

`material.py` deploys what `agent` spec §3.1 declares as three lists of *file
paths* — ``rules``, ``hooks``, ``skills``. That shape stops at the first real
component: a skill is a **directory**, a plugin marketplace is a directory of
directories, an MCP server is a process to register rather than a file to place.
This module is the route for those, and it keeps `material.py`'s rule — *a file
is placed, not read* — for everything except the four documents whose contents
are an interface (`settings.json`, `.mcp.json`, `marketplace.json`, and a
`*.tooldef.py`'s ``TOOLS``).

## The three levels

| level | what | declared how |
|---|---|---|
| L1 | industry components: serena, a marketplace plugin, an apt/pip tool | ``recipes: [...]`` -> an `env_mgr` recipe YAML |
| L2 | components this repository ships | ``components: [...]`` -> `agent_sys/components/<name>/` |
| L3 | components one task package carries for one agent | **undeclared** — `<agent assets>/.claude/` |

**L2 and L3 have the same on-disk shape**, so one installer serves both and
promoting a component is moving a directory. **L3 is undeclared** because the
declaration would be a second statement of what the directory already says, and
the two would drift; `components/README.md` states the same contract from the
author's side.

## Order, and what it buys

L1, then L2, then L3 — so that when two levels bind the same name, **the
package's own copy wins**. That is `material.deploy`'s existing precedence
carried to a second kind of material: *an author saying so outranks a default*.

One ordering constraint cuts across it and is **measured**, 2026-09-03: ``claude
plugin marketplace add`` and ``claude plugin install`` *merge* into an existing
``settings.json`` rather than clobbering it — a hand-written ``hooks`` key
survived a subsequent install, which only added ``enabledPlugins`` and
``extraKnownMarketplaces``. Merging is not commutative with creating: the file
has to exist first. So the settings document is assembled from every level and
**written before any install runs**, and this module writes it rather than
returning it for `material.py` to write. A caller cannot write it at the right
moment without knowing which levels resolved, and that is this module's
knowledge (`engineer_principle.md` §1 — one writer, and it is the one that owns
the ordering).

## L1 runs the shipped machinery as a SUBPROCESS, and that is a contract

    <sys.executable> -m env_mgr <stage> <recipe> --json

**This is a compatibility surface, not an implementation detail.** `env-mgr`'s
argument surface — the four `STAGES` sub-parsers, `--json`, and `render_json`'s
``{status, outcomes: [{level, message, details}]}`` — is depended on by this
module, and the next reader must not "simplify" it into an import. Four reasons
it is a subprocess, and the last two would not be recovered by an import:

1. **The decoupling wall stays literally true.** `env_mgr` spec §9 says *nothing
   new imports the installer machinery*, and `tests/env_mgr/test_imports.py`
   checks it with `cli.py` as the single named exception. An import here would
   mean amending the invariant to fit the feature.
2. **It is not the `importlib.import_module` dodge.** A lazy import hides an edge
   that still exists in-process; a subprocess genuinely has no edge, and the
   coupling becomes a CLI contract — weaker, and visible to a reader.
3. **A child takes `env=`; this process's `os.environ` does not.**
   `installers/base.py::run_cmd` builds its subprocesses from `os.environ` and
   takes no environment argument, so an in-process call could only set
   ``CLAUDE_CONFIG_DIR`` and the ``UV_*`` roots by mutating the supervisor's own
   environment. `agent/runner.py` is threaded by construction, so two concurrent
   prepares would see each other's values — a silent wrong-value bug. Handing a
   child an environment removes the window entirely and changes nothing in the
   shipped machinery.

4. **A child can be bounded and an in-process call cannot.** `run_cmd` has no
   timeout, so a networked L1 install that goes wrong hangs `prepare` for ever —
   and `agent-sys --timeout` is a ceiling on the whole run, not a cure for a
   stuck child. `subprocess.run(timeout=)` here bounds the
   ``python -m env_mgr`` process, which is precisely what `run_cmd` cannot bound
   from inside. See `RECIPE_TIMEOUT_SECONDS` for the value and why it is a
   parameter.

**`PYTHONPATH` is pinned from this module's own location, and that is not
belt-and-braces.** Measured on this host 2026-09-03: the editable install
`agent-sys-helper 0.1.0` resolves `env_mgr` to a *different worktree*
(`infera.aiopt.all`), so a bare ``python3 -m env_mgr`` runs somebody else's
checkout — and mostly works, which is the failure class `claude_sdk.py:399-440`
already documents for the `claude` CLI. `_PACKAGE_ROOT` is
``Path(__file__).resolve().parents[1]``, so the child runs the tree that spawned
it whatever is installed globally.

**The `UV_*` roots are a parameter, not a constant.** Measured (probe D):
``uv tool install`` with no environment writes ``~/.local/share/uv``,
``~/.local/bin`` and ``~/.cache/uv`` — host state on a shared box — and it
*succeeds* while doing it. A recipe that needs them declares them through the
agent spec's `env` block, which is in `environ` by the time this runs.

## What is returned rather than applied

`AgentMaterial` — because this module does not own the executor's process, does
not own `Prepared`, and does not own the backend's options. `material.deploy`
already returns an environment for that reason and this follows it.

## Two things this module refuses to do quietly

**Declared and absent is an error.** `material.py:62-86` settled that, on a
measured failure: a skipped copy is invisible at every point where anyone could
act on it, and the agent meets the absence hours later as ``Unknown skill`` from
inside its own session with nothing naming the cause. Undeclared and absent is
simply absent — that is L3's normal shape.

**A failed install is a named `InstallOutcome`, never a silent skip.** Every
subprocess's return code and output lands in `AgentMaterial.report`.

## The one place package-authored code is imported

A ``*.tooldef.py`` is executed in **the supervisor's own process** to read its
module-level ``TOOLS``. That is not a sandboxed act and nothing here pretends it
is: the supervisor holds the API credentials the whole confinement design exists
to keep away from a task body. Three narrowings, and they are the whole defence:
it is done only for that exact suffix, only under ``tools/`` of a resolved
component, and only from the **staged** package or this repository's own
`components/` — never from `Context.package`, which is the operator's live
checkout. An in-process tool is worth this because the alternative measured
shape (a stdio MCP server) costs a subprocess per tool and cannot share the
supervisor's objects; a component that does not need that should ship a
``*.mcp.py`` instead.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

from env_mgr import paths
from env_mgr.fs.layout import copy_out
from env_mgr.fs.path import contained
from env_mgr.protocols import PrepareRefused

#: The directory holding the `env_mgr` package — i.e. what has to be on a child
#: process's `PYTHONPATH` for ``-m env_mgr`` to reach **this** tree.
#:
#: **Derived from this file, never from the environment, and that is measured
#: rather than defensive.** On the development host the editable install
#: ``agent-sys-helper`` resolves `env_mgr` and `cli` to a *different worktree*
#: (2026-09-03: ``python3 -c "import cli.main"`` landed in
#: ``infera.aiopt.all/agent_sys`` while the work was in
#: ``infera.aiopt.real.task_package``). A bare ``-m env_mgr`` would therefore run
#: somebody else's checkout, succeed, and report outcomes about the wrong code —
#: `interfaces.md` §4.11's family, and the same shape as the bundled-vs-`PATH`
#: `claude` binary that `agent/backends/claude_sdk.py:399` refuses. Deriving the
#: path from ``__file__`` means the child runs whichever tree spawned it.
_PACKAGE_ROOT = str(pathlib.Path(__file__).resolve().parents[1])

__all__ = [
    "CLAUDE_DIRNAME",
    "COMPONENTS_ROOT",
    "MARKETPLACE_MANIFEST",
    "SETTINGS_FILENAME",
    "RECIPE_TIMEOUT_SECONDS",
    "AgentMaterial",
    "InstallOutcome",
    "install",
]

#: Claude Code's own directory name, inside a component. Ours nowhere: the
#: contract is *place a file in the harness's layout*, and a name of our own
#: would be a format to convert between.
CLAUDE_DIRNAME = ".claude"

#: The components this repository ships. **A repository path, not a
#: configurable root**: ``components: [<name>]`` takes a bare name precisely so
#: that a task package cannot point L2 anywhere, and a knob here would give back
#: what the name shape removed.
COMPONENTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "components")

#: Written into the zone's config directory. Nothing else writes this file.
SETTINGS_FILENAME = "settings.json"

#: What makes a directory a local plugin marketplace. Measured 2026-09-03:
#: ``claude plugin validate <dir>`` requires it, and it carries
#: ``{name, owner, plugins: [{name, source, description}]}``.
MARKETPLACE_MANIFEST = os.path.join(".claude-plugin", "marketplace.json")

#: Where an external MCP declaration lives inside a component. **Our normalised
#: spot rather than the harness's**, and the one place this module names a path
#: Claude Code does not: measured 2026-09-03, ``CLAUDE_CONFIG_DIR`` relocates
#: ``.claude.json`` along with everything else, but MCP still reaches the model
#: through the SDK's typed ``mcp_servers`` option rather than through that file.
#: So a component declares servers here and this module carries them to the
#: backend as data.
MCP_FILENAME = ".mcp.json"

#: Subdirectory of a component holding executable tool definitions.
TOOLS_DIRNAME = "tools"

#: A stdio MCP server the component ships; auto-registered under its stem.
MCP_SUFFIX = ".mcp.py"

#: A module exposing ``TOOLS: list[ToolDef]``, imported into the supervisor.
TOOLDEF_SUFFIX = ".tooldef.py"

#: What a ``*.tooldef.py`` must expose. A module-level name rather than a
#: factory call, so that reading the file tells a reviewer what it publishes
#: without running it in their head.
TOOLS_ATTR = "TOOLS"

#: The install report's filename inside ``<zone>/logs``.
INSTALL_REPORT_FILENAME = "agent_assets.install.json"

#: Where a component's marketplace is copied to before it is registered. Under
#: the zone's config directory, and probe F is why it is copied at all rather
#: than registered where it lies — see `_install_plugins`.
MARKETPLACES_DIRNAME = "marketplaces"

#: `bootstrap`, not `install`. `runner.run` defines that stage as install *then*
#: bootstrap, so a recipe item needing a post-install step — a marketplace add, a
#: login, a cache warm — is usable when `_run_recipe` returns. Under `install` it
#: would be installed and not usable, and the failure would surface inside the
#: agent's session rather than in the report.
_RECIPE_STAGE = "bootstrap"

#: ``${NAME}`` only. See `_expand` for why the bare ``$NAME`` form is excluded.
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: How long one L1 recipe child may run, in seconds. **A default, and `install`
#: takes it as a parameter** — a site with a slow mirror is a fact about that
#: site, not about this code.
#:
#: **This bound exists only because L1 is a child process**, and it is the third
#: independent argument for that route. `installers/base.py::run_cmd` is
#: ``subprocess.run(shell=True)`` with **no timeout**, so a networked install
#: that goes wrong hangs `prepare` for ever; `agent-sys --timeout` is a ceiling
#: on the whole run and not a cure for a stuck child. In-process there was
#: nothing to be done without changing the shipped machinery. Out of process the
#: timeout applies to the ``python -m env_mgr`` process rather than to the shell
#: it spawns internally, which is exactly the thing `run_cmd` cannot bound
#: itself.
#:
#: **Twenty minutes, chosen against a measurement rather than defensively.**
#: Probe D's real ``uv tool install "git+https://github.com/oraios/serena"``
#: completed well inside fifteen minutes on this host from a **cold** cache, and
#: a warm-cache re-run pays only the `check`. So this is that measurement plus
#: headroom, not a round number.
RECIPE_TIMEOUT_SECONDS = 20 * 60

#: `_run_cmd`'s return code for a killed child. Negative so that it cannot
#: collide with an exit status a real child produced.
_TIMED_OUT = -1


class InstallOutcome(NamedTuple):
    """**The receiving type for `report.render_json`'s wire format.**

    That sentence is the definition and not a description, because the obvious
    reading of this class is the wrong one. It has the same three fields as
    `env_mgr.outcome.Outcome` and it is **deliberately not that class, and must
    not later be "unified" with it.** `outcome` is below the decoupling wall
    (spec §9); this module is above it and imports nothing from there, which is
    the whole reason L1 is a subprocess rather than a call. The data arrives as
    JSON across a process boundary — ``{level, message, details}``, exactly what
    `render_json` emits — so importing the machinery's class to rebuild it would
    reintroduce the edge purely for nominal identity. A three-field record on
    each side of a process boundary is a DTO, not a duplicate.

    **`env_mgr.outcome.LEVELS` remains the owner of the vocabulary** — ``ok`` /
    ``info`` / ``warn`` / ``fail`` — and `level` carries one of its values
    verbatim. Not redefined here and not validated here: the CLI's own values
    arrive, a second vocabulary would make a recipe's `warn` and a plugin's
    `warn` different words for one thing, and rejecting an unrecognised level
    would discard the report saying something went wrong in a way nobody
    anticipated.
    Not validated: this is a value carrying what a child reported, and rejecting
    an unrecognised level would discard the report that says something went
    wrong in a way we did not anticipate.
    """

    level: str
    message: str
    #: **No default, and that is `prepare.Prepared`'s rule rather than strictness
    #: for its own sake.** A mutable default on a `NamedTuple` is one object
    #: shared by every instance, which is one `details[...] = x` away from one
    #: install's evidence appearing in another's. Every construction here passes
    #: a fresh mapping.
    details: dict[str, Any]


class AgentMaterial(NamedTuple):
    """Everything the three levels produced, as a value.

    Five fields because they have five different destinations, and collapsing
    any two would make a caller unpack what this module already separated:
    `env` joins `Prepared.environment`, `mcp_servers` and `tools` cross to the
    backend through `Assignment`, `settings` is a **report of what was written**
    to ``<config>/settings.json``, and `report` is the per-install record.

    `settings` is a report and not an instruction. The file is written here —
    see the module docstring's ordering constraint — and this carries the merged
    document so that a caller and a test can see what landed without re-reading
    it and re-deciding what a merge means.
    """

    env: dict[str, str]
    mcp_servers: dict[str, Any]
    #: `env_mgr.remote.tools.ToolDef`-shaped. Typed loosely for `Prepared.tools`'
    #: reason: it crosses to `agent`, which may not import this package, and a
    #: package-authored `ToolDef` is duck-typed by construction.
    tools: tuple[Any, ...]
    settings: dict[str, Any]
    report: tuple[InstallOutcome, ...]


def install(
    agent_spec: Any,
    *,
    staged_package: str | None,
    config_dir: str,
    workspace: str | None = None,
    logs_dir: str | None = None,
    environ: Mapping[str, str] | None = None,
    recipe_timeout: float | None = None,
) -> AgentMaterial:
    """Install this agent's L1, L2 and L3 components into `config_dir`.

    `staged_package` is the **copy in the zone** (`interfaces.md` §4.16), never
    `Context.package`. Both are the same package and only one of them is inside
    the granted set: a path resolved against the original root points outside
    every grant, so a component installed from there would place files the
    agent's own session cannot read — and, for a ``*.tooldef.py``, would import
    the operator's live checkout instead of the copy this run was pinned to.

    `config_dir` is ``<zone>/config``, which `material.deploy` has already
    pointed ``CLAUDE_CONFIG_DIR`` at. Measured 2026-09-03: ``claude plugin
    marketplace add`` and ``claude plugin install`` fully respect it —
    ``plugins/``, ``settings.json`` and even ``.claude.json`` land in the
    relocated directory, and ``~/.claude/plugins/marketplaces`` was untouched.

    `environ` is the **zone environment `material.deploy` has already built**,
    including the agent spec's declared block. It is what every subprocess here
    runs under and what ``${VAR}`` in a component's `.mcp.json` expands against —
    one environment, so a server's declared command and the process that
    installed it cannot disagree. `None` means *this process's*, which is the
    shape a direct call in a test has.

    `logs_dir` is ``<zone>/logs``. The install report is written there as JSON
    and its path is exported: an agent that must state what it has is otherwise
    guessing, and `examples/env_checker`'s `check_capabilities_genuine` decides
    an ``unavailable`` verdict *against that file*.

    Raises `PrepareRefused` when something **declared** does not exist. It does
    not raise when an install *fails*: that is a named `InstallOutcome` in the report,
    because a failed `apt` on a `suggested` item is not a reason to stop a task
    and the recipe machinery already grades that distinction by `importance`.
    """
    os.makedirs(config_dir, exist_ok=True)
    child_env = _child_env(environ, config_dir)
    # An item's `cwd`, handed to the child as `--path`. The workspace when there
    # is one; the config directory otherwise, because it is inside the zone and
    # `install` has just created it — see `_run_recipe` on why this may not be
    # left to the recipe's own placeholder.
    recipe_cwd = workspace if workspace and os.path.isdir(workspace) else config_dir
    timeout = RECIPE_TIMEOUT_SECONDS if recipe_timeout is None else recipe_timeout

    trees = _component_trees(agent_spec, staged_package=staged_package)

    # **Before any install**, for the measured merge-not-clobber reason in the
    # module docstring. Later levels win, which is `trees`' order.
    settings: dict[str, Any] = {}
    for _, tree in trees:
        settings = _merge(settings, _read_json(os.path.join(tree, SETTINGS_FILENAME)))
    if settings:
        _write_json(os.path.join(config_dir, SETTINGS_FILENAME), settings)

    report: list[InstallOutcome] = []
    for path in _recipe_paths(agent_spec, staged_package=staged_package):
        report.extend(_run_recipe(path, child_env, recipe_cwd, timeout))

    mcp_servers: dict[str, Any] = {}
    tools: list[Any] = []
    for origin, tree in trees:
        outcomes, servers, defs = _install_tree(
            tree,
            origin=origin,
            config_dir=config_dir,
            environ=child_env,
            recipe_cwd=recipe_cwd,
            timeout=timeout,
        )
        report.extend(outcomes)
        # `update`, not `setdefault`: a later level is a nearer owner, and
        # the collision is reported so it is not silent either way.
        for name in set(servers) & set(mcp_servers):
            report.append(
                InstallOutcome(
                    "warn",
                    f"MCP server {name!r} redeclared by {origin}; the nearer component wins",
                    {"component": origin},
                )
            )
        mcp_servers.update(servers)
        tools.extend(defs)

    env: dict[str, str] = {}
    assets = _assets_dir(agent_spec, staged_package=staged_package)
    if assets is not None:
        env[paths.AGENT_ASSETS_ENV_VAR] = assets
    if _sequence(agent_spec, "components"):
        # **Only when declared, because only then is it granted.** `prepare`
        # composes a read grant on this directory from the same condition
        # (`isolation/policy.py::component_grants`), and `paths.py`'s rule for
        # this whole family is that exported and granted agree by construction —
        # an exported path we did not grant is the body failing on our own
        # instruction.
        env[paths.COMPONENTS_ROOT_ENV_VAR] = COMPONENTS_ROOT
    if logs_dir is not None:
        env[paths.INSTALL_REPORT_ENV_VAR] = _write_report(logs_dir, report)

    return AgentMaterial(
        env=env,
        mcp_servers=mcp_servers,
        tools=tuple(tools),
        settings=settings,
        report=tuple(report),
    )


# --------------------------------------------------------------------------- #
# Resolving what was declared


def _assets_dir(agent_spec: Any, *, staged_package: str | None) -> str | None:
    """``<staged package>/<AgentSpec.assets>``, or `None`.

    `None` for three different situations — no `assets` key, no staged package,
    or a spec from before the field existed — and they are one answer here on
    purpose: each means *this run has no agent asset directory*, and a caller
    that had to tell them apart would be reading our resolution rules.

    **Declared and absent raises**, because `spec_loader` fills this field from
    a directory it found. A value that no longer resolves means the staged copy
    is not the package the spec was loaded from, and that is worth stopping for.
    """
    rel = _get(agent_spec, "assets")
    if not rel or not staged_package:
        return None
    path = os.path.join(staged_package, str(rel))
    if not os.path.isdir(path):
        raise PrepareRefused(
            f"agent {_name(agent_spec)!r} declares assets {rel!r} and "
            f"{path!r} is not a directory in the staged package. spec_loader fills "
            f"that field from a directory it found, so the staged copy is not the "
            f"package this spec was loaded from"
        )
    return path


def _component_trees(agent_spec: Any, *, staged_package: str | None) -> list[tuple[str, str]]:
    """Every ``.claude/`` tree to install, as ``(origin, path)``, L2 then L3.

    `origin` is what a message names — ``component 'envchk-baseline'`` or
    ``the package's own assets`` — because by the time a `.claude/` tree fails
    to install, its path is a copy inside a zone and says nothing about who
    wrote it.

    An L2 component with no ``.claude/`` is an **error**: `components/README.md`
    marks that directory REQUIRED, and a component that installs nothing is
    indistinguishable from one whose contents were forgotten. An L3 directory
    with no ``.claude/`` is not — L3 is undeclared, so its absence is the normal
    shape of an agent that carries no components.
    """
    trees: list[tuple[str, str]] = []

    for name in _sequence(agent_spec, "components"):
        root = os.path.join(COMPONENTS_ROOT, name)
        if not os.path.isdir(root):
            raise PrepareRefused(
                f"agent {_name(agent_spec)!r} declares component {name!r} and "
                f"{root!r} does not exist. `components:` takes a bare name under "
                f"agent_sys/components/, never a path"
            )
        tree = os.path.join(root, CLAUDE_DIRNAME)
        if not os.path.isdir(tree):
            raise PrepareRefused(
                f"component {name!r} has no {CLAUDE_DIRNAME}/ directory. That is "
                f"what a component *is* (agent_sys/components/README.md), and "
                f"installing it would be a no-op nothing reports"
            )
        trees.append((f"component {name!r}", tree))

    assets = _assets_dir(agent_spec, staged_package=staged_package)
    if assets is not None:
        tree = os.path.join(assets, CLAUDE_DIRNAME)
        if os.path.isdir(tree):
            trees.append(("the package's own assets", tree))
    return trees


def _recipe_paths(agent_spec: Any, *, staged_package: str | None) -> list[str]:
    """L1 recipe YAMLs, resolved, in declaration order.

    Two spellings, tried in that order and both admitted because they answer
    different questions. A **package-relative path** is *this package's own
    recipe*, and resolves against the staged copy. A **bare name** is
    ``env_mgr/recipes/<name>.yaml``, one this repository ships. A package cannot
    shadow a shipped recipe by accident, because its own path has to resolve
    first for the name form never to be reached — and if it does resolve, the
    author wrote a file at that path and meant it.

    An L2 component's own ``recipe.yaml`` is **not** here: it is found beside
    the component's `.claude/` and run in `_install_tree`'s place in the order,
    so that a component's prerequisites install with the component rather than
    with the agent's unrelated L1 list.
    """
    out: list[str] = []
    for declared in _sequence(agent_spec, "recipes"):
        candidates = []
        if staged_package:
            candidates.append(os.path.join(staged_package, declared))
        candidates.append(os.path.join(os.path.dirname(__file__), "recipes", f"{declared}.yaml"))
        for path in candidates:
            if os.path.isfile(path):
                out.append(path)
                break
        else:
            raise PrepareRefused(
                f"agent {_name(agent_spec)!r} declares recipe {declared!r} and none "
                f"of {candidates!r} exists. It would have been skipped and the agent "
                f"would meet the absence as a failure of its own"
            )
    return out


# --------------------------------------------------------------------------- #
# Installing one component


def _install_tree(
    tree: str,
    *,
    origin: str,
    config_dir: str,
    environ: Mapping[str, str],
    recipe_cwd: str,
    timeout: float,
) -> tuple[list[InstallOutcome], dict[str, Any], list[Any]]:
    """One ``.claude/`` tree: its recipe, skills, plugins, MCP servers and tools.

    ``settings.json`` is **not** read here — it was merged and written before
    any of this ran, for the ordering reason in the module docstring. Reading it
    a second time here would be the same fact with two readers, and the second
    one would run after the plugin installs had already added their own keys to
    the file.
    """
    outcomes: list[InstallOutcome] = []

    # A component's own prerequisites, beside `.claude/` rather than inside it:
    # `recipe.yaml` is not part of Claude Code's layout and must not appear to be.
    prereq = os.path.join(os.path.dirname(tree), "recipe.yaml")
    if os.path.isfile(prereq):
        outcomes.extend(_run_recipe(prereq, environ, recipe_cwd, timeout))

    outcomes.extend(_install_skills(tree, origin=origin, config_dir=config_dir))
    outcomes.extend(_install_plugins(tree, origin=origin, config_dir=config_dir, environ=environ))

    servers, mcp_outcomes = _mcp_servers(tree, origin=origin, environ=environ)
    outcomes.extend(mcp_outcomes)

    tools, tool_outcomes = _tooldefs(tree, origin=origin)
    outcomes.extend(tool_outcomes)

    return outcomes, servers, tools


def _install_skills(tree: str, *, origin: str, config_dir: str) -> list[InstallOutcome]:
    """Copy each ``skills/<name>/`` into the zone's config directory.

    A **copy**, not a symlink or a grant, for `material.deploy`'s reason: the
    zone is what the confined session can read, and a link out of it resolves to
    a path the kernel refuses. `copy_out` is reused rather than `shutil.copytree`
    so that the refusal to copy a path onto itself is one rule in one place.
    """
    src_root = os.path.join(tree, "skills")
    if not os.path.isdir(src_root):
        return []
    out: list[InstallOutcome] = []
    for name in sorted(os.listdir(src_root)):
        src = os.path.join(src_root, name)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(config_dir, "skills", name)
        copy_out(src, dst)
        out.append(InstallOutcome("ok", f"skill {name!r} from {origin}", {"path": dst}))
    return out


def _install_plugins(
    tree: str, *, origin: str, config_dir: str, environ: Mapping[str, str]
) -> list[InstallOutcome]:
    """Register the component's local marketplace and install every plugin in it.

    Measured 2026-09-03: a local marketplace needs
    ``<dir>/.claude-plugin/marketplace.json`` carrying
    ``{name, owner, plugins: [{name, source, description}]}``, each plugin
    directory needs its own ``.claude-plugin/plugin.json``, and
    ``claude plugin validate <dir>`` checks both. So the manifest is **read** —
    not to convert it, but because the install command needs one argument per
    plugin and ``<plugin>@<marketplace>`` is the only spelling that names one.

    **The marketplace is copied into the zone before it is registered, and that
    is probe F rather than tidiness.** Measured 2026-09-03: a session loading a
    plugin's skill reported its base directory as the *marketplace source path*,
    not the config directory — ``settings.json`` records
    ``extraKnownMarketplaces: {<mp>: {source: {path: <dir>}}}`` and the plugin is
    read from that path **at run time**. Nothing is copied by the install. So
    registering `agent_sys/components/<name>/.claude/plugins` directly would
    install cleanly, report success, and then fail to load under confinement,
    because that path is outside every grant — a plausible value consumed as if
    it were right (`interfaces.md` §4.11). Copying first makes the registered
    path inside the zone *by construction*, and the assertion below is what says
    the construction held.

    A manifest that does not parse is a `fail` `InstallOutcome` rather than a raise:
    the directory exists, so this is not the declared-and-absent case, and the
    author gets the parser's complaint next to every other install result.
    """
    source = os.path.join(tree, "plugins")
    if not os.path.isdir(source):
        return []

    manifest_path = os.path.join(source, MARKETPLACE_MANIFEST)
    try:
        manifest = json.loads(_read_text(manifest_path))
    except (OSError, json.JSONDecodeError) as error:
        return [
            InstallOutcome(
                "fail",
                f"{origin} ships plugins/ with no readable {MARKETPLACE_MANIFEST}: {error}",
                {"path": manifest_path},
            )
        ]

    market = str(manifest.get("name") or "")
    if not market:
        return [
            InstallOutcome(
                "fail",
                f"{origin}'s {MARKETPLACE_MANIFEST} declares no 'name', and the "
                f"install spelling is <plugin>@<marketplace>",
                {"path": manifest_path},
            )
        ]

    out: list[InstallOutcome] = []
    root = copy_out(source, os.path.join(config_dir, MARKETPLACES_DIRNAME, market))
    if not contained(root, config_dir):
        # The construction above should make this impossible; it is asserted
        # because the failure it guards is silent. A marketplace outside the
        # granted set installs, reports success, and then serves nothing —
        # probe F is what makes that concrete rather than theoretical.
        raise PrepareRefused(
            f"the marketplace for {origin} would be registered at {root!r}, which is "
            f"not inside {config_dir!r}. Claude Code reads a plugin from its "
            f"marketplace source path at run time, so a path outside the zone is a "
            f"plugin that installs cleanly and never loads"
        )
    rc, text = _run_cmd(["claude", "plugin", "marketplace", "add", root], environ)
    out.append(
        InstallOutcome(
            "ok" if rc == 0 else "fail",
            f"marketplace {market!r} from {origin}: rc={rc}",
            {"rc": rc, "output": text, "path": root},
        )
    )
    if rc != 0:
        # Every subsequent install would fail for the same reason and say so
        # once each. One cause, one message.
        return out

    for plugin in manifest.get("plugins") or []:
        name = str(plugin.get("name") or "") if isinstance(plugin, Mapping) else str(plugin)
        if not name:
            continue
        rc, text = _run_cmd(["claude", "plugin", "install", f"{name}@{market}"], environ)
        out.append(
            InstallOutcome(
                "ok" if rc == 0 else "fail",
                f"plugin {name}@{market} from {origin}: rc={rc}",
                {"rc": rc, "output": text},
            )
        )
    return out


def _mcp_servers(
    tree: str, *, origin: str, environ: Mapping[str, str]
) -> tuple[dict[str, Any], list[InstallOutcome]]:
    """External servers from ``.mcp.json``, plus one per ``tools/*.mcp.py``.

    The two are one mapping because they are one thing to the model: an MCP
    server it can call. What differs is who wrote the launch command — the
    component's author, or this function, which spells a bundled server as
    ``sys.executable <path>`` so that it runs under the interpreter the
    supervisor is running under rather than whatever ``python3`` the zone's
    derived ``PATH`` resolves to.

    **A declared entry is expanded against the zone environment**, everywhere in
    it — command, args, and the server's own `env` block. That is what lets a
    component write ``"${CLAUDE_CONFIG_DIR}/servers/x.py"`` or
    ``"${UV_TOOL_BIN_DIR}/serena"`` and get an absolute path that exists,
    without knowing where the zone is. `_expand` refuses an unresolved name; see
    it for why a pass-through would be the worst of the three options.

    The last of those is also why the uv bin directory does **not** need to be on
    the agent's `PATH`: `executable_path(policy)` derives `PATH` at prepare step
    2 and that directory does not exist until step 6b installs it, so naming the
    binary absolutely is the only route that needs no reordering. It is also
    exactly what probe E measured working.
    """
    servers: dict[str, Any] = {}
    out: list[InstallOutcome] = []

    path = os.path.join(tree, MCP_FILENAME)
    declared = _read_json(path).get("mcpServers")
    if isinstance(declared, Mapping):
        for key, value in declared.items():
            servers[str(key)] = _expand(value, environ, where=f"{path} ({key})")
        out.append(
            InstallOutcome("ok", f"{len(declared)} external MCP server(s) from {origin}", {})
        )

    for path in _tool_files(tree, MCP_SUFFIX):
        name = os.path.basename(path)[: -len(MCP_SUFFIX)]
        servers[name] = {"type": "stdio", "command": sys.executable, "args": [path]}
        out.append(
            InstallOutcome("ok", f"bundled MCP server {name!r} from {origin}", {"path": path})
        )
    return servers, out


def _tooldefs(tree: str, *, origin: str) -> tuple[list[Any], list[InstallOutcome]]:
    """Import each ``tools/*.tooldef.py`` and take its module-level ``TOOLS``.

    **This executes package-authored code in the supervisor.** The module
    docstring states the three narrowings that are the whole defence; this
    function adds only the fourth, which is that a module failing to import is a
    `fail` `InstallOutcome` and not an exception. A component whose tool module is
    broken must not take the task down with it — the agent then runs without
    those tools, which is a degradation the report names, and stopping the run
    instead would make one bad file in one component fatal to every task using
    it.

    Loaded under a private module name so that two components shipping
    ``tools/util.tooldef.py`` do not overwrite one another in `sys.modules`.

    **Registered in `sys.modules` before `exec_module`, and it stays registered.**
    Measured 2026-09-03, CPython 3.13, by `pkg-author` against a real artefact:
    without the registration, an artefact using ``@dataclass`` raises
    ``AttributeError: 'NoneType' object has no attribute '__dict__'`` at import,
    because `dataclasses._is_type` resolves a string annotation through
    ``sys.modules[cls.__module__]`` and finds `None`. It is left in place rather
    than popped afterwards for the same reason one step later: anything that
    resolves an annotation lazily — `typing.get_type_hints`, a pydantic model, a
    second dataclass built at call time — does the same lookup after this
    function has returned.

    The failure that costs is not the exception; it is that a tool whose module
    fails to import is indistinguishable, from the model's side, from a tool
    that was never installed. `test_agent_assets.py` carries a fixture using
    ``@dataclass`` for exactly this.

    **The cost of retention, named rather than left to be discovered.** A
    long-lived supervisor accumulates one `sys.modules` entry per tooldef per
    *attempt*, and each holds a reference into a zone that may since have been
    removed. It is bounded by the number of attempts times the number of tool
    modules, which is small, and nothing here reclaims it. Stated so that a
    reader chasing memory finds a known cost instead of a surprise; popping the
    entry is not the fix, for the reason in the paragraph above.
    """
    tools: list[Any] = []
    out: list[InstallOutcome] = []
    for path in _tool_files(tree, TOOLDEF_SUFFIX):
        stem = os.path.basename(path)[: -len(TOOLDEF_SUFFIX)]
        module_name = f"_agent_sys_tooldef_{abs(hash(path)):x}_{stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"no import machinery accepts {path!r}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            declared = getattr(module, TOOLS_ATTR, None)
        except Exception as error:  # noqa: BLE001 — reported, not swallowed
            out.append(InstallOutcome("fail", f"{path} did not import: {error!r}", {"path": path}))
            continue
        if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)):
            out.append(
                InstallOutcome(
                    "fail",
                    f"{path} defines no module-level {TOOLS_ATTR}: list[ToolDef], so "
                    f"nothing in it reaches the model",
                    {"path": path},
                )
            )
            continue
        tools.extend(declared)
        out.append(
            InstallOutcome(
                "ok", f"{len(declared)} in-process tool(s) from {origin}", {"path": path}
            )
        )
    return tools, out


def _tool_files(tree: str, suffix: str) -> list[str]:
    root = os.path.join(tree, TOOLS_DIRNAME)
    if not os.path.isdir(root):
        return []
    return [os.path.join(root, n) for n in sorted(os.listdir(root)) if n.endswith(suffix)]


# --------------------------------------------------------------------------- #
# Running a recipe


def _run_recipe(
    path: str, environ: Mapping[str, str], cwd: str, timeout: float
) -> list[InstallOutcome]:
    """One recipe YAML through the shipped machinery, **as a child process**.

        <sys.executable> -m env_mgr bootstrap <recipe> --json --path <cwd>

    The module docstring gives the three reasons this is a subprocess and states
    that the CLI's argument surface is therefore a contract. Three details of
    the invocation are load-bearing:

    - ``-m env_mgr`` with `PYTHONPATH` pinned to `_PACKAGE_ROOT`, never the
      `env-mgr` console script: measured, the script on this host resolves to a
      different worktree's install.
    - ``--path``, always. `Target.path` is what `installers/base.py::run_cmd`
      passes as each item's `cwd`, and a shipped recipe cannot know where a zone
      is — `recipes/serena.yaml` carries a documented placeholder, because a
      real path in a data file is one machine's answer. Left unoverridden,
      `subprocess.run` **raises** rather than returning non-zero, so this is a
      crash rather than a bad outcome. The value is the agent's workspace when
      there is one: it is inside the zone, it exists, and a `kind: workspace`
      target means exactly that.
    - a **bounded** child. `installers/base.py::run_cmd` has no timeout, so a
      networked install that goes wrong hangs `prepare` for ever and
      `agent-sys --timeout` is a ceiling on the run rather than a cure. The
      bound applies to the ``python -m env_mgr`` process, which is the thing
      `run_cmd` cannot bound from inside. See `RECIPE_TIMEOUT_SECONDS`.
    - no filter flags. An agent naming a recipe is asking for the recipe, and a
      filter would be this module deciding which of its items the author meant.
      `--tag` / `--installer` / `--importance` / `--item` and `--on-conflict`
      are all reachable if that ever changes.

    **The JSON round-trip is lossless and that is why it costs nothing.**
    `report.render_json` emits ``{level, message, details}`` per outcome and
    `InstallOutcome` is a dataclass of exactly those three fields, so what this
    reconstructs is what an in-process call would have returned.

    A malformed recipe needs no special case here: `cli.main` already catches
    `RecipeError` into a `fail` outcome and exits 2, which is the shape this
    parses. What *is* handled is the child producing no usable JSON at all — an
    interpreter that could not start, a crash before `print`. That becomes one
    `fail` carrying the return code and the raw output, because a recipe whose
    result is unknown must not read as a recipe that succeeded.
    """
    rc, text = _run_cmd(
        [sys.executable, "-m", "env_mgr", _RECIPE_STAGE, path, "--json", "--path", cwd],
        environ,
        timeout,
    )
    if rc == _TIMED_OUT:
        return [
            InstallOutcome(
                "fail",
                f"recipe {os.path.basename(path)}: killed after {timeout:g}s. Whatever "
                f"it had already installed is still installed, and this run does not "
                f"know how much that was",
                {"path": path, "timeout": timeout, "output": text.strip()},
            )
        ]
    try:
        document = json.loads(text)
        outcomes = [
            InstallOutcome(str(o["level"]), str(o["message"]), dict(o.get("details") or {}))
            for o in document["outcomes"]
        ]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        return [
            InstallOutcome(
                "fail",
                f"recipe {os.path.basename(path)}: the installer child produced no "
                f"usable report ({error}); rc={rc}",
                {"path": path, "rc": rc, "output": text},
            )
        ]
    return [
        InstallOutcome(
            "info",
            f"recipe {os.path.basename(path)}: {document.get('status')}",
            {"path": path, "rc": rc},
        ),
        *outcomes,
    ]


def _child_env(environ: Mapping[str, str] | None, config_dir: str) -> dict[str, str]:
    """The environment every subprocess here runs under.

    **Built once and passed, never set on this process.** That is the third
    reason in the module docstring, and it is the whole difference between this
    and the earlier in-process design: `agent/runner.py` is threaded, so a
    global ``CLAUDE_CONFIG_DIR`` is a value two concurrent prepares can take
    from each other.

    `environ` is the zone environment `material.deploy` built, so the ``UV_*``
    roots an agent spec declared are already in it. ``CLAUDE_CONFIG_DIR`` is
    forced rather than defaulted: it is *this* call's config directory, and an
    inherited one would send a plugin install somewhere the session will not
    read. `PYTHONPATH` is prepended, not replaced — a caller's entries stay
    reachable behind ours.
    """
    env = dict(os.environ if environ is None else environ)
    env["CLAUDE_CONFIG_DIR"] = config_dir
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([_PACKAGE_ROOT, existing]).rstrip(os.pathsep)
    return env


def _run_cmd(
    argv: Sequence[str], environ: Mapping[str, str], timeout: float | None = None
) -> tuple[int, str]:
    """A command, its return code and its combined output. Never raises.

    An argv list rather than `installers/base.py`'s shell string: every argument
    here is a filesystem path or a name out of a component's manifest, and a
    marketplace called ``a; rm -rf b`` would otherwise be a shell injection out
    of a JSON file. The shipped `run_cmd` keeps its shell form, which its
    recipes' ``run:`` strings need.

    **A timeout returns rather than raises**, and the return code says which
    case it was. ``-1`` is not an errno and cannot be confused with one a child
    produced; `_run_recipe` turns it into a `fail` naming the bound. The child
    is killed, and whatever it had already installed stays installed — said out
    loud in the message, because a partial install that nobody mentions is worse
    than one that is reported.
    """
    try:
        proc = subprocess.run(
            list(argv), capture_output=True, text=True, env=dict(environ), timeout=timeout
        )
    except subprocess.TimeoutExpired as expired:
        return -1, _text(expired.stdout) + _text(expired.stderr)
    except OSError as error:
        return 127, f"{argv[0]}: {error}"
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _text(stream: Any) -> str:
    """`TimeoutExpired.stdout` is `bytes` even under ``text=True``. Measured, and
    it is the kind of thing that turns a timeout report into a `TypeError`."""
    if stream is None:
        return ""
    return stream.decode("utf-8", "replace") if isinstance(stream, bytes) else str(stream)


def _write_report(logs_dir: str, report: Sequence[InstallOutcome]) -> str:
    """The install report, as JSON inside the zone, and its path.

    **Promised rather than discoverable.** An agent asked to state what it has
    installed would otherwise have to find a file nobody named, and
    `examples/env_checker`'s `check_capabilities_genuine` decides whether an
    ``unavailable`` verdict is honest *by reading this* — an ``unavailable``
    beside a clean install report is a failure. A validator that cannot find the
    file fails the report for an avoidable reason.

    `<zone>/logs` is already the zone's own directory and already granted, so
    unlike the components root this needs no new grant — `paths.py`'s rule is
    satisfied without one.
    """
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, INSTALL_REPORT_FILENAME)
    _write_json(
        path,
        {
            "outcomes": [
                {"level": o.level, "message": o.message, "details": o.details} for o in report
            ]
        },
    )
    return path


def _expand(value: Any, environ: Mapping[str, str], *, where: str) -> Any:
    """``${VAR}`` against the zone environment, recursively, over a JSON value.

    **An unresolved name is an error, not a pass-through**, and that is the whole
    reason this exists rather than `os.path.expandvars`, which leaves an unknown
    name in place. Left in place, the server does not start and the symptom the
    operator sees is *a server with no tools* — no error, no cause, which is
    `interfaces.md` §4.11's family and precisely what this package exists to
    catch.

    Expanded against the environment `material.deploy` has already built, so a
    component's ``"${UV_TOOL_BIN_DIR}/serena"`` resolves to the same absolute
    path the recipe installed to. `${VAR}` only — no ``$VAR`` — because a
    JSON string full of shell-looking text should not have to be escaped, and
    the braced form is what every component in this tree writes.
    """
    if isinstance(value, str):

        def one(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in environ:
                raise PrepareRefused(
                    f"{where} references ${{{name}}} and nothing in this zone's "
                    f"environment defines it. Unexpanded, the server would fail to "
                    f"start and be reported as a server with no tools, which names "
                    f"no cause"
                )
            return environ[name]

        return _VAR_RE.sub(one, value)
    if isinstance(value, Mapping):
        return {k: _expand(v, environ, where=where) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, environ, where=where) for v in value]
    return value


# --------------------------------------------------------------------------- #
# Small readers


def _merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive for mappings, replace for everything else.

    **Lists replace rather than concatenate**, and that is the conservative
    reading rather than the convenient one: ``settings.json``'s ``hooks`` values
    are lists of matcher objects, and concatenating two components' lists would
    run both sets of hooks on every matching tool call — a behaviour neither
    author wrote. A component that means to add to another's list is asking for
    something this format cannot express, and it should say so loudly by
    replacing.
    """
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_json(path: str) -> dict[str, Any]:
    """A JSON object, or ``{}`` when the file is absent.

    **A file that exists and does not parse raises**, `harness.harness_env`'s
    rule and for its reason: it is an operator or author error one character
    wide, and continuing produces a session missing its hooks that blames
    itself.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise PrepareRefused(
            f"{path!r} exists and could not be read: {error}. It configures the "
            f"agent's session, so continuing would start an agent missing what it "
            f"declares, with nothing naming this as the cause"
        ) from error
    return document if isinstance(document, dict) else {}


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _write_json(path: str, document: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or os.curdir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _sequence(agent_spec: Any, key: str) -> tuple[str, ...]:
    value = _get(agent_spec, key)
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(v) for v in value)
    return ()


def _name(agent_spec: Any) -> str:
    return str(_get(agent_spec, "name") or "?")


def _get(agent_spec: Any, key: str) -> Any:
    """`material.py`'s reader, and the duplication is deliberate.

    Both modules accept either a mapping or a model, because both are called
    with an `AgentSpec` in production and with a dict in tests. Importing one
    from the other would put a private helper on a module's surface for no gain
    over four lines.
    """
    if isinstance(agent_spec, dict):
        return agent_spec.get(key)
    return getattr(agent_spec, key, None)
