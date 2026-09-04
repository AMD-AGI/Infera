# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-agent components, from three origins, into one Claude Code configuration.

`material.py` deploys what `agent` spec §3.1 declares as three lists of *file
paths* — ``rules``, ``hooks``, ``skills``. That shape stops at the first real
component: a skill is a **directory**, a plugin marketplace is a directory of
directories, an MCP server is a process to register rather than a file to place.
This module is the route for those, and it keeps `material.py`'s rule — *a file
is placed, not read* — for everything except the four documents whose contents
are an interface (`settings.json`, `.mcp.json`, `marketplace.json`, and a
`*.tooldef.py`'s ``TOOLS``).

**Every measurement cited below is evidence about `claude` 2.1.246** — the build
`cli/environment.py` pins and the one `material.deploy` now hands this module as
`agent_cli`. Naming the build is what makes these evidence rather than folklore:
probes B, C and F were first taken against the SDK's *bundled* 2.1.251, because
`_find_cli` prefers its own bundle over `PATH`, and had to be re-measured as
B'/C'/F' once that was noticed. A probe's own provenance is a fact to open, not
to assume.

## The three origins

**Not levels.** They were numbered L1/L2/L3 and the numbers were a vocabulary
three documents each restated slightly differently, while carrying no ordering
a reader could not get from the table itself. What actually differs is **who owns
the directory**:

| owner | what | declared how |
|---|---|---|
| upstream | serena, a marketplace plugin, an apt/pip tool | ``recipes: [...]`` -> an `env_mgr` recipe YAML |
| this repository | what `agent_sys` ships | ``agent_plugins: [...]`` -> `agent_sys/agent_plugins/<name>/`; an *install* rather than a tree is a recipe carrying ``tags: [internal]`` |
| one task package | what it carries for one agent | **undeclared** — `<agent assets>/.claude/` |

**The last two have the same on-disk shape**, so one installer serves both and
promoting is moving a directory. **A package's own material is undeclared**
because the declaration would be a second statement of what the directory
already says, and the two would drift; `agent_plugins/README.md` states the same
contract from the author's side.

## Order, and what it buys

Upstream, then this repository, then the package — so that when two origins bind
the same name, **the package's own copy wins**. That is `material.deploy`'s
existing precedence carried to a second kind of material: *an author saying so
outranks a default*.

One ordering constraint cuts across it and is **measured**, 2026-09-03:
``claude plugin marketplace add`` and ``claude plugin install`` *merge* into an
existing ``settings.json`` rather than clobbering it — a hand-written ``hooks`` key
survived a subsequent install, which only added ``enabledPlugins`` and
``extraKnownMarketplaces``. Merging is not commutative with creating: the file
has to exist first. So the settings document is assembled from every level and
**written before any install runs**, and this module writes it rather than
returning it for `material.py` to write. A caller cannot write it at the right
moment without knowing which levels resolved, and that is this module's
knowledge (`engineer_principle.md` §1 — one writer, and it is the one that owns
the ordering).

## A recipe runs the shipped machinery as a SUBPROCESS, and that is a contract

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
   timeout, so a networked recipe install that goes wrong hangs `prepare` for ever —
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
simply absent — that is the normal shape for a package's own material.

**A failed install is a named `InstallOutcome`, never a silent skip.** Every
subprocess's return code and output lands in `AgentMaterial.report`.

## The one place package-authored code is imported — and what that costs it

A ``*.tooldef.py`` is executed in **the supervisor's own process** to read its
module-level ``TOOLS``. That is not a sandboxed act and nothing here pretends it
is: the supervisor holds the API credentials the whole confinement design exists
to keep away from a task body. Three narrowings, and they are the whole defence:
it is done only for that exact suffix, only under ``tools/`` of a resolved
component, and only from the **staged** package or this repository's own
`agent_plugins/` — never from `Context.package`, which is the operator's live
checkout. An in-process tool is worth this because the alternative measured
shape (a stdio MCP server) costs a subprocess per tool and cannot share the
supervisor's objects; a component that does not need that should ship a
``*.mcp.py`` instead.

**The same fact has a second consequence, and nobody had connected it: an
in-process tool cannot see the environment the run declared.** Measured
2026-09-03. The security half above was written first — package code runs in the
supervisor — and this follows from that same sentence. The tool's ``call`` executes in the
supervisor's process, so it reads the **supervisor's** ``os.environ``;
``Prepared.environment``, which is where an agent spec's declared ``env`` block
lands, is handed to the **CLI child** (`claude_sdk.py`'s
``options.setdefault("env", …)``) and to nothing else. Driven through
`claude_sdk._adapt_tool` with a probe tool and no model call:

    supervisor  pid=784037   ENVCHK_NONCE=None
    tool saw    pid=784037   thread='asyncio_0'   ENVCHK_NONCE=<unset>

Found by a real run: a tool computed a token from an empty ``$ENVCHK_NONCE``,
returned a well-formed result, and the value was right about the wrong
environment. **The other two routes do not share this** — an external
``.mcp.json`` entry has its own ``env`` block, and a bundled ``*.mcp.py`` gets
one from `_mcp_servers` (and got the value by inheritance even before that).

**This is what is true today, not a statement about what the route should be.**
Whether the in-process route *ought* to be run-aware is an open decision and is
not this module's to take: the loader already holds the zone environment and
already loads each module once per attempt, so binding it at load — an additive
contract rather than a global — is available and unbuilt. Until that is ruled,
a component needing a run-specific value should ship a ``*.mcp.py`` rather than
a ``*.tooldef.py``. Written this way on purpose: a reader who finds a constraint
recorded as permanent cannot tell it from one that has since been removed.

**So the positive rule, which is what an author needs: a tool's per-run context
arrives as a tool argument. It never arrives through the environment, because
the tool runs in the supervisor.**

**Forced by the process model, not a style preference**, and that is the part
worth keeping: the supervisor is shared by every attempt and `agent/runner.py`
is threaded by construction, so a tool reading *any* process-global state cannot
be per-attempt **even if the value were populated correctly**. An argument binds
per call; a closure binds per construction; the environment binds per process,
and the process outlives the attempt. That is also why setting ``os.environ``
around the handler was rejected rather than merely disliked — same root cause,
not a second one.

`remote/tools.py` is the shape done properly and is **not currently reachable
from a package**, which is the honest form of a precedent you cannot follow:
`tools(...)` is a **factory**, `prepare` calls it with the zone and connection in
hand, and each `ToolDef.call` closes over them. A ``*.tooldef.py`` exposes a
module-level ``TOOLS`` and nothing ever hands that module anything, so the
closure half of the rule above is available to this package and not to its
callers. An author has the argument half and only that.

**The open question, dated 2026-09-03 and surfaced by run 2.** `env_mgr`'s own
tools would have this exact bug if they were declared the way packages must
declare theirs; they are immune only because they get a call site. **The missing
piece is not documentation, it is the factory `env_mgr` already gives itself and
does not give packages.** Whether to give it is undecided at the time of
writing. If it was decided against, this is the record of what was rejected; if
it was decided for, this is its specification. Deliberately the same words
either way.
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
from env_mgr.fs.path import contained, contained_syntactically
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
    "AGENT_PLUGINS_ROOT",
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

#: The agent plugins this repository ships. **A repository path, not a
#: configurable root**: ``agent_plugins: [<name>]`` takes a bare name precisely so
#: that a task package cannot point this anywhere, and a knob here would give back
#: what the name shape removed.
AGENT_PLUGINS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent_plugins")

#: The **default** recipe layer — the one nobody names, because it always
#: applies. It sits in ``env_mgr/`` rather than in ``env_mgr/recipes/``, and
#: that placement is the whole of how a reader tells it from the recipes beside
#: it: ``recipes/`` is the namespace of things you *name* in ``recipes: [x]``,
#: and this is not in it. No field says "this one is the default"; the path does.
#:
#: **Absent is normal.** Nothing declares it, so a missing file is simply
#: absent — not the declared-and-absent error, which only the agent layer can
#: reach.
DEFAULT_RECIPE = os.path.join(os.path.dirname(__file__), "default.env_recipe.yaml")

#: The **package** recipe layer's one admitted filename, under the staged
#: package's ``assets/``. See `_package_recipe_path` for why this is a single
#: spelling where the agent layer takes any permutation.
PACKAGE_RECIPE_BASENAME = "main.env_recipe.yaml"

#: Written into the zone's config directory. Nothing else writes this file.
SETTINGS_FILENAME = "settings.json"

#: What makes a directory a local plugin marketplace. Measured 2026-09-03 on
#: ``claude`` 2.1.246:
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

#: **Everything a `.claude/` tree may hold is placed in the zone except these,
#: and each one is here because it is read or relocated rather than skipped.**
#:
#: This set is the whole answer to *"can a component carry something a consumer
#: names and the installer does not place?"*. Placing is the default, so the
#: answer is: only if it is listed here, and each entry says where it goes
#: instead. The inverse — a list of what to copy — is what let `hooks/` and
#: `servers/` be named by `settings.json` and an `.mcp.json` and placed by
#: nobody (`reviewer`, 2026-09-03).
#:
#: | member | why not placed |
#: |---|---|
#: | ``settings.json`` | **read**, merged across levels, and written by `install` *before* any install runs — copying it here would overwrite the merge with one level's copy, after the plugin installs had already added their own keys |
#: | ``.mcp.json`` | **read** as an interface document and carried to the backend as `Prepared.mcp_servers`. Placing it would put a file in the zone that nothing reads, which reads as a configuration the session honours |
#: | ``plugins/`` | **relocated**, not skipped: copied to ``<config>/marketplaces/<name>/`` and registered there. `<config>/plugins/` is where Claude Code puts *installed* plugins, so a component's marketplace may not land on that name |
_NOT_PLACED = frozenset({SETTINGS_FILENAME, MCP_FILENAME, "plugins"})

#: A stdio MCP server the component ships; auto-registered under its stem.
MCP_SUFFIX = ".mcp.py"

#: A module exposing ``TOOLS: list[ToolDef]``, imported into the supervisor.
TOOLDEF_SUFFIX = ".tooldef.py"

#: The MCP server name an in-process `ToolDef` is addressed under — the model
#: calls ``mcp__env_mgr__<tool>``.
#:
#: **Owned by `agent/backends/claude_sdk.py::_TOOL_SERVER`, and duplicated here
#: because this package may not import `agent`** (`interfaces.md` §4.6, checked
#: by `test_env_mgr_imports_nothing_of_ours_but_task_graph`). That is a second
#: writer for one fact and it is a real cost, accepted for a narrow reason: this
#: copy is only ever *reported*, never used to address anything, so a divergence
#: makes the install report wrong and breaks no call. If the two ever need to
#: agree functionally, the name belongs in a place both may import — which does
#: not exist today, and inventing one for a report string would be
#: `engineer_principle.md` §2's failure mode.
IN_PROCESS_SERVER = "env_mgr"

#: What a ``*.tooldef.py`` must expose. A module-level name rather than a
#: factory call, so that reading the file tells a reviewer what it publishes
#: without running it in their head.
TOOLS_ATTR = "TOOLS"

#: The install report's filename inside ``<zone>/logs``.
INSTALL_REPORT_FILENAME = "agent_assets.install.json"

#: Where a component's marketplace is copied to before it is registered. Under
#: the zone's config directory, and probe F is why it is copied at all rather
#: than registered where it lies — see `_install_plugins`.
#:
#: **`marketplaces/` and emphatically not `plugins/`, which is a collision with
#: the harness's own namespace rather than a naming preference.** Measured
#: (probe A, `claude` 2.1.246): ``claude plugin install`` writes
#: ``<CLAUDE_CONFIG_DIR>/plugins/`` itself —
#: ``installed_plugins.json``, ``known_marketplaces.json``, ``marketplaces/``,
#: ``cache/``. A component's *source* marketplace copied onto that name would be
#: placed among the CLI's own bookkeeping, by us, before the CLI writes it. So
#: `plugins/` is in `_NOT_PLACED` — relocated, never skipped — and this is where
#: it goes. `test_a_components_marketplace_never_lands_on_the_harnesss_own_name`
#: keeps the two apart.
MARKETPLACES_DIRNAME = "marketplaces"

#: `bootstrap`, not `install`. `runner.run` defines that stage as install *then*
#: bootstrap, so a recipe item needing a post-install step — a marketplace add, a
#: login, a cache warm — is usable when `_run_recipe` returns. Under `install` it
#: would be installed and not usable, and the failure would surface inside the
#: agent's session rather than in the report.
_RECIPE_STAGE = "bootstrap"

#: ``${NAME}`` only. See `_expand` for why the bare ``$NAME`` form is excluded.
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: How long one recipe child may run, in seconds. **A default, and `install`
#: takes it as a parameter** — a site with a slow mirror is a fact about that
#: site, not about this code.
#:
#: **This bound exists only because a recipe runs as a child process**, and it is the third
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
    the whole reason a recipe is a subprocess rather than a call. The data arrives as
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
    agent_cli: str | None = None,
) -> AgentMaterial:
    """Install everything this agent asks for, from all three origins, into `config_dir`.

    `staged_package` is the **copy in the zone** (`interfaces.md` §4.16), never
    `Context.package`. Both are the same package and only one of them is inside
    the granted set: a path resolved against the original root points outside
    every grant, so a component installed from there would place files the
    agent's own session cannot read — and, for a ``*.tooldef.py``, would import
    the operator's live checkout instead of the copy this run was pinned to.

    `config_dir` is ``<zone>/config``, which `material.deploy` has already
    pointed ``CLAUDE_CONFIG_DIR`` at. Measured 2026-09-03 on ``claude`` 2.1.246:
    ``claude plugin
    marketplace add`` and ``claude plugin install`` fully respect it —
    ``plugins/``, ``settings.json`` and even ``.claude.json`` land in the
    relocated directory, and ``~/.claude/plugins/marketplaces`` still held only
    ``claude-plugins-official`` afterwards.

    **Not evidenced by ``~/.claude.json``'s checksum**, and that citation is
    withdrawn rather than merely dropped: measured later the same morning, with
    no probe running at all, that file changed twice in 75 seconds because every
    live Claude Code session on this host rewrites it. The earlier "unchanged"
    readings were luck. The two facts above are what the conclusion rests on.

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

    trees = _agent_plugin_trees(agent_spec, staged_package=staged_package)

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
            agent_cli=agent_cli,
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
    if _sequence(agent_spec, "agent_plugins"):
        # **Only when declared, because only then is it granted.** `prepare`
        # composes a read grant on this directory from the same condition
        # (`isolation/policy.py::agent_plugin_grants`), and `paths.py`'s rule for
        # this whole family is that exported and granted agree by construction —
        # an exported path we did not grant is the body failing on our own
        # instruction.
        env[paths.AGENT_PLUGINS_ROOT_ENV_VAR] = AGENT_PLUGINS_ROOT
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
    # **Joined under a check, not with `os.path.join`.** `assets` is
    # author-written and `Path(staged) / "/abs"` is `/abs` — F-D18, which
    # `spec_loader.AssetIndex.resolve_folder`'s docstring already records. The
    # lesson was applied to the loader's *output* and not to this consumer of
    # it, so a hand-bound `assets: ../../..` or an absolute one reached outside
    # the staged copy. That matters more here than anywhere else in this module:
    # `_tooldefs` imports `*.tooldef.py` from this directory **into the
    # supervisor**, and the module docstring's narrowing is that it never comes
    # from outside the staged package.
    path = contained_syntactically(str(rel), staged_package)
    if path is None:
        raise PrepareRefused(
            f"agent {_name(agent_spec)!r} declares assets {rel!r}, which does not "
            f"stay inside the staged package {staged_package!r}. A tool module is "
            f"imported from that directory into this process, so it may not be "
            f"reached by climbing out of the copy this run was pinned to"
        )
    if not os.path.isdir(path):
        raise PrepareRefused(
            f"agent {_name(agent_spec)!r} declares assets {rel!r} and "
            f"{path!r} is not a directory in the staged package. spec_loader fills "
            f"that field from a directory it found, so the staged copy is not the "
            f"package this spec was loaded from"
        )
    return path


def _agent_plugin_trees(agent_spec: Any, *, staged_package: str | None) -> list[tuple[str, str]]:
    """Every ``.claude/`` tree to install, as ``(origin, path)``: this
    repository's, then the package's own.

    `origin` is what a message names — ``component 'envchk-baseline'`` or
    ``the package's own assets`` — because by the time a `.claude/` tree fails
    to install, its path is a copy inside a zone and says nothing about who
    wrote it.

    A declared agent plugin with no ``.claude/`` is an **error**: `agent_plugins/README.md`
    marks that directory REQUIRED, and a component that installs nothing is
    indistinguishable from one whose contents were forgotten. A package's own
    assets directory with no ``.claude/`` is not — it is undeclared, so its
    absence is the normal
    shape of an agent that carries no components.
    """
    trees: list[tuple[str, str]] = []

    for name in _sequence(agent_spec, "agent_plugins"):
        root = os.path.join(AGENT_PLUGINS_ROOT, name)
        if not os.path.isdir(root):
            raise PrepareRefused(
                f"agent {_name(agent_spec)!r} declares component {name!r} and "
                f"{root!r} does not exist. `agent_plugins:` takes a bare name under "
                f"agent_sys/agent_plugins/, never a path"
            )
        tree = os.path.join(root, CLAUDE_DIRNAME)
        if not os.path.isdir(tree):
            raise PrepareRefused(
                f"component {name!r} has no {CLAUDE_DIRNAME}/ directory. That is "
                f"what a component *is* (agent_sys/agent_plugins/README.md), and "
                f"installing it would be a no-op nothing reports"
            )
        trees.append((f"component {name!r}", tree))

    assets = _assets_dir(agent_spec, staged_package=staged_package)
    if assets is not None:
        tree = os.path.join(assets, CLAUDE_DIRNAME)
        if os.path.isdir(tree):
            trees.append(("the package's own assets", tree))
    return trees


def _package_recipe_path(*, staged_package: str | None) -> str | None:
    """The package layer's recipe inside the staged copy, or `None`.

    **One spelling, and that is a cost paid deliberately.** The agent layer is
    found by `spec_loader`'s filename convention, which admits every `.`-joined
    permutation — `env_recipe.<agent>.yaml`, `<agent>.env_recipe.yaml`, and the
    rest. **This layer admits exactly `assets/main.env_recipe.yaml` and nothing
    else.** `main.env_recipe.package.yaml`, `env_recipe.main.yaml` and every
    other spelling a reader would reasonably expect from the agent layer
    **silently do nothing here.**

    The asymmetry is forced, not chosen. `spec_loader` owns that convention and
    `env_mgr` imports `spec_loader` **nowhere** — they are two independent
    components and a test enforces the partition. For this layer to use the same
    machinery, `spec_loader` would have to find the file and hand the path over
    on a *field*, and the only schema that could hold one is the task's — which
    would give every task in the graph its own recipe layer. That is a fourth
    layer arriving by accident, and two stated discovery rules are better than
    one silent extra layer.

    `main` and not the package's directory name: `main.yaml` is the reserved
    package-entry filename (`spec_loader`'s `ENTRY_FILENAME`), so `main` already
    means *this package* rather than any object in it.

    Resolved against the **staged** copy for `_assets_dir`'s reason — the
    operator's live checkout is not what the run executes.
    """
    if not staged_package:
        return None
    path = os.path.join(staged_package, "assets", PACKAGE_RECIPE_BASENAME)
    return path if os.path.isfile(path) else None


def _recipe_paths(agent_spec: Any, *, staged_package: str | None) -> list[str]:
    """Every recipe YAML to run, in execution order: **default, package, agent**.

    ## Three layers, and the layer is *where the file is*

    | layer | where | declared |
    |---|---|---|
    | default | `env_mgr/default.env_recipe.yaml` | never — it always applies |
    | package | `<staged package>/assets/main.env_recipe.yaml` | never — auto-detected |
    | agent | ``recipes: [...]`` on the agent spec | by name or package-relative path |

    No item carries a layer and none can: the **path** already says which layer
    a file is, and a field saying it again is a second writer of one fact. That
    is the reasoning that removed `Item.layer`, applied one level out.

    Order is most-general to most-specific, so a later layer runs later.

    ## They CONCATENATE — a later layer adds, it does not override

    This is the thing a reader will get wrong, because "layers" suggests
    override. It is not override, and it does not need to be: a recipe item is
    an *install action*, and every installer gates on `check` before `install`,
    so an item that two layers both declare is done once and reported twice.
    Nothing is discarded and nothing has to be reconciled.

    ## What is NOT checked across layers, and what closing it would cost

    **A version conflict between two layers is not detected.** `runner.detect_conflicts`
    would catch it — it is scoped to a single `run()` call, and `_run_recipe`
    spawns **one child process per recipe file**, so three layers are three
    independent conflict checks that never see each other. A default-layer
    ``uv <0.11`` and an agent-layer ``uv >=0.12`` both simply run, and the last
    one wins by execution order, silently.

    Closing it means parsing all three files **in this process** to compare
    their items before any child starts — which is exactly the in-process
    coupling the subprocess design exists to avoid (see the module docstring's
    three reasons). The gap is left open deliberately, with that cost named:
    a gap whose price is stated is a decision, and the same gap alone reads as
    an oversight.

    Worth knowing while reading the above: `detect_conflicts` fires only on
    **incompatible version constraints**, never on a repeated name. Two layers
    naming the same item is not an error *within* one file either, so the
    cross-file gap is narrower than it first sounds.

    ## Absence

    A layer that is **not declared and not there** is simply absent — the
    default and package layers are never declared, so a missing file is their
    normal shape. A layer that **is declared and not there** is an error;
    that is `material.py:62-86`'s existing rule and only the agent layer can
    reach it. There is no third case.

    ## The agent layer's two spellings

    A **package-relative path** is *this package's own recipe* and resolves
    against the staged copy. A **bare name** is ``env_mgr/recipes/<name>.yaml``,
    one this repository ships. A package cannot shadow a shipped recipe by
    accident, because its own path has to resolve first for the name form never
    to be reached — and if it does resolve, the author wrote a file at that path
    and meant it.

    An agent plugin's own ``recipe.yaml`` is **not** here: it is found beside
    the component's `.claude/` and run in `_install_tree`'s place in the order,
    so that a component's prerequisites install with the component rather than
    with the agent's unrelated ``recipes:`` list.
    """
    out: list[str] = []
    shipped = os.path.join(os.path.dirname(__file__), "recipes")

    if os.path.isfile(DEFAULT_RECIPE):
        out.append(DEFAULT_RECIPE)
    package_recipe = _package_recipe_path(staged_package=staged_package)
    if package_recipe is not None:
        out.append(package_recipe)

    for declared in _sequence(agent_spec, "recipes"):
        candidates = []
        if staged_package:
            # **Checked, for `_assets_dir`'s reason.** `../../..` climbs out of
            # the staged copy and an absolute value replaces it outright, and a
            # recipe is a file this module hands to a subprocess to execute.
            # `None` simply means *not a package-relative recipe*, so the bare
            # name below is still tried and the refusal names both candidates.
            inside = contained_syntactically(declared, staged_package)
            if inside is not None:
                candidates.append(inside)
        # The bare-name form. `os.path.basename` because this one is ours: a
        # declared name may not select a file outside the shipped directory by
        # spelling a path, and unlike the package-relative form there is no
        # legitimate reading in which it contains a separator.
        candidates.append(os.path.join(shipped, f"{os.path.basename(declared)}.yaml"))
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
    agent_cli: str | None,
) -> tuple[list[InstallOutcome], dict[str, Any], list[Any]]:
    """One ``.claude/`` tree, placed in the zone. **Place by default; name every
    exception.**

    This was *"copy `skills/`, copy `plugins/`"* and it had a hole with no
    bottom: `hooks/` and `servers/` were named by consumers and placed by
    nobody. Measured by `reviewer`, 2026-09-03, against the real package with a
    real `claude` — ``settings.json`` landed in the zone naming
    ``$CLAUDE_CONFIG_DIR/hooks/envchk_session_start.py`` with the script absent,
    and `envchk-baseline`'s server was reported ``ok  1 external MCP server(s)``
    with ``args[0]`` pointing at a file that did not exist. That second one is
    the failure `_expand`'s docstring claims to make impossible: the variable
    resolved, and the file was not there.

    **So the enumeration is inverted.** The question a reader must be able to
    answer is not *are `hooks/` and `servers/` copied* but *is there anything a
    `.claude/` tree can carry that a consumer will name and this will not
    place*, and that is only answerable if placing is the default. Everything
    under `.claude/` is copied into `<zone>/config/` except the members of
    `_NOT_PLACED`, and each of those is a **read** or a **relocation**, listed
    there with its reason. A directory Claude Code adds next year — `agents/`,
    `commands/`, an output style — is placed on the day it appears, without an
    edit here.

    `recipe.yaml` sits beside `.claude/` rather than inside it, so it is not
    part of this and is not placed: it is not part of Claude Code's layout and
    must not appear to be.
    """
    outcomes: list[InstallOutcome] = []

    prereq = os.path.join(os.path.dirname(tree), "recipe.yaml")
    if os.path.isfile(prereq):
        outcomes.extend(_run_recipe(prereq, environ, recipe_cwd, timeout))

    outcomes.extend(_place_tree(tree, origin=origin, config_dir=config_dir))
    outcomes.extend(
        _install_plugins(
            tree,
            origin=origin,
            config_dir=config_dir,
            environ=environ,
            agent_cli=agent_cli,
        )
    )

    servers, mcp_outcomes = _mcp_servers(
        tree, origin=origin, environ=environ, config_dir=config_dir
    )
    outcomes.extend(mcp_outcomes)

    tools, tool_outcomes = _tooldefs(tree, origin=origin, config_dir=config_dir)
    outcomes.extend(tool_outcomes)

    return outcomes, servers, tools


def _place_tree(tree: str, *, origin: str, config_dir: str) -> list[InstallOutcome]:
    """Copy every member of a ``.claude/`` tree into the zone, bar the exceptions.

    A **copy**, not a symlink and not a grant, for `material.deploy`'s reason:
    the zone is what the confined session can read, and a link out of it
    resolves to a path the kernel refuses. That is also the ruling for
    `servers/` specifically — *copy into the zone, do not reference out of it* —
    chosen over having components name ``${AGENT_SYS_AGENT_PLUGINS_ROOT}/...``
    because it makes the three docstrings that already assert a copy true, the
    copy dies with the zone, and it gives **one rule for both origins**: a package's
    `tools/*.mcp.py` used to be registered at its *source* path and worked only
    because that happened to lie inside the staged package.

    `copy_out` rather than `shutil.copytree` so that the refusal to copy a path
    onto itself stays one rule in one place.

    **Symlinks are resolved, at every depth, and that is `dereference=True`.**
    `copy_out`'s default is asymmetric — measured 2026-09-03, a *top-level*
    symlink goes through `copy2` and arrives as a real file, while one *nested*
    inside a placed directory goes through ``copytree(symlinks=True)`` and
    arrives still a symlink. Same input, two results, decided by depth.

    Resolving is the choice rather than preserving or refusing, and the
    measurement is what picks it: a preserved link pointing outside the zone
    **fails `contained`** — checked, `contained(<zone>/link -> /outside, <zone>)`
    is `False` while the dereferenced copy is `True` — so it is a path the
    confined session cannot follow, reported as installed. That is probe F's
    installs-cleanly-never-loads shape wearing a different hat, and *copy into
    the zone, do not reference out of it* is already this module's rule for the
    marketplace. A preserved link **is** a reference out of the zone.

    What it costs, stated: a component that links to host content gets that
    content copied into the zone at prepare time, on its author's behalf.
    Refusing links instead would reject a layout nobody has written yet, and
    would leave the depth asymmetry in place for the ones that stay inside the
    tree.
    """
    if not os.path.isdir(tree):
        return []
    out: list[InstallOutcome] = []
    for name in sorted(os.listdir(tree)):
        if name in _NOT_PLACED:
            continue
        target = os.path.join(config_dir, name)
        # **An overwrite between components is reported, never silent.** Levels
        # are installed in order and `copy_out` merges with `dirs_exist_ok`, so
        # a member two components both ship — `skills/x/SKILL.md`,
        # `tools/util.tooldef.py` — ends up holding only the later one's bytes.
        # That is the same event `_mcp_servers` already warns about for a server
        # name, and it was unreported here for a *file*, which is how one
        # component's artefact can be absent from the zone while its report says
        # `ok`. Precedence is unchanged — later wins, which is the origin order's rule —
        # and only the silence is.
        existing = _existing_files(target)
        collisions = sorted(existing & _relative_files(os.path.join(tree, name)))
        if collisions:
            out.append(
                InstallOutcome(
                    "warn",
                    f"{origin} replaces {len(collisions)} already-placed file(s) "
                    f"under {name!r}; the nearer level wins",
                    {"path": target, "files": collisions[:20]},
                )
            )
        dst = copy_out(os.path.join(tree, name), target, dereference=True)
        out.append(InstallOutcome("ok", f"placed {name!r} from {origin}", {"path": dst}))
    return out


def _relative_files(root: str) -> set[str]:
    """Every file under `root`, relative to it. `root` itself for a plain file.

    The unit of a collision is a **file**, not a directory: two components both
    shipping `skills/` collide only if they ship the same skill, and reporting
    the directory would cry wolf on every second component.
    """
    if os.path.isfile(root):
        return {os.path.basename(root)}
    return {
        os.path.relpath(os.path.join(where, f), root)
        for where, _, files in os.walk(root)
        for f in files
    }


def _existing_files(target: str) -> set[str]:
    if not os.path.exists(target):
        return set()
    return _relative_files(target)


def _install_plugins(
    tree: str,
    *,
    origin: str,
    config_dir: str,
    environ: Mapping[str, str],
    agent_cli: str | None,
) -> list[InstallOutcome]:
    """Register the component's local marketplace and install every plugin in it.

    Measured 2026-09-03 on ``claude`` 2.1.246: a local marketplace needs
    ``<dir>/.claude-plugin/marketplace.json`` carrying
    ``{name, owner, plugins: [{name, source, description}]}``, each plugin
    directory needs its own ``.claude-plugin/plugin.json``, and
    ``claude plugin validate <dir>`` checks both. So the manifest is **read** —
    not to convert it, but because the install command needs one argument per
    plugin and ``<plugin>@<marketplace>`` is the only spelling that names one.

    **The marketplace is copied into the zone before it is registered, and that
    is probe F rather than tidiness.** Measured 2026-09-03 (re-measured as F' on
    ``claude`` 2.1.246 after the first run used the SDK's bundled 2.1.251): a
    session loading a
    plugin's skill reported its base directory as the *marketplace source path*,
    not the config directory — ``settings.json`` records
    ``extraKnownMarketplaces: {<mp>: {source: {path: <dir>}}}`` and the plugin is
    read from that path **at run time**. Nothing is copied by the install. So
    registering `agent_sys/agent_plugins/<name>/.claude/plugins` directly would
    install cleanly, report success, and then fail to load under confinement,
    because that path is outside every grant — a plausible value consumed as if
    it were right (`interfaces.md` §4.11). Copying first makes the registered
    path inside the zone *by construction*, and the assertion below is what says
    the construction held.

    **The CLI is the pinned one, absolute, and never the bare name `claude`.**
    Measured on this host 2026-09-03, and it is the same defect
    `claude_sdk._prepared_cli` refuses from the other side of the seam:

        Prepared.agent_cli      ~/.local/bin/claude -> versions/2.1.246
        bare `claude` on the policy-derived PATH   /usr/local/bin/claude  2.1.197

    `agent_cli_grants` grants the CLI's *install* directory, not the shim
    directory that holds `~/.local/bin/claude`, so the shim is not on the
    derived `PATH` and a bare name resolves to a different build — an npm-owned
    2.1.197 from July. The session would then talk to 2.1.246 while its plugins
    were installed by 2.1.197, through one shared `CLAUDE_CONFIG_DIR`. Probe A,
    the entire basis for believing plugin installs honour that variable, was
    measured on 2.1.246 and says nothing about 2.1.197.

    This is the rule the rest of this module already follows twice — a bundled
    MCP server runs under `sys.executable`, and a serena entry names
    ``${UV_TOOL_BIN_DIR}/serena`` — stated once more because here it was got
    wrong: **name the binary, do not search for it.**

    **The hazard was written down in this repository before this module
    existed.** `cli/environment.py:512-519`, on the line that sets
    ``agent_cli``: *"a run with this left `None` would either refuse or, if the
    backend let it through, execute a different build from the one `env_mgr`
    installed plugins into and succeed without them."* That was about the
    **session** running a build other than the installs'; what happened here is
    the mirror image, the **installs** running a build other than the session's,
    and it arrived because this was new code that looked a binary up by name
    while every existing consumer had already been made to name it. Two
    consumers of one ``CLAUDE_CONFIG_DIR`` must be one binary — in either
    direction.

    And the mismatch is **structural, not incidental**: `cli/environment.py`
    resolves `claude` with `shutil.which` in the *supervisor*, whose ``PATH``
    carries ``~/.local/bin``, while the child's ``PATH`` is derived from the
    granted policy, which does not. The two answers differ by construction, so
    no amount of care in the child's environment fixes it. Only naming the
    binary does.

    **An absent CLI is a `fail` outcome, not a raise, and not a fallback.** Not a
    fallback, because silently running an unknown build is exactly what the
    measurement above condemns. Not a raise, because it is the same event as
    ``claude plugin install`` exiting non-zero — the plugin did not install, the
    agent will run without it, and the report says so — and raising would make
    one component's `plugins/` directory fatal to every task that names it,
    which is the argument `_tooldefs` already lost for its own case. A run whose
    agent ships no marketplace never reaches this line, so *no `claude` on the
    machine* stays a working configuration.

    A manifest that does not parse is a `fail` `InstallOutcome` rather than a raise:
    the directory exists, so this is not the declared-and-absent case, and the
    author gets the parser's complaint next to every other install result.
    """
    source = os.path.join(tree, "plugins")
    if not os.path.isdir(source):
        return []

    if not agent_cli:
        return [
            InstallOutcome(
                "fail",
                f"{origin} ships plugins/ and this run pinned no `claude` CLI, so "
                f"there is nothing to install them with. Falling back to a bare "
                f"`claude` would run whichever build the derived PATH happens to "
                f"reach, which is not the build the session uses",
                {"component": origin},
            )
        ]

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

    # **Checked BEFORE the copy, on the un-joined name.** This was
    # `copy_out(...)` and *then* `contained()`, with a comment saying the
    # construction made an escape impossible — and the construction was
    # `os.path.join` with `manifest["name"]`, an author-controlled JSON string.
    # Measured by `reviewer` 2026-09-03 with `"name": "../../../ESCAPED"`: the
    # tree **was written outside the zone**, and only then did the refusal fire.
    # An absolute name is worse — `os.path.join` discards `config_dir` outright
    # and `copy_out`'s `dirs_exist_ok=True` merges into whatever is there.
    #
    # `contained_syntactically` is the right check and it existed unused: it
    # rejects an absolute path and any `..` that climbs out, **without touching
    # the filesystem**, which is what makes it usable before the directory it
    # would create exists. `contained` cannot serve here for that exact reason —
    # it calls `resolve_strict` on both sides and the destination is not there
    # yet.
    #
    # On this host a write outside the tree is not a hypothetical: the
    # repository's no-delete rule was bought by a real one.
    # **The question is "is this a single directory name", not "does it stay
    # inside the zone".** Those differ, and the difference is a real hole that
    # `contained_syntactically` alone does not close: measured by `reviewer`,
    # ``name: ".."`` normalises ``marketplaces/..`` to ``.`` and returns
    # `config_dir` **unchanged**, so the component's `plugins/` was emptied
    # straight into the config root and ``claude plugin marketplace add
    # <config>`` registered the whole zone configuration directory as a
    # marketplace. The trailing `contained()` passed, because `config_dir` is
    # contained in itself. ``"."`` and ``"a/.."`` do the same.
    #
    # Nothing escaped the zone, so the fix above still holds — what this defeats
    # is the *other* decision: `_NOT_PLACED`'s `plugins` row and
    # `MARKETPLACES_DIRNAME` exist to keep a component's marketplace off the
    # harness's namespace, and this landed it one level **above** that, on top
    # of `settings.json` and everything `_place_tree` had just written.
    #
    # `market` is also an identifier, not only a path component — it is half of
    # the ``<plugin>@<marketplace>`` spec handed to the CLI — so a value that is
    # not a plain name is wrong twice over.
    if market != os.path.basename(market) or market in (os.curdir, os.pardir):
        return [
            InstallOutcome(
                "fail",
                f"{origin}'s {MARKETPLACE_MANIFEST} declares the marketplace name "
                f"{market!r}, which is not a single directory name. It is used both "
                f"as a directory under the zone config and as the right-hand side of "
                f"<plugin>@<marketplace>. Nothing was copied",
                {"path": manifest_path, "name": market},
            )
        ]

    relative = os.path.join(MARKETPLACES_DIRNAME, market)
    destination = contained_syntactically(relative, config_dir)
    if destination is None:
        # Kept as the second half rather than replaced by the check above: that
        # one says *this is a name*, this one says *the join stays inside*. A
        # single check would have to mean both, and the pair is what makes each
        # message say which rule was broken.
        return [
            InstallOutcome(
                "fail",
                f"{origin}'s {MARKETPLACE_MANIFEST} declares the marketplace name "
                f"{market!r}, which does not stay inside the zone when used as a "
                f"directory name. Nothing was copied",
                {"path": manifest_path, "name": market},
            )
        ]

    root = copy_out(source, destination)
    if not contained(root, config_dir):
        # The syntactic check above cannot see a symlink; this one resolves both
        # sides and runs now that the directory exists. Belt and braces here is
        # earned rather than reflexive — a marketplace outside the granted set
        # installs, reports success and then serves nothing (probe F), so the
        # failure it guards is silent.
        raise PrepareRefused(
            f"the marketplace for {origin} would be registered at {root!r}, which is "
            f"not inside {config_dir!r}. Claude Code reads a plugin from its "
            f"marketplace source path at run time, so a path outside the zone is a "
            f"plugin that installs cleanly and never loads"
        )
    rc, text = _run_cmd([agent_cli, "plugin", "marketplace", "add", root], environ)
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
        rc, text = _run_cmd([agent_cli, "plugin", "install", f"{name}@{market}"], environ)
        out.append(
            InstallOutcome(
                "ok" if rc == 0 else "fail",
                f"plugin {name}@{market} from {origin}: rc={rc}",
                {"rc": rc, "output": text},
            )
        )
    return out


def _mcp_servers(
    tree: str, *, origin: str, environ: Mapping[str, str], config_dir: str
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

    **A bundled server is registered at its PLACED path, never its source.**
    This read `os.path.join(tree, ...)` and worked only by luck: a package's tree
    is inside the staged package, which is inside the zone. An agent plugin
    shipping the same file registers a path under `AGENT_PLUGINS_ROOT` — outside
    every grant — and that is probe F's failure one directory over: it installs
    cleanly, reports `ok`, and serves nothing. `_place_tree` has already copied
    `tools/` into the config directory, so the placed path both exists and is
    reachable. Same rule as the marketplace, and now the same rule at both
    levels.
    """
    servers: dict[str, Any] = {}
    out: list[InstallOutcome] = []

    path = os.path.join(tree, MCP_FILENAME)
    declared = _read_json(path).get("mcpServers")
    if isinstance(declared, Mapping):
        for key, value in declared.items():
            servers[str(key)] = _expand(value, environ, where=f"{path} ({key})")
        out.append(
            InstallOutcome(
                "ok",
                f"{len(declared)} external MCP server(s) from {origin}",
                # **The names, not the count.** This recorded `{}`, and it is
                # the one route of the three whose key is chosen by an author in
                # a data file — so it is the only one that can silently be about
                # a *different* server than the reader expects. `Prepared.mcp_servers`
                # goes supervisor -> backend and is written down nowhere else, so
                # the install report is the only artefact that could carry the
                # evidence, and it carried a number.
                #
                # What this buys is narrow and worth stating: it catches
                # *declared nowhere* at prepare time, before the agent starts. It
                # does **not** catch a server that is declared and fails to start
                # — that is the `args[0]` existence assertion, or a handshake.
                {"names": sorted(str(k) for k in declared)},
            )
        )

    for source in _tool_files(tree, MCP_SUFFIX):
        name = os.path.basename(source)[: -len(MCP_SUFFIX)]
        placed = os.path.join(config_dir, TOOLS_DIRNAME, os.path.basename(source))
        if name in servers:
            # **The same-tree collision, which the cross-tree one already
            # reported and this did not.** `.mcp.json` declaring `x` beside
            # `tools/x.mcp.py` is the likelier author mistake of the two, and
            # silently overwriting leaves them with a server they did not write
            # and no message. Reported and the bundled one wins, because it is
            # the one whose file this function can prove exists.
            out.append(
                InstallOutcome(
                    "warn",
                    f"{origin} declares MCP server {name!r} in {MCP_FILENAME} and also "
                    f"ships {TOOLS_DIRNAME}/{os.path.basename(source)}; the bundled "
                    f"server wins",
                    {"path": placed},
                )
            )
        # **`env` stated, not inherited**, and the entry is the only place it can
        # be stated. Measured (run 2, capability 5): the bundled server *did*
        # see the run's `ENVCHK_NONCE` with no `env` key here at all — the SDK
        # hands `Prepared.environment` to the CLI child, the CLI spawns this
        # server as *its* child, and the value arrives by inheritance. So this
        # is not a fix for something broken; it is a fix for **nothing stating
        # why it works**, and what makes it work is a third party's process
        # model. One CLI change from silent, and silent here means a
        # well-formed answer computed against the wrong environment — which is
        # exactly what the in-process route was measured doing.
        #
        # **The whole mapping, not a subset, and that is deliberate under an
        # unmeasured semantics.** Whether the SDK *merges* this `env` with the
        # child's or *replaces* it is not measured. A subset would be correct
        # under merge and would strip `PATH` under replace, so the server would
        # stop starting; the full mapping is correct under both. The available
        # evidence points at merge — `envchk-baseline`'s `.mcp.json` declares
        # only `ENVCHK_NONCE` and its server starts, which a replaced
        # environment with no `PATH` makes hard to explain — but that is an
        # inference from one run, not a measurement, and it is recorded as such
        # rather than relied on.
        #
        # What is in it: this attempt's zone paths, the operator's harness block
        # and the agent spec's declared `env`. The harness block carries
        # credentials, and they already reached this server by inheritance — so
        # the exposure is unchanged and is now **visible**, which is the point.
        servers[name] = {
            "type": "stdio",
            "command": sys.executable,
            "args": [placed],
            "env": dict(environ),
        }
        out.append(
            InstallOutcome(
                "ok",
                f"bundled MCP server {name!r} from {origin}",
                # `server` recorded rather than left to be parsed out of the
                # message or the path stem. Re-deriving a name a producer
                # already knew is the same defect one layer down, and it is what
                # a reader of this report would otherwise have to do.
                {"server": name, "path": placed},
            )
        )
    return servers, out


def _tooldefs(tree: str, *, origin: str, config_dir: str) -> tuple[list[Any], list[InstallOutcome]]:
    """Import each ``tools/*.tooldef.py`` and take its module-level ``TOOLS``.

    **This executes package-authored code in the supervisor.** The module
    docstring states the three narrowings that are the whole defence; this
    function adds only the fourth, which is that a module failing to import is a
    `fail` `InstallOutcome` and not an exception. A component whose tool module is
    broken must not take the task down with it — the agent then runs without
    those tools, which is a degradation the report names, and stopping the run
    instead would make one bad file in one component fatal to every task using
    it.

    **Enumerated from the source tree and imported from the PLACED copy**, which
    is `_mcp_servers`' shape for the same reason one line over: the run owns the
    copy in the zone and does not own the directory it came from. For a package's own material the two
    were the same file, so this looked settled; for an **agent plugin** the
    source is `agent_sys/agent_plugins/<name>/.claude/tools/…`, so the supervisor
    was importing **the repository** rather than the copy this attempt was
    pinned to. That is the bare-`claude` defect's class — two consumers, one of
    them reading a path the run does not own — and the module docstring's
    narrowing (*"the staged package or this repository's own `agent_plugins/`"*) is
    the wording that made it read as fine.

    It also puts ``__pycache__`` in the zone, where it dies with the zone,
    instead of writing it into the repository during `prepare`. That is a
    consequence rather than the reason, and it is why no `sys.dont_write_bytecode`
    is needed — a process-global of exactly the kind the subprocess route removed.

    **Enumerated from the source, though, and that part is load-bearing.**
    `<config>/tools/` accumulates every level's files as each is placed, so
    listing *it* would re-import the previous component's modules under this
    component's name and register their tools twice. The source tree names
    exactly this component's files; the placed path is only where each one is
    read from.

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
    for source in _tool_files(tree, TOOLDEF_SUFFIX):
        path = os.path.join(config_dir, TOOLS_DIRNAME, os.path.basename(source))
        if not os.path.exists(path):
            # `_place_tree` ran first and copies `tools/`, so this cannot happen
            # without the placement rule having changed under this function.
            # Reported rather than asserted, because the consequence is a missing
            # capability and the run can continue without it.
            out.append(
                InstallOutcome(
                    "fail",
                    f"{source} was not placed, so there is nothing to import from "
                    f"{path!r}. A tool module is read from the zone copy, never from "
                    f"the directory it came from",
                    {"path": path},
                )
            )
            continue
        stem = os.path.basename(source)[: -len(TOOLDEF_SUFFIX)]
        # **Keyed on `source`, loaded from `path`, and the two halves are not
        # interchangeable.** `99d3aea` moved the *load* to the placed copy and
        # took the module name with it — and the placed path is
        # ``<config>/tools/<basename>``, identical for every component shipping
        # the same file name. Measured: two components each shipping
        # ``tools/util.tooldef.py`` produced one module name twice, the second
        # import replaced the first in `sys.modules`, and
        # ``get_type_hints`` on the first component's tool raised
        # ``NameError: name 'AlphaArgs' is not defined`` — its annotations
        # resolved against the *other* component's namespace. Four `ok`s in the
        # report and nothing said anything.
        #
        # That is precisely the state the registration below exists to prevent,
        # arriving one component later instead of at import. `source` is unique
        # per component, so it is what identifies the module; `path` is only
        # where the bytes are read from.
        module_name = f"_agent_sys_tooldef_{abs(hash(source)):x}_{stem}"
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
                "ok",
                f"{len(declared)} in-process tool(s) from {origin}",
                # **`tools` is the load-bearing key here, not `server`.**
                # `IN_PROCESS_SERVER` is a constant — every run has it the moment
                # any `ToolDef` exists — so a reader comparing only against that
                # would have a check that cannot fail. The name worth recording
                # is the **tool's**, which is the half of
                # ``mcp__env_mgr__<tool>`` an author chooses and the half that
                # gets mis-addressed.
                #
                # `getattr` because a `ToolDef` here is duck-typed by
                # construction — it is package-authored and this module does not
                # define the class. A tool with no `.name` cannot be addressed by
                # the model at all; recorded as such rather than guessed at.
                {
                    "server": IN_PROCESS_SERVER,
                    "tools": [str(getattr(d, "name", "<unnamed>")) for d in declared],
                    "path": path,
                },
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
    unlike the agent plugins root this needs no new grant — `paths.py`'s rule is
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
