# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-agent material -- recipes and one copied tree -- into one Claude Code configuration.

`agent` spec section 3.1's ``rules``/``hooks``/``skills`` file-path lists stop at
the first real component: a skill is a directory, a marketplace a directory of
directories, an MCP server a process to register. This module handles those,
keeping `material.py`'s rule that a file is placed, not read, except for the
three interface documents (``settings.json``, ``.mcp.json``, ``marketplace.json``).

Two routes, per `docs/spec.provisioning.md` (normative): upstream/repo material
installs by recipe; a task package's own material is undeclared, copied from
``<agent assets>/.claude/`` after recipes run, so its file outranks a default.
A recipe runs as a child ``env_mgr`` process, not an import. Raises
`PrepareRefused` for a declared-and-absent asset or recipe; a failed install is
a named `InstallOutcome`, never a silent skip.
"""

from __future__ import annotations

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

#: The directory holding the `env_mgr` package -- what a child process's
#: `PYTHONPATH` needs for ``-m env_mgr`` to reach this tree. Derived from this
#: file, never from the environment, so the child always runs the tree that
#: spawned it, whatever else is installed globally.
_PACKAGE_ROOT = str(pathlib.Path(__file__).resolve().parents[1])

__all__ = [
    "CLAUDE_DIRNAME",
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

#: The default recipe -- applies to every agent, unnamed. Lives in ``env_mgr/``
#: rather than ``env_mgr/recipes/`` so its path alone distinguishes it from the
#: recipes an agent names in ``recipes: [...]``. Absent is normal: nothing
#: declares it.
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

#: Where an external MCP declaration lives inside a component. MCP reaches the
#: model through the SDK's typed ``mcp_servers`` option, not this file, so this
#: module reads it and carries the servers to the backend as data.
MCP_FILENAME = ".mcp.json"

#: Subdirectory of a component holding executable tool definitions.
TOOLS_DIRNAME = "tools"

#: Everything a `.claude/` tree may hold is placed in the zone except these,
#: each here because it is read or relocated rather than skipped:
#: ``settings.json`` is read and merged; ``.mcp.json`` is read as data for
#: `Prepared.mcp_servers`; ``plugins/`` is relocated to
#: ``<config>/marketplaces/<name>/`` to avoid the harness's own `plugins/`.
_NOT_PLACED = frozenset({SETTINGS_FILENAME, MCP_FILENAME, "plugins"})

#: A stdio MCP server the component ships; auto-registered under its stem.
MCP_SUFFIX = ".mcp.py"

#: The install report's filename inside ``<zone>/logs``.
INSTALL_REPORT_FILENAME = "agent_assets.install.json"

#: Where a component's marketplace is copied before registration, under the
#: zone's config directory. Not `plugins/`: ``claude plugin install`` writes its
#: own bookkeeping there (``installed_plugins.json``, ``marketplaces/``,
#: ``cache/``), so a component's marketplace must land on a different name.
MARKETPLACES_DIRNAME = "marketplaces"

#: `bootstrap`, not `install`. `runner.run` defines that stage as install *then*
#: bootstrap, so a recipe item needing a post-install step — a marketplace add, a
#: login, a cache warm — is usable when `_run_recipe` returns. Under `install` it
#: would be installed and not usable, and the failure would surface inside the
#: agent's session rather than in the report.
_RECIPE_STAGE = "bootstrap"

#: ``${NAME}`` only. See `_expand` for why the bare ``$NAME`` form is excluded.
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: How long one recipe child may run, in seconds. A default; `install` takes it
#: as a parameter, since a slow mirror is a fact about a site, not this code.
#: Bounds the ``python -m env_mgr`` child directly, which `run_cmd`'s own
#: shell timeout cannot do for a networked install that hangs.
RECIPE_TIMEOUT_SECONDS = 20 * 60

#: `_run_cmd`'s return code for a killed child. Negative so that it cannot
#: collide with an exit status a real child produced.
_TIMED_OUT = -1


class InstallOutcome(NamedTuple):
    """The receiving type for `report.render_json`'s wire format, across the
    subprocess boundary. Deliberately not `env_mgr.outcome.Outcome`, below the
    decoupling wall (spec section 9). `level` carries one of
    `env_mgr.outcome.LEVELS`'s values verbatim, not re-validated.
    """

    level: str
    message: str
    #: No default: a mutable default on a `NamedTuple` would be shared by every
    #: instance. Every construction here passes a fresh mapping.
    details: dict[str, Any]


class AgentMaterial(NamedTuple):
    """Everything the recipes and the agent's own tree produced, as a value.

    `settings` reports what was merged into ``<config>/settings.json``;
    `report` is the per-install record.
    """

    env: dict[str, str]
    mcp_servers: dict[str, Any]
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
    """Install everything this agent asks for -- recipes, then its own tree -- into `config_dir`.

    Raises `PrepareRefused` for a declared-and-absent item; a failed install
    is a named `InstallOutcome` in the returned report instead.
    """
    os.makedirs(config_dir, exist_ok=True)
    child_env = _child_env(environ, config_dir)
    # An item's `cwd`, handed to the child as `--path`. The workspace when there
    # is one; the config directory otherwise, because it is inside the zone and
    # `install` has just created it — see `_run_recipe` on why this may not be
    # left to the recipe's own placeholder.
    recipe_cwd = workspace if workspace and os.path.isdir(workspace) else config_dir
    timeout = RECIPE_TIMEOUT_SECONDS if recipe_timeout is None else recipe_timeout

    trees = _addon_trees(agent_spec, staged_package=staged_package)

    # **Before any install**, for the measured merge-not-clobber reason in the
    # module docstring.
    settings: dict[str, Any] = {}
    for _, tree in trees:
        settings = _merge(settings, _read_json(os.path.join(tree, SETTINGS_FILENAME)))
    if settings:
        _write_json(os.path.join(config_dir, SETTINGS_FILENAME), settings)

    report: list[InstallOutcome] = []
    for path in _recipe_paths(agent_spec, staged_package=staged_package):
        report.extend(_run_recipe(path, child_env, recipe_cwd, timeout))

    # A name collision between this mapping and the SDK's other MCP sources is
    # `claude_sdk.py`'s to catch, not this loop's.
    mcp_servers: dict[str, Any] = {}
    for origin, tree in trees:
        outcomes, servers = _install_tree(
            tree,
            origin=origin,
            config_dir=config_dir,
            environ=child_env,
            recipe_cwd=recipe_cwd,
            timeout=timeout,
            agent_cli=agent_cli,
        )
        report.extend(outcomes)
        mcp_servers.update(servers)

    env: dict[str, str] = {}
    assets = _assets_dir(agent_spec, staged_package=staged_package)
    if assets is not None:
        env[paths.AGENT_ASSETS_ENV_VAR] = assets
    if logs_dir is not None:
        env[paths.INSTALL_REPORT_ENV_VAR] = _write_report(logs_dir, report)

    return AgentMaterial(
        env=env,
        mcp_servers=mcp_servers,
        settings=settings,
        report=tuple(report),
    )


# --------------------------------------------------------------------------- #
# Resolving what was declared


def _assets_dir(agent_spec: Any, *, staged_package: str | None) -> str | None:
    """``<staged package>/<AgentSpec.assets>``, or `None`.

    `None` covers no `assets` key, no staged package, or a pre-field spec.
    Declared and absent raises.
    """
    rel = _get(agent_spec, "assets")
    if not rel or not staged_package:
        return None
    # Joined under a check, not `os.path.join`: `assets` is author-written and
    # must not resolve outside the staged copy.
    path = contained_syntactically(str(rel), staged_package)
    if path is None:
        raise PrepareRefused(
            f"agent {_name(agent_spec)!r} declares assets {rel!r}, which does not "
            f"stay inside the staged package {staged_package!r}. This run is "
            f"pinned to that copy, so its material may not be reached by "
            f"climbing out of it"
        )
    if not os.path.isdir(path):
        raise PrepareRefused(
            f"agent {_name(agent_spec)!r} declares assets {rel!r} and "
            f"{path!r} is not a directory in the staged package. spec_loader fills "
            f"that field from a directory it found, so the staged copy is not the "
            f"package this spec was loaded from"
        )
    return path


def _addon_trees(agent_spec: Any, *, staged_package: str | None) -> list[tuple[str, str]]:
    """Every ``.claude/`` tree to install, as ``(origin, path)`` -- at most one.

    Per `docs/spec.provisioning.md` section 3 the copy route is the agent's own
    assets and nothing else; a missing ``.claude/`` inside it is not an error.
    """
    trees: list[tuple[str, str]] = []

    assets = _assets_dir(agent_spec, staged_package=staged_package)
    if assets is not None:
        tree = os.path.join(assets, CLAUDE_DIRNAME)
        if os.path.isdir(tree):
            trees.append(("the package's own assets", tree))
    return trees


def _package_recipe_path(*, staged_package: str | None) -> str | None:
    """The package layer's recipe inside the staged copy, or `None`.

    Admits exactly ``assets/main.env_recipe.yaml`` -- one spelling, unlike the
    agent layer's filename-convention search.
    """
    if not staged_package:
        return None
    path = os.path.join(staged_package, "assets", PACKAGE_RECIPE_BASENAME)
    return path if os.path.isfile(path) else None


def _recipe_paths(agent_spec: Any, *, staged_package: str | None) -> list[str]:
    """Every recipe YAML to run, in order: default, package, then agent-declared.

    See `docs/spec.provisioning.md` section 9.1. Layers concatenate rather
    than override; a version conflict *between* layers is not detected.
    """
    out: list[str] = []
    shipped = os.path.join(os.path.dirname(__file__), "recipes")

    if os.path.isfile(DEFAULT_RECIPE):
        out.append(DEFAULT_RECIPE)
    package_recipe = _package_recipe_path(staged_package=staged_package)
    if package_recipe is not None:
        out.append(package_recipe)

    for declared in _sequence(agent_spec, "recipes"):
        out.append(_resolve_recipe(declared, agent_spec, staged_package, shipped))
    return out


#: The two roots a recipe reference may name, and the whole of the vocabulary.
#: A tuple rather than a chain of ``elif``s so the refusal below can print it and
#: cannot drift from what is accepted.
RECIPE_SCHEMES = ("agent_sys", "package")

#: What a reference with no scheme is told. A migration guard for the removed
#: bare-name form; can be deleted once no ``recipes:`` list carries one.
_BARE_RECIPE_REMOVED = (
    "agent {agent!r} declares recipe {declared!r}, which names no root. The bare "
    "form was removed on 2026-09-04. Write 'agent_sys:{declared}' for a recipe "
    "this repository ships under env_mgr/recipes/, or 'package:<relpath>' for one "
    "this task package carries. It used to resolve by trying the package-relative "
    "path first and falling back to the shipped directory, so the declaration "
    "could not be read without knowing the resolution order, and a typo in a "
    "package path silently ran a different recipe"
)


def _resolve_recipe(
    declared: str, agent_spec: Any, staged_package: str | None, shipped: str
) -> str:
    """One ``<scheme>:<ref>`` reference to an existing file, or `PrepareRefused`.

    The scheme selects exactly one path; no candidate list, no
    first-that-exists.
    """
    scheme, sep, ref = declared.partition(":")
    if not sep:
        raise PrepareRefused(
            _BARE_RECIPE_REMOVED.format(agent=_name(agent_spec), declared=declared)
        )
    if scheme not in RECIPE_SCHEMES:
        raise PrepareRefused(
            f"agent {_name(agent_spec)!r} declares recipe {declared!r} and "
            f"{scheme!r} is not a recipe root. The roots are 'agent_sys:<name>' — "
            f"a recipe this repository ships under env_mgr/recipes/ — and "
            f"'package:<relpath>', one this task package carries, resolved against "
            f"the staged copy"
        )

    if scheme == "agent_sys":
        # **A name, and a separator in it is an error rather than a basename.**
        # The previous code called `os.path.basename` here, which silently turned
        # ``a/b`` into ``b`` — a normalisation nobody asked for, of exactly the
        # kind this change exists to remove. This root is ours and flat, so there
        # is no reading in which a separator was meant.
        if not ref or "/" in ref or os.sep in ref:
            raise PrepareRefused(
                f"agent {_name(agent_spec)!r} declares recipe {declared!r}; "
                f"'agent_sys:' takes the bare name of a file in env_mgr/recipes/, "
                f"never a path"
            )
        path = os.path.join(shipped, f"{ref}.yaml")
    else:
        # **``package:`` with nothing staged is a declaration that cannot be
        # honoured**, and saying so beats resolving to nothing. It is not the same
        # event as the file being missing and does not share its message.
        if not staged_package:
            raise PrepareRefused(
                f"agent {_name(agent_spec)!r} declares recipe {declared!r} and this "
                f"run has no staged task package to resolve 'package:' against"
            )
        # **Checked, for `_assets_dir`'s reason.** ``../../..`` climbs out of the
        # staged copy and an absolute value replaces it outright, and a recipe is
        # a file this module hands to a subprocess to execute. There is no
        # fallback behind this any more, so `None` here is final.
        inside = contained_syntactically(ref, staged_package)
        if inside is None:
            raise PrepareRefused(
                f"agent {_name(agent_spec)!r} declares recipe {declared!r}, which "
                f"does not stay inside the staged package {staged_package!r}. A "
                f"recipe is executed, so it may not be reached by climbing out of "
                f"the copy this run was pinned to"
            )
        path = inside

    if not os.path.isfile(path):
        raise PrepareRefused(
            f"agent {_name(agent_spec)!r} declares recipe {declared!r} and {path!r} "
            f"does not exist. Declared and absent is an error: skipped, the agent "
            f"would meet the absence as a failure of its own"
        )
    return path


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
) -> tuple[list[InstallOutcome], dict[str, Any]]:
    """One ``.claude/`` tree, placed in the zone. Place by default; name every
    exception in `_NOT_PLACED`. ``recipe.yaml`` sits beside ``.claude/``, not
    inside it, and is never placed.
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

    return outcomes, servers


def _place_tree(tree: str, *, origin: str, config_dir: str) -> list[InstallOutcome]:
    """Copy every member of a ``.claude/`` tree into the zone, bar the exceptions.

    A copy, not a symlink: a link out of the zone resolves to a path the
    kernel refuses. Symlinks are resolved at every depth (``dereference=True``).
    """
    if not os.path.isdir(tree):
        return []
    out: list[InstallOutcome] = []
    for name in sorted(os.listdir(tree)):
        if name in _NOT_PLACED:
            continue
        target = os.path.join(config_dir, name)
        # An overwrite between components is reported, never silent: `copy_out`
        # merges with `dirs_exist_ok`, so a file two components both ship ends
        # up holding only the later one's bytes. Precedence (later wins) is
        # unchanged; only the reporting is new.
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

    A collision is measured per file, not per directory, so two components
    shipping `skills/` collide only when they ship the same skill.
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

    Copied into the zone before registering, since Claude Code reads it from
    its source path at run time. An absent CLI or bad manifest is a `fail`.
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

    # Checked before the copy, on the un-joined name, with `contained_syntactically`
    # rather than `contained`: it rejects an absolute path or a climbing ``..``
    # without touching the filesystem, which the destination does not have yet.
    # `market` is also half of the ``<plugin>@<marketplace>`` CLI argument, so a
    # value that is not a plain directory name is wrong twice over.
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
        # A distinct check from the one above: that says "this is a name",
        # this says "the join stays inside".
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
        # The syntactic check above cannot see a symlink; this resolves both
        # sides now that the directory exists.
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

    A bundled server runs under `sys.executable`, registered at its *placed*
    path. A declared entry is expanded against the zone environment.
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
                # The names, not just the count: this is the only artefact that
                # records which servers a component declared. Catches
                # declared-nowhere at prepare time; does not catch one that
                # fails to start.
                {"names": sorted(str(k) for k in declared)},
            )
        )

    for source in _tool_files(tree, MCP_SUFFIX):
        name = os.path.basename(source)[: -len(MCP_SUFFIX)]
        placed = os.path.join(config_dir, TOOLS_DIRNAME, os.path.basename(source))
        if name in servers:
            # A same-tree collision: `.mcp.json` declaring `x` beside
            # `tools/x.mcp.py`. Reported, and the bundled one wins, because
            # this function can prove its file exists.
            out.append(
                InstallOutcome(
                    "warn",
                    f"{origin} declares MCP server {name!r} in {MCP_FILENAME} and also "
                    f"ships {TOOLS_DIRNAME}/{os.path.basename(source)}; the bundled "
                    f"server wins",
                    {"path": placed},
                )
            )
        # `env` stated explicitly, not left to inherit: this is the only place
        # a bundled server's environment can be stated. The whole zone mapping
        # is passed rather than a subset, since whether the SDK merges or
        # replaces the child's own environment is not relied on either way.
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
    """One recipe YAML through ``<sys.executable> -m env_mgr bootstrap <recipe>
    --json --path <cwd>``, as a child process. Bounded by `timeout`; a child
    producing no usable JSON becomes one `fail` outcome.
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

    Built once and passed, never set on this process: `agent/runner.py` is
    threaded and a global ``CLAUDE_CONFIG_DIR`` would race.
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

    An argv list, not a shell string, so a manifest-supplied value cannot be
    interpreted as shell syntax. A timeout returns ``-1`` rather than raising.
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

    Promised rather than discoverable: an agent needs a named file to state
    what it has installed. ``<zone>/logs`` is already granted.
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

    An unresolved name raises rather than passing through unexpanded, unlike
    `os.path.expandvars`. ``${VAR}`` only, no bare ``$VAR``.
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

    Lists replace rather than concatenate: two components' ``hooks`` lists
    concatenated would silently run both sets on every matching call.
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

    A file that exists and does not parse raises, rather than being skipped:
    continuing would start a session silently missing what it declares.
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
    """`material.py`'s reader, duplicated deliberately rather than imported.

    Accepts either a mapping or a model: an `AgentSpec` in production, a dict
    in tests.
    """
    if isinstance(agent_spec, dict):
        return agent_spec.get(key)
    return getattr(agent_spec, key, None)
