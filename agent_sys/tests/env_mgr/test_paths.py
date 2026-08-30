# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The path environment-variable system — `refine.task_package.define.md` item 3.

Two halves are asserted here and the second matters more than the first: the
names that **are** exported, and the four the user asked for that are **not**,
with the measurement that decided it. A negative ruling with no test is a ruling
the next agent reverses in good faith.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from env_mgr import paths
from env_mgr.fs.domain import DomainKind, DomainRegistry
from env_mgr.fs.path import contained
from env_mgr.prepare import PACKAGE_ENV_VAR, prepare
from env_mgr.protocols import Tier
from env_mgr.sync import Direction, _ends, remote_root

from .stubs import AgentSpec, Task, context


@pytest.fixture
def main_repo(tmp_path: Path) -> str:
    repo = tmp_path / "main"
    repo.mkdir()
    env = dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    git("config", "extensions.preciousObjects", "true")
    (repo / "f.txt").write_text("base\n")
    git("add", "f.txt")
    git("commit", "-qm", "base")
    return str(repo)


def _ctx(tmp_path: Path, main_repo: str, *, kinds, mapping=None, package=None):
    """A `Context` whose registered domain **kinds** are the parameter.

    Which kinds are registered is what decides which subdirectories
    `layout.create` makes, and therefore which of these variables can exist at
    all — so it is the axis this file varies rather than a fixture constant.
    """
    from env_mgr.isolation.policy import interpreter_grants

    reg = DomainRegistry()
    for name, kind in kinds:
        reg.register(name, str(tmp_path / "root"), kind)
    ctx = context(
        domains=reg,
        store_root=str(tmp_path / "store"),
        main_repo=main_repo,
        interpreter_grants=interpreter_grants(),
        tier=Tier.PRODUCTION,
        mapping=mapping,
    )
    return ctx._replace(package=package) if package is not None else ctx


ALL_KINDS = (
    ("store", DomainKind.HANDOFF_STORAGE),
    ("ws", DomainKind.WORKSPACE),
    ("play", DomainKind.PLAYGROUND),
)


# --------------------------------------------------- the names that are exported


def test_every_zone_directory_has_a_declared_name(tmp_path: Path, main_repo: str) -> None:
    """The user's `my_agent_workspace` / `my_agent_playground`, and the two
    neighbours that had no name either.

    Before this, `<zone>/workspace`, `playground`, `handoffs` and `logs` existed,
    were granted read-write, and a body could reach none of them except by
    parsing this module's directory layout — which
    `engineer_principle.md` §4.4 names as the smell, and which
    `examples/demo/logic/store.py` had already become a reader of.
    """
    ctx = _ctx(tmp_path, main_repo, kinds=ALL_KINDS)
    task = Task()
    env = prepare(task, task.push_execution(), ctx).environment

    zone = env[paths.ZONE_ENV_VAR]
    assert env[paths.WORKSPACE_ENV_VAR] == os.path.join(zone, "workspace")
    assert env[paths.PLAYGROUND_ENV_VAR] == os.path.join(zone, "playground")
    assert env[paths.HANDOFFS_ENV_VAR] == os.path.join(zone, "handoffs")
    assert env[paths.LOGS_ENV_VAR] == os.path.join(zone, "logs")


def test_the_zone_variable_is_the_attempts_own_root(tmp_path: Path, main_repo: str) -> None:
    """`AGENT_SYS_MY_ZONE` is `Prepared.zone.root` and not a recomputation of it."""
    ctx = _ctx(tmp_path, main_repo, kinds=ALL_KINDS)
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)
    assert prepared.environment[paths.ZONE_ENV_VAR] == prepared.zone.root


def test_every_exported_path_is_one_the_policy_grants(tmp_path: Path, main_repo: str) -> None:
    """**The rule this family had to obey to exist at all.**

    `env_mgr/README.md` states it for `AGENT_SYS_OUTPUT_<KIND>`: *exported and
    granted agree by construction*, because "an exported path we did not grant
    would be the evaporating allow-list one level up: the body failing on our own
    instruction". `prepare` grants `Granted(zone.root, Mode.READ_WRITE)` and
    permissions cover a subtree recursively, so containment in the zone **is**
    the grant — which is why every name in this family is zone-relative and why
    the four `*_root` names below are not in it.
    """
    ctx = _ctx(tmp_path, main_repo, kinds=ALL_KINDS)
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)
    ours = [paths.ZONE_ENV_VAR, *(name for name, _ in paths._SUBDIRS)]
    for name in ours:
        assert contained(prepared.environment[name], prepared.zone.root), name


def test_a_directory_that_does_not_exist_gets_no_name(tmp_path: Path, main_repo: str) -> None:
    """A zone's subdirectories are one per **registered domain kind**, so a run
    with no `PLAYGROUND` domain has no `<zone>/playground`.

    Exporting the name anyway would instruct a body to use a path that is not
    there. `grants.output_paths` settled the shape: absent, never
    present-and-empty. The two positive controls are the point — without them a
    missing key could mean the whole family failed to export.
    """
    ctx = _ctx(
        tmp_path,
        main_repo,
        kinds=(("store", DomainKind.HANDOFF_STORAGE), ("ws", DomainKind.WORKSPACE)),
    )
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)

    assert not os.path.isdir(os.path.join(prepared.zone.root, "playground"))
    assert paths.PLAYGROUND_ENV_VAR not in prepared.environment
    # Controls: the family did export, and it exported the neighbour.
    assert paths.ZONE_ENV_VAR in prepared.environment
    assert paths.WORKSPACE_ENV_VAR in prepared.environment


# ------------------------------------------------- the package, folded in not moved


def test_the_task_package_variable_is_unchanged(tmp_path: Path, main_repo: str) -> None:
    """`AGENT_SYS_TASK_PACKAGE` was the first member of this family and keeps its
    spelling, its value, and its importability from `env_mgr.prepare`.

    `tests/cli/test_isolation_shown.py` imports `PACKAGE_ENV_VAR` from there and
    `examples/demo/**` reads the literal, so moving the definition into `paths`
    had to leave both alone.
    """
    package = tmp_path / "pkg"
    (package / "bin").mkdir(parents=True)
    (package / "bin" / "run.sh").write_text("#!/bin/sh\n")
    ctx = _ctx(tmp_path, main_repo, kinds=ALL_KINDS, package=str(package))
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)

    assert PACKAGE_ENV_VAR == "AGENT_SYS_TASK_PACKAGE"
    assert prepared.environment[PACKAGE_ENV_VAR] == prepared.staged_package
    assert Path(prepared.staged_package, "bin", "run.sh").exists()


def test_the_package_name_has_exactly_one_writer(tmp_path: Path, main_repo: str) -> None:
    """`prepare.PACKAGE_ENV_VAR` is `paths.PACKAGE_ENV_VAR`, not a second copy.

    Two module-level string literals would satisfy every other assertion in this
    file and drift on the first rename. `engineer_principle.md` §1: never let an
    invariant have two writers.
    """
    assert PACKAGE_ENV_VAR is paths.PACKAGE_ENV_VAR


def test_no_package_configured_exports_no_package_name(tmp_path: Path, main_repo: str) -> None:
    """`stage_package` returns `None` when `Context.package` is unset, and this
    module refuses to recompute `<zone>/package` behind its back — a path that
    would name a directory nothing created."""
    ctx = _ctx(tmp_path, main_repo, kinds=ALL_KINDS)
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)
    assert prepared.staged_package is None
    assert PACKAGE_ENV_VAR not in prepared.environment
    assert paths.ZONE_ENV_VAR in prepared.environment  # control: the family ran


# ------------------------------------------------------------------- the remote half


def test_no_mapping_exports_no_remote_name(tmp_path: Path, main_repo: str) -> None:
    """Today's every production configuration. `cli/environment.py` passes an
    empty mapping and states that it is legitimate rather than a degradation, so
    this is the shape that ships."""
    ctx = _ctx(tmp_path, main_repo, kinds=ALL_KINDS)
    task = Task()
    env = prepare(task, task.push_execution(), ctx).environment
    assert [k for k in env if k.endswith(paths.REMOTE_SUFFIX)] == []
    assert paths.ZONE_ENV_VAR in env  # control


def test_a_mapping_gives_every_local_name_a_far_side(tmp_path: Path, main_repo: str) -> None:
    """The user's `_romote` half, read as `_remote`.

    The far side mirrors the near side because `sync` rsyncs the zone whole and
    creates `playground/` on the far side explicitly even though it excludes the
    contents — spec §6.4's *"remote workspace, playground, handoff storage"*.
    """
    far = tmp_path / "far"
    ctx = _ctx(tmp_path, main_repo, kinds=ALL_KINDS, mapping={str(tmp_path / "root"): str(far)})
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)
    env = prepared.environment

    rel = os.path.relpath(prepared.zone.root, str(tmp_path / "root"))
    assert env[paths.remote_name(paths.ZONE_ENV_VAR)] == os.path.join(str(far), rel)
    for name, subdir in paths._SUBDIRS:
        assert env[paths.remote_name(name)] == os.path.join(str(far), rel, subdir)


def test_the_remote_playground_is_named_although_it_is_not_synced(
    tmp_path: Path, main_repo: str
) -> None:
    """Spec §6.2 says the playground's **contents** are not synced; §6.4 says the
    far side still has one, and `sync` creates the directory there explicitly
    after excluding everything in it.

    So the path is derivable and correct while the contents are independent, and
    the variable carries the former. Flagged for the user: §6.4's *"not mapped at
    all"* admits a reading in which this name should not exist.
    """
    far = tmp_path / "far"
    ctx = _ctx(tmp_path, main_repo, kinds=ALL_KINDS, mapping={str(tmp_path / "root"): str(far)})
    task = Task()
    env = prepare(task, task.push_execution(), ctx).environment
    assert env[paths.remote_name(paths.PLAYGROUND_ENV_VAR)].endswith("playground")


def test_remote_root_answers_none_where_ends_refuses(tmp_path: Path) -> None:
    """One walk over the mapping, two callers, two shapes of miss.

    `_ends` cannot build an rsync without both ends and raises with the
    diagnostic it always did; an environment variable for an unconfigured side is
    simply absent. Extracting the walk had to preserve the first exactly.
    """
    from env_mgr.fs.zone import Zone

    zone = Zone(task_id="t", attempt=0, root=str(tmp_path / "root" / "task.t.0.aa"))
    covered = {str(tmp_path / "root"): "/far"}

    assert remote_root(zone, covered) == os.path.join("/far", "task.t.0.aa")
    assert remote_root(zone, {"/somewhere/else": "/far"}) is None
    assert _ends(zone, covered, Direction.LOCAL_TO_REMOTE) == (
        zone.root,
        os.path.join("/far", "task.t.0.aa"),
    )
    with pytest.raises(KeyError, match="no mapping covers zone"):
        _ends(zone, {"/somewhere/else": "/far"}, Direction.LOCAL_TO_REMOTE)


# -------------------------------------------------- the four names that are refused


def test_the_root_names_the_user_asked_for_are_not_exported(tmp_path: Path, main_repo: str) -> None:
    """**A measurement, not an omission.**

    `refine.task_package.define.md` item 3 asks for `agent_workspace_root`,
    `agent_handoff_root` and `agent_playground_root`. Each would resolve to a
    registered *domain* root, which sits outside the zone. Measured against a
    real Landlock ruleset built from exactly the policy `prepare` composes, with
    an in-zone positive control succeeding in the same confined child
    (`scratch/ui-yaml-2026-08/w2/p13_are_the_root_paths_reachable.py`): all four
    give `EACCES`, and all four read cleanly unconfined — a denial, not an
    absence.

    Exporting them would break the rule
    `test_every_exported_path_is_one_the_policy_grants` holds for the rest of the
    family. `AGENT_SYS_MY_ZONE` is the one root in the user's sense that is
    granted and is exported in their place.

    This test exists so that adding them later is a deliberate act with the
    measurement in front of whoever does it, rather than a helpful completion of
    a list.
    """
    ctx = _ctx(tmp_path, main_repo, kinds=ALL_KINDS)
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)

    refused = ("AGENT_SYS_WORKSPACE_ROOT", "AGENT_SYS_HANDOFF_ROOT", "AGENT_SYS_PLAYGROUND_ROOT")
    for name in refused:
        assert name not in prepared.environment
    # And, whatever they were named, no exported path is outside the zone.
    outside = {
        key: value
        for key, value in prepared.environment.items()
        if key.startswith("AGENT_SYS_MY_") and not contained(value, prepared.zone.root)
    }
    assert outside == {}


def test_a_declared_env_still_outranks_every_path_name(tmp_path: Path, main_repo: str) -> None:
    """The new source sits **before** `material.deploy`, so an agent spec's `env`
    still wins — the precedence
    `test_a_declared_env_overrides_every_contributor` pins for the five sources
    that came first, extended to the sixth rather than excepted from it."""
    ctx = _ctx(tmp_path, main_repo, kinds=ALL_KINDS)
    task = Task()

    plain = prepare(task, task.push_execution(), ctx).environment
    names = [paths.ZONE_ENV_VAR, paths.WORKSPACE_ENV_VAR, paths.PLAYGROUND_ENV_VAR]
    for name in names:
        assert name in plain, f"{name} is not set at all, so overriding it proves nothing"

    declared = {name: f"/declared/{name.lower()}" for name in names}
    task2 = Task()
    overridden = prepare(task2, task2.push_execution(), ctx, AgentSpec(env=declared)).environment
    for name, value in declared.items():
        assert overridden[name] == value
