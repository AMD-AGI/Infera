# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The composition. Design §11 — **the order is the design.**

Since the step-7 split, `prepare` **checks** that a mechanism exists and
`Prepared.spawn` **applies** it in the child. So nothing here confines the
pytest process and no test has to fork to stay safe from it — the tests that
exercise real confinement go through `spawn`, which is where the syscall now
happens.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from env_mgr.fs.domain import DomainKind, DomainRegistry
from env_mgr.fs.path import contained as contained_path
from env_mgr.isolation.policy import interpreter_grants
from env_mgr.isolation.probe import Availability
from env_mgr.prepare import PACKAGE_ENV_VAR, EnvManager, Prepared, prepare
from env_mgr.protocols import Mode, NoConfinement, PrepareRefused, Tier, UnresolvedGrant
from task_graph.ids import HandoffId

from .stubs import AgentSpec, Execution, Grant, Handoff, Permissions, Task, context


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


@pytest.fixture
def ctx(tmp_path: Path, main_repo: str):
    reg = DomainRegistry()
    reg.register("store", str(tmp_path / "root"), DomainKind.HANDOFF_STORAGE)
    reg.register("ws", str(tmp_path / "root"), DomainKind.WORKSPACE)
    reg.register("play", str(tmp_path / "root"), DomainKind.PLAYGROUND)
    # The interpreter is granted, as a real `Context` would: it lives under
    # `$HOME` on every conda / pyenv / uv / venv install and the default set
    # deliberately excludes it. Without this a confined child cannot exec
    # python3 at all, and `subprocess` reports it **in the parent**, naming the
    # interpreter rather than the sandbox — which is measurement M3, and which
    # this fixture reproduced the first time `spawn` ran a real child.
    return context(
        domains=reg,
        store_root=str(tmp_path / "store"),
        main_repo=main_repo,
        interpreter_grants=interpreter_grants(),
        tier=Tier.PRODUCTION,
    )


def test_the_zone_and_the_workspace_exist_before_confinement(ctx) -> None:
    """Everything before step 7 writes outside the zone by design, and none of
    it is possible afterwards."""
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)
    assert os.path.isdir(prepared.zone.root)
    assert os.path.isdir(os.path.join(prepared.workspace.path, ".git"))
    # Since the split, step 7 reports the confinement `spawn` **will** apply
    # rather than one it has applied. Nothing has been confined yet — asserted
    # by the fact that this process can still write outside the zone, which is
    # the property a runner thread depends on.
    assert prepared.confinement.mechanism in ("bwrap", "landlock")
    open(os.path.join(str(Path(prepared.zone.root).parents[2]), "still-free.txt"), "w").close()


def test_the_policy_grants_the_zone_read_write(ctx) -> None:
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)
    zone_entry = next(g for g in prepared.policy.granted if g.path == prepared.zone.root)
    assert zone_entry.mode is Mode.READ_WRITE


def test_the_granted_set_is_assembled_in_the_order_the_spec_lists(ctx) -> None:
    """Default system set, the task's own zone, whatever its permissions name,
    nothing else."""
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)
    paths = [g.path for g in prepared.policy.granted]
    assert paths[0] == "/usr"
    assert paths[len(paths) - 1 - paths[::-1].index(prepared.zone.root)] >= ""
    assert prepared.zone.root in paths


def test_an_unresolved_grant_stops_the_task(ctx, tmp_path: Path) -> None:
    """`prepare` does not catch it, for the same reason it does not catch
    `NoConfinement`: the task does not start."""
    hid = HandoffId.new()
    task = Task(inputs=[hid], permissions=Permissions((Grant(kind="trace"),)))
    execution = Execution(attempt=0, input_versions={hid: 0})
    ctx_with = context(
        domains=ctx.domains,
        store_root=ctx.store_root,
        main_repo=ctx.main_repo,
        handoffs={hid: Handoff(id=hid, type="")},  # the field nobody filled
    )
    with pytest.raises(UnresolvedGrant):
        prepare(task, execution, ctx_with)


def test_a_mapping_makes_prepare_mirror_the_zone(tmp_path: Path, main_repo: str) -> None:
    """The copy path, composed rather than called directly — and the artefact,
    not the return value.

    `test_sync.py` covers `sync()` and `test_paths.py` covers the `_REMOTE`
    environment names, both green for as long as they have existed. Neither
    looks at the far side of a tree that `prepare` produced, and `prepare` is
    what a run calls. `SyncReport.sent` is not evidence either: it is parsed out
    of rsync's own summary, so a `sent` of 3 over a copy that landed somewhere
    else would read identically. This asserts on the directory.

    Its control is `test_no_mapping_leaves_no_far_side` immediately below, which
    is the configuration everything shipped with.
    """
    import shutil

    if shutil.which("rsync") is None:
        pytest.skip("rsync is not installed")

    local = tmp_path / "root"
    far = tmp_path / "far"
    reg = DomainRegistry()
    reg.register("store", str(local), DomainKind.HANDOFF_STORAGE)
    reg.register("play", str(local), DomainKind.PLAYGROUND)
    ctx_mapped = context(
        domains=reg,
        store_root=str(tmp_path / "store"),
        main_repo=main_repo,
        interpreter_grants=interpreter_grants(),
        mapping={str(local): str(far)},
    )
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx_mapped)

    mirrored = far / os.path.relpath(prepared.zone.root, local)
    assert mirrored.is_dir(), f"{mirrored} is not there; nothing was copied"
    # A directory alone would pass on an rsync that copied an empty tree, so the
    # assertion is that a file the zone has arrived with its bytes.
    local_files = {
        os.path.relpath(os.path.join(d, f), prepared.zone.root)
        for d, _, fs in os.walk(prepared.zone.root)
        for f in fs
        if os.path.relpath(os.path.join(d, f), prepared.zone.root).split(os.sep)[0] != "playground"
    }
    assert local_files, "the zone had no files at all; this proves nothing"
    for rel in local_files:
        assert (mirrored / rel).is_file(), f"{rel} did not cross"

    # Criterion 16's first half: the playground's *directory* is there, which
    # `sync` creates explicitly after excluding everything in it. That the
    # *contents* are excluded is not asserted here and could not be — a zone
    # `prepare` has just built has an empty playground, so the assertion would
    # be vacuous. `test_sync.py::test_playground_not_synced` proves it against a
    # playground with a file in it.
    assert (mirrored / "playground").is_dir()


def test_no_mapping_leaves_no_far_side(tmp_path: Path, main_repo: str) -> None:
    """The non-vacuity control for the test above, and the shipped default.

    Same zone, same `prepare`, no mapping: `if ctx.mapping:` is false, nothing
    is copied, and the far side is not so much as created. Without this, a test
    that found a mirror could not tell a working sync from a directory some
    other part of `prepare` had made.
    """
    local = tmp_path / "root"
    far = tmp_path / "far"
    reg = DomainRegistry()
    reg.register("store", str(local), DomainKind.HANDOFF_STORAGE)
    reg.register("play", str(local), DomainKind.PLAYGROUND)
    ctx_plain = context(
        domains=reg,
        store_root=str(tmp_path / "store"),
        main_repo=main_repo,
        interpreter_grants=interpreter_grants(),
    )
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx_plain)

    assert Path(prepared.zone.root).is_dir()  # control: the zone was built
    assert not far.exists()


def test_a_sync_conflict_refuses(tmp_path: Path, main_repo: str) -> None:
    """Refusing converts silent data loss into a stopped task, which is the
    difference the open question is actually about."""
    import shutil

    if shutil.which("rsync") is None:
        pytest.skip("rsync is not installed")

    local = tmp_path / "local"
    remote = tmp_path / "remote"
    reg = DomainRegistry()
    reg.register("store", str(local), DomainKind.HANDOFF_STORAGE)
    task = Task()
    execution = task.push_execution()

    from env_mgr.fs import layout

    zone = layout.create(task, execution, reg)
    (Path(zone.root) / "handoffs" / "in.txt").write_text("LOCAL edit")
    far = Path(remote) / os.path.relpath(zone.root, local) / "handoffs"
    far.mkdir(parents=True)
    (far / "in.txt").write_text("REMOTE edit")

    ctx = context(
        domains=reg,
        store_root=str(tmp_path / "store"),
        main_repo=main_repo,
        mapping={str(local): str(remote)},
    )
    with pytest.raises(PrepareRefused, match="both sides"):
        prepare(task, execution, ctx)


def test_no_confinement_propagates_out_of_prepare(ctx) -> None:
    """Criterion 8 at the composition level: nothing between `select` and the
    caller converts the refusal into a warning."""
    task = Task()
    with pytest.raises(NoConfinement):
        prepare(
            task,
            task.push_execution(),
            ctx,
            availability=Availability(bwrap=None, landlock_abi=None),
        )


def test_material_is_deployed_before_confinement(ctx, tmp_path: Path) -> None:
    """Step 6b. Deploying is writing into the zone, and step 7 makes writing
    impossible — so it is beside handoff staging, not after the sandbox."""
    rule = tmp_path / "rules" / "style.md"
    rule.parent.mkdir()
    rule.write_text("# house style\n")
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx, AgentSpec(rules=(str(rule),)))
    assert Path(prepared.zone.root, "config", "rules", "style.md").read_text() == "# house style\n"


def test_env_manager_exposes_exactly_these(ctx) -> None:
    """The guard, kept through the amendment rather than deleted under it.

    It was `test_env_manager_has_exactly_one_method`, and the rule it guarded
    named the runner as the hazard: *"a second is how the runner would start
    making environment decisions."* `prepare_validation` was ruled in after two
    modules were found answering *where does a validation go*, and it is a
    different caller asking the layout owner a question only the layout owner
    can answer.

    **What the guard was for survives**: pinning the set means the next method
    still fails a test and still needs a decision. Deleting it would have
    removed the pressure; leaving it at one would have blocked a correct change.

    It has since fired twice and been right both times. `place_zone` is the
    third, added after `demo` found that a non-leaf never gets a zone and
    `agent` measured that the obvious repair — calling `prepare` — would be
    refused. The guard is what made each of those a decision with a stated
    reason rather than an accretion.
    """
    public = {
        name
        for name in dir(EnvManager)
        if not name.startswith("_") and callable(getattr(EnvManager, name))
    }
    assert public == {"place_zone", "prepare", "prepare_validation"}


def test_env_manager_satisfies_the_frozen_two_argument_call(ctx) -> None:
    """`interfaces.md` §4.6 and `protocols.py` both declare
    ``prepare(task, execution)``; design rev. 4 §11.5 adds ``agent_spec``. The
    third parameter has a default, so both calls work and neither side is
    quietly changed. The README reports the seam."""
    import inspect

    from env_mgr.protocols import EnvManager as EnvManagerProtocol

    signature = inspect.signature(EnvManager.prepare)
    assert list(signature.parameters) == ["self", "task", "execution", "agent_spec"]
    assert signature.parameters["agent_spec"].default is None
    # The Protocol is not `runtime_checkable`, so the agreement is asserted on
    # the shape rather than with `isinstance`: every parameter the frozen
    # declaration names is positional here, in the same order.
    declared = list(inspect.signature(EnvManagerProtocol.prepare).parameters)
    assert list(signature.parameters)[: len(declared)] == declared

    # And the two-argument call really runs — in a forked child, because
    # `EnvManager` does not expose `confine` and confining pytest would poison
    # every later test. That is the design's whole reason for forking.
    task = Task()
    execution = task.push_execution()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never reports coverage
        try:
            prepared = EnvManager(ctx).prepare(task, execution)  # two arguments
            os._exit(0 if prepared.confinement is not None else 1)
        except BaseException:
            os._exit(2)
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


def test_prepare_does_not_confine_and_spawn_does(ctx, tmp_path: Path) -> None:
    """The step-7 split, stated as the two halves it became.

    `prepare` **checks**: a mechanism must exist or the task does not start, and
    it refuses before the workspace is cut rather than after. `spawn`
    **applies**, in the child.

    Design §11.1's *"confinement last"* survives in the form that mattered. That
    rule existed so the supervisor and every prior process stay outside the
    domain, and moving the syscall into the child achieves it **by construction
    rather than by ordering** — this test needs no fork to stay safe, where the
    version before the split did.
    """
    import errno

    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)

    # prepare checked, and applied nothing: this process is still free.
    outside = tmp_path / "supervisor-writes-this.txt"
    outside.write_text("still here")
    assert prepared.confinement.filesystem is True

    # spawn applied: the child is not.
    script = f"import sys\ntry:\n open({str(outside)!r}, 'a')\nexcept OSError as e:\n sys.exit(e.errno)\nsys.exit(0)"
    assert prepared.spawn([os.sys.executable, "-c", script]).wait(timeout=60) == errno.EACCES

    # And the supervisor is *still* free afterwards, which is the property a
    # runner thread depends on and which the pre-split arrangement destroyed.
    outside.write_text("and again")
    assert outside.read_text() == "and again"


def test_prepared_carries_the_agents_environment(ctx, tmp_path: Path) -> None:
    """`material.deploy` computes it and rev. 1 dropped it on the floor.

    `CLAUDE_CONFIG_DIR` is the load-bearing one: measured, with ``~/.claude``
    granted a confined agent read the **operator's personal** `CLAUDE.md` and
    obeyed its language rule. Pointing it into the zone removes the ``$HOME``
    grant entirely, and the runner cannot do that with a value it never sees.
    """
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx, AgentSpec(env={"HF_HOME": "/models"}))
    assert prepared.environment["CLAUDE_CONFIG_DIR"] == os.path.join(prepared.zone.root, "config")
    assert prepared.environment["CLAUDE_CODE_TMPDIR"] == os.path.join(prepared.zone.root, "tmp")
    assert prepared.environment["HF_HOME"] == "/models"


def test_the_environment_is_not_writable_through(ctx) -> None:
    """Never hand out a mutable handle to internal state. A `dict` default on a
    NamedTuple is shared by every instance, which is one edit from one task's
    environment leaking into another's."""
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx, AgentSpec())
    with pytest.raises(TypeError):
        prepared.environment["INJECTED"] = "x"  # type: ignore[index]


def test_prepared_matches_the_declared_surface(ctx) -> None:
    """`prepare.Prepared` implements what `protocols.Prepared` declares.

    Two declarations of one shape, which is the duplication
    `engineer_principle.md` §1 names — admissible here for the same reason
    `Policy.with_` and `Zone.contains` are, because the declaration's bodies are
    `...` and a `NamedTuple` cannot carry a real method and stay a declaration.
    **The price of that is this test**, and without it the honest move would be
    to have only one.

    **Three things can diverge and this checked one of them.** A `NamedTuple`
    pair can drift in field *names*, in *defaults*, and in *annotations*; the
    names were asserted and the other two were not. The gap was found the
    ordinary way — a field arrived whose default was spelled `_NO_ENV` on one
    side and `MappingProxyType({})` on the other, and nothing here had an
    opinion. A test whose stated justification is *being the price of two
    declarations* has to price all of what two declarations can cost.

    All three are now asserted: names and defaults by equality, annotations by
    equality **with a named exemption list**.

    An exemption list rather than a subtyping rule, on `main`'s ruling. The
    principle is *the implementation may be wider than the declaration where it
    cannot name the narrow type* — but "cannot name" is a fact about a module's
    import surface, not about the types, and encoding it as a compatibility
    check would make the test cleverer than the thing it guards. A list makes a
    **third** exemption a decision someone has to justify rather than a line
    someone adds.

    **One entry, and it is forced.** Naming `HandoffId` means importing
    `task_graph` into `prepare.py`; `protocols.py` may (`:22`), `prepare.py`
    imports nothing from it, and the seam is deliberately one-way.

    It reached one entry by both of the other two leaving, in opposite
    directions, and that is the point of keeping a list rather than a rule:

    `zone` was exempt and should not have been — `Any` only because it sat
    beside something that had to be, when `from env_mgr.fs.zone import Zone` is
    intra-package and breaks nothing. Narrowed in `af97f51`. **An exemption list
    earns its keep only if every entry is forced**; the moment a fixable thing
    rests in one because of its neighbours, the list stops meaning *cannot* and
    starts meaning *did not*, which is a stale docstring wearing a structure
    built to prevent them.

    `confinement` **was** a third entry and is now fixed rather than exempted:
    the declaration typed it non-optional while `None` is how the system says
    *unconfined*. It was accurate until `ad730a2` made `None` reachable, and the
    comment explaining `confinement is None` sat nine lines above the annotation
    denying it the whole time. Widened on a ruling, because the frozen
    cross-module surface is not something a test should quietly encode either
    answer to.
    """
    import inspect

    from env_mgr.protocols import Prepared as Declared

    assert Prepared._fields == Declared._fields
    assert inspect.signature(Prepared.wrap_argv) == inspect.signature(Declared.wrap_argv)

    # Value equality, not identity: the two sides may reach the same empty
    # mapping by different names, and that is a spelling difference rather than
    # a divergence. What must not differ is what a caller who omits the field
    # gets -- and a type change (an empty `dict` where the other side has a
    # `MappingProxyType`) is a real divergence that `==` alone would miss.
    assert Prepared._field_defaults == Declared._field_defaults
    for name, declared in Declared._field_defaults.items():
        assert type(Prepared._field_defaults[name]) is type(declared), (
            f"{name}: implementation defaults to {type(Prepared._field_defaults[name])}, "
            f"the declaration to {type(declared)} -- same value, different mutability "
            f"contract"
        )

    #: Fields the implementation may annotate more loosely, and only these. See
    #: the docstring for why each one cannot say what the declaration says.
    WIDER_IN_THE_IMPLEMENTATION = {"output_paths"}

    # Compared as source text rather than resolved types: `from __future__ import
    # annotations` makes both sides strings, and resolving them would need
    # `prepare.py` to be able to import the very names it deliberately does not.
    for name in Declared._fields:
        if name in WIDER_IN_THE_IMPLEMENTATION:
            continue
        implemented = Prepared.__annotations__[name]
        declared_type = Declared.__annotations__[name]
        assert str(implemented) == str(declared_type), (
            f"{name}: the implementation says {implemented}, the declaration says "
            f"{declared_type}. If the implementation cannot name the declared type, "
            f"add it to WIDER_IN_THE_IMPLEMENTATION with the reason -- but check "
            f"first that it is forced, because `confinement` looked forced and was "
            f"a defect"
        )

    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)
    (
        zone,
        workspace,
        policy,
        confinement,
        sync,
        environment,
        agent_cli,
        permissions_enforced,
        output_paths,
        staged_package,
    ) = prepared
    assert (zone, policy, environment) == (prepared.zone, prepared.policy, prepared.environment)
    assert agent_cli is prepared.agent_cli
    # `staged_package` is the §4.16 copy, and `agent` resolves a package-relative
    # `entry` against it. Unpacked here rather than only attribute-read, because
    # a positional unpack is what breaks first when a field is inserted rather
    # than appended -- which is the whole reason this test exists.
    assert staged_package is prepared.staged_package
    assert permissions_enforced is prepared.permissions_enforced
    assert output_paths is prepared.output_paths


def test_wrap_argv_is_a_no_op_under_landlock(ctx, availability) -> None:
    """The process is **already** confined when `prepare` returned, so there is
    nothing to prepend. Run in a forked child, because it really confines."""
    if not availability.landlock_abi or availability.bwrap:
        pytest.skip("this machine does not select Landlock")
    task = Task()
    execution = task.push_execution()

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never reports coverage
        os.close(read_fd)
        try:
            prepared = prepare(task, execution, ctx)
            same = prepared.wrap_argv(["python3", "-c", "pass"]) == ["python3", "-c", "pass"]
            os.write(write_fd, b"1" if same else b"0")
        except BaseException:
            os.write(write_fd, b"e")
        finally:
            os.close(write_fd)
        os._exit(0)

    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as fh:
        verdict = fh.read()
    os.waitpid(pid, 0)
    assert verdict == b"1", f"the child reported {verdict!r}"


def test_wrap_argv_builds_the_bwrap_command(ctx, tmp_path: Path, monkeypatch) -> None:
    """Rung 1's policy becomes real at ``exec``, and this is where.

    ``bwrap`` is absent on this machine, so the binary is faked to exercise the
    branch. What is asserted is the argv — the two unshares that are rung 1's
    extra properties, the zone bound writable, and the command last.
    """
    fake = tmp_path / "bin" / "bwrap"
    fake.parent.mkdir()
    fake.write_text('#!/bin/sh\nexec "$@"\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake.parent) + os.pathsep + os.environ["PATH"])

    task = Task()
    prepared = prepare(
        task,
        task.push_execution(),
        ctx,
        availability=Availability(bwrap=str(fake), landlock_abi=None),
    )
    assert prepared.confinement.mechanism == "bwrap"
    argv = prepared.wrap_argv(["python3", "work.py"])
    assert argv[0] == str(fake)
    assert "--unshare-net" in argv and "--unshare-pid" in argv
    # The zone is bound writable. Searched for as a pair rather than by the
    # first `--bind`, which is `/dev/null` — writable since git dies without it.
    pairs = [(argv[i], argv[i + 1]) for i in range(len(argv) - 1)]
    assert ("--bind", prepared.zone.root) in pairs
    assert ("--ro-bind", "/usr") in pairs
    assert argv[-2:] == ["python3", "work.py"]
    assert argv[-3] == "--"


def test_wrap_argv_refuses_when_the_binary_vanished(ctx, tmp_path: Path, monkeypatch) -> None:
    """Resolved at exec time rather than remembered from probe time — the same
    rule as canonicalising per check. If bubblewrap was selected and is gone,
    the task does not start rather than running unwrapped."""
    fake = tmp_path / "bin" / "bwrap"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake.parent) + os.pathsep + os.environ["PATH"])

    task = Task()
    prepared = prepare(
        task,
        task.push_execution(),
        ctx,
        availability=Availability(bwrap=str(fake), landlock_abi=None),
    )
    fake.unlink()
    monkeypatch.setenv("PATH", os.environ["PATH"].split(os.pathsep, 1)[1])
    with pytest.raises(NoConfinement, match="does not start"):
        prepared.wrap_argv(["python3"])


def test_wrap_argv_refuses_when_nothing_was_confined(ctx) -> None:
    """A `Prepared` carrying no confinement must not silently produce a runnable
    command line. Since the split `prepare` always sets one, so this constructs
    the state directly — the guard protects against a future caller building a
    `Prepared` by hand, which is exactly when it would matter."""
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)._replace(confinement=None)
    with pytest.raises(NoConfinement, match="no executor to start"):
        prepared.wrap_argv(["python3"])


def test_prepared_environment_carries_a_derived_path(ctx) -> None:
    """`PATH` is in the block, and every directory in it is granted.

    Raised by `validator`: a body handed an environment block with no `PATH`
    does not get an empty one — POSIX `sh` substitutes a built-in default, and
    that value is a property of the shell binary rather than of anything anyone
    configured. It is not an isolation hole, because the allow-list decides what
    may execute; it is a **determinism** hole, and this closes it.
    """
    from env_mgr.fs.path import contained

    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)
    dirs = prepared.environment["PATH"].split(os.pathsep)
    assert dirs and all(dirs)
    granted = [g.path for g in prepared.policy.granted]
    for d in dirs:
        assert any(contained(d, root) for root in granted), f"{d} is on PATH and not granted"


def test_a_declared_env_may_override_the_derived_path(ctx) -> None:
    """An author saying so outranks a default. An override naming an ungranted
    directory is simply unreachable, and nothing here can make it otherwise —
    which is the honest behaviour, not a guard worth adding."""
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx, AgentSpec(env={"PATH": "/usr/bin"}))
    assert prepared.environment["PATH"] == "/usr/bin"


# ------------------------------------------- spawn: one verb, three mechanisms


def test_spawn_confines_a_child_from_an_unconfined_parent(ctx, tmp_path: Path) -> None:
    """The case `agent` actually has: a threaded runner that must stay writable.

    The parent applies nothing — it has a store to write afterwards — and the
    **child** is confined between fork and exec. Row 1 of the threading
    measurement is why this shape exists at all: a runner thread that confines
    itself can no longer do its own job, irreversibly.
    """
    import errno

    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)
    outside = tmp_path / "supervisor-only.txt"
    outside.write_text("mine")

    script = (
        "import sys\n"
        f"try:\n    open({str(outside)!r}, 'a')\n"
        "except OSError as e:\n    sys.exit(e.errno)\n"
        "sys.exit(0)\n"
    )
    proc = prepared.spawn([os.sys.executable, "-c", script])
    assert proc.wait(timeout=60) == errno.EACCES

    # And the positive control that makes the denial mean something: the same
    # child, writing inside its own zone, succeeds.
    inside = os.path.join(prepared.zone.root, "ok.txt")
    ok = prepared.spawn([os.sys.executable, "-c", f"open({inside!r}, 'w').write('x')"])
    assert ok.wait(timeout=60) == 0
    assert Path(inside).read_text() == "x"


def test_spawn_leaves_the_parent_writable(ctx, tmp_path: Path) -> None:
    """The property the whole shape exists for. If this ever fails, the runner
    has lost the ability to record what it just ran."""
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)
    prepared.spawn([os.sys.executable, "-c", "pass"]).wait(timeout=60)

    store_record = tmp_path / "store-record.json"
    store_record.write_text("{}")
    assert store_record.read_text() == "{}"


def test_spawn_refuses_when_there_is_no_mechanism(ctx, monkeypatch) -> None:
    """*No isolation, no start* now has two places to hold, and both do.

    Since the split the refusal comes **at prepare**, before the workspace is
    cut — which is the improvement the split bought. `spawn` keeps its own
    refusal for a `Prepared` that reaches it carrying none, because a second
    reader of that state is exactly where a silent no-sandbox would appear.
    """
    from env_mgr.isolation.probe import Availability

    nothing = Availability(bwrap=None, landlock_abi=None)
    task = Task()
    with pytest.raises(NoConfinement):
        prepare(task, task.push_execution(), ctx, availability=nothing)

    prepared = prepare(Task(), Task().push_execution(), ctx)._replace(confinement=None)
    monkeypatch.setattr("env_mgr.prepare.probe", lambda: nothing)
    with pytest.raises(NoConfinement):
        prepared.spawn([os.sys.executable, "-c", "pass"])


def test_spawn_uses_bwrap_when_that_is_the_mechanism(ctx, tmp_path: Path, monkeypatch) -> None:
    """One verb, and the caller branches on nothing. The fake `bwrap` here is a
    shim that execs its tail, so a successful run proves the argv was built and
    handed over rather than that the policy was applied — which is all that can
    be proved where `bwrap` is absent."""
    fake = tmp_path / "bin" / "bwrap"
    fake.parent.mkdir()
    fake.write_text('#!/bin/sh\nwhile [ "$1" != "--" ]; do shift; done\nshift\nexec "$@"\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake.parent) + os.pathsep + os.environ["PATH"])

    task = Task()
    prepared = prepare(
        task,
        task.push_execution(),
        ctx,
        availability=Availability(bwrap=str(fake), landlock_abi=None),
    )
    assert prepared.confinement.mechanism == "bwrap"
    proc = prepared.spawn([os.sys.executable, "-c", "raise SystemExit(7)"])
    assert proc.wait(timeout=60) == 7


def test_an_ungranted_interpreter_blames_the_policy_not_python(tmp_path: Path, main_repo) -> None:
    """M3, caught at the surface that produces it.

    A `Context` with no interpreter grant makes every spawn fail — and
    `subprocess` reports it in the **parent** as
    ``PermissionError: /home/…/bin/python3``, naming the interpreter. That is
    this module's characteristic failure, *the symptom names the wrong cause*,
    arriving in our own output. It cost an afternoon when first measured and it
    reappeared here unprompted the first time `spawn` ran a real child.
    """
    reg = DomainRegistry()
    for name, kind in (
        ("store", DomainKind.HANDOFF_STORAGE),
        ("ws", DomainKind.WORKSPACE),
        ("play", DomainKind.PLAYGROUND),
    ):
        reg.register(name, str(tmp_path / "root"), kind)
    bare = context(
        domains=reg, store_root=str(tmp_path / "store"), main_repo=main_repo
    )  # no interpreter_grants
    task = Task()
    prepared = prepare(task, task.push_execution(), bare)

    with pytest.raises(PermissionError, match="not the interpreter") as excinfo:
        prepared.spawn([os.sys.executable, "-c", "pass"])
    assert "interpreter_grants" in str(excinfo.value)
    assert os.sys.executable in str(excinfo.value)


def test_a_granted_executable_failing_is_not_blamed_on_the_sandbox(ctx) -> None:
    """The other direction, and it is the reason the check exists rather than a
    blanket rewrite: attributing *every* exec failure to the policy would be the
    same defect pointing the other way. A missing file inside the granted set
    still reports itself as a missing file."""
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)
    missing = os.path.join(prepared.zone.root, "no-such-binary")
    with pytest.raises(FileNotFoundError) as excinfo:
        prepared.spawn([missing])
    assert "not the interpreter" not in str(excinfo.value)


def test_prepared_reports_the_cli_it_was_provisioned_for(tmp_path: Path, main_repo) -> None:
    """`agent`'s O2, measured: the SDK's `_find_cli` returns its **bundled**
    binary before it ever calls `shutil.which("claude")`, so an agent runs one
    CLI while this package's recipe configured plugins into another — two
    versions on this machine, and the agent silently lacks the plugins.

    The fix is for the backend to pin `cli_path`, which needs one fact it cannot
    obtain: *which binary was provisioned*. Declared on `Context`, reported here.
    """
    reg = DomainRegistry()
    for name, kind in (
        ("store", DomainKind.HANDOFF_STORAGE),
        ("ws", DomainKind.WORKSPACE),
        ("play", DomainKind.PLAYGROUND),
    ):
        reg.register(name, str(tmp_path / "root"), kind)
    cli = tmp_path / "opt" / "bin" / "claude"
    cli.parent.mkdir(parents=True)
    cli.write_text("#!/bin/sh\n")
    cli.chmod(0o755)

    ctx = context(
        domains=reg,
        store_root=str(tmp_path / "store"),
        main_repo=main_repo,
        interpreter_grants=interpreter_grants(),
        agent_cli=str(cli),
        tier=Tier.PRODUCTION,
    )
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)

    assert prepared.agent_cli == str(cli.resolve())
    # And it is executable under the policy, or pinning it would trade a silent
    # wrong binary for a confident refusal at exec.
    assert any(contained_path(str(cli), g.path) for g in prepared.policy.granted), (
        f"{cli} is reported but not granted: {[g.path for g in prepared.policy.granted]}"
    )


def test_no_declared_cli_reports_none_and_grants_nothing(ctx) -> None:
    """Absence is a case. A run with no declared backend CLI is the ordinary
    program-task run, and it must not acquire a grant for a path nobody named."""
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)
    assert prepared.agent_cli is None


def test_a_declared_cli_that_does_not_resolve_refuses(tmp_path: Path, main_repo) -> None:
    """Principle 3, and the same rule as every other declared path here: cannot
    resolve, cannot decide, deny — at prepare rather than at exec, where it
    would arrive as the backend blaming its own binary."""
    reg = DomainRegistry()
    for name, kind in (
        ("store", DomainKind.HANDOFF_STORAGE),
        ("ws", DomainKind.WORKSPACE),
        ("play", DomainKind.PLAYGROUND),
    ):
        reg.register(name, str(tmp_path / "root"), kind)
    ctx = context(
        domains=reg,
        store_root=str(tmp_path / "store"),
        main_repo=main_repo,
        agent_cli=str(tmp_path / "no-such-cli"),
    )
    task = Task()
    with pytest.raises(ValueError, match="does not resolve"):
        prepare(task, task.push_execution(), ctx)


# ---------------------------------- the kill switch: AGENT_SYS_NO_PERMISSIONS


def test_the_switch_is_read_in_exactly_one_place() -> None:
    """**A switch with three readers is three switches.**

    Structural, not behavioural: nothing outside `prepare.permissions_enforced`
    may look the variable up. `grants` and `layout` take it as an argument, and
    `agent` and `demo` read `Prepared.permissions_enforced` rather than the
    environment. Asserted over the source because "one reader" is a claim about
    the code and the code is checkable.
    """
    import ast
    import pathlib

    from env_mgr.prepare import NO_PERMISSIONS_ENV_VAR

    root = pathlib.Path(__file__).resolve().parents[2] / "env_mgr"
    readers = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        # Docstrings mention the switch by name in three modules, which is
        # documentation rather than a second reader — so a plain substring
        # search answers the wrong question. A string constant used as a bare
        # expression statement is a docstring; anything else is code.
        docstrings = {
            id(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        }
        for node in ast.walk(tree):
            named = (
                isinstance(node, ast.Constant)
                and node.value == NO_PERMISSIONS_ENV_VAR
                and id(node) not in docstrings
            ) or (isinstance(node, ast.Name) and node.id == "NO_PERMISSIONS_ENV_VAR")
            if named:
                readers.append(path.relative_to(root).as_posix())
                break
    assert readers == ["prepare.py"], f"the switch is read in {readers}"


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "anything"])
def test_these_spellings_turn_it_off(value: str) -> None:
    from env_mgr.prepare import permissions_enforced

    assert permissions_enforced({"AGENT_SYS_NO_PERMISSIONS": value}) is False


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "  OFF  "])
def test_these_spellings_leave_it_on(value: str) -> None:
    """`AGENT_SYS_NO_PERMISSIONS=0` must not read as *set*.

    An operator who writes `0` means off, and treating every non-empty string as
    truthy would disable enforcement for exactly the person trying to say "no
    thanks" — a switch that fails open on a typo.

    **Since 2026-08-30 these spellings are the only way to enforcement at all**
    (§4.22f), so this stopped being a courtesy to a typo and became the opt-in
    path. It is also the non-vacuity control for `test_unset_leaves_it_off`
    below: without it, a `permissions_enforced` hard-wired to `False` would pass
    the default test.
    """
    from env_mgr.prepare import permissions_enforced

    assert permissions_enforced({"AGENT_SYS_NO_PERMISSIONS": value}) is True


def test_unset_leaves_it_off() -> None:
    """**The 2026-08-30 ruling, at the one place that decides it.**

    *关闭agent_sys的权限系统（干脆改成默认关闭吧）* — off, and off as the default
    rather than as an opt-in. `interfaces.md` §4.22f.

    Its non-vacuity control is `test_these_spellings_leave_it_on` immediately
    above: **`AGENT_SYS_NO_PERMISSIONS=0` still returns `True`**, so this test
    is measuring a default rather than a function that has stopped being able
    to say `True` at all.
    """
    from env_mgr.prepare import permissions_enforced

    assert permissions_enforced({}) is False


def test_by_default_a_run_that_would_be_refused_proceeds(ctx, monkeypatch) -> None:
    """**The behavioural half of the default flip, with the switch nowhere.**

    `test_with_the_switch_off_an_unresolvable_grant_does_not_stop_the_run` pins
    the same pair by *setting* the variable, which after the flip proves only
    that an explicit `1` still works. This one deletes it — the state a fresh
    shell is actually in — and shows the run proceeding.

    **And it is not vacuous**: the first half is the control, running under the
    directory fixture's explicit `AGENT_SYS_NO_PERMISSIONS=0`, and it must still
    raise. If enforcement had been removed rather than defaulted off, this test
    fails on its first assertion rather than passing quietly.
    """
    from .stubs import Grant, Permissions

    hid = HandoffId.new()
    task = Task(
        outputs=[hid], kinds={hid: "facts"}, permissions=Permissions((Grant(kind="facts"),))
    )

    # Control: enforcement is still reachable, and it still refuses.
    with pytest.raises(UnresolvedGrant) as caught:
        prepare(task, task.push_execution(), ctx)
    assert "facts" in str(caught.value)

    # And with the variable simply absent, which is the new default.
    monkeypatch.delenv("AGENT_SYS_NO_PERMISSIONS")
    prepared = prepare(task, task.push_execution(), ctx)
    assert prepared.permissions_enforced is False
    assert prepared.confinement is None


def test_with_the_switch_off_step_seven_is_not_attempted(ctx, monkeypatch) -> None:
    """**Not attempted, rather than attempted and discarded.**

    `select` raises `NoConfinement` when no mechanism exists, and `bwrap` is
    absent here — computing it only to throw it away would put a live exception
    on the path of a run that asked for no permission management at all. The
    probe is replaced with one that fails the test if it is called.
    """
    from env_mgr import prepare as prepare_mod

    monkeypatch.setenv("AGENT_SYS_NO_PERMISSIONS", "1")
    monkeypatch.setattr(
        prepare_mod, "probe", lambda: pytest.fail("step 7 was attempted with the switch off")
    )

    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)

    assert prepared.permissions_enforced is False
    assert prepared.confinement is None


def test_the_switch_is_stated_on_prepared_not_inferred(ctx, monkeypatch) -> None:
    """§4.17a. `confinement is None` would mean *unconfined* for two different
    reasons, and a reader cannot tell them apart. One reason too many."""
    task = Task()
    on = prepare(task, task.push_execution(), ctx)
    assert on.permissions_enforced is True

    monkeypatch.setenv("AGENT_SYS_NO_PERMISSIONS", "1")
    off = prepare(task, task.push_execution(), ctx)
    assert off.permissions_enforced is False


def test_with_the_switch_off_an_unresolvable_grant_does_not_stop_the_run(ctx, monkeypatch) -> None:
    """Step 2 best-effort. A run that asked for no permission management must
    not be stopped by a permission that failed to resolve.

    **The grant must name a kind the task DECLARES, and that is load-bearing.**
    Under the `_participates` skip ruling a grant for an *undeclared* kind
    resolves to `()` and never raises — grants are inherited wholesale from a
    root, so a wide model that raised on its own width could not be used with
    inheritance. Simplify the fixture to an inherited-looking kind and the
    second half passes because **nothing raised in the first place**, not
    because the switch swallowed a raise: the assertion survives and stops
    meaning anything. `env-mgr` caught exactly that in the first draft.

    The control half is therefore not decoration. It asserts both that a raise
    happens *and* which of the two conditions produced it, so a fixture that
    drifts into the vacuous shape fails here rather than passing quietly.
    """
    from .stubs import Grant, Permissions

    hid = HandoffId.new()
    task = Task(
        outputs=[hid], kinds={hid: "facts"}, permissions=Permissions((Grant(kind="facts"),))
    )

    with pytest.raises(UnresolvedGrant) as caught:
        prepare(task, task.push_execution(), ctx)
    # Which condition, not merely that something raised.
    assert "facts" in str(caught.value)

    monkeypatch.setenv("AGENT_SYS_NO_PERMISSIONS", "1")
    prepared = prepare(task, task.push_execution(), ctx)
    assert prepared.permissions_enforced is False


def test_with_the_switch_off_materialisation_is_unchanged(ctx, monkeypatch) -> None:
    """**The line the switch must not cross.**

    Staging, the workspace, `staged_package` and `environment` make a file
    appear where a task needs it. Turning any of them off would break the run
    rather than unrestrict it, so the switch leaves them exactly as they are.
    """
    monkeypatch.setenv("AGENT_SYS_NO_PERMISSIONS", "1")
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)

    assert prepared.zone is not None
    assert prepared.workspace is not None
    assert "PATH" in prepared.environment


def test_with_the_switch_off_spawn_runs_the_child_unconfined(ctx, monkeypatch) -> None:
    """The switch has to survive into `spawn`, or it is not a switch.

    `spawn` falls back to `select(probe())` when `confinement is None`, which
    raises on a machine with no mechanism — so without this the kill switch
    would fail **closed** in the one mode whose whole point is not to.
    """
    monkeypatch.setenv("AGENT_SYS_NO_PERMISSIONS", "1")
    task = Task()
    prepared = prepare(task, task.push_execution(), ctx)

    proc = prepared.spawn([sys.executable, "-c", "print('ran')"], stdout=subprocess.PIPE, text=True)
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 0
    assert out.strip() == "ran"
    # And `wrap_argv` is a no-op rather than a raise.
    assert prepared.wrap_argv(["/bin/true"]) == ["/bin/true"]


def test_the_switch_does_not_move_a_body_s_input(ctx, monkeypatch, tmp_path) -> None:
    """**The switch must not widen `stage`, and that reverses a ruled row.**

    Widening moves every staged input **down one level** — the artefact's files
    land at `<materials>/<hid>/v<N>/content/…` instead of at
    `<materials>/<hid>/v<N>/…`. `examples/demo/bin/render.py:67` reads the narrow
    shape, so the switch would break a body **by moving its input**, and it
    would present as a body reading one level short rather than as a switch.

    Applying the ruling's own line rather than overriding it: *materialisation
    is not permission management; if you find yourself disabling something that
    makes a file appear where a task needs it, you have crossed the line.* And
    widening buys nothing here — with nothing confined a body can read the store
    directly, so narrowing the copy denies it nothing it could not already
    reach. What is left is a path convention.
    """
    from env_mgr.fs import layout

    from .stubs import Handoff

    hid = HandoffId.new()
    content = Path(ctx.store_root, str(hid), "v0", "content")
    content.mkdir(parents=True)
    (content / "README.md").write_text("# the artefact")
    Path(ctx.store_root, str(hid), "v0", "claim").mkdir()

    monkeypatch.setenv("AGENT_SYS_NO_PERMISSIONS", "1")
    task = Task(inputs=[hid], kinds={hid: "summary"})
    execution = task.push_execution()
    execution.input_versions = {hid: 0}
    ctx.handoffs[hid] = Handoff(id=hid, type="summary")
    prepared = prepare(task, execution, ctx)

    staged = Path(prepared.environment["AGENT_SYS_INPUT_SUMMARY"])
    # The artefact's own files, directly — not a `content/` hop, and no `claim/`.
    assert (staged / "README.md").read_text() == "# the artefact"
    assert not (staged / "content").exists()
    assert not (staged / "claim").exists()
    # And the same shape with the switch off, which is the whole point.
    monkeypatch.delenv("AGENT_SYS_NO_PERMISSIONS")
    again = layout.stage([hid], {hid: 0}, str(tmp_path / "control"), ctx.store_root)
    assert sorted(os.listdir(again[hid])) == sorted(os.listdir(staged))


def test_a_declared_env_wins_over_every_other_contributor(ctx, tmp_path: Path) -> None:
    """**`spec_loader`'s schema now asserts this ordering, in a file we do not read.**

    `a816f39` tells every package author that a declared `env` is applied last
    and wins. That is true of five contributors to `Prepared.environment` and
    only one of them was pinned: `test_a_declared_env_may_override_the_derived_path`
    checks `PATH`, set at the top of the block, while `PACKAGE_ENV_VAR`,
    `output_env` and `input_env` sit **between** it and `material.deploy`.

    So the old pin passes if `deploy` moves above any of those three, while a
    declared `env` silently loses to a grant-derived variable — **one test, five
    contributors, and the one it checks is the furthest away.** Reported by
    `agent`, who looked before passing `spec_loader`'s question on.

    This pins the claim the schema actually makes. It fails on any reordering
    that puts `deploy` earlier, not only the one that reaches `PATH`.
    """
    from .stubs import Handoff

    package = tmp_path / "pkg"
    (package / "bin").mkdir(parents=True)
    ctx = ctx._replace(package=str(package))

    given, made = HandoffId.new(), HandoffId.new()
    for hid in (given, made):
        Path(ctx.store_root, str(hid), "v0", "content").mkdir(parents=True)
        ctx.handoffs[hid] = Handoff(id=hid, type="summary" if hid is given else "facts")
    Path(ctx.store_root, str(given), "v0", "content", "x.txt").write_text("x")

    task = Task(inputs=[given], outputs=[made], kinds={given: "summary", made: "facts"})
    execution = task.push_execution()
    execution.input_versions = {given: 0}
    execution.output_versions = {made: 0}

    # Establish that all five keys are set at all, or the override below would
    # pass by overriding nothing — the vacuous shape this suite keeps finding.
    plain = prepare(task, execution, ctx).environment
    contributors = ("PATH", PACKAGE_ENV_VAR, "AGENT_SYS_OUTPUT_FACTS", "AGENT_SYS_INPUT_SUMMARY")
    for key in contributors:
        assert key in plain, f"{key} is not set at all, so overriding it proves nothing"

    declared = {key: f"/declared/{key.lower()}" for key in contributors}
    overridden = prepare(task, execution, ctx, AgentSpec(env=declared)).environment
    for key, value in declared.items():
        assert overridden[key] == value, f"a declared env lost to whatever sets {key}"


def test_output_paths_addresses_slots_where_the_env_var_addresses_names(ctx) -> None:
    """**The hole `output_env` names and declines to close, closed for one reader.**

    A kind naming two output slots is exported for **neither** variable, because
    an author who wrote `outputs: ['facts', 'facts']` cannot address either one
    *by name*. A `HandoffId` has no such collision — it addresses slots.

    So the **agent** can be told about both, which is the case the ruling's
    constraint is about: an agent told about two of three outputs writes two and
    finishes successfully. A shell **body** still has only the variable, so this
    closes the hole for the reader that can use an id and leaves it open for the
    one that cannot. That is the honest state, not a full fix.
    """
    from .stubs import Handoff

    a, b = HandoffId.new(), HandoffId.new()
    for hid in (a, b):
        Path(ctx.store_root, str(hid), "v0", "content").mkdir(parents=True)
        ctx.handoffs[hid] = Handoff(id=hid, type="facts")
    task = Task(outputs=[a, b], kinds={a: "facts", b: "facts"})
    execution = task.push_execution()
    execution.output_versions = {a: 0, b: 0}

    prepared = prepare(task, execution, ctx)

    # The variable: neither, deliberately.
    assert [k for k in prepared.environment if k.startswith("AGENT_SYS_OUTPUT")] == []
    # The mapping: both, addressed by slot.
    assert set(prepared.output_paths) == {a, b}
    assert prepared.output_paths[a] != prepared.output_paths[b]


def test_an_output_with_no_pinned_version_is_absent_not_empty(ctx) -> None:
    """`agent` enumerates `task.outputs` itself and renders the difference as
    *"no resolved path"*. Absence therefore has to be unambiguous — a slot we
    could not resolve must not arrive looking like one we resolved to nothing.
    """
    from .stubs import Handoff

    hid = HandoffId.new()
    ctx.handoffs[hid] = Handoff(id=hid, type="facts")
    task = Task(outputs=[hid], kinds={hid: "facts"})
    prepared = prepare(task, task.push_execution(), ctx)  # no output_versions

    assert hid not in prepared.output_paths
    assert prepared.output_paths == {}


def test_a_declared_skill_reaches_the_zone(ctx, tmp_path: Path) -> None:
    """**The mechanism was wired, working, and no package had ever used it.**

    `env_mgr` redirects `CLAUDE_CONFIG_DIR` into the zone (`material.py`), and a
    Claude Code session reads personal skills from `$CLAUDE_CONFIG_DIR/skills/`
    — measured with a planted skill whose name cannot come from the model's
    prior, `scratch/single-real-task-2026-08/r0_probe_skill_in_config_dir.sh`.
    So a skill arrives in a zone only if the agent spec declares it.

    None did. Measured 2026-08-31, mid-run: the agent called
    `Skill{"experiment-result-packup"}` — which its own brief calls *"the
    authority"* for the packup layout, and which the mission names as a
    requirement — got `Unknown skill`, and started hunting the filesystem for
    the directory by hand. Its zone's `config/` held the CLI's runtime
    directories and no `skills/` at all.
    """
    skill = tmp_path / "experiment-result-packup"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: experiment-result-packup\n---\nlayout\n")

    task = Task()
    prepared = prepare(task, task.push_execution(), ctx, AgentSpec(skills=[str(skill)]))

    landed = Path(prepared.environment["CLAUDE_CONFIG_DIR"]) / "skills"
    assert (landed / "experiment-result-packup" / "SKILL.md").is_file()


def test_a_declared_material_that_is_absent_is_refused_not_skipped(ctx, tmp_path: Path) -> None:
    """**The non-vacuity control, and it is the same bug wearing a fix.**

    `material.deploy` was `if os.path.exists(src): copy_out(...)` with no else.
    A declared skill whose path is wrong was skipped in silence — no error, no
    warning — and the agent met the absence hours later, from inside its own
    session, as a failure of its own with nothing anywhere naming the cause.

    That is exactly the failure the test above records, one layer up: with the
    declaration added and this guard missing, a typo in the path produces a run
    that looks fixed and is not. `fail closed` is this package's own rule.

    No shipped package declared any material when this landed, so nothing
    existing changed behaviour.
    """
    task = Task()
    missing = str(tmp_path / "not-there")
    with pytest.raises(PrepareRefused) as caught:
        prepare(task, task.push_execution(), ctx, AgentSpec(skills=[missing]))

    text = str(caught.value)
    assert missing in text, "the message must name the path that was wrong"
    assert "skills" in text, "and which material key it came from"
