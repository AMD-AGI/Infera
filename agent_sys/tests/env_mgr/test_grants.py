# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Resolving a `Grant` — design §6, and deviation D4.

The `Grant` / `Access` / `Permissions` shapes are `task_graph` rev.-12 material
being written in parallel; `stubs.py` carries them exactly as its design §3.5
declares them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from env_mgr.fs.domain import DomainKind, DomainRegistry
from env_mgr.fs.layout import handoff_version_dir
from env_mgr.grants import input_env, mode_for, output_env, resolve, resolve_all
from env_mgr.protocols import Mode, UnresolvedGrant
from task_graph.ids import HandoffId

from .stubs import Access, Execution, Grant, Handoff, Permissions, Task, context


@pytest.fixture
def store(tmp_path: Path) -> str:
    root = tmp_path / "store"
    root.mkdir()
    return str(root)


def _task_with(kind: str, *, version: int = 2) -> tuple[Task, Execution, dict]:
    hid = HandoffId.new()
    task = Task(inputs=[hid], permissions=Permissions((Grant(kind=kind),)))
    execution = Execution(attempt=1, input_versions={hid: version})
    return task, execution, {hid: Handoff(id=hid, type=kind)}


def test_a_kind_resolves_to_the_version_this_attempt_has(store: str) -> None:
    """No manifest read and no store access: the mapping is already on the
    runtime object, because ``Handoff.type`` carries the kind name."""
    task, execution, handoffs = _task_with("trace", version=2)
    (granted,) = resolve(task.permissions.grants[0], task, execution, handoffs, store)
    # `content/`, **not** `v2/`. Under §4.14 the manifest is the seal, so a
    # version directory granted whole lets its producer publish its own unsealed
    # version. A read grant gets one path because a consumer has nothing to claim.
    assert granted.path == os.path.join(handoff_version_dir(store, task.inputs[0], 2), "content")
    assert granted.mode is Mode.READ_EXEC


def test_a_retry_resolves_to_a_different_version(store: str) -> None:
    """``N`` lives on the `Execution`, so a retry has a different granted set.
    That is why `prepare` takes an attempt and why the zone rebuilds."""
    task, first, handoffs = _task_with("trace", version=0)
    second = Execution(attempt=1, input_versions={task.inputs[0]: 1})
    grant = task.permissions.grants[0]
    (a,) = resolve(grant, task, first, handoffs, store)
    (b,) = resolve(grant, task, second, handoffs, store)
    assert a.path.endswith("/v0/content")
    assert b.path.endswith("/v1/content")


def test_unresolved_grant_raises_rather_than_covering_nothing(store: str) -> None:
    """**Raised, not returned empty.**

    ``Handoff.type`` defaults to ``""`` and nothing yet requires it to be a
    registered kind name, so an unfilled type silently matched no grant and the
    agent received an empty granted set instead of an error — a graph that fails
    at the first read, in a way that looks like the agent's fault. Being loud
    costs one raise, and it makes whichever route `task_graph` takes for
    ``type`` loud when it is forgotten.
    """
    task, execution, handoffs = _task_with("trace")
    hid = task.inputs[0]
    handoffs[hid] = Handoff(id=hid, type="")  # the field nobody filled
    with pytest.raises(UnresolvedGrant, match="no slot has that kind") as excinfo:
        resolve(task.permissions.grants[0], task, execution, handoffs, store)
    # And it says which of the two conditions was unmet, because both used to
    # fall into one message and a caller who fixed the first got a
    # byte-identical error for the second.
    assert "Task.kinds" in str(excinfo.value)


def test_write_access_maps_to_read_write(store: str) -> None:
    """The seam between two vocabularies. `Access` is what the author declared;
    `Mode` is what the kernel gets, and `READ_EXEC` has no declaration-side
    meaning at all."""
    assert mode_for(Access.READ) is Mode.READ_EXEC
    assert mode_for(Access.WRITE) is Mode.READ_WRITE
    task, execution, handoffs = _task_with("trace")
    grant = Grant(access=Access.WRITE, kind="trace")
    content, claim = resolve(grant, task, execution, handoffs, store)
    assert content.mode is Mode.READ_WRITE
    assert claim.mode is Mode.READ_WRITE


# ------------------------------------------------------------- deviation D4


def test_the_two_unresolvable_conditions_say_which(store: str) -> None:
    """A kind that matches nothing, and a kind that matches with no version.

    The second is F-D12's shape: an **output** grant cannot resolve before the
    output is written, because a version is pinned for an input at dispatch and
    for an output only when the attempt closes. Reported by `demo`, who fixed a
    genuinely different bug and got the identical error message.
    """
    hid = HandoffId.new()
    task = Task(outputs=[hid], permissions=Permissions((Grant(kind="facts"),)))
    execution = Execution(attempt=0)  # output_versions is empty until close
    handoffs = {hid: Handoff(id=hid, type="facts")}

    with pytest.raises(UnresolvedGrant, match="none has a version on this attempt"):
        resolve(task.permissions.grants[0], task, execution, handoffs, store)

    with pytest.raises(UnresolvedGrant, match="no slot has that kind"):
        resolve(Grant(kind="absent"), task, execution, handoffs, store)


def test_a_non_canonical_grant_path_is_rejected(store: str) -> None:
    """Exact equality and realpath disagree on every form tried, always in the
    direction `closure` §6.3 forbids. Requiring canonical form makes the two
    interpreters agree by construction rather than by care."""
    task, execution, handoffs = _task_with("trace")
    for bad in ("/usr/", "/usr/.", "/usr/../etc", "relative/path", "/usr/*"):
        with pytest.raises(UnresolvedGrant, match="canonical"):
            resolve(Grant(path=bad), task, execution, handoffs, store)


def test_a_symlinked_grant_path_is_rejected(tmp_path: Path, store: str) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    task, execution, handoffs = _task_with("trace")
    with pytest.raises(UnresolvedGrant, match="realpath"):
        resolve(Grant(path=str(link)), task, execution, handoffs, store)
    assert resolve(Grant(path=str(real)), task, execution, handoffs, store)[0].path == str(real)


# ---------------------------------------------------------------- resolve_all


def test_resolve_all_flattens_every_grant(tmp_path: Path, store: str) -> None:
    a, b = HandoffId.new(), HandoffId.new()
    task = Task(
        inputs=[a, b],
        permissions=Permissions((Grant(kind="trace"), Grant(path="/usr", access=Access.READ))),
    )
    execution = Execution(attempt=0, input_versions={a: 0, b: 0})
    handoffs = {a: Handoff(id=a, type="trace"), b: Handoff(id=b, type="trace")}

    reg = DomainRegistry()
    reg.register("store", store, DomainKind.HANDOFF_STORAGE)
    ctx = context(domains=reg, store_root=store, handoffs=handoffs)

    granted = resolve_all(task, execution, ctx)
    assert len(granted) == 3
    assert {g.path for g in granted} == {
        os.path.join(handoff_version_dir(store, a, 0), "content"),
        os.path.join(handoff_version_dir(store, b, 0), "content"),
        "/usr",
    }


def test_no_permissions_grants_nothing_extra(store: str) -> None:
    reg = DomainRegistry()
    reg.register("store", store, DomainKind.HANDOFF_STORAGE)
    ctx = context(domains=reg, store_root=store)
    assert resolve_all(Task(), Execution(), ctx) == ()


# ------------------- inherited width: a no-op, not an error (interfaces §4.16)


def test_a_grant_for_a_kind_this_task_has_no_part_in_is_a_no_op(store: str) -> None:
    """`demo`'s measurement: grants are inherited wholesale from a root, so two
    of three subtasks carried a kind-named grant for a kind they have no slot
    for, and resolution raised on every one.

    Under `interfaces.md` §4.16 permission management is **wide** and its only
    job is stopping agents cross-contaminating. **A wide model that raises on
    its own width cannot be used with inheritance**, and a permission for
    something absent grants nothing, so it is not a contamination risk either.
    """
    facts = HandoffId.new()
    task = Task(
        outputs=[facts],
        kinds={facts: "facts"},  # this task participates in `facts` only
        permissions=Permissions((Grant(kind="facts"), Grant(kind="summary"))),
    )
    execution = Execution(attempt=0, output_versions={facts: 0})
    handoffs = {facts: Handoff(id=facts, type="facts")}

    assert resolve(Grant(kind="summary"), task, execution, handoffs, store) == ()
    assert len(resolve(Grant(kind="facts"), task, execution, handoffs, store)) == 1


def test_a_declared_kind_that_does_not_resolve_still_raises(store: str) -> None:
    """The half the no-op must not swallow.

    A kind the task **declares** and cannot resolve is the forgotten
    `Handoff.type` defect the raise was added for. `demo`'s case and this one
    differ by exactly one fact — whether `Task.kinds` names the kind — and that
    is the whole discriminator.
    """
    hid = HandoffId.new()
    task = Task(
        outputs=[hid],
        kinds={hid: "facts"},
        permissions=Permissions((Grant(kind="facts"),)),
    )
    execution = Execution(attempt=0)  # no version pinned
    handoffs = {hid: Handoff(id=hid, type="facts")}

    with pytest.raises(UnresolvedGrant, match="none has a version on this attempt"):
        resolve(task.permissions.grants[0], task, execution, handoffs, store)


def test_an_empty_kinds_cannot_discriminate_so_it_still_raises(store: str) -> None:
    """The conservative edge, stated because silence here would undo the raise.

    An unfilled `Task.kinds` **is** the forgotten-`type` bug, so it cannot be
    read as *"this task does not participate"* — that would restore the silent
    empty granted set the raise replaced.
    """
    hid = HandoffId.new()
    task = Task(outputs=[hid], permissions=Permissions((Grant(kind="facts"),)))
    execution = Execution(attempt=0, output_versions={hid: 0})

    with pytest.raises(UnresolvedGrant, match="no slot has that kind"):
        resolve(task.permissions.grants[0], task, execution, {hid: Handoff(id=hid)}, store)


# ------------------------------------ one output, two granted paths (the ruling)


def test_a_write_grant_resolves_to_content_and_a_claim_directory(store: str) -> None:
    """**The user's ruling, and it corrects a contradiction between two others.**

    `done_by_self_check`'s claim was ruled a *sibling* of `content/`, and the
    grant was separately ruled to narrow *to* `content/`. Under both the agent
    cannot write the thing it is asked to write. So an output resolves to two
    paths, and `manifest.yaml` and `validation.yaml` are reachable by neither.

    The narrowing is not tidiness: under §4.14 the **manifest is the seal**, so
    an agent granted `v<N>/` could publish its own unsealed version.
    """
    hid = HandoffId.new()
    task = Task(
        outputs=[hid],
        kinds={hid: "facts"},
        permissions=Permissions((Grant(access=Access.WRITE, kind="facts"),)),
    )
    execution = Execution(attempt=0, output_versions={hid: 0})
    handoffs = {hid: Handoff(id=hid, type="facts")}
    os.makedirs(os.path.join(store, str(hid), "v0", "content"), exist_ok=True)

    granted = resolve(task.permissions.grants[0], task, execution, handoffs, store)
    version = os.path.join(store, str(hid), "v0")
    assert [g.path for g in granted] == [
        os.path.join(version, "content"),
        os.path.join(version, "claim"),
    ]
    # Neither the seal nor the verdict is inside a granted path.
    assert version not in {g.path for g in granted}


def test_a_read_grant_gets_content_alone(store: str) -> None:
    """A consumer has nothing to claim, and an input's claim is the producer's.

    What this forecloses is named rather than assumed: a body can no longer read
    its input's `manifest.yaml`. Nobody has said whether one needs to; reported
    to `main` rather than decided here.
    """
    task, execution, handoffs = _task_with("trace", version=2)
    granted = resolve(task.permissions.grants[0], task, execution, handoffs, store)
    assert len(granted) == 1
    assert granted[0].path.endswith(os.path.join("v2", "content"))


def test_resolve_creates_nothing_and_the_allocator_owns_both_directories(store: str) -> None:
    """§4.18: **the allocator creates every directory it expects to be granted.**

    `handoff.allocate` makes `content/` and `claim/`; `handoff/store.py:334` is
    the second one. A granted path that does not exist is either a
    `FileNotFoundError` that kills every output dispatch — `landlock.py:198`
    opens every granted path and `Granted.optional` defaults `False` — or, if
    made optional, a rule dropped silently. **And the agent cannot create it
    either**: `mkdir` inside `v<N>/` needs write on `v<N>/`, which is what the
    narrowing removed.

    This module briefly created `claim/` itself, because the user left the
    *name* here and `handoff` could not act without one. Asserted rather than
    assumed now, because `allocate`'s `os.mkdir` is **not** `exist_ok` — a
    resolver that raced it would turn a dispatch into `FileExistsError`.
    """
    hid = HandoffId.new()
    task = Task(
        outputs=[hid],
        kinds={hid: "facts"},
        permissions=Permissions((Grant(access=Access.WRITE, kind="facts"),)),
    )
    execution = Execution(attempt=0, output_versions={hid: 0})
    handoffs = {hid: Handoff(id=hid, type="facts")}
    version = os.path.join(store, str(hid), "v0")
    os.makedirs(os.path.join(version, "content"), exist_ok=True)

    granted = resolve(task.permissions.grants[0], task, execution, handoffs, store)
    assert [g.path for g in granted] == [
        os.path.join(version, "content"),
        os.path.join(version, "claim"),
    ]
    assert not os.path.exists(os.path.join(version, "claim"))


def test_an_unallocated_version_is_still_only_resolved_never_created(store: str) -> None:
    """A hole must not be made to look allocated.

    `resolve` names paths; it does not decide a version exists. The distinction
    matters because §4.14 pre-allocates at dispatch and a *failed* attempt
    leaves a `v<N>/` that `latest` skips — so "the directory is there" and "the
    version is real" are already two different facts.
    """
    hid = HandoffId.new()
    task = Task(
        outputs=[hid],
        kinds={hid: "facts"},
        permissions=Permissions((Grant(access=Access.WRITE, kind="facts"),)),
    )
    execution = Execution(attempt=0, output_versions={hid: 0})
    handoffs = {hid: Handoff(id=hid, type="facts")}

    resolve(task.permissions.grants[0], task, execution, handoffs, store)
    assert not os.path.exists(os.path.join(store, str(hid), "v0"))


# ------------------------------- the output's declared name (demo's F-D17)


def test_an_output_is_exported_under_its_declared_kind(store: str) -> None:
    """**`demo`'s F-D17**, and it is `AGENT_SYS_TASK_PACKAGE`'s argument one slot
    over: a path known only at prepare time that the body cannot compute.

    Since §4.14 the directory exists at dispatch and is granted; until this it
    lived only in `prepared.policy.granted`, which no body ever sees, and a
    demo body died with `KeyError` that reached the run as `output_absent`.
    """
    hid = HandoffId.new()
    task = Task(outputs=[hid], kinds={hid: "facts"})
    execution = Execution(attempt=0, output_versions={hid: 0})

    assert output_env(task, execution, store) == {
        "AGENT_SYS_OUTPUT_FACTS": os.path.join(handoff_version_dir(store, hid, 0), "content"),
    }


def test_the_exported_path_is_one_the_policy_grants(store: str) -> None:
    """Exported and granted must agree **by construction**, not by care.

    A path we told the body to use and did not grant is the evaporating
    allow-list one level up: the body fails on our own instruction.
    """
    hid = HandoffId.new()
    task = Task(
        outputs=[hid],
        kinds={hid: "facts"},
        permissions=Permissions((Grant(access=Access.WRITE, kind="facts"),)),
    )
    execution = Execution(attempt=0, output_versions={hid: 0})
    handoffs = {hid: Handoff(id=hid, type="facts")}

    exported = set(output_env(task, execution, store).values())
    granted = {
        g.path for g in resolve(task.permissions.grants[0], task, execution, handoffs, store)
    }
    assert exported <= granted


def test_a_kind_naming_two_outputs_is_exported_for_neither(store: str) -> None:
    """**A hole, named rather than resolved.**

    Discriminating them by `HandoffId` would invent a naming scheme no author
    can write against: a closure declares `outputs: ['facts', 'facts']`, so the
    author cannot address one of them either. Choosing one silently is exactly
    the failure this whole export exists to remove — an output never written,
    reaching the run as `output_absent` with no cause.
    """
    a, b = HandoffId.new(), HandoffId.new()
    task = Task(outputs=[a, b], kinds={a: "facts", b: "facts"})
    execution = Execution(attempt=0, output_versions={a: 0, b: 0})

    assert output_env(task, execution, store) == {}


def test_two_kinds_that_collide_as_variable_names_are_both_dropped(store: str) -> None:
    """`my-facts` and `my_facts` are distinct kinds and one variable name.

    Detected by counting rather than by trusting `_env_name` to be injective —
    it is lossy on purpose and its inverse is never taken.
    """
    a, b = HandoffId.new(), HandoffId.new()
    task = Task(outputs=[a, b], kinds={a: "my-facts", b: "my_facts"})
    execution = Execution(attempt=0, output_versions={a: 0, b: 0})

    assert output_env(task, execution, store) == {}


def test_an_output_with_no_version_on_this_attempt_is_not_exported(store: str) -> None:
    """Naming a directory that was never allocated would send a body to write
    somewhere nothing granted and nothing will seal."""
    hid = HandoffId.new()
    task = Task(outputs=[hid], kinds={hid: "facts"})
    assert output_env(task, Execution(attempt=0), store) == {}


def test_inputs_are_not_exported(store: str) -> None:
    """A body writes its outputs; its inputs are staged into the zone by
    `layout.stage_handoffs`, which is a different route and a different copy."""
    hid = HandoffId.new()
    task = Task(inputs=[hid], kinds={hid: "facts"})
    execution = Execution(attempt=0, input_versions={hid: 0})
    assert output_env(task, execution, store) == {}


# ---------------------------- the input's declared name (demo's mirror report)


def test_a_staged_input_is_exported_under_its_declared_kind() -> None:
    """`demo` asked whether the asymmetry was deliberate. **It was not.**

    Outputs had a declared name and inputs did not, and the proof that it was an
    oversight was already in the code: `prepare` called `stage_handoffs`, which
    returns handoff id → staged path, and **discarded the mapping**. The only
    remaining way to find a staged input was to parse this module's directory
    layout, which `examples/demo/logic/store.py` had already become a reader of.
    """
    hid = HandoffId.new()
    task = Task(inputs=[hid], kinds={hid: "summary"})
    assert input_env(task, {hid: "/zone/handoffs/x/v0"}) == {
        "AGENT_SYS_INPUT_SUMMARY": "/zone/handoffs/x/v0",
    }


def test_the_exported_input_is_the_staged_copy_not_the_store() -> None:
    """Spec §6.3 rule 2: an agent works on a copy. The store path is a place a
    body must **not** read from, so it is not the value we hand it."""
    hid = HandoffId.new()
    task = Task(inputs=[hid], kinds={hid: "summary"})
    staged = "/zone/handoffs/" + str(hid) + "/v0"
    assert input_env(task, {hid: staged})["AGENT_SYS_INPUT_SUMMARY"] == staged


def test_two_inputs_of_one_kind_are_exported_for_neither() -> None:
    """The same rule as outputs, and the collision is **more** likely here:
    `validator` spec §4.1 makes many-to-many first class, so several inputs of
    one kind is ordinary rather than exotic. Still neither, because an author
    who declared two of a kind cannot address either one by name.
    """
    a, b = HandoffId.new(), HandoffId.new()
    task = Task(inputs=[a, b], kinds={a: "summary", b: "summary"})
    assert input_env(task, {a: "/zone/a", b: "/zone/b"}) == {}


def test_an_input_that_staged_nothing_is_not_exported() -> None:
    """A slot absent from the staged mapping staged nothing — a hole, or a
    version with no `content/`. Naming a path that was never copied would send a
    body to read an empty directory and call it the artefact."""
    hid = HandoffId.new()
    task = Task(inputs=[hid], kinds={hid: "summary"})
    assert input_env(task, {}) == {}


def test_the_two_declared_names_point_at_the_same_level(tmp_path, store: str) -> None:
    """**Both names hand a body the artefact's own files.** Measured, not argued.

    `demo` read the two path *shapes* — `<zone>/handoffs/<hid>/v<N>` against
    `<store>/<hid>/v<N>/content` — and reported them as one directory apart,
    which they were until `stage` narrowed. Since then `stage` copies
    ``<v>/content`` **to** ``<into>/<hid>/v<N>``, so the staged directory holds
    the content's own files and there is no `content/` hop on the input side.

    The strings still differ and the levels do not, which is worth a test
    precisely because the strings are what a reader compares. It pins the
    property — *what a body finds at the end of each name* — rather than either
    path's spelling, so it survives a layout move and fails a level move.
    """
    from env_mgr.fs import layout
    from env_mgr.fs.domain import DomainKind, DomainRegistry

    from .stubs import context

    hid_in, hid_out = HandoffId.new(), HandoffId.new()
    content = Path(store, str(hid_in), "v0", "content")
    content.mkdir(parents=True)
    (content / "README.md").write_text("# the artefact")
    # Beside it, and neither name may reach these.
    Path(store, str(hid_in), "v0", "manifest.yaml").write_text("digest: x")
    Path(store, str(hid_out), "v0", "content").mkdir(parents=True)

    reg = DomainRegistry()
    reg.register("zones", str(tmp_path / "zones"), DomainKind.HANDOFF_STORAGE)
    task = Task(inputs=[hid_in], outputs=[hid_out], kinds={hid_in: "summary", hid_out: "facts"})
    execution = Execution(attempt=0, input_versions={hid_in: 0}, output_versions={hid_out: 0})
    zone = layout.create(task, execution, reg)
    staged = layout.stage_handoffs(task, execution, zone, context(domains=reg, store_root=store))

    given = input_env(task, staged)["AGENT_SYS_INPUT_SUMMARY"]
    made = output_env(task, execution, store)["AGENT_SYS_OUTPUT_FACTS"]

    # The artefact's own files are directly at the end of the input name.
    assert (Path(given) / "README.md").read_text() == "# the artefact"
    assert not (Path(given) / "content").exists(), "an input needs no `content/` hop"
    # And the output name likewise names a directory the body writes files into.
    assert Path(made).is_dir() and Path(made).name == "content"
    # Neither reaches the manifest: `stage` copies `content/` and nothing else.
    assert not (Path(given).parent / "manifest.yaml").exists()
