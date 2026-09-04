# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Criteria 2, 13, 17 — the nested layout, the validation sibling, and resume."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from env_mgr.fs import layout
from env_mgr.fs.domain import DomainKind, DomainRegistry
from env_mgr.fs.path import contained
from env_mgr.isolation.policy import Granted, Mode

from .conftest import attempt, base_policy, run_confined
from .stubs import Task


@pytest.fixture
def domains(tmp_path: Path) -> DomainRegistry:
    reg = DomainRegistry()
    reg.register("store", str(tmp_path / "root"), DomainKind.HANDOFF_STORAGE)
    reg.register("ws", str(tmp_path / "root"), DomainKind.WORKSPACE)
    reg.register("play", str(tmp_path / "root"), DomainKind.PLAYGROUND)
    return reg


# ----------------------------------------------------------- criterion 2


def test_subtask_nested_under_parent(domains: DomainRegistry) -> None:
    parent = Task()
    child = Task(parent=parent.id)
    parent_zone = layout.create(parent, parent.push_execution(), domains)
    child_zone = layout.create(child, child.push_execution(), domains)

    assert contained(child_zone.root, parent_zone.root)
    assert os.path.dirname(child_zone.root) == parent_zone.root


def test_reach_is_containment(domains: DomainRegistry) -> None:
    """A task's reach is decided by canonical containment against its own
    subtree — one function serving both questions, which is what the nesting is
    for."""
    parent = Task()
    child = Task(parent=parent.id)
    other = Task()
    parent_zone = layout.create(parent, parent.push_execution(), domains)
    child_zone = layout.create(child, child.push_execution(), domains)
    other_zone = layout.create(other, other.push_execution(), domains)

    assert parent_zone.contains(child_zone.root) is True
    assert parent_zone.contains(other_zone.root) is False
    assert child_zone.contains(parent_zone.root) is False


def test_the_zone_has_the_four_directories(domains: DomainRegistry) -> None:
    task = Task()
    zone = layout.create(task, task.push_execution(), domains)
    for sub in ("handoffs", "workspace", "playground", layout.LOGS):
        assert os.path.isdir(os.path.join(zone.root, sub)), sub


def test_a_retry_gets_its_own_zone(domains: DomainRegistry) -> None:
    """A zone belongs to an **attempt**, not a task. Grants resolve to
    ``<root>/<hid>/v<N>/`` and ``N`` lives on the `Execution`."""
    task = Task()
    first = layout.create(task, task.push_execution(), domains)
    second = layout.create(task, task.push_execution(), domains)
    assert first.root != second.root
    assert (first.attempt, second.attempt) == (0, 1)


# ---------------------------------------------------------- criterion 13


def test_validation_is_a_sibling_not_a_descendant(domains: DomainRegistry) -> None:
    """Design D5. Criterion 13 says containment resolves the producer/validator
    separation, and it is untrue without this: the layout is five things and
    none of them is a validation, and the only place it has room is inside the
    producing subtree, which is reachable."""
    task = Task()
    zone = layout.create(task, task.push_execution(), domains)
    validation = layout.validation_zone(task, "output", domains)

    assert contained(validation, zone.root) is False
    assert os.path.dirname(validation) == os.path.dirname(zone.root)


def test_producer_cannot_read_validation(domains: DomainRegistry) -> None:
    """And the sibling is denied for free under the allow-list: no rule of its
    own, and nothing in the granted set names it."""
    task = Task()
    zone = layout.create(task, task.push_execution(), domains)
    validation = layout.validation_zone(task, "output", domains)
    standard = os.path.join(validation, "checking-standard.md")
    Path(standard).write_text("the standard the producer must not see")
    own = os.path.join(zone.root, layout.LOGS, "own.txt")
    Path(own).write_text("mine")

    policy = base_policy(Granted(zone.root, Mode.READ_WRITE))
    denied, control = run_confined(policy, lambda: (attempt(standard, "r"), attempt(own, "r")))
    assert control == 0, "the positive control failed"
    assert denied == errno.EACCES


def test_the_placement_is_what_denies_it(domains: DomainRegistry) -> None:
    """The **negative control** for criterion 13, and it exists because of an
    observation `validator` made about their own suite.

    Their criterion 10 and 21 tests assert what a validation can *observe*, not
    where it *sits* — and were silently benefiting from a placement nothing
    asserted. The same question applies here: does the test above pass because
    the sibling placement works, or would it pass under any placement?

    So this puts the same file inside the producing subtree and asserts the
    producer **can** read it. That is the state the layout would be in if
    anybody moved a validation under the task it validates, and criterion 13
    would be false with every test above still green. A property held by an
    accident of location is not a property; this is what makes it one.
    """
    task = Task()
    zone = layout.create(task, task.push_execution(), domains)
    as_descendant = os.path.join(zone.root, "validation", "checking-standard.md")
    os.makedirs(os.path.dirname(as_descendant))
    Path(as_descendant).write_text("the standard, misplaced")

    policy = base_policy(Granted(zone.root, Mode.READ_WRITE))
    assert run_confined(policy, lambda: attempt(as_descendant, "r")) == 0, (
        "a descendant of the producing zone was NOT readable, so the test above "
        "proves nothing about placement — the granted set must have changed"
    )
    # And the two placements differ in exactly the way the criterion turns on.
    assert zone.contains(as_descendant) is True
    assert zone.contains(layout.validation_zone(task, "output", domains)) is False


# ---------------------------------------------------------- criterion 17


def test_playground_survives_resume(domains: DomainRegistry) -> None:
    """**Half of the criterion only.**

    The survival half is observable and is tested here. *"Nothing depends on its
    contents having survived"* is a property of all future code, not an
    observable of a run — it is a review rule, and the design says so rather
    than approximating it with a test that would pass vacuously.
    """
    task = Task()
    execution = task.push_execution()
    zone = layout.create(task, execution, domains)
    note = Path(zone.root) / "playground" / "half-finished.txt"
    note.write_text("work in progress")

    # A fresh process, re-declaring the same domains, resuming the same attempt.
    reloaded = DomainRegistry()
    for domain in domains:
        reloaded.register(domain.name, domain.root, domain.kind)
    again = layout.create(task, execution, reloaded)
    assert again.root == zone.root
    assert note.read_text() == "work in progress"


def test_find_zone_dir_picks_the_latest_attempt(domains: DomainRegistry) -> None:
    task = Task()
    layout.create(task, task.push_execution(), domains)
    latest = layout.create(task, task.push_execution(), domains)
    assert layout.find_zone_dir(domains.storage_root(), task.id) == latest.root


# ------------------------------------------------- the label on a directory
#
# A run tree is something a person reads. Every directory `agent_sys` creates
# now leads with what it *is* and carries the name the system already knew, so
# `ls` answers "which task is this" without a lookup. The tests below hold the
# two properties that make the label safe: nothing resolves through it, and a
# tree written before it existed still resolves.


def test_a_zone_is_named_after_its_closure(domains: DomainRegistry) -> None:
    task = Task(closure="describe")
    zone = layout.create(task, task.push_execution(), domains)
    assert os.path.basename(zone.root).startswith(f"task.describe.{task.id}.")


def test_a_task_with_no_closure_keeps_the_unlabelled_name(domains: DomainRegistry) -> None:
    """The label is optional and its absence is not a placeholder: a task that
    has no closure gets exactly the name it got before labels existed."""
    task = Task()
    zone = layout.create(task, task.push_execution(), domains)
    assert os.path.basename(zone.root).startswith(f"task.{task.id}.")


def test_a_closure_name_cannot_add_a_field(domains: DomainRegistry) -> None:
    """`find_zone_dir` reads the attempt from ``parts[-2]``, so a ``.`` in a
    closure name would make it read the wrong field. The slug is what stops it."""
    task = Task(closure="a.b/c d")
    execution = task.push_execution()
    zone = layout.create(task, execution, domains)
    name = os.path.basename(zone.root)
    assert name == f"task.a-b-c-d.{task.id}.0.{name.rsplit('.', 1)[-1]}"
    assert layout.find_zone_dir(domains.storage_root(), task.id) == zone.root


def test_find_zone_dir_still_finds_an_unlabelled_zone(domains: DomainRegistry) -> None:
    """The compatibility claim, asserted rather than argued: a zone directory in
    the shape written before labels existed is still this task's."""
    task = Task()
    legacy = os.path.join(domains.storage_root(), f"task.{task.id}.0.deadbeef")
    os.makedirs(legacy)
    assert layout.find_zone_dir(domains.storage_root(), task.id) == legacy


def test_a_validation_zone_is_named_after_its_closure(
    domains: DomainRegistry,
) -> None:
    task = Task(closure="describe")
    layout.create(task, task.push_execution(), domains)
    root = layout.validation_zone(task, "output_validation", domains)
    assert os.path.basename(root).startswith(f"validation.describe.{task.id}.output_validation.")


def test_a_staged_input_keeps_the_store_directory_s_name(
    domains: DomainRegistry, tmp_path: Path
) -> None:
    """The staged copy inherits the label instead of recomputing it, so a body's
    ``materials/`` reads like the store and no kind map has to be threaded
    through `stage`."""
    from task_graph.ids import HandoffId

    from .stubs import context

    store = tmp_path / "store"
    hid = HandoffId.new()
    published = store / f"handoff.facts.{hid}" / "v1" / "content"
    published.mkdir(parents=True)
    (published / "result.json").write_text("{}")

    task = Task(inputs=[hid])
    execution = task.push_execution()
    execution.input_versions = {hid: 1}
    zone = layout.create(task, execution, domains)

    ctx = context(domains=domains, store_root=str(store))
    staged = layout.stage_handoffs(task, execution, zone, ctx)

    assert Path(staged[hid]).parent.name == f"handoff.facts.{hid}"


def test_a_parent_without_a_zone_is_an_error(domains: DomainRegistry) -> None:
    orphan = Task(parent=Task().id)
    with pytest.raises(ValueError, match="has no zone under"):
        layout.create(orphan, orphan.push_execution(), domains)


# ------------------------------- the validation zone, as `validator` receives it


def test_prepare_validation_places_a_sibling_and_stages_the_materials(
    domains: DomainRegistry, tmp_path: Path
) -> None:
    """`interfaces.md` §5.8 dissolves here: a body was handed handoff **ids as
    strings** in a zone with nothing pointing at the store, so it could not read
    what it was validating. It could not, because the zone was allocated by a
    module that does not know the layout — so nothing staged anything into it.

    Placement and staging arrive together because they are one question.
    """
    from env_mgr.prepare import prepare_validation
    from task_graph.ids import HandoffId

    from .stubs import context

    store = tmp_path / "store"
    hid = HandoffId.new()
    # `content/`, because that is the only shape the store produces: `allocate`
    # creates it at dispatch and `put` writes into it. A fixture fabricating a
    # bare `v<N>/` was asserting against a layout that never exists, which is why
    # it kept passing while `stage` copied the version directory whole.
    body = store / str(hid) / "v1" / "content"
    body.mkdir(parents=True)
    (body / "result.json").write_text('{"ok": true}')

    task = Task(outputs=[hid])
    execution = task.push_execution()
    execution.output_versions = {hid: 1}
    zone = layout.create(task, execution, domains)
    ctx = context(domains=domains, store_root=str(store))

    prepared = prepare_validation(task, execution, "output_validation", ctx)

    assert contained(prepared.root, zone.root) is False, "a descendant is reachable"
    assert os.path.dirname(prepared.root) == os.path.dirname(zone.root)
    # Keyed by handoff id, so a body taking more than one input knows which
    # copy is which without parsing the path.
    assert list(prepared.materials) == [hid]
    assert Path(prepared.materials[hid], "result.json").read_text() == '{"ok": true}'
    # A copy: spec §6.3 rule 2, so a validation cannot edit what it validates.
    assert (body / "result.json").read_text() == '{"ok": true}'
    assert prepared.materials[hid] != str(body)


def test_prepare_validation_stages_inputs_for_an_input_phase(
    domains: DomainRegistry, tmp_path: Path
) -> None:
    """An input validation checks what the task was **given**; an output
    validation what it **produced**. The phase decides, and the versions come
    off the attempt so a retry validates that attempt's artefacts."""
    from env_mgr.prepare import prepare_validation
    from task_graph.ids import HandoffId

    from .stubs import context

    store = tmp_path / "store"
    given, made = HandoffId.new(), HandoffId.new()
    for hid, name in ((given, "given.txt"), (made, "made.txt")):
        # `content/`: `stage` copies the artefact, not the version directory,
        # so a fixture that fabricated a bare `v0/` would stage nothing.
        d = store / str(hid) / "v0" / "content"
        d.mkdir(parents=True)
        (d / name).write_text(name)

    task = Task(inputs=[given], outputs=[made])
    execution = task.push_execution()
    execution.input_versions = {given: 0}
    execution.output_versions = {made: 0}
    layout.create(task, execution, domains)
    ctx = context(domains=domains, store_root=str(store))

    on_input = prepare_validation(task, execution, "input_validation", ctx)
    on_output = prepare_validation(task, execution, "output_validation", ctx)

    assert list(on_input.materials) == [given]
    assert list(on_output.materials) == [made]
    assert str(given) in on_input.materials[given]
    assert str(made) in on_output.materials[made]
    # Two phases, two zones — the phase is in the directory name.
    assert on_input.root != on_output.root


def test_prepare_validation_reads_the_phase_structurally(
    domains: DomainRegistry, tmp_path: Path
) -> None:
    """`PhaseKind` is `validator`'s and the two packages do not import each
    other, so the phase is read for its value — the same structural read this
    module already uses for `task_graph.Access`."""
    import enum

    from env_mgr.prepare import prepare_validation

    from .stubs import context

    class PhaseKind(str, enum.Enum):
        INPUT = "input_validation"
        OUTPUT = "output_validation"

    task = Task()
    execution = task.push_execution()
    layout.create(task, execution, domains)
    ctx = context(domains=domains, store_root=str(tmp_path / "store"))

    assert prepare_validation(task, execution, PhaseKind.OUTPUT, ctx).phase == "output_validation"
    assert prepare_validation(task, execution, "output_validation", ctx).phase == (
        "output_validation"
    )


# ---------------------------- F-D10: a non-leaf's zone, and why it must exist


def test_a_subtask_cannot_be_placed_before_its_parent(domains: DomainRegistry) -> None:
    """The failure `demo` hit in a live run, reproduced.

    A non-leaf never executes — the scheduler runs its main phase by unfolding —
    so it reaches no path that calls `prepare` and never gets a zone. Its
    children nest inside it, so they cannot be placed at all. **No nested graph
    can run.** Both sides were right alone, which is why neither suite saw it:
    `tests/env_mgr` builds parent zones itself and `tests/agent` has no
    `env_mgr`.

    The raise stays. Naming the parent is what let a live failure be diagnosed
    from one stored record instead of by bisection.
    """
    parent = Task()
    child = Task(parent=parent.id)
    with pytest.raises(ValueError, match="has no zone under"):
        layout.create(child, child.push_execution(), domains)


def test_place_zone_closes_it(domains: DomainRegistry, tmp_path: Path) -> None:
    """And the verb that closes it, from the caller's side.

    Whichever caller wins — the scheduler at `unfold`, or the attempt before it
    releases its thread — the verb is the same, which is why building it does
    not pre-empt that decision.
    """
    from env_mgr.prepare import place_zone

    from .stubs import context

    ctx = context(domains=domains, store_root=str(tmp_path / "store"))
    parent = Task()
    child = Task(parent=parent.id)

    parent_zone = place_zone(parent, parent.push_execution(), ctx)
    child_zone = layout.create(child, child.push_execution(), domains)

    assert contained(child_zone.root, parent_zone.root)
    assert os.path.dirname(child_zone.root) == parent_zone.root


def test_place_zone_confines_nothing_and_cuts_nothing(
    domains: DomainRegistry, tmp_path: Path
) -> None:
    """It is `prepare`'s first step and none of the rest.

    A non-leaf must not pay for a `git clone --shared`, must not have handoffs
    staged for work it will never do, and above all must not be confined — its
    thread is handed back and re-entered for output validation, and Landlock is
    irreversible.

    `main_repo` is deliberately a path that does not exist: if this ever starts
    cutting a workspace, it fails here rather than somewhere subtler.
    """
    from env_mgr.prepare import place_zone

    from .stubs import context

    ctx = context(
        domains=domains,
        store_root=str(tmp_path / "store"),
        main_repo=str(tmp_path / "no-such-repo"),
    )
    task = Task()
    zone = place_zone(task, task.push_execution(), ctx)

    assert os.path.isdir(zone.root)
    assert list(Path(zone.root, "workspace").iterdir()) == []
    assert list(Path(zone.root, "handoffs").iterdir()) == []
    # Still unconfined: the test process can write outside the zone afterwards.
    probe = tmp_path / "supervisor-still-works.txt"
    probe.write_text("yes")
    assert probe.read_text() == "yes"


@pytest.mark.xfail(
    strict=True,
    reason="criterion 13 has a second route. Staging MOVED it rather than closing "
    "it (interfaces.md §4.16): the precondition is TODO.md 4a, the package layout, "
    "and it is the user's. Strict, so closing it fails here and is noticed.",
)
def test_a_staged_package_still_carries_the_validators(
    domains: DomainRegistry, tmp_path: Path
) -> None:
    """**Criterion 13's honest number under staging, re-derived rather than assumed.**

    F19 reversed to *stage, not grant*, and the tempting conclusion is that the
    package route closed with it. **It did not.** `test_producer_cannot_read_validation`
    tests the *zone* route and the sibling placement really does close that one;
    this is the other route, and staging relocates it from
    ``<package>/validators/`` to ``<zone>/package/validators/``. The zone is
    granted read-**write**, so the copy is not merely readable — it is more
    reachable than the grant it replaced.

    That is `interfaces.md` §4.16's own statement, stated and accepted rather
    than discovered: *"until [TODO.md 4a] holds, staging moves the leak rather
    than closing it."* `stage_package`'s `include` allow-list is the mechanism
    that will close it, and it cannot be populated until a task's executable set
    is nameable without ``validators/`` in it — a package-layout guarantee,
    which is the user's and not this module's.

    Kept as a strict `xfail` for the same reason as before: when 4a lands and
    `Context.package_stage` names the set, this **XPASSes and fails the suite**,
    so the closing is noticed instead of quietly making a permanent xfail
    obsolete. The name changed with the mechanism — *granted* became *staged* —
    because a test whose name describes a mechanism nobody uses reads as history.
    """
    import errno

    package = tmp_path / "package"
    (package / "validators").mkdir(parents=True)
    standard = package / "validators" / "checking-standard.md"
    standard.write_text("the standard the producer must not see")
    (package / "bin").mkdir()
    (package / "bin" / "collect.py").write_text("# the body's launcher\n")

    task = Task()
    zone = layout.create(task, task.push_execution(), domains)
    staged = layout.stage_package(str(package), zone)
    assert staged is not None
    policy = base_policy(Granted(zone.root, Mode.READ_WRITE))

    copied = os.path.join(staged, "validators", "checking-standard.md")
    assert run_confined(policy, lambda: attempt(copied, "r")) == errno.EACCES


def test_a_staged_package_is_reachable_and_the_original_is_not(
    domains: DomainRegistry, tmp_path: Path
) -> None:
    """The property staging buys, and the cost it accepts, in one test.

    **Buys:** the package root is no longer granted, so criterion 12's *a read
    outside the granted set is denied* now covers every task package on the
    machine rather than exempting this one. The body still runs, because it
    launches out of the copy.

    **Costs, and §4.16 accepts it explicitly:** the copy lands in the zone,
    which the agent can write, so *a task may not modify the package it was
    loaded from* stops being kernel-enforced. Under a scope where permission
    management exists only to stop agents cross-contaminating, an agent editing
    **its own** body is not ours to prevent — so this asserts the write
    succeeds rather than pretending otherwise.
    """
    import errno

    package = tmp_path / "package"
    (package / "bin").mkdir(parents=True)
    launcher = package / "bin" / "collect.py"
    launcher.write_text("# the body's launcher\n")

    task = Task()
    zone = layout.create(task, task.push_execution(), domains)
    staged = layout.stage_package(str(package), zone)
    policy = base_policy(Granted(zone.root, Mode.READ_WRITE))

    copy = os.path.join(str(staged), "bin", "collect.py")
    assert run_confined(policy, lambda: attempt(copy, "r")) == 0
    assert run_confined(policy, lambda: attempt(str(launcher), "r")) == errno.EACCES
    # The accepted cost, asserted rather than left implicit.
    assert run_confined(policy, lambda: attempt(copy, "w")) == 0


def test_stage_package_takes_an_allow_list_not_a_deny_list(tmp_path: Path, domains) -> None:
    """**The list shape is the point, not an ergonomic choice.**

    Criterion 14 holds because a sibling zone nobody anticipated is *absent from
    a list*. A deny-list over the package would give exactly the
    anticipated-only guarantee §4.5 rejects — a validator directory added next
    month would not be on it and would stage. So when `TODO.md` 4a makes a
    task's executable set nameable, criterion 13 closes here by the same
    construction that already closes 14.
    """
    package = tmp_path / "package"
    (package / "bin").mkdir(parents=True)
    (package / "bin" / "collect.py").write_text("x")
    (package / "validators").mkdir()
    (package / "validators" / "standard.md").write_text("secret")

    task = Task()
    zone = layout.create(task, task.push_execution(), domains)
    staged = layout.stage_package(str(package), zone, ("bin",))

    assert os.path.exists(os.path.join(str(staged), "bin", "collect.py"))
    assert not os.path.exists(os.path.join(str(staged), "validators"))


def test_a_declared_staged_path_that_is_absent_is_refused(tmp_path: Path, domains) -> None:
    """Principle 3. A named entry that is missing is an error, not an empty
    stage — the alternative is a body that launches nothing and blames itself."""
    package = tmp_path / "package"
    package.mkdir()
    task = Task()
    zone = layout.create(task, task.push_execution(), domains)
    with pytest.raises(ValueError, match="declares 'bin'"):
        layout.stage_package(str(package), zone, ("bin",))


def test_no_package_stages_nothing(tmp_path: Path, domains) -> None:
    """Absence is a case: a run with no package configured stages nothing and
    exports nothing, and every existing caller keeps working."""
    task = Task()
    zone = layout.create(task, task.push_execution(), domains)
    assert layout.stage_package(None, zone) is None


def test_a_package_that_does_not_resolve_is_refused(tmp_path: Path, domains) -> None:
    """Cannot resolve, cannot decide, deny — the same rule as every other path
    here, carried over from the grant this replaced."""
    task = Task()
    zone = layout.create(task, task.push_execution(), domains)
    with pytest.raises(ValueError, match="does not resolve"):
        layout.stage_package(str(tmp_path / "no-such-package"), zone)
