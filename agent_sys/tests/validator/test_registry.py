"""Criteria 3, 12, 14 and 15 — the registry, and its two indexes.

The indexes are the interesting part. They are separate objects that **disagree**,
and every system surveyed keeps them separate for that reason. What each of them
must not do is the failure they were copied away from: dbt drops an id present in
`run_results` and gone from the manifest by a silent set intersection, and Great
Expectations reports `success=True` for an empty suite because its denominator
counts results produced.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from spec_loader.protocols import SpecInconsistent, SpecNotFound
from tests.validator.conftest import validator_record, write_body
from validator.protocols import Dimension, ValidatorInvalid
from validator.registry import EDGE_KINDS, RunState, ValidatorSpecRegistry


@pytest.fixture
def registry() -> ValidatorSpecRegistry:
    return ValidatorSpecRegistry()


def test_it_is_the_shared_base(registry: ValidatorSpecRegistry) -> None:
    """Design §10.1: *"the base supplies the dict, the collision policy, and the
    error shape."* Pinned so nobody re-hand-rolls a fifth copy of it — which is
    what this class was until `spec_loader` shipped `BaseSpecRegistry`.

    `origin_of` comes with it, and §9.3 check 4's conflict report needs it: the
    rule is *name both sides*, and a cross-registry pass holds neither side's
    file path otherwise.
    """
    from spec_loader import BaseSpecRegistry

    assert isinstance(registry, BaseSpecRegistry)
    registry.add("shape", validator_record("shape"), origin="pkg/shape.jsonnet")
    assert registry.origin_of("shape") == "pkg/shape.jsonnet"


def test_one_spec_under_two_names_is_rejected(registry: ValidatorSpecRegistry) -> None:
    """One validator under two names would run twice and record two verdicts
    against one handoff version. `pluggy` has the same rule for the same reason,
    and `BaseSpecRegistry` enforces it as a byte-identity check.

    **For this kind that check is unreachable, and the reason is worth recording
    rather than leaving as an unexercised branch.** A validator spec carries its
    own `name`, and `_validate` requires it to match the key, so two keys imply
    two different records and identity can never collide. Our rejection fires
    first and names the actual fault, which is the more useful message.

    Criterion 12's two parameterised instances are unaffected either way — they
    differ in `name` and `args`.
    """
    record = validator_record("shape")
    registry.add("shape", record, origin="a.jsonnet")
    with pytest.raises(SpecInconsistent, match="names itself 'shape'"):
        registry.add("shape_copy", record, origin="b.jsonnet")
    assert registry.names() == ["shape"]


def test_duplicate_name_raises(registry: ValidatorSpecRegistry) -> None:
    """Criterion 3. **It does not overwrite.** Great Expectations logs
    `Overwriting declaration` and proceeds; Inspect AI's `registry_add` is a bare
    dict assignment with no check at all. pandera raises with a message that names
    the collision, and so does this."""
    registry.add("shape", validator_record("shape"), origin="a.jsonnet")
    with pytest.raises(SpecInconsistent, match="shape"):
        registry.add("shape", validator_record("shape", strength="weak"), origin="b.jsonnet")
    assert registry.spec("shape").strength.value == "strong"  # the first one stands


def test_identical_reregistration_is_a_noop(registry: ValidatorSpecRegistry) -> None:
    """`fsspec`'s shape: an error by default, an identical re-registration free.
    Two packages symlinking one general spec must not be a collision."""
    record = validator_record("shape")
    registry.add("shape", record, origin="a.jsonnet")
    registry.add("shape", dict(record), origin="b.jsonnet")
    assert registry.names() == ["shape"]


def test_a_name_that_does_not_resolve_enumerates_the_candidates(
    registry: ValidatorSpecRegistry,
) -> None:
    registry.add("shape", validator_record("shape"), origin="a")
    with pytest.raises(SpecNotFound, match="shape"):
        registry.get("shpae")


def test_the_spec_must_name_itself(registry: ValidatorSpecRegistry) -> None:
    with pytest.raises(SpecInconsistent, match="names itself"):
        registry.add("other", validator_record("shape"), origin="a")


def test_an_inadmissible_spec_never_enters(registry: ValidatorSpecRegistry) -> None:
    bad = validator_record("shape")
    del bad["brief"]
    with pytest.raises(ValidatorInvalid):
        registry.add("shape", bad, origin="a.jsonnet")
    assert registry.names() == []


# --------------------------------------------------------------------------- #
# Criterion 12 — reuse without copy-paste


def test_two_instances_one_body(registry: ValidatorSpecRegistry) -> None:
    """Criterion 12. Two validators differing only in a threshold are two registry
    entries over one body folder — **the args live in the spec**, one spec per
    instance (D5)."""
    shared = {"readme": "checks/threshold/readme.md", "entry": "checks/threshold/entry.sh"}
    for name, limit in (("p95_under_10ms", 10), ("p95_under_50ms", 50)):
        record = validator_record(name, readme=shared["readme"], entry=shared["entry"])
        record["args"] = {"limit_ms": limit}
        registry.add(name, record, origin=f"{name}.jsonnet")

    a, b = registry.spec("p95_under_10ms"), registry.spec("p95_under_50ms")
    assert a.body == b.body
    assert a.args != b.args


def test_entry_sh_runs_without_the_registry(tmp_path: Path) -> None:
    """Criterion 12's second half: *"the shared logic is a plain function that is
    directly testable"*.

    Rev. 2 kept that by convention — pandera documents delegating to a plain
    function because their decorator otherwise makes it unreachable from a test.
    A body needs no such convention: `entry.sh` is runnable from a shell. Run it
    with no registry, no phase runner and no system at all.
    """
    root = write_body(tmp_path / "pkg")
    zone = tmp_path / "zone"
    zone.mkdir()
    (zone / "inputs.json").write_text(json.dumps(["h1", "h2"]))
    subprocess.run(["/bin/sh", str(root / "entry.sh")], cwd=zone, check=True)
    assert json.loads((zone / "verdict.json").read_text()) == {"h1": True, "h2": True}


# --------------------------------------------------------------------------- #
# Criterion 14 — two indexes


def test_two_indexes_disagree(registry: ValidatorSpecRegistry) -> None:
    """Criterion 14. "Who uses this" and "who has used it" answer different
    questions and must be allowed to differ: a validator bound by a kind and never
    run is a real state, and so is one that ran and is bound by nothing."""
    registry.add("bound", validator_record("bound"), origin="a")
    registry.add("ran", validator_record("ran"), origin="b")
    registry.bind("trace", ["bound"])
    registry.record_run("ran")

    assert registry.users_of("bound") == ["handoff_kind:trace"]
    assert registry.has_ever_run("bound") is None
    assert registry.users_of("ran") == []
    assert registry.has_ever_run("ran") is not None


def test_never_run_is_a_state(registry: ValidatorSpecRegistry) -> None:
    """Criterion 14. Stryker is the model — `Pending` is first-class, serialised
    and assertable. Not a set difference computed at the call site."""
    registry.add("idle", validator_record("idle"), origin="a")
    registry.add("busy", validator_record("busy"), origin="b")
    assert registry.run_state("idle") is RunState.NEVER_RUN
    assert registry.never_run() == ["busy", "idle"]

    registry.record_run("busy")
    registry.record_run("busy")
    assert registry.run_state("busy") is RunState.RAN
    assert registry.has_ever_run("busy").runs == 2  # runs constantly vs ran once
    assert registry.never_run() == ["idle"]


def test_history_survives_a_deleted_validator(registry: ValidatorSpecRegistry) -> None:
    """Airflow tombstones and keeps the record; dbt drops the orphan in a silent
    set intersection. **"Has this ever run" is meaningless if deletion erases the
    answer.** §15 O8 leaves pruning open; retention is not open."""
    registry.record_run("since_deleted")
    assert "since_deleted" not in registry
    assert registry.has_ever_run("since_deleted") is not None


def test_a_composite_edge_is_one_of_the_enumerated_kinds(
    registry: ValidatorSpecRegistry,
) -> None:
    """§10.5. A derived static index goes wrong by forgetting a reference kind —
    Airflow reported live assets dead because materialisation through an
    `AssetAlias` was not one of the four tables it joined over. Both edge kinds
    come from one table, so a third is added in one place."""
    registry.add("member", validator_record("member"), origin="a")
    registry.add(
        "pair",
        validator_record("pair", members=("member",), reduce="all", entry=None),
        origin="b",
    )
    registry.bind("trace", ["member"])
    assert registry.users_of("member") == ["composite:pair", "handoff_kind:trace"]
    assert {u.split(":")[0] for u in registry.users_of("member")} <= set(EDGE_KINDS)


def test_a_re_registration_does_not_double_an_edge(registry: ValidatorSpecRegistry) -> None:
    """The index is built in `_admitted`, which runs only on the branch that
    actually stores.

    `add` returns as a **no-op** on a byte-identical re-registration *after*
    `_validate` has run, so a subclass indexing in either `_validate` or an `add`
    override records twice for one spec — and main spec §4.3 makes the same kind
    vendored in two packages a supported case, so it is a live path, not a
    hypothetical.

    It is invisible today because `_edges` holds sets. This pins it for the day
    that changes, which is the only reason the test is worth having.
    """
    member = validator_record("member")
    pair = validator_record("pair", members=("member", "other"), reduce="all")
    registry.add("member", member, origin="a.jsonnet")
    registry.add("pair", pair, origin="b.jsonnet")
    once = registry.users_of("member")

    registry.add("pair", dict(pair), origin="vendored/b.jsonnet")  # the no-op path
    assert registry.users_of("member") == once == ["composite:pair"]


def test_a_closure_phase_edge_is_recorded(registry: ValidatorSpecRegistry) -> None:
    """`interfaces.md` §5.4's **third** edge kind — *"a closure naming a phase
    validator"* — which `EDGE_KINDS` was missing until `demo` found it.

    Its absence was precisely the failure the module docstring cites: Airflow
    reported live assets dead because one reference kind was not in its join, and
    a `users_of` counting only two of three edges reports a validator that two
    closures run as used by nothing. Quoting #58058 and then committing it is
    worth a test rather than a quiet correction.

    A phase validator is a property of the **task**, a kind's validators of the
    **artefact**, and one validator reached both ways must say so.
    """
    registry.add("grounded", validator_record("grounded"), origin="a")
    registry.bind("trace", ["grounded"])
    registry.bind_phase("analyse_trace", ["grounded"])

    assert registry.users_of("grounded") == ["closure:analyse_trace", "handoff_kind:trace"]
    assert {u.split(":")[0] for u in registry.users_of("grounded")} <= set(EDGE_KINDS)
    assert "closure" in EDGE_KINDS


def test_users_of_is_recoverable_but_punishes_a_careless_split(
    registry: ValidatorSpecRegistry,
) -> None:
    """The tagged string is **injective**, and the correct recovery is one
    character from the wrong one.

    No edge tag contains a colon, so the first colon is always the separator and
    `split(":", 1)[1]` round-trips exactly — measured here over names chosen to
    break it. The naive `split(":")[1]` does not.

    **This test asserted the opposite an hour ago and was wrong.** `closure`
    reported the naive parse's behaviour as the format's, I reproduced their
    snippet rather than measuring `users_of`, and "the format is lossy" reached a
    commit with "measured" attached to it. Re-measured from scratch in
    `scratch/impl-2026-08/validator/p4_users_of_recovery.py`. The guard is still
    worth having on the true, weaker premise: a display format that looks
    parseable invites a parse, and the easy one is wrong.
    """
    registry.add("shape", validator_record("shape"), origin="a")
    for awkward in ("a:b", "::", ":y", "x:", "analyse"):
        registry.bind_phase(awkward, ["shape"])

    rows = registry.users_of("shape")
    assert sorted(r.split(":", 1)[1] for r in rows) == ["::", ":y", "a:b", "analyse", "x:"]
    assert sorted(r.split(":")[1] for r in rows) != sorted(r.split(":", 1)[1] for r in rows)


def test_a_validator_only_a_closure_names_is_not_dead(registry: ValidatorSpecRegistry) -> None:
    """The false-positive deadness itself, as the assertion.

    Before the third edge kind existed this returned `[]` — indistinguishable
    from a validator nothing anywhere names, which is what "who uses this" exists
    to answer.
    """
    registry.add("grounded", validator_record("grounded"), origin="a")
    registry.bind_phase("analyse_trace", ["grounded"])
    assert registry.users_of("grounded") == ["closure:analyse_trace"]


def test_an_edge_to_an_unregistered_validator_is_kept(
    registry: ValidatorSpecRegistry,
) -> None:
    """The disagreement is the useful part. dbt's `build_node_edges` silently
    drops any edge whose target is outside a hand-chained set (#14436): the
    `depends_on` is correct, the `child_map` is incomplete, and there is no
    error."""
    registry.bind("trace", ["absent"])
    assert registry.users_of("absent") == ["handoff_kind:trace"]


# --------------------------------------------------------------------------- #
# Criterion 15 — dimensions


def test_binding_symlink_disagreeing_with_inputs_is_reported(tmp_path: Path) -> None:
    """§10.4. The symlinks are a **second, redundant statement of the binding, and
    redundant statements disagree.** `inputs` is the source of truth, so the
    loader reads the field; a symlink naming a kind the field does not is still a
    fault, and §10.3's check 3 catches only the reverse."""
    from validator.spec import admit, check_binding_symlinks

    folder = tmp_path / "shape"
    kinds = tmp_path / "kinds"
    (kinds / "kernel").mkdir(parents=True)
    (kinds / "trace").mkdir(parents=True)
    folder.mkdir()
    (folder / "trace").symlink_to(kinds / "trace")
    (folder / "kernel").symlink_to(kinds / "kernel")

    spec = admit(validator_record("shape", inputs=["trace"]), origin="o")
    (problem,) = check_binding_symlinks(spec, folder)  # the declared one is silent
    assert str(folder / "kernel") in problem.message
    assert "not in inputs ['trace']" in problem.message


def test_dangling_binding_symlink_is_a_load_error(tmp_path: Path) -> None:
    """§10.4, and spec §9.1 says it outright: *"a load error naming the path, not
    a puzzle for the loader to solve"*.

    The *direction* is what has to be spelled out: for a binding symlink,
    unresolvable means **error**; for §9's separation check, unresolvable means
    **reject the validator**. Two uses of one primitive, two failure directions,
    both loud.
    """
    from validator.spec import admit, check_binding_symlinks

    folder = tmp_path / "shape"
    folder.mkdir()
    (folder / "trace").symlink_to(tmp_path / "kinds/gone")

    spec = admit(validator_record("shape", inputs=["trace"]), origin="o")
    (problem,) = check_binding_symlinks(spec, folder)
    assert str(folder / "trace") in problem.message
    assert problem.fatal is True


def test_list_by_dimension(registry: ValidatorSpecRegistry) -> None:
    """Criterion 15, so *"nothing checks trustworthiness on this kind"* is
    answerable rather than a thing somebody has to notice."""
    registry.add("a", validator_record("a", dimension="completeness"), origin="1")
    registry.add("b", validator_record("b", dimension="completeness"), origin="2")
    registry.add("c", validator_record("c", dimension="usability"), origin="3")
    assert registry.list_by_dimension(Dimension.COMPLETENESS) == ["a", "b"]
    assert registry.list_by_dimension(Dimension.TRUSTWORTHINESS) == []
