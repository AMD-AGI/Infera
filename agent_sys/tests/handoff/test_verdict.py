"""Criteria 7 and 8: recording a verdict does not move the digest, and the
history is complete.

Criterion 7 is a **structural** fact: `validation.yaml` is a sibling of
`content/`, so the digest cannot see it. The alternative — hash the version
directory and exclude the file by name — is not stable, because it re-hashes on
every rewrite unless the exclusion is applied at every level of the walk.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from handoff import Manifest, Verdict
from handoff import verdict as verdict_mod
from handoff.errors import Malformed
from task_graph.ids import AgentId, TaskId
from tests.handoff.conftest import make_content


def _verdict(name: str = "check_trace_shape", *, result: bool = True) -> Verdict:
    return Verdict(
        validator=name,
        result=result,
        strength="strong",
        dimension="shape",
        task_id=TaskId.new(),
        agent_id=AgentId.new(),
        environment={"gpu": "MI300X", "image": "rocm:7.0"},
        at=datetime.now(timezone.utc),
    )


def test_verdict_does_not_move_digest(kinded_store, tmp_path: Path) -> None:
    store, hid = kinded_store
    version = store.put(hid, make_content(tmp_path / "c"), producer=TaskId.new())
    before = store.get_manifest(hid, version).digest

    store.record_verdict(hid, version, _verdict())

    assert store.get_manifest(hid, version).digest == before
    store.copy_out(hid, version, tmp_path / "out")  # re-verifies; would raise


def test_rewriting_verdict_does_not_move_digest(kinded_store, tmp_path: Path) -> None:
    store, hid = kinded_store
    version = store.put(hid, make_content(tmp_path / "c"), producer=TaskId.new())
    before = store.get_manifest(hid, version).digest

    for i in range(3):
        store.record_verdict(hid, version, _verdict(f"v{i}"))
    assert store.get_manifest(hid, version).digest == before
    assert len(store.read_verdicts(hid, version)) == 3
    store.copy_out(hid, version, tmp_path / "out")


def test_history_is_complete(kinded_store, tmp_path: Path) -> None:
    """Criterion 8: per validator, each execution result plus the task, the
    versioned agent, and the environment — so "has this been checked, by what,
    when, and did it pass" is answerable from the handoff alone."""
    store, hid = kinded_store
    version = store.put(hid, make_content(tmp_path / "c"), producer=TaskId.new())

    first = _verdict("shape", result=False)
    second = _verdict("shape", result=True)
    third = _verdict("usable", result=True)
    for v in (first, second, third):
        store.record_verdict(hid, version, v)

    got = store.read_verdicts(hid, version)
    assert [v.validator for v in got] == ["shape", "shape", "usable"]
    assert [v.result for v in got] == [False, True, True]
    assert got[0].task_id == first.task_id
    assert got[0].agent_id == first.agent_id
    assert got[0].environment == {"gpu": "MI300X", "image": "rocm:7.0"}
    assert got[0].strength == "strong" and got[0].dimension == "shape"
    assert got[0].at == first.at


def test_an_empty_list_and_a_missing_file_mean_different_things(
    kinded_store, tmp_path: Path
) -> None:
    """`validation.yaml` is created empty at publication rather than on first
    verdict. SVR requires an empty array where no policy applies, and Nix's
    source carries the rule verbatim: *"absent whitelist, and present but empty
    whitelist mean very different things."*"""
    store, hid = kinded_store
    version = store.put(hid, make_content(tmp_path / "c"), producer=TaskId.new())
    path = store.root / str(hid) / f"v{version}" / verdict_mod.VERDICT_FILE

    assert path.is_file() and store.read_verdicts(hid, version) == []

    path.unlink()
    with pytest.raises(Malformed, match="missing"):
        store.read_verdicts(hid, version)


def test_an_unreadable_row_is_named(tmp_path: Path) -> None:
    path = tmp_path / verdict_mod.VERDICT_FILE
    path.write_text("verdicts:\n  - validator: x\n", encoding="utf-8")
    with pytest.raises(Malformed, match="unreadable verdict row"):
        verdict_mod.read(path)


def test_a_verdict_with_no_agent_round_trips_as_null(kinded_store, tmp_path: Path) -> None:
    """A **script body has no agent**, and the record says so by having none.

    Before this, `Verdict.agent_id` was required, so `validator` fell back to
    *the producing agent's* id with `attributed: False` beside it in
    `environment` — a record whose own attribution field named the producer, in
    the one artefact criterion 8 asserts over. `validator/phase.py` said in a
    docstring that it did this only because the field was non-optional.
    """
    store, hid = kinded_store
    version = store.put(hid, make_content(tmp_path / "c"), producer=TaskId.new())

    unattributed = Verdict(
        validator="pytest_harness",
        result=True,
        strength="strong",
        dimension="shape",
        task_id=TaskId.new(),
        agent_id=None,
        environment={"source": "script"},
        at=datetime.now(timezone.utc),
    )
    store.record_verdict(hid, version, unattributed)

    (got,) = store.read_verdicts(hid, version)
    assert got.agent_id is None
    assert got == unattributed, "the whole record round-trips, not just the field"

    raw = yaml.safe_load(
        (store.root / str(hid) / f"v{version}" / verdict_mod.VERDICT_FILE).read_text()
    )
    assert raw["verdicts"][0]["agent_id"] is None
    assert "agent_id" in raw["verdicts"][0], (
        "written as an explicit null, not omitted: an absent key and a null "
        "value are two different records"
    )


def test_an_attributed_and_an_unattributed_verdict_coexist(kinded_store, tmp_path: Path) -> None:
    """The history holds both, and they are distinguishable without a side channel."""
    store, hid = kinded_store
    version = store.put(hid, make_content(tmp_path / "c"), producer=TaskId.new())

    agent_run = _verdict("by_an_agent")
    script_run = Verdict(
        validator="by_a_script",
        result=True,
        strength="weak",
        dimension="shape",
        task_id=TaskId.new(),
        agent_id=None,
        environment={},
        at=datetime.now(timezone.utc),
    )
    store.record_verdict(hid, version, agent_run)
    store.record_verdict(hid, version, script_run)

    by_name = {v.validator: v for v in store.read_verdicts(hid, version)}
    assert by_name["by_an_agent"].agent_id == agent_run.agent_id
    assert by_name["by_a_script"].agent_id is None


def test_a_row_that_never_mentioned_the_field_is_unreadable(tmp_path: Path) -> None:
    """`null` is a statement; a **missing key** is a row written by something
    that did not know the field exists. Not the same, and not silently equal."""
    path = tmp_path / verdict_mod.VERDICT_FILE
    path.write_text(
        "verdicts:\n"
        "  - validator: x\n"
        "    result: true\n"
        "    strength: strong\n"
        "    dimension: shape\n"
        f"    task_id: {TaskId.new()}\n"
        "    environment: {}\n"
        "    at: '2026-08-28T00:00:00+00:00'\n",
        encoding="utf-8",
    )
    with pytest.raises(Malformed, match="unreadable verdict row"):
        verdict_mod.read(path)


def test_the_manifest_does_not_carry_done_by_self_check() -> None:
    """A **deadline**, guarded rather than remembered — `interfaces.md` §5.14.

    `monitor` §4.1.2 and §9 carry `done_by_self_check` as an unbuilt `handoff`
    item, and the obvious home is `Manifest`. **Do not put it there.** §5.14
    resolved publication to a supervisor-side pull, and under that model the
    completeness gate runs *before any manifest exists* — so a manifest field
    could not carry it even once written.

    Measured today: the name appears only in `agent`'s test stubs, and
    `agent/gate.py` reads it with `getattr(manifest, "done_by_self_check",
    None)`, treating absent as not-a-failure. So the gate is currently inert
    and nothing breaks — which is exactly why this is easy to "fix" the wrong
    way.

    > *"Cheap to land as a zone artefact now and expensive to move after
    > `handoff` builds it on the manifest."*

    This asserts the **absence of the mechanism**, not the presence of a
    consequence — the rule that caught the escape-hatch gap closing. If a field
    of this name lands on `Manifest`, this fails and points here.
    """
    fields = {f.name for f in dataclasses.fields(Manifest)}
    assert "done_by_self_check" not in fields, (
        "§5.14: publication is a supervisor-side pull and the gate runs before "
        "a manifest exists, so this cannot live on the Manifest. It belongs in "
        "the zone. Moving it later is the expensive path this test exists to "
        "prevent"
    )
    assert fields == {"digest", "algorithm", "kind", "producer", "created_at"}, (
        "the Manifest's field set changed; if that was deliberate, say so here"
    )
