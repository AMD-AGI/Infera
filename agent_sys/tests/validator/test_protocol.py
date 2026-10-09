"""Criterion 4 — one verdict per input handoff, for a validator taking three."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from task_graph.ids import HandoffId
from validator.protocols import Dimension, PhaseKind, Strength, ValidatorInvalid
from validator.report import PhaseOutcome


class ThreeInputs:
    """A leaf in the shape `Validator` declares. Not registered anywhere: the
    Protocol is a static type and the registry admits *specs*, not objects."""

    name = "three"
    brief = "checks three kinds at once — completeness dimension"
    inputs = ("trace", "kernel", "config")
    dimension = Dimension.COMPLETENESS
    strength = Strength.STRONG

    def __call__(self, handoffs: Mapping[HandoffId, Any]) -> dict[HandoffId, bool]:
        return {h: True for h in handoffs}


def test_three_handoffs_three_verdicts() -> None:
    """Criterion 4. **Keyed by uuid, not by kind**: a task with two inputs of the
    same kind would be ambiguous under a kind key."""
    ids = [HandoffId.new() for _ in range(3)]
    result = ThreeInputs()({h: object() for h in ids})
    assert set(result) == set(ids)
    assert len(result) == 3


def test_two_inputs_of_one_kind_are_two_entries() -> None:
    """Why the key is a uuid, asserted rather than argued."""
    a, b = HandoffId.new(), HandoffId.new()
    result = ThreeInputs()({a: object(), b: object()})
    assert result == {a: True, b: True}


def test_no_outcome_field_defaults_to_success() -> None:
    """§11.2. The guarantee is the shape, so it is asserted over the fields.

    JUnit XML makes pass the **structural default** — a `testcase` with no child
    element is a pass — so a producer that forgets to emit `<skipped/>` emits a
    pass and nothing detects it.
    """
    import dataclasses

    outcome = PhaseOutcome.fold(PhaseKind.INPUT)
    assert outcome.empty is True
    assert outcome.passed is False
    # There is no argument list to `fold` that produces a pass from nothing.
    with pytest.raises(TypeError):
        PhaseOutcome.fold(PhaseKind.INPUT, True)  # type: ignore[misc]

    # And **no field has a default at all**, so the constructor cannot be used to
    # leave one unsaid either. That is stronger than "nothing defaults to
    # success": `empty` had defaulted to `True`, which is safe, and it was still
    # the wrong shape on the one object whose point is that nothing defaults.
    # `tests/interfaces/test_declaration_conformance.py` found it by comparing
    # defaults rather than names.
    defaulted = [
        f.name
        for f in dataclasses.fields(PhaseOutcome)
        if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
    ]
    assert defaulted == []


def test_outcome_matches_the_declared_shape() -> None:
    """`report.PhaseOutcome` and `protocols.PhaseOutcome` are one contract in two
    files, and this is that duplication's price — the same bargain
    `tests/interfaces/test_pushable.py` makes for `Pushable`.

    `protocols.py` is declarations only, so it cannot carry `fold`; the fields and
    the declared `passed` are what must agree.
    """
    import dataclasses

    from validator import protocols

    declared = {f.name for f in dataclasses.fields(protocols.PhaseOutcome)}
    implemented = {f.name for f in dataclasses.fields(PhaseOutcome)}
    assert declared == implemented
    assert isinstance(protocols.PhaseOutcome.passed, property)
    assert isinstance(PhaseOutcome.passed, property)


def test_the_store_stub_matches_the_real_seam() -> None:
    """The stub `conftest.py` uses is reconciled against `handoff`, not left to
    drift.

    `docs/implementation-stage.md` §4.1 tells wave 1 to *satisfy the Protocol with
    a stub in your own tests/*. A stub that is never checked against the real
    thing is a merge conflict scheduled for later, so the two verdict operations
    this module calls are compared signature for signature. Importing `handoff`
    here is fine: tests are not under `interfaces.md` §4's import rule.
    """
    import inspect

    from handoff.protocols import HandoffStore
    from tests.validator.conftest import MemoryHandoffStore

    for name in ("record_verdict", "read_verdicts"):
        declared = inspect.signature(getattr(HandoffStore, name))
        stubbed = inspect.signature(getattr(MemoryHandoffStore, name))
        assert list(declared.parameters) == list(stubbed.parameters), name
        assert str(declared.return_annotation) == str(stubbed.return_annotation), name


def test_the_validation_zone_stub_matches_the_real_seam() -> None:
    """The second time a neighbour's type changed under this package, and the
    first time it would have shipped.

    `env_mgr.ValidationZone.materials` was `tuple[str, ...]` and became
    `Mapping[HandoffId, str]` (789796d) — at my own request. `phase.py` did
    `tuple(placed.materials)`, which over a mapping yields the **keys**, so
    `materials.json` would have contained handoff ids where a body expects paths.
    **Every test passed**, because the stub in `conftest.py` still returned a
    tuple: a stub that has drifted asserts the old contract with full confidence.

    So the field types are compared against the real `NamedTuple` rather than
    trusted. Importing `env_mgr` here is fine — tests are not under
    `interfaces.md` §4's import rule, and this is the same bargain
    `test_the_store_stub_matches_the_real_seam` makes for `handoff`.
    """
    import typing

    from env_mgr.protocols import ValidationZone

    declared = typing.get_type_hints(ValidationZone)
    assert set(declared) == {"root", "phase", "materials"}

    # `materials` is a mapping, not a sequence — the distinction that broke.
    origin = typing.get_origin(declared["materials"])
    assert origin is not None and issubclass(origin, Mapping)


def test_a_member_omitting_a_declared_handoff_is_a_fault() -> None:
    """§6.4, at the leaf. `None` folded as falsy is indistinguishable from a
    genuine `False`, so the omission is named rather than defaulted."""
    from validator.composite import Composite
    from validator.reducers import AllReducer

    class Forgetful(ThreeInputs):
        name = "forgetful"
        inputs = ("trace",)

        def __call__(self, handoffs):
            return {}

    hid = HandoffId.new()

    class Slot:
        type = "trace"

    composite = Composite(
        "pair",
        brief="b",
        dimension=Dimension.COMPLETENESS,
        strength=Strength.STRONG,
        members=[Forgetful()],
        reduce=AllReducer(),
    )
    with pytest.raises(ValidatorInvalid, match="no verdict"):
        composite({hid: Slot()})
