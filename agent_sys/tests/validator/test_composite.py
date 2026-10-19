"""Criterion 13, and §6.4's two silent passes.

The reduce axis is the trap: a composite of *m* validators over *n* handoffs is an
*m×n* grid collapsing to an *n*-entry dict, and the shape a naive implementation
reaches for is the broadcast — the only one that both keeps the declared return
type and lets the reducer reduce over validators. Under it a handoff that passed
every member is recorded `False`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from spec_loader import schema_for, validate
from task_graph.ids import HandoffId
from tests.validator.conftest import validator_record
from validator.composite import Composite, check_coverage
from validator.protocols import Dimension, NestedComposite, Strength, ValidatorInvalid
from validator.reducers import AllReducer, get_reducer

#: The one schema, read through `spec_loader`'s accessor rather than by path.
#: There were two for a while — one here, one in `spec_loader/schemas/` — and two
#: schemas for one spec kind is `engineer_principle.md` §1's two writers. The copy
#: in this package is deleted; `schema_for` is the single reader, which is what
#: stops four modules hand-rolling an `importlib.resources` read.
SCHEMA = schema_for("validator")


def _faults(record: Mapping[str, Any]) -> list[Any]:
    """Schema-check one record through `spec_loader.validate`.

    Not `jsonschema.validate` directly: the schema `$ref`s `_common.schema.json`,
    which needs a registry, and `validate` is the system's single enforcement
    point and already carries one. Faults are **returned**, never raised — one
    broken spec must not hide the other nine.

    The record goes in as a mapping. `validate` took `bytes` and parsed until
    rev. 10; the parse moved to the package with the source format, so a caller
    hands over the document it already has instead of serialising it for
    somebody else to read back.
    """
    return validate(record, SCHEMA, origin="<test>")


class Slot:
    """The runtime handoff a composite sees. `task_graph.Handoff.type`."""

    def __init__(self, kind: str) -> None:
        self.type = kind


class Leaf:
    def __init__(self, name: str, kinds: tuple[str, ...], answers: Mapping[HandoffId, bool]):
        self.name = name
        self.brief = name
        self.inputs = kinds
        self.dimension = Dimension.COMPLETENESS
        self.strength = Strength.STRONG
        self._answers = dict(answers)

    def __call__(self, handoffs: Mapping[HandoffId, Any]) -> dict[HandoffId, bool]:
        return {h: self._answers[h] for h in handoffs}


def composite(members, reduce=None) -> Composite:
    return Composite(
        "pair",
        brief="two checks, one verdict",
        dimension=Dimension.COMPLETENESS,
        strength=Strength.STRONG,
        members=members,
        reduce=reduce or AllReducer(),
    )


def test_composite_reduces_per_handoff() -> None:
    """Criterion 13, and the reduce axis.

    Inspect AI's shape, arrived at independently: walk the keys and reduce each
    key **across members**. `h1` passed both members and must stay `True` — the
    broadcast shape would report it `False` because some other handoff failed.
    """
    h1, h2 = HandoffId.new(), HandoffId.new()
    slots = {h1: Slot("trace"), h2: Slot("trace")}
    result = composite(
        [
            Leaf("a", ("trace",), {h1: True, h2: True}),
            Leaf("b", ("trace",), {h1: True, h2: False}),
        ]
    )(slots)
    assert result == {h1: True, h2: False}


def test_a_composite_is_type_substitutable_for_a_leaf() -> None:
    """What keeps the reducer out of the phase: `run_phase` cannot tell them
    apart, so spec §5.5's one rule stays the phase's only rule."""
    h = HandoffId.new()
    leaf = Leaf("a", ("trace",), {h: True})
    both = [leaf({h: Slot("trace")}), composite([leaf])({h: Slot("trace")})]
    assert both[0] == both[1] == {h: True}


def test_ragged_membership_is_permitted() -> None:
    """Inspect rejects mismatched keys outright, because its keys are epochs of
    one sample. **Ours are handoffs and members legitimately declare different
    input kinds**, so adopting that rejection would forbid a composite the spec
    permits."""
    trace, kernel = HandoffId.new(), HandoffId.new()
    slots = {trace: Slot("trace"), kernel: Slot("kernel")}
    result = composite(
        [
            Leaf("t", ("trace",), {trace: True}),
            Leaf("k", ("kernel",), {kernel: False}),
        ]
    )(slots)
    assert result == {trace: True, kernel: False}


def test_composite_refuses_an_uncovered_handoff() -> None:
    """§6.4. **`all([])` is `True`** — a handoff no member declares would be
    folded from an empty list and pass, having been checked by nothing. Vacuous
    truth arriving from the standard library is exactly the silent pass spec §1
    exists to prevent."""
    assert all([]) is True  # the standard library's part, stated
    orphan = HandoffId.new()
    with pytest.raises(ValidatorInvalid, match="covers no validator"):
        composite([Leaf("t", ("trace",), {})])({orphan: Slot("nobody_checks_this")})


def test_coverage_is_checked_at_admission() -> None:
    """Membership and declared inputs are both static, so the fault is an
    author's and belongs in the load report, not in a run."""
    check_coverage([("trace",), ("kernel",)], ["trace", "kernel"])
    with pytest.raises(ValidatorInvalid, match=r"\['deploy'\]"):
        check_coverage([("trace",)], ["trace", "deploy"])


def test_member_omitting_a_declared_handoff_raises() -> None:
    """§6.4. DeepEval's unreached DAG node leaves `score` as `None`,
    `is_successful` catches the `TypeError` and sets `success = False`, so an
    unreached terminal and a real zero report identically."""
    h = HandoffId.new()

    class Silent(Leaf):
        def __call__(self, handoffs):
            return {}

    with pytest.raises(ValidatorInvalid, match="no verdict for"):
        composite([Silent("s", ("trace",), {})])({h: Slot("trace")})


def test_nested_composite_rejected_by_guard() -> None:
    """The safety net for the path that bypasses the loader — a composite can be
    constructed directly in a test."""
    inner = composite([Leaf("a", ("trace",), {})])
    with pytest.raises(NestedComposite, match="may not contain a composite"):
        composite([inner])


def test_nested_composite_rejected_by_schema() -> None:
    """Criterion 13's enforcement half, and it is OpenAI's mechanism: `GraderMulti`
    appears in the three top-level grader unions and is **absent from
    `GraderMulti.properties.graders`**. Not a runtime check and not a
    documentation note — the nested union simply omits it.

    Ours has one list to maintain: `members` is `array of string`, so a composite
    inlined as a member is not expressible at all, and a composite *named* as a
    member is caught by §10.3 check 5 once both are loaded.
    """
    assert not _faults(validator_record("pair", members=["a", "b"], reduce="all"))

    nested = validator_record("pair", members=["a", "b"], reduce="all")
    nested["members"] = [validator_record("inner", members=["a", "b"], reduce="all")]
    assert _faults(nested)


def test_the_schema_and_the_model_agree_on_what_is_forbidden() -> None:
    """The caution carried from reading OpenAI's schema: they maintain two unions
    by hand and the two disagree in both directions. Ours has one list, and this
    is what notices if it drifts from the model.

    **`_PENDING` lived here and is gone**, which is the scaffolding working. The
    `agent` key had to cross a boundary two people own, and exact equality made
    either side landing alone a red shared suite — symmetric, so ordering did not
    help. The set named the key while it crossed and carried an assertion that
    **failed once both sides had it**, so the commit that landed the field could
    not forget to remove the set. It duly failed on me. `task_graph`'s move on
    their `task_of` drift test, and worth reaching for again.

    **Which direction to keep if this coupling ever stops being worth it**, since
    the reasoning is the part that would be lost: the schema runs *first* (main
    design §4), so a **schema key with no model field is a loud rejection** at
    admission, while a **model field with no schema key is a dead field no
    document can reach** — the `environment` defect exactly. Relax the loud
    direction, never the silent one.
    """
    from validator.spec import ValidatorSpec

    assert set(SCHEMA["properties"]) == set(ValidatorSpec.model_fields)
    assert SCHEMA["additionalProperties"] is False


def test_an_absent_reducer_enumerates_the_table() -> None:
    """§6.2. `all` and `any` are **ours**, not adopted: Inspect's registered set is
    `collect, at_least, pass_at, pass_k, max, mean, median, mode`, and
    `multi_scorer([...], "all")` raises `LookupError` there. The alpha registers
    one, and the table is what makes the others additions."""
    assert get_reducer("all").name == "all"
    with pytest.raises(ValidatorInvalid, match=r"registered: \['all'\]"):
        get_reducer("at_least(2)")
