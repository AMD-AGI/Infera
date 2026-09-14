"""Criteria 1 and 2 — what admission rejects, and what it cannot express.

The interesting half is criterion 2. Spec §3.5's two structural constraints have
no field at all, and that is the *strongest* form they can take: the violation is
unrepresentable rather than checked. It is also the kind of guarantee a later
field addition silently removes, which is why the shape itself is asserted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import pytest

from tests.validator.conftest import validator_record
from validator.protocols import Dimension, Strength, Validator, ValidatorInvalid
from validator.spec import Cost, LogicSource, ValidatorSpec, admit, check_body_resolves


@pytest.mark.parametrize("field", ["brief", "dimension", "strength"])
def test_each_missing_field_rejected(field: str) -> None:
    """Criterion 1. An unlabelled validator would default to being trusted."""
    record = validator_record("v")
    del record[field]
    with pytest.raises(ValidatorInvalid) as exc:
        admit(record, origin="pkg/v.jsonnet")
    assert field in str(exc.value)
    assert "pkg/v.jsonnet" in str(exc.value)


def test_no_field_has_a_default() -> None:
    """Criterion 1, as a shape. `brief`, `dimension` and `strength` are required,
    so nothing can be omitted into a trusted default."""
    required = {n for n, f in ValidatorSpec.model_fields.items() if f.is_required()}
    assert {"brief", "dimension", "strength"} <= required


def test_subtask_field_rejected() -> None:
    """Criterion 2. A validator is single-node: a check that expands into a
    workflow needs its own checks, and the recursion has to stop somewhere."""
    record = validator_record("v") | {"subtasks": ["another"]}
    with pytest.raises(ValidatorInvalid, match="subtasks"):
        admit(record, origin="pkg/v.jsonnet")


def test_own_input_validation_rejected() -> None:
    """Criterion 2. Otherwise validating a validator's inputs needs a validator,
    without bound."""
    record = validator_record("v") | {"input_validation": ["schema"]}
    with pytest.raises(ValidatorInvalid, match="input_validation"):
        admit(record, origin="pkg/v.jsonnet")


def test_neither_field_exists_on_the_model() -> None:
    """Criterion 2, the guarantee rather than the check.

    `extra="forbid"` is what rejects them, and it only does so while the fields
    stay absent. Asserted directly so that adding either one fails here rather
    than quietly turning an unrepresentable violation into a representable one.
    """
    assert "subtasks" not in ValidatorSpec.model_fields
    assert "input_validation" not in ValidatorSpec.model_fields
    assert ValidatorSpec.model_config["extra"] == "forbid"


def test_protocol_is_not_the_admission_gate() -> None:
    """§3.2, and the shipped fact is **stronger** than the design measured.

    The design measured a `runtime_checkable` Protocol: `issubclass` raises on
    non-method members, and `isinstance` is presence-only, so `strength=None`
    passes. `validator/protocols.py`'s `Validator` is not `runtime_checkable` at
    all, so *both* raise — there is no way to use it as a gate even by mistake.
    The presence-only half is re-measured on a local runtime-checkable copy, so
    the recorded fact does not quietly stop being asserted if the decorator is
    ever added.
    """

    class NoneEverywhere:
        brief = None
        inputs = None
        dimension = None
        strength = None

        def __call__(self, handoffs):  # pragma: no cover - never called
            return {}

    with pytest.raises(TypeError, match="runtime_checkable"):
        isinstance(NoneEverywhere(), Validator)
    with pytest.raises(TypeError, match="runtime_checkable|non-method members"):
        issubclass(NoneEverywhere, Validator)  # type: ignore[misc]

    @runtime_checkable
    class Checkable(Protocol):
        brief: str
        inputs: tuple[str, ...]
        dimension: Dimension
        strength: Strength

        def __call__(self, handoffs):  # pragma: no cover - never called
            ...

    assert isinstance(NoneEverywhere(), Checkable)  # presence-only: None passes
    with pytest.raises(TypeError, match="non-method members"):
        issubclass(NoneEverywhere, Checkable)  # type: ignore[misc]

    # The gate that does reject it:
    with pytest.raises(ValidatorInvalid):
        admit(validator_record("v") | {"strength": None}, origin="o")


def test_inputs_as_a_bare_string_rejected() -> None:
    """§3.2's specific trap: a bare string is iterable, so one declared kind
    silently becomes five nonexistent ones and §10.3's resolution check then
    fails five times with a useless message."""
    with pytest.raises(ValidatorInvalid, match="inputs"):
        admit(validator_record("v") | {"inputs": "trace"}, origin="o")


def test_a_list_is_coerced_to_a_tuple() -> None:
    """jsonnet renders a JSON array; the model wants a tuple."""
    assert admit(validator_record("v", inputs=["a", "b"]), origin="o").inputs == ("a", "b")


def test_logic_source_is_recorded_and_unverified() -> None:
    """§3.3. The field round-trips and **nothing checks it** — the graph shape is
    what makes `external_dynamic` trustworthy and the registry cannot see the
    graph (spec open question 3). Asserted so the gap stays visible."""
    spec = admit(validator_record("v", logic_source="external_dynamic"), origin="o")
    assert spec.logic_source is LogicSource.EXTERNAL_DYNAMIC
    # One writer: the tag is the writer and the field is a read.
    assert "logic_source" not in ValidatorSpec.model_fields
    assert spec.tags.logic_source is spec.logic_source


def test_cost_orders_cheap_first() -> None:
    """§5.3 principle 2 — a schema check that costs milliseconds runs before a
    benchmark that costs GPU-hours."""
    assert Cost.SECONDS.rank < Cost.MINUTES.rank < Cost.GPU_HOURS.rank


def test_body_paths_resolve_at_load(tmp_path: Path) -> None:
    """§3.8. A dangling `entry` or `material` is a load error **naming the path**
    — the same rule spec §9.1 gives a binding symlink."""
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "readme.md").write_text("# x")
    spec = admit(validator_record("v", entry="entry.sh"), origin="o")
    with pytest.raises(ValidatorInvalid) as exc:
        check_body_resolves(spec, root)
    assert str(root / "entry.sh") in str(exc.value)

    (root / "entry.sh").write_text("#!/bin/sh\n")
    check_body_resolves(spec, root)  # now resolves

    missing = admit(validator_record("v", materials=("data/fixture.json",)), origin="o")
    with pytest.raises(ValidatorInvalid, match="fixture.json"):
        check_body_resolves(missing, root)


@pytest.mark.parametrize(
    "body",
    [
        {"readme": "", "entry": "entry.sh"},
        {"readme": "readme.md", "entry": ""},
        {"readme": "readme.md", "materials": [""]},
    ],
)
def test_an_empty_path_is_a_fault_not_an_absence(body: dict) -> None:
    """`entry: ""` is the one that bites, and it bites **silently**.

    An empty string is falsy, so `runner_for` reads it as *no entry* and runs the
    validator as agent-bodied — a programmatic check quietly becoming an agent's
    opinion, with no error anywhere. Wrong but not *type*-wrong, which is
    `interfaces.md` §4.11's family and the third instance of it here today, after
    `inputs="trace"` and `kind is PhaseKind.INPUT`.

    `_common.schema.json#/$defs/body` gives all three `minLength: 1`, so this is
    also the two-gates rule turned on ourselves: a document the schema rejects
    must not be one the model accepts.
    """
    record = validator_record("v")
    record["body"] = body
    with pytest.raises(ValidatorInvalid, match="empty path"):
        admit(record, origin="pkg/v.jsonnet")


def test_an_agent_bodied_validator_needs_no_entry() -> None:
    """The case the withdrawn Python callable could not express."""
    spec = admit(validator_record("judge", entry=None), origin="o")
    assert spec.body == {"readme": "readme.md"}
    assert spec.body.get("entry") is None


def test_an_existing_body_is_never_empty() -> None:
    """**The whole safety argument for `{}` being the spelling of absent**, and it
    is invisible in the code, so it is pinned here.

    `_common.schema.json` makes `readme` required with `minLength: 1`, so a body
    that exists always has a non-empty `readme` and therefore is never `{}`.
    `{}` is unambiguous absence rather than "a body whose optional fields were all
    omitted", because there are no all-optional bodies.

    If the schema ever makes `readme` optional this goes red and somebody
    re-reads that decision instead of discovering it.

    **Not the `entry: ""` trap.** That one falsy-checks a field *inside* a body,
    where `""` is a wrong value the schema separately forbids. This falsy-checks
    the body *itself*, where `{}` is the designed absent — the argument for the
    `TypedDict` over a dataclass in the first place. They look alike and are
    opposite.
    """
    common = json.loads(
        (
            Path(__file__).resolve().parents[2] / "spec_loader/schemas/_common.schema.json"
        ).read_text()
    )
    body = common["$defs"]["body"]
    assert body["required"] == ["readme"]
    assert body["properties"]["readme"]["minLength"] == 1

    # So the two are distinguishable, and a composite is the empty one.
    leaf = admit(validator_record("leaf"), origin="o")
    composite = admit(validator_record("pair", members=("a", "b"), reduce="all"), origin="o")
    assert leaf.body and not composite.body


def test_a_reducer_without_members_is_rejected() -> None:
    """`reduce` is a composite's field. A leaf naming one is an author saying
    something the shape cannot mean."""
    with pytest.raises(ValidatorInvalid, match="reduce"):
        admit(validator_record("v", reduce="all"), origin="o")
    with pytest.raises(ValidatorInvalid, match="reducer"):
        admit(validator_record("v", members=("a",)), origin="o")


def test_the_vocabulary_is_the_spec_s() -> None:
    assert {d.value for d in Dimension} == {"completeness", "usability", "trustworthiness"}
    assert {s.value for s in Strength} == {"strong", "long_term_strong", "weak"}


def test_validator_invalid_is_a_spec_invalid() -> None:
    """`load_package` must catch it, and the type says so rather than the docstring.

    It was a bare `ValueError`. `load_package` caught `SpecInvalid` and
    `SpecInconsistent`, so this package's rejections **escaped it entirely** —
    and because `load_package` loads every package in one pass, one validator
    spec aborted the whole multi-package load, against its first stated property:
    *one broken spec must not hide the other nine.*

    `spec_loader` repaired it at their end by catching `ValueError` (80e2f42),
    which holds without four packages remembering a docstring. This is the
    belt-and-braces half: `handoff` and `agent` raise `SpecInvalid` itself, and
    now the kinship is in the type rather than resting on both happening to
    descend from `ValueError`.
    """
    from spec_loader.protocols import SpecInvalid
    from validator.protocols import NestedComposite

    assert issubclass(ValidatorInvalid, SpecInvalid)
    assert issubclass(ValidatorInvalid, ValueError)  # unchanged, so nothing narrows
    assert issubclass(NestedComposite, SpecInvalid)  # the guard funnels through too

    # **The behaviour, not just the relationship** — and the old shape beside it,
    # so the assertion is shown capable of failing rather than merely passing.
    # `handoff` pinned their inverse property by applying the refactor that would
    # break it and watching the test go red; this is the cheap version of that.
    class BareValueError(ValueError):
        """What `ValidatorInvalid` was until 6f5c6ef."""

    caught = []
    for cls in (BareValueError, ValidatorInvalid, NestedComposite):
        try:
            raise cls("rejected")
        except SpecInvalid:
            caught.append(cls.__name__)
        except ValueError:
            pass

    assert caught == ["ValidatorInvalid", "NestedComposite"]
    assert "BareValueError" not in caught  # the old shape escapes a narrow handler
