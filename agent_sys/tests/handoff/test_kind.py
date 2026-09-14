"""Criteria 1, 4 and 12 — the kind's own load-time checks.

Spec §8 lists five. Check 1 is the registry's collision policy, check 3 runs in
the closure pass (`test_binding.py`), and 2, 4 and 5 are here.
"""

from __future__ import annotations

import pytest

from handoff import HandoffSpecRegistry, Scope
from handoff import kind as kind_mod
from handoff.errors import Malformed

SPEC = {
    "name": "trace",
    "content_type": "reproducible",
    "scope": "fixed.required",
    "items_schema": {
        "type": "object",
        "properties": {"script": {}, "result": {}, "env": {}},
        "required": ["env"],
        "additionalProperties": False,
    },
    "validators": ["check_trace_shape"],
}


def _spec(**over) -> dict:
    out = dict(SPEC)
    out.update(over)
    return out


def test_a_well_formed_kind_admits() -> None:
    reg = HandoffSpecRegistry()
    assert reg.check(SPEC, origin="h/trace.jsonnet") == []
    reg.add("trace", SPEC, origin="h/trace.jsonnet")
    assert reg.names() == ["trace"]
    assert reg.kind_of("trace").scope is Scope.FIXED_REQUIRED


@pytest.mark.parametrize("key", kind_mod.REQUIRED_KEYS)
def test_missing_key_rejected(key: str) -> None:
    """Criterion 1, first half: rejected with the file path and the offending key."""
    spec = {k: v for k, v in SPEC.items() if k != key}
    problems = HandoffSpecRegistry().check(spec, origin="h/trace.jsonnet")
    assert len(problems) == 1
    assert problems[0].origin == "h/trace.jsonnet"
    assert f"$.{key}" in problems[0].message
    assert problems[0].fatal


def test_an_absent_validators_key_is_reported_not_a_construction_error() -> None:
    """The schema does not require `validators`, so an absent one must reach
    check 4 and be *reported* — that is the whole of spec §5.3's escape hatch.
    A construction error here would turn a reportable finding into a crash one
    layer too early."""
    spec = {k: v for k, v in SPEC.items() if k != "validators"}
    problems = HandoffSpecRegistry().check(spec, origin="h/loose.jsonnet")
    assert [p.path for p in problems] == ["$.validators"]

    reg = HandoffSpecRegistry(allow_no_validator=True)
    reg.add("trace", spec, origin="h/loose.jsonnet")
    assert list(reg.load_report().without_validator) == ["trace"]


def test_an_absent_items_schema_permits_anything_and_check_five_says_so() -> None:
    spec = {k: v for k, v in SPEC.items() if k != "items_schema"}
    assert kind_mod.from_spec(spec, origin="x").items_schema == {}
    problems = HandoffSpecRegistry().check(spec, origin="h/trace.jsonnet")
    assert any("'env'" in p.message for p in problems)


def test_items_schema_not_a_schema_is_one_error() -> None:
    """Criterion 1, second half. **One** error: `$ref`-ing the metaschema turns
    this into eight identical ones, and a package author reading eight copies
    of `is not of type 'object', 'boolean'` learns nothing."""
    problems = HandoffSpecRegistry().check(
        _spec(items_schema={"type": "nonsense"}), origin="h/trace.jsonnet"
    )
    assert len(problems) == 1
    assert problems[0].path == "$.items_schema"
    assert "h/trace.jsonnet" in problems[0].message


def test_an_unknown_content_type_or_scope_is_rejected() -> None:
    for over in ({"content_type": "prose"}, {"scope": "fixed.mandatory"}):
        problems = HandoffSpecRegistry().check(_spec(**over), origin="h/x.jsonnet")
        assert len(problems) == 1 and problems[0].fatal


def test_script_without_env_rejected_at_kind_admission() -> None:
    """Criterion 4, and the phase is the subtle part: **no content exists** at
    kind-admission time, so the question is whether a document satisfying this
    `items_schema` *could* carry `script` with no `env`."""
    open_ended = _spec(items_schema={"type": "object", "properties": {"script": {}, "result": {}}})
    problems = HandoffSpecRegistry().check(open_ended, origin="h/trace.jsonnet")
    assert any("'env'" in p.message for p in problems)

    built = kind_mod.from_spec(SPEC, origin="ok")
    assert not kind_mod.permits_exec_without_env(built)


def test_a_closed_schema_that_cannot_carry_script_needs_no_env() -> None:
    closed = _spec(
        items_schema={
            "type": "object",
            "properties": {"result": {}},
            "additionalProperties": False,
        }
    )
    assert HandoffSpecRegistry().check(closed, origin="h/x.jsonnet") == []


def test_no_validator_rejected() -> None:
    """Criterion 12: off by default. 'Checkable by construction' is spec §2's
    third principle and a kind with no validator cannot be admitted."""
    problems = HandoffSpecRegistry().check(_spec(validators=[]), origin="h/loose.jsonnet")
    assert len(problems) == 1
    assert problems[0].fatal
    assert problems[0].path == "$.validators"


def test_flag_reports_by_name() -> None:
    """The other half of criterion 12: with the flag the kind loads **and** its
    name appears in the report. A list on a value, not a `logging.warning` — an
    assertion over a log capture is a test of the logging configuration."""
    reg = HandoffSpecRegistry(allow_no_validator=True)
    spec = _spec(name="loose", validators=[])
    problems = reg.check(spec, origin="h/loose.jsonnet")
    assert len(problems) == 1
    assert not problems[0].fatal, "the escape hatch is report severity, not fatal"
    assert "loose" in problems[0].message

    reg.add("loose", spec, origin="h/loose.jsonnet")
    reg.add("trace", SPEC, origin="h/trace.jsonnet")
    report = reg.load_report()
    assert list(report.without_validator) == ["loose"]
    assert list(report.admitted) == ["loose", "trace"]


def test_the_flag_disables_no_existing_validator() -> None:
    reg = HandoffSpecRegistry(allow_no_validator=True)
    reg.add("trace", SPEC, origin="h/trace.jsonnet")
    assert reg.validators_for("trace") == ["check_trace_shape"]
    assert list(reg.load_report().without_validator) == []


def test_a_duplicate_name_is_an_error_and_an_identical_one_is_a_no_op() -> None:
    """Deliberately the opposite of `task_graph.Registry`, which overwrites so a
    test can swap a component. Two specs claiming one name is a fault."""
    from spec_loader.protocols import SpecInconsistent

    reg = HandoffSpecRegistry()
    reg.add("trace", SPEC, origin="a.jsonnet")
    reg.add("trace", dict(SPEC), origin="a.jsonnet")  # byte-identical: a no-op
    with pytest.raises(SpecInconsistent) as exc:
        reg.add("trace", _spec(content_type="text"), origin="b.jsonnet")
    assert "a.jsonnet" in str(exc.value) and "b.jsonnet" in str(exc.value)


def test_get_enumerates_the_candidates() -> None:
    from spec_loader.protocols import SpecNotFound

    reg = HandoffSpecRegistry()
    reg.add("trace", SPEC, origin="a.jsonnet")
    with pytest.raises(SpecNotFound, match="have: trace"):
        reg.get("trace_getter")


def test_version_is_maintenance_metadata_and_nothing_reads_it() -> None:
    built = kind_mod.from_spec(_spec(version="3"), origin="x")
    assert built.version == "3"
    assert kind_mod.from_spec(SPEC, origin="x").version is None


def test_from_spec_names_the_origin_it_was_given() -> None:
    with pytest.raises(Malformed, match="h/trace.jsonnet"):
        kind_mod.from_spec({"name": "x"}, origin="h/trace.jsonnet")


def test_malformed_is_not_a_spec_fault_and_the_conversion_is_deliberate() -> None:
    """**`Malformed` must not subclass `SpecInvalid`**, and `_validate` must convert.

    `spec-loader` asked whether `Malformed` should gain `SpecInvalid` kinship,
    the way `validator`'s `ValidatorInvalid` did, so that `load_package` catches
    it by inheritance rather than by coincidence. The answer is no, and the
    reason is what `Malformed` actually means here.

    Measured: this package raises it in about twenty places, and **six** are
    load-time spec faults (all in `kind.py`). The rest are runtime artefact
    faults — a FIFO in the digest walk, `-0.0`, an unreadable manifest, a store
    with no root, a `tree` item opened as a stream, a locality violation, a
    `copy_out` destination that already exists. Making the class a `SpecInvalid`
    would assert that a device node in a content tree is *a spec that failed its
    schema*. That is the same kind of lie as the one it would prevent, pointing
    the other way, and `engineer_principle.md` §2 is the rule against it.

    So the conversion stays explicit, at the one boundary where a `Malformed`
    genuinely *is* a spec fault — `_validate`. This asserts the **mechanism**,
    not the consequence: a tidy-up that simplifies that line to re-raise the
    `Malformed` it already holds — which reads like a cleanup and loses no
    information — fails here.
    """
    from handoff.errors import BindingConflict, Malformed
    from spec_loader.protocols import SpecInconsistent, SpecInvalid

    assert not issubclass(Malformed, SpecInvalid), (
        "Malformed is mostly a runtime artefact fault, not a spec fault; "
        "kinship here would be a false claim about a FIFO in a content tree"
    )
    assert issubclass(Malformed, ValueError), (
        "but it must stay a ValueError: load_package catches ValueError so that "
        "one package's choice of exception type cannot abort a multi-package load"
    )

    reg = HandoffSpecRegistry()
    with pytest.raises(SpecInvalid) as exc:
        reg.add("trace", _spec(items_schema="notaschema"), origin="h/trace.jsonnet")
    assert not isinstance(exc.value, Malformed), (
        "_validate must convert rather than re-raise: a caller distinguishing a "
        "spec fault from an artefact fault has only the type to go on"
    )

    # `validator`'s pattern, and it is better than mine here: **keep the broken
    # shape in the assertion** rather than applying the refactor once by hand
    # and watching it go red. A tamper-and-restore proves the guard worked on
    # the day someone ran it; this proves it still can, on every run, including
    # after somebody refactors the guard itself.
    class TidiedUp(HandoffSpecRegistry):
        """The "simplification" this test exists to catch: re-raise the
        `Malformed` already in hand instead of converting it."""

        def _validate(self, name, spec, *, origin):  # noqa: ANN001, ANN202
            kind_mod.from_spec(spec, origin=origin)  # lets Malformed out

    with pytest.raises(Malformed) as leaked:
        TidiedUp().add("trace", _spec(items_schema="notaschema"), origin="h/x.jsonnet")
    assert not isinstance(leaked.value, SpecInvalid), (
        "the tidied-up form must still be catchable as the defect it is; if this "
        "starts failing, Malformed has gained SpecInvalid kinship and the "
        "assertion above has quietly stopped meaning anything"
    )

    # The one class here that *is* kin, and it says so because it genuinely is:
    # two loaded specs disagreeing is exactly SpecInconsistent's meaning.
    assert issubclass(BindingConflict, SpecInconsistent)
