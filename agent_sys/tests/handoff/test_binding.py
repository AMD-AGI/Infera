"""Criteria 9 and 10: the many-to-many binding, and what a mismatch looks like.

**No precedent was found for this check.** SQLAlchemy's `back_populates` — the
closest analogue — does not verify that the two sides agree; GraphQL Federation
deleted the requirement in Fed 2; Kubernetes treats a dangling ownerRef as
absent. The design implements the spec, and the deviation is that nobody else
does this (`design.md` D2, O6).
"""

from __future__ import annotations

import pytest

from handoff import HandoffSpecRegistry
from handoff.errors import BindingConflict
from spec_loader.protocols import SpecInconsistent, SpecNotFound


class FakeValidatorRegistry:
    """Satisfies `SpecRegistry` with a dict. A neighbour's implementation is
    not imported to run this package's tests.

    **`get` raises `SpecNotFound`, not `KeyError`, and that is not cosmetic.**
    It used to be a bare `self._specs[name]`, which is a different exception
    from the real `spec_loader/registry.py:193` — *"raises `SpecNotFound`,
    naming the kind, the name, and the candidates."* Unreachable today, because
    `check_bindings` guards with `vname not in validators` before it calls
    `get`, so the only divergence sits behind a membership test. **A double
    whose faithfulness depends on a caller's guard is faithful by luck**, and
    the guard is in the other package.

    Found by applying `agent`'s stand-in/instrument rule after their fourth
    instance of one bug: production calls `__contains__`, `names` and `get`, so
    those three must match contract; `add` is an instrument nothing here
    exercises.
    """

    kind = "validator"

    def __init__(self, specs: dict[str, dict]) -> None:
        self._specs = specs

    def add(self, name, spec, *, origin):  # pragma: no cover - an instrument
        self._specs[name] = spec

    def get(self, name):
        try:
            return self._specs[name]
        except KeyError:
            have = ", ".join(self.names()) or "nothing admitted"
            raise SpecNotFound(f"no {self.kind} named {name!r} (have: {have})") from None

    def names(self):
        return sorted(self._specs)

    def __contains__(self, name):
        return name in self._specs


def _kind(name: str, validators: list[str]) -> dict:
    return {
        "name": name,
        "content_type": "text",
        "scope": "fixed.required",
        "items_schema": {"type": "object"},
        "validators": validators,
    }


def test_binding_resolves_both_directions() -> None:
    """Criterion 9: a validator over two handoffs, and a handoff with three
    validators. Both directions are genuinely many."""
    kinds = HandoffSpecRegistry()
    kinds.add("trace", _kind("trace", ["shape", "usable", "trusted"]), origin="h/trace.jsonnet")
    kinds.add("config", _kind("config", ["shape", "pair"]), origin="h/config.jsonnet")

    validators = FakeValidatorRegistry(
        {
            "shape": {"inputs": ["trace", "config"]},
            "usable": {"inputs": ["trace"]},
            "trusted": {"inputs": ["trace"]},
            "pair": {"inputs": ["config", "trace"]},
        }
    )
    kinds.check_bindings(validators)

    assert kinds.validators_for("trace") == ["shape", "usable", "trusted"]
    # The reverse index, built at admission rather than searched per call.
    assert kinds.kinds_for("shape") == ["config", "trace"]
    assert kinds.kinds_for("usable") == ["trace"]
    assert kinds.kinds_for("nobody") == []


def test_conflict_names_both_sides_and_paths() -> None:
    """Criterion 10: a mismatch crashes at load, naming both sides. Nothing is
    silently resolved — a silently-resolved conflict means one of the two
    records is lying and nobody finds out which."""
    kinds = HandoffSpecRegistry()
    kinds.add("trace", _kind("trace", ["check_trace_shape"]), origin="handoff/trace.jsonnet")
    # A real kind, with a validator of its own: `add` enforces check 4, so a
    # fixture that names none is not admissible and never was.
    kinds.add(
        "trace_v2", _kind("trace_v2", ["check_trace_shape"]), origin="handoff/trace_v2.jsonnet"
    )
    validators = FakeValidatorRegistry({"check_trace_shape": {"inputs": ["trace_v2"]}})

    with pytest.raises(BindingConflict) as exc:
        kinds.check_bindings(validators)

    message = str(exc.value)
    assert "'trace'" in message and "'check_trace_shape'" in message  # both sides
    assert "handoff/trace.jsonnet" in message  # the origin
    assert "differing:" in message  # the specific element, not "they differ"
    assert "fix:" in message and "Both are valid repairs" in message
    assert "hint:" in message and "trace_v2" in message  # one was renamed
    assert exc.value.kind == "trace" and exc.value.validator == "check_trace_shape"


def test_a_conflict_is_catchable_as_spec_inconsistent() -> None:
    """It crosses the seam, so a caller knowing only `spec_loader`'s three
    error classes still catches it."""
    assert issubclass(BindingConflict, SpecInconsistent)


def test_an_unresolvable_validator_is_not_found_not_inconsistent() -> None:
    """JPMS separates "not found" from "found, but inconsistent", and the
    distinction is load-bearing: a missing validator is a typo, a mismatch
    means one of two records is lying."""
    kinds = HandoffSpecRegistry()
    kinds.add("trace", _kind("trace", ["absent"]), origin="handoff/trace.jsonnet")

    with pytest.raises(SpecNotFound) as exc:
        kinds.check_bindings(FakeValidatorRegistry({"present": {"inputs": ["trace"]}}))
    assert "present" in str(exc.value)  # the candidates are enumerated
    assert not isinstance(exc.value, BindingConflict)


def test_the_binding_field_is_inputs() -> None:
    """`ValidatorSpec` declares `inputs` and the model is `extra="forbid"`, so
    a spec carrying `binds_to` is rejected at admission. Two designs had two
    names for one field, and the agreement check read the one that cannot
    exist."""
    from handoff.registry import BINDS_KEY

    assert BINDS_KEY == "inputs"

    kinds = HandoffSpecRegistry()
    kinds.add("trace", _kind("trace", ["shape"]), origin="h.jsonnet")
    with pytest.raises(BindingConflict):
        kinds.check_bindings(FakeValidatorRegistry({"shape": {"binds_to": ["trace"]}}))


def test_the_double_refuses_an_unknown_name_the_way_the_real_registry_does() -> None:
    """The double is a stand-in for `spec_loader`'s registry, so its refusal
    must be the same *type*, not merely a refusal.

    Pinned rather than left to the docstring: the divergence it replaces was
    invisible because `check_bindings` guards with a membership test before it
    calls `get`, so nothing here would ever have reached it.
    """
    from spec_loader.registry import BaseSpecRegistry

    class RealValidatorRegistry(BaseSpecRegistry):
        # `kind` is a class attribute on every concrete registry —
        # `agent/registry.py:44`, `closure/registry.py:31`. Subclassing is how
        # the real `get` is reached, so this exercises production's code path
        # rather than a second copy of it.
        kind = "validator"

    for source in (FakeValidatorRegistry({"check_shape": {}}), RealValidatorRegistry()):
        with pytest.raises(SpecNotFound):
            source.get("nobody")
