"""`BaseSpecRegistry` — the collision policy the four registries share.

`docs/design.md` §5.2 and §5.3. This is the base four other packages subclass, so
its policy is four modules' policy and a change here is a change to all of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from spec_loader import (
    BaseSpecRegistry,
    SpecInconsistent,
    SpecInvalid,
    SpecNotFound,
    load_package,
)

TRACE = {"name": "trace", "content_type": "reproducible"}


class Kinds(BaseSpecRegistry):
    kind = "handoff kind"


@pytest.fixture
def registry() -> Kinds:
    return Kinds()


def test_a_duplicate_name_is_an_error(registry: Kinds) -> None:
    """Error by default, not overwrite.

    `fsspec`'s shape. The alternative is on record as a mistake twice over:
    Great Expectations logs "Overwriting declaration" and proceeds, and Inspect
    AI assigns with no check at all. Two specs claiming one name is a fault.
    """
    registry.add("trace", TRACE, origin="a/trace.jsonnet")

    with pytest.raises(SpecInconsistent) as caught:
        registry.add("trace", {**TRACE, "content_type": "text"}, origin="b/trace.jsonnet")

    message = str(caught.value)
    assert "a/trace.jsonnet" in message and "b/trace.jsonnet" in message, (
        "a collision report names both sides — docs/design.md §6.2, and OPA#7806 "
        "is a user who still could not tell which side was wrong without them"
    )


def test_an_identical_re_registration_is_a_no_op(registry: Kinds) -> None:
    """So loading one package twice is harmless and loading two that disagree is not."""
    registry.add("trace", TRACE, origin="a/trace.jsonnet")
    registry.add("trace", dict(reversed(list(TRACE.items()))), origin="b/trace.jsonnet")

    assert registry.names() == ["trace"]
    assert registry.origin_of("trace") == "a/trace.jsonnet"


def test_the_reverse_collision_is_rejected(registry: Kinds) -> None:
    """One spec under two names.

    `pluggy` rejects this and the reason transfers exactly: for them it silently
    doubles hook invocations, and here one validator under two names would run
    twice and record two verdicts against one handoff version.
    """
    registry.add("trace", TRACE, origin="a/trace.jsonnet")

    with pytest.raises(SpecInconsistent) as caught:
        registry.add("trace_v2", TRACE, origin="b/trace_v2.jsonnet")

    assert "byte-identical" in str(caught.value)
    assert registry.names() == ["trace"]


def test_not_found_enumerates_the_candidates(registry: Kinds) -> None:
    """pytest sets the bar — `fixture 'x' not found` plus every available one —
    and it is already a convention here: `env_mgr/registry.py` raises
    `unknown installer {name!r} (have {sorted(REGISTRY)})`."""
    registry.add("trace", TRACE, origin="a")
    registry.add("deploy_config", {"name": "deploy_config"}, origin="b")

    with pytest.raises(SpecNotFound) as caught:
        registry.get("trace_getter")

    message = str(caught.value)
    assert "handoff kind" in message
    assert "trace_getter" in message
    assert "deploy_config, trace" in message


def test_not_found_on_an_empty_registry_says_so(registry: Kinds) -> None:
    with pytest.raises(SpecNotFound, match="nothing admitted"):
        registry.get("trace")


def test_a_subclass_check_runs_before_anything_is_stored() -> None:
    """The override point exists so a subclass cannot get the ordering wrong.

    Each module's spec lists its own load-time checks — `handoff` §8 has five,
    `validator` §9.3 has five — and none of them is in the base, because none of
    them is shared. What *is* shared is that a rejected spec leaves no trace.
    """

    class Strict(BaseSpecRegistry):
        kind = "handoff kind"

        def _validate(self, name: str, spec: Mapping[str, Any], *, origin: str) -> None:
            if not spec.get("validators"):
                raise SpecInvalid(f"{origin}: handoff kind {name!r} names no validator")

    registry = Strict()

    with pytest.raises(SpecInvalid, match="names no validator"):
        registry.add("trace", TRACE, origin="a/trace.jsonnet")

    assert registry.names() == []
    assert "trace" not in registry


def test_load_package_collects_a_registry_rejection_rather_than_raising(tmp_path) -> None:
    """A kind's own check failing is a `Problem`, like every other load fault.

    Same reason as a schema violation: one broken spec must not hide the other
    nine. The composition root raises once, over everything.
    """

    class Strict(BaseSpecRegistry):
        kind = "handoff kind"

        def _validate(self, name: str, spec: Mapping[str, Any], *, origin: str) -> None:
            raise SpecInvalid(f"{origin}: {name!r} refused by the kind's own check")

    result = load_package(_package(tmp_path, {"trace": "trace"}), _Registries(Strict()))

    assert result.admitted == ()
    (problem,) = result.problems
    assert problem.keyword == "invalid"
    assert "refused by the kind's own check" in problem.message


def test_the_registry_is_the_opposite_of_the_component_registry() -> None:
    """Stated as a test because the instinct on seeing two registries is to unify them.

    `task_graph.Registry.register` overwrites **on purpose** — spec §4.1 requires
    that a test can swap an implementation after wiring. `SpecRegistry` errors on
    purpose. The collision policies are irreconcilable, which is why they are two
    classes (`docs/design.md` §5.3).

    The general form, and it is evidence rather than assertion: every canonical
    "one generic registry" turns out to be N typed containers with N policies.
    Kubernetes' `runtime.Scheme` is one struct holding seven separately-typed
    maps with three collision policies — panic, silent last-wins, and
    unconditional overwrite.
    """
    from task_graph.registry import Registry

    component = Registry()
    component.register("runner", "first")
    component.register("runner", "second")
    assert component.get("runner") == "second"

    specs = Kinds()
    specs.add("trace", TRACE, origin="a")
    with pytest.raises(SpecInconsistent):
        specs.add("trace", {"name": "trace"}, origin="b")


def test_load_package_collects_a_registry_raising_its_own_error_type(tmp_path) -> None:
    """The contract must not depend on four packages remembering it.

    `_validate`'s docstring says a subclass raises `SpecInvalid` or
    `SpecInconsistent`. Measured against the four real registries: `handoff` and
    `agent` do, and `validator` raises `ValidatorInvalid` — a `ValueError`, but
    neither of them. It escaped `load_package` entirely, so **one package's
    choice of exception type aborted the whole multi-package load**: "collect,
    do not raise" became "die on the first", silently, for every other package
    in the run.

    Found by driving the real `RegistryViews` and the four real registry
    subclasses instead of this file's doubles —
    `scratch/impl-2026-08/spec_loader/probe_real_registries.py`. Nothing in
    `tests/spec_loader` could have caught it, because every registry here is a
    `BaseSpecRegistry` with no `_validate` of its own.
    """

    class OwnErrorType(ValueError):
        """What a module invents when it does not subclass ours."""

    class Strict(BaseSpecRegistry):
        kind = "handoff kind"

        def _validate(self, name: str, spec: Mapping[str, Any], *, origin: str) -> None:
            raise OwnErrorType(f"{origin}: {name!r} refused with a type spec_loader never named")

    package = _package(tmp_path, {"keeps_going": "keeps_going", "refused": "refused"})
    result = load_package(package, _Registries(Strict()))

    assert len(result.problems) == 2, "both were refused; neither may be hidden by the other"
    assert {p.keyword for p in result.problems} == {"invalid"}
    assert all("never named" in p.message for p in result.problems)


def test_a_registry_bug_is_not_collected_as_a_rejection(tmp_path) -> None:
    """`ValueError` and not `Exception`, so the widening stays a repair.

    A rejection is a statement about the spec; an `AttributeError` is a bug in
    the registry. Collecting the second as a `Problem` would report a broken
    package where the fault is broken code — the silent-empty family again, one
    level up, and the reason this catch is not simply `except Exception`.
    """

    class Broken(BaseSpecRegistry):
        kind = "handoff kind"

        def _validate(self, name: str, spec: Mapping[str, Any], *, origin: str) -> None:
            raise AttributeError("a genuine bug in the registry")

    with pytest.raises(AttributeError, match="genuine bug"):
        load_package(_package(tmp_path, {"trace": "trace"}), _Registries(Broken()))


def test_the_protocol_declares_what_a_whole_catalogue_pass_calls_unguarded() -> None:
    """`origin_of` is on `SpecRegistry`, not only on `BaseSpecRegistry`.

    The gap was found from the other side. `closure.check_closures` needs a
    spec's file path for `docs/design.md` §6.2's *name both sides* rule, so it
    called `origin_of` — and, seeing it absent from the Protocol, guarded it
    with a `getattr` fallback to the spec's **name**. When `build_registry`
    began taking `registries=` from a caller, the fallback became reachable and
    labelled every `Problem` with a name where a path belongs: indistinguishable
    from a real origin in a message, and nothing raises.

    They removed the guard and declared it an obligation. This is that
    obligation in the contract, so the next implementer of `SpecRegistry` is
    told rather than left to infer it from a base class they need not subclass.

    The general rule, which is why this test is here rather than in `closure`:
    **a method a whole-catalogue pass calls unguarded belongs on the Protocol,
    not only on the base.** A `Registries` satisfying the declared Protocol and
    not subclassing `BaseSpecRegistry` is legal, and the docstring on
    `Registries` invites exactly that — *"a test supplies five dicts"*.
    """
    from spec_loader.protocols import SpecRegistry

    # `kind` is an annotation rather than an assignment, so it is in
    # `__annotations__` and not in `vars()`. Both halves are the declaration.
    methods = {n for n in vars(SpecRegistry) if not n.startswith("_")} | {"__contains__"}
    attributes = set(SpecRegistry.__annotations__)
    assert "origin_of" in methods
    assert attributes == {"kind"}

    unprovided = sorted(n for n in methods if not callable(getattr(BaseSpecRegistry, n, None)))
    assert unprovided == [], (
        f"the shared base does not provide {unprovided}, which the Protocol promises"
    )
    assert isinstance(BaseSpecRegistry.kind, str)


# --------------------------------------------------------------------------- #
# Helpers for the three `load_package` tests above.
#
# They exercise what a registry does when `load_package` admits into it, so the
# package they are handed only has to be real enough to produce documents. It is
# a real `YamlPackage` rather than a double, because the thing under test is the
# *boundary* between the two.


def _package(root, kinds: Mapping[str, str]):
    """A minimal well-formed package declaring one handoff kind per entry.

    **No `main.yaml`, and that is the point rather than an omission.** Main spec
    criterion 18 (rev. 11): a package with none is a *library*, its documents are
    admitted, and its absence of a graph is a statement rather than a fault. A
    bundle of handoff kinds is exactly that shape, so writing one here would be
    inventing a graph to satisfy a rule that no longer exists.
    """
    from spec_loader import YamlPackage

    (root / "assets").mkdir(parents=True, exist_ok=True)
    (root / "kinds.yaml").write_text(
        "".join(
            f"- module: handoff\n  name: {name}\n  description: d\n"
            f"  content_type: text\n  scope: fixed.required\n"
            for name in kinds
        )
    )
    return YamlPackage(root=root)


class _Registries:
    """Five registries, with one of them supplied by the test.

    `for_kind` sends `handoff` to the supplied one and everything else to an
    inert `Kinds()`, so the test's registry sees only what it is about.
    """

    def __init__(self, handoff_specs: BaseSpecRegistry) -> None:
        self.handoff_specs = handoff_specs
        self.validator_specs = Kinds()
        self.task_specs = Kinds()
        self.agent_specs = Kinds()
        self.closures = Kinds()

    def for_kind(self, kind: str) -> BaseSpecRegistry:
        return self.handoff_specs if kind == "handoff" else self.closures
