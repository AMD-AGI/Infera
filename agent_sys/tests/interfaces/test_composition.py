"""The assembled system, driven rather than described — gate G3's two halves.

Every other suite in this repository proves one package's half of a seam. That
is what let four defects through in one day, each of the same shape: **a seam
failing by producing a plausible empty value instead of raising.** None of them
errored, each left a check reporting nothing, and a check that reports nothing
reads exactly like a check that found nothing.

Two of the four are caught here and are catchable nowhere else:

| defect | why no package suite saw it |
|---|---|
| `FilesystemStore(handoff_root)` built with no `KindSource` published a handoff missing four of its five required README sections, with `kind: ""` in the manifest — criteria 2 and 3 unenforced | all 137 `tests/handoff` tests inject a resolver |
| the composition root reached for `load_report`, the registry spelled it `report`, and `getattr(..., lambda: None)` turned the mismatch into `None` — `closure`'s escape-hatch check then skipped itself | `handoff`, `closure`, `spec_loader` and `task_graph` each asserted their own half returned the right value |

So the rule this file exists to enforce, `implementation-stage.md` §6.0:
**the assembled system must report something on a package built to fail**, and
must actually work on one built to succeed. Both halves, because a gate that
only checks the failure path passes when nothing runs at all.

`docs/interfaces.md` §4's import rule binds packages, not tests — §4.9 says so
explicitly — which is what lets one file import both sides of a seam.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from handoff import HandoffSpecRegistry
from handoff import content as content_mod
from spec_loader import YamlPackage, load_package
from task_graph.bootstrap import build_registry
from task_graph.ids import HandoffId, TaskId

# --------------------------------------------------------------------------- #
# One admissible handoff kind, as a task package would declare it.

#: **`module: handoff` claims the kind, and it is a key rather than a
#: directory.** Under the jsonnet loader this lived in `handoffs/` and
#: `DirectoryPackage` read the kind off its location.
KIND_SOURCE = """module: handoff
name: trace
description: a profiling trace, for a human
content_type: reproducible
scope: fixed.required
items_schema:
  type: object
  properties: {script: {}, result: {}, env: {}}
  required: [env]
  additionalProperties: false
validators: [shape]
"""

#: A kind naming no validator. Refused unless the escape-hatch flag is set —
#: `handoff` spec §5.3, criterion 12.
LOOSE_SOURCE = KIND_SOURCE.replace("validators: [shape]", "validators: []").replace(
    "name: trace", "name: loose"
)

README = (
    "# Trace\n\n"
    "## Purpose\n\nWhy this run exists.\n\n"
    "## How to run\n\nsh items/script\n\n"
    "## Result\n\n42 ms.\n\n"
    "## Environment\n\nMI300X, ROCm 7.0.\n\n"
    "## Watch out\n\nThe first iteration is cold.\n"
)


def _content(root: Path, *, readme: str = README) -> Path:
    """A `content/` directory a `reproducible` kind accepts."""
    (root / "items").mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text(readme, encoding="utf-8")
    (root / "items" / "script").write_text("echo hi\n", encoding="utf-8")
    (root / "items" / "result").write_text("42\n", encoding="utf-8")
    content_mod.write_items_json(root / "items", {"env": {"gpu": "MI300X"}})
    return root


def _shell(pkg: Path) -> Path:
    """The two names every package must carry, and nothing else.

    Split out because three tests below write their own documents and only need
    a package to put them in. `assets/` is empty on purpose: these packages
    declare handoff kinds, which name no body, so there is nothing for the
    convention to find.
    """
    (pkg / "assets").mkdir(parents=True, exist_ok=True)
    return pkg


def _package(root: Path, name: str, source: str) -> YamlPackage:
    """A package holding one handoff kind and no graph — a **library**.

    No `main.yaml`. Criterion 18 (rev. 11) makes its absence the statement that
    this package is not a run's entry, and `assets/` alone is what makes it a
    package. Several tests below build **two** of these, which is the case that
    forced the split: demanding an entry file of each would answer "where does a
    run start" twice and therefore not at all.
    """
    pkg = root / name
    (pkg / "assets").mkdir(parents=True, exist_ok=True)
    (pkg / "kinds.yaml").write_text(source, encoding="utf-8")
    return YamlPackage(root=pkg)


class _Views:
    """`Registries` over a real registry and four dicts.

    `load_package` takes a `Registries`, and only `handoff_specs` is exercised
    here. A Protocol rather than a class is what makes this possible without a
    second composition root.
    """

    def __init__(self, registry) -> None:  # noqa: ANN001 - a task_graph Registry
        self.handoff_specs = registry.get("handoff_specs")
        self.validator_specs: dict = {}
        self.task_specs: dict = {}
        self.agent_specs: dict = {}
        self.closures: dict = {}

    def for_kind(self, kind: str):
        return {"handoff": self.handoff_specs}[kind]


class _EmptyRegistry:
    """A `SpecRegistry` holding nothing.

    Four of the five slots need one of these rather than a `{}`. A dict is not
    a `SpecRegistry`: the root walks `agent_specs.names()` to pre-register
    agents, and `{}` has no `names`. That is the second time in this file a
    plausible stand-in has been caught by the root doing something real with
    it — and both times it raised on the spot, because a missing *method* is a
    shape and the runtime checks shapes for free.
    """

    kind = "empty"

    def add(self, name, spec, *, origin):  # pragma: no cover - nothing is admitted
        raise AssertionError("this registry admits nothing")

    def get(self, name):
        raise KeyError(name)

    def names(self) -> list[str]:
        return []

    def __contains__(self, name: object) -> bool:
        return False


class _Five:
    """The five spec registries, as `build_registry(registries=...)` takes them.

    An object with the five attributes rather than a dict: the root does
    `getattr(registries, key)`, so a dict raises there — measured, and loudly,
    which is the good kind of guess to make wrong.
    """

    def __init__(self, handoff_specs) -> None:  # noqa: ANN001
        self.handoff_specs = handoff_specs
        self.validator_specs = _EmptyRegistry()
        self.task_specs = _EmptyRegistry()
        self.agent_specs = _EmptyRegistry()
        self.closures = _EmptyRegistry()


@pytest.fixture
def system(tmp_path: Path):
    """A real registry, with real roots. Nothing here is a stub."""
    return build_registry(
        handoff_root=str(tmp_path / "handoffs"),
        knowledge_root=str(tmp_path / "knowledge"),
    )


# --------------------------------------------------------------------------- #
# The positive half


def test_a_declared_typed_handoff_publishes_and_round_trips(system, tmp_path: Path) -> None:
    """`put` → `copy_out` → manifest, through the assembled root.

    The store is reached by the name the composition root registers it under,
    the kind by the name a task package declared it under, and the slot by the
    id `declare(types=...)` typed — three lookups that each worked in isolation
    long before they worked together.
    """
    load_package(_package(tmp_path / "pkg", "trace", KIND_SOURCE), _Views(system))

    hid, producer = HandoffId.new(), TaskId.new()
    system.get("handoff_mgr").declare([hid], producer, types={hid: "trace"})
    assert system.get("handoff_mgr").type_of(hid) == "trace"

    store = system.get("handoff_store")
    version = store.put(hid, _content(tmp_path / "produced"), producer=producer)

    manifest = store.get_manifest(hid, version)
    assert manifest.kind == "trace", "the manifest records a real kind, not ''"
    assert manifest.algorithm == "agent_sys.handoff.tree.v1"
    assert manifest.producer == producer

    # copy_out verifies the digest on the way out; a mismatch would raise here.
    content = store.copy_out(hid, version, tmp_path / "playground")
    assert sorted(content.items) == ["env", "result", "script"]
    assert store.read_verdicts(hid, version) == [], "created empty, not absent"


def test_the_two_stores_are_two_roots_and_one_implementation(system, tmp_path: Path) -> None:
    """`handoff` spec §6.2. Both are wired, and an artefact lands in exactly one."""
    load_package(_package(tmp_path / "pkg", "trace", KIND_SOURCE), _Views(system))

    hid, producer = HandoffId.new(), TaskId.new()
    system.get("handoff_mgr").declare([hid], producer, types={hid: "trace"})
    knowledge = system.get("knowledge_store")
    knowledge.put(hid, _content(tmp_path / "produced"), producer=producer)

    assert knowledge.exists(hid)
    assert not system.get("handoff_store").exists(hid)
    assert type(knowledge) is type(system.get("handoff_store"))


# --------------------------------------------------------------------------- #
# The negative half — `implementation-stage.md` §6.0


def test_the_assembled_system_refuses_a_malformed_handoff(system, tmp_path: Path) -> None:
    """**The regression test for the defect this file exists for.**

    Before `e81a81c` the store as the root builds it had no `KindSource`, so
    `put` had no content type to take required sections from and no
    `items_schema` to check against — and published this exact directory as v0
    with `kind: ""`. Criteria 2 and 3, unenforced on the assembled path, with
    every package suite green.
    """
    from handoff.errors import Malformed

    load_package(_package(tmp_path / "pkg", "trace", KIND_SOURCE), _Views(system))
    hid, producer = HandoffId.new(), TaskId.new()
    system.get("handoff_mgr").declare([hid], producer, types={hid: "trace"})

    bad = _content(tmp_path / "bad", readme="# Trace\n\n## Purpose\n\nOnly one section.\n")
    with pytest.raises(Malformed, match="How to run"):
        system.get("handoff_store").put(hid, bad, producer=producer)

    assert system.get("handoff_store").list_versions(hid) == [], "and nothing was created"


def test_an_untyped_slot_cannot_publish_and_says_why(system, tmp_path: Path) -> None:
    """The other direction: the kind resolves through `type_of`, so a slot
    nobody typed is a slot nobody can publish to. `None` here is a real answer
    the store acts on, not a default standing in for one."""
    from handoff.errors import Malformed

    load_package(_package(tmp_path / "pkg", "trace", KIND_SOURCE), _Views(system))
    hid, producer = HandoffId.new(), TaskId.new()
    system.get("handoff_mgr").declare([hid], producer)  # no types=

    assert system.get("handoff_mgr").type_of(hid) == ""
    with pytest.raises(Malformed, match="has no kind for it"):
        system.get("handoff_store").put(hid, _content(tmp_path / "c"), producer=producer)


def test_a_package_built_to_fail_is_reported_by_the_assembled_system(
    system, tmp_path: Path
) -> None:
    """§6.0 as written: the system reports **something**.

    Two failures in one package, and neither hides the other — one broken spec
    must not conceal the other nine.
    """
    broken = _shell(tmp_path / "broken")
    # Two documents in two files, which is what makes the "one does not hide the
    # other" assertion below mean something. The scan reaches every YAML under
    # the root, so where they sit is the package's business, not the loader's.
    (broken / "kinds.yaml").write_text(LOOSE_SOURCE, encoding="utf-8")
    (broken / "bad_schema.yaml").write_text(
        KIND_SOURCE.replace("name: trace", "name: bad_schema").replace(
            "  type: object", "  type: nonsense"
        ),
        encoding="utf-8",
    )

    report = load_package(YamlPackage(root=broken), _Views(system))

    assert list(report.admitted) == [], "neither kind is admissible"
    assert len(report.problems) == 2, "both are reported; one does not hide the other"
    messages = " ".join(p.message for p in report.problems)
    assert "names no validator" in messages
    assert "not a valid schema" in messages
    assert all(p.fatal for p in report.problems)


def test_the_accessor_the_root_calls_yields_a_report(system, tmp_path: Path) -> None:
    """**The second defect this file exists for**, and the one no package could see.

    `task_graph/bootstrap.py` reaches for `load_report` on the handoff spec
    registry. It once spelled that method `report`, and a `getattr` default
    turned the mismatch into `None` — after which `closure`'s check 3 returned
    early and an escape-hatch admission went unreported by the assembled system,
    with four package suites green.

    So this asserts the **composition root's own expression** against the
    registry the root actually built, rather than a method name on a class in
    isolation. That distinction is the whole defect: both halves were correct
    on their own.
    """
    load_package(_package(tmp_path / "pkg", "trace", KIND_SOURCE), _Views(system))
    specs = system.get("handoff_specs")

    report = getattr(specs, "load_report", lambda: None)()
    assert report is not None, (
        "the composition root's own expression must yield a report; `None` here "
        "is what silently disabled closure's escape-hatch check"
    )
    assert list(report.admitted) == ["trace"]
    assert list(report.without_validator) == [], "nothing used the hatch"
    assert all(type(n) is str for n in report.admitted)


def test_the_default_root_admits_no_kind_without_a_validator(system, tmp_path: Path) -> None:
    """Criterion 12, first half, through the assembled system: **off by default.**

    "Checkable by construction" is spec §2's third principle, and the flag not
    being set is what makes it hold for anyone who does not ask otherwise.
    """
    pkg = _shell(tmp_path / "pkg")
    (pkg / "kinds.yaml").write_text(LOOSE_SOURCE, encoding="utf-8")

    report = load_package(YamlPackage(root=pkg), _Views(system))
    assert list(report.admitted) == []
    assert "names no validator" in report.problems[0].message
    assert system.get("handoff_specs").load_report().without_validator == []


def test_the_escape_hatch_flag_reaches_the_root_and_reports(tmp_path: Path) -> None:
    """Criterion 12, second half: the flag **admits and reports**, from the root.

    This test replaces one that asserted the opposite. When it was written
    `bootstrap.py` built the five registries itself as `parts[name]()`, so
    nothing in the assembled system could turn the flag on and criterion 12's
    second half was reachable only from a direct construction — a capability
    with no route, the family `put`-with-no-caller belongs to.

    `build_registry(*, registries=...)` closed it: the caller constructs the
    five, so a CLI can pass `HandoffSpecRegistry(allow_no_validator=cfg.loose)`
    and no flag needs a parameter of its own.

    **The marker test is what found that the gap had closed**, and only because
    its second assertion checked for the route rather than for the default
    being strict — under `registries=`, the default path still builds a strict
    registry, so a test asserting only that would have gone on passing and gone
    on reporting a gap that no longer existed.
    """
    permissive = HandoffSpecRegistry(allow_no_validator=True)
    system = build_registry(
        handoff_root=str(tmp_path / "handoffs"),
        knowledge_root=str(tmp_path / "knowledge"),
        registries=_Five(permissive),
    )
    assert system.get("handoff_specs") is permissive, "the root registered what it was given"

    pkg = _shell(tmp_path / "pkg")
    (pkg / "kinds.yaml").write_text(LOOSE_SOURCE, encoding="utf-8")
    report = load_package(YamlPackage(root=pkg), _Views(system))

    assert list(report.admitted) == ["loose"], "the flag admits it"
    # ...and the root's own expression is what carries the name onward.
    named = getattr(system.get("handoff_specs"), "load_report", lambda: None)()
    assert list(named.without_validator) == ["loose"], "and reports it by name"


# --------------------------------------------------------------------------- #
# Stubs, driven against the real thing — `docs/interfaces.md` §8.7


def test_check_bindings_agrees_with_a_real_validator_registry() -> None:
    """`handoff`'s binding stub, checked against `ValidatorSpecRegistry` itself.

    §8.7 names two failures a conformance guard must survive, and only one is
    drift: **a stub can be wrong on the day it is written**, matching nothing,
    which drift-detection structurally cannot see.

    `tests/handoff/test_binding.py` drives `check_bindings` against a
    `FakeValidatorRegistry` whose `get()` returns a plain dict. Everything there
    would pass if the real registry returned a **model** instead — `.get()`
    would not exist, `check_bindings` would raise `AttributeError` in the
    assembled system, and criterion 10 would be dead while green.

    So this drives both real registries. Verified rather than reasoned:
    `ValidatorSpecRegistry` inherits `BaseSpecRegistry.get`, which returns the
    raw admitted mapping, and the admitted document keeps `inputs` — which is
    also why `BINDS_KEY` is `inputs` and not the `binds_to` two designs once
    named.
    """
    from handoff.errors import BindingConflict
    from validator.registry import ValidatorSpecRegistry

    validator_spec = {
        "name": "shape",
        "brief": "checks the trace's shape",
        "inputs": ["trace"],
        "dimension": "completeness",
        "strength": "strong",
        "tags": {"logic_source": "external_static", "cost": "seconds"},
        "body": {"readme": "readme.md"},
    }
    kind_spec = {
        "name": "trace",
        "description": "d",
        "content_type": "text",
        "scope": "fixed.required",
        "items_schema": {"type": "object"},
        "validators": ["shape"],
    }

    validators = ValidatorSpecRegistry()
    validators.add("shape", validator_spec, origin="v/shape.jsonnet")
    assert validators.get("shape").get("inputs") == ["trace"], (
        "the stub in tests/handoff models `get()` as returning a mapping with "
        "an `inputs` key; if this fails, that stub was never right"
    )

    agreeing = HandoffSpecRegistry()
    agreeing.add("trace", kind_spec, origin="h/trace.jsonnet")
    agreeing.check_bindings(validators)  # agrees: no raise

    disagreeing = HandoffSpecRegistry()
    disagreeing.add("other", {**kind_spec, "name": "other"}, origin="h/other.jsonnet")
    with pytest.raises(BindingConflict) as exc:
        disagreeing.check_bindings(validators)
    assert "'other'" in str(exc.value) and "'shape'" in str(exc.value)


def test_an_allocated_version_does_not_look_delivered_to_the_gate(tmp_path: Path) -> None:
    """§4.14's own unmeasured question, pinned against the real reader.

    The ruling named as *not yet measured* whether a pre-allocated empty
    `v<N>` pollutes any other reader. **It does**, and the reader is
    `agent/gate.py`. Measured before the seal marker existed:

    ```
    nothing at all           -> OUTPUT_ABSENT "declared output … never delivered"
    a pre-allocated empty v0 -> RAISED FileNotFoundError: …/v0/manifest.yaml
       store.exists -> True      store.list_versions -> [0]
    ```

    So naive pre-allocation destroyed **criterion 5's** distinction between
    *refused* and *never attempted*, and destroyed it as an uncaught `OSError`
    rather than as a verdict — the worst way for a completeness gate to fail,
    because the gate exists to turn exactly that situation into a report.

    `manifest.yaml` as the seal marker is what fixes it, and it fixes it
    without `agent` learning that allocation exists. **This lives here rather
    than in `tests/handoff/` because neither package can assert it alone**: the
    store's own conformance suite proves an allocated version is invisible to
    `list_versions`, and `tests/agent` would have to build a real store to see
    the gate's side. The defect was in the join.
    """
    from agent.gate import _one_output
    from handoff import FilesystemStore, version_dir

    store = FilesystemStore(tmp_path / "s")
    hid = HandoffId.new()

    absent = _one_output(hid, store)
    assert [f.kind.name for f in absent] == ["OUTPUT_ABSENT"], "nothing at all"

    allocated = store.allocate(hid)
    assert version_dir(tmp_path / "s", hid, allocated).is_dir(), (
        "the directory must exist — env_mgr's Landlock layer opens every granted "
        "path and a reservation-only number is fatal before the body starts"
    )

    after = _one_output(hid, store)
    assert [f.kind.name for f in after] == ["OUTPUT_ABSENT"], (
        "an allocated-but-unsealed version must still read as never delivered; "
        "before the seal marker this raised FileNotFoundError out of the gate"
    )
    assert not store.exists(hid) and store.list_versions(hid) == []
    assert store.latest(hid) is None
