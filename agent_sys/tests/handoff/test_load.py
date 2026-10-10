"""The registry through the **real** loader, across two packages.

Everything else in this directory tests `HandoffSpecRegistry` by calling it.
This file goes through `spec_loader.load_package` — jsonnet rendered, schema
validated, `_validate` hook fired, admitted — because three of the claims in
`handoff` design §3.5 and §8.5 are about *where in the pipeline* a check runs,
and a direct call cannot tell a correct ordering from a lucky one.

It is also the evidence for one open question: `docs/interfaces.md` §2 calls
`merged(reports)` to fold `HandoffLoadReport`s before `check_closures`, and no
§4 row assigns that function to anybody. `test_the_registry_is_already_the_merge`
is why it needs no owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from handoff import HandoffSpecRegistry
from handoff import kind as kind_mod
from spec_loader import YamlPackage, load_package, schema_for
from spec_loader.protocols import SpecRegistry

#: A minimal admissible kind. `description` is the schema's requirement and not
#: a field of `HandoffKind`, which is exactly the split `REQUIRED_KEYS` records.
#:
#: **`module: handoff` is how the kind is claimed now, and it is a key rather
#: than a directory.** Under the jsonnet loader this file lived in `handoffs/`
#: and `DirectoryPackage` read the kind off its location; a YAML document says
#: what it is (main spec §4.4).
SOURCE = """module: handoff
name: %(name)s
description: a kind, for a human
content_type: text
scope: %(scope)s
items_schema: {type: object}
validators: %(validators)s
"""


@dataclass
class Views:
    """`Registries` satisfied with one real registry and four dicts.

    A Protocol rather than a class is what makes this possible, and it is why
    this package's tests need no composition root.
    """

    handoff_specs: SpecRegistry
    validator_specs: SpecRegistry = field(default_factory=dict)
    task_specs: SpecRegistry = field(default_factory=dict)
    agent_specs: SpecRegistry = field(default_factory=dict)
    closures: SpecRegistry = field(default_factory=dict)

    def for_kind(self, kind: str) -> SpecRegistry:
        return {"handoff": self.handoff_specs}[kind]


def _package(root: Path, name: str, *, scope: str = "fixed.required", validators: str) -> Path:
    """A package holding one handoff kind and no graph — a **library**.

    No `main.yaml`, and that is the point rather than an omission. Main spec
    criterion 18 (rev. 11) split what criterion 16 had fused: `assets/` is about
    *being a package* and is required of every one, while `main.yaml` is about
    *being a run's entry*, which is one per run and not one per package. A
    package without one is a library and loads.

    This fixture briefly wrote its kind INTO `main.yaml`, under the earlier
    unconditional rule. That is now the one shape criterion 18 rejects — a file
    whose whole definition is "the outermost graph's entry" cannot be an entry to
    nothing — so the workaround is deleted rather than replaced, and the fixture
    is the case the criterion was split to permit.
    """
    pkg = root / name
    (pkg / "assets").mkdir(parents=True, exist_ok=True)
    (pkg / "kinds.yaml").write_text(
        SOURCE % {"name": name, "scope": scope, "validators": validators}, encoding="utf-8"
    )
    return pkg


def _load(reg: HandoffSpecRegistry, *roots: Path):
    views = Views(handoff_specs=reg)
    return [load_package(YamlPackage(root=r), views) for r in roots]


def test_a_kind_survives_render_schema_and_the_validate_hook(tmp_path: Path) -> None:
    reg = HandoffSpecRegistry()
    (report,) = _load(reg, _package(tmp_path, "trace", validators='["shape"]'))

    assert list(report.admitted) == ["trace"]
    assert list(report.problems) == []
    assert reg.kind_of("trace").content_type == "text"
    assert reg.validators_for("trace") == ["shape"]


def test_the_escape_hatch_end_to_end(tmp_path: Path) -> None:
    """Criterion 12, through the pipeline rather than beside it.

    Off by default the kind is **refused** and the loader carries the problem;
    with the flag it is **admitted and named**. Both halves have to be true of
    the same object, which is what a direct call to `check()` cannot show.
    """
    loose = _package(tmp_path, "loose", validators="[]")

    strict = HandoffSpecRegistry()
    (refused,) = _load(strict, loose)
    assert list(refused.admitted) == []
    assert "names no validator" in refused.problems[0].message
    assert "loose" not in strict

    permissive = HandoffSpecRegistry(allow_no_validator=True)
    (admitted,) = _load(permissive, loose)
    assert list(admitted.admitted) == ["loose"]
    assert list(permissive.load_report().without_validator) == ["loose"]


def test_the_registry_is_already_the_merge(tmp_path: Path) -> None:
    """**`merged(reports)` needs no owner, because the merge is structural.**

    `interfaces.md` §2 folds N `LoadReport`s into the `HandoffLoadReport` that
    `check_closures` takes, and no §4 row owns the folding. It does not have to:
    one `HandoffSpecRegistry` receives *every* package, so `load_report()` is
    the whole-catalogue answer by construction and a fold would be a second writer
    of a fact the registry already holds.

    Asserted over two packages, one of which uses the escape hatch.
    """
    a = _package(tmp_path, "trace", validators='["shape"]')
    b = _package(tmp_path, "loose", validators="[]")

    reg = HandoffSpecRegistry(allow_no_validator=True)
    reports = _load(reg, a, b)

    assert [list(r.admitted) for r in reports] == [["trace"], ["loose"]]
    merged = reg.load_report()
    assert list(merged.admitted) == ["loose", "trace"]  # both packages, sorted
    assert list(merged.without_validator) == ["loose"]


def test_a_second_package_may_not_redefine_a_kind(tmp_path: Path) -> None:
    """The collision policy is the base's, and it survives the pipeline: the
    loader collects the `SpecInconsistent` as a problem rather than dying, so
    one clashing package does not hide the rest."""
    first = _package(tmp_path / "one", "trace", validators='["shape"]')
    second = _package(tmp_path / "two", "trace", scope="addons.temp", validators='["shape"]')

    reg = HandoffSpecRegistry()
    ok, clash = _load(reg, first, second)

    assert list(ok.admitted) == ["trace"]
    assert list(clash.admitted) == []
    assert "already held by a different spec" in clash.problems[0].message
    assert reg.can_satisfy_required("trace"), "the first definition stands"


def test_an_identical_kind_in_two_packages_is_a_no_op_not_a_clash(tmp_path: Path) -> None:
    """Main spec §4.3 makes cross-package references a supported case, so the
    same kind vendored twice must load. And it must not index twice — the
    reverse index is appended to in `add`, which is why that append is guarded
    by the base's no-op return rather than living in `_validate`."""
    first = _package(tmp_path / "one", "trace", validators='["shape"]')
    second = _package(tmp_path / "two", "trace", validators='["shape"]')

    reg = HandoffSpecRegistry()
    for report in _load(reg, first, second):
        assert list(report.problems) == []

    assert reg.names() == ["trace"]
    assert reg.kinds_for("shape") == ["trace"], "indexed once, not twice"


def test_the_accessor_is_named_what_the_composition_root_reaches_for() -> None:
    """`task_graph/bootstrap.py` does

        getattr(r.get("handoff_specs"), "load_report", lambda: None)()

    and this registry once spelled that method `report`. The mismatch did not
    fail — the `getattr` default turned it into `None`, `closure`'s check 3
    returned early on `None`, and an escape-hatch admission went unreported in
    the assembled system while every test in both packages passed.

    So the name is pinned from this side. The string in `bootstrap.py` is
    `task_graph`'s and pinning it is theirs.
    """
    reg = HandoffSpecRegistry()
    assert getattr(reg, "load_report", None) is not None, (
        "the composition root reaches for `load_report`; renaming it here "
        "silently disables closure's escape-hatch check"
    )


def test_the_report_is_always_present_and_never_none(tmp_path: Path) -> None:
    """ "No escape-hatch admissions" and "nobody told me" are different facts.

    An empty `without_validator` states the first. `load_report()` returns a
    report on an empty registry, so the second is not representable from this
    side and a caller never has to treat `None` as if it meant "none".
    """
    empty = HandoffSpecRegistry().load_report()
    assert list(empty.admitted) == [] and list(empty.without_validator) == []

    reg = HandoffSpecRegistry()
    _load(reg, _package(tmp_path, "trace", validators='["shape"]'))
    clean = reg.load_report()
    assert list(clean.admitted) == ["trace"]
    assert list(clean.without_validator) == [], "admitted normally, hatch unused"


def test_the_report_carries_bare_kind_names(tmp_path: Path) -> None:
    """`closure`'s check 3 is a set intersection against `named_kinds(task)`, so
    an element that is anything but a bare kind-name string makes that check
    match nothing — and a check that silently reports nothing is the worst
    failure available to it, because criterion 6 requires an escape-hatch
    closure to load *and report that it did*."""
    reg = HandoffSpecRegistry(allow_no_validator=True)
    _load(reg, _package(tmp_path, "loose", validators="[]"))
    report = reg.load_report()

    assert list(report.without_validator) == ["loose"]
    assert all(type(n) is str for n in report.without_validator)
    assert all(type(n) is str for n in report.admitted)
    assert list(report.without_validator) == sorted(report.without_validator)


def test_the_schema_and_the_constructor_agree_on_required_keys() -> None:
    """One writer per invariant, read through `schema_for` rather than by
    hand-rolling an `importlib.resources` read — which is the failure main
    design §2.2 records, reintroduced once per module."""
    required = set(schema_for("handoff")["required"])
    assert required - {"description"} == set(kind_mod.REQUIRED_KEYS)
    assert "description" not in kind_mod.REQUIRED_KEYS, (
        "description is for a human and is not a field of HandoffKind, "
        "so the schema is its only writer"
    )


def test_schema_for_rejects_a_kind_that_is_not_one() -> None:
    from spec_loader.protocols import SpecNotFound

    with pytest.raises(SpecNotFound):
        schema_for("handoffs")
