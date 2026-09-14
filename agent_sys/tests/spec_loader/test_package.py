"""Main spec criteria 4, 6, 16 and 18 — what a package may vary, and what it may not.

- **4** *"The loader never sees a package's source. It is handed parsed documents
  and has no parameter through which a path could arrive, so two packages whose
  files are organised wildly differently … are indistinguishable to it and both
  load."*
- **6** *"A task package resolves without the loader knowing its layout: a package
  whose validator symlinks point into a second package loads, and a dangling
  symlink fails naming the path."*
- **16** *"`assets/` is required of every package, and it is the whole of the
  layout a package must have."*
- **18** *"`main.yaml` states that a package is runnable, and says what it is."*

4 and 16 are opposite halves and 6 is 16's neighbour: 6 is what may vary, 16 is
what may not, and 4 is the reason the loader cannot tell.

**16 and 18 were one criterion until rev. 11**, and the split is the reason these
tests read the way they do. Rev. 10 demanded both names of every package; the two
have different *arity* — `assets/` is about being a package, `main.yaml` about
being a **run's** entry, which is one per run and not one per package — so
demanding both of every package answered *"where does a run start"* N times, and
made a kinds-only library inexpressible three paragraphs after §4.3 permits one.
"""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import pytest

from spec_loader import (
    ENTRY_FILENAME,
    KINDS,
    PackageContents,
    SpecDocument,
    YamlPackage,
    load_package,
    schema_for,
)

from .conftest import HANDOFF, ONE_OF_EACH, VALIDATOR, FakeRegistries, PackageBuilder

# --------------------------------------------------------------------------- #
# Criterion 4 — the loader never sees a package's source
#
# The document below is one handoff kind. Two packages produce it: one with a
# file per object under a directory per kind, and one with every object inline
# in a single `main.yaml`. Criterion 4 names exactly this pair.

PER_FILE_LAYOUT = {
    "handoffs/trace.yaml": (
        "module: handoff\n"
        "name: trace\n"
        "description: a captured kernel trace\n"
        "content_type: reproducible\n"
        "scope: fixed.required\n"
        "validators: [check_trace_shape]\n"
    ),
}

#: The same object, written where it is used, inside the entry — and reached
#: through two levels of nesting so that hoisting is doing real work rather than
#: unwrapping a top-level key.
ALL_INLINE = """\
module: task
name: main
description: the mandatory entry
handoffs: []
validators: []
agent:
  module: agent
  name: main_agent
  kind: program
  description: does the work
task:
  goal: hold the package together
  version: "1"
  inputs: []
  outputs:
    - module: handoff
      name: trace
      description: a captured kernel trace
      content_type: reproducible
      scope: fixed.required
      validators: [check_trace_shape]
"""


def test_two_packages_same_document_indistinguishable(tmp_path: Path) -> None:
    """One document, two wildly different packages, and both load.

    **Strengthened at rev. 10 rather than replaced.** It used to compare two
    *jsonnet sources* — a plain object against a function-and-overlay pile — and
    assert that rendering both gave one document. There is no rendering, so the
    thing it demonstrated moved: what varies now is the **package's layout**,
    which is what the amended criterion names, and the property is no longer
    "the loader renders first" but "the loader is handed documents and cannot
    tell". The name is kept because the criterion it maps to is the same one.
    """
    per_file = _build(tmp_path / "per_file", PER_FILE_LAYOUT)
    inline = _build(tmp_path / "inline", {ENTRY_FILENAME: ALL_INLINE})

    left = _by_name(per_file.documents().documents)
    right = _by_name(inline.documents().documents)

    assert left["trace"].doc == right["trace"].doc
    assert left["trace"].kind == right["trace"].kind == "handoff"

    for pkg in (per_file, inline):
        registries = FakeRegistries()
        result = load_package(pkg, registries)
        assert result.problems == (), result.problems
        assert registries.handoff_specs.get("trace")["description"] == "a captured kernel trace"


def test_load_package_opens_no_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """The type boundary, made observable.

    Criterion 4's *"no parameter through which a path could arrive"* is a claim
    about a signature, and a signature claim should fail loudly when someone adds
    a convenience read (`test_validate_takes_no_path` is the same idea one layer
    down). Handing `load_package` a `TaskPackage` that touches no filesystem and
    then failing the test if anything is opened is what makes it structural
    rather than reviewed.

    Before rev. 10 this test could not have been written: `load_package` called
    `render`, which read every source.

    **The bundled schemas are warmed first and that is not a loophole.** They are
    this package's own resource, read once through `importlib.resources` and
    memoised; a *package's* source is the thing criterion 4 is about. Warming
    them separates the two, and the spy is left recording afterwards so that a
    read of anything else — including a schema re-read — fails the test.
    """
    for kind in KINDS:
        schema_for(kind)

    opened: list[Any] = []
    real_open, real_text, real_bytes = builtins.open, Path.read_text, Path.read_bytes
    monkeypatch.setattr(builtins, "open", lambda *a, **k: (opened.append(a), real_open(*a, **k))[1])
    monkeypatch.setattr(
        Path, "read_text", lambda self, *a, **k: (opened.append(self), real_text(self, *a, **k))[1]
    )
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self, *a, **k: (opened.append(self), real_bytes(self, *a, **k))[1],
    )

    result = load_package(_InMemory(), FakeRegistries())

    assert opened == [], f"load_package opened {opened}"
    assert result.admitted == ("trace",)


class _InMemory:
    """A `TaskPackage` with no directory behind it at all.

    Legal by the Protocol — `root` is a `Path` and nothing says it exists — and
    the point: a package is whatever produces documents, and the loader has no
    way to ask where they came from.
    """

    root = Path("/nowhere")

    def documents(self) -> PackageContents:
        return PackageContents(
            documents=(
                SpecDocument(
                    kind="handoff",
                    doc={
                        "name": "trace",
                        "description": "a captured kernel trace",
                        "content_type": "reproducible",
                        "scope": "fixed.required",
                    },
                    origin="memory://trace",
                ),
            ),
            problems=(),
        )


# --------------------------------------------------------------------------- #
# Criterion 16 — `assets/`, and that is the whole of it


def test_a_package_without_assets_fails_naming_the_root(tmp_path: Path) -> None:
    """The message names the root, because that is the thing the author has to
    look at — there is no file to point at.

    `assets/` is required of **every** package and not only a runnable one:
    every document may write an unqualified path, so there is no package for
    which the question does not arise.
    """
    root = tmp_path / "pkg"
    root.mkdir()
    (root / ENTRY_FILENAME).write_text("module: task\nname: a\n")

    (problem,) = YamlPackage(root=root).documents().problems

    assert problem.keyword == "package"
    assert problem.origin == str(root)
    assert "assets/" in problem.message
    assert problem.fatal


# --------------------------------------------------------------------------- #
# Criterion 18 — `main.yaml` states that a package is runnable


def test_a_package_without_main_yaml_is_a_library_and_loads(tmp_path: Path) -> None:
    """*"A package with no `main.yaml` loads, and its documents are admitted — a
    package shipping only shared handoff kinds is a library, and its absence of a
    graph is a statement rather than a fault."*

    **This test asserts the opposite of the one it replaces.** Under rev. 10 the
    same package was a fatal error, and this file's
    `test_a_package_without_main_yaml_fails_naming_the_root` said so. That rule
    defeated §4.3's own reason for fixing the name: a run over several packages
    would hold several files each claiming to declare the outermost graph. The
    reversal is recorded rather than quietly applied, because a test whose
    assertion inverts is the one a reader most needs told about.
    """
    root = tmp_path / "pkg"
    (root / "assets").mkdir(parents=True)
    (root / "kinds.yaml").write_text(
        "module: handoff\nname: trace\ndescription: d\ncontent_type: text\nscope: fixed.required\n"
    )

    registries = FakeRegistries()
    result = load_package(YamlPackage(root=root), registries)

    assert result.problems == (), result.problems
    assert result.admitted == ("trace",)


def test_a_main_yaml_declaring_no_task_is_an_entry_to_nothing(tmp_path: Path) -> None:
    """*"A `main.yaml` that is present but declares no `module: task` is rejected
    naming the file, because a file whose whole definition is 'the outermost
    graph's entry' cannot be an entry to nothing."*

    This is what survives of the unconditional rule, and it is the half that was
    always per-package: the file's *contents* are checkable here, while "exactly
    one entry package per run" is a per-run fact no module owns today (§10).
    """
    root = tmp_path / "pkg"
    (root / "assets").mkdir(parents=True)
    (root / ENTRY_FILENAME).write_text("module: agent\nname: a\nkind: program\n")

    (problem,) = YamlPackage(root=root).documents().problems

    assert problem.keyword == "package"
    assert problem.origin == str(root / ENTRY_FILENAME), "names the file, not the root"
    assert "entry to nothing" in problem.message
    assert problem.fatal


def test_a_task_written_inline_in_main_yaml_counts(tmp_path: Path) -> None:
    """The check is over the emitted documents, not a re-read of the file.

    An object written inline is hoisted out of its host and registered under its
    own name; a check that re-read `main.yaml` looking for a top-level `module:
    task` would reject a package whose entry is perfectly well formed.
    """
    root = tmp_path / "pkg"
    (root / "assets").mkdir(parents=True)
    (root / "assets" / "inner.md").write_text("x")
    (root / ENTRY_FILENAME).write_text(
        "- module: handoff\n  name: trace\n  description: d\n"
        "  content_type: text\n  scope: fixed.required\n"
        "- module: task\n  name: inner\n  description: d\n"
        "  agent: a\n  handoffs: []\n  validators: []\n"
        '  task: {goal: g, version: "1", inputs: [], outputs: []}\n'
    )

    assert YamlPackage(root=root).documents().problems == ()


def test_a_main_yaml_that_did_not_parse_reports_the_parse_error_only(tmp_path: Path) -> None:
    """One fault, not two.

    A file that did not parse declares no `module: task` *because it declared
    nothing at all*, and telling the author it is "an entry to nothing" on top of
    the syntax error names the wrong problem. This is the derived-error family
    `failed_names` exists to suppress, one layer earlier.
    """
    root = tmp_path / "pkg"
    (root / "assets").mkdir(parents=True)
    (root / ENTRY_FILENAME).write_text("module: task\nname: x\n  bad: indent\n")

    problems = YamlPackage(root=root).documents().problems

    assert [p.keyword for p in problems] == ["parse"]


def test_a_root_that_is_not_a_directory_says_so(tmp_path: Path) -> None:
    (problem,) = YamlPackage(root=tmp_path / "absent").documents().problems

    assert "not a directory" in problem.message


def test_one_file_and_two_hundred_objects_are_equally_well_formed(tmp_path: Path) -> None:
    """Criterion 16's second sentence: *"a package that declares every object in
    a single file and one that gives each its own both load, with no
    directory-per-kind anywhere."*

    The layout here is deliberately hostile to any convention the loader might
    have been tempted to infer: no directory per kind, files named after nothing,
    and the entry holding an object of a different kind from its own.
    """
    root = tmp_path / "pkg"
    (root / "assets").mkdir(parents=True)
    (root / "assets" / "everything.md").write_text("x")
    # A library: 200 handoff kinds and no graph, so no `main.yaml` (criterion 18).
    (root / "solo.yaml").write_text(
        "module: agent\nname: everything\nkind: program\ndescription: the only agent\n"
    )
    (root / "z" / "y" / "x").mkdir(parents=True)
    (root / "z" / "y" / "x" / "unrelated-name.yaml").write_text(
        "\n".join(
            f"- module: handoff\n"
            f"  name: k{i}\n"
            f"  description: kind {i}\n"
            f"  content_type: text\n"
            f"  scope: fixed.required\n"
            for i in range(200)
        )
    )

    registries = FakeRegistries()
    result = load_package(YamlPackage(root=root), registries)

    assert result.problems == (), result.problems
    assert len(registries.handoff_specs.names()) == 200
    assert registries.agent_specs.names() == ["everything"]


# --------------------------------------------------------------------------- #
# Criterion 6 — a package resolves without the loader knowing its layout


def test_cross_package_symlink_loads(tmp_path: Path) -> None:
    """A package reaching into a second one through a relative symlink.

    Main spec §4.3 permits exactly this — *"two packages may reference each
    other, and they do it themselves: a relative symlink from one package into
    another"* — and the loader must not need to know that packages can be
    nested, adjacent, or shared.

    **What it demonstrates moved with the seam, and the criterion says so.** It
    used to be a jsonnet `import` crossing the boundary, which reached the
    loader through `render`. A package now resolves its own files, so what
    crosses is a *scanned document* and an *asset* found through a link — and
    criterion 6's rev. 10 note is explicit that a test reaching into the loader
    would no longer be demonstrating it.
    """
    other = tmp_path / "other_package"
    (other / "shared").mkdir(parents=True)
    (other / "shared" / "check_trace_shape.md").write_text("shared readme")
    (other / "shared" / "borrowed.yaml").write_text(
        f"module: validator\nname: check_trace_shape\n{VALIDATOR}"
    )

    pkg = PackageBuilder(tmp_path / "pkg")
    (pkg.root / "borrowed.yaml").symlink_to(
        Path("..") / "other_package" / "shared" / "borrowed.yaml"
    )
    (pkg.root / "assets" / "check_trace_shape.md").symlink_to(
        Path("..") / ".." / "other_package" / "shared" / "check_trace_shape.md"
    )

    registries = FakeRegistries()
    result = load_package(pkg.package(), registries)

    assert result.problems == (), result.problems
    spec = registries.validator_specs.get("check_trace_shape")
    assert spec["dimension"] == "completeness"
    assert spec["body"]["readme"] == "assets/check_trace_shape.md"


def test_dangling_symlink_names_path(tmp_path: Path) -> None:
    """A symlink that dangles is a load error naming the path, not a puzzle."""
    pkg = PackageBuilder(tmp_path / "pkg")
    broken = pkg.root / "gone.yaml"
    broken.symlink_to(Path("nowhere.yaml"))

    result = load_package(pkg.package(), FakeRegistries())

    (problem,) = [p for p in result.problems if p.keyword == "unreadable"]
    assert str(broken) == problem.origin
    assert "gone.yaml" in problem.message


# --------------------------------------------------------------------------- #
# Scanning


def test_assets_is_not_scanned(builder: PackageBuilder) -> None:
    """A `*.yaml` under `assets/` is an asset, not an object.

    Main spec §4.3 gives this as the *reason* `assets/` is mandatory: without a
    separate root, the scanned documents and the assets share one namespace and
    a file is both.
    """
    builder.asset("data/fixture.yaml", "module: handoff\nname: sneaky\n")

    result = load_package(builder.package(), FakeRegistries())

    assert result.admitted == ("main",)


def test_an_empty_file_is_not_a_problem(builder: PackageBuilder) -> None:
    """A package may hold a placeholder, and silence is the right answer."""
    builder.write("draft.yaml", "")

    result = load_package(builder.package(), FakeRegistries())

    assert result.problems == ()


def test_yml_is_scanned_too(builder: PackageBuilder) -> None:
    """The same format under a second spelling.

    Scanning it is the choice between accepting a `.yml` and rejecting it; what
    is not on the table is ignoring it, because a package whose objects silently
    do not exist is the failure mode this tree has thirteen recorded instances
    of.
    """
    builder.asset("trace.md", "x")
    builder.write("kinds.yml", f"module: handoff\nname: trace\n{HANDOFF}")

    result = load_package(builder.package(), FakeRegistries())

    assert sorted(result.admitted) == ["main", "trace"]


def test_documents_come_out_in_a_deterministic_order(builder: PackageBuilder) -> None:
    """`LoadReport.admitted` is compared in tests and read by humans, so the
    order is fixed: sorted by path, with the entry last."""
    for module, (name, body) in ONE_OF_EACH.items():
        builder.one(module, name, body)

    first = [d.origin for d in builder.package().documents().documents]
    second = [d.origin for d in builder.package().documents().documents]

    assert first == second
    assert first[-1].endswith(ENTRY_FILENAME)


# --------------------------------------------------------------------------- #
# What `load_package` deliberately does not do


def test_load_package_runs_no_cross_registry_check(
    builder: PackageBuilder, registries: FakeRegistries
) -> None:
    """`docs/design.md` D8 — the closure pass moved out, and this is what holds it out.

    A task naming a validator, an agent and a handoff kind that exist in **no
    registry** loads without complaint. That is not laxity: `load_package` runs
    once *per package*, so with two packages the pass would fire with the
    second's specs in no registry, and main spec §4.3 makes cross-package
    references a supported case. The pass runs once, at the composition root,
    after every package is loaded (`docs/interfaces.md` §2 step 5).
    """
    builder.one("task", "collect_trace", _UNRESOLVABLE)

    result = load_package(builder.package(), registries)

    assert result.problems == ()
    assert sorted(result.admitted) == ["collect_trace", "main"]
    assert registries.handoff_specs.names() == []


#: Every name in it resolves to nothing at all, and the package still loads.
_UNRESOLVABLE = """\
description: every name below resolves to nothing at all
agent: nobody_defined_this
handoffs: [nor_this]
validators: [nor_this_either]
task:
  goal: collect a kernel trace
  version: "1"
  inputs: [nor_this]
  outputs: []
"""


def _build(root: Path, files: dict[str, str]) -> YamlPackage:
    (root / "assets").mkdir(parents=True)
    (root / "assets" / "main.md").write_text("the root task")
    if ENTRY_FILENAME not in files:
        from .conftest import MAIN

        files = {ENTRY_FILENAME: MAIN, **files}
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return YamlPackage(root=root)


def _by_name(documents: Any) -> dict[str, SpecDocument]:
    return {d.doc["name"]: d for d in documents}
