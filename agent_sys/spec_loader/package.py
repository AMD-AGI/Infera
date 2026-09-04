"""A task package, and loading one.

`load_package` is the one function in this package whose *body* is the design, so
`docs/design.md` §3.6 writes it out as a body rather than a signature. Three
properties of the ordering, each with a reason:

- **Failures are collected, not raised.** One broken spec must not hide the
  other nine, and a loader that dies on the first makes fixing a package an
  N-round trip. `check-jsonschema` goes further and returns parse failure as a
  value, which the package does here too.
- **Admission happens after all validation.** Half a package in a registry is a
  state nothing else in the system knows how to reason about.
- **Nothing cross-registry happens here at all.** Not the closure pass, not
  `handoff`'s two-way binding check, not `validator`'s separation check: each
  needs a registry this call may not have filled yet, and `load_package` runs
  once *per package*. They run once, at the composition root
  (`docs/interfaces.md` §2 step 5).

That last one is `docs/design.md` D8, and it is the correction `closure` design
D3 made to rev. 1 of the main design.

**What changed at rev. 10.** `load_package` no longer renders, no longer opens a
file, and no longer knows a source format exists. It is handed
`SpecDocument`s — parsed, substituted, discriminated — and its whole job is
validate-and-admit. Main spec §4.4's promise that *"the loader does not read,
audit, or constrain"* a package's source was an ordering convention inside this
function; it is now a property of what crosses the seam, which is what criterion
4 pins.

The work that left is in `YamlPackage`, below: everything about the *format*
lives on the package's side of the seam, where a second format would be a second
`TaskPackage` and not a change to this function.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping, MutableSequence, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .assets import (
    ASSETS_DIRNAME,
    AssetIndex,
    fill_agent_assets,
    fill_agent_env_recipe,
    fill_body,
)
from .bundled import schema_for
from .protocols import (
    LoadReport,
    PackageContents,
    Problem,
    Registries,
    SpecDocument,
    SpecInconsistent,
    TaskPackage,
)
from .validate import validate
from .variables import ASSETS_VAR, substitute
from .yaml_source import position_of, read_yaml

__all__ = ["ENTRY_FILENAME", "MODULE_KEY", "YamlPackage", "load_package"]

#: The discriminator. `refine.task_package.define.md` §1.1.1: an object is a
#: validator because it says `module: validator`, not because of where it sits.
MODULE_KEY = "module"

#: The entry's name (main spec §4.3). Fixed rather than inferred: a package holds
#: many documents and more than one may declare a graph, and inferring the entry —
#: "the one nothing references" — would make a package's meaning depend on a
#: global property of the tree, so adding an unreferenced draft would silently
#: move it.
#:
#: **The name is fixed; the file is not required of every package** (criterion 18,
#: rev. 11). Presence is the statement: a package carrying it is runnable, one
#: without is a library. What *is* checked per package is that a `main.yaml`
#: present declares a `module: task` — see `YamlPackage._entry_problems`.
ENTRY_FILENAME = "main.yaml"

#: What a user writes, to the schema kind it produces. Four words in and five
#: kinds out is not a mismatch: **users write `task`, never `closure`**
#: (`closure` spec §2), and `task` produces the closure document with the task
#: spec nested inside it — which is what `closure/check.py:709` already does by
#: splitting one out of the other.
_MODULES: Mapping[str, str] = {
    "handoff": "handoff",
    "validator": "validator",
    "agent": "agent",
    "task": "closure",
}

#: The keys whose values name another object. **This is the one place the
#: package layer knows the schemas' key names**, and it is here rather than
#: derived because the cost of the two errors is not symmetric: a missing
#: reference key means a forward reference goes unreported and the composition
#: root says *"does not resolve"* instead, while a spurious one **rejects a valid
#: package**. So the list is explicit and conservative.
#:
#: `froms` is deliberately absent. It names siblings inside one subgraph and the
#: rule over it is *"the listing order must be a valid topological order"*, which
#: `docs/ui-stage.md` §4 assigns to W5 along with the cross-check against the
#: derived edges. Two owners reporting one violation is worse than one.
#:
#: **`inputs` is absent too, and that one is a measurement.** A validator's
#: `inputs` names handoff kinds and a handoff's `validators` names validators, so
#: the two keys are a **cycle in the reference graph** — not a hypothetical one:
#: in `examples/demo/steps/produce.yaml` the handoff `facts` says
#: `validators: [check_facts]` and the validator `check_facts` says
#: `inputs: [facts]`, **in one file**. With both keys in this set, no ordering of
#: those two objects is legal, and the rule would forbid a package the system
#: already ships. One of the two
#: edges has to go and `inputs` is the weaker: `handoff` spec §5.3 already makes
#: "a kind with no validator" a named check, while `inputs` is not reliably a
#: reference at all — every general spec under `validator/general_specs/` writes
#: `inputs: ['any']`, which names nothing.
_REFERENCE_KEYS = frozenset(
    {
        "agent",  # closure -> agent spec
        "handoffs",  # closure -> handoff kinds
        "validators",  # closure, handoff -> validators
        "outputs",  # task -> handoff kinds
        "closure",  # subgraph entry -> another task
        "members",  # composite validator -> validators
    }
)


@dataclass(frozen=True)
class YamlPackage:
    """A task package: YAML anywhere, `assets/` at the root, `main.yaml` if runnable.

    ::

        my_package/
        |- main.yaml            the outermost graph's entry, iff this is runnable
        |- <anything>/**/*.yaml scanned. an object may live in any of them
        \\- assets/             MANDATORY. bodies, found by filename convention

    **The two fixed names have different arity and rev. 11 is where that was
    settled.** `assets/` is about *being a package* — every document may write an
    unqualified path, so the question arises for all of them — and is required of
    every one. `main.yaml` is about *being a run's entry*, which is one per
    **run**: a package with none is a library and loads (criterion 18).

    Beyond those two names nothing about the layout is fixed — not how many
    objects a file holds, not which directory a kind lives in, not whether there
    is a directory per kind at all (main spec §4.3).

    `variables` is the package-level set §4.4 names, and it is **caller-supplied**
    because measurement says that is the only kind anything uses: across all 21
    jsonnet sources the tree replaces, every variable reference was to
    `config.package_root`, `config.outside` or `config.inputs`, and not one
    package declared a constant of its own. A `vars:` block would be a construct
    with no measured user, which is `engineer_principle.md` §2's "do not put it
    anywhere yet". `${ASSETS_VAR}` is added on top and cannot be overridden — it
    is a fact about the package, not about the run.
    """

    root: Path
    variables: Mapping[str, str] = field(default_factory=dict)

    # -- the seam ----------------------------------------------------------- #

    def documents(self) -> PackageContents:
        """Scan, parse, substitute, discriminate, expand, order-check, emit."""
        if problems := self._structural_problems():
            return PackageContents(documents=(), problems=tuple(problems))

        index = AssetIndex(self.root / ASSETS_DIRNAME)
        variables = {**self.variables, ASSETS_VAR: ASSETS_DIRNAME}

        documents: list[SpecDocument] = []
        problems = []
        for path in self._scan():
            docs, probs = self._read_one(path, index=index, variables=variables)
            documents.extend(docs)
            problems.extend(probs)

        problems += self._entry_problems(documents, problems)
        return PackageContents(documents=tuple(documents), problems=tuple(problems))

    # -- structure ---------------------------------------------------------- #

    def _structural_problems(self) -> list[Problem]:
        """Main spec criterion 16 — `assets/`, and that is the whole of it.

        **`main.yaml`'s absence is not checked here, and rev. 11 is why.** Rev. 10
        demanded both names of every package and this method implemented that.
        `spec-author` split the rule after measuring that the two have different
        *arity*: `assets/` is about **being a package** — every document may
        write an unqualified path, so the question arises for all of them —
        while `main.yaml` is about **being a run's entry**, which is one per
        *run*. `task_graph/bootstrap.py:47` takes `packages: Sequence[Any]` and
        loads each into one shared set of registries, so demanding the file of
        every package answers *"where does a run start"* N times and therefore
        not at all. It also made a kinds-only library package inexpressible,
        three paragraphs after §4.3 permits exactly that.

        So **presence is the statement**: a package carrying `main.yaml` is
        runnable, one without is a library, and absence is not a fault. What is
        still checked per package is the file's *contents* — see
        `_entry_problems`. The per-run half (exactly one entry package) has no
        owner and is §10's, not this method's to invent.

        Reported as fatal `Problem`s rather than raised, so that a caller gets
        one answer from one call (`engineer_principle.md` §1).
        """
        origin = str(self.root)
        if not self.root.is_dir():
            return [
                Problem(
                    origin=origin,
                    path="$",
                    keyword="package",
                    message="not a directory, so it is not a task package",
                )
            ]

        if not (self.root / ASSETS_DIRNAME).is_dir():
            return [
                Problem(
                    origin=origin,
                    path="$",
                    keyword="package",
                    message=(
                        f"no {ASSETS_DIRNAME}/ at the package root. It is what an "
                        f"unqualified path in a document is relative to, and the tree "
                        f"a body's files are found in."
                    ),
                )
            ]
        return []

    def _entry_problems(
        self, documents: Sequence[SpecDocument], problems: Sequence[Problem]
    ) -> list[Problem]:
        """Main spec criterion 18 — a `main.yaml` that is an entry to nothing.

        *"A `main.yaml` that is present but declares no `module: task` is
        rejected naming the file, because a file whose whole definition is 'the
        outermost graph's entry' cannot be an entry to nothing."*

        Checked over the **emitted documents** rather than by re-reading the
        file, so an object written inline inside the entry counts — a hoisted
        `module: task` is a graph the author really did declare, and a re-read
        looking for a top-level key would reject a well-formed package.

        **A file that did not parse is skipped, and the first draft did not skip
        it.** It declared no `module: task` *because it declared nothing at all*,
        so the check fired on top of the syntax error and named the wrong
        problem — the derived-error family `failed_names` exists to suppress,
        one layer earlier. `test_a_main_yaml_that_did_not_parse_reports_the_parse
        _error_only` is what found it; the docstring had already claimed the case
        was handled before the code did any such thing.

        It does not demand a `subgraph`. A package with one task has a degenerate
        graph and no specification says that is ill formed, so requiring the key
        would be inventing a rule — the same line this method's neighbour stopped
        drawing at rev. 11.
        """
        entry = self.root / ENTRY_FILENAME
        if not entry.is_file():
            return []  # a library, and that is a statement rather than a fault
        origin = str(entry)
        if any(p.origin.startswith(origin) for p in problems):
            return []
        if any(d.kind == "closure" and d.origin.startswith(origin) for d in documents):
            return []
        return [
            Problem(
                origin=origin,
                path="$",
                keyword="package",
                message=(
                    f"{ENTRY_FILENAME} declares no `{MODULE_KEY}: task`, so it is an "
                    f"entry to nothing. It is the one file whose definition is "
                    f"'the outermost graph's entry'; a package with no graph is a "
                    f"library and should not carry it at all."
                ),
            )
        ]

    def _scan(self) -> list[Path]:
        """Every YAML under the root except `assets/`, in a deterministic order.

        `assets/` is excluded because a `*.yaml` in there would be both an object
        and an asset, which is the namespace collision main spec §4.3 gives as
        the reason the directory is mandatory in the first place.

        **`main.yaml` sorts last, and the order is not semantically load-bearing
        across files.** The forward-reference rule is *within* a file — see
        `_order_problems`, which records the measurement that no total file order
        can work. This one is chosen so that `LoadReport.admitted` is
        reproducible and so that the entry, which by definition references the
        graph below it, comes after what it names.

        `.yml` is scanned too. It is the same format under a second spelling, and
        ignoring it silently is worse than either accepting or rejecting it.

        **`is_symlink()` is there for the dangling case and is criterion 6's
        second half.** `is_file()` follows the link and answers `False` for one
        that points nowhere, so a scan filtered on it alone would make a broken
        link *invisible* rather than an error — the criterion asks for "fails
        naming the path", and a silently skipped file names nothing.
        """
        assets = self.root / ASSETS_DIRNAME
        entry = self.root / ENTRY_FILENAME
        found = [
            p
            for pattern in ("*.yaml", "*.yml")
            for p in self.root.rglob(pattern)
            if (p.is_file() or p.is_symlink()) and assets not in p.parents and p != entry
        ]
        return sorted(set(found)) + ([entry] if entry.is_file() else [])

    # -- one file ------------------------------------------------------------ #

    def _read_one(
        self,
        path: Path,
        *,
        index: AssetIndex,
        variables: Mapping[str, str],
    ) -> tuple[list[SpecDocument], list[Problem]]:
        origin = str(path)
        tree, problems = read_yaml(path, origin=origin)
        if problems:
            return [], problems
        if tree is None:
            # An empty file. Legal and uninteresting — a package may hold a
            # placeholder — and silence is the right answer rather than a
            # problem about a file that says nothing.
            return [], []

        # Substituted over the whole file at once, before anything is hoisted,
        # so an inline definition is filled exactly like a top-level one. The
        # JSONPath in a fault is then file-relative (`$[1].task.body.readme`),
        # which locates it better than a document-relative one would in a file
        # holding twenty objects.
        problems += substitute(tree, variables, origin=origin)

        raw, probs = _objects_in(tree, origin=origin)
        problems += probs

        documents: list[SpecDocument] = []
        for obj, pointer in raw:
            docs, probs = self._one_object(
                obj, pointer=pointer, path=path, index=index, container=tree
            )
            documents.extend(docs)
            problems += probs

        problems += _order_problems(documents, origin=origin)
        return documents, problems

    def _one_object(
        self,
        obj: MutableMapping[str, Any],
        *,
        pointer: str,
        path: Path,
        index: AssetIndex,
        container: Any,
    ) -> tuple[list[SpecDocument], list[Problem]]:
        """One declared object, plus every object defined inside it.

        Post-order: an inline definition is emitted **before** the object that
        references it, so that "defined before use" holds for the one case where
        the author had no choice about the order.
        """
        documents: list[SpecDocument] = []
        problems: list[Problem] = []

        for inline, inline_pointer in _hoist_inline(obj, pointer):
            docs, probs = self._one_object(
                inline, pointer=inline_pointer, path=path, index=index, container=container
            )
            documents.extend(docs)
            problems += probs

        origin = _origin(path, pointer)
        module = obj.pop(MODULE_KEY, None)
        kind = _MODULES.get(module) if isinstance(module, str) else None
        if kind is None:
            problems.append(
                Problem(
                    origin=origin,
                    path=f"$.{MODULE_KEY}",
                    keyword="module",
                    message=(
                        f"{module!r} is not a module. Every object declares one of: "
                        f"{', '.join(sorted(_MODULES))}."
                        if module is not None
                        else f"no {MODULE_KEY}: key, so this object declares nothing. "
                        f"Write one of: {', '.join(sorted(_MODULES))}."
                    ),
                    line=_line(obj),
                )
            )
            return documents, problems

        name = obj.get("name")
        if isinstance(name, str) and name:
            # Three fillers, one `try`, and **three different `$.` paths in the
            # fault**. They fill different fields from the same index and any of
            # them can find two candidates; catching them separately would
            # triplicate the handler, and merging their `path` into one would
            # point a reader at `$.body` for a conflict between two `assets/`
            # directories.
            for filler, pointer in (
                (fill_body, "$.body"),
                (fill_agent_assets, "$.assets"),
                (fill_agent_env_recipe, "$.recipes"),
            ):
                try:
                    problems += filler(
                        obj, index, kind=kind, name=name, origin=origin, line=_line(obj)
                    )
                except SpecInconsistent as exc:
                    problems.append(
                        Problem(
                            origin=origin,
                            path=pointer,
                            keyword="inconsistent",
                            message=str(exc),
                            line=_line(obj),
                        )
                    )

        at = position_of(obj)
        documents.append(
            SpecDocument(
                kind=kind,
                doc=obj,
                origin=origin,
                line=at.line if at else None,
                column=at.column if at else None,
            )
        )
        return documents, problems


# --------------------------------------------------------------------------- #
# Reading the shape of a file


def _objects_in(tree: Any, *, origin: str) -> tuple[list[tuple[Any, str]], list[Problem]]:
    """The declared objects at a file's root, each with its JSON pointer.

    A file holds **one mapping** or **a list of mappings** — the two shapes
    `refine.task_package.define.md` §1.1 names. The pointer is empty for the
    first, which is what keeps a single-object file's `origin` reading exactly as
    it did before rev. 10.
    """
    if isinstance(tree, Mapping):
        return [(tree, "")], []
    if isinstance(tree, Sequence) and not isinstance(tree, (str, bytes)):
        out: list[tuple[Any, str]] = []
        problems: list[Problem] = []
        for i, item in enumerate(tree):
            if isinstance(item, MutableMapping):
                out.append((item, f"/{i}"))
            else:
                problems.append(
                    Problem(
                        origin=origin,
                        path=f"$[{i}]",
                        keyword="shape",
                        message=(
                            f"a list entry must be an object declaring a "
                            f"{MODULE_KEY}:, and this one is a "
                            f"{type(item).__name__}"
                        ),
                        line=(p.line if (p := position_of(tree, i)) else None),
                    )
                )
        return out, problems
    return [], [
        Problem(
            origin=origin,
            path="$",
            keyword="shape",
            message=(
                f"a spec file holds one object or a list of them, and this holds a "
                f"{type(tree).__name__}"
            ),
        )
    ]


def _hoist_inline(obj: Any, pointer: str) -> Iterator[tuple[MutableMapping[str, Any], str]]:
    """Every object defined inline below `obj`, removed and replaced by its name.

    **The rule is the discriminator and nothing else**: a mapping carrying a
    `module:` key anywhere below the root is a definition written at its point of
    use, and is replaced by the string in its `name`. Keying on `module:` rather
    than on *which key it sits under* is what keeps this from needing to know the
    schemas — a subgraph entry's `closure:`, a closure's `agent:`, a handoff's
    `validators:` list and anywhere a later schema puts one all work without a
    line here.

    An object with no usable `name` is left where it is, deliberately: it then
    reaches the schema as a mapping where a string belongs and is reported by the
    enforcement point, rather than being hoisted under a name this function made
    up.
    """
    for container, key, child in _children(obj):
        here = f"{pointer}{_pointer_step(key)}"
        if isinstance(child, MutableMapping) and MODULE_KEY in child:
            name = child.get("name")
            if isinstance(name, str) and name:
                container[key] = name
                yield child, here
                continue
        # Sequences are descended too, and that is not an afterthought: a
        # closure's `outputs`, a handoff's `validators` and a subgraph are all
        # lists, so a hoist that only walked mappings would miss every place an
        # inline definition is most natural to write.
        if isinstance(child, (MutableMapping, MutableSequence)) and not isinstance(
            child, (str, bytes)
        ):
            yield from _hoist_inline(child, here)


def _children(node: Any) -> Iterator[tuple[Any, Any, Any]]:
    """`(container, key, value)` for everything one level down, mappings and
    sequences alike, so a caller assigns back through the container it was
    handed."""
    if isinstance(node, MutableMapping):
        for key in list(node):
            yield node, key, node[key]
    elif isinstance(node, MutableSequence) and not isinstance(node, (str, bytes)):
        for i in range(len(node)):
            yield node, i, node[i]


def _pointer_step(key: Any) -> str:
    """One JSON Pointer segment (RFC 6901): `~` and `/` are escaped."""
    return "/" + str(key).replace("~", "~0").replace("/", "~1")


def _origin(path: Path, pointer: str) -> str:
    """`<path>` for a file holding one object, `<path>#<pointer>` otherwise.

    Absolute, because `str(source.path)` was absolute before rev. 10 and two
    packages with the same internal layout must not produce the same origin —
    `task_graph/bootstrap.py`'s `_names_for` bridges origins to names by exact
    equality, and a collision there would map one package's failure onto
    another's spec. `docs/ui-stage.md` §2 writes the shape as
    `steps/collect.yaml#/2`; the pointer half is adopted and the relative half is
    not, for that reason.
    """
    return f"{path}#{pointer}" if pointer else str(path)


def _line(node: Any) -> int | None:
    at = position_of(node)
    return at.line if at else None


# --------------------------------------------------------------------------- #
# Definition order


def _order_problems(documents: Sequence[SpecDocument], *, origin: str) -> list[Problem]:
    """A reference to a name defined **later in the same file** is an error.

    `refine.task_package.define.md` §1.1.2, and the scope is the finding.

    **Within a file, not across the package**, and that is measured rather than
    chosen for convenience. `examples/demo` is the only real package in the tree,
    and no total file order satisfies its own reference graph: sorted by path,
    `closures/produce` (index 4) references `handoffs/facts` (5) and
    `validators/check_facts` (7); reversed, it references `agents/collect`, which
    is then last. No ordering by kind works either, because the reference edges
    genuinely cycle between two of them — a handoff names its validators and a
    validator names its input handoff kinds.

    So a package-wide rule would be one no author could satisfy except by
    renaming files, and the user's own sentence sits under the *"several
    definitions in one file as a list"* bullet. Cross-file forward references are
    caught where they always were: the composition root's passes say a name does
    not resolve. **Reported to the lead as a narrowing, not decided quietly.**
    """
    problems: list[Problem] = []
    defined: set[str] = set()
    later = {d.doc.get("name") for d in documents if isinstance(d.doc.get("name"), str)}

    for document in documents:
        for path, referenced in _references(document.doc):
            if referenced in defined or referenced not in later:
                continue
            problems.append(
                Problem(
                    origin=document.origin,
                    path=path,
                    keyword="order",
                    message=(
                        f"{referenced!r} is referenced here and defined later in this "
                        f"file. Definitions are read in order; move it above."
                    ),
                    line=document.line,
                )
            )
        name = document.doc.get("name")
        if isinstance(name, str):
            defined.add(name)
    return problems


def _references(node: Any, path: str = "$") -> Iterator[tuple[str, str]]:
    """`(jsonpath, name)` for every value under a `_REFERENCE_KEYS` key."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            here = f"{path}.{key}"
            if key in _REFERENCE_KEYS:
                if isinstance(value, str):
                    yield here, value
                elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    for i, item in enumerate(value):
                        if isinstance(item, str):
                            yield f"{here}[{i}]", item
                        else:
                            yield from _references(item, f"{here}[{i}]")
            else:
                yield from _references(value, here)
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        for i, item in enumerate(node):
            yield from _references(item, f"{path}[{i}]")


# --------------------------------------------------------------------------- #
# The loader


def load_package(pkg: TaskPackage, registries: Registries) -> LoadReport:
    """Validate and admit one package's documents.

    Two steps and no third. There is no parameter through which a path could
    reach this function and no branch on where a document came from, which is
    main spec criterion 4 expressed as code rather than as a convention.
    """
    contents = pkg.documents()
    problems: list[Problem] = list(contents.problems)

    validated: list[SpecDocument] = []
    for document in contents.documents:
        errs = validate(document.doc, schema_for(document.kind), origin=document.origin)
        if errs:
            problems.extend(_located(errs, document))
            continue
        validated.append(document)

    admitted: list[str] = []
    for document in validated:
        name = document.doc["name"]
        try:
            registries.for_kind(document.kind).add(name, document.doc, origin=document.origin)
        except ValueError as exc:
            # A kind's own load-time checks live in its registry subclass and
            # raise; this call collects rather than propagates, for the same
            # reason step 1 does. The composition root raises once, over
            # everything (`docs/interfaces.md` §2 step 5).
            #
            # **`ValueError`, not `(SpecInvalid, SpecInconsistent)`**, and that
            # is a repair rather than a widening. `BaseSpecRegistry._validate`
            # documents those two as what a subclass raises, and measured
            # against the four real registries, `validator` raises
            # `ValidatorInvalid` — a `ValueError`, but neither of them. It
            # escaped, and one package's choice of exception type aborted the
            # whole multi-package load: "collect, do not raise" became "die on
            # the first", silently, for every other package in the run.
            #
            # A contract four packages must remember is one this function should
            # not depend on. Every module exception measured is a `ValueError`,
            # as are both of ours, so this holds without anyone changing — while
            # still letting a `TypeError` or `AttributeError` out, because those
            # are bugs in a registry rather than rejections of a spec.
            problems.append(
                Problem(
                    origin=document.origin,
                    path="$",
                    keyword=_keyword_of(exc),
                    message=str(exc),
                    line=document.line,
                )
            )
            continue
        admitted.append(name)

    return LoadReport(admitted=tuple(admitted), problems=tuple(problems))


def _located(problems: Sequence[Problem], document: SpecDocument) -> list[Problem]:
    """Stamp the document's position onto the schema's findings.

    **The document's, not the field's**, and that limit is worth stating rather
    than papering over. A schema error knows its `json_path`; joining that back
    onto a source position needs the parse tree, and `validate` cannot see one —
    that is the whole of what makes it path-free. So a diagnostic says *"this
    object, which starts at line 12, has a bad `$.body.entry`"*, which is enough
    to find in a file holding twenty objects and is not a guess.
    """
    if document.line is None:
        return list(problems)
    return [
        Problem(
            origin=p.origin,
            path=p.path,
            keyword=p.keyword,
            message=p.message,
            fatal=p.fatal,
            line=document.line,
            column=document.column,
        )
        for p in problems
    ]


def _keyword_of(exc: BaseException) -> str:
    """`inconsistent` for the two-records-disagree case, `invalid` otherwise.

    JPMS separates "not found" from "found, but inconsistent" and the
    distinction is load-bearing here too: a missing validator is a typo, while a
    two-way mismatch means one of two records is lying and nobody knows which.
    A registry raising its own `ValueError` subclass lands on `invalid`, which
    is right — it is a spec that failed its kind's own check.
    """
    return "inconsistent" if isinstance(exc, SpecInconsistent) else "invalid"
