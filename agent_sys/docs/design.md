# Agent Work System — Design

| | |
|---|---|
| Status | Draft — stage two of spec → design → test & code |
| Revision | 3 — 2026-08-29. **The user-interface stage: the render step is deleted.** §3 was written around `render.py` and jsonnet, and neither exists. `render.py`, `ImportResolver`, `SpecSource` and `DirectoryPackage` are gone; §3.2 is now the YAML source reader, §3.3 the variable set, §3.4 `ruamel.yaml`'s diagnostics, and §3.6 the two-step `load_package` the code actually has. **Three measurements are kept as history rather than deleted** and each says which it is (§3.2): jsonnet's GIL-release and fixed ~23 ms per-render, which no longer describe anything and whose *conclusion* — parallelise — is unmeasured for YAML; the import-containment evidence, which outlived the resolver it was attached to and is now O3; and the IEEE-754 finding, which the format change **closes** for integers (measured) and leaves untouched for floats. Adopt table, criteria map, D4, O1, O2, O4 follow. (rev. 2: 2026-08-27. **The stage-three consistency pass.** The closure pass moves out of `load_package` and into the composition root (§3.6, §7 — `closure` design D3); §7 becomes the *whole* composition root rather than the spec layer's half of it (§7, and [`interfaces.md`](interfaces.md) §2, which is now normative for it); `Registries` is defined here rather than used undefined (§3.7); `cli/`'s layout is corrected to what `demo` design D2 measured (§2). (rev. 1: 2026-08-26. Initial) |
| Implements | [`spec.md`](spec.md) rev. 11, criteria 1–18 |
| Language | Python ≥ 3.10. pydantic v2, `ruamel.yaml`, jsonschema. PyYAML remains a dependency and no longer touches a package document (§3.2) |
| Scope | The loader, the five schemas, the four spec registries, the closure check, and the composition root |

---

## 1. Scope

This document turns [`spec.md`](spec.md) into files, classes, and interfaces.
**It adds no requirements.** Where it makes a choice the spec left open, the
choice is stated here; where implementing the spec exposed a contradiction in
it, §11 says so rather than papering over it.

The spec's 15 system-level acceptance criteria are the definition of done. §9
maps every one of them to a named test, or to the module design that owns it.

**This document specifies interfaces, not bodies.** A method appears as a
signature and a sentence of semantics; a body appears only where the ordering of
steps *is* the design decision — which here is `load_package` (§3.6) and the
closure pass (§6.3).

### 1.1 What this document covers

The parts of the system that have no other home, because they are what every
component shares:

- **The loader** — scan, parse, validate, admit. §3.
- **The five JSON Schemas** — the spec of the spec. §4.
- **The four spec registries**, and why they stay four. §5.
- **The closure check**, which is a pass rather than a component. §6.
- **The composition root**, extending the existing one. §7.

### 1.2 What it defers

Each module's own `docs/design.md`. **All seven now exist**; this document was
written when none did, and every deferral it made has since been taken up:

| Deferred | To | Since |
|---|---|---|
| What a handoff kind's schema actually declares, and storage | `handoff` | **rev. 2** |
| The `Validator` protocol, composites, the phase runner | `validator` | **rev. 2** |
| Subgraphs, validation phases, task-owned transitions | `task_graph` | **rev. 12** |
| Backends, the SDK adapter, the two interface levels | `agent` | **rev. 4** |
| The closure's own query helpers | `closure` | **rev. 3** |
| Isolation, domains, sync | `env_mgr` | **rev. 2** |
| The demo package and its CLI | `demo` | **rev. 3** |

**Four of the seven reach back into this document**, which is why it is at rev. 2
rather than rev. 1: `task_graph` put a graph-level load pass in the composition
root, `closure` moved the closure pass there too and gave the reason (§3.6),
`handoff` registers two stores nothing here listed, and `demo` measured that
`cli/` must be an installed package.

**The composition root is therefore not this document's alone**, and pretending
otherwise is what let four documents each write a different half of it.
[`interfaces.md`](interfaces.md) §2 is now the single normative listing; §7 below
shows the spec layer's contribution to it and defers the whole to that file.

This document specifies the *mechanism* those seven use. Where it names a
schema key, it names it as an example; the authority is the module's spec.

---

## 2. Layout and import graph

`agent_sys/` holds independent top-level packages. Two exist; this stage adds
five. Nothing importable sits at the `agent_sys/` top level, and that stays
true.

```
agent_sys/
├── pyproject.toml          declares every package; ruff and pytest settings
├── docs/
│   ├── spec.md
│   └── design.md           this document
├── env_mgr/                shipped — not this document's subject
├── task_graph/             shipped — not this document's subject
├── spec_loader/            NEW. The mechanism every registry shares
│   ├── __init__.py
│   ├── errors.py           SpecNotFound, SpecInvalid, SpecInconsistent
│   ├── protocols.py        the vocabulary: Problem, SpecDocument, the Protocols
│   ├── yaml_source.py      text -> a tree that still knows its line numbers
│   ├── variables.py        ${...} expansion over that tree, in place
│   ├── assets.py           AssetIndex — a body found by filename convention
│   ├── package.py          YamlPackage: scan, discriminate, emit. Reads source
│   ├── validate.py         doc -> [Problem]. Takes no path and no bytes
│   ├── access.py           accessors over the vocabulary, so nobody re-derives
│   ├── bundled.py          the schemas, as a `referencing` Registry
│   ├── report.py           Problem -> a line a human can act on
│   ├── registry.py         SpecRegistry — the shared base of the four
│   └── schemas/
│       ├── handoff.schema.json
│       ├── validator.schema.json
│       ├── task.schema.json
│       ├── agent.schema.json
│       └── closure.schema.json
├── handoff/                NEW package; docs/ already exists
├── validator/              NEW package; docs/ already exists
├── agent/                  NEW package; docs/ already exists
├── closure/                NEW package; docs/ already exists
├── demo/                   NEW. The installed runner — cli, build, stream, renderers
├── general_specs/          workflow-independent specs (spec §4.5)
├── examples/demo/          the demo task package — YAML and data, NOT a Python
│                           package and imported by nobody (demo design D2, §3)
└── tests/
    ├── env_mgr/
    ├── task_graph/
    ├── spec_loader/
    ├── handoff/  validator/  agent/  closure/
```

### 2.1 Why one package per module

Spec §4.1 says the four registries are deliberately separate. A directory
listing is where that is cheapest to see and hardest to erode: unifying four
packages is a visible refactor, whereas merging four classes in one file is a
quiet afternoon.

It also matches what is already here. `env_mgr` and `task_graph` are flat
top-level packages, each owning its `docs/`; `handoff/` and friends join that
pattern rather than introducing a second one.

### 2.2 The schemas live inside `spec_loader/`, and that is forced

[`../README.md`](../README.md)'s projected layout puts `schemas/` at the
`agent_sys/` top level. **That is not installable as written**, and it was worth
checking rather than assuming:

```python
>>> find_packages(where=".", include=["env_mgr*","task_graph*","agent_sys_helper*"])
['env_mgr', 'task_graph', 'env_mgr.installers']
```

A bare directory of `.json` files is not a package — it has no `__init__.py`, so
`find_packages` cannot see it, and setuptools will not install it without a
`package_data` rule. Anything reading it would be reaching outside the installed
distribution at a path that exists in a git checkout and not in a wheel.

So the schemas sit at `spec_loader/schemas/` and are read through
`importlib.resources.files("spec_loader") / "schemas"`, which works identically
from a checkout, a wheel, and a zipimport. `pyproject.toml` gains
`"spec_loader*"` plus the four module packages in `packages.find.include`, and a
`package-data` entry for `*.json`.

**This is a correction to the README's projection, not to the spec** — spec §4.3
says only that this repository holds the schemas, which it still does.

### 2.3 Import graph

```
                        spec_loader/     (ruamel.yaml, jsonschema, referencing)
                             ▲
        ┌──────────┬─────────┼─────────┬──────────┐
    handoff    validator   task*     agent     closure
        ▲          ▲                   ▲          ▲
        └──────────┴─────────┬─────────┴──────────┘
                             │
                    bootstrap (§7) — the only module importing all of them

    task_graph/  ──── imports nothing above. Nothing above imports it either,
                      except through the Registry, by name, at use time.
    env_mgr/     ──── likewise.
```

`*` The **task** spec registry lives in `closure/`, not in a package of its own.
A task spec is not independently loadable — spec §2 of the closure document
declares it inside the closure, as the `task` key — so a `task/` package would
contain one registry and no other reason to exist. The four registries are still
four objects; three of them have their own package and one is homed with the
document that declares its contents. **Stated rather than hidden**, because a
reader counting packages will otherwise count four and find three.

**No module package imports another.** They resolve collaborators through the
`Registry` by name at call time, exactly as `task_graph`'s managers already do
([`../task_graph/docs/design.md`](../task_graph/docs/design.md) §2). That is
what keeps the graph acyclic without anyone maintaining it.

`spec_loader` imports nothing from this repository. It is the leaf, and it must
stay one: the moment it imports `handoff` to understand a handoff spec, the
"the loader does not interpret a package's content" claim stops being
structural.

### 2.4 Source-file conventions

`env_mgr`'s 19 modules carry an SPDX header; `task_graph`'s carry none. The new
packages follow **`task_graph`** — it is the closer sibling in age, style, and
subject. The inconsistency is pre-existing and is not resolved here; it is
recorded in §12 because someone should decide it once rather than per package.

---

## 3. The loader — `spec_loader/`

Spec §4.4 fixes the pipeline: a package delivers parsed documents, each is
validated, and only then is the spec admitted. This section turns that ordering
into module boundaries, because the ordering is the design.

```
 package.py               validate.py            registry.py
┌──────────────┐   doc   ┌──────────────┐  doc  ┌──────────────┐
│ scan · parse │────────►│   validate   │──────►│ name -> spec │
│ substitute   │         └──────────────┘       └──────────────┘
│ discriminate │                ▲                       ▲
└──────────────┘        never sees a path       never sees a file
       ▲                     or bytes
 the only part that
 touches source
```

**Rev. 2 had a `render.py` in front of that and it is deleted**, along with
jsonnet, `ImportResolver`, `SpecSource` and `DirectoryPackage`. Spec §4.4 rev. 10
carries the measurement: the templating language was carrying constants, string
concatenation, and default-if-absent, and nothing else. §3.2 below keeps the
findings from that module that still teach something, marked as history, and says
of each whether it transfers.

### 3.1 The boundary is a type signature, not a promise

**`validate` takes a parsed document. It has no parameter through which a path
could reach it, and since rev. 3 not even bytes.**

```python
def validate(doc: Any, schema: Mapping, *, origin: str) -> list[Problem]:
    """Check `doc` against `schema`. `origin` is a label for messages —
       an opaque string, never opened."""
```

This is the one place where a design choice makes a specification claim
*enforced* rather than *asserted*. Spec §4.4 says the loader "does not read,
audit, or constrain" how a package produced its documents; with this signature, a
future maintainer who wants to cannot do it by accident.

**Rev. 3 strengthened it and the change is worth noticing rather than skimming.**
Rev. 2's signature was `validate(data: bytes, …) -> tuple[Any, list[Problem]]`:
no path, but it still parsed, so this module still had an opinion about the source
*format*. It does not any more — parsing moved to the package (§3.2), the return
type lost the document because the caller already has it, and the claim went from
"cannot see where it came from" to "cannot see what it was written in".

The precedent is Grafana's dashboard validator, whose entire input is
`data []byte` reached via `json.Marshal` — there is no channel through which
provenance could arrive. Helm's `KubeClient.Build(reader io.Reader, …)` and
Kustomize's validator-plugin contract (resources arrive **on stdin**, and the
plugin cannot modify them) are the same shape.

Two honesty notes, because the spec should not overclaim:

- **No maintainer anywhere states this as a principle.** The support is
  convergent — function signatures, Kubernetes' *Declarative Application
  Management* ("embedded code renders the configuration unparsable by other
  tools"), Kustomize's *Eschewed Features* ("it's no longer *data*, it's now
  logic that must be compiled"), and Helm #1463, where source-level template
  analysis was **tried and abandoned** because `{{...}}` fragments cannot be
  executed standalone. It is our principle, with convergent support.
- **OPA is a counter-example and is not cited as support.** Its bundles retain
  rego source and the runtime recompiles on every activation, so its *manifest*
  validation has to open the source AST — a module's data path lives in its
  `package` declaration. It is a demonstration of the failure mode: an
  artefact-level invariant leaks back into the source language when the artefact
  keeps source.

**A note the deletion earned.** Both Kustomize quotations above are arguments
against embedding logic in configuration. Rev. 2 cited them while adopting a
templating language, which was defensible — a template is data templating, not
embedded code (spec §7.1) — and is simply no longer a tension worth managing.

`origin` is a plain string used only in messages. cdk8s solves the same problem
the same way — an out-of-band sidecar joined to violations *after* the plugin
returns, degrading to `'N/A'` when absent. Since rev. 3 it carries a path **plus
an optional JSON pointer**, `steps/collect.yaml#/2`, because one file may hold
many objects; it is still never opened.

### 3.2 `yaml_source.py` — text to a tree that knows where it came from

`render.py`'s successor, and much smaller: no runtime seam, no fallback binding,
no thread pool. It keeps the one property the old module had for a different
reason — **a fault names a place**.

```python
def read_yaml(path: Path, *, origin: str) -> tuple[Any, list[Problem]]:
def position_of(node: Any, key: str | int | None = None) -> Position | None:
```

`ruamel.yaml` in **round-trip mode**, which spec §7 adopts. The choice rests on
four facts, and `spec_loader/README.md` carries the probes; the fourth is what
makes this a thin wrapper rather than a layer:

| | |
|---|---|
| Positions survive nesting | `lc.item(i)`, `lc.key(k)`, `lc.value(k)`, all 0-based, so everything here adds one. An inline definition has no file of its own, so a diagnostic that could only name a file would name the wrong thing |
| Syntax errors carry `problem_mark` | line and column, 0-based |
| **Duplicate keys are rejected**, with a position | PyYAML's `safe_load` silently keeps the last |
| **The tree is `dict` and `list`** | `CommentedMap` / `CommentedSeq` subclass them, so `jsonschema` validates the position-carrying tree directly and `err.json_path` is correct. Nothing is converted, so nothing is lost on the way to `validate` |

**Two traps that jsonnet used to close, and how they are closed now.** Rev. 2's
§8 argued that "`norway: NO` parses as `False` under PyYAML; jsonnet always
quotes strings, so the rendered form does not", and that "jsonnet rejects
duplicate keys statically". **Both premises died with the render step**, and a
hand-written document has no upstream to lean on. Neither is avoided by
convention: `ruamel.yaml`'s round-trip loader is **YAML 1.2**, so `NO` is the
string `NO`, and duplicate keys raise. The library closed both, which is why this
module is the only place a package document is parsed.

#### What `render.py` measured, and which of it survives

Deleting a module is not licence to delete what measuring it taught. Three
findings, each labelled:

**1. jsonnet released the GIL, and cost a fixed ~23 ms per render — ARCHIVED,
and its conclusion is UNMEASURED here.** 8 threads × 200 renders took 0.95 s
against 6.98 s serial with all 200 results correct, and `{}` cost the same as a
2000-element tree, so the cost was per-call VM construction rather than warm-up.
That shaped rev. 2's whole module: 100 specs at ~2.3 s serial versus ~0.3 s
pooled is the difference between a loader a developer notices and one they do
not, so `load_package` rendered through a `ThreadPoolExecutor`.

> **None of it describes YAML parsing, and it must not be assumed to.** The
> jsonnet number was a *fixed* per-call cost, which is what makes parallelism pay
> at any package size; YAML parse cost is presumably proportional to document
> size and has **no VM to construct** — and that word "presumably" is the
> problem, because it has not been measured in this tree. `load_package` (§3.6)
> is serial today. The measurement that would settle it is the one rev. 2 ran,
> re-run against `read_yaml`: parse time versus document size, and pooled versus
> serial over a package of ~100 documents. Until then, "parallelise the loader"
> is a hypothesis with a dead experiment behind it.

**2. Import containment — the resolver is gone, the QUESTION SURVIVES**, and
§3.3 of rev. 2 is where the evidence was. `ImportResolver` existed so that a
containing resolver would be a substitution rather than a rewrite, and there are
no imports left to contain. But the underlying question — *may a package reach
outside its own root?* — is unchanged, because spec §4.3 still permits a relative
symlink from one package into another, and `_scan` still walks whatever the
filesystem presents. The evidence, kept because it took the research:

| System | What it does |
|---|---|
| Kustomize | resolves symlinks **then** enforces containment, and calls `LoadRestrictions` "an intentional security feature" |
| Helm | permits escape with a log line; CVE-2025-53547's advisory reads *"Helm warns of the symlinked file but did not stop execution due to symlinking"* |
| dbt | permits arbitrary local paths, and closed the resulting relocatability bug as `not_planned` |

This is `spec_loader` design **O3** and spec §7 records it as the one adoptable
idea left in Kustomize. **It is open, and the deletion moved it rather than
answering it** — from "which resolver does `render` take" to "what does `_scan`
refuse to walk", which is a smaller and better-placed question.

**3. Numbers were IEEE 754 doubles — CLOSED for integers by the format change,
unchanged for floats.** jsonnet rendered `12345678901234567890` as
`12345678901234567168`, and rev. 2's O4 concluded that any field needing an exact
large integer must be declared as a string. Measured on this machine against both
parsers, the same document:

```
big: 12345678901234567890   ruamel round-trip -> int  12345678901234567890  (exact)
                            PyYAML safe_load  -> int  12345678901234567890  (exact)
```

YAML integers are Python `int`, arbitrary precision. **The specific hazard O4
recorded cannot occur any more** and the workaround it prescribed is not needed.
Floats are still doubles — `0.1` arrives as a `ScalarFloat`, which subclasses
`float` — but that is true of JSON Schema `number` everywhere and was never a
property of this system. O4 is amended in §12 rather than deleted, because
"a previous revision required string-typed large integers" is the kind of rule
that outlives its reason if nobody writes down that it stopped applying.

### 3.3 `variables.py` — what replaced the `config` fill

Spec §4.4 rehomes the three things jsonnet was carrying. Two of them land here:
a package-level variable set, expanded over the parsed tree.

```python
def substitute(tree, variables, *, origin: str, path: str = "$") -> list[Problem]:
```

Three decisions, and the second is the one a reader will want justified.

**Expansion is in place.** A rebuilt tree would be plain `dict`s and `list`s, and
the `lc` positions would go with them — the same failure OmegaConf was rejected
for, reintroduced by hand. Assigning back into a `CommentedMap` preserves the
positions exactly; measured, before and after are equal.

**An unresolved reference is a fault, not a value left literal.** `${NOPE}/x.md`
left alone is a path that resolves to nothing later, in another module, with
nothing to say why. `examples/demo`'s own history holds that bug: an unfilled
value concatenated as `'' + "/leak.txt"` and produced a plausible absolute path
that demonstrated nothing.

**The values are caller-supplied, and there is no `vars:` block in a package.**
This is narrower than spec §4.4's table reads, and deliberately: measured across
all 21 jsonnet sources the tree replaced, every variable reference was to
`config.package_root`, `config.outside` or `config.inputs` — all run-level facts —
and **not one package declared a constant of its own**. A declaration block would
be a construct with no measured user, which is `engineer_principle.md` §2's "do
not put it anywhere yet". `${TASK_PACKAGE_ASSERT_DIR}` is added on top and cannot
be overridden: it is a fact about the package, not about the run.

### 3.4 `ruamel.yaml`'s errors need no adapter, and `_jsonnet`'s did

Rev. 2 had a section here on parsing `_jsonnet`'s output. It is worth one
paragraph of history, because the contrast is the argument for the library.

`_jsonnet` raised `RuntimeError` with a message beginning `STATIC ERROR:` or
`RUNTIME ERROR:`, and **two of four measured error classes carried no location in
the first line** — for a RUNTIME ERROR it was recoverable only from a trailing
tab-delimited stack, with no machine-readable output at all. That is
google/jsonnet#786, open since 2020 and raised by a Flycheck maintainer:
*"jsonnet only supports human-readable output"*. So `render.py` carried a parser
for error strings, and `max_trace` had to be raised from 20 to 40 because the
default truncated the stack that held the location.

`MarkedYAMLError.problem_mark` is a structured line and column, on every syntax
error probed. **The adapter is one `+ 1` for 0-based indexing**, in
`Position.from_ruamel`. A whole module's worth of string parsing became a
subtraction, and that is the clearest single statement of what the format change
bought.

### 3.5 `validate.py` and `report.py` — error reporting

```python
@dataclass(frozen=True)
class Problem:
    origin: str          # the label passed in; never opened
    path: str            # JSONPath: "$", "$.items_schema", "$.validators[0]"
    keyword: str         # the failing JSON Schema keyword
    message: str
```

`path` is `ValidationError.json_path`, which is supported API and emits
JSONPath (`$.spec.cmd`) rather than an RFC 6901 pointer.

**The report format is `check-jsonschema`'s, adopted whole**:

```
<origin>::<path>: <message>
Best Match: <path>: <message>
Best Deep Match: <path>: <message>          (only when it differs)
N other errors were produced. Use --verbose to see all errors.
```

Adopted rather than invented because the project shipped a **second** heuristic
after finding stock `best_match` insufficient — `_deep_match_relevance`,
maximising `len(absolute_path)` over a flattened recursive walk, where stock
`relevance` minimises depth at the top level and then descends. Two guesses plus
a verbose escape is the state of the art; a third guess is not an improvement.

**This is not theoretical for us.** Measured against a schema shaped like
`handoff`'s:

| Checking `items_schema` by | `{"type": "nonsense"}` | `"notaschema"` |
|---|---|---|
| `$ref`-ing the 2020-12 metaschema | 1 clear error | **8 identical errors** |
| `check_schema` as a named step | `SchemaError: 'nonsense' is not valid…` | `SchemaError: 'notalist' is not of type 'array'` |

The metaschema's `anyOf` branches each fail the same way, and a package author
reading eight copies of `is not of type 'object', 'boolean'` learns nothing.
Both `best_match` and dedupe-by-`(path, keyword, message)` collapse it to 1;
`best_match` picked the correct error in every case measured.

**Rule for the schemas (§4): a field carrying a nested user-supplied schema is
typed loosely and checked by `check_schema` as a named load-time check.** Not by
`$ref`. The `$ref` form works and produces a message nobody can act on.

`report.py` is separate from `validate.py` so a machine-readable emitter is a
second function over the same `Problem` list, not a flag threaded through the
validator.

### 3.6 `package.py` — loading one task package

The one place where the ordering is the design, so it appears as a body. **Two
steps and no third**, which is what rev. 2's five became:

```python
def load_package(pkg: TaskPackage, registries: Registries) -> LoadReport:
    # 1. ask the package for its documents. No path, no format, no branch on
    #    where one came from — main spec criterion 4 as code (§3.1). The
    #    package's own faults arrive alongside them and are carried through.
    contents = pkg.documents()
    problems = list(contents.problems)

    # 2. validate each against its kind's schema. Collected, not raised: one
    #    broken spec must not hide the other nine. A schema fault is stamped
    #    with the DOCUMENT's line, not the field's — see below.
    validated = []
    for document in contents.documents:
        errs = validate(document.doc, schema_for(document.kind), origin=document.origin)
        if errs:
            problems.extend(_located(errs, document))
        else:
            validated.append(document)

    # 3. admit. A duplicate name is an error here, not an overwrite (§5.2), and
    #    a kind's own load-time checks live in its registry subclass and raise.
    #    Caught as `ValueError` — see below, it is a repair.
    admitted = []
    for document in validated:
        name = document.doc["name"]
        try:
            registries.for_kind(document.kind).add(name, document.doc, origin=document.origin)
        except ValueError as exc:
            problems.append(Problem(origin=document.origin, path="$",
                                    keyword=_keyword_of(exc), message=str(exc),
                                    line=document.line))
        else:
            admitted.append(name)

    # 4. NO closure pass here. It runs once, at the composition root, after
    #    every package is loaded — §7, and closure design D3 for why.
    return LoadReport(admitted=tuple(admitted), problems=tuple(problems))
```

**Rev. 2's steps 1 and 2 — discover, then render in parallel — are one call to
`pkg.documents()`.** That is the seam change: scanning, parsing, variable
substitution, discrimination and ordering all moved *inside* the package, and
what crosses is `PackageContents`, a pair of document and problem tuples. The
function that used to know about files knows about neither files nor bytes.

Two properties of the code that are not obvious from the shape, both first-hand
from `spec_loader/package.py`:

**A schema fault carries the document's position, not the field's**, and the
limit is stated rather than papered over. `validate` reports a field as
`json_path`; joining that onto a source position needs the parse tree, which
`validate` deliberately cannot see (§3.1). So a diagnostic says *"this object,
which starts at line 12, has a bad `$.body.entry`"* — enough to find in a file
holding twenty objects, and not a guess.

**Step 3 catches `ValueError`, not `(SpecInvalid, SpecInconsistent)`, and that is
a repair rather than a widening.** `BaseSpecRegistry` documents those two as what
a subclass raises; measured against the four real registries, `validator` raises
`ValidatorInvalid` — a `ValueError`, but neither of them. It escaped, and one
package's choice of exception type aborted the whole multi-package load: "collect,
do not raise" became "die on the first", silently, for every other package in the
run. A contract four packages must remember is one this function should not
depend on. `TypeError` and `AttributeError` still propagate, because those are
bugs in a registry rather than rejections of a spec.

**Step 5 was in this function in rev. 1, and moving it out is the one correction
this revision makes to the loader.** `closure` design §7.1 found two defects, and
the second is the serious one. `check_closures` is `closure`'s, so calling it
here would make `spec_loader` import a module of ours — the leaf rule §2.3 calls
structural. And `load_package` runs **once per package** (§7), so with two
packages the pass would fire with the second package's specs in no registry —
exactly what §6.1 forbids in its own words, and main spec §4.3 makes
cross-package references a supported case.

Fixing the ordering fixes the import as a side effect: `bootstrap` already
imports everything.

Three properties of the remaining ordering, each with a reason:

**Failures are collected, not raised.** `check-jsonschema` goes further and
returns parse failure *as a value* so parse and validation errors travel
separate channels. A loader that dies on the first bad spec makes fixing a
package an N-round trip.

**Admission happens after all validation.** Half a package in a registry is a
state nothing else in the system knows how to reason about.

**Nothing cross-registry happens here at all.** §6 says why the closure pass
cannot, and the same argument covers `handoff`'s two-way binding check and
`validator`'s separation check: each needs a registry this call may not have
filled yet.

### 3.7 `Registries` — the read-only view the passes take

Rev. 1 used this name three times and defined it nowhere; `closure` design §7.3
worked out the shape the six closure checks need. It is defined **here**, because
`load_package` takes one and `spec_loader` may not import `closure` (§2.3).

```python
class Registries(Protocol):
    """A read-only view over the five spec registries. Nothing on it mutates."""
    handoff_specs:   SpecRegistry
    validator_specs: SpecRegistry
    task_specs:      SpecRegistry
    agent_specs:     SpecRegistry
    closures:        SpecRegistry

    def for_kind(self, kind: str) -> SpecRegistry: ...
```

A `Protocol` rather than a class, so a test supplies five dicts and no
`bootstrap`.

**`closure` §7.3 also wanted a `handoff_report` field on it**, for check 3's
escape-hatch intersection. It is **not** here, and the reason is the leaf rule
again: that field is typed `handoff.LoadReport`, so putting it on this Protocol
would make `spec_loader` name a type in `handoff`. The report is passed to
`check_closures` as its own argument instead —
[`interfaces.md`](interfaces.md) §3.2 carries the signature.

---

## 4. The five schemas

`spec_loader/schemas/*.json`, hand-written, Draft 2020-12, one per object plus
the closure's.

### 4.1 Hand-written, and no pydantic spec models

The tempting alternative is a pydantic model per kind with the schema generated
from it. Measured: `model_json_schema()` on a model with `extra="forbid"`, a
`Literal` of one, and `Field(description=...)` emits exactly

```json
{"additionalProperties": false,
 "properties": {"kind": {"const": "reproducible", "description": "…"}},
 "required": ["kind", "name"]}
```

— `const`, `additionalProperties: false`, and `description`, the exact three
mechanisms spec §4.4 names. So the option is real. It is **not taken**, for
three reasons in descending order of weight:

1. **A generated model does not enforce what a schema enforces.**
   `datamodel-code-generator` publishes `UNSUPPORTED_SCHEMA_KEYWORD_REASONS`
   naming the keywords whose semantics generated pydantic models do not
   represent — `if`/`then`/`else`, `not`, `unevaluatedProperties`,
   `propertyNames`, `dependentRequired`, `patternProperties`, `uniqueItems`. A
   model built from such a schema **accepts instances the schema rejects**, and
   spec §4.4 makes the schema the *only* enforcement point.
2. **The schema is the artefact a package author reads.** Spec §4.4 calls it "a
   better artefact than the source for the person deciding whether this is the
   kind they want" — and since rev. 3 collapsed tier ② (§4.5) it is the *only*
   place a kind's shape is written down. A build product is a worse contract
   than a maintained file.
3. **It removes the double-validation question entirely.** Four of the six
   surveyed loaders — `check-jsonschema`, `openapi-spec-validator`, `prance`,
   `pre-commit` — validate against a schema and never build a typed object.

**So a spec is a plain `dict` throughout**, and access is `spec["content_type"]`
rather than `spec.content_type`. That cost is real and is accepted; §12 records
the alternative (`pre-commit`'s field projection into a container structurally
incapable of re-validating) as the escape hatch if it becomes painful.

One trap this avoids, worth stating because it is the obvious shortcut: JSON
Schema checks structure, while pydantic additionally **coerces**. Validating
with JSON Schema and then calling `model_construct` gives typed fields still
holding raw JSON strings — a silently wrong object, not a faster one. No
production instance of that pattern was found in the survey.

### 4.2 The idiom, with its measured messages

| Intent | Expressed as | Rejection message, measured |
|---|---|---|
| A field a template must not change | `const`, or a tight `enum` | `'reproducible' was expected` |
| A field nobody may add | `additionalProperties: false` | `Additional properties are not allowed ('sneak' was unexpected)` |
| A field that must be present | `required` | `'name' is a required property` |
| What a field is for | `description` | — it is the documentation |

These are spec criterion 3's exact strings, verified against `jsonschema`
4.26.0.

### 4.3 A nested user-supplied schema is checked, not `$ref`-ed

`handoff.items_schema` is a JSON Schema supplied by a package. Its outer
declaration is `{"type": "object"}` and its validity is a **named load-time
check** calling `Draft202012Validator.check_schema`, for the error-quality
reason §3.5 measured.

This is the openapi-spec-validator shape: a declarative pass, then a registry of
hand-written checks for what the declarative layer cannot express well. Each
module's spec already lists its own such checks — `handoff` §8 has five,
`validator` §9.3 has five, `closure` §4 has six — and they run in step 3 of §3.6.

### 4.4 Two properties the pipeline used to get for free, and now buys

Rev. 2 recorded both of these as free consequences of rendering, and closed with
*"a hand-written YAML spec would not have that property"*. **Rev. 10 of the spec
made every spec hand-written**, so this section is the one place in the design
where the deletion took something away rather than simplifying. Both were
recovered, by choosing the parser rather than by convention (§3.2):

| Property | Rev. 2: free, because | Rev. 3: bought, by |
|---|---|---|
| No YAML 1.1 type coercion | jsonnet always quotes strings, so `norway: NO` could not reach a YAML loader unquoted. Verified as `json.loads(rendered) == yaml.safe_load(rendered)` over `null`, `true`, `1e3`, `"yes"`, `"NO"` and a large integer | `ruamel.yaml`'s round-trip loader is **YAML 1.2**. `NO` is the string `NO`, `on` is `on`, `12:30` is not 750 |
| Duplicate keys rejected | `STATIC ERROR: duplicate field: a`, where both `json.loads` and `yaml.safe_load` silently keep the last | `ruamel.yaml` raises `DuplicateKeyError`, **with a position** |

**The reason to state it as a purchase rather than a fact.** A property that
holds because of a library choice can be lost by a later library choice, and the
argument that made it free is gone. `yaml_source.py` is the only module that may
parse a package document, and that exclusivity is what the two rows above now
rest on — not on anything about the format.

### 4.5 Where the three tiers land in this pipeline

Spec §2.3's third cut is the shape this whole document implements, and it is
worth naming here because the mechanism is spread across §3 and §4 and a reader
should be able to check it serves the model:

| Tier | Who writes it | Where it is enforced here |
|---|---|---|
| ① **Contract** | this repository | `spec_loader/schemas/*.json` — §4.1. Fillable by nobody |
| ② ~~**Template**~~ | — | **Gone at rev. 3.** Spec §2.3 collapsed three tiers to two: this was jsonnet, and it stopped existing rather than moving. Its customisation job is now ③'s, through the package's variable set (§3.3) |
| ③ **Declaration** | a task package, or `general_specs/` | `package.py` — §3.6, §5.5. Where the semantics enter |

**The ordering is what makes ① unevadable**, and it is a property of the
pipeline rather than a rule anyone enforces: the package produces its documents
first (§3.2, §3.3), the schema runs on what it delivered (§3.6 step 2), so
nothing a package does can reach past it. The collapse of ② did not weaken this
— ① was always downstream of everything, and there is now less upstream. Spec §2.3 says the same thing from the other side — "a field
sealed here is sealed for good … no amount of cleverness upstream evades it".

**One claim of spec §2.3 this pipeline only partly delivers, and it is worth
being plain about.** §2.3: "Nothing runs until all three are present for all
four objects. **A missing ③ is an unfilled template and a load error**, not a
run-time surprise."

That holds when the field is *absent*: `required` rejects it, with
`'name' is a required property` (§4.2). It does **not** hold when the author
writes a placeholder — `name: TODO`, `description: ""` — because the document is
then structurally valid and the loader admits it. Nothing in the pipeline
distinguishes a filled field from a filled-in-with-nothing one. **The gap
survived the tier collapse unchanged**, and is arguably more exposed: there is no
template layer left to leave a field absent in the first place, so every
placeholder is now written by hand into the declaration.

This is the failure module 2 measured on Hugging Face model cards: a card whose
entire prose is `[More Information Needed]` returns HTTP 200, and that exact
string appears in 636,321 repositories. **A presence check cannot tell a value
from a placeholder.** Whether a template may carry placeholder defaults at all —
the cheap structural fix — is a **spec** question, and O9 records it.

---

## 5. The four spec registries

### 5.1 Four objects, one shared base

One registry per kind: `handoff/registry.py`, `validator/registry.py`,
`closure/task_registry.py`, `agent/registry.py`. Each is a `SpecRegistry`
subclass adding its own load-time checks and its own indexes — the validator
registry maintains the two-way binding and the used-by index (validator spec
§9.3), the handoff registry maintains the reverse index from validator to kinds
(handoff spec §8), and neither needs the other's.

Spec §4.1 accepts the cost of four places to change when the loading mechanism
changes, and names the fix if it hurts: "a shared *loader* the four registries
call — not a shared registry". `SpecRegistry` is that shared loader-facing base.
It holds the dict, the collision policy, and the error shape; everything a kind
does differently is in the subclass.

### 5.2 Duplicate registration is an error

```python
class SpecRegistry:
    kind: str

    def add(self, name: str, spec: Mapping, *, origin: str) -> None:
        """Admit a spec. Raises SpecInconsistent on a name already held by a
           different spec; a byte-identical re-registration is a no-op."""

    def get(self, name: str) -> Mapping:
        """Raises SpecNotFound, naming the kind, the name, and the candidates."""

    def names(self) -> list[str]: ...
```

`fsspec`'s shape: **error by default, identical re-registration a no-op**, and
an explicit opt-in if one is ever needed rather than a silent overwrite. The
alternative is on record as a mistake — Great Expectations' registry logs
`"Overwriting declaration"` and proceeds, and Inspect AI's assigns with no check
at all.

**The reverse collision is rejected too**: the same spec admitted under two
names. `pluggy` does this and the reason transfers exactly — for them it
silently doubles hook invocations; for us one validator under two names would
run twice and record two verdicts against one handoff version.

### 5.3 This is the opposite of the component `Registry`, deliberately

`task_graph`'s `Registry.register` **overwrites on purpose**, and
[`../task_graph/docs/design.md`](../task_graph/docs/design.md) §4 says why:
spec §4.1 requires that a test can swap an implementation after wiring, and
rejecting duplicates would forbid exactly that.

Two objects, two jobs:

| | Component `Registry` | `SpecRegistry` |
|---|---|---|
| Holds | live components | admitted specs |
| Keyed by | a wiring name (`"handoff_mgr"`) | a spec name (`"collect_trace"`) |
| Duplicate | **overwrite** — the swap mechanism | **error** — two specs claiming one name |
| Written by | `bootstrap`, and tests | the loader, once per package |

Stated as a table because the natural instinct on seeing two registries is to
unify them, and the collision policies are irreconcilable.

That instinct is worth answering once more generally. **Every canonical "one
generic registry" turns out to be N typed containers.** Kubernetes'
`runtime.Scheme` — the textbook example — is one struct holding seven
separately-typed maps with three key types and **three different collision
policies**: `AddKnownTypeWithName` panics, `AddTypeDefaultingFunc` silently
last-wins and documents it, `AddFieldLabelConversionFunc` overwrites
unconditionally. Airflow's `ProvidersManager` is ~25 dicts and sets, first-wins
with a warning for four kinds and silent for six.

So a generic registry cannot hold *one* policy; it grows N of them
inconsistently. That is a sharper form of spec §4.1's argument and it is
evidence rather than assertion.

The recorded harm is the kind that is expensive to diagnose: a `GroupVersion`
constant typo in cluster-api-provider-aws surfaced as
`panic: Double registration of different types…` three frames from its cause,
fixed by a one-line diff (#1211/#1212); the panic is undocumented (k8s#138028,
open); ~30 registration call sites are annotated
`// ERROR RESULT IS IGNORED BELOW` (k8s#51457, frozen). dbt's shared keyspace
produced the user-visible `"dbt found two docss"` (dbt-core#5352).

### 5.4 Errors enumerate the candidates

```
SpecNotFound: no handoff kind named 'trace_getter'
  (have: collect_trace, deploy_config, kernel_patch, trace_analysis)
```

pytest sets the bar — `fixture 'x' not found` followed by every available
fixture and how to list them — and dbt prints `expected one of {sorted(...)}`.

**This is already a repository convention**: `env_mgr/registry.py` raises
`KeyError(f"unknown installer {name!r} (have {sorted(REGISTRY)})")`. The new
registries follow the local precedent, which happens to be the surveyed best
practice.

`task_graph`'s component `Registry.get` says only
`no component registered as 'handof_mgr'`. Adding a candidate list there is a
one-line improvement to shipped code and is **not made in this document** — it
is a `task_graph` design change, and §12 records it for module 4.

### 5.5 Disk-driven loading, and the failure mode it avoids

Specs are admitted by the loader reading files, never by import side effects.

Worth naming because the alternative is common and its failure is documented.
Great Expectations registers in a metaclass, so a class exists in the registry
only if something imported it; their own docstring concedes *"we need to hope
that core Expectations are imported somewhere in our import graph — if not, our
registry will be empty"*, and the fix is a JIT force-import that counts entries
before and after.

Inspect AI's shape is the one to know if a decorator API is ever wanted
alongside: one registry, and loading from disk is *import-then-read* —
`create_file_tasks` imports the module, firing the decorators, then reads the
registry. **The loader never writes to the registry.** One writer, two entry
paths. Nothing here needs it yet.

---

## 6. The closure check is a pass, not a component

Spec §1.1 of the closure document is emphatic that a closure does nothing at
runtime. This section says where its *load-time* check lives, which the spec
leaves open.

### 6.1 It runs after all four registries are fully loaded

Not during admission. The reason is structural: closure spec §4 check 2 requires
that every handoff kind the task names resolves, and handoff spec §5.1 requires
that the **two-way binding agrees** — a handoff kind names its validators, a
validator names its kinds, and a mismatch crashes. **Resolve-during-load cannot
see the far side of a binding**, because the far side may not be loaded yet.

The shape is dbt's `ManifestLoader.load()`: parse everything, rebuild the lookup
index, *then* run named whole-graph passes (`process_refs`,
`check_valid_access_property`, …). OPA checks cross-bundle root overlap **before
any mutation**, in the same transaction.

### 6.2 Three rules for the report

**Name both sides, sort for determinism, and add a hint.** OPA blames the pair
symmetrically and sorts the names (`util.KeysSorted`) so the message is
reproducible — and OPA#7806 is a user who *still* could not tell which bundle
was misconfigured, with two PRs merged in response. Symmetric blame is correct
and insufficient.

```
SpecInconsistent: handoff kind 'trace' and validator 'check_trace_shape' disagree
  handoff/trace.yaml            validators: [check_trace_shape, check_trace_cov]
  validators/check_trace_shape.yaml      inputs: [trace_v2]
  hint: 'check_trace_shape' binds to 'trace_v2', not 'trace'. One of the two
        was renamed and the other not.
```

**A layering gate: skip the closure check for a spec that already failed its own
checks.** Kubernetes CRD validation does exactly this — CEL rules are skipped
when the schema itself failed, because the alternative is *"CEL validation error
messages that are not actionable"*. A closure reporting "your task's handoff
kind does not resolve" when that kind failed its own schema is noise on top of
the real error. This is the `skip=` argument in §3.6 step 5.

**Two exception classes, not one.** JPMS separates `FindException` ("not found")
from `ResolutionException` ("found, but inconsistent"), and the distinction is
load-bearing here: a missing validator is a typo or a missing file, while a
two-way mismatch means **one of two records is lying and nobody knows which**.

```python
class SpecNotFound(LookupError): ...      # a name does not resolve
class SpecInvalid(ValueError): ...        # a spec failed its schema or own checks
class SpecInconsistent(ValueError): ...   # two specs that both loaded disagree
```

### 6.3 What it checks, and what it cannot

The six checks are closure spec §4's, unchanged. Check 6 — that a task's
permissions cover its handoffs — is the one only this pass can perform, because
it needs the task's handoffs and its permissions together and neither registry
sees both.

`closure` spec §4.1 explicitly defers graph-level composition — cycles,
reachability, whether every input has a producer. This design does not
smuggle a partial version in. `task_graph` spec §3.2.4 has since given that
check a concrete reason to exist (`replace_with`'s containment claim depends on
it), which strengthens the case for a home; it is still not this pass's.

**No prior art was found for our exact check** — symmetric name-vs-name
agreement where disagreement crashes. Bazel visibility is set membership;
Kubernetes CEL is single-object by design; OPA and Cargo match in *error shape*
but check different things. The survey constrains how to report it, not whether
it is right. Recorded so nobody later assumes it was copied from somewhere.

---

## 7. The composition root

`bootstrap.build_registry()` already takes every component as an override
([`../task_graph/docs/design.md`](../task_graph/docs/design.md) §8.8). The spec
layer joins it there and nowhere else.

```python
    r.register("handoff_specs",   HandoffSpecRegistry())
    r.register("validator_specs", ValidatorSpecRegistry())
    r.register("task_specs",      TaskSpecRegistry())
    r.register("agent_specs",     AgentSpecRegistry())
    r.register("closures",        ClosureRegistry())
    for pkg in packages:
        report = load_package(pkg, views)     # per package; no cross-registry check
    problems  = check_closures(views, report, skip=...)   # ← once, all packages loaded
    problems += check_graph(r.get("task_specs"), skip=...)
    r.get("closures").freeze()                # closure design §8.2
```

**That is the spec layer's contribution and not the whole function.** Four other
documents also add to `build_registry` — `task_graph` design §8.8 (the managers,
the runner, the policy, the pools, the scheduler), `handoff` design §6.5 (two
stores), `agent` design §7.1 (the real runner), `env_mgr` (the environment
manager the runner resolves) — and rev. 1 of this document wrote its own half as
though it were the function.

**It is not, and the consequence was concrete**: `agent` design §7.1 resolves
`env_mgr` and the validator's phase runner by name, and no document registered
either. [`interfaces.md`](interfaces.md) §2 is the single normative listing; this
snippet is the spec layer's rows of it.

**The scheduler is registered after the packages are loaded**, which is not an
ordering constraint so much as a statement: a graph cannot be assembled from
specs that have not been admitted. Rev. 1 continued *"and the scheduler is what
assembles it"* — **that clause is withdrawn.** `closure` criterion 8 forbids the
scheduler reading a closure, and `closure` design D5 identified this sentence as
the only attribution of a job nobody owns. `Scheduler.submit` takes `Task`
objects a caller already built; who builds the root is
[`interfaces.md`](interfaces.md) §4.6, and today it is `cli/build.py`
(`demo` D3).

#### The graph pass lives here, and `task_graph` rev. 11 put it here

`check_graph` is `task_graph`'s (its design §8.7), and it runs at this line
because this is the only moment when every spec is present and nothing has run.
It carries the two graph-level checks `closure` spec §4.1 explicitly declines —
that no handoff produced inside a subgraph is consumed outside it, and that a
non-leaf declares no resources (`task_graph` criteria 50 and 53).

**§6.3 of this document said the check was "still not this pass's", and that
remains true**: it is not the *closure* pass. It is a separate pass, over task
specs rather than over closures, sequenced after both. The composition root is
where the two meet, and neither owns the other.

**Nothing in `task_graph` changes *because of the spec layer*.** The scheduler
resolves collaborators by name at use time and has no name for a spec registry;
it never acquires one. Spec criterion 10 requires `test_authority.py` to pass
unchanged with the spec layer present, and the reason it will is that there is
no edge from the scheduler to any of these five names.

That sentence is narrower than it first reads, and `task_graph` design §3.4
depends on the narrowing: **the prohibition is on the *scheduler*.** A `Task`
transition may resolve `closures` — `replace_with` does, to satisfy criterion 51
— and that adds no `Scheduler → spec registry` edge. `task_graph` rev. 11 also
changes plenty on its own account; what it does not do is acquire a spec name in
the scheduler.

`general_specs/` (spec §4.5) loads through the same call as an ordinary package
whose `config` is empty. The uniformity is the point — the main repository gets
no private path for its own specs — and it means the general specs are exercised
by the same code path a task package uses, on every run.

---

## 8. Build versus adopt

Per module, as the task definition requires. `README.md` carries the summary for
readers who do not open `docs/`.

| Module | Considered | Chosen | Why |
|---|---|---|---|
| ~~`render`~~ | Jinja2, Kustomize overlays, `rjsonnet`, `_gojsonnet` | **`jsonnet` — adopted at rev. 2, removed at rev. 3** | The module is gone (§3). Spec §7 rev. 10 carries the measurement: across all 21 sources the templating was constants, string concatenation and default-if-absent. Kept as a row rather than dropped, because the aarch64-wheel caveat and the two-runtime seam it forced (O2) are the concrete cost of the decision, and a table that only lists what is currently carried teaches nobody what a dependency costs |
| `validate` | pydantic-generated schemas, `fastjsonschema`, `cfgv` | **`jsonschema` 4.26** | Already installed. Spec §4.4 makes JSON Schema the only enforcement point, and §4.1 above records why a generated model is not a substitute. `fastjsonschema` compiles to Python for speed we do not need and has weaker error objects — no `json_path`, no `context` tree, which §3.5 depends on |
| parse | `json.loads`, PyYAML `safe_load` | **`ruamel.yaml`, round-trip mode** | §3.2. Rev. 2 chose `safe_load` on the argument that rendered JSON is a YAML subset and that the YAML loader kept the door open to a non-jsonnet source — the door it kept open is the one the system walked through, and the argument for `safe_load` did not survive it. Round-trip mode is what carries `lc` positions into a diagnostic, and it is YAML 1.2, which closes the `norway: NO` trap the render step used to close (§4.4). **PyYAML stays a dependency** — `env_mgr/recipe.py`, `handoff/verdict.py` and `handoff/store.py` use it — and no longer parses a package document |
| `report` | writing our own relevance heuristic | **`jsonschema.best_match` + check-jsonschema's four-part format** | §3.5. The project shipped a *second* heuristic after finding one insufficient; a third invented here would be worse than adopting both |
| `registry` | `pluggy`, entry points, one generic registry | **own, ~40 lines** | §5.3. `pluggy` solves 1-to-N hook broadcast with ordering and wrappers; a spec lookup is a dict with a collision policy. Its duplicate-rejection *behaviour* is adopted; its machinery is not. Entry points come later, if specs ever ship out of tree |
| `package` | dbt packages, Helm dependencies, Ansible collections | **own** | §3.2, §12. No surveyed system has a lockfile with content hashes, so there is nothing to adopt for the part that matters. What is adopted is the *shape* of the discovery-then-resolve pass |
| error types | `jsonschema`'s own exceptions | **three of our own** | §6.2. `ValidationError` cannot express "two specs that both validated disagree", which is the failure this system most needs to report well |

**Rev. 3 removes one runtime dependency and adds one.** jsonnet goes (with
`rjsonnet`, the fallback binding it forced); `ruamel.yaml` arrives, and spec §7
rev. 10 adopts it. Whether `pyproject.toml` reflects either is **not checked in
this pass** — O1.

---

## 9. Test plan

`pytest`. Tests live in `agent_sys/tests/spec_loader/` and the four module test
directories, each with an `__init__.py` so pytest's `prepend` import mode does
not put a test directory on `sys.path` and module basenames cannot collide
across packages — the same reason
[`../task_graph/docs/design.md`](../task_graph/docs/design.md) §11 gives.

Every test builds its own registries. Nothing is process-global.

### 9.1 Spec criteria, mapped

| # | Criterion | Test | File |
|---|---|---|---|
| 1 | The four objects are uniform | `test_four_kinds_instantiate_from_disk` | `tests/spec_loader/test_uniformity.py` |
| 2 | A schema violation is rejected at load, naming path and field | `test_rejects_with_path_and_field` | `tests/spec_loader/test_validate.py` |
| 3 | The schema is the only enforcement point | `test_const_override_rejected`, `test_undeclared_field_rejected` | `tests/spec_loader/test_schemas.py` |
| 4 | The loader never sees a package's source | `test_two_packages_same_document_indistinguishable` | `tests/spec_loader/test_package.py` |
| 5 | No workflow-specific spec in this repository | `test_repo_holds_only_schemas_general_and_demo` | `tests/spec_loader/test_repo_contents.py` |
| 6 | A package resolves without the loader knowing its layout | `test_cross_package_symlink_loads`, `test_dangling_symlink_names_path` | `tests/spec_loader/test_package.py` |
| 7 | The producer cannot reach the validation's context | — | **`env_mgr` design.** OS confinement, not a loader property |
| 8 | Every handoff kind resolves to ≥1 validator, or the flag is reported | `test_kind_without_validator_rejected`, `test_escape_hatch_is_reported` | `tests/handoff/test_registry.py` |
| 9 | A closure is well-formed or rejected | `test_closure_six_checks` | `tests/closure/test_check.py` |
| 10 | The scheduler never writes handoff state, never sees a validation | `test_authority.py` **unchanged** | `tests/task_graph/` — already green |
| 11 | The six-step reference workflow is expressible | `test_reference_workflow_loads` | `tests/closure/test_reference_workflow.py` |
| 12 | An agent's work is reconstructible after a restart | — | **`env_mgr` design** (§6.2 there) |
| 13 | Swapping the backend changes no other component | — | **`agent` and `demo` designs** |
| 14 | No isolation, no start | — | **`env_mgr` design.** CI-enforced there |
| 15 | A scripted bypass is blocked | — | **`env_mgr` design.** CI-enforced there |
| 16 | `assets/` is required of every package, and is the whole of the layout | `test_a_package_without_assets_fails_naming_the_root` | `tests/spec_loader/test_package.py` |
| 17 | No source format survives the deletion | `test_no_source_format_survives_the_deletion` | `tests/interfaces/test_import_rules.py` |
| 18 | `main.yaml` states that a package is runnable, and says what it is | `test_a_package_without_main_yaml_is_a_library_and_loads`, `test_a_main_yaml_declaring_no_task_is_an_entry_to_nothing` | `tests/spec_loader/test_package.py` |

Nine criteria are this document's (1–6, 16–18), three belong to a module registry
this document supplies the base for (8, 9, 11), one is already green (10), and
five are other modules' (7, 12, 13, 14, 15). Each of the five is named in the
table rather than omitted, so a reader can see that nothing is unassigned.

**Criterion 18's second clause is not in the table and that is deliberate**:
spec §8 states plainly that "a run has exactly one entry package" has no owner
and no test, and §10 there carries it. The two tests above cover what a *package*
can be asked; nothing yet can ask what a *run* is.

### 9.2 Tests beyond the criteria

Two properties are worth their own tests because they are measured facts a
future change could silently break:

| Test | Guards |
|---|---|
| `test_validate_takes_no_path` | §3.1. Asserts `validate`'s signature has no path-typed parameter — the boundary is the signature, so the signature is what is tested |
| `test_nested_schema_error_is_one_line` | §3.5. The `items_schema` case produces one actionable message, not eight |

**Rev. 2 listed a third, `test_render_is_parallel_and_idempotent`**, guarding
that a package rendered each source exactly once and that N sources did not cost
N × 23 ms serially. It is gone with the module, and **nothing replaced it**:
`load_package` is serial and no test asserts anything about load cost. §3.2 says
what would have to be measured before one is worth writing.

`test_validate_takes_no_path` deserves a note: it is a test about a type
signature, which is unusual. It exists because §3.1's whole claim is that the
boundary is structural rather than disciplinary, and a claim of that kind should
fail loudly when someone adds a convenience overload.

---

## 10. Implementation order

Test first, in dependency order. Each step is independently runnable and green
before the next begins.

| # | Module | Depends on |
|---|---|---|
| 1 | `errors` | — |
| 2 | `validate` + `report` | 1 |
| 3 | `schemas/` — the five files | 2 (they are tested through it) |
| 4 | `yaml_source` + `variables` + `assets` | 1 |
| 5 | `registry` — `SpecRegistry` | 1 |
| 6 | `package` — `YamlPackage` and `load_package` | 2–5 |
| 7 | the four subclasses, with their own load-time checks | 5, 6 |
| 8 | the closure pass | 7 |
| 9 | `bootstrap` extension | 6–8 |

Steps 1–5 are small and independent; the interesting work is 6–8. Step 3 carries
more than its size suggests — the schemas *are* the enforcement, so an
over-permissive one is a hole nothing downstream can close.

---

## 11. Deviations from the spec

Places where implementing the specification literally does not work. None
changes an acceptance criterion.

| # | Spec says | Design does | Why |
|---|---|---|---|
| D1 | `README.md` projects `agent_sys/schemas/` at top level | `spec_loader/schemas/`, read through `importlib.resources` | §2.2. A bare directory of `.json` is not a package: `find_packages` with the declared `include` does not see it and setuptools does not install it. Reading it by relative path works from a checkout and fails from a wheel. **This corrects the README's projection, not the spec** — §4.3 says only that this repository holds the schemas |
| D2 | Four independent registries (§4.1) | Four registry *objects*, in three packages — the task registry lives in `closure/` | §2.3. A task spec is not independently loadable; closure spec §2 declares it as the closure's `task` key. A `task/` package would hold one registry and nothing else. The four objects and their four collision policies are intact |
| D3 | "The loader resolves a path; it does not interpret a package's layout" (spec §4.3) | **Both halves changed at rev. 3 and the direction is opposite for each.** The loader resolves no path at all now — it is handed documents (§3.1), which is stronger than the sentence claimed. But spec §4.3 rev. 10 fixes two names, `main.yaml` and `assets/`, so *something* interprets layout; §4.3 rev. 11 places that on the package rather than on the loader | §3.2, §3.6. `ImportResolver` is gone with the imports; O3 restates the containment question against `_scan` |
| D4 | A spec is "a rendered YAML document" (spec §4) | It is a position-carrying `dict` — a `ruamel.yaml` `CommentedMap`, which subclasses `dict` — and the YAML text is never retained | §4.1. Nothing downstream reads the serialised form, and keeping it would invite someone to re-parse instead of using the loaded object. **Amended at rev. 3**: rev. 2 said "the rendered bytes are available to `report` while a load is in flight", and there are no rendered bytes. What survives instead is better — the *tree* carries `lc`, so a diagnostic gets a line without anyone holding the text (§3.2) |
| D5 | §4.6 says "a `ValidatorId` joins them on the same terms" as `TaskId` / `AgentId` / `HandoffId` | no `ValidatorId` exists in any design | **Reported, not resolved.** `grep -rn "ValidatorId"` over every design returns nothing, and the `validator` design identifies a validator by **name** throughout — its spec registry is keyed by name, and the implementation registry is joined to it by name. So the spec asks for a fourth typed id and three designs have quietly gone another way. The two are not equivalent: a name is a vocabulary entry that must be unique across a registry, an id is minted per object and survives renaming. Which one a validator needs depends on whether a validator is ever *instantiated* per run the way an `Agent` is — `validator` design §3.2 says it is not, which is the argument for names, but that argument is nowhere written down as a departure from §4.6. Whose change and which direction is a **spec** question |
| D7 | §2 of rev. 1 projected `demo/` as "docs only — the package itself is `examples/demo/`" | `cli/` is an installed Python package; `examples/demo/` is YAML and data and is **not** importable | `demo` design D2 measured it (M13): a console script pointing into an unpackaged directory installs successfully and dies with `ModuleNotFoundError` when run. Making `examples*` a Python package would fix that and break `demo` spec §1.1's rule that the demo uses nothing an out-of-repository package could use. The split is what makes that rule checkable |
| D8 | §3.6 of rev. 1 ran the closure pass inside `load_package` | it runs once at the composition root | `closure` design D3, adopted in full: `spec_loader` may not import `closure` (§2.3), and `load_package` runs once per package so the pass would fire before a second package's specs exist (§6.1). **This is the first correction another module's design made to this one that this document has accepted rather than merely recorded** |
| D6 | §7 is written as though this document's composition root is the last word on it | `check_graph` was added to it by `task_graph` design rev. 11, after this document was written | Recorded because the coupling runs *backwards* from the usual direction. This document defers module detail to seven module designs (§1.2); `task_graph` rev. 11 is the first of them to put something back — a pass that must run in the composition root, between package loading and scheduler registration, because that is the only point at which every spec is present and nothing has run. §7 now shows it. Expect the same from `env_mgr` and `demo`, and read §7's `build_registry` as a shared surface rather than as this document's alone |

---

## 12. New open questions

Found by this design, and **not** in spec §10.

| # | Question |
|---|---|
| **O1** | **Runtime dependencies were undeclared; the jsonnet half of the problem is gone and the rest is not verified here.** Rev. 2 measured (CPython 3.13.13) that `_jsonnet` 0.22.0, `rjsonnet` 0.5.6, `jsonschema` 4.26.0, `jsonpath-ng` 1.8.0, `jsonpointer` 3.1.1, `markdown-it-py` 4.0.0 and `rfc8785` 0.1.4 were all installed and none declared, while `python-jsonpath` — the one `handoff` design §11 chose — was not installed. **Rev. 3 changes two rows of that and re-measures none of the others**: the two jsonnet bindings are no longer imported anywhere (`tests/interfaces/test_import_rules.py::test_no_source_format_survives_the_deletion` enforces it, and passes), and `ruamel.yaml` is now a real runtime import from `spec_loader/yaml_source.py`. Whether `agent_sys/pyproject.toml` declares it, and whether the other five are still undeclared, is **not checked in this pass**. [`interfaces.md`](interfaces.md) §7 carries the declaration |
| **O2** | ~~**`_jsonnet` ships no Linux aarch64 wheel.**~~ **Closed at rev. 3 by deletion, not by resolution.** Four abi3 wheels — macOS arm64, manylinux x86_64, musllinux x86_64, win_amd64 — so ARM Linux fell back to an sdist build needing a compiler, and the fix was a runtime seam normalising `rjsonnet`'s `str` `import_callback` against `_jsonnet`'s `bytes` (as `kapitan`'s `select_jsonnet_runtime` does). It was never built, and `§3.2`'s `render` — the one place it would have gone — no longer exists. **Recorded rather than dropped, because it is the concrete cost of the adopt decision §8 now marks as removed**: a dependency with four wheels and a compiled extension forced a second binding and a seam, for a language nothing was using |
| **O3** | **Should the package scanner enforce containment?** §3.2. **Restated at rev. 3 and not answered**: there is no import resolver any more, and the question moved from "which resolver does `render` take" to "what does `_scan` refuse to walk" — a smaller and better-placed version of the same thing, since spec §4.3 still permits a relative symlink from one package into another. Spec §4.3 permits cross-package reference and says the loader does not interpret layout — the *permissive* branch, and the one with the CVEs. Kustomize resolves symlinks then enforces containment, calls it "an intentional security feature", and names the cost: it **breaks relocatability**. Helm permits escape with a log line — CVE-2025-53547, *"Helm warns of the symlinked file but did not stop execution"*. dbt permits arbitrary local paths and closed the resulting bug `not_planned`. **This is a specification question**, raised here because the design must not pre-empt it |
| **O4** | **Spec numbers: the integer hazard is closed, the float one is not this system's.** Rev. 2 recorded that jsonnet rendered `12345678901234567890` as `…567168`, and concluded that any field needing an exact large integer must be declared as a string. **Measured at rev. 3 over the same document**: `ruamel.yaml` round-trip and PyYAML `safe_load` both give Python `int`, exact. The workaround is not needed and no schema should adopt it. Floats remain IEEE 754 doubles — `0.1` arrives as `ScalarFloat`, a `float` subclass — which is true of JSON Schema `number` everywhere and was never special here. **Kept as a row rather than deleted**, because "a previous revision required string-typed large integers" is exactly the rule that outlives its reason if nobody writes down that it stopped applying |
| **O5** | **`Registry.get` does not list candidates.** §5.4. The new spec registries do, `env_mgr` already does, and `task_graph`'s component registry does not. It is a one-line change to shipped code and belongs to module 4's design, not to this document |
| **O6** | **SPDX headers are inconsistent.** `env_mgr`'s 19 modules carry one; `task_graph`'s carry none. The new packages follow `task_graph`. Somebody should decide it once for the repository rather than per package |
| **O7** | **Nothing yet sweeps a stale spec.** A registry is loaded once at composition and never invalidated. Reloading a package after editing a spec means rebuilding the registry, which means restarting the system. Acceptable for the alpha — spec §6 makes graphs static — but it interacts badly with a long-running system, and it is the first thing a developer editing a task package will hit |
| **O8** | **A typed accessor may be wanted later.** §4.1 chose plain dicts. If `spec["content_type"]` becomes a source of typos, the escape hatch with prior art is `pre-commit`'s: a field projection into a frozen dataclass that is structurally incapable of re-validating. Recorded so the answer is not "add pydantic models", which §4.1 rejected for a reason that will still hold |
| **O9** | **A placeholder passes every check a real value passes.** §4.5: `required` catches a field a template left *absent*, and nothing catches one a template filled with `"TODO"` or `""`. Spec §2.3 asserts "a missing ③ is an unfilled template and a load error", which is true only of the first case. Three candidate answers, none free: forbid defaults on a template's `config` keys, so an unfilled one is always absent — cheap, structural, and a **spec** change; add a placeholder denylist, which module 2 measured failing at scale (Hugging Face's own template emits `[More Information Needed]`, present in 636,321 repositories); or accept it and say so, on the grounds that a package author who writes `"TODO"` has been told. The first is the only one that makes the harmful case unrepresentable. Not decided |
