# `spec_loader`

| | |
|---|---|
| What | The loader, the five JSON Schemas, and the vocabulary every other package shares |
| Wave | 0, and W3+W4 of the user-interface stage. Six packages wait on `Problem` and `SpecRegistry` |
| Specified by | [`../docs/spec.md`](../docs/spec.md) §4.3–§4.5 · [`../docs/design.md`](../docs/design.md) §3–§5 · [`../docs/interfaces.md`](../docs/interfaces.md) §3, §4.1 |
| Imports of ours | **none, ever** |

```
  package.py / yaml_source.py        validate.py            registry.py
 ┌────────────────────────────┐    ┌─────────────┐        ┌──────────────┐
 │ *.yaml -> SpecDocument     │───►│ against the │───────►│ name -> spec │
 │ scan, parse, discriminate, │doc │ JSON Schema │  doc   └──────────────┘
 │ expand, substitute, order  │    └─────────────┘
 └────────────────────────────┘        ▲
     the only side that opens a file   never sees a path — and, since
                                       rev. 10, never sees a byte either
```

**The seam moved at rev. 10 of the main spec.** It was `SpecSource(path, kind)` —
one file, one object, kind claimed by the directory it sat in — and three of the
user-interface stage's five requirements break that shape at once: several
objects per file, kind claimed by a `module:` key, and inline definitions with no
file of their own. What crosses now is `SpecDocument`s, and `load_package` is
validate-and-admit. A second source format would be a second `TaskPackage`, not
a change to the loader.

---

## Done: the criterion-to-test mapping

`implementation-stage.md` §5.1 calls this *"the deliverable, not a formality"*.
`docs/design.md` §9.1 assigns main spec criteria **1–6** to this package; each
maps to a named test that exists and passes, under the name the design gave it.

| # | Criterion | Test | File |
|---|---|---|---|
| 1 | The four objects are uniform | `test_four_kinds_instantiate_from_disk` | `tests/spec_loader/test_uniformity.py` |
| 2 | A schema violation is rejected at load, naming path and field | `test_rejects_with_path_and_field` | `test_validate.py` |
| 3 | The schema is the only enforcement point | `test_const_override_rejected`, `test_undeclared_field_rejected` | `test_schemas.py` |
| 4 | The loader never sees a package's source | `test_two_packages_same_document_indistinguishable`, `test_load_package_opens_no_file` | `test_package.py` |
| 5 | No workflow-specific spec in this repository | `test_repo_holds_only_schemas_general_and_demo`, `test_the_scan_finds_something_to_check` | `test_repo_contents.py` |
| 6 | A package resolves without the loader knowing its layout | `test_cross_package_symlink_loads`, `test_dangling_symlink_names_path` | `test_package.py` |
| 16 | `assets/` is required of every package, and is the whole of the layout one must have | `test_a_package_without_assets_fails_naming_the_root`, `test_one_file_and_two_hundred_objects_are_equally_well_formed` | `test_package.py` |
| 18 | `main.yaml` states that a package is runnable, and says what it is | `test_a_package_without_main_yaml_is_a_library_and_loads`, `test_a_main_yaml_declaring_no_task_is_an_entry_to_nothing`, `test_a_task_written_inline_in_main_yaml_counts` | `test_package.py` |

**Criterion 4's test kept its name and was strengthened, not replaced**, and it
is worth saying which because the two are easy to confuse. It used to compare two
*jsonnet sources* — a plain object against a function-and-overlay pile — and
assert that rendering both produced one document; the property was an *ordering*,
"render happens before validate". Rev. 10 amends the criterion to *"the loader is
handed parsed documents and has no parameter through which a path could
arrive"*, so what varies in the test is now the **package's layout**: one
document per file under a directory per kind against every object inline in
`main.yaml`. `test_load_package_opens_no_file` is the structural half and could
not have been written before — `load_package` called `render`, which read every
source. It moved out of the deleted `test_render.py` into `test_package.py`,
beside criteria 6 and 16, which are the other two about what a package may and
may not vary.

**Criterion 5's guard went green and stopped guarding, and the second test is the
repair.** Its scan was `.jsonnet`-only, so when the format changed there was
nothing left to find: `stray` was a filtered subset of an empty set and
`assert not stray` held over it. Measured before repairing — a planted `.jsonnet`
still failed the test, a planted `.yaml` did not — so the guard was **aimed at a
dead format**, not dead, and the fix is to widen the scan while *keeping*
`.jsonnet` as criterion 17's tripwire rather than rewriting it.

Two things came out of it that generalise:

- **`assert not scan()` still works when the scan is empty; `assert not [p for p
  in scan() if pred]` does not.** Empty is the claim in the first and vacuous in
  the second. `test_no_source_format_survives_the_deletion` is the first shape
  and needs nothing; this file is the second and now carries the non-vacuity
  assertion `tests/env_mgr/test_imports.py:225` already had.
- **A `.yaml` glob would have been the wrong widening.** Ten `.yaml` files live
  under `agent_sys/` and one — `env_mgr/recipes/sglang.repo.yaml` — is not a
  spec, so an extension scan condemns a valid repository. A file is a spec source
  iff it declares `module:`, which is the format's own discriminator and keeps
  one definition of "a spec" in the system.

Verified by breaking it on purpose, both ways
(`scratch/ui-yaml-2026-08/w5/probe_stray_spec_guard.py` and `w3/probe_criterion_5_guard.py`).

**Criteria 16 and 18 were one criterion, and the split reversed a test.** Rev. 10
demanded both `main.yaml` and `assets/` of every package and this package
implemented that. `spec-author` split it at rev. 11 after measuring that the two
have different **arity**: `assets/` is about *being a package* and is required of
every one, while `main.yaml` is about *being a run's entry*, which is one per
**run** — `task_graph/bootstrap.py:47` takes `packages: Sequence[Any]`, so
demanding the file of each answers *"where does a run start"* N times and
therefore not at all, and it made a kinds-only library package inexpressible
three paragraphs after §4.3 permits one.

So `test_a_package_without_main_yaml_fails_naming_the_root` **now asserts the
opposite** and is renamed `..._is_a_library_and_loads`. What survives of the old
rule is the half that was always per-package: a `main.yaml` that is *present* and
declares no `module: task` is an entry to nothing and is rejected naming the file.
The per-run half — exactly one entry package per run — has no owner and is main
spec §10's.

**Criterion 17** — no `.jsonnet` or `.libsonnet` remains, nothing imports
`_jsonnet` or `rjsonnet`, neither is a declared dependency — is only partly this
package's. `render.py`, its two dependencies and `RenderError` are gone from
here; `examples/demo/**` and `validator/general_specs/**` still hold `.jsonnet`
sources and are W6's and `validator`'s.

Design §9.2's three, which guard measured facts rather than criteria:

| Test | Guards |
|---|---|
| `test_validate_takes_no_path` | §3.1. The boundary is the signature, so the signature is tested — a claim that structural should fail loudly when someone adds a convenience overload |
| `test_nested_schema_error_is_one_line` | §3.5. The `items_schema` case produces one actionable message, not eight |
| `test_a_package_is_read_once_per_call_and_gives_the_same_answer_twice` | §3.2's surviving half. The **parallel** half went with the thread pool: it existed because a jsonnet render cost a fixed ~23 ms of VM construction whatever the spec's size, so 100 specs were ~2.3 s serially. A `read_text` plus a parse has no such floor, and adding a pool to it would be optimising something nobody has measured. Idempotence is a correctness property and stays |

**Criteria 7, 12, 13, 14 and 15 are other modules'** (`env_mgr`, `agent`, `demo`)
and §9.1 names them rather than omitting them, so a reader can see nothing is
unassigned. **8, 9 and 11** belong to `handoff` and `closure`; this package
supplies the registry base and the schemas they need. **10** is `task_graph`'s
and was already green.

## What this package owes, and what it is owed

Nothing outstanding either way at the time of writing. What crossed a boundary
and might need re-checking if either side moves:

| Owed to | |
|---|---|
| `handoff`, `validator`, `closure`, `agent` | `BaseSpecRegistry` — the dict, the collision policy, `origin_of`, and the `_validate` / `_admitted` hooks. Its docstrings carry three traps found by three registries; changing the hook ordering breaks four packages |
| all four | `schema_for(kind)`. **They own the content, this package owns the file.** Every shape in the five schemas was confirmed by its owner; four of the five needed the owner's *design*, not just their spec, to be right |
| `task_graph` | `subgraph_of` returns entries **unnormalised** — they own `SubgraphEntry` and default the marks on top. `failed_names` / `rejected` return **origins**, and `check_closures` filters by **name**; only the composition root holds the map between them |
| `closure` | `task_of`, `body_of`, `Body`. `task_of` was disputed, `main` reversed on the objection, and `closure` then retracted it after measuring — settled, one writer, and **the reasoning is in `closure/README.md`** rather than restated here. The criterion it was settled on, which is the part that outlives it: *is the duplication forced by the import graph?* |

## Libraries adopted, and why

Mission rule 5: *prefer a mature, widely adopted library over writing it
yourself; if a de facto standard exists, use it directly; if a thin wrapper
suffices, wrap it; implement it yourself only when nothing fits, and say why.*

| Need | Adopted | Why, and what was rejected |
|---|---|---|
| **Parsing, with positions** | **`ruamel.yaml` 0.18.16, round-trip mode** | Main spec §7 adopts it and this package settles what §7 hands over. Four things measured here rather than read from a changelog (`scratch/ui-yaml-2026-08/w3/probe_ruamel_positions.py`, `probe_ruamel_semantics.py`) — see "What the parser settled" below. It is a **thin wrapper**: `yaml_source.py` is ~90 lines of load-plus-adapt, because the tree the library returns needs no conversion |
| Templating | ~~`jsonnet` / `rjsonnet`~~ **deleted** | Main spec §7 rev. 10 and criterion 17. Not rejected on principle: adopted at rev. 4, shipped, then measured. Across every non-comment line of all 21 `.jsonnet` / `.libsonnet` sources the whole computation surface was constants, string concatenation and default-if-absent, and every general spec used one construct. A compiled extension plus a fallback binding, for three things a variable set and a schema `default` do for nothing |
| Schema constraint | **`jsonschema` 4.26** | Main spec §4.4 makes JSON Schema the *only* enforcement point, and this is the reference implementation. `fastjsonschema` compiles to Python for speed we do not need and has weaker error objects — no `json_path`, no `context` tree, and `report` depends on both. A pydantic model generated from the schema was measured and rejected: `datamodel-code-generator` publishes the keywords generated models do not represent, so such a model **accepts instances the schema rejects** |
| Cross-schema `$ref` | **`referencing`** | Ships *with* `jsonschema` >= 4.18, so it adds no dependency. It is what lets `closure.schema.json` say `{"$ref": "task.schema.json"}` instead of inlining the task shape — two declarations of one shape being the duplication `engineer_principle.md` §1 forbids. The alternative was writing a local-`$ref` inliner, which is the wheel this library is |
| Parsing, second parser | ~~PyYAML `safe_load`~~ **removed from this package's path** | It read the *rendered* document and argued that "neither the YAML 1.1 `norway: NO` trap nor the duplicate-key trap can reach us — jsonnet quotes every string and rejects a duplicate field statically". Both premises are gone. Worse, keeping it would mean **two parsers over one document**: measured, `12:30` is `'12:30'` under `ruamel`'s 1.2 and the integer `750` under PyYAML's 1.1, and `1e3` is `1000.0` against the string `'1e3'`. It stays a repository dependency for other packages; nothing here calls it |
| **Variable substitution** | **own, ~40 lines** (`variables.py`) | Three candidates measured, all failed on something structural. `string.Template.safe_substitute` returns `'${inputs:-any}'` unchanged — no default-if-absent, which is one of the three needs. `os.path.expandvars` likewise, and it reads the *process* environment. `OmegaConf` 2.3.1 has both interpolation and a default form and **raises `ValidationError: Object of unsupported type: 'CommentedMap'`** — it will not accept the position-carrying tree at all, and costs `antlr4-python3-runtime` plus a second bundled PyYAML. That result answers for a **class**: `dynaconf`, `hydra` and `pydantic-settings` all own their container types too, so any of them means parsing twice or parsing without positions. `probe_substitution_libraries.py` |
| **Assets discovery** | **own, ~120 lines** (`assets.py`) | Nothing was looked for and nothing would fit: the convention is `refine.task_package.define.md` §2.3's, invented for this system, and it resolves against this system's `body` shape. What *is* adopted is the error: a conflict raises `SpecInconsistent`, which is `registry.py`'s existing answer to "two things claiming one name" |
| Error relevance | **`jsonschema.best_match`, plus `check-jsonschema`'s deep-match rule** | The format is adopted whole rather than invented, because the project shipped a **second** heuristic after finding stock `best_match` insufficient. Two guesses plus an escape hatch is the state of the art; a third invented here would be worse. `check-jsonschema` itself is not a dependency — it is a CLI whose internals are not an API, so the ~10 lines of `_best_deep_match` are the wrapper |
| Parallel rendering | ~~`ThreadPoolExecutor`~~ **removed** | It was there for a measurement that no longer describes anything: a jsonnet render cost a fixed ~23 ms of VM construction whatever the spec's size, and `_jsonnet` released the GIL, so 8 threads beat serial 0.95 s to 6.98 s. A `read_text` plus a parse has no such floor. **Not replaced by a faster pool — removed, because nobody has measured a problem**, and a pool over an unmeasured cost is complexity bought on a guess |
| Resource access | **`importlib.resources`** | `docs/design.md` D1. Behaves identically from a checkout, a wheel, and a zipimport, where a relative path works from the first and dies from the second |
| The registry | **own, ~120 lines** | `pluggy` was considered and rejected: it solves 1-to-N hook broadcast with ordering and wrappers, and a spec lookup is a dict with a collision policy. Its duplicate-rejection *behaviour* is adopted (as is `fsspec`'s error-by-default, identical-re-registration-is-a-no-op shape); its machinery is not |
| The package format | **own** | dbt packages, Helm dependencies and Ansible collections were surveyed. None has a lockfile with content hashes, so there is nothing to adopt for the part that matters. What *is* adopted is the shape: discover, then resolve |
| Error types | **three of our own** | `jsonschema.ValidationError` cannot express "two specs that both validated disagree", which is the failure this system most needs to report well. JPMS separates `FindException` from `ResolutionException` for the same reason |

## What the parser settled, and the one thing it did not

Main spec §7 records `ruamel.yaml`'s `lc` accessors from **web research** and
hands three open questions to this package: who parses what, with which loader,
and whether the two traps that jsonnet's deletion re-opened are live. Measured
first-hand, because §7 was written before anything in this tree parsed a
hand-written document and the shapes that matter — a **list at a file's root**
and a **nested inline definition** — did not exist under jsonnet.

| Question | Answer |
|---|---|
| Do positions survive nesting? | Yes. `lc.item(i)` on a root sequence, `lc.key(k)` / `lc.value(k)` on a mapping three levels down. Both 0-based; `Position.from_ruamel` is the one place that adds one |
| Syntax errors? | `MarkedYAMLError.problem_mark`, line and column, on all three malformed documents probed |
| **Duplicate keys?** | **Rejected**, with a mark, by both `typ='rt'` and `typ='safe'`. The trap is closed by the library rather than avoided. PyYAML `safe_load` silently keeps the last value |
| **YAML 1.1 or 1.2?** | **1.2.** `NO`, `yes`, `on` and `12:30` stay strings; PyYAML gives `False`, `True`, `True` and `750`. The `norway: NO` trap does not need avoiding — a package author's values mean what they look like |
| Does the tree validate as-is? | Yes. `CommentedMap` and `CommentedSeq` subclass `dict` and `list`, so every `type` keyword and every `err.json_path` is correct over the position-carrying tree. **Nothing is converted, so nothing is lost between the parse and `validate`** — this is what makes the wrapper thin |

**Substitution happens after the parse, and that is not a style choice.** Main
spec §7 rejects Jinja2 because text templating *"can emit a document that is not
valid YAML at all"*, and the argument is about the operation rather than about
Jinja2 — so it applies to a twenty-line regex over the source text just as much.
Checked over six values a package author could reasonably supply
(`probe_substitute_shape.py`):

    a plain path        parsed, ok        a newline           BROKE: ScannerError
    a colon in prose    BROKE: Scanner    a hash              parsed, WRONG: 'run'
    a leading dash      BROKE: Scanner    a windows-ish root  parsed, ok

Four of six, and the fourth is the dangerous one: `run #3` becomes `run`, because
`#` starts a comment — a document that validates and admits with a truncated
value. Substituting into the parsed tree cannot do either, and the positions are
byte-identical before and after.

### The one thing round-trip mode does not do, recorded rather than guarded

`typ='rt'` does **not execute** an unknown tag — `!!python/object/apply:os.system
['echo …']` comes back as a plain `CommentedSeq` and `!!python/name:os.system` as
a `TaggedScalar`, and nothing runs. But it does not *refuse* it either, the way
`typ='safe'` does, and `typ='safe'` has no `lc`. So a document can carry a tag
the schema cannot see.

The exposure is a hypothetical downstream unsafe round trip and this package
performs none, so it is written down rather than guarded. **The probe's first run
labelled this "CONSTRUCTED — unsafe" purely because no exception was raised**;
the correction is in the probe, because a wrong label in the evidence is worse
than no probe.

## Where the five schemas live, and why `task` is not discoverable

`spec_loader/schemas/*.json`, hand-written, Draft 2020-12, read through
`importlib.resources` — `docs/design.md` D1 measured that a top-level
`agent_sys/schemas/` is not installable at all.

`task.schema.json` is one of the five and **no package ships a task spec as a
file**: `closure` spec §2 declares it inside the closure as the `task` key, so
`closure.schema.json` `$ref`s it. Four kinds are discoverable, five schemas
exist. `_common.schema.json` is a sixth *file* and not a sixth kind — it holds
the shapes more than one schema names, today just `body`, which `closure` §2.6
and `validator` §6.1 describe as deliberately the same thing.

A field carrying a **nested user-supplied schema** — `handoff.items_schema` — is
typed `{"type": "object"}` and its validity is left to a named load-time check
calling `check_schema`. Not a `$ref` to the metaschema: `docs/design.md` §3.5
measured the `$ref` form producing eight identical unactionable errors for one
mistake, because the metaschema's `anyOf` branches each fail the same way.

## What this package exports that `interfaces.md` §4.1 does not list

§4.1 says: §3's whole table, plus `render`, `validate`, `load_package`, `report`.
**`render` no longer exists**, so §4.1 needs that row *removed* — the first time
this package has asked for a deletion rather than an addition, and it is an
`interfaces.md` edit nobody here may make. The other names below are exported
because something the frozen contract already names would otherwise have nowhere
to come from. **They are proposed additions to §4.1, not decisions** —
`interfaces.md` §1.1.

| Name | Why it cannot be internal |
|---|---|
| `format_problems`, `failed_names`, `rejected` | §2's composition root calls all three and no §4 row assigns them. See below |
| `Body`, `body_of`, `subgraph_of` | Ruled here by `main`. One `$defs.body` in `_common.schema.json`, three Python declarations over it. See below |
| `BaseSpecRegistry` | §3 says four registries **subclass** `SpecRegistry` — but that is a `Protocol` with `...` bodies, so subclassing it yields no dict and no collision policy, and four packages would each write their own. This is the implementation `docs/design.md` §5.1 calls *"a shared **loader** the four registries call"*. The Protocol stays the type |
| `YamlPackage` | `build_registry` takes `packages: Sequence[TaskPackage]` and no module constructs one. Without this, `demo` writes the whole front end from scratch. It replaces `DirectoryPackage`, and `FileImportResolver` and `RenderError` went with `render` |
| `MODULE_KEY`, `ENTRY_FILENAME`, `ASSETS_DIRNAME`, `ASSETS_VAR` | The four spellings a package author has to get right — `module:`, `main.yaml`, `assets/`, `${TASK_PACKAGE_ASSERT_DIR}`. Main spec §4.3 fixes two of them as **names** (which is not the same as demanding both files of every package — §4.3's arity subsection), so a test or a package generator that hard-codes the string is a second writer of a fact this module owns |
| `AssetIndex` | The assets resolver, exported so `demo` can show what convention found what without re-walking the tree |
| `schema_for`, `KINDS` | The five schemas live here and four other modules own what is *in* them. Without one accessor each of those four hand-rolls an `importlib.resources` read, which is D1's failure mode reintroduced four times |

### The composition root's derivations — three of §2's four

`interfaces.md` §2 step 5 calls `format_problems`, `failed_names`, `merged` and
`rejected`, and no §4 row assigned any of them. Three are here, because they are
operations over `Problem` and `LoadReport` and the module that owns a type owns
the operations over it (`engineer_principle.md` §3):

| | |
|---|---|
| `format_problems` | One line, delegating to `report`. Both names are normative — §2 writes one, §4.1 lists the other — so one is expressed in terms of the other rather than reimplemented (main spec §3.1 principle 10). Collapsing them is an `interfaces.md` edit |
| `failed_names` | The origins of specs that failed. **Fatal only**, and that guard is live rather than defensive: `closure/check.py`'s check 3 is the one producer of `fatal=False` in the system, for a closure built from a kind admitted under the escape hatch. Gating on it would skip the very closure whose reporting `closure` criterion 6 requires |
| `rejected` | The origins a whole-catalogue pass rejected. Fatal only, same reason |

**`merged` is deliberately absent.** It folds `handoff.HandoffLoadReport`, whose
shape is `handoff`'s to define and whose constructor this package may not name —
and §2's own line passes it `reports`, which are `spec_loader.LoadReport`s and
have no `without_validator`. The two types cannot be bridged by a function here.
`engineer_principle.md` §2: *"If the right home does not exist, say so."*

**The one that does not compose yet, stated plainly.** `failed_names` and
`rejected` return **origins** — a `Problem` identifies a file — and
`closure.check_closures` filters with `if name in skip`, where `name` is a
*closure name*. So the layering gate does not close today. Neither side is wrong
on its own: `Problem` has carried `origin` since `protocols.py` was frozen, and a
registry is keyed by name. Bridging it needs the origin-to-name map, which only
the registries hold — one line in the composition root, or a field on a frozen
type. Reported, not decided.

## Two adaptations of `check-jsonschema`'s report format

Its shape is four parts:

```
<origin>::<path>: <message>
Best Match: <path>: <message>
Best Deep Match: <path>: <message>          (only when it differs)
N other errors were produced. Use --verbose to see all errors.
```

The headline here already *is* the best match — `validate` sorted it there — so
repeating it under a `Best Match:` label would print one error twice. And the
escape hatch is `verbose=True` rather than a CLI flag, because this package has
no CLI.

## Layout

| File | |
|---|---|
| `protocols.py` / `.pyi` | **Frozen.** The contract, declaration only (`interfaces.md` §8) |
| `errors.py` | The three from `protocols.py`, re-exported rather than redeclared. `RenderError` was a fourth and went with `render.py` |
| `yaml_source.py` | text → a tree that still knows its own line numbers. `render.py`'s successor, and **the only module that reads a package's file** |
| `variables.py` | `${NAME}` and `${NAME:-default}`, expanded on the parsed tree |
| `assets.py` | `assets/` — a body's files, found by filename convention |
| `validate.py` | document → `[Problem]`. **Takes no path, and since rev. 10 no bytes either** |
| `report.py` | Problems → a line a human can act on |
| `registry.py` | `BaseSpecRegistry` — the dict, the collision policy, the error shape |
| `bundled.py` | The five schemas, and the `referencing` registry that resolves `$ref` between them |
| `package.py` | `YamlPackage` — the whole front end, on the package's side of the seam — and `load_package`, whose body is the design |
| `schemas/` | Five kinds plus `_common` |

## Open, and reported rather than closed

- **`items_schema` and the `check_schema` call.** The schema types it loosely on
  purpose; the named check is `handoff`'s registry's, not this package's.
- **`tags.domain` accepts a string or a list.** `validator` spec §9.2 reads as
  one value; the three general specs under `validator/general_specs/` all write a
  list. Free-form metadata nothing enforces is the wrong place to make a package
  author lose, so both are accepted until the owner decides.
- **A placeholder passes every check a real value passes** — main design O9.
  `required` catches a field an unfilled template left *absent* and nothing
  catches one filled with `"TODO"`.
  `tests/spec_loader/test_schemas.py::test_a_placeholder_is_admitted_and_that_is_open`
  asserts the limit so it stays visible.
- **Containment is not enforced** — main design O3, a specification question, and
  it survives jsonnet's deletion in a changed form. There is no import resolver
  to substitute any more; the scan is rooted at the package and
  `test_the_scan_does_not_leave_the_package` pins that, but a **symlink** from
  inside still reaches out and must — `test_cross_package_symlink_loads` requires
  it, and main spec §4.3 calls it the supported way two packages share. So what
  is open is narrower and sharper than before: *may a symlink leave the root?*
  Kustomize's `LoadRestrictions` is the prior art and answers "no"; this system
  answers "yes" and has a test that says so. One of the two has to give.
- **A package-declared variable set does not exist, and nothing has asked for
  one.** Main spec §4.4 says shared constants live in *"a package-level variable
  set"* without saying where it is written. `YamlPackage` takes `variables=` from
  the caller, which is what `config=` was. Measured over all 21 jsonnet sources,
  **every** variable reference was to a caller-supplied value —
  `config.package_root`, `config.outside`, `config.inputs` — and not one package
  declared a constant of its own. A `vars:` block would be a construct with no
  measured user (`engineer_principle.md` §2), so it is not built. If a package
  ever wants one, this is the note that says the decision was deliberate.
- **Two of the four kinds have no body to fill from `assets/`.** The user's rule
  finds a readme for *"某个命名的 agent/task/handoff/validator"*; measured against
  the schemas, only `task.body` and `validator.body` exist. `agent` has
  `knowledge` / `rules` / `skills` and `handoff` has `readme_sections`, and none
  of those is a path. Inventing a field to fill would be §2's failure mode, so
  the gap is reported — and `test_an_agent_and_a_handoff_have_no_body_to_fill`
  keeps it visible in the suite rather than only here.
- **The forward-reference rule is within a file, not across the package.**
  `docs/ui-stage.md` §4 W3 step 6 reads as package-wide. Measured against
  `examples/demo`, no total file order satisfies its own reference graph — sorted
  by path, `closures/produce` references `handoffs/facts` and
  `validators/check_facts`, both later; reversed, it references `agents/collect`,
  which is then last — and no ordering by kind works either, because
  `handoff.validators` and `validator.inputs` are a real 2-cycle. `validator.inputs`
  is therefore **not** a reference key, which leaves exactly one legal order for
  that pair. Both facts are pinned by tests. Reported as a narrowing.
- **A round-trip parse preserves an unknown tag rather than refusing it.** See
  "What the parser settled". Nothing executes and nothing here dumps, so it is
  recorded and not guarded.
- **Spec numbers are IEEE 754 doubles** — main design O4. Any field needing an
  exact large integer must be declared as a string, and none currently is.
- **`subgraph`'s key and entry shape are named by no specification.**
  `task_graph` chose `task.subgraph` as `[{closure, is_start?, is_end?}]` with
  absent marks defaulting to first and last. Hosting the accessor gives the key
  one reader; it does **not** promote the convention to a rule.
- **The covering relation has two implementations and they disagree.**
  `task_graph.Permissions.covers` implements *a WRITE grant implies READ* and
  says so in its docstring; `closure.check.covers` matches kind and access
  exactly. `task.schema.json` now states the vocabulary and **not** the relation,
  because a description asserting an implication the checker does not perform
  would tell an author a grant covers something load will reject. Neither type is
  this package's, so the conflict is reported to `task-graph` and `closure`
  rather than settled here.

## Schema-checking in a test: use `validate`, not a bare `Draft202012Validator`

Two modules have reached for `jsonschema` directly and found the schemas
unresolvable. `handoff.schema.json`, `task.schema.json`, `validator.schema.json`
and `closure.schema.json` all `$ref` across files, and a bare
`Draft202012Validator(schema)` carries no registry to resolve them with.

`validate(doc, schema_for(kind), origin=...)` already holds one, built over
every bundled schema and keyed by `$id`, and it is the single enforcement point
besides. Hand-rolling the `referencing` registry works until it does not, and
the failure is a resolution error a long way from its cause.

## `Body` and the two accessors — why a `TypedDict`

`main` ruled `Body`, `body_of` and `subgraph_of` here: `_common.schema.json`
holds **one** `$defs.body` and Python declared it three times — `closure` as a
`TypedDict`, `agent` and `validator` as frozen dataclasses.

**The `TypedDict` wins, and the reason is §4.1's**, one level down. A spec is a
plain `dict` throughout; typing is all that is wanted. A dataclass has to
*construct*, and constructing means inventing a value for a field the document
does not have — `agent.body_of({})` returned `Body(readme="")`, an object that is
**truthy** and reports a body that is present and empty where the task had none.
`{}` is falsy, so `if body_of(task):` means what it looks like.

**`subgraph_of` returns entries unnormalised**, and the split is the point:
this package owns that the key exists and is called `subgraph`; `task_graph` owns
what an entry *means* and defaults `is_start` / `is_end` on top, because those
mean something only once an entry is a `SubgraphEntry` — a type this package may
not name.

**The line this package does not cross.** It may *declare and expose* the
vocabulary; it may not *act* on it during a load. Exporting `body_of` is
declaration-side. Having `load_package` reach into an admitted closure for its
`task` key would be action-side, and that is what main spec §4.4 makes
structural. The two look alike and are not.
