# `closure`

The predefined binding of a task's handoffs, its agent, and its validators.

```
closure = < handoff spec set, task spec, agent spec, validator set >
```

Three things and nothing more: **a composition of four parts**, **a load
checker**, and **read-only query helpers**. Nothing at runtime — a closure is
consulted when a graph is assembled and never again.

| | |
|---|---|
| Spec | [`docs/spec.md`](docs/spec.md) rev. 9, 11 acceptance criteria |
| Design | [`docs/design.md`](docs/design.md) rev. 4 |
| Contract | [`../docs/interfaces.md`](../docs/interfaces.md) §4.5, and [`protocols.py`](protocols.py) |
| Tests | [`../tests/closure/`](../tests/closure/) |

---

## Layout

| File | What |
|---|---|
| `model.py` | The two document aliases and the accessors. **Every schema key this package reads is read here** — and `Body`, `body_of`, `subgraph_of` and `task_of` are re-exported from `spec_loader` rather than declared, so it is one place to look without being a second writer |
| `registry.py` | `ClosureRegistry` — the table, the frozen index, and the six queries |
| `task_registry.py` | `TaskSpecRegistry` |
| `check.py` | The eight per-closure checks, `covers`, and `check_closures` — which is the whole *pass*: it also keys each nested task spec, feeds `validator_specs.bind_phase`, and runs one **catalogue** check that is not per-closure (`_check_validator_agents`) |
| `query.py` | The reverse index and the joins behind the queries |

The two schemas this module *owns the content of* — `closure.schema.json` and
`task.schema.json` — live in `spec_loader/schemas/` and are read through
`spec_loader.schema_for`. Main design §2.2 measured why all five belong in one
installable package: a bare directory of `.json` is not a package, so reading one
by relative path works from a checkout and fails from a wheel.

`closure` imports **`spec_loader`, and nothing else in this repository**. It
reaches the other four modules through the `Registries` handle it is given, by
name, at call time. That rule matters more here than anywhere: this is the module
whose whole job is looking at four other modules' objects, so it is where an
import would be easiest to justify and hardest to remove.
`tests/closure/test_conformance.py` is the enforcement.

---

## Done: every criterion, and the test that holds it

`closure/docs/spec.md` rev. 8 has eleven. Each maps to the test name
`docs/design.md` §11 specifies, in the file it specifies — the mapping is the
deliverable rather than a formality, so a rename here is a change to the design's
test plan and not a tidy-up.

| # | Criterion | Test | File |
|---|---|---|---|
| 1 | An unresolvable kind is rejected, with the kind named | `test_unresolved_kind_names_it` | `test_check.py` |
| 2 | An input absent from declared `handoffs` is rejected; the reverse loads | `test_declared_handoffs_one_directional` | `test_check.py` |
| 3 | A missing agent spec is rejected, **and so is naming none** | `test_missing_agent_spec_rejected`, `test_no_agent_key_rejected` | `test_check.py` |
| 4 | A phase validator resolving to a general task is rejected | `test_phase_validator_is_not_a_task` | `test_check.py` |
| 5 | Permissions not covering handoffs, naming both | `test_permissions_cover_handoffs` | `test_permissions.py` |
| 6 | An escape-hatch kind loads, **and reports that it did** | `test_escape_hatch_reported` | `test_check.py` |
| 7 | Two closures may share a kind, an agent, a validator | `test_sharing_is_legal` | `test_registry.py` |
| 8 | The scheduler never reads a closure | `test_scheduler_never_reads_closure`, `test_the_spy_would_catch_a_scheduler_read` | `test_authority.py` |
| 9 | Every query answers without a join; none mutates | `test_queries_need_no_join`, `test_queries_do_not_mutate` | `test_query.py` |
| 10 | The six reference steps are each one closure | `test_six_step_shape_loads` | `test_reference_shape.py` |
| 11 | A closure declares no version; each member declares its own | `test_closure_version_rejected`, `test_no_runtime_version_read` | `test_schema.py` |

**Three checks exist beyond the spec's six**, and none of them is a criterion:

| | |
|---|---|
| **Check 7 — the body** | `readme` always, `entry` and a subgraph mutually exclusive. Design §3.6; `test_body.py`. Narrower than the design asks — see below |
| **Check 8 — subgraph targets** | An entry naming a closure the catalogue does not hold. Asked for by `task_graph`: `Task.unfold` raises on the same fault, but hours into a run for a subgraph nested three deep, rather than beside every other reason the graph is not admissible |
| **The load path** | `test_load_path.py` runs a real `YamlPackage` through the real `load_package`, this pass, and `task_graph.check_graph` — which is where `task_graph` criteria 50 and 53 are exercised end to end |
| **A validator's agent resolves** | `validator` §8.2 row 1's load-time layer. **Not per-closure** — the fault is a property of the validator spec, so it is reported once, keyed to `validator_specs.origin_of(name)`, and it catches a validator bound only to a handoff kind that no closure names. `test_registry.py` |

**One check here reads a document this package does not own**, and it is the only
one: `_check_validator_agents` reads a validator spec's `agent` key. It does so
through `spec_loader.validator_agent_of` rather than `spec["agent"]`, because the
key is `validator`'s and `closure` may not import them — measured, not assumed,
by adding the import and watching `test_import_rules` fail. The accessor is named
`validator_agent_of` and not `agent_of` because a leaf exporting two accessors of
one name over two document types is a shadowing hazard nobody downstream could
alias around.

**Two tests here assert against something outside this package on purpose**, because a guard that stops at its own boundary is the failure mode this package has now hit twice: `test_the_real_users_of_stops_under_reporting` uses `validator`'s real registry, and `test_load_path.py` uses `task_graph`'s real `check_graph`.

---

## Libraries adopted, and why

Mission rule 5. **Nothing outside the standard library and the repository's
existing dependency set is adopted**, and each row says why the alternative was
not built.

| Concern | Chosen | Why |
|---|---|---|
| Schema validation | **`jsonschema` ≥ 4.18** (`Draft202012Validator`) | Already a design-set dependency and the only enforcement point the main spec admits. Its `json_path` and `context` tree are what the report format depends on; `fastjsonschema` compiles for speed nobody needs and has weaker error objects |
| "did you mean" in a message | **`difflib.get_close_matches`** (stdlib) | The survey's differentiator is a *computed repair drawn from what is in scope*, which is a candidate list plus one close match. A hand-written edit distance would be a second implementation of a stdlib function |
| The registry base | **`spec_loader.BaseSpecRegistry`** | It already holds the dict, the collision policy, `SpecNotFound` with its candidate list, and `origin_of`. A fifth bespoke registry would be a fifth policy to keep in step, and Kubernetes' `runtime.Scheme` is what that becomes: one struct, seven typed maps, three different collision policies |
| `$ref` between the two schemas | **`spec_loader.bundled_registry`** | `referencing`, wrapped once in the package that owns the files, rather than four modules each hand-rolling the same read |
| The document type | **`Mapping` alias + accessors** | A `TypedDict` or a pydantic model is a second declaration of the schema's shape. A typed object built from a schema *accepts instances the schema rejects* — `datamodel-code-generator` publishes the list of keywords whose semantics it cannot represent |
| `Body`, `body_of`, `subgraph_of` | **`spec_loader`'s** | One `$defs.body` in `_common.schema.json` had grown three Python declarations, and `subgraph`'s key two readers. `spec_loader` already names every key in the system, so a reader of one adds no interpretation of a package's *content* — and it is the one package everyone may import |
| The reverse index | **eager and frozen, written here** | Nobody surveyed maintains one incrementally, so there is nothing to adopt. dbt rebuilds O(V+E) at six call sites; Sphinx's outlives its build and charges a `clear_doc` and a `merge_domaindata` obligation to every owner including third-party extensions; Cargo throws its inversion away per invocation. Our world is closed, so the index is built once and mutation is made impossible |
| The covering relation | **written here, and it exists twice** | Forced, not careless — see below. Exact equality on the *kind name*, `WRITE` covering a `READ` requirement on the *access*. kubernetes#122154 is what an under-specified name grammar costs: their check wrongly rejects a legal delegation because a second component gave `example.com/*` a meaning the authorization API never defined, and they shipped the wrongness rather than open the grammar |
| Criterion 8's spy | **`sys._getframe`** (stdlib) | The alternative is a `caller=` parameter on `SpecRegistry.get`, which changes a base class shared by five registries to satisfy one test. Fragile, and chosen anyway — recorded as design O5 |

Two external systems shaped decisions without contributing code — Kubernetes
RBAC's covering relation and dbt's manifest maps — and both entered as
constraints, not as designs to copy.

---

## Four things worth knowing before changing this package

### Check 7 checks a declaration, not a path on disk

Design §3.6 asks that a body's `readme`, `entry` and `materials` paths *resolve*.
`check_closures` is handed registries and an opaque `origin` label, and resolving
a path would need the package root — which is deliberately unreachable from here,
because main design §3.1 makes the loader's no-path property structural rather
than disciplinary.

So check 7 asserts the declaration is present and coherent — `readme` always,
`entry` and a subgraph mutually exclusive — and **path existence belongs to
whoever holds the `TaskPackage`**. Reported rather than worked around.

### Criterion 6 was met at this boundary long before it was met at all

*"A closure assembled from a handoff kind admitted under the escape-hatch flag
loads, **and reports that it did**."* `check_closures` produces a non-fatal
`Problem` for it, and `test_escape_hatch_reported` has asserted that from the
start.

**The value reached nobody for most of a day, through three correct repairs.**
`handoff` renamed `report()` to `load_report()` to match what the composition
root reached for; the root dropped a `getattr(..., lambda: None)` default that
had been swallowing it; this package made `handoff_report` required and a `None`
loud. Each was right. After all three, `build_registry` still computed the
non-fatal problems and then filtered them away — `[p for p in problems if
p.fatal]` — so nothing was printed and criterion 6 was unmet in the assembled
system while this package's suite was green.

The fourth repair is `task_graph/bootstrap.py`, which now logs them. **A path can
be repaired at every point and still not deliver**, which is `docs/interfaces.md`
§4.13; the instance is recorded here because the criterion is this package's.

Verified where it matters rather than at this boundary —
`scratch/impl-2026-08/closure/probe_criterion_6_arrives.py`, real package on
disk, real `build_registry`, escape hatch on:

```
the report ARRIVED:
  1 admitted with reservations:
  .../closures/collect.yaml::$.handoffs: closure 'collect' is assembled from
  handoff kind 'loose', admitted under the no-validator escape-hatch flag...

control — the same package with the kind given a validator:
  reservations reported: 0  (expected 0)
```

**The control is not decoration.** Without it the probe says *a warning fired*,
not *a warning fired about this* — and a probe showing a pass deserves the same
scepticism as one showing a failure.

### `task_of` is settled, and it is not this package's

**Settled, and this is where the reasoning lives** — `spec_loader/README.md`
points here rather than restating it, which is the right shape: a row asserting
a *state* goes stale in a file its owner does not control, and this one did once
already.

One writer, `spec_loader.access.task_of`; `task_graph/models.py:20` imports it;
`closure/model.py` re-exports it so this file stays the one place a `closure`
reader looks for a key name, and it is deliberately absent from `__all__` and
from `protocols.py`.

I argued against the move and was wrong on the claim that decided it. The
argument was that `body` and `subgraph` are keys of the *task spec*, sharing one
`$defs.body`, while `task` is a key of the *closure document* — this package's
subject — and that `task_graph` could delete its copy by resolving `task_specs`
by name. **Measured, the second half is false**: `_admit_task_specs` runs inside
`check_closures` and not inside `ClosureRegistry.add`, so a closure can be
declared with its task spec absent, and `unfold` reading `task_specs` would raise
on a closure that is demonstrably declared. `task_graph._instantiate` also holds
the document already, for the membership check and for `doc["agent"]`.

The criterion that replaced both arguments is `spec_loader`'s and is the durable
part:

> **Is the duplication forced by the import graph?** Two packages needing one
> accessor, across an edge one of them may not cross, with no lookup either can
> make instead.

That is `Pushable`'s test rather than a count. `task_of` passes it; `agent`'s copy
failed it, which is why deleting the *need* was right there and wrong here.
`docs/interfaces.md` §4.5 carries it.

### The covering relation, and why it exists twice

`covers` matches the kind name exactly and treats **`WRITE` as covering a `READ`
requirement**. That second half was not this module's choice and was not its
first answer: rev. 1 matched access exactly, on the argument that Kubernetes RBAC
verbs do not imply one another and Android lint is bare set membership. Both true,
and both about a greenfield choice. `task_graph/permissions.py::Permissions.covers`
had already shipped the implication — with a docstring naming *this* check as the
reason it exists — and the deciding argument is `env_mgr`'s: a write grant on a
directory without read and execute is unusable, because a file cannot be created
in a directory that cannot be traversed. The exact version's failure was
over-rejection, refusing a spec that would have run.

**The relation is obliged to exist twice, and that is the finding rather than the
defect.** At load there is no `Task`: a task spec is a `Mapping`, and building a
`Permissions` from one would mean importing `task_graph`, which §4.5 forbids. So
one relation has two bodies, one over pydantic objects and one over raw dicts.
`tests/interfaces/test_covers_agreement.py` is the price of that, on the terms
`docs/interfaces.md` §8 sets for `Pushable`.

**The name axis did not move with it.** Access is a two-element order on a closed
enum; a kind name is an open string that invites a second interpreter, and
kubernetes#122154 is what that costs. The agreement test pins both axes so that
widening one is never read as licence to widen the other.

---

## What this package owes anyone: nothing

Checked against the tree rather than remembered, because the answer being "none"
is only useful if somebody looked.

| Raised with | State |
|---|---|
| `validator` | `bind_phase` is fed from `check_closures` and asserted against their **real** registry. `closures_using_validator` survives beside their `users_of`, answering a different question — `docs/interfaces.md` §4.5 |
| `task_graph` | Check 8 landed at their request; criterion 50's shape landed in `test_load_path.py` with the negative row they asked for. `_task_of` is gone from their side |
| `spec_loader` | Both schemas and the registry base are theirs and adopted. `task_of` settled — above |
| `handoff` | `HandoffLoadReport`'s four properties confirmed; `handoff_report` is required here and a `None` raises |
| `agent`, `monitor` | Review findings, all closed by their owners |

**Two open items are named here so a reader does not go looking in this package
for them, and neither is `closure`'s**: `docs/interfaces.md` §4.4's *Resolves* row
disagreeing with §2.1, and `validator_executor` wanting a row in §2.1's table.
Both are the interface file's.

---

## The seams: one open, one closed, one declined

All three are `../docs/interfaces.md` §5, and none was the implementer's to decide alone.

**§5.1b — nobody wraps `materials` into a handoff.** Spec §2.5 gives a task spec
a `materials` key and `validator` design §3.8 gives a validator body the same one;
nothing reads either. The schema admits the key, `body_of` exposes it, and check 7
treats it as a declaration. Whoever needs materials resolves them through
`task.closure` → the task spec, and *how they become a handoff* stays one call
behind a name.

**§5.4 — which reference kinds "who uses this" enumerates. Closed.** There are
three edge kinds in three modules: a handoff kind naming a validator, a composite
naming a member, and a closure naming a phase validator. The design's answer was
a union at the composition root, which *"does not scale to a fourth"*. What
happened instead is better: `validator`'s `users_of` records the **edge kind** on
each entry, and `check_closures` feeds it the third kind through `bind_phase`
(`check.py::_bind_phase_validators`). One owner, one representation, no union.

`closures_using_validator` survives beside it, answering a different question —
*which closures name this as a phase validator*, typed, within one kind, where
`users_of` answers *who names this and how*, across kinds. It was withdrawn on
the reasoning that it was a filter over `users_of`, and the withdrawal was
reversed; `docs/interfaces.md` §4.5 records both arguments, including one of mine
that measurement showed to be false.

Airflow #58058 and dbt#14436 are what the unfed-edge failure looked like
elsewhere: silent false-negative deadness, in both cases.

**And one this package declines: turning a closure into the root `Task`.** A
helper returning a `Task` would make `closure` import `task_graph`, and would make
this module the thing that decides a task's initial permissions. `cli/build.py`
holds it, and the whole-system CLI inherits it.
