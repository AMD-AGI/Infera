# Closure — Design

| | |
|---|---|
| Status | Draft — stage two of spec → design → test & code |
| Revision | 4 — 2026-08-27. **A task spec has a `body`**, following spec §2.6 rev. 9 — three accessors and a seventh load check, `entry.sh` mutually exclusive with a subgraph while `readme.md` is required of every task (§3.6). `repos` and `monitor` join the accessors. (rev. 3: 2026-08-27. **The stage-three consistency pass.** §5's check table said "if one is named" where §3.3 and criterion 3 make check 4 unconditional — one stale cell from rev. 2. D2 and D3 are both **adopted upstream** and retire: `task_graph` design §3.5 rev. 12 types the grant by kind name, and main design §3.6 rev. 2 moves the pass to the composition root. §7.3's `Registries` is now defined in `spec_loader`, without the `handoff_report` field that would have made the leaf import `handoff` (main design §3.7). (rev. 2: 2026-08-27. `agent` is **required**, following spec §2.2 rev. 8: §3.3 rewritten, criterion 3's tests renamed, D1 retired. The loader still synthesises nothing. (rev. 1: initial))) |
| Implements | [`spec.md`](spec.md) rev. 9, acceptance criteria 1–11 |
| Language | Python ≥ 3.10. No pydantic models here — a spec is a `dict` (main design §4.1) |

---

## 1. Scope

This document turns [`spec.md`](spec.md) into files, classes, and interfaces. It
adds no requirements. Where it makes a choice the spec left open, the choice is
stated here; where implementing the spec exposed a contradiction, §13 says so
rather than papering over it.

The spec's 11 acceptance criteria are the definition of done. §11 maps every one
to a named test.

**This module is unusual, and it is worth saying why before anything else.**
Every other module in this stage had a subject nobody else had described. This
one does not: [`../../docs/design.md`](../../docs/design.md) §6 already gives the
closure check a home and a report format, its §2.3 and D2 put the task spec
registry inside this package, and
[`../../task_graph/docs/design.md`](../../task_graph/docs/design.md) §3.5 and
§8.5 name two of its collaborators. So the research for this module was mostly
**reading the four designs against each other**, and what it produced was six
collisions rather than six discoveries. Four of them change something here, and
§13 records them.

**This document specifies interfaces, not bodies.** A method is a signature and
a sentence. A body appears only where the ordering of steps *is* the design
decision — which is §5.1's check sequence and §7.1's placement, and nothing else.

### 1.1 What this module owns

- The **closure document** and the **task spec** nested inside it: their schemas,
  their accessors, and the type names the rest of the design set already uses
  without defining (§3).
- **Two registries** — `ClosureRegistry` and `TaskSpecRegistry` (§4).
- **The six load checks** (§5), and check 6 in particular, which is the one only
  this module can perform (§6).
- **Where the closure pass runs**, which the spec leaves open and the main design
  gets wrong (§7).
- **The read-only query helpers** and the index behind them (§8).
- **Proving that nothing reads a closure at run time** (§9).

### 1.2 What it does not

| Deferred to | What |
|---|---|
| `task_graph` design §8.7 | **Graph-level composition** — cycles, reachability, containment. Spec §4.1 declines it; `check_graph` claims it. §7.2 sequences the two |
| `env_mgr` spec §4 | **Interpreting a permission.** Nothing here resolves a path, compares a prefix, or decides containment. §6.2 is emphatic about the boundary and about why it has to be |
| `task_graph` spec §3.2.2 | **Owning permissions.** They are a versioned *task* attribute. This module reads them for one check and stores none |
| `validator` design §10.5 | **The reverse index from a validator to its users.** §8.5 contributes an edge to it and does not host it |
| `spec_loader` | **Rendering, schema validation, and admission.** This module supplies two schemas and two registries into that pipeline |
| The whole-system CLI — [`../../docs/TODO.md`](../../docs/TODO.md) | **Turning a closure into the root `Task`.** §9.2 says why that is not here; `demo` design §6 and D3 record who does it in the meantime |

---

## 2. Layout and import graph

```
closure/
├── __init__.py
├── model.py           the two document types, their accessors, and TaskSpec
├── registry.py        ClosureRegistry(SpecRegistry) — and the index (§8)
├── task_registry.py   TaskSpecRegistry(SpecRegistry)
├── check.py           the six checks (§5), and check_closures (§7)
├── query.py           the five read-only helpers (§8.1)
└── docs/
    ├── spec.md
    └── design.md      this document
```

Two of the five schemas are this module's contribution to `spec_loader`, and
they live there rather than here, for the reason main design §2.2 measured — a
bare directory of `.json` is not an installable package:

```
spec_loader/schemas/task.schema.json
spec_loader/schemas/closure.schema.json
```

### 2.1 Import graph

```
                        spec_loader/
                             ▲
                             │
                         closure/            imports spec_loader, and nothing
                             ▲               else in this repository
                             │
                        bootstrap (main design §7)
```

`closure` imports `spec_loader` for `SpecRegistry`, the three error classes, and
`Problem`. It imports **no other module package** — not `handoff`, not
`validator`, not `agent`, and not `task_graph` — even though the six checks
resolve names against all four registries. It reaches them through the
`Registries` handle it is given (§7.3), by name, at call time. That is the same
discipline main design §2.3 states for the module packages generally, and here it
is load-bearing rather than stylistic: this is the only module whose whole job is
to look at four other modules' objects, so it is the one where an import would be
easiest to justify and hardest to remove.

**Nothing imports `closure` except `bootstrap`.** `task_graph` resolves
`closures` through the component `Registry` by name at use time
([`../../task_graph/docs/design.md`](../../task_graph/docs/design.md) §3.4), which
is what lets `Task.unfold` instantiate a closure without a `task_graph → closure`
edge.

---

## 3. The two documents

### 3.1 A spec is a `dict`, and `TaskSpec` is a name for one

Main design §4.1 settles this for every spec kind:

> **So a spec is a plain `dict` throughout**, and access is
> `spec["content_type"]` rather than `spec.content_type`.

`task_graph` design §8.7 nonetheless types its graph pass
`check_graph(specs: Mapping[str, TaskSpec])`, and `TaskSpec` is defined nowhere
in the design set. It is this module's to define, because this is where the task
registry lives, and the honest definition given main design §4.1 is an alias:

```python
TaskSpec: TypeAlias = Mapping[str, Any]        # a rendered, schema-validated task
ClosureDoc: TypeAlias = Mapping[str, Any]      # a rendered, schema-validated closure
```

**An alias, not a `TypedDict`.** A `TypedDict` would express the shape and would
be checked statically, which is tempting. It is not taken because it duplicates
the schema — main spec §4.4 makes the schema *the only enforcement point*, and a
second declaration of the same shape is a second thing to keep in step, with
`total=False` on every optional key and no way to express `const`,
`additionalProperties: false`, or a nested user-supplied schema. Main design §4.1
already rejected generated pydantic models for the same reason and named the
failure mode: a typed object built from a schema **accepts instances the schema
rejects**. The alias buys the name without the second source of truth.

What the alias does not buy is autocompletion, so the accessors in §3.2 exist to
keep key names in one file rather than spread across `check.py` and `query.py`.

### 3.2 The task spec is nested, and it is registered under the closure's name

Spec §2's key table nests the task inside the closure, and the nested object has
no `name` of its own — it carries `goal`, `inputs`, `outputs`, `resources`,
`permissions`, its own `version`, and possibly a subgraph. Something has to key
it in `TaskSpecRegistry`, and the choice is this design's.

**It is registered under the closure's `name`.** A closure names a workflow step
(`prepare_e2e`, `collect_trace`) and its task *is* that step; a separate name
would be a second thing to keep unique, a second thing to typo, and a second
thing for `Task.closure` and `check_graph` to disagree about. `task_graph`
design §3.2's `Task.closure: str | None` already resolves a *closure* name, and
`check_graph` runs over task specs — so keying both by the same string is what
makes those two pass over one object.

The consequence has to be stated rather than discovered:

> **`ClosureRegistry` and `TaskSpecRegistry` share a key space, and that is a
> decision, not an accident.** Every closure name is a task spec name and the
> reverse holds too. Main design §5.2's "a duplicate name is an error" therefore
> fires in both registries for one duplicate closure, and the two messages name
> two different kinds.

The second message is noise. So `TaskSpecRegistry.add` is called only from the
closure admission path, never from a discovery pass of its own, and the
duplicate is reported once — by `ClosureRegistry`, because that is the file the
author wrote. A task spec is not independently loadable (main design §2.3), so
there is no other path for it to arrive by.

### 3.3 `agent` is required, and this module still synthesises nothing

Spec §2.2 rev. 8 and criterion 3: a closure **always names an agent spec**, and
the load checker demands one. Main spec §4.8 rev. 9 says the same — every task
has an agent, and `kind` (`ai`, `human`, `program`) is what varies.

**This module implements exactly that: `agent` is in the closure schema's
`required` list, and check 4 (§5) always runs.** A closure with no `agent` is
rejected, naming the file, and `agent_of(closure)` never returns `None`.

That is a change of direction from rev. 1 of this document, which implemented
`agent` as optional and left the gap deliberately visible. The gap is now closed
upstream, and the reason it closed the way it did is worth keeping: the
alternative required `task_graph` to grow `agent_spec: str | None`, and

- measured on the shipped model, `Task(agent_spec=None)` raises
  `ValidationError: string_type` and `Task()` raises `missing`, so there was no
  spelling for the absent case at all;
- `scheduler._dispatch_pass` calls `agent_mgr.instantiate(task.agent_spec, tid)`
  unconditionally before `push_execution`, and `Execution.agent_id` is required,
  so the absence would have had to be handled at dispatch as well as at load.

**What has not changed is that this module synthesises nothing.** `agent` design
D1 assumed a `kind: program` spec would be supplied by the loader; it is supplied
by the **package author**, as an ordinary agent spec that the ordinary registry
admits. The loader's job is to insist that one is named — which is check 4 — and
nothing else.

### 3.4 The closure schema does not declare `version`, and the four member schemas do

Criterion 11 — *"A `version` key on a closure is rejected at load"* — needs no
check. Measured against `jsonschema` 4.26.0:

| Mechanism | Message |
|---|---|
| `additionalProperties: false`, `version` simply absent from `properties` | `Additional properties are not allowed ('version' was unexpected)` |
| additionally `not: {required: ["version"]}` | the same, **plus** `{...} should not be valid under {'required': ['version']}` |

The first names the offending key and is actionable; the second adds a message
nobody can act on beside it. So spec §1.2 is expressed as a **schema
difference** — the closure schema omits `version`, the four member schemas
declare it — rather than as a hand-written check. That is main spec §4.4's
"the schema is the only enforcement point" applied to a *negative* requirement,
and it is the cheapest form the requirement has.

The second half of criterion 11 — *"nothing at runtime reads a member's
`version`"* — is not a schema property and is tested structurally (§11).

### 3.5 Accessors, so key names live in one file

```python
def task_of(doc: ClosureDoc) -> TaskSpec: ...
def declared_handoffs(doc: ClosureDoc) -> tuple[str, ...]: ...
def named_kinds(task: TaskSpec) -> tuple[str, ...]:
    """Every handoff kind the task names, inputs and outputs together, in
       declaration order with duplicates removed."""
def phase_validators(doc: ClosureDoc) -> tuple[str, ...]: ...
def agent_of(doc: ClosureDoc) -> str: ...
def body_of(task: TaskSpec) -> Body: ...          # readme, entry, materials. §3.6
def repos_of(task: TaskSpec) -> tuple[str, ...]: ...
def monitor_of(task: TaskSpec) -> str | None: ...
def permissions_of(task: TaskSpec) -> Mapping[str, Any]: ...
```

Nine functions over two dicts, and the only reason they exist is that §3.1 chose
an alias over a typed model. They are pure, they are in `model.py`, and no check
reads a raw key directly.

### 3.6 `body`, and the seventh check

Spec §2.6 rev. 9 gives a task a **body** — always a `readme.md`, plus an
`entry.sh` when the work is programmatic, plus `materials`. It is how a package
author says what the task *is*, what to run, and how; a `goal` of at most 100
characters cannot carry that, and nothing else in the closure document did.

```python
class Body(TypedDict):
    readme: str                    # a path in the package. ALWAYS present
    entry: NotRequired[str]        # a path. Present iff the task is programmatic
    materials: NotRequired[list[str]]
```

The one place this module deviates from its own §3.1 rule: `Body` is a
`TypedDict` where every other document type is a `Mapping` alias. It is three
keys with no nesting and no user-supplied schema inside it, so the argument §3.1
makes — a second declaration of the shape drifts from the schema — is worth much
less here than the readability is. **Recorded rather than done silently**, since
§3.1 argues the other way at length.

**Check 7** joins §5's six, and it is the cheapest kind:

| | |
|---|---|
| `readme` present and resolving | Always. A task nobody can read is a step nobody can review, **and that holds for a non-leaf too** |
| `entry` resolving, when declared | Existence only |
| **`entry` and a subgraph are mutually exclusive** | A non-leaf's work *is* its subgraph (§2.1). The exclusion is `entry`-versus-subgraph, **not** body-versus-subgraph |
| every `material` resolving | Existence only |

Existence, never content. What a script does and whether a readme is any good are
not load-time questions, and §5.2's rule about actionable messages applies: a
dangling path is reported with the path.

---

## 4. The two registries

Both are `spec_loader`'s `SpecRegistry` (main design §5.2), which supplies
`add(name, spec, *, origin)`, `get(name)`, `names()`, the duplicate-is-an-error
policy, and `SpecNotFound` carrying the kind, the name, and the candidates.

```python
class ClosureRegistry(SpecRegistry):
    kind = "closure"
    # plus the index and the five queries — §8

class TaskSpecRegistry(SpecRegistry):
    kind = "task"
```

`TaskSpecRegistry` adds nothing to the base. It exists as a separate object
because main spec §4.1 makes the four registries deliberately separate, and
because `check_graph` takes it alone (`task_graph` design §8.7). Main design D2
already recorded why it has no package of its own; §3.2 above records why it has
no name space of its own either.

**Neither registry is a component `Registry`.** Main design §5.3 draws that
distinction and it holds here: the component `Registry` resolves collaborators
late and permits replacement, while a spec registry is a name table that refuses
one. `bootstrap` registers both spec registries *in* the component registry, so
both distinctions are live in the same file.

---

## 5. The six checks — `check.py`

Spec §4's six, unchanged, with the registry each one reads.

| # | Check | Reads | Criterion |
|---|---|---|---|
| 1 | The YAML validates against the schema; the name is unique | — | done by `spec_loader` before this pass; restated by the spec for completeness |
| 2 | Every kind the task names resolves, **and** appears in declared `handoffs` | `handoff_specs`, the closure | 1, 2 |
| 3 | Every kind has at least one validator — re-asserted so an escape-hatch admission is reported | `handoff_specs`' load report | 6 |
| 4 | An agent spec is named, **and it exists** | `agent_specs` | 3 |
| 5 | Every phase validator resolves, and is a validator rather than a general task | `validator_specs`, `task_specs` | 4 |
| 6 | The task's permissions cover its handoffs | the closure alone — §6 | 5 |

### 5.1 The order is the design, so it appears as a body

```python
def check_closure(doc: ClosureDoc, regs: Registries, *, origin: str) -> list[Problem]:
    problems: list[Problem] = []
    task = task_of(doc)

    # 2. resolution first, and declaration second. A kind that does not exist
    #    is a different error from a kind that exists and was not declared,
    #    and reporting the second for a typo would send the author to the
    #    wrong file.
    declared = set(declared_handoffs(doc))
    for kind in named_kinds(task):
        if kind not in regs.handoff_specs:
            problems.append(unresolved_kind(kind, origin, regs))
        elif kind not in declared:
            problems.append(undeclared_kind(kind, origin))

    # 3. only over kinds that resolved. An escape-hatch admission is a report,
    #    not a failure (criterion 6).
    problems += escape_hatch_report(named_kinds(task), regs, origin)

    # 4. absent and present-and-wrong are both errors (§3.3). The schema's
    #    `required` catches the first, so reaching here means it is named.
    if (agent := agent_of(doc)) not in regs.agent_specs:
        problems.append(unresolved_agent(agent, origin, regs))

    # 5. resolves, then is-the-right-kind. The second message is only reachable
    #    when the first passed, which is what makes it specific.
    for name in phase_validators(doc):
        if name in regs.validator_specs:
            continue
        problems.append(general_task_used_as_validator(name, origin)
                        if name in regs.task_specs
                        else unresolved_validator(name, origin, regs))

    # 6. the only check that reads nothing outside this document. §6.
    problems += check_permissions_cover(task, origin)
    return problems
```

Three properties of that ordering, each with a reason:

**Resolution before declaration, in check 2.** Both failures are about one kind
name, and the author's next action differs: an unresolved kind means "you typed
it wrong, or the file is missing", while an undeclared one means "add it to
`handoffs`". Reporting the second when the first is true sends them to the wrong
file. Spec §2.3 makes the direction explicit — the check is one-directional, and
a declared kind the task does not name is **legal**, because a closure may
declare a kind its subgraph uses internally.

**Every check appends; none raises.** The pass returns `list[Problem]` for the
reason main design §3.6 gives about the loader generally: *"A loader that dies on
the first bad spec makes fixing a package an N-round trip."* A closure with a
typo'd kind and a missing agent should report both.

**Nothing here catches an exception to decide.** `kind not in regs.handoff_specs`
is a membership test, not a `try: get() except SpecNotFound`. `SpecNotFound`
exists for a caller that wanted the spec; a checker that wanted the answer should
ask the question.

### 5.2 The report follows main design §6.2, and adds one thing

Main design §6.2's three rules are adopted as written — name both sides and sort
for determinism; **skip a closure check for a spec that already failed its own
checks** (the `skip=` gate, from Kubernetes CRD validation's "CEL validation
error messages that are not actionable"); and two exception classes,
`SpecNotFound` versus `SpecInconsistent`, from JPMS's `FindException` versus
`ResolutionException`.

The one addition is **a computed repair**, and it comes from the survey. Every
system that ships this class of message names both sides; the ones users called
actionable name a third thing — what to do, enumerated from what is actually in
scope. Dagster's is the model
(`resource_requirement.py:64`, read first-hand):

```
io manager with key 'foo' required by SourceAsset with key ["foo"] was not provided.
Please provide a IOManagerDefinition to key 'foo', or change the required key to
one of the following keys which points to an IOManagerDefinition: [...]
```

`SpecRegistry.get` already carries the candidate list for `SpecNotFound` (main
design §5.2). This makes the same list part of the closure pass's own messages:

```
closure 'collect_trace' (packages/perf/closures/collect_trace.jsonnet)
  names handoff kind 'trace_v2', which does not resolve.
  known kinds: [trace, trace_summary, kernel_ir]
  hint: 'trace' is one character away.
```

A notable negative from the same survey, recorded because it corrects an
assumption this design would otherwise have made: a tracker search on the
Kubernetes coverage message returned three issues and **none** complained it was
unreadable — all three disputed the verdict. Message legibility is not where the
pain is in these systems. Correctness of the check is (§6.3).

### 5.3 Check 3 reads a report, not a registry

Criterion 6: *"A closure assembled from a handoff kind admitted under the
escape-hatch flag loads, and reports that it did."*

The escape hatch is `handoff` design §8.5's, and it is deliberately **a return
value rather than a log line**:

```python
@dataclass(frozen=True)
class LoadReport:                     # handoff's
    admitted: list[str]
    without_validator: list[str]      # sorted. Empty is the normal case
```

So check 3 is an intersection between `named_kinds(task)` and
`without_validator`, and it produces a `Problem` at report severity rather than
an error. It does **not** re-derive the coverage — `handoff` spec §5.3 makes a
kind with no validator unadmittable in the first place, so anything in
`without_validator` is already known and already permitted.

**A collision worth naming, because both names are in scope in this file.**
`spec_loader` also has a `LoadReport`, with fields `(admitted, problems)` (main
design §3.6). Two dataclasses, one name, one shared field name, different
meanings. This module imports `handoff`'s under an alias:

```python
from spec_loader import LoadReport                       # (admitted, problems)
HandoffLoadReport = ...                                  # (admitted, without_validator)
```

and §14 O1 records that renaming one of them is the better fix and is not this
module's to make.

---

## 6. Check 6, and the two-level naming problem

This is the check spec §4 says only a closure can perform — *"it needs the task's
handoffs and its permissions together, and neither registry sees both"* — and
criterion 5 is the strongest claim in the spec:

> A closure whose task permissions do not cover its handoffs is rejected at
> load, naming both the handoff and the missing permission — **rather than
> deadlocking at dispatch**.

### 6.1 The type it was promised does not exist

`task_graph` design §3.5 introduced `Permissions` explicitly for this check, and
its consumption table names this check as the reason the type distinguishes read
from write. The signature it gives is

```python
class Grant(Model):
    path: str
    access: Access = Access.READ
    handoff: HandoffId | None = None

def covers(self, hid: HandoffId, access: Access) -> bool: ...
```

**`HandoffId` is a `uuid.UUID` subclass, and this check runs at load, where a
closure names handoff *kinds by name*.** Measured against the shipped
`task_graph/ids.py`:

```
  HandoffId('trace')         -> ValueError: badly formed hexadecimal UUID string
  HandoffId('collect_trace') -> ValueError: badly formed hexadecimal UUID string
  _coerce('trace')           -> ValueError: badly formed hexadecimal UUID string
```

The pydantic coercion path raises identically, so a declared grant naming a kind
cannot be loaded into the declared type at all. The type is correct for a
runtime question and unusable for this one.

### 6.2 The permission model speaks declared names, and instance ids never enter it

Three candidate shapes were on the table: two fields, one `str` field, or a
resolution step. The survey settles it, and it settles it with a **negative
result across three independent systems**:

| System | The artefact reference in its permission model |
|---|---|
| Kubernetes RBAC | `RoleRef` is `{APIGroup, Kind, Name}`. **There is no UID field anywhere in the RBAC model**, and `RoleRef` is immutable across updates, which pins the name for the object's life |
| Dagster | `ExternalAssetIOManagerRequirement(key=..., asset_key=key.to_string())` — the declaration-time key, stringified. Both sides of the pair are strings at check time |
| Android lint | Both sides are bare permission-name strings; `hasPermission` is set membership |

**No surveyed permission model references a runtime instance id, and none
carries one type with both meanings.** The consistent answer is a resolution
step: the model holds a declared name, and something maps name → instance when
the instance exists.

So this design's position is:

> **A grant references a handoff *kind name*. `Permissions` never holds a
> `HandoffId`.** Check 6 asks `covers(kind_name, access)` and compares strings.
> Where a runtime component needs the instance, it resolves the name at that
> point — which is `env_mgr`, at the moment it builds the zone, and which it
> already does for paths.

That is a change to `task_graph` design §3.5, so it is **reported, not made**:
§13 D2. Nothing in this document depends on it being accepted in that exact
form — what it depends on is that the two levels do not share one field.

### 6.3 The covering grammar is closed, total, and owned by this function

The sharpest finding in the survey is a failure, and it is one this check is
directly exposed to.

Kubernetes' `Covers` **wrongly rejects a legal delegation** (kubernetes#122154):
a role holding `resourceNames: ["example.com/*"]` cannot grant
`resourceNames: ["example.com/specific"]`, because `resourceNames` has no glob
support in RBAC — while the CSR admission plugin had given that string a
meaning. The maintainer's reply, verbatim:

> `example.com/* covering example.com/specific is a semantic specific to the CSR
> admission plugin, it is not part of the authorization API or RBAC.`

followed by `/remove-kind bug`. **They shipped the wrongness rather than open the
grammar**, and the check has been wrong in that case ever since.

Our grant is `(path, read|write, kind name)` and paths are exactly the kind of
string that invites a second interpreter — a trailing slash, a `*`, a symlink,
a `..`. So:

- **The covering relation is one function in this module**, and it is **total
  over the syntax the schema admits**. If the schema admits `*`, this function
  defines what `*` means, and it is the only definition.
- **No other component may interpret a path in a way this function does not.**
  `env_mgr` resolves paths to real locations and enforces containment
  (`task_graph` design §3.5 is explicit that `task_graph` carries the field
  without interpreting it); what it must not do is decide that some path *form*
  grants something this function thinks it does not.
- **Fail closed on the unbounded side.** Kubernetes asserts this with a named
  test, read first-hand — `TestCoversEnumerationNotCoveringVerbStar`: an
  exhaustive enumeration on the grant side does **not** cover a `*` on the
  requirement side. Where a requirement cannot be enumerated at all, Kubernetes
  demands the maximal grant rather than skipping the check.

**The alpha's grammar is the smallest one that is total: exact string equality,
no wildcards.** That is Android lint's held-side model (`mGrantedPermissions
.contains(permission)`), and it is the only grammar with no ambiguous case. If a
wildcard is ever wanted, adding it is a change to one function and to the schema
that admits its syntax, in that order. §14 O2.

### 6.4 The decision form and the message form are different

Kubernetes' comparator decomposes the requirement into atoms so that each is
either wholly covered by one grant or not covered at all, and then **re-compacts
the uncovered atoms** for the human — with the recompaction added later, from a
`TODO` in the original:

> `Because the breakdown is down to the most atomic level, we're guaranteed that
> each mini-servant rule will be either fully covered or not covered by a single
> owner rule` … `TODO: it might be nice to collapse the list down into something
> more human readable`

With §6.3's exact-equality grammar our atom *is* `(kind, access)`, so there is
nothing to decompose and nothing to recompact — the two forms coincide. This is
recorded because it stops being true the moment a wildcard is added, and the
recompaction is real work rather than formatting.

The message names three things, per §5.2:

```
closure 'collect_trace' (packages/perf/closures/collect_trace.jsonnet)
  task produces handoff 'trace' but its permissions grant no write for it.
  grants held: trace(read), kernel_ir(read)
  hint: add a write grant for 'trace', or remove it from the task's outputs.
```

### 6.5 Existence and coverage are separate questions, and they fail separately

Kubernetes checks rule coverage statically and deliberately does **not** check
that a `RoleBinding`'s target exists — a dangling reference is legal, resolved at
evaluation, and logged-and-continued past. The two questions are separable and it
separates them.

Here they are also separable, and this design keeps them apart in the *report*
while failing both at load:

- **Existence** is checks 2, 4 and 5, and each names a registry and a candidate
  list.
- **Coverage** is check 6, and it reads nothing outside the closure document.
  It is decidable with no registry at all, which is why §5.1 puts it last and
  passes it no `Registries`.

The practical payoff is the layering gate: a closure whose kind does not resolve
gets one message about the kind, and check 6 still runs and still reports a
missing grant, because the grant is about a *name* the author wrote and can fix
whether or not the kind exists.

### 6.6 Why the criterion is right, and what would make it wrong

The survey's answer to "does anyone actually do this statically" is **yes, and
the dividing line is not maturity**:

| System | Static coverage check | Why |
|---|---|---|
| **Dagster** | Yes, blocking, raises | Ops *declare* their required resource keys |
| **Kubernetes RBAC** | Yes, blocking at admission | Rules are declared on both sides |
| **Android lint** | Exists, **fails open** at four separate points | The requirement must be **inferred** from code |
| **systemd** | **Absent** — `systemd-analyze security` never reads `ReadWritePaths=` | A unit never declares what it will touch, so the question is unaskable |

**Every implementation that actually performs the check is blocking.** Nobody
ships an advisory version anyone treats as sufficient.

The corollary is the thing to guard: our check is reachable **because a closure
declares its handoffs**. Lint fails open because a false positive on inferred
code is unfixable except by suppression; that reasoning does not transfer to us,
since a rejection here is always attributable to something the author wrote.
**But any feature that lets a task consume a handoff it did not declare converts
us from Dagster's position into lint's**, and the pressure to add a suppression
switch would follow immediately. If one is ever added, systemd's form is the
cheapest known: a per-item marker inside the declaration (a leading `-` on the
path, meaning tolerate absence), visible in review, with no separate config
surface — not a global flag.

---

## 7. Where the pass runs

Spec §1.1 is emphatic that a closure does nothing at runtime and leaves the
load-time placement open. Main design §6 answers it — and the answer, as
written, has two defects.

### 7.1 It moves to the composition root

Main design §3.6 puts the pass inside `spec_loader/package.py::load_package`:

```python
def load_package(pkg: TaskPackage, registries: Registries) -> LoadReport:
    ...
    # 5. the closure pass — only now, and only over specs that got this far (§6).
    problems += check_closures(registries, skip=failed_names(problems))
```

**Defect one, the import.** `check_closures` is this module's; main design §2.3
says *"`spec_loader` imports nothing from this repository. It is the leaf, and it
must stay one."*

**Defect two, the ordering, which is the serious one.** Main design §7's
composition root calls `load_package` **once per package**:

```python
    for pkg in packages:
        load_package(pkg, Registries(r))
```

With two packages the closure pass runs after the first — with the second
package's specs in no registry — and again after the second. That is exactly what
main design §6.1 forbids in its own words: *"Resolve-during-load cannot see the
far side of a binding, because the far side may not be loaded yet."* And main
spec §4.3 makes cross-package references a supported case: *"Two packages may
reference each other, and they do it themselves."*

So the call leaves `load_package` and joins `check_graph` at the composition
root:

```python
def build_registry(..., packages: Sequence[TaskPackage] = ()) -> Registry:
    ...
    for pkg in packages:
        load_package(pkg, Registries(r))       # no closure pass inside
    problems = check_closures(Registries(r))   # ← here. All packages loaded
    check_graph(r.get("task_specs"))           # task_graph design §8.7
    r.register("scheduler", Scheduler(r))
```

Main design §7 already justifies this line for `check_graph`, and the
justification is the closure pass's word for word: *"it runs at this line because
this is the only moment when every spec is present and nothing has run."* Fixing
the ordering fixes the import as a side effect — `bootstrap` is already *"the
only module importing all of them"*.

This is a change to the main design, so §13 D3 records it.

**Independent confirmation, from a system that hit this in production.** Dagster
makes its per-job coverage check *conditional* and defers the mandatory one to
repository build, with the reason in a source comment read first-hand
(`repository_data_builder.py:460`):

```python
# Late validate all jobs' resource requirements are satisfied, since
# they may not be applied until now
```

What happens in between is top-level resource binding — a job that looks
incomplete in isolation is legal, because something later supplies the rest. Our
analogue is a closure whose handoff kind lives in another package.

### 7.2 It is not the graph pass, and the two are sequenced

`check_graph` carries the two graph-level checks spec §4.1 explicitly declines —
that no handoff produced inside a subgraph is consumed outside it, and that a
non-leaf declares no resources. It is `task_graph`'s (its design §8.7), it runs
over **task specs** rather than over closures, and it runs **after** this pass.

The ordering matters in one direction only: a task spec whose closure failed
should not be walked for containment, because its handoff names may not resolve.
So `check_graph` receives the same `skip` set. Three modules touch one object at
this line — `closure` owns the registry, `task_graph` owns the pass,
`bootstrap` calls it — and this is the only document positioned to say so.

### 7.3 `Registries`

Main design uses the name three times and defines it nowhere. The shape the six
checks need is a read-only view, and nothing more:

```python
class Registries(Protocol):
    handoff_specs:   SpecRegistry
    validator_specs: SpecRegistry
    task_specs:      SpecRegistry
    agent_specs:     SpecRegistry
    closures:        SpecRegistry
    def for_kind(self, kind: str) -> SpecRegistry: ...
```

A `Protocol` rather than a class, so a test supplies five dicts and no
`bootstrap`. It is defined in `spec_loader` and not here, because `load_package`
takes one too and `spec_loader` may not import `closure`. **Main design §3.7 rev. 2
now carries that definition**, which is where a reader will look for it.

**Rev. 2 of this document also wanted a `handoff_report` field on it, and that is
withdrawn.** Its type is `handoff.LoadReport`, so a Protocol in `spec_loader`
declaring it would make the leaf name a type in `handoff` — the same rule that
moved the closure pass out of `load_package` in the first place (§7.1, defect
one). Check 3's report is a separate argument:

```python
def check_closures(regs: Registries, handoff_report: HandoffLoadReport, *,
                   skip: Set[str] = frozenset()) -> list[Problem]:
    """Every closure in `regs.closures`, in sorted name order, except those in
       `skip`. Returns problems; raises nothing."""
```

A parameter rather than a field also says something true: the escape-hatch report
is a fact about **this load**, while the five registries outlive it.

**Sorted name order**, so a package with two broken closures reports them the
same way twice — the determinism rule main design §6.2 takes from OPA's
`util.KeysSorted`.

---

## 8. The read-only query helpers

### 8.1 The five queries

Spec §3.2, with the shape §8.3 argues for:

```python
class ClosureRegistry(SpecRegistry):
    def handoff_kinds(self, closure: str) -> tuple[str, ...]:
        """Every kind this closure touches, inputs and outputs. Raises
           SpecNotFound if `closure` is not a closure."""

    def validators_for(self, closure: str) -> tuple[str, ...]:
        """Every validator that will run: the phase validators, plus the
           per-handoff ones joined through the handoff registry."""

    def closures_using_kind(self, kind: str) -> tuple[str, ...]:
        """Reverse. Raises SpecNotFound if `kind` is not a known handoff kind;
           returns () for a known kind no closure uses."""

    def closures_using_agent(self, agent: str) -> tuple[str, ...]:
        """Same, for an agent spec."""

    def agent_of(self, closure: str) -> str:
        """The agent spec name. Always present — §3.3."""
```

Spec §3.2 spells the reverse pair `closures_using(handoff_kind)` and
`closures_using(agent_spec)` — one name, two argument types. They are two
methods here because the argument is a `str` in both cases and nothing in the
call would say which registry to validate it against. A single method would need
a discriminator argument, which is the same two methods with a worse signature.

### 8.2 The index is built once and frozen, and post-load mutation is impossible

Nobody in the survey maintains a reverse map incrementally, and the two ends of
the spectrum are both instructive:

- **dbt** has no invalidation at all. `build_parent_and_child_maps` is a full
  O(V+E) rebuild, and it is called at **six** separate sites, each immediately
  before a consumer, with the policy stated in a comment: `# parent and child
  maps will be rebuilt by write_manifest`. That is what "eager but not frozen"
  actually costs.
- **Sphinx** is the one system whose index outlives the load, and the price is a
  `clear_doc` **and** a `merge_domaindata` obligation on every owner — including
  third-party extensions, through the `env-purge-doc` event — with
  `StandardDomain.clear_doc` degenerating into five independent linear scans and
  an unresolved `# XXX duplicates?` in the merge path.
- **Cargo** inverts the graph destructively per invocation and throws it away;
  consistency is a non-question because the graph does not outlive the call.

Our world is closed: five registries, all loaded before any query, and spec §1.1
says a closure is read to assemble a graph and never again. So the index is built
**once, at the end of the closure pass, over the closures that passed**, and then
frozen.

**Frozen structurally, not by convention.** `ClosureRegistry.add` raises after
the index is built:

```python
def freeze(self) -> None:
    """Build the reverse index and refuse further registration."""
```

Sphinx is the argument for making that impossible rather than discouraged: an
index that can outlive its build is one somebody will eventually have to purge.

### 8.3 "Not found" and "found, used by nothing" are different answers

dbt has this right in the data and loses it at every call site. The map
pre-populates `{n.unique_id: [] for n in nodes}`, so a known-but-unused node is a
present key with `[]` and an unknown id is a `KeyError` — and then all six
consumers guard with `if unique_id in child_map`, collapsing the two. The
user-facing message inherits the conflation and has to hedge across three causes:

> `The selection criterion '{spec_raw}' does not match any enabled nodes`

— which covers a typo, a real-but-unused node, and a real-but-disabled one
alike.

Cargo keeps them apart more cheaply: it resolves the `--invert` argument against
the package catalogue **before** touching the graph, so an unknown package is an
error from spec resolution and a real-but-unused package yields a one-node tree.
**The two cases never share a code path.**

That is the shape adopted here. `closures_using_kind` validates its argument
against `handoff_specs` first and raises `SpecNotFound`; only then does it read
the index, which returns `()` for a known kind nobody uses. The lookup that
decides *which* answer applies is a distinct step from the index read.

This matters more here than in dbt, because the question these queries exist to
answer is *"what breaks if I change this"* — and an empty answer is the one a
caller acts on.

### 8.4 Membership is derived, not restated

dbt's `build_parent_and_child_maps` chains a hand-written tuple of seven
collections, and `build_node_edges` silently discards any edge whose target is
outside that set (`if unique_id in forward_edges.keys()`). Issue 14436 is that
bug reaching users: `depends_on` correct, `child_map` incomplete, no error
raised.

That is the second independent instance of a failure this design set has already
recorded once — `validator` design §10.5 took it from Airflow's asset orphanage,
where materialisation through an `AssetAlias` is not one of the four tables the
join covers. Two systems, same shape: **a derived static index goes wrong by
forgetting a reference kind, and it goes wrong silently.**

So the index loop does not restate what participates:

- Its **sources** are the accessors in §3.5, so adding a referential key to the
  closure schema without adding it to the index is a change in one file that
  fails to compile in the other.
- An edge whose target is **not** in the corresponding registry does not get
  dropped. It cannot occur — checks 2, 4 and 5 already rejected the closure, and
  the index is built only over closures that passed (§8.2). If it occurs anyway,
  that is a programming error and the builder raises rather than skipping.

The second point is the one dbt gets wrong, and it is free for us only because
the pass runs first.

### 8.5 The third edge kind, and `validator` O7

`validator` design §10.5 defines the reverse index for a validator:

```python
def users_of(self, name: str) -> list[str]:
    """Static: which handoff kinds bind this validator. From the specs."""
```

**Handoff kinds only.** But spec §2.4 of this document says a closure's phase
validators *"are a property of the task, not of any one handoff kind, which is
why the handoff specs cannot carry them"*. So a validator that two closures run
in every `output_validation` phase is invisible to `users_of`, which reports it
as used by nothing — the exact false-negative §8.4 is about, in the module that
recorded the failure mode.

`validator` design O7 asks who owns the enumeration of reference kinds. **This
module is what makes the question concrete**, because it adds the third kind.

The answer here is the narrow one: `ClosureRegistry` owns the closure→validator
edge and exposes it, and `ValidatorSpecRegistry.users_of` unions its own answer
with this one at the composition root, where both registries exist. Neither
imports the other.

```python
def closures_using_validator(self, name: str) -> tuple[str, ...]:
    """Reverse, for a phase validator. The edge `users_of` cannot see."""
```

This is a **sixth query, not in spec §3.2**, and §13 D4 declares it. It is added
because leaving it out ships a known-wrong answer from a different module, and
because §3.2's stated purpose — *"what breaks if I change this"* — is precisely
what the missing edge breaks.

O7 itself is not closed: nothing yet owns the enumeration, and now there are
three edge kinds in three modules. §14 O3.

### 8.6 The queries answer within the loaded world, and say so

Bazel's `rdeps` makes the caller name the universe, and the documentation's own
worked example shows the failure being a **confident empty answer** rather than
an error:

> But the result of that query with `--universe_scope` is only `//my:target`;
> **none of the reverse dependencies of `//my:target` are in the universe, by
> construction!**

We get the universe for free — every registry is loaded before any query — but
"what breaks if I change this" is only true of the loaded catalogue, and a task
package that is not loaded is not in the answer. Every reverse query's docstring
says so. That is the whole mitigation, and it is proportionate: unlike Bazel we
cannot get the universe *wrong*, only narrow.

### 8.7 They stay one-hop

dbt ended up with the reverse relation in two representations — `child_map` for
point queries, a networkx `DiGraph` for `+model` traversal — which is
duplication, but also a boundary that held: `child_map` never grew traversal
operators. Bazel's is the stronger precedent in the other direction: when node
identity gained a configuration dimension, `cquery` **removed** `allrdeps` rather
than generalising it.

So these five (six) queries are one-hop lookups and stay that way. Transitive
questions — "everything downstream of this kind" — are graph questions and belong
to whatever eventually owns graph-level composition (spec §6). If a closure could
ever reference one handoff kind under more than one context, `closures_using_kind`
stops having a single well-defined answer, and the precedent says restrict the
query rather than widen it.

### 8.8 What they cannot answer, and it is a real gap

Spec §6: the reverse indexes answer *who is affected* when a shared kind changes;
what nothing answers is whether the change is **safe**.

The survey confirms that as a genuine gap rather than something we are failing to
copy. dbt ships **both halves and never joins them**:

- `check_for_model_deprecations` walks `child_map` and warns each consumer.
- `same_contract` is a real structural compatibility diff, with named breakage
  categories (`columns_removed`, `column_type_changes`,
  `enforced_model_constraint_removed`, …) rather than a bool.

And the severity decision consults neither: warn-vs-error keys on whether the
model *declares a version*. `same_contract(self, old, adapter_type)` does not
take the manifest, so it structurally cannot see consumers.

Two details are transferable if this is ever closed: gate the expensive
structural walk behind a cheap fingerprint (dbt compares a contract checksum
first), and return **named breakage categories** rather than a boolean, because
that is what makes the message actionable. Not built here. §14 O4.

---

## 9. Nothing at runtime

### 9.1 Criterion 8 is proved by attribution, not by absence

Criterion 8: *"The scheduler never reads a closure. Verified the way criterion 14
is: a spy over the closure registry records no read from a scheduler frame across
a full submit → dispatch → complete cycle."*

The pattern is shipped. `tests/task_graph/test_authority.py` subclasses the real
component (`SpyHandoffMgr(HandoffMgr)`), registers it in place of the real one,
asserts over `spy.log`, and — the part that makes a negative result mean
something — ships a meta-test, `test_the_spy_would_catch_a_scheduler_write`,
proving the spy would have noticed.

**But the assertion here cannot be "no read at all", and that is the whole
difficulty.** Main design §7 narrows the prohibition deliberately:

> the prohibition is on the *scheduler*. A `Task` transition may resolve
> `closures` — `replace_with` does, to satisfy criterion 51 — and that adds no
> `Scheduler → spec registry` edge.

And `Task.unfold` reads one on every non-leaf's `enter_phase(RUNNING)`. So a run
that exercises a subgraph — which criterion 8's "full cycle" must, or it proves
nothing — contains legitimate closure reads.

`test_authority.py`'s existing `spans()` helper attributes reads to *agent
spans*, which is a different question. This test attributes to the **calling
frame**:

```python
class SpyClosureRegistry(ClosureRegistry):
    def get(self, name):
        self.log.append(_caller_module())      # inspect.stack, one frame up
        return super().get(name)
```

and asserts that no entry names `task_graph.scheduler`. Reads from
`task_graph.models` (`unfold`, `replace_with`) are expected and are asserted to
be *present*, because a test that passes because nothing read anything is a test
that proves nothing. The meta-test plants a read inside a scheduler method and
asserts the spy catches it.

Frame inspection is fragile and it is chosen anyway: the alternative is a
`caller` argument threaded through `SpecRegistry.get`, which changes a shared
base class to satisfy one test. §14 O5 records the fragility.

### 9.2 Who builds the root `Task` is unowned, and this module does not claim it

Spec §3.1: *"Whoever assembles a graph looks one up; nothing else does."* The
assembler is never named, in this spec or any other.

The **subgraph** half is owned: `Task.unfold` instantiates `self.closure`'s
declared expansion and returns subtasks the caller submits (`task_graph` design
§8.5). The **root** half is owned by nobody, and the one sentence in the design
set that assigns it names the component criterion 8 forbids — main design §7's
*"the scheduler is what assembles it"*.

Almost certainly that clause is loose prose about registration order:
`Scheduler.submit` takes `Task` objects a caller already built, and `submit` is
`task_graph`'s entry point. But as written it is the only attribution of the job,
so §13 D5 records it rather than letting it sit.

**This module does not grow an `instantiate(name) -> Task` helper**, and the
reason is spec §1.1's:

> This is what keeps it from becoming a fifth object. The closure is the one
> place that sees all four parts, so every cross-object question wants to live
> here; resisting that is the design.

A helper that returns a `Task` would make `closure` import `task_graph`, which
§2.1 forbids, and would make the closure module the thing that decides what a
task's initial `permissions`, `is_start` and `is_end` are. The two named callers
in the spec set are both outside — `demo` spec §4.1's `show` verb, and its
`--dry-run` (criterion 11: *"resolves every closure, validates every spec, and
dispatches nothing"*). Those are also the **only named non-test callers of the
query helpers anywhere in the spec set**, which is worth knowing when judging how
much §8 is worth.

### 9.3 The demo's two verbs are the acceptance surface

`show` needs `handoff_kinds`, `validators_for` and `agent_of`; `--dry-run` needs
`check_closures` with no scheduler registered. Both are satisfied by what §7 and
§8 already build, and neither needs a runtime object — which is the concrete form
of "nothing at runtime" rather than a restatement of it.

---

## 10. Build versus adopt

| Concern | Considered | Chosen | Why |
|---|---|---|---|
| The registry base | A fifth bespoke registry; `SpecRegistry` | **`SpecRegistry`** | Main design §5.2 already fixes the collision policy, the error types and the candidate list. A fifth policy would be a fifth thing to keep in step |
| The document type | `TypedDict`; pydantic model; `Mapping` alias | **`Mapping` alias + accessors** | §3.1. A second declaration of the schema's shape is a second source of truth, and main design §4.1 measured the failure mode |
| The reverse index | On-demand scan; eager and mutable; **eager and frozen** | **Eager and frozen** | §8.2. Sphinx priced the mutable option: a purge and a merge hook on every owner, forever |
| The covering grammar | Glob; prefix; **exact equality** | **Exact equality** | §6.3. kubernetes#122154 is what an under-specified grammar costs, and they shipped the wrongness rather than fix it |
| Criterion 8's spy | `caller=` on `SpecRegistry.get`; frame inspection | **Frame inspection** | §9.1. The alternative changes a shared base class to satisfy one test |
| Error reporting | Bespoke; `spec_loader`'s `Problem` + main design §6.2's rules | **`Problem` + §6.2** | One report format across five spec kinds, and §5.2 adds only the computed repair |

Nothing is adopted from outside the repository. Two external systems shaped
decisions without contributing code — Kubernetes RBAC's covering relation (§6.3,
§6.4) and dbt's manifest maps (§8.3, §8.4) — and both entered as constraints,
not as designs to copy.

---

## 11. Test plan

`tests/closure/`. Every criterion maps to a named test.

| # | Criterion | Test | File |
|---|---|---|---|
| 1 | An unresolvable kind is rejected, with the kind name in the message | `test_unresolved_kind_names_it` | `test_check.py` |
| 2 | An input absent from declared `handoffs` is rejected; the reverse loads | `test_declared_handoffs_one_directional` | `test_check.py` |
| 3 | A missing agent spec is rejected, and **so is a closure naming none** | `test_missing_agent_spec_rejected`, `test_no_agent_key_rejected` | `test_check.py` |
| 4 | A phase validator resolving to a general task is rejected | `test_phase_validator_is_not_a_task` | `test_check.py` |
| 5 | Permissions not covering handoffs is rejected, naming both | `test_permissions_cover_handoffs` | `test_permissions.py` |
| 6 | An escape-hatch kind loads, and reports that it did | `test_escape_hatch_reported` | `test_check.py` |
| 7 | Two closures may share a kind, an agent and a validator | `test_sharing_is_legal` | `test_registry.py` |
| 8 | The scheduler never reads a closure | `test_scheduler_never_reads_closure`, `test_the_spy_would_catch_a_scheduler_read` | `test_authority.py` |
| 9 | Every §3.2 query answers without a caller-written join; none mutates | `test_queries_need_no_join`, `test_queries_do_not_mutate` | `test_query.py` |
| 10 | The six reference-workflow steps are each one closure, and the set loads | `test_six_step_shape_loads` | `test_reference_shape.py` |
| 11 | A closure declares no version; each member declares its own | `test_closure_version_rejected`, `test_no_runtime_version_read` | `test_schema.py` |

Six tests carry more than their criterion, and are worth naming:

**`test_unresolved_kind_names_it` also asserts the ordering of §5.1.** A closure
with a kind that neither resolves nor is declared must report the *resolution*
failure, not the declaration one.

**`test_permissions_cover_handoffs` asserts the fail-closed direction.**
Kubernetes' `TestCoversEnumerationNotCoveringVerbStar` is the model: the test
that matters is the one proving the check does **not** accept something it cannot
justify. Ours: a grant list that enumerates every kind still does not cover a
kind it does not name.

**`test_queries_do_not_mutate` is a structural test, not a behavioural one.** It
calls `freeze()`, then asserts `add()` raises — §8.2's claim is that mutation is
impossible, and a test that merely observes no mutation would not distinguish
that from "nobody happened to".

**Criterion 10 has no artefact to check against, and the test says so.** The six
reference steps live in a task package that does not exist — `demo` spec §1.3
puts the reference workflow out of scope, and the demo's own graph is three
tasks. So `test_six_step_shape_loads` builds a **fixture package of six
closures** named for the kickoff report's loop (prepare e2e, collect, analyse,
optimise, integrate, verify), each with the handoffs and phase validators the
step needs and a trivial task body, and asserts the set loads with no problems.
That tests **expressibility** — the schema and the six checks admit the shape —
which is what criterion 10 says. It does not test the real workflow, and the
test's docstring says which of the two it is. §13 D6.

**`test_no_runtime_version_read` is criterion 11's second half**, and it is a
grep-shaped test rather than a behavioural one: no module outside `spec_loader`
and the schemas reads a `version` key. The same shape as
`task_graph`'s criterion-42 test, which blanks the structural fields and asserts
nothing changes.

**Property tests are not used here.** Every check is a membership test over a
finite declared set; the interesting inputs are named cases, not generated ones.

---

## 12. Implementation order

| Step | What | Depends on |
|---|---|---|
| 1 | `model.py` — the two aliases and the six accessors | — |
| 2 | The two schemas, in `spec_loader/schemas/` | main design §4 |
| 3 | `task_registry.py`, `registry.py` without the index | `spec_loader` §5.2 |
| 4 | `check.py` — checks 2, 4, 5, then 3 | 1–3, `Registries` |
| 5 | Check 6 and the covering function | 4, and D2's resolution |
| 6 | The composition-root move (§7.1) | 4, main design §7 |
| 7 | The index and `query.py` | 3, 6 — the index is built after the pass |
| 8 | `test_authority.py` and the spy | 7 |

Step 5 is the one that can stall: it needs `Permissions` to hold a kind name
rather than a `HandoffId` (D2). Until that lands, check 6 is implementable
against the closure document alone — it reads no registry (§6.5) — so the block
is on the *shared type*, not on this module's work.

---

## 13. Deviations from the spec

| # | Where | This design | Why |
|---|---|---|---|
| **D1** | ~~Spec §2.2, criterion 3 — a closure may name no agent spec~~ | **No longer a deviation.** Spec §2.2 rev. 8 makes `agent` required; §3.3 implements it, and `agent_of()` never returns `None` | Rev. 1 implemented the old wording as written and declined to synthesise a spec into the gap, on the ground that doing so would make §2.2's "names no agent spec" true of the file and false of the loaded object. That reasoning held, and the resolution went the other way: main spec §4.8 rev. 9 removed the gap instead of filling it. The part of rev. 1 that survives unchanged is the refusal — **the loader still synthesises nothing**; a `kind: program` spec is written by the package author and admitted by the ordinary registry |
| **D2** | ~~`task_graph` design §3.5 — `Grant.handoff: HandoffId`~~ | **No longer a deviation.** `task_graph` design §3.5 rev. 12 adopts it: `Grant.kind: str | None`, `covers(kind, access)` | The type was introduced for this check and cannot serve it: check 6 runs at load over kind names, and `HandoffId('trace')` raises `ValueError: badly formed hexadecimal UUID string`, coercion path included. The survey is a three-system negative — Kubernetes RBAC has **no UID field anywhere in its permission model**, Dagster carries `asset_key: str`, lint compares bare strings — so name-plus-resolution-step is the only shape with precedent. A `task_graph` change; reported |
| **D3** | ~~Main design §3.6 step 5 — the closure pass inside `load_package`~~ | **No longer a deviation.** Main design §3.6 rev. 2 adopts it and D8 there records it | Two defects: `spec_loader` may not import `closure` (§2.3), and `load_package` runs per package, so the pass fires before a second package's specs exist — which §6.1 forbids and §4.3 of the main spec makes a supported case. Dagster hit the same ordering problem and its fix is the same shape (`# Late validate … since they may not be applied until now`). A main-design change; reported |
| **D4** | Spec §3.2's five queries | **Six.** `closures_using_validator` is added | A closure's phase validators are an edge `ValidatorSpecRegistry.users_of` structurally cannot see — §2.4 says the handoff specs cannot carry them — so without it `users_of` reports a validator two closures run as used by nothing. That is the failure `validator` design §10.5 records from Airflow #58058, and dbt#14436 is a second instance |
| **D5** | Main design §7 — *"the scheduler is what assembles it"* | **Not implemented, and contradicted** | Criterion 8 forbids the scheduler reading a closure. The clause is almost certainly loose prose about registration order, but it is the only attribution of the job in the design set, and the job is real: nothing owns turning a closure into the root `Task`. §9.2 |
| **D6** | Criterion 10 — the six reference steps load | Tested against a **fixture package**, not the real workflow | The six closures have no artefact in this repository: `demo` spec §1.3 puts the reference workflow out of scope. The fixture tests expressibility, which is what the criterion says; the test's docstring states which of the two it is. **Also: criterion 10 cites main spec criterion 7, which is "the producing agent cannot reach the validation's context". It means criterion 11.** A cross-reference error in a frozen spec; reported, not edited |

---

## 14. New open questions

| # | Question |
|---|---|
| **O1** | **Two types are named `LoadReport`** — `spec_loader`'s `(admitted, problems)` and `handoff`'s `(admitted, without_validator)`. Both are in scope in `check.py`, which imports the second under an alias (§5.3). Renaming one is the better fix and belongs to whichever module is next to touch it, not here |
| **O2** | **The covering grammar is exact equality, and the first wildcard is a decision, not an increment.** §6.3 argues it is the only *total* grammar available. Adding `*` means defining what it means in one function and admitting its syntax in the schema, in that order — and kubernetes#122154 is what happens when a second component gives the same string a different meaning. Nobody has asked for a wildcard yet; when they do, this is the question |
| **O3** | **`validator` O7 is now concrete and still unowned.** There are three edge kinds for "who uses this" — a kind naming a validator, a composite naming a member, and a closure naming a phase validator — in three modules, and nothing owns the enumeration. §8.5 unions two of them at the composition root, which works and does not scale to a fourth |
| **O4** | **"Is this change safe" needs a diff, and nobody has one.** Spec §6 records the gap; the survey confirms dbt has both halves and never joins them, keying warn-vs-error on whether a model declares a version rather than on whether anything consumes it. Two transferable details when it is built: gate the structural walk behind a cheap fingerprint, and return named breakage categories rather than a bool (§8.8) |
| **O5** | **Criterion 8's spy inspects the calling frame**, which is fragile — a decorator, a `functools.wraps`, or a C-level call would change what it sees. The alternative is a `caller=` parameter on `SpecRegistry.get`, which changes a base class shared by five registries to satisfy one test. Chosen the fragile one; recording that it is fragile (§9.1) |
| **O6** | **The two registries share a key space** (§3.2), and the choice is defensible but has one untested consequence: a package that wants two closures over one task spec cannot express it, and a task spec reused by two closures is inexpressible too. Whether that is a limitation or a property is not settled — spec §2 nests the task inside the closure, which implies one-to-one, but never says so |
| **O7** | **Main design O9's placeholder problem is nearly unrepresentable here, and that narrows O9 rather than closing it.** Four of a closure's keys are referential — `agent`, `handoffs`, `validators`, and the task's `inputs`/`outputs` — so `"TODO"` does not resolve and the closure pass rejects it. The residue is `name`, `description` and `task.goal`. Worth knowing before paying for O9's structural fix: in this module it would buy three free-prose fields |
