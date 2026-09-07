# Validator — Design

| | |
|---|---|
| Status | Draft — stage two of spec → design → test & code |
| Revision | 3 — 2026-08-27. **A validator's implementation is a body, following spec §6.1 rev. 8.** `readme.md` always, `entry.sh` when programmatic, plus its own `materials` (§3.8) — the same shape `closure` spec §2.6 gives a task. The registered Python callable, its implementation registry and its `inspect.signature` argument check are all withdrawn (§10.2, §10.6); D6 records what that costs, because it costs a real check. Everything about *verdicts* is untouched. (rev. 2: 2026-08-27. **The stage-three consistency pass.** The binding field is `inputs`, not the `binds_to` this document named in §10.3 and `handoff` §8.3 read. `Verdict` is `handoff`'s and is re-exported, not re-declared (§2, §11.1). The SDK's measured cost replaces the superseded figures (§8.1, §13). §8.2's per-phase-session decision is reconciled with `agent` O6 rather than left as two answers. (rev. 1: 2026-08-26. Initial)) |
| Implements | [`spec.md`](spec.md) rev. 8, acceptance criteria 1–21 |
| Language | Python ≥ 3.10. pydantic v2; the harness SDK behind a seam (§8.1) |

---

## 1. Scope

This document turns [`spec.md`](spec.md) into files, classes, and interfaces. It
adds no requirements. Where it makes a choice the spec left open, the choice is
stated here; where implementing the spec exposed a contradiction, §14 says so
rather than papering over it.

The spec's 21 acceptance criteria are the definition of done. §12 maps every one
to a named test.

**This document specifies interfaces, not bodies.** A method is a signature and
a sentence. A body appears only where the ordering of steps *is* the design
decision — which is §5.2's phase runner, §6.3's fold, and nothing else.

### 1.1 What this module owns

- The `Validator` and `Reducer` protocols, and the pydantic model that admits a
  validator spec (§3).
- `ValidatorSpecRegistry` — one of the four the main design reserves — with its
  load-time checks and its two indexes (§10).
- The composite, one level deep (§6).
- The two validation phases inside `TaskRunner`, and their skip logic (§5, §7).
- The load-time separation check, criterion 11 (§9).
- The hook seam through which a producer is denied the checking standard (§8).

### 1.2 What it does not

| Deferred to | What |
|---|---|
| [`../../handoff/docs/design.md`](../../handoff/docs/design.md) | the validation record's storage, its digest exclusion, `HandoffKind`'s side of the binding |
| [`../../docs/design.md`](../../docs/design.md) | the loader, the schemas, `SpecRegistry`, the closure pass |
| [`../../env_mgr/docs/spec.md`](../../env_mgr/docs/spec.md) | building a sandbox, and enforcing a permission zone |
| [`../../task_graph/docs/design.md`](../../task_graph/docs/design.md) | the scheduler, the pools, `TaskRunner`'s dispatch contract |

Two of those are not merely deferred but **currently absent**, and the design
must not read as though they exist: `env_mgr` has no sandbox implementation
(§8.3), and `TaskStatus` does not yet carry the two validating members (§5.1).

---

## 2. Layout and import graph

The `validator/` package that [`../../docs/design.md`](../../docs/design.md) §2
reserved. `docs/` already exists.

```
agent_sys/validator/
├── __init__.py
├── docs/
│   ├── spec.md
│   └── design.md              this document
├── protocol.py                Validator, Reducer, Dimension, Strength
│                              (Verdict is re-exported from handoff, not declared)
├── spec.py                    ValidatorSpec, Body — the admission model. §3.2, §3.8
├── registry.py                ValidatorSpecRegistry(SpecRegistry). §10
├── composite.py               the one-level composite and the reducer table. §6
├── reducers.py                `all` — and nothing else in the alpha. §6.2
├── phase.py                   the two validation phases. §5
├── history.py                 the verdict stack, and the skip decision. §7
├── separation.py              criterion 11's declaration comparison. §9
├── report.py                  PhaseOutcome and how a qualified pass renders. §11
└── errors.py                  ValidatorInvalid, SeparationViolation, NestedComposite
```

The schema is **not** here. `validator.schema.json` lives in
`spec_loader/schemas/` with the other four, for the packaging reason
[`../../docs/design.md`](../../docs/design.md) §2.2 measures.

### 2.1 The import graph

```
             spec_loader ────────────────┐
                  │                      │
                  ▼                      ▼
validator.errors ◄── validator.spec ── validator.registry
      ▲                    │                    │
      │                    ▼                    ▼
      ├── protocol ◄── composite ◄── reducers    separation
      ├── history                                    │
      └── report ◄─────── phase ◄───────────────────┘
```

Acyclic, and two rules keep it so.

**`reducers`, `history` and `separation` import nothing from this package except
`errors` and `protocol`.** They are pure functions over, respectively, a list of
verdicts, a record, and two sets of paths. That is what makes each testable
without a registry, a runner, or a store.

**`phase` is the only module that imports a manager.** It resolves
`handoff_mgr` and `agent_mgr` from the component `Registry` at call time, never
by import — the rule `task_graph` design §2 already establishes and that this
package does not get to break.

---

## 3. A validator, and what admits one

### 3.1 The two protocols

Spec §4.1's four elements, plus the reducer §6 injects.

```python
class Dimension(str, Enum):
    COMPLETENESS = "completeness"
    USABILITY = "usability"
    TRUSTWORTHINESS = "trustworthiness"


class Strength(str, Enum):
    STRONG = "strong"
    LONG_TERM_STRONG = "long_term_strong"
    WEAK = "weak"


class Validator(Protocol):
    brief: str
    inputs: tuple[str, ...]          # handoff kind names, declared
    dimension: Dimension
    strength: Strength

    def __call__(self, handoffs: dict[HandoffId, Handoff]) -> dict[HandoffId, bool]:
        """One verdict per input handoff. Every key of `handoffs` appears in
           the result; a validator that omits one is rejected by §6.4.

           **This is the shape of a verdict, not the shape of an
           implementation.** From rev. 3 the implementation is a body — a
           `readme.md` and optionally an `entry.sh` (§3.8) — and this Protocol
           is what the phase runner sees after that body has run. A composite
           satisfies it directly; a leaf is satisfied by the adapter that
           executes the body and collects the result."""


class Reducer(Protocol):
    name: str

    def __call__(self, verdicts: Sequence[bool]) -> bool:
        """Fold several members' verdicts on ONE handoff into one verdict."""
```

`Strength` is `str, Enum` rather than `StrEnum`, which is 3.11 and the target is
3.10 — the constraint `CLAUDE.md` records and the development machine cannot
surface.

**`Reducer` is a protocol because the reducer is injected, not chosen here.**
§6.2 says what the alpha registers.

### 3.2 The Protocol is a static type. It is not the admission gate

Measured (`scratch/design/validator/t_protocol.py`, `t_protocol2.py`, 3.13.13):

| | result |
|---|---|
| `issubclass(AnyClass, Validator)` | **`TypeError: Protocols with non-method members don't support issubclass()`** |
| `isinstance(obj, Validator)` with `brief = 7`, `strength = "STRONG"` | **`True`** |
| `isinstance(obj, Validator)` with all four set to `None` | **`True`** |
| `isinstance(plain_function, Validator)` after bolting attributes on | `True` |

Three consequences, and together they decide the design:

1. **`issubclass` is unavailable at all.** A registration decorator receives a
   class or a function; the class form raises rather than answering.
2. **`isinstance` is presence-only.** `strength = None` passes, so criterion 1 —
   *"rejected at load. There are no defaults"* — is **not** satisfied by the
   Protocol.
3. **`inputs = "trace"` is the specific trap.** A bare string is iterable, so it
   passes a presence check and then iterates as `['t','r','a','c','e']`: one
   declared input kind silently becomes five nonexistent ones, and §10.3's
   resolution check then fails five times with a useless message.

So the gate is a pydantic model over the **spec record**:

```python
class ValidatorSpec(BaseModel, extra="forbid"):
    name: str
    brief: str                        # no default. Criterion 1
    inputs: tuple[str, ...]
    dimension: Dimension              # no default
    strength: Strength                # no default
    body: Body                        # readme.md, entry.sh, materials. §3.8
    logic_source: LogicSource         # external_static | external_dynamic | agent_written
    tags: Tags                        # §3.5
    version: str | None = None        # maintenance metadata. §3.3
    members: tuple[str, ...] = ()     # non-empty iff this is a composite. §6
    reduce: str | None = None
    args: Mapping[str, Any] = {}      # §10.6
```

Measured on the same probe: pydantic rejects the wrong type, the missing field,
and — with `extra="forbid"` — the unexpected key, each naming the field in
`loc`. It also coerces `list → tuple`, which is wanted, since jsonnet renders a
JSON array.

This is [`../../docs/design.md`](../../docs/design.md) §4's finding arriving
from the other side. There, JSON Schema runs *before* pydantic because generated
models silently drop keywords. Here, pydantic runs *after* the schema because no
schema keyword reaches an in-process Python object. **Both passes exist and
neither replaces the other.**

### 3.3 Two fields with no `Validator` counterpart, and one constraint with no field

Three of `ValidatorSpec`'s fields do not appear on the `Validator` protocol, and
the asymmetry is deliberate.

**`logic_source`** (spec §5.2) classifies where the checking logic came from:
`external_static`, `external_dynamic`, or `agent_written`. It is a *tag on the
spec*, not a property of the body, and **nothing verifies it** — spec §5.2's
"there is no third row" is a rule about what may be registered, and spec open
question 3 already records that the graph shape is what makes `external_dynamic`
trustworthy while the registry cannot see the graph. This design does not close
that; it carries the field so the claim is at least written down and reviewable.

**`args`** is §10.6's parameterisation, checked against the implementation's
signature at registration.

Spec §3.5's two structural constraints have **no field at all**, and criterion 2
rejects a spec that violates either:

| Constraint | How it is rejected |
|---|---|
| A validator is single-node — no subtasks | A validator spec is not a task spec: it has no `subtasks`, and `extra="forbid"` makes naming one a load error. The constraint is enforced by the schema having no place to express its violation |
| A validator's own input validation is empty | Same shape. Nothing on `ValidatorSpec` declares a validation phase |

That is the strongest form the constraint can take — not a check that runs, but
a shape in which the violation is unrepresentable. §12.1 asserts both, because
"the field does not exist" is exactly the kind of guarantee a later field
addition silently removes.

### 3.8 `body` — a validator's implementation is a task's body

Spec §6.1 rev. 8: a validator's checking logic is a `readme.md`, plus an
`entry.sh` when the check is programmatic, plus its own `materials`. Exactly
`closure` spec §2.6's shape, and the same mechanism serves both.

```python
class Body(BaseModel, extra="forbid"):
    readme: str                       # a path in the package. ALWAYS present
    entry: str | None = None          # a path. Present iff the check is programmatic
    materials: tuple[str, ...] = ()   # paths this check needs for itself
```

Paths into the validator's own folder (spec §9.1), resolved against the package
and checked for existence at load. A dangling one is a load error naming the
path, which is the same rule §10.4 already applies to a binding symlink.

**Two kinds of validator, one shape.**

| | Runs | Result comes from |
|---|---|---|
| `entry` present | The script, in the rebuilt environment §8.4 describes | Its **exit status and a verdict file** it writes into the zone |
| `entry` absent | An agent, given `readme.md` as its instruction | The same verdict file, written by the agent |

The verdict file is what makes the two substitutable at the seam: the phase
runner reads one thing either way, and `Validator.__call__` (§3.1) is what it
looks like once read. A code-shaped check gets the shipped pytest harness and
puts its assembly and run commands in `entry.sh`.

**Rev. 2 had a registered Python callable here, and it is withdrawn.** The
reason is spec §6.1's and it is not about taste: a callable cannot express *a
validator an agent is responsible for* without a wrapper whose whole job is to
run an agent — so the callable becomes a layer that exists to be worked around,
and the two kinds of validator stop being one thing. §14 D6 records what that
costs, because it costs something real.

**`args` still lives on the spec** (§10.6), and reaches the body as a JSON file
in the validation's zone rather than as parameters. That is the one shape that
works for a script and for an agent without either learning about the other:
`entry.sh` reads it, and a `readme.md` may refer to it.

### 3.4 `version` is not read at runtime, and this design does not start

Spec §9.3: *"**Maintenance metadata only**: nothing at runtime reads it"*, and
[`../../closure/docs/spec.md`](../../closure/docs/spec.md) §1.2 says the same of
every spec `version`: *"Nothing at runtime reads it, nothing pins to it."*

Recorded here because §7's skip decision is exactly where a designer reaches for
it — "has the validator changed since that verdict?" — and the answer is that a
spec `version` is not the runtime identity of anything. §7 uses the verdict
record instead, which is a runtime fact.

### 3.5 Tags are a dictionary, and one of them is queried

Spec §9.2's five keys. `dimension` and `strength` are also fields, because
criterion 1 requires them present and criterion 15 requires querying by
dimension; the tag dictionary carries the rest.

```python
class Tags(BaseModel, extra="allow"):
    logic_source: LogicSource
    cost: Cost                        # seconds | minutes | gpu_hours. §5.3 orders by it
    domain: tuple[str, ...] = ()      # free-form: trace, kernel, deploy, eval
```

`extra="allow"` here and `extra="forbid"` on `ValidatorSpec`, deliberately: the
spec calls `domain` free-form and a tag dictionary is where a site adds its own
key, while a stray *top-level* field is a typo. **`cost` is the one tag the
system reads** (§5.3); the others are for a human deciding which validator they
want, which is what spec §4's `purpose` argument is about.

### 3.6 The result type, and why it is per-handoff

`dict[HandoffId, bool]`, spec §4.2, boolean in v1. A score type is reserved in
the schema and unbuilt, for spec §4.2's reason: a threshold is meaningless until
run-to-run variance has been measured.

**Keyed by uuid, not by kind.** Spec §4.1 and handoff §5.1 both say so, and the
reason is concrete: a task with two inputs of the same kind would be ambiguous
under a kind key. The pointer that addresses *into* one handoff's content is
[`../../handoff/docs/design.md`](../../handoff/docs/design.md) §8.4's — RFC 6901,
three-way failure — and this module consumes it rather than restating it.

### 3.7 What the shape of `Validator` refuses to say

Spec §7 draws a line this design must not cross: **how a check's own code is
structured is the external test system's**, not ours.

So `Validator` has no base class, no assertion helpers, no fixture notion, and
no timing hook.

**Rev. 3 makes this stronger rather than weaker**, and by accident. Rev. 2 kept
the checking logic directly testable by convention — a registered validator
*delegates* to a plain function, which pandera documents doing because their
decorator otherwise makes the underlying function unreachable from a test. A body
(§3.8) needs no such convention: `entry.sh` is runnable from a shell, and
`readme.md` is readable by a person. **There is nothing wrapping it that could
make it unreachable.**

Criterion 12 is what pins it — the shared logic must be *"a plain function that
is directly testable"* — and §12.1 now tests it by running the body with no
registry, no phase runner, and no system at all.

---

## 4. Where a validator runs — one dispatch, three phases

Spec §3. `TaskRunner` runs all three phases for the one task the scheduler
dispatched; the scheduler sees phase 2 and nothing else.

This design adds no scheduling. What it adds is the two phase objects and the
rule for what runs inside them.

```
scheduler dispatches ONE task, holding ONE lease
   │
   ▼
┌──────────────────┐   ┌──────────────┐   ┌───────────────────┐
│ 1 INPUT          │──►│ 2 MAIN       │──►│ 3 OUTPUT          │
│   VALIDATION     │   │   the agent, │   │   VALIDATION      │
│   over inputs    │   │   or a       │   │   over outputs    │
│                  │   │   subgraph   │   │                   │
└──────────────────┘   └──────────────┘   └───────────────────┘
   this module            not this            this module
```

---

## 5. The phases

### 5.1 What `task_graph` must supply, and does not yet

`task_graph` spec §3.2.2 rev. 12 specifies `INPUT_VALIDATING` and
`OUTPUT_VALIDATING`, with their transitions. Measured: `TaskStatus`
(`task_graph/models.py:44`) has **eight** members and contains neither.

**That is `task_graph`'s change, not this module's** — module 4 raises the design
from rev. 10 to rev. 11 and its criteria 36–54 include them. This design is
written against the specified state and §15 O5 records the dependency.

### 5.2 The phase runner

The first place the ordering is the design:

```python
def run_phase(self, kind: PhaseKind, task: Task, registry: Registry) -> PhaseOutcome:
    selected = self._select(kind, task, registry)        # bound validators, cheap-first
    if not selected:
        return PhaseOutcome.empty(kind)                  # NOT a pass. §11.2

    ran, reused, skipped = [], [], []
    for v in selected:
        prior = history.top(task, v, registry)           # §7.1
        if history.may_skip(prior, self.strict_level):   # §7.2
            reused.append(prior); skipped.append(SkipRecord(v.name, prior)); continue
        env = self._build_environment(kind, task, v)     # §8.2 — a rebuild, always
        ran.append(self._invoke(v, task, env, registry))

    return PhaseOutcome.fold(kind, ran=ran, reused=reused, skipped=skipped)
```

Four properties, each with a reason:

**An empty selection is not a pass.** §11.2.

**The fold is over `ran ∪ reused`, never over `ran` alone.** §7.3 gives the
argument; it is the only reading under which spec §5.5's "one fails, all fail"
survives a skip.

**A skipped validator produces a `SkipRecord`, not silence.** Criterion 7. It is
a value on the outcome object, not a log line, for the reason
[`../../handoff/docs/design.md`](../../handoff/docs/design.md) §8.5 gives:
an assertion over a log capture is a test of the logging configuration.

**The environment is built per validator, inside the loop.** Spec §8.2 —
rebuilt, never inherited. §8.2 here measures what that costs.

### 5.3 Selection, and cost-aware ordering

`_select` returns the bound validators for the phase, ordered by the `cost` tag
(spec §9.2, §2 principle 2 — cheap gates before expensive ones).

**Ordering by a declared cost tag has no prior art in anything surveyed.** Every
system that consumes a cost signal does one of three other things: admission
control (Bazel's `size` feeds `ResourceSet`; `ResourceSet` never appears in the
ordering path), measured bin-packing (pytest-split), or a human-declared
dependency graph (CI `needs`). Nothing orders by a coarse self-reported
order-of-magnitude tag.

So this is a place where we are ahead of the prior art rather than behind it,
and the design owes the failure mode rather than a citation. **The declared tag
can be wrong, and nothing here detects it.** The two nearest precedents bracket
the risk without solving it: Bazel warns when a test's runtime leaves its
declared envelope — advisory, off by default, and with an irreducible
false-positive rate against variance (#5015: a test legitimately varying
100–500 s trips the warning whichever size is chosen). pytest-split **silently
substitutes the population mean** for an unknown test and **silently discards
orphaned durations**, so a partially-stale index degrades with no warning at all.

The design's answer is the cheap one and it is recorded rather than built:
`_invoke` records actual duration into §10.5's historical index, so a future
change can report disagreement with the declared tag. Ordering is by the tag
today; §15 O6 carries the rest.

### 5.4 A failed phase is an ordinary task failure

Spec §3.4. Nothing in this module cascades, invalidates, or notifies. The phase
returns a `PhaseOutcome`; `TaskRunner` turns a failing one into a task failure;
the scheduler's response is to stop scheduling. Reacting to it belongs to the
monitor ([`../../docs/ROADMAP.md`](../../docs/ROADMAP.md) §2).

Recorded because the temptation at this exact seam is to add a retry or an
escalation, and either would give the runner an opinion about *what*, which is
the boundary the whole system rests on (main spec §5.1).

### 5.5 What the composition root registers, and what calls it

`phase.py`'s `run_phase` is a method, and rev. 1 never said what it is a method
*on* or how `agent`'s `Runner` gets hold of one. `agent` design §7.1 says the
runner *"resolves the validator's phase runner from the registry, by name, at use
time"* — and no document registered anything. Found in the stage-three
consistency pass, along with the same gap on `env_mgr`'s side.

```python
r.register("phase_runner", PhaseRunner(strict_level))
```

```python
class PhaseRunner:
    """The two validation phases. Registered once; called twice per dispatch."""
    def __init__(self, strict_level: StrictLevel) -> None: ...
    def run_phase(self, kind: PhaseKind, task: Task,
                  registry: Registry) -> PhaseOutcome: ...       # §5.2
```

**`strict_level` is bound at construction and `registry` is passed per call**, and
the asymmetry is the design. The strict level is a run-wide policy — it arrives
from `--validation-strict-level` and cannot change mid-run without changing what
criterion 20 means — while the registry is how the phase reaches `handoff_mgr`
and `agent_mgr` at the moment it needs them (§2.1). Binding the registry instead
would make this object hold a collaborator handle across dispatches, which is the
thing `task_graph` design §2 keeps out of every manager.

**One method, and it stays one.** A second — `configure`, `set_level` — would put
the knob back inside the object the knob is not allowed to reach (§7.3).

---

## 6. The composite

### 6.1 The reducer is injected

Spec §6.2 writes the combinator as
`composite(validators=[...], reduce="all" | "any" | "at_least(k)")`. **The
reducer is a `Reducer` (§3.1) resolved by name from a table.** The three names
in the spec are the anticipated table, not a closed set the design must
implement.

```python
class Composite:
    """One level deep. Members are leaf validators; the reducer folds their
       verdicts per handoff."""
    members: tuple[Validator, ...]
    reduce: Reducer

    def __call__(self, handoffs) -> dict[HandoffId, bool]: ...
```

The reduction is **per handoff**: for each handoff, fold the verdicts its
members gave *it*. That keeps the result type `dict[HandoffId, bool]`, so a
composite is type-substitutable for a leaf and criterion 4 holds of both.

**Type-substitutability is what keeps the reducer out of the phase.** Spec §5.5
is explicit that the reducers are *"scoped **within** a composite validator and
say nothing about a phase: a composite reduces its members to one verdict, and
that verdict then meets this rule like any other."* Because a composite returns
the same type a leaf does, `run_phase` (§5.2) cannot tell them apart and has no
place to apply a reducer even if someone wanted to — the phase's rule stays the
one rule spec §5.5 gives it. A composite whose reducer is `any` is therefore a
way to express "either of these will do" *inside* one check, and never a way to
soften the phase.

It is also Inspect AI's shape, arrived at independently — its reducers walk the
keys of the first dict and reduce each key across scorers. Two differences worth
recording, because they say what we cannot borrow:

- Inspect's keys are **epochs of one sample**, so it can and does reject
  mismatched keys outright: *"Cannot reduce dictionary scores with mismatched
  keys… Every epoch must score the same keys."* **Our members legitimately
  declare different input kinds** (spec §4.1), so that rejection would forbid a
  composite the spec permits. §6.4 is our answer instead.
- Inspect's registered reducers are `collect, at_least, pass_at, pass_k, max,
  mean, median, mode`. `all` and `any` are **not** among them —
  `multi_scorer([...], "all")` raises `LookupError`. Spec §6.2's names are ours.

### 6.2 The alpha registers one reducer

```python
REDUCERS: dict[str, Reducer] = {"all": AllReducer()}
```

`and` semantics, and nothing else, for the alpha. A `reduce` naming anything
absent is a load-time error enumerating the table, per
[`../../docs/design.md`](../../docs/design.md) §5.2's rule that a failed lookup
names its candidates.

`any` and `at_least(k)` are the anticipated additions and the table is what
makes them additions rather than changes. One note for whoever adds
`at_least(k)`: Inspect validates `k` against the number of things being reduced
and raises `PrerequisiteError` when `k` exceeds it, so a threshold that can never
be met is rejected up front rather than failing silently forever. Under a ragged
membership — where members declare different inputs — the count differs *per
handoff*, so the check is per handoff too. Not built; `all` has no `k`.

### 6.3 Nesting is rejected by the schema, not at runtime

Criterion 13. `validator.schema.json` describes a composite's `members` as
naming **leaf** validators; the composite shape is absent from that list. A
nested composite therefore fails schema validation in step 3 of the loader
([`../../docs/design.md`](../../docs/design.md) §3.6) and never reaches
admission.

This is OpenAI's mechanism, verified in its published schema: `GraderMulti`
appears in the three top-level grader unions and **is absent from
`GraderMulti.properties.graders`**. Not a runtime check and not a documentation
note — the nested union simply omits it. Their docs say it in prose too — *"You
cannot nest one multigrader inside another"* — but the schema is what enforces
it.

A caution carried from reading that schema: OpenAI maintains the two unions by
hand and they disagree in both directions — `graders` is declared as a bare
`oneOf` with **no `type: array`** while the example on the same object passes a
list, and `GraderLabelModel` is in the nested union but not the top-level one.
Our schema has one list to maintain and §12 asserts the rejection, so the same
drift is detectable.

**A registry-level guard is kept as well**, because a composite may be
constructed in a test without passing through the loader:
`Composite.__init__` raises `NestedComposite` if any member is one. Two layers,
because the schema is the enforcement and the guard is the safety net for the
path that bypasses it.

Spec §6.2's citation for the prohibition needs correcting, and §14 D1 records
it: **Inspect AI does not reject nesting** — `_multi.py` (0.3.260) is 69 lines
with no depth check, and a 3-deep composite constructs and executes. The rule
stands on OpenAI and Gatekeeper.

Gatekeeper's four reasons for refusing cross-template reuse are worth keeping,
because two of them are ours. Verbatim, gatekeeper#1204: *"Dependency version
conflicts · **Eventual consistency** — what if the template is ingested before
the shared library is available? · **Non-hermeticity** — you now need to install
two objects to get a given constraint template to work · Non-isolation"*.
Eventual consistency is [`../../docs/design.md`](../../docs/design.md) §6.1's own
argument for a closure pass; non-hermeticity is why a composite naming a member
in another package invisibly is a bad idea. Non-isolation does *not* transfer —
Rego imports share a namespace, while our members are registry entries whose
duplicate-rejection §5.2 already guarantees.

### 6.4 Two silent passes the composite must refuse

Both measured (`scratch/design/validator/t_reduce.py`), both arriving from the
standard library rather than from a bug.

**`all([])` is `True`.** A handoff that *no member validator declares* would be
folded from an empty list and pass, having been checked by nothing. That is a
vacuous truth reported as evidence — precisely the silent pass spec §1 exists to
prevent.

```python
covered = set().union(*(v.inputs for v in self.members))
missing = {h for h in handoffs if kind_of(h) not in covered}
if missing:
    raise ValidatorInvalid(f"composite covers no validator for {sorted(missing)}")
```

Checked at **admission**, not per call: membership and declared inputs are both
static, so the fault is an author's and belongs in the load report. The runtime
guard remains as an assertion because a composite can be built directly.

**A member that returns no entry for a handoff it declared.** `dict.get` would
yield `None`, and `None` folded as falsy is indistinguishable from a genuine
`False`. DeepEval demonstrates the cost: its unreached DAG node leaves `score`
as `None`, `is_successful` catches the resulting `TypeError` and sets
`success = False`, so **an unreached terminal and a real zero report
identically**, with no signal that the graph terminated early. Our `__call__`
raises `ValidatorInvalid` naming the validator and the missing handoff.

---

## 7. Re-validation is decided from the verdict record

### 7.1 The record, and the top of it

handoff spec §5.2: *"**Every versioned handoff carries a validation history**:
for each validator, the result of each execution, plus enough context to
interpret it — which task and which versioned agent ran the validation, and the
environment"*. Criterion 8 of that spec asserts exactly those fields.

So the history is per validator, and each execution appends. **The decision
reads the top; everything beneath it is retained as the record.**

```python
def top(task, validator, registry) -> VerdictRecord | None:
    """The most recent execution of `validator` against this handoff version.
       None if it has never run against it."""
```

This is the whole mechanism, and it is worth saying what it replaces. The
natural instinct at this seam is a content-addressed verdict cache keyed on the
handoff digest plus something identifying the checker. Five such key schemes were
measured against six things that can change the answer
(`scratch/design/validator/t_cachekey.py`), and **"the validator's implementation
changed" is a stale hit under every one of them**, because implementation source
appears in no spec file. The record answers the question the cache was being
built to answer — *did this exact validator run against this exact version, and
what did it say* — without needing a key at all, because the record **is** the
answer rather than an index into one.

**It does not, and does not claim to, detect that the validator's code changed
since.** That is stated in §7.4 rather than left to be discovered.

### 7.2 What may be skipped

Spec §3.3: by config, or because this exact version was already validated. Both
are governed by `--validation-strict-level`, and criterion 20 says the level
*"changes which phases run, and never which verdicts bind"*.

```python
def may_skip(prior: VerdictRecord | None, level: StrictLevel) -> bool:
    """True iff a prior verdict exists for this exact version and the level
       permits reusing it. Never consults the verdict's value."""
```

**The last clause is load-bearing.** A `may_skip` that reused a pass and re-ran a
failure would be the level deciding a verdict by the back door. Bazel does make
that asymmetry deliberately — `--cache_test_results=auto` reuses a cached pass
and re-runs a cached failure — and it is a coherent policy, but it is a *policy*
and ours is not it: spec §5.5 has one rule, and a recorded failure must bind
exactly as a fresh one does. §7.3 is how.

### 7.3 The fold is over produced ∪ reused

`PhaseOutcome.fold(ran, reused, skipped)` folds **both** lists. A skipped
validator contributes its prior verdict — including a prior **failure**, which
therefore still fails the phase.

The shape is deliberate and it is borrowed. ESLint's `--quiet` once erased an
*error*-severity finding and flipped the exit code to 0 (#14202); the fix (PR
#14242) was not an enumeration of interactions but a **variable split** —
introduce `resultsToPrint` for the filtered set, keep `countErrors(results)` on
the unfiltered one, with the comment *"Errors and warnings from the original
unfiltered results should determine the exit code."*

That the split is the lesson and enumeration is not is shown by what happened
next: the optimisation was reintroduced one layer down by an RFC that
**explicitly claimed verdict-neutrality** and handled the one exception it knew
about — and #19625 found the missed case **22 months later**, closed as
working-as-intended. ESLint enumerated and was still wrong.

So here the knob decides membership of the **run set** only, and the verdict
folds over a set the knob cannot reach. That makes criterion 20 structural
rather than a property to be maintained by care.

### 7.4 What this does not detect

**A validator whose implementation changed after recording a verdict.** The
record names the validator, the task, the agent and the environment; it does not
digest the implementation, and spec §9.3 forbids reading `version` at runtime.
So reusing a verdict is a statement that *this validator, as configured then,
said this* — not that a re-run today would agree.

Nix names the same distinction and says the second half is unavailable: its
build trace is *"a memoization table for builds"*, and *"**In general, there is
no way to audit a build trace entry except for by performing the build again
from scratch.** … the decision of whether to trust a counterparty's build trace
is a fundamentally subjective policy choice."*

That is the honest description of `--validation-strict-level`: **a trust policy
over recorded verdicts, not a correctness mechanism.** Re-running is what
converts a recorded verdict into a current one, and the strict level is the knob
that decides how much of that you buy. Stating it this way is better than any
invalidation scheme the design could invent, and §15 O2 carries the question of
whether a stronger identity is wanted later.

Where the design does have an advantage worth claiming deliberately: **our
dependency edges are declared, not inferred.** dbt solves the analogous problem
by digesting `macro_sql` transitively, and its whole #6455 / #5202 / #8526 /
#14403 bug family is one bug — an edge that its Jinja parser did not see. A
validator's `inputs` and a composite's `members` are fields, so the equivalent
mistake is not available.

---

## 8. Isolation — the producer cannot read the standard

### 8.1 The hook seam

Criterion 10. Spec §8.1: by hook, not by convention.

Measured against `claude-agent-sdk` **0.2.144**
(`scratch/design/probes-validator/p1_sdk_hook_dispatch.py`): a single
synchronous `PreToolUse` callback **logs every attempt before deciding, and then
denies** — so criterion 10's "spy" and criterion 10's "the hook denies" are the
same object, not two. The async form cannot block (*"Async outputs can't block,
modify, or inject context into the operation since the agent has already moved
on"*), so logging-only is not an available optimisation. Composition is safe:
any hook returning `deny` wins.

**The SDK is behind a seam, because the repository has not committed to it.**
`pip show claude-agent-sdk` → not found. Rev. 1 recorded *"the wheel is 103 MB and
pulls `mcp` and `sniffio`"*; **both figures were wrong**, and `agent` design §8.1
measured the real ones: 99 MB wheel, **376 MB installed** (328 MB of it a single
bundled executable), **26 extra packages**, and **~1.3 s to import**. That last
number is what decided it — `agent` §8.1 makes the SDK an **extra**, imported
inside `_probe` rather than at module scope, so `env-mgr check` does not pay 1.3 s
for a package it will never use. This module's seam is unaffected; the numbers are
corrected because §13 reasons from them.

```python
class BoundaryHook(Protocol):
    def on_tool_use(self, event: ToolUseEvent) -> Decision: ...
    def log(self) -> Sequence[ToolUseEvent]: ...
```

One implementation adapts the SDK; the test double is the other. Which package
backs it, and whether it is a hard dependency, is a coding-stage decision this
seam exists to defer.

Two registration details measured, recorded because each costs an afternoon:
`hook_callbacks` is populated only inside `Query.initialize()`
(`_internal/query.py:242-259`), not `__init__` — a `Query` built with hooks but
never initialised dispatches nothing. And `HookMatcher` matches the **tool
name**; there is no agent-scoped matcher, so a hook is session-wide and filters
on identity itself.

### 8.2 Attribution, and why each phase is its own session

Criterion 10 says *"no read **originating in a producer frame**"*. Measured: the
union of every field over every `HookInput` type in 0.2.144 contains **nothing**
denoting a stack, caller, frame, origin or parent. The only identity fields are
`agent_id` and `agent_type`, both **optional**, and the SDK's own docstring says
why:

> *"agent_id: … **Present only when the hook fires from inside a Task-spawned
> sub-agent; absent on the main thread.**"*

**So a phase running on the main thread is unattributable, and criterion 10 is
not testable for it.** This design therefore requires that **each phase be
separately attributable** — which on this SDK means a subagent or a distinct
session, because those are the only shapes that carry `agent_id`.

**That is a requirement, not a mechanism, and the distinction was worth making.**
Rev. 1 wrote it as *"runs each phase as its own subagent or session"*, and
`agent` design O6 records the same question as still open, having narrowed it:
one client with several `session_id`s is **ruled out**, because `interrupt()`
takes no `session_id` and acts on the whole connection. The stage-three
consistency pass found the two documents giving different answers.

They are reconciled by splitting the question, and neither document was wrong
about its half:

| | Owner |
|---|---|
| **That** a phase must be attributable — criterion 10 is untestable otherwise | **this design**, §8.2 |
| **How** the backend delivers it — subagent, `fork_session`, `resume`, or a second client | **`agent` design O6**, still open |

`_build_environment` therefore asserts that the phase it is about to run has an
`agent_id`, and fails loudly if it does not, rather than assuming a mechanism
`agent` has not chosen. That is the same discipline §8.3 takes from SWE-bench:
**absence is a property to assert, not to arrange.**

`SubagentStart` and `SubagentStop` carry `agent_id` as **required**, which makes
them structurally the `<agent>` / `</agent>` markers
`tests/task_graph/test_authority.py` already uses. The technique transfers whole,
including its justification, which is worth quoting because it is the reason the
test is honest: *"'Who called this' is recorded rather than inferred from the
call stack, which would be unimplementable in any honest way."*

Note for a Python implementation: `SessionStart` / `SessionEnd` are **absent from
the Python `HookEvent` literal** and are TypeScript-only as SDK callbacks. In
Python they are shell hooks in `settings.json` — a file `env_mgr` §4.4 requires
to live outside the agent's writable set (CVE-2026-48124).

### 8.3 The hook is the attributable layer; the sandbox is the enforcing one

Measured, and it is the whole reason the design has two layers:
`Bash{'command': 'python3 reader.py'}` returned **ALLOW**. There is no path in
the payload, so there is nothing to match. `env_mgr` spec §4.1 found this; it
reproduces against the current SDK. Anthropic documents the same for declarative
rules — deny rules *"don't apply to arbitrary subprocesses that read or write
files indirectly"*.

| Layer | Answers | Does not |
|---|---|---|
| The hook (§8.1) | *who attempted what*, and denies direct tool calls | see an indirect read |
| `env_mgr`'s allow-list (§4.5) | makes the standard unreachable at all | say who tried |

Criterion 10 asserts both halves, and the design must not let the first read as
though it were the second. This is the same two-layer argument
[`../../handoff/docs/design.md`](../../handoff/docs/design.md) §7.3 makes for
path containment, and it transfers intact.

**`env_mgr` has no sandbox implementation today** — a grep for
landlock / bwrap / unshare / seccomp hits only its spec. So the enforcing layer
is specified and unbuilt, and criterion 10's test asserts the hook half plus the
declaration half (§9), not the kernel half.

**Absence is a property to assert, not merely to arrange.** SWE-bench's answer
key was physically absent from the container for two years and still leaked:
`git remote remove origin` leaves the fix commit reachable through
`git cat-file --batch-all-objects` (reproduced,
`probes-validator/e10_answerkey/git_object_leak.sh`), and issue #465 names real
cheating trajectories. Their fix is the design lesson — `git_clone_timesafe` now
ends with `COMMIT_COUNT=$(git log --oneline --all --after=…); [ "$COMMIT_COUNT"
-eq 0 ] || exit 1`. So `_build_environment` **asserts** the checking standard is
unreachable from the constructed zone and fails loudly, rather than arranging
for it and trusting the arrangement.

### 8.4 A fresh environment, and what that costs

Spec §8.2: rebuilt from configuration, never inherited. `_build_environment`
takes a resolved configuration and returns a new environment; the chain for
which configuration is spec §8.2's four-row table and this design does not
restate it.

**There is no cost argument against rebuilding.** Measured on this machine
(`probes-validator/e21_isolation_cost.sh`, `e21_venv_cost.py`):

| | |
|---|---|
| `mkdir` a fresh zone | 0.03 ms |
| `unshare --user --mount` + bind + remount-ro | **13.8 ms** |
| `python3 -c pass` — the subprocess floor | **61–66 ms** |
| `docker run --rm … /bin/true` | 786 ms |

**A full private, read-only, bind-mounted namespace costs about 4.4× *less* than
starting the Python interpreter that runs inside it.** All variants ran
unprivileged, with no `bwrap` present. The roadmap item contemplating a
relaxation of §8.2 ([`../../docs/ROADMAP.md`](../../docs/ROADMAP.md) §6) should
be told this number.

**Criterion 21 cannot be tested by a directory check**, and the design's test
must not pretend otherwise. Measured: a fresh zone directory closes **one** of
the channels a producer leaves state in. `/tmp`, `os.environ`, an inherited
`cwd`, `sys.path`, `$HOME`, and same-path reuse all still carry it. Inherited
fds are clean, CPython defaulting them non-inheritable. `env_mgr` §4.5.1 already
excludes `$HOME` from the granted set — one channel closed by policy; the rest
are unmeasurable until `env_mgr` implements the sandbox its own spec §4.2
specifies — its design stage, not this one.

Nix specifies its own version of this guarantee in the right shape, and this
design copies the shape: *"what matters for determinism is what the build
process can observe… Nix can achieve this through many different sandboxing
strategies… but the process shouldn't be able to tell them apart. We therefore
specify building from the process's perspective, not Nix's."* Stating criterion
21 as *what the validation can observe* rather than as a mechanism list is what
lets `env_mgr` change mechanisms without invalidating it.

Two failure modes to design away from, both first-hand from the survey:

**Freshness must come from allocation, never from cleanup.** pytest's
`tmp_path` is a *new numbered directory*; its cleanup is explicitly best-effort
(`rmtree(..., ignore_errors=True)`). A guarantee that depends on a teardown
succeeding is not a guarantee.

**A staleness check with a hidden off-switch is worse than none.** nox's, in
full (`nox/virtualenv.py:133`):

```python
def _check_reused_environment_interpreter(self) -> bool:
    if not os.environ.get("NOX_ENABLE_STALENESS_CHECK", ""):
        return True
```

Disabled for a Python 2.7 bug and never re-enabled. That is the
silently-applied-default-upstream-of-a-check lesson
[`../../docs/design.md`](../../docs/design.md) already records from OPA,
recurring in a third system.

---

## 9. Criterion 11 — the producer may not legislate its own standard

### 9.1 What the criterion is about

> *A validator whose logic lives in the producing task's permission zone is
> rejected — the check is structural, not declarative.*

The principle is a **separation of legislative and executive power**. A producing
task has, and must have, the power to *execute* code in its own zone. What it may
not have is the power to *write the rule it will be judged by*. Spec §8.1's table
says it directly — the producer cannot "Write the checking logic" — and appendix
A of the kickoff report names the failure it prevents: *the candidate writes the
exam, the answer key, and grades it.*

A validator whose implementation sits where the producing task's permissions
reach hands the producer both powers at once. That is the shape this check
refuses.

### 9.2 It is a comparison of two declarations, at load

Both sides are spec fields available at load:

| Side | Where it comes from |
|---|---|
| the producing task's reach | `Task.permissions`, a field of the task spec, **versioned with the task** (`task_graph` spec §3.2.2, criterion 44) |
| the validator's logic location | `ValidatorSpec.logic`, resolved against its package (spec §9.1) |

```python
def check_separation(validator: ValidatorSpec, producer: TaskSpec) -> None:
    """Raise SeparationViolation if the producing task's declared permissions
       reach the validator's logic. Both sides are declarations, compared at
       load; enforcing them at run time is env_mgr's."""
```

**`env_mgr` builds a sandbox at task start, and that is irrelevant here.** The
sandbox *executes* the declaration; this check reads the declaration. Nothing
about it needs a live zone, which is why criterion 11 belongs in spec §9.3's
load-time list.

It runs in the **closure pass**
([`../../docs/design.md`](../../docs/design.md) §6), not at admission, for that
section's reason: it needs the task registry and the validator registry both
fully loaded, and the far side may not be loaded yet at admission time. The
layering gate applies — a task whose own spec failed is skipped.

### 9.3 "Structural" means the layout, not anyone's assertion

The check consults no field in which a validator or a task asserts its own
independence. It compares where things *are*. That is the whole content of
"structural, not declarative", and it is what makes the check unfoolable by an
author who is simply wrong about their own package.

Comparing paths means resolving them, and the failure modes are the ones
[`../../handoff/docs/design.md`](../../handoff/docs/design.md) §7.2 measured. Two
are worth restating because this module gets them differently:

**Resolution is mandatory here, not prudent**, because spec §9.1 *sanctions*
cross-package symlinks: *"A symlink may point outside its own package, and that
is how two packages share a handoff kind."* The benign case (a link pointing
*away* from the zone) is handled correctly by every check, lexical or resolved.
The dangerous case is its inverse — a link in a neutral package pointing **into**
the producer's zone — which is lexically innocent and executes the producer's
bytes. Measured over five checks × six layouts, wrong answers were
3 / 2 / 2 / 1 / **0**; only `realpath` plus a trailing separator gets all six
right (`probes-validator/p2_c11_symlink_vs_zone.py`).

**The fail-closed direction is inverted relative to the sibling modules**, and
this is the trap. In `handoff` §7 and `env_mgr` §4.3, *contained* means allow, so
an unresolvable path is denied. Here *contained* means **reject the validator**,
so an unresolvable path must be treated as **inside**. Importing
`check_contained` and negating at the call site would negate the fail-closed
behaviour too, and **a dangling validator symlink would be accepted**. §12.2
tests the unresolvable case explicitly for that reason.

Go's `internal` check is the sharpest prior art and it inverts for us. It checks
lexically, then resolves symlinks **only to widen** access — a link can never
turn an allowed import into a denied one (`cmd/go/internal/load/pkg.go:1548-1575`,
with the comment *"Look for symlinks before reporting error"*). Ours is a
rejection, so the corresponding asymmetry is: resolution may only ever move a
verdict toward *accept*. **A deliberate inversion of Go's risk posture, not a
copy of it.**

Its error-message discipline is worth copying outright: `ImportErrorf` **panics**
if the offending path is absent from the message.

### 9.4 The honest ceiling

Stated here for the reason `env_mgr` spec §4.6 states its own — because it
decides what the system may claim.

> **This check refuses one shape of a broader problem.** It rejects a validator
> whose implementation file sits where the producing task's declared permissions
> reach. It does not, and cannot, establish that the producer had no hand in the
> checking logic. Logic that lives elsewhere but was authored by the producing
> agent, influence through a shared upstream, a standard weakened before the
> producer ran — none of these have a path to compare.

Spec §5.2's taxonomy is what actually carries that weight: logic is
`external_static`, `external_dynamic`, or `agent_written`, and *"logic written by
the agent under test is not a validator, it is the producer's opinion with extra
steps."* Nothing verifies that classification at load — spec open question 3
already says so — and this check does not close it.

What this check does buy is that the most obvious form cannot be committed by
accident, and that is worth having as long as nobody reads it as more.

---

## 10. The registry

### 10.1 One of four

`ValidatorSpecRegistry(SpecRegistry)`, per
[`../../docs/design.md`](../../docs/design.md) §5.1. The base supplies the dict,
the collision policy, and the error shape; this subclass adds its own load-time
checks and its own indexes.

**A duplicate name raises.** Criterion 3, and the base's policy already:
`fsspec`'s shape — error by default, an identical re-registration a no-op. The
alternative is on record as a mistake, verified first-hand: Great Expectations
logs `Overwriting declaration` and proceeds, and Inspect AI's `registry_add` is
a bare dict assignment with no check at all (`_util/registry.py:141`). Spec §6.1
names pandera as the system that raises, and pandera does — with a message that
names the collision: `method with name 'lt_limit' already defined. Check methods
must have a unique method name.`

### 10.2 One registry, and how a body is found

Rev. 2 had **two** registries here — the spec registry, and an implementation
registry a `@validator` decorator wrote at import time, joined by name through
`ValidatorSpec.logic`. **Rev. 3 has one.**

A body is a set of paths in the validator's own folder (§3.8), so finding it is
resolving a path against the package the spec was loaded from. There is no second
table, no decorator, no import-time side effect, and no join to keep in step.

That also removes a failure mode this design had recorded and then inherited:
Great Expectations' registry is populated by a metaclass, so a class is present
only if something imported it, and their own docstring concedes *"we need to hope
that core Expectations are imported somewhere in our import graph — if not, our
registry will be empty"*. Loading from disk was already the rule for specs
([`../../docs/design.md`](../../docs/design.md) §5.5); now it is the rule for
implementations too, because there is only one kind of thing left to load.

**What is lost, and it was worth having.** A dotted path to a Python callable is
checkable by import; a path to a script is checkable only for existence. A body
that exists and does the wrong thing is not detectable at load in either shape,
but a body whose *entry point is missing a function it needs* was, and is not any
more. §14 D6.

### 10.3 The five load-time checks

Spec §9.3's list, each failing with the file path:

| # | Check | Notes |
|---|---|---|
| 1 | schema, then `ValidatorSpec`; the name is unique | §3.2. Two passes, neither replacing the other |
| 1b | **the body resolves** — `readme` always, `entry` when declared, every `material` | §3.8. Existence only; a dangling path is a load error naming it |
| 2 | `brief`, `dimension`, `strength` present | No defaults — pydantic, since an unlabelled validator would default to being trusted |
| 3 | every declared input kind resolves in the handoff registry | closure pass; enumerates candidates on a miss |
| 4 | the binding agrees with the handoff registry's side | closure pass. [`../../handoff/docs/design.md`](../../handoff/docs/design.md) §8.3 owns the message; this side supplies **`inputs`** — rev. 1 of both documents called it `binds_to`, a key `ValidatorSpec` does not have and `extra="forbid"` rejects, so the agreement check read a field that cannot exist |
| 5 | a composite's members resolve and none is a composite | §6.3 |

Checks 3, 4 and 5 need the other registry and so run in the closure pass; 1 and
2 are admission. Criterion 11's separation check (§9) joins 3–5 there.

### 10.4 A folder, and the symlinks in it

Spec §9.1: a validator is a **folder**, in a task package (main spec §4.3) or in
this repository's general-spec directory (main spec §4.5). Its folder *"carries
relative symlinks to the handoff kinds it binds to, so the binding is visible in
a directory listing and not only in the registry."*

Two things follow that the registry has to get right.

**The symlinks are a second, redundant statement of the binding, and redundant
statements disagree.** They are not the registry's source of truth —
`ValidatorSpec.inputs` is — so the loader reads the field and the symlinks are
for a human with `ls`. But a symlink naming a kind the spec does not is a fault,
and §10.3 check 3 catches only the reverse. The design's answer is the cheap
one: the discovery step reports a symlink with no matching `inputs` entry as a
load problem, in the same collected-not-raised style as everything else in
[`../../docs/design.md`](../../docs/design.md) §3.6.

**A dangling symlink is a load error naming the path**, which spec §9.1 says
outright — *"not a puzzle for the loader to solve"*. That is the same
`resolve(strict=True)` fail-closed §9.3 uses, and the reason it must be spelled
out here is that the *direction* differs: for a binding symlink, unresolvable
means **error**; for §9's separation check, unresolvable means **reject the
validator**. Two uses of one primitive, two failure directions, both loud.

Note that these two symlink roles are unrelated: the binding symlinks point at
handoff kinds, while §9.3's concern is where the *logic* resolves to. A
package may legitimately have both.

### 10.5 The two indexes

Criterion 14: *"who uses it now and who has used it"* — one static, one
historical.

```python
class ValidatorSpecRegistry(SpecRegistry):
    def users_of(self, name: str) -> list[str]:
        """Static: which handoff kinds bind this validator. From the specs."""

    def has_ever_run(self, name: str) -> RunRecord | None:
        """Historical: from the verdict records. None means never."""
```

**They are separate objects that disagree**, and every system surveyed keeps
them separate for that reason: dbt's `manifest.json` and `run_results.json` share
only `unique_id` as a join key (measured: the manifest contains zero
`execution_time`, `status` or `timing` fields); Bazel has three graphs and none
is "what executed"; Airflow keeps its serialised DAG and its task instances
apart.

Two failure modes taken from what they do about the disagreement:

**A record whose validator has been deleted is retained, not dropped.** Airflow
tombstones — `REMOVED = "Task vanished from DAG before it ran"` is a *state*, the
reconciliation is bidirectional (a restored task is un-removed), and it never
runs while the DAG run is active. dbt does the opposite and it is worse: a
`unique_id` present in `run_results` and gone from the manifest is dropped by a
silent set intersection (`graph/selector_methods.py:822`) with no error, warning
or count. "Has this ever run" is meaningless if deletion erases the answer.

**A derived static index goes wrong by forgetting a reference kind.** Airflow's
asset orphanage reports live assets dead because materialisation through an
`AssetAlias` is not one of the four tables it joins over (#58058, with a live
comment in the source saying so). Ours has two edge kinds today — a kind naming
a validator, and a composite naming a member — and §15 O7 records that nothing
owns that enumeration.

**"Never run" must be a state, not a set difference.** Stryker is the model:
`Pending` ("generated, but not run yet"), `No coverage`, `Ignored` are
first-class, serialised, and assertable — and it publishes **two denominators
side by side**, one of which deliberately excludes uncovered mutants. The
counter-example is Great Expectations, whose suite success is
`successful == evaluated` where `evaluated = len(results)` — a tautology whose
denominator counts results produced. Measured on GE 1.21.0: **an empty suite
reports `success=True`**, and the only trace that nothing ran is
`success_percent=None`, an incidental artefact of a division guard. That is the
bug criterion 14 exists to forbid.

### 10.6 Parameterised instances, and the check that is no longer available

Spec §6.1's middle row: one implementation, many registry entries. **The args
live in the validator spec — one spec per instance.** Two validators differing
only in a threshold are two `ValidatorSpec` records naming one body folder.

That satisfies criterion 12, matches
[`../../docs/design.md`](../../docs/design.md) §5's one-spec-one-name shape, and
avoids a question the alternatives raise: if the *binding* supplied the args,
`validators_for(kind)` would change type and handoff design §8.3 would have to
rule on two kinds binding one validator with different args.

**They reach the body as a file, not as parameters.**

```
<validation zone>/args.json        written by the phase runner, before the body runs
```

One shape that works for both kinds of validator without either learning about
the other: `entry.sh` reads it, and a `readme.md` may refer to it. Parameters
would have worked for a callable and not for an agent, which is the same reason
§3.8 gives for the callable going away.

#### The signature check is gone, and this is the one place rev. 3 is weaker

Rev. 2 checked `spec.args` against the implementation's signature at
registration:

```python
sig = inspect.signature(resolve(spec.logic))
for key in spec.args:
    if key not in sig.parameters:
        raise ValidatorInvalid(...)
```

**There is no signature to read on a shell script or on a description**, so the
check is not available in this shape. It is worth saying plainly what that
forfeits, because pandera shipped those four lines for a reason and the reason
transfers: pandera#480 — *"Registered checks with no statistics can be called
with any positional arguments. The arguments are ignored"* — whose reproduction
was a check that validated a frame it should have rejected. Their in-source
comment names it better than we could: **"Reject silent argument loss … producing
checks that look configured but ignore their inputs."**

Three things stand in its place, and together they are less than it was:

- **The args are schema-checked at load.** `validator.schema.json` declares what
  `args` may contain, so a *malformed* args block is rejected. A well-formed one
  naming a key the body ignores is not.
- **`args.json` is in the zone and in the record**, so "what was this configured
  with" is answerable after the fact rather than inferred.
- **A body that ignores its args is a body that is wrong**, and the verdict it
  produces is the thing under test — which is a validator-of-validators argument
  and not a load-time one.

**Recorded as a real loss, not talked away.** dbt's cautionary version is worth
keeping in view: `validate_macro_args` defaults to `False`, warns rather than
errors, compares by positional `zip`, and is broken in both directions — #11792
always warns without a cached manifest, #12574 stops warning after a second
parse. **A signature check reading from a cache of a previous parse is worse than
no check.** Ours is now no check, which is at least not that.

## 11. Reporting a verdict

### 11.1 Where a qualified pass lives

Criterion 19: a passing `weak` validator is reported as a low-confidence pass and
is distinguishable from a passing `strong` one.

Measured against the shipped `task_graph`
(`scratch/design/validator/t_verdict_shape.py`), there are three candidate homes
and only one is free:

| | |
|---|---|
| a third `HandoffStatus` | **Rejected.** `check_if_latest_valid` is `status is VALID`, so a `WEAK_VALID` **silently blocks every consumer** unless the scheduler's eligibility test changes too — a scheduler change made for a reporting requirement |
| a field on `HandoffVersion` | **Rejected.** Its fields are `{content, producer_agent_id, producer_task_id, status, timestamp, version}`, and the verdict record is deliberately *not* on the version — it is the sibling file, outside the digest |
| the validation record | **Chosen.** Already beside the artefact and outside the digest ([`../../handoff/docs/design.md`](../../handoff/docs/design.md) §4.1) |

So the persisted record carries the validator's `strength` alongside its result,
and `PhaseOutcome` renders it.

**Two names, one fact, and the ownership is `handoff`'s.** The persisted shape is
`handoff.Verdict` (`handoff` design §6.1) — that module writes it, reads it, and
has to keep it readable across versions. `VerdictRecord` here is this module's
*view* of one, which `top()` returns and `may_skip()` reads. Rev. 1 of both
documents declared a type called `Verdict`; the stage-three pass found it, and
two records of one fact is `engineer_principle.md` §1's failure.

**There are three strengths, not two, and `long_term_strong` needs saying.**
Spec §5.3: it is *"not a weaker `strong`. The rigour is the same; the **timing**
is what differs"* — the quality *"cannot be verified when the handoff is
produced"*. So a `long_term_strong` validator run in an output-validation phase
is answering a question its own label says is not yet answerable.

This design does not treat it specially, and that is a choice with a
consequence:

- **In the fold it behaves as `strong`.** Spec §5.4's table has two columns and
  spec §5.5 has one rule; nothing in the spec makes timing change what binds.
- **In the rendering it is its own label**, not folded into `strong`, because
  a pass that is *"evidence, once the loop closes"* is not the same claim as
  evidence now. §11.2's argument applies to it as much as to `weak`.
- **Nothing here schedules the later look-back.** Spec §10.7's global validator
  is `long_term_strong` and runs *"at the end of the run"*; a phase inside the
  producing task cannot be where that happens. §15 O10 records that the spec
  names the label and the examples but not the mechanism that revisits one.

### 11.2 The aggregate must carry the qualification, and an empty phase is not a pass

**Criterion 19 is close to unprecedented, and the one precedent is a cautionary
tale.** No surveyed system qualifies a *pass* in its aggregate view. Dagster's
`AssetCheckResult(passed=True, severity=WARN)` is constructible and then ignored
at every consumption site — `execute_step.py:521` requires `not passed` before
severity is read — so a qualified success is structurally unrepresentable there.
Great Expectations is the same. SARIF chose a different axis entirely: confidence
is a `kind` (`pass` / `open` / `notApplicable` / `fail`), and if `kind ≠ fail`
then `level` **SHALL** be `none` and `rank` **SHALL be absent** — you type the
pass, you do not grade it. CodeQL's answer is to not run low-precision queries at
all, which we cannot adopt because a weak validator's *failure* still binds
(spec §5.4).

The precedent that does exist is pytest's XPASS, and it fails in practice.
Measured, pytest 9.1.1: `xpassed` is a separate count with its own `X` progress
character and its own summary prefix — **and the exit code is 0 and the bar is
green.** Issue #11467, opened by a pytest core developer and still open: *"I
recently hit multiple cases where **it was missed that a test was fixed**. That's
not an acceptable default."* Distinguishable rendering was demonstrably not
enough, because the aggregate verdict stayed green. pytest 3.0 also made
non-strict xfail results *"appear as passing tests"* in the JUnit XML — the
qualification erased in exactly the artefact a dashboard reads.

Two consequences for `PhaseOutcome`:

**The aggregate carries the qualification, not only the per-item line.** A phase
whose passes are all `weak` renders differently from one with a `strong` pass,
and the machine-readable form carries the distinction, because that is where
pytest lost it.

**An empty phase is not a pass.** `PhaseOutcome.empty()` is its own outcome. If
the strict level, a config, or a full set of reused verdicts leaves nothing run,
that must not render as the green of "every strong validator passed". Four
systems reached this independently and **none of them spells it "pass"**: pytest
exits **5** for no-tests-collected; Bazel's `test_status.proto` opens with
`NO_STATUS = 0` *before* `PASSED = 1`; GitHub leaves a skipped workflow's checks
**Pending, blocking merge**; SARIF has `notApplicable`.

The counter-example is the one to design against. JUnit XML makes **pass the
structural default** — a `testcase` with no child element is a pass — so a
producer that forgets to emit `<skipped/>` emits a pass and nothing detects it.
Measured: a pytest run with one skipped test yields `failures=0 errors=0` and a
naive consumer reads it as green. **No field of `PhaseOutcome` defaults to
success, and an unrecognised outcome is an error rather than a pass.**

### 11.3 A skip is reported

Criterion 7. Every `SkipRecord` names the validator, why it was skipped, and the
verdict that was reused. Reported unconditionally rather than behind a verbosity
flag: pytest's `-r` defaults to `fE` — failures and errors only — so a skip's
*reason* needs `-rs` to appear at all, and our skip count is small enough that
there is nothing to save.

The incident that makes this worth a sentence: codeql-action#3156, where
`upload-sarif` 3.30.4 *"silently stopped uploading"*. *"The only difference
between the below two runs is the upload-sarif version. The file being uploaded
is identical."* The job stayed green; the failure was visible only as the absence
of alerts.

---

## 12. Test plan

`pytest`. Tests in `agent_sys/tests/validator/`, with an `__init__.py` for the
import-mode reason
[`../../task_graph/docs/design.md`](../../task_graph/docs/design.md) §11 gives.
Every test builds its own `Registry` via `bootstrap.build_registry(...)` with a
`MemoryStoreMgr` and a `FakeRunner`; nothing is process-global.

### 12.1 Spec criteria, mapped

| # | Criterion | Test | File |
|---|---|---|---|
| 1 | A spec missing `brief` / `dimension` / `strength` is rejected; no defaults | `test_each_missing_field_rejected`, `test_no_field_has_a_default` | `test_spec.py` |
| 2 | A validator declaring a subtask or its own input validation is rejected | `test_subtask_field_rejected`, `test_own_input_validation_rejected`, `test_neither_field_exists_on_the_model` | `test_spec.py` |
| 3 | A duplicate name raises; it does not overwrite | `test_duplicate_name_raises`, `test_identical_reregistration_is_a_noop` | `test_registry.py` |
| 4 | One verdict per input handoff, for a validator taking three | `test_three_handoffs_three_verdicts` | `test_protocol.py` |
| 5 | **Invisible to the scheduler**: one dispatch, no pool, no `select` | `test_one_dispatch_for_three_phases`, `test_no_validator_reaches_the_policy`, `test_the_spy_would_catch_a_dispatch` | `test_invisibility.py` |
| 6 | No output handoff; the record persists; the digest is unchanged | `test_phase_produces_no_handoff`, `test_verdict_does_not_move_digest` | `test_phase.py` |
| 7 | A skipped phase is **reported**, and the task advances | `test_skip_by_config_is_reported`, `test_skip_by_prior_verdict_is_reported` | `test_history.py` |
| 8 | `--validation-strict-level` changes which skips are permitted | `test_strict_level_governs_skips` | `test_history.py` |
| 9 | A fresh environment; the §8.2 configuration chain | `test_environment_is_rebuilt`, `test_configuration_chain_order` | `test_isolation.py` |
| 10 | **The producer cannot read the standard**; the spy sees it; the hook denies | `test_no_producer_frame_reads_the_standard`, `test_hook_denies_and_logs` | `test_isolation.py` |
| 11 | A validator in the producing task's zone is rejected, structurally | `test_logic_inside_producer_permissions_rejected`, `test_shared_package_symlink_admitted`, `test_symlink_into_zone_rejected` | `test_separation.py` |
| 12 | Reuse without copy-paste; the shared logic is directly testable | `test_two_instances_one_body`, `test_entry_sh_runs_without_the_registry` | `test_registry.py` |
| 13 | A composite reduces; **a nested composite is rejected at load** | `test_composite_reduces_per_handoff`, `test_nested_composite_rejected_by_schema`, `test_nested_composite_rejected_by_guard` | `test_composite.py` |
| 14 | "Who uses this" and "who has used it", separately; never-run is distinguishable | `test_two_indexes_disagree`, `test_never_run_is_a_state` | `test_registry.py` |
| 15 | All three dimensions present; the registry lists by dimension | `test_list_by_dimension`, `test_every_dimension_is_represented` | `test_registry.py` |
| 16 | Every step of the §10 reference workflow is expressible | `test_reference_workflow_resolves`, `test_system_level_validators_resolve` | `test_reference.py` |
| 17 | A failed phase is indistinguishable from any other task failure | `test_failed_phase_is_an_ordinary_failure`, `test_nothing_downstream_is_cancelled` | `test_phase.py` |
| 18 | **A failing `weak` fails the phase**, as a `strong` one does | `test_weak_failure_binds_like_strong` | `test_report.py` |
| 19 | **A passing `weak` is a low-confidence pass**, distinguishable | `test_weak_pass_is_qualified_in_the_aggregate` | `test_report.py` |
| 20 | The strict level changes which phases run, **never which verdicts bind** | `test_strict_level_cannot_reach_the_fold`, `test_reused_failure_still_fails` | `test_history.py` |
| 21 | A validation environment is a **rebuild, not a reuse** | `test_producer_leavings_absent`, `test_rebuild_not_reuse_across_consecutive_runs` | `test_isolation.py` |

### 12.2 Tests beyond the criteria

Measured facts a future change could silently break:

| Test | Guards |
|---|---|
| `test_protocol_is_not_the_admission_gate` | §3.2. Asserts `isinstance(obj, Validator)` is `True` for `strength=None`, so nobody later replaces the pydantic gate with it |
| `test_inputs_as_a_bare_string_rejected` | §3.2. `inputs="trace"` iterating as five characters |
| `test_composite_refuses_an_uncovered_handoff` | §6.4. `all([])` is `True`; the vacuous pass |
| `test_member_omitting_a_declared_handoff_raises` | §6.4. `None` folded as falsy is not a verdict |
| `test_unresolvable_logic_path_is_rejected` | §9.3. **The inverted fail-closed.** A dangling validator symlink must be *rejected*, where the same helper in `handoff` denies |
| `test_prefix_sibling_is_not_contained` | §9.3. `zone` versus `zone-EVIL`, which the lexical checks fail |
| `test_empty_phase_is_not_a_pass` | §11.2. The JUnit failure mode |
| `test_no_outcome_field_defaults_to_success` | §11.2. Asserted over the model's fields, since the guarantee is the shape |
| `test_binding_symlink_disagreeing_with_inputs_is_reported` | §10.4. The redundant statement that can disagree |
| `test_dangling_binding_symlink_is_a_load_error` | §10.4. Spec §9.1's "a load error naming the path" |
| `test_long_term_strong_renders_as_itself` | §11.1. Three strengths, not two — it must not fold into `strong` |
| `test_logic_source_is_recorded_and_unverified` | §3.3. Asserts the field round-trips and that nothing checks it, so the gap stays visible |
| `test_the_runner_never_names_the_scheduler` | §12.3 |
| `test_args_reach_the_body_as_a_file` | §10.6. `args.json` in the zone, readable by a script and referable by a readme |
| `test_body_paths_resolve_at_load` | §3.8. A dangling `entry` or `material` is a load error naming the path |
| `test_agent_bodied_and_script_bodied_validators_are_substitutable` | §3.8. One verdict file, two ways of producing it — the property the callable could not have |

### 12.3 Two tests that carry more weight than their size

**`test_invisibility.py`** is criterion 5, and it reuses all three devices of
`tests/task_graph/test_authority.py`: subclass-and-log rather than stack
inspection, bracket-and-assert-set-membership, and a third test that **plants the
erosion** so the spy is known to be able to fail.

There are exactly three surfaces to spy, verified against the real scheduler
(`probes-validator/p4_c5_invisibility.py`): `runner.start` (`scheduler.py:246`),
`resource:<name>.take` (`:227`), and `policy.select` (`:203`). A validation phase
adds to none of them, **because the runner never returns to the scheduler between
phases** — that structural fact is what makes the criterion assertable rather
than aspirational.

Two things the test must get right, both measured:

**`select` fires more often than there are dispatches** — four passes for two
tasks, two of them with an empty eligible list. So the assertion is *"no
validator name ever appears in a `select` argument"*, not a call count.

**"Pool" means the resource pool.** `Scheduler.pools` (`scheduler.py:26`) is a
derived index over the whole `TaskStatus` enum, so adding `INPUT_VALIDATING` and
`OUTPUT_VALIDATING` **creates two index pools by construction**. A test asserting
"no validator occupies a pool" over `Scheduler.pools` would fail on a correct
implementation. A task sitting in the `OUTPUT_VALIDATING` index is correct; the
real assertion is one lease taken once and held across all three phases, which is
`task_graph` criterion 40.

**`test_the_runner_never_names_the_scheduler`** is the static half, and it must
walk the **AST**. Measured today: `"scheduler" in runner.py` source text is
**`True`** — two docstring mentions — while an AST walk over names, attributes
and imports returns **0**. `test_authority.py`'s existing static check *is* a
substring grep (`test_the_scheduler_never_takes_a_mutable_handle`), so copying it
naively produces a test that fails for the wrong reason.

### 12.4 Naming the freshness tests after their ancestors

Criterion 21 decomposes into two tests, and only two surveyed systems test this
property at all. Their names are ours, deliberately:

- `test_producer_leavings_absent` ← Bazel's `test_sandbox_undeclared_deps`, and
  Concourse's `It("doesn't mount its file system into the next task")`.
- `test_rebuild_not_reuse_across_consecutive_runs` ← Bazel's
  `test_sandbox_old_contents_not_reused_in_consecutive_builds`.

The conformance test itself is ~40 lines and sub-millisecond in process
(`probes-validator/e21_leak_test.py`), so there is no cost argument for asserting
the property instead of testing it. It must enumerate **channels**, not check a
directory — §8.4 lists the six a fresh directory does not close.

---

## 13. Build versus adopt

| Module | Considered | Chosen | Why |
|---|---|---|---|
| `protocol` | `abc.ABC`, `typing.Protocol`, pydantic model | **`Protocol`, for typing only** | Structural typing suits a seam an external package implements. It is explicitly *not* the runtime gate — §3.2 measures why `issubclass` raises and `isinstance` is presence-only |
| `spec` | JSON Schema alone, `dataclasses`, `attrs` | **pydantic v2** | Already installed. Catches the wrong type, the missing field, and the extra key that no schema keyword reaches once the document is a Python object. Coerces `list → tuple`, which jsonnet's output needs |
| `registry` | a generic registry shared with the other three | **`SpecRegistry` subclass** | [`../../docs/design.md`](../../docs/design.md) §5.1. Four registries with three different index sets; the shared part is the loader-facing base |
| `composite` | Inspect AI's `multi_scorer`, DeepEval's `DAGMetric`, OpenAI's multigrader | **own, ~40 lines** | The shape is Inspect's (reduce per key across members) and is worth copying; the code is not adoptable — Inspect's reducers are `Score`-typed and epoch-oriented, and it rejects the mismatched keys spec §4.1 permits |
| `reducers` | `all`/`any` builtins, Inspect's registry | **stdlib `all`, behind the Protocol** | The alpha needs one reducer and it is a builtin. The Protocol is what makes the second one an addition |
| the body | a registered Python callable; **a task-shaped body** | **a task-shaped body** |
| `history` | a content-addressed verdict cache, `pytest-cache`, dbt `state:` | **the verdict record** | §7.1. The record already carries the answer; a cache would be an index into it, and every key scheme measured has a stale hit on "the implementation changed" |
| `separation` | Bazel visibility, `import-linter`, `dependency-cruiser` | **own, ~20 lines** | The comparison is two declared path sets. Every candidate is a tool for a different graph, and two of the three are symlink-defeated (§9.3) |
| the hook seam | `claude-agent-sdk` directly | **a Protocol, one adapter** | §8.1. Undeclared; 376 MB installed, 26 extra packages, ~1.3 s to import (`agent` design §8.1, correcting rev. 1's figures). The design cannot pin a seam the repository has not chosen, and `agent` made it an optional extra for the import cost alone |
| tests | — | `pytest` | Already the repository's |

**Nothing new is adopted.** pydantic is already installed and already used by
`task_graph`. The one dependency this module *would* add — the agent SDK — is
behind a Protocol precisely so the choice can be made when there is code to
make it against.

---

## 14. Deviations

Each is a place where implementing the spec literally does not work, or where
the spec's stated evidence does not hold. **None changes an acceptance
criterion.**


| # | Spec says | This design does | Why |
|---|---|---|---|
| **D1** | §6.2: *"Inspect AI ships exactly this… **Nesting is not permitted**"* and *"DeepEval's `DAGMetric` is the model"* for one level deep | Rejects nesting by schema omission, citing **OpenAI and Gatekeeper** | Both citations are wrong, verified first-hand. Inspect's `_multi.py` (0.3.260) is 69 lines with **no depth check** and a 3-deep composite executes; DeepEval's `DeepAcyclicGraph` imposes two structural rules, neither a depth limit. **The one-level rule itself stands** — OpenAI forbids it *structurally*, by omitting `Multi` from the nested union in its published schema, which is stronger than what the spec claims for Inspect |
| **D2** | §6.2: `reduce="all" \| "any" \| "at_least(k)"` | A `Reducer` Protocol with a name table; the alpha registers **`all` only** | The spec's three names are ours, not adopted — Inspect's registered set is `collect, at_least, pass_at, pass_k, max, mean, median, mode`, and `multi_scorer([...], "all")` raises `LookupError`. Injecting the reducer makes the other two additions rather than changes, and the alpha needs one |
| **D3** | Spec §6.0 cites Dagster #16569 as convergence on separating severity from blocking | Cites it as a **contrast** | The reading is backwards. #16569 is Dagster *deliberately decoupling* the two, not users conflating them. And `AssetCheckResult(passed=True, severity=WARN)` is constructible then **ignored at every consumption site**, so a qualified pass is structurally unrepresentable there. Ours is decided statically, so their stated objection does not bind us — but the convergence claim does not hold |
| **D4** | §9.3 lists five load-time checks; criterion 11 is separate | Criterion 11 runs as a **sixth check in the closure pass** | It needs the task registry and the validator registry both loaded, which is [`../../docs/design.md`](../../docs/design.md) §6.1's argument for the pass existing. Placement, not substance |
| **D5** | §6.1: instances come from a `{name, args}` table | The args live in the **validator spec**, one spec per instance | §10.6. The alternatives — args in the binding (dbt) or a schema shipped by the implementation (Gatekeeper) — both have precedent, and the first would change the type of `validators_for(kind)` and force handoff design §8.3 to rule on differing args per binding. Recorded as a choice because spec open question 2 leaves it open |
| **D6** | Spec §6.1 rev. 8 — a validator's logic is a body | **Implemented as written, and it costs a check that rev. 2 had** | §3.8, §10.2, §10.6. The model is right and the user's argument decides it: a callable cannot express a validator an agent is responsible for without a wrapper that runs an agent. What goes with the callable is **pandera's `inspect.signature` check** — four lines that rejected args a check would silently ignore, shipped because pandera#480 was a check that validated a frame it should have rejected. A shell script has no signature and a description has none either. Three weaker things stand in its place (§10.6) and they do not add up to it. Recorded as a deviation from *this document's rev. 2* rather than from the spec, because the spec never asked for the check — but a reader comparing the two revisions should not have to work out what was traded |

---

## 15. New open questions

Found by this design, and **not** in spec §12.

| # | Question |
|---|---|
| **O1** | **Criterion 5's "no validator occupies a pool" is ambiguous, and the literal reading is unsatisfiable.** `Scheduler.pools` comprehends over the whole `TaskStatus` enum, so the two validating statuses create two index pools by construction. §12.3 reads it as *resource* pool and asserts `task_graph` criterion 40 instead, but the criterion's wording should be tightened |
| **O2** | **A reused verdict does not prove a re-run would agree.** §7.4. The record names the validator, not its implementation, and §9.3 forbids reading `version` at runtime. Nix names this exactly — *"there is no way to audit a build trace entry except for by performing the build again"* — and calls trust in it a subjective policy choice. If a stronger identity is ever wanted, Bazel's is the shape: the tool's bytes plus a hand-bumped GUID per action class, under the contract *"if the work to be performed changes, the key must change"* |
| **O3** | **Nothing checks that a body reads the args it was configured with.** §10.6. Rev. 2 asked this only of *agent-run* validators, because `inspect.signature` covered the code-backed ones; rev. 3's body has no signature in either case, so the asymmetry is gone and the gap now applies to both. Gatekeeper's answer is still the one with precedent — the implementation ships a schema for its own args — and it would fit a body: a validator folder could carry an `args.schema.json` beside its `entry.sh`. Not built. **This is the one place rev. 3 is strictly weaker than rev. 2**, and §14 D6 says so |
| **O4** | **May a handoff kind supply args to the validator it binds?** Spec open question 2 raises it. Under D5 it cannot, and the question does not arise; if it later can, `validators_for(kind)` changes type and handoff design §8.3 must rule on two kinds binding one validator with different args |
| **O5** | **`TaskStatus` lacks `INPUT_VALIDATING` and `OUTPUT_VALIDATING`.** `task_graph` spec §3.2.2 rev. 12 specifies them; `models.py:44` has eight members and neither. Module 4's change, but this design's §5 is written against a state that does not exist yet |
| **O6** | **Nothing detects a wrong `cost` tag.** §5.3. Ordering by a declared cost tag has no prior art at all; the two nearest systems warn advisorily (Bazel, off by default, with an irreducible false-positive rate under variance) or degrade silently (pytest-split substitutes the population mean and discards orphans). The design records actual durations so a later change can report disagreement; nothing consumes them yet |
| **O7** | **Who owns the enumeration of reference kinds for "who uses this"?** §10.5. There are two edges today — a kind naming a validator, a composite naming a member. Airflow's asset orphanage reported live assets dead because one reference kind was missing from its join (#58058). Nothing here owns that list, and a missing edge produces false-positive deadness, which is the failure mode of every derived static index |
| **O8** | **Does a validator's deletion erase its history?** §10.5. Airflow tombstones and keeps the record; dbt drops orphans in a silent set intersection. "Has this ever run" is meaningless if deletion erases the answer and unbounded if nothing ever prunes. This is the same GC problem [`../../handoff/docs/design.md`](../../handoff/docs/design.md) §15 O3 records between an artefact and its verdict, arriving from the registry side |
| **O9** | **Does "every validator here was weak" deserve its own treatment**, distinct from "some weak, some strong"? §11.2 qualifies the aggregate, but the case carrying *no* strong evidence at all is different in kind, and pytest's third-colour precedent is the only prior art for it |
| **O10** | **`long_term_strong` has a label and examples but no mechanism.** §11.1. Spec §5.3 says the rigour is the same and only the *timing* differs, and §10.7's global validator is one — *"looks back at the end of the run"*. But a phase inside the producing task is over long before then, and nothing in the spec set says what revisits a `long_term_strong` verdict, or what its verdict means in the meantime. This design folds it as `strong` and renders it as itself, which is the least-surprising reading of §5.4, not an answer |
| **O11** | **Should a failed name lookup enumerate its candidates by default?** [`../../docs/design.md`](../../docs/design.md) §5.2 says yes, from pytest and dbt. Bazel is moving the other way — #25941, #25933, #25940, all open — toward one greppable line with detail behind `--verbose_visibility_errors`, because multi-line output *"makes grepping harder"*. Ours are short lists today, so the default is fine; the question is whether it stays fine |
