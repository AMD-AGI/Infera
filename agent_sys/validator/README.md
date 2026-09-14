# validator

**A handoff is only a contract if something checks it.** Validators are the sole
standard by which a handoff is judged, and this package is where a check plugs
in, what it may claim, and what stops the producer from grading itself.

| | |
|---|---|
| Specification | [`docs/spec.md`](docs/spec.md) rev. 8 — 21 acceptance criteria |
| Design | [`docs/design.md`](docs/design.md) rev. 3 |
| Seam | [`../docs/interfaces.md`](../docs/interfaces.md) §4.3 — normative |
| Contract | [`protocols.py`](protocols.py) + `protocols.pyi`, declarations only |

---

## What is here

| File | |
|---|---|
| `protocols.py` | The frozen half of §4.3. Enums, the two Protocols, `Body`, the records. **Every module below imports its vocabulary from here rather than re-declaring it** |
| `spec.py` | `ValidatorSpec` — the pydantic admission gate — plus `Tags`, `Cost`, `LogicSource`, and the two load-time path checks |
| `registry.py` | `ValidatorSpecRegistry(spec_loader.BaseSpecRegistry)`: the shared name table, plus admission and the two indexes |
| `composite.py` | One level deep, reducing **per handoff** |
| `reducers.py` | `all`, behind the `Reducer` Protocol. One entry in the alpha |
| `history.py` | The skip decision, read from the verdict record |
| `separation.py` | Criterion 11 — the producer may not legislate its own standard |
| `environment.py` | A validation environment is a rebuild, never a reuse |
| `boundary.py` | The hook seam: the spy and the denier are one object |
| `report.py` | `PhaseOutcome`. No field defaults to success |
| `phase.py` | `PhaseRunner` — the seam `agent.Runner` calls, twice per dispatch |
| `general_specs/` | The three shipped workflow-independent validators, one per dimension |

---

## The 21 criteria, and the test that holds each

`spec.md` §11. **Checked against the tree rather than transcribed** —
`scratch/impl-2026-08/validator/p7_criteria_map.py` parses every
`tests/validator/test_*.py` and fails if a name here does not exist. Two of these
had been renamed since they were last written down in a message, which is why the
probe is kept.

| # | What it requires | Test |
|---|---|---|
| 1 | No `brief` / `dimension` / `strength` ⇒ rejected. No defaults | `test_each_missing_field_rejected`, `test_no_field_has_a_default` |
| 2 | A subtask or its own input validation ⇒ rejected | `test_subtask_field_rejected`, `test_own_input_validation_rejected`, `test_neither_field_exists_on_the_model` |
| 3 | A duplicate name raises; it does not overwrite | `test_duplicate_name_raises`, `test_identical_reregistration_is_a_noop` |
| 4 | One verdict per input handoff | `test_three_handoffs_three_verdicts`, `test_two_inputs_of_one_kind_are_two_entries` |
| 5 | Invisible to the scheduler | `test_one_dispatch_for_three_phases`, `test_no_validator_reaches_the_policy`, `test_the_spy_would_catch_a_dispatch` |
| 6 | No output handoff; the record persists; the digest is unchanged | `test_phase_produces_no_handoff`, `test_verdict_does_not_move_digest` |
| 7 | A skipped phase is **reported** | `test_skip_by_config_is_reported`, `test_skip_by_prior_verdict_is_reported` |
| 8 | The strict level changes which skips are permitted | `test_strict_level_governs_skips` |
| 9 | A fresh environment ✅; §8.2's configuration chain **⚠ ¼ unreachable** | `test_environment_is_rebuilt`, `test_configuration_chain_order`, `test_the_bound_row_takes_the_named_agents_env`, **`test_the_producer_row_is_live_when_the_runner_has_one`**, `test_a_non_leafs_empty_configuration_falls_through_to_global`, `test_a_validator_naming_no_agent_still_takes_the_global_row` |
| 10 | The producer cannot read the standard | `test_hook_denies_and_logs`, `test_no_producer_frame_reads_the_standard`, `test_the_verdict_names_the_checking_agent_not_the_producer` |
| 11 | A validator in the producer's zone ⇒ rejected, structurally | `test_logic_inside_producer_permissions_rejected`, `test_shared_package_symlink_admitted`, `test_symlink_into_zone_rejected` |
| 12 | Reuse without copy-paste; the logic is directly testable | `test_two_instances_one_body`, `test_entry_sh_runs_without_the_registry` |
| 13 | A composite reduces; a nested one is rejected at load | `test_composite_reduces_per_handoff`, `test_nested_composite_rejected_by_schema`, `test_nested_composite_rejected_by_guard` |
| 14 | "Who uses this" and "who has used it", separately | `test_two_indexes_disagree`, `test_never_run_is_a_state` |
| 15 | All three dimensions shipped; listable by dimension | `test_list_by_dimension`, `test_every_dimension_is_represented` |
| 16 | Every step of §10's reference workflow is expressible | `test_reference_workflow_resolves`, `test_system_level_validators_resolve` |
| 17 | A failed phase is an ordinary task failure | `test_failed_phase_is_an_ordinary_failure`, `test_nothing_downstream_is_cancelled` |
| 18 | A failing `weak` fails the phase, as a `strong` one does | `test_weak_failure_binds_like_strong` |
| 19 | A passing `weak` is a low-confidence pass, distinguishable | `test_weak_pass_is_qualified_in_the_aggregate` |
| 20 | The level changes which phases run, never which verdicts bind | `test_strict_level_cannot_reach_the_fold`, `test_reused_failure_still_fails` |
| 21 | A validation environment is a **rebuild, not a reuse** | `test_producer_leavings_absent`, `test_rebuild_not_reuse_across_consecutive_runs` |

44 named tests carry the criteria; the rest of the ~142 are design §12.2's guards
over measured facts a future change could silently break, and the findings other
packages brought in.

### An unchecked output is a fault, and it blocks — `interfaces.md` §4.15

**Ruled by the user, against what this package had concluded.** `report.py` had
derived that the answer *cannot* be "it blocks": `StrictLevel.NONE` folds every
phase to `empty`, so a blocking `empty` would be the knob deciding an outcome.
The ruling makes the two `empty`s distinguishable instead, and the field that
does it is **`PhaseOutcome.verdicts_expected`**.

| `empty` and… | means | blocks |
|---|---|---|
| nothing was asked — level is `NONE`, or the task has no handoff in this position | no validation was asked for | no |
| verdicts were expected, phase is **output** | **nothing checked what this task produced** | **yes**, `Evidence.UNCHECKED` |
| verdicts were expected, phase is **input** | this task consumes nothing | no |

**Criterion 20 is untouched and this was checked rather than assumed.** The
level reaches the *choice of constructor* and nothing else — `fold`'s parameters
are unchanged, so `test_strict_level_cannot_reach_the_fold` passes verbatim, and
the fault rule is identical at every non-`NONE` level. The two "nothing was
asked" sites call `PhaseOutcome.nothing_expected`.

**The reading is the narrow one, deliberately.** §4.15's sentence is *nothing
checked what this task **produced***, so a task with no output has nothing
unchecked. Measured: `examples/demo/closures/` gives `main` (`outputs: []`,
`validators: []`) and `consume` (`outputs: []`, one validator declared), so the
wide reading blocks the demo's **root** task. No demo task has an output with
nothing bound, so the fault arm is unreachable in `demo run` as it stands.
`test_a_task_with_no_output_has_nothing_unchecked` carries it.

### **20¾ of 21**, and criterion 9 is the quarter

**The table is deliberately not uniform**, following `env_mgr`'s 21½-of-22: a
mapping is a claim about the **system**, not about the tests, and a row that reads
like the other twenty while covering part of a criterion is worth less than a row
that says so.

Criterion 9 is *"a validation runs in a fresh agent environment, and the default
configuration follows §8.2's chain."* The **fresh environment is real** —
allocated, never reused, `CHANNELS`-tested. **Three of the chain's four rows are
now real**: `bound`, `producer` and `global` work; `consumer` has no source and
**cannot have one**, which is a different claim from the other three and the
reason this is a quarter rather than a gap somebody could close.

| §8.2 row | Reachable | Needs |
|---|---|---|
| bound — the agent's declared environment | **yes** | `ValidatorSpec.agent` names a spec; `agent_specs.get(name)["env"]` is the row. **Not** an inline block: `agent.schema.json` already has `env`. Three packages to land one row — `spec_loader`'s key, this field and resolve, `closure`'s catalogue check |
| consumer — for input validation | **no** | **Unreachable in principle, not unbuilt.** The configuration is `env_mgr.Prepared.environment`; `env.prepare` has one call site (`agent/runner.py:668`, from `_deploy`) and `_one_phase` reaches `_main` only in `RUNNING` — strictly after `INPUT_VALIDATING`. §8.2's phrase for this row is *"the task about to run"*, and about-to-run is exactly before `prepare` |
| producer — for output validation | **yes**, since `agent` `3155ca2` | `attempt_of(task.id).environment` — a read-only `Mapping[str, str]` carried on the `TaskAttempt` from `_deploy` onwards. It was a discarded local of `_deploy`; `agent` added the field on request. **Empty reads as absent**: `{}` means the task never deployed, which is every **non-leaf**, and §8.2's row is *the configuration already resolved* |

**Absent and unresolvable are different questions and do not share an answer**, which is `closure`'s correction of a conflation of mine. A validator naming no agent takes the global row; one naming `profilr` **raises**, because falling back would hand the author a *working* environment that is not the one they configured. The silent version's symptom is the bad one: a validator that **runs**, in the wrong environment, producing a verdict somebody trusts.
| global | yes | the `validation_env` component |

`CONFIGURATION_SOURCES` in `phase.py` enumerates all four with the reason each
does or does not have a source, and `test_only_the_global_row_is_reachable_today`
asserts it **through the phase runner** — which is where the gap is, and where
`test_configuration_chain_order` could not see it.

**How it hid, because the shape recurs** (`interfaces.md` §8.8). The three rows
were fed by `getattr(spec, "environment", None)` and
`getattr(task, "environment", None)`, and neither type has that field —
`ValidatorSpec` sets `extra="forbid"`, so no document can add one. A `getattr`
with a default is not a field access; it is dead code that reads as live.
`test_configuration_chain_order` is a **correct** unit test of a pure function
and calls it directly, so all four rows pass forever: *its coverage was real and
its implication was not.* The tell is one question — **does anything assert that
the caller can produce these inputs?**

---

## Libraries adopted, and why

**Nothing new is adopted.** Every dependency below was already installed and
already declared or already used by a sibling package.

| Concern | Considered | Chosen | Why |
|---|---|---|---|
| The name table | a private dict, a generic registry shared with the other three | **`spec_loader.BaseSpecRegistry`** | Design §10.1: the base supplies the dict, the collision policy and the error shape, and the subclass adds this kind's checks through `_validate`. It was a private dict here until `spec_loader` shipped the base; subclassing removed ~35 lines and brought `origin_of`, which §9.3 check 4's *name both sides* report needs |
| The admission gate | JSON Schema alone, `dataclasses`, `attrs`, `typing.Protocol` | **pydantic v2** | Already installed and already `task_graph`'s. It catches the three faults no schema keyword reaches once the document is a Python object — the wrong type, the missing field, the extra key — each naming the field in `loc`, and it coerces `list → tuple`, which a YAML sequence needs (it was jsonnet's JSON array output that needed it before, and the coercion is the same one). Re-measured in `scratch/impl-2026-08/validator/p1_pydantic_shapes.py` rather than taken from the design |
| The static type | `abc.ABC`, pydantic model | **`typing.Protocol`, for typing only** | Structural typing suits a seam an external package implements. It is explicitly **not** the runtime gate, and the shipped fact is stronger than the design's: `Validator` is not `runtime_checkable`, so `isinstance` *and* `issubclass` both raise. There is no way to use it as a gate even by mistake |
| Schema validation in tests | `jsonschema` directly | **`spec_loader.validate` + `schema_for`** | The schema lives in `spec_loader/schemas/` (design §2) and `$ref`s `_common.schema.json`, so it needs a registry; `validate` is the system's single enforcement point and already carries one. Calling `jsonschema` directly here would be a second reader of a file this package does not own |
| Reading the shipped general specs | hand-written Python records, PyYAML, `ruamel.yaml` called directly | **`spec_loader.yaml_source.read_yaml`** | Main spec §4.5: the main repository gets no private path for its own specs, so the files on disk are ordinary YAML documents and `tests/validator/test_reference.py` reads them from disk rather than transcribing them into three dicts. **Amended 2026-08-29**: this row said `_jsonnet`, and said `spec_loader.render` was the real path to swap to. Both halves resolved in opposite directions — `render` no longer exists (main spec §7 rev. 10), and the intent survives. Not PyYAML and not `ruamel` called here: measured, round-trip is YAML 1.2 and `safe_load` is 1.1, so `12:30` is a string on one side and 750 on the other, and a second parser in a test is how a document comes to mean two things. `load_package` is deliberately *not* used — these are documents, not a package, and they have neither of §4.3's two mandatory names because they are not one |
| Fresh zone allocation | `shutil.rmtree` + recreate, `tempfile.TemporaryDirectory` | **`tempfile.mkdtemp`** | **Freshness comes from allocation, never from cleanup.** pytest's `tmp_path` is a new *numbered* directory and its cleanup is explicitly best-effort; a guarantee that depends on a teardown succeeding is not a guarantee. `mkdtemp` also never returns a path twice, which is the property criterion 21 actually turns on |
| The composite | Inspect AI `multi_scorer`, DeepEval `DAGMetric`, OpenAI multigrader | **own, ~60 lines** | The *shape* is Inspect's and is worth copying — reduce per key across members. The *code* is not adoptable: Inspect's reducers are `Score`-typed and epoch-oriented, and it rejects the mismatched keys spec §4.1 permits, because its keys are epochs of one sample while ours are handoffs |
| The reducer | `all`/`any` builtins, Inspect's registry | **stdlib `all`, behind the Protocol** | The alpha needs one reducer and it is a builtin. The Protocol is what makes the second one an addition rather than a change. Inspect's registered set does not contain `all` at all — `multi_scorer([...], "all")` raises `LookupError` there — so the spec's three names are ours |
| The skip decision | a content-addressed verdict cache, `pytest-cache`, dbt `state:` | **the verdict record** | Five key schemes were measured against six things that can change the answer, and *"the validator's implementation changed"* is a stale hit under **every** one, because implementation source appears in no spec file. The record **is** the answer rather than an index into one |
| Separation | Bazel visibility, `import-linter`, `dependency-cruiser` | **own, ~30 lines** | The comparison is two declared path sets. Every candidate is a tool for a different graph, and two of the three are symlink-defeated |
| Containment | `handoff.check_contained` | **own** | **The fail-closed direction is inverted.** There *contained* means allow, so unresolvable means deny; here *contained* means **reject**, so unresolvable must be treated as inside. Importing it and negating at the call site would negate the fail-closed behaviour too, and a dangling validator symlink would be accepted |
| The hook | `claude-agent-sdk` | **a Protocol, one SDK-free implementation** | 376 MB installed, 26 extra packages, ~1.3 s to import, and `agent` §8.1 made it an optional extra for the import cost alone. The design cannot pin a seam the repository has not chosen |
| Tests | — | **`pytest`** | Already the repository's |

### `python-jsonpath` — chosen by `handoff`, and this package has **no caller**

Worth recording because three documents say otherwise and two of them are mine.
`handoff` §8.4 measured six libraries and chose `python-jsonpath` for the RFC 6901
resolver, on a property nothing else has: **three outcomes, three answers.**
`jsonpointer` raises one class for both a malformed pointer and a missing value,
collapsing the distinction the whole design turns on; `jsonpath-ng` leaks bare
`IndexError` / `KeyError` out of validly-parsed queries; and no *JSONPath*
library can work at all, because RFC 9535 §2.5.1.2 forbids a valid query from
erroring, so none can tell a wrong path from an absent value.

The trap it walked past is worth keeping: **both rejected libraries were
installed on this machine and the chosen one was not**, so a Pointer test written
before 2026-08-28 would have passed locally against a rejected library and failed
on a clean install.

**`validator` imports none of it.** Measured — `handoff.resolve` has zero callers
anywhere in the tree. After spec rev. 8 a validator's implementation is a
**body**, so whatever addresses into content is a shell script or an agent inside
the zone, and there is no Python here between the spec and the content for a
pointer to be resolved by. `interfaces.md` §4.3, `validator` spec §4.1 and
`validator` design §3.6 all say this module consumes it; §5.8 records that it does
not, and that the spec line cannot be closed by editing a design.

---

## The four properties worth knowing before changing anything here

**A validation phase is invisible to the scheduler.** One dispatch, one lease,
three phases. There are exactly three surfaces to spy — `runner.start`,
`resource:<name>.take`, `policy.select` — and a validation phase adds to none of
them, *because the runner never returns to the scheduler between phases*. That
structural fact is what makes criterion 5 assertable rather than aspirational.

**`PhaseOutcome` never defaults to success.** `PhaseOutcome.fold()` with nothing
folded is `empty`, and `empty` is **not** a pass. Four systems reached that
independently and none of them spells the third state "pass". JUnit XML is the
counter-example: pass is its *structural* default, so a producer that forgets to
emit `<skipped/>` emits a pass.

**The strict level cannot reach a verdict.** `fold` takes no level, so there is
no argument through which the knob could touch one — the ESLint variable split as
a signature rather than as care. ESLint tried the other way twice and #19625
found the second miss 22 months later.

**A failure binds at every strength; the label qualifies the pass.** A failing
`weak` validator fails the phase exactly as a `strong` one does. What differs is
what a *pass* is worth, and the qualification is carried by the **aggregate**
(`PhaseOutcome.evidence`), not only the per-item line — because that is where
pytest's XPASS lost it.

### Building a `ValidatorSpec` fixture

**Use `tests/validator/conftest.py::validator_record`.** It produces one
admissible record and is the shorter answer for anyone who can import it. The
rest of this section is for anyone who cannot.

**Say which gate you mean.** A spec crosses four and they do not agree:
`validator.schema.json`, then `ValidatorSpec(...)`, then `admit()`, then
`ValidatorSpecRegistry.add()`. A fixture author is going through the last two.
Measured:

| | model ctor | `admit()` |
|---|---|---|
| the six required fields, no `body` | accepts | **rejects** — *declares neither a body nor members* |
| `body: {}` | accepts | **rejects** — *body.readme: Field required* |
| `body: {"readme": ...}` | accepts | accepts |
| `members` + `reduce`, no `body` | accepts | accepts |

So the smallest thing a fixture can actually use is a **leaf** with a body:

```python
admit({
    "name": "x", "brief": "b", "inputs": ["trace"],
    "dimension": "completeness", "strength": "strong",
    "tags": {"logic_source": "external_static", "cost": "seconds"},
    "body": {"readme": "readme.md", "entry": "entry.sh"},
}, origin="s.yaml")
```

Six required fields — `name`, `brief`, `inputs`, `dimension`, `strength`, `tags`
— and inside `tags`, `logic_source` and `cost` are required while `domain`
defaults to `()`. The enums are the other trip: `strength` is
`strong` / `long_term_strong` / `weak`, **not** `blocking`.

**`body` is not required by the model** — it defaults to `{}`. It *is* required
of a leaf by `admit`, and a composite is excused because `members` is its
implementation. Both halves matter and each on its own is wrong in a different
direction: `closure` reported `body` as required and generalised from one error
message; I corrected them with a true statement about the model, which their
fixture then failed against `add()`. **Neither of us said which gate we meant,
and each of us meant the one we had most recently hit.**

The reason the distinction is worth the space rather than a footnote: a fixture
author who adds a stub body to satisfy a requirement they think the model has
can make an agent-bodied validator by accident — the `entry: ""` failure by
another route — while one who omits it gets a loud, accurate rejection. Same
missing sentence, two failures, only one of them noisy.

---

## Reported, not decided quietly

Each names both sides.

**0. `Verdict`'s field names are not criterion 8's words, and `agent_id` is now
optional.** Criterion 8 lists *"which task and which versioned agent ran the
validation"* and the prose says **task, agent, timestamp**; the fields are
**`task_id`**, **`agent_id`**, **`at`**. `handoff` flagged it before either of us
shipped against the other — a wrong guess is a construction `TypeError`, which is
at least loud. Two corrections followed, and both were mine:

- Every verdict named the **producing** agent as its own checker, until `agent`'s
  executor returned a fresh unbound `AgentId` per phase. A record saying the
  producer validated its own artefact is the claim spec §8.1 forbids, in the one
  artefact that outlives the run.
- A **script** body has no agent, and `agent_id` was non-optional, so it fell back
  to the producer's id. `handoff` widened the field to `AgentId | None` (`f9142aa`)
  rather than take a sentinel: a sentinel `AgentId` is a UUID, so a reader who
  does not know it takes it for a real agent and one who looks it up finds
  nothing. `None` cannot be mistaken for an agent.

`strength` and `dimension` are plain `str` on the persisted record deliberately —
the enums are this module's, and a stored file stays readable when a member is
added.

**1. `PhaseOutcome.empty` is a field and a constructor at once.**
`validator/protocols.py` declares `empty: bool` as a field; `docs/interfaces.md`
§4.3 and design §5.2 call `PhaseOutcome.empty()` a classmethod. A dataclass
cannot have both. **The field wins here**, because it is the importable contract
that the `.pyi` and `tests/interfaces/test_stub_agreement.py` guard, and
`PhaseOutcome.fold(kind)` with nothing folded is the constructor. One name has to
give; which one is not this package's call alone.

**2. `logic_source` had two writers.** Design §3.2's model lists it as a
top-level field, and §3.5 says the tag dictionary carries everything that is not
`dimension` or `strength`. It is implemented **on `Tags` only**, with
`ValidatorSpec.logic_source` as a read-only delegation, because two records of
one fact is `engineer_principle.md` §1's failure. Internal to this module; noted
so a reader of the design is not surprised.

**3. ~~Two `validator.schema.json`~~ — closed.** `spec_loader/schemas/` holds
the one schema; the copy here is deleted and the criterion-13 test reads
`schema_for("validator")`. `spec-loader` wrote theirs from the spec before
reading `general_specs/`, and all three shipped specs passed on the first try
including the `body` shape, which no spec fixes and which we each had to choose
independently. Five rows disagreed and all five are settled: `args` and top-level
`members` + `reduce` by `main`'s ruling (a schema that rejects what an acceptance
criterion requires is the schema being wrong); `tags` extra-keys, `cost` as an
enum and `domain` as a list to the model's side. `body` becoming optional and
`description` being accepted were theirs, and both were right.

One residual asymmetry, small and recorded rather than fixed: the schema gives
`tags.domain` `minItems: 1`, so an explicit `[]` is a fault, while the model's
default `()` means "not declared". A spec that writes an empty list is rejected
by one gate and not the other.

**4. `general_specs/` has no agreed home.** Main spec §4.5 says general specs
live in *"their own directory"* and does not say where. They are here because
this is the module that owns validators. If the repository grows one general-spec
directory for all five spec kinds, they move there unchanged.

**5. §8.1's table has no row for *the producer supplies the evidence*, and
`demo` has now built the case.** With `main`, and it is a spec question rather
than a code one.

The user ruled that grounding data reaching a validator is a **task-declaration**
problem: a task author passes their own input through to their own output, and
the system does not carry a second artefact in for a check. `demo` followed it —
the `summary` kind now requires `items/grounding/`, a verbatim copy of the
producer's input, so `check_grounded` compares two halves that are both inside
the one staged handoff and it works confined.

**The exposure is inherent to the ruling, not to `demo`'s implementation.** The
reference data a differential check compares against now arrives **through the
producer being judged.** §8.1's fourth row is *the producer cannot see the hidden
inputs a differential comparison uses*, and this is a strictly stronger case than
the one that row forbids: the producer does not merely see the reference, it
**authors** it. The table has no row for that because it was written assuming the
reference comes from somewhere else.

So what a `strong` verdict from such a validator asserts has narrowed: **the
summary is internally consistent with the grounding copy the producer carried** —
not that it is grounded in the facts. The two differ exactly when the copy is not
faithful, and **nothing in the confined path can tell them apart**: a body cannot
reach the store (see the two-routes section below), and the ruling forecloses
declaring the original as a second input. Closing it would need a party that sees
both, and today there is none.

`demo` chose a verbatim `cp -r` over *"carry the numerals you used"*, against
`demo-2`'s own suggestion, and that is the right call for the reason they gave:
**copying is mechanical, selecting is a judgement, and the judgement is the thing
under test.** A curated list would be the producer choosing what it may be judged
against, in the one check whose subject is whether it invented a number. That
keeps the exposure as narrow as the ruling allows. It does not remove it.

**Not added as a row here, because the spec is frozen** — a fifth row in §8.1 is
`main`'s and the user's to place. Recorded so the narrowing is visible to a
reader of a `trustworthiness / strong` verdict, who would otherwise read it as
the claim it made this morning.

## Left out, and why

**The kernel isolation layer.** `env_mgr` has no sandbox implementation — a grep
for landlock / bwrap / unshare / seccomp hits only its spec — so `environment.py`
is the **process-perspective** layer only: a freshly allocated zone, an explicit
environment block, `TMPDIR` and `HOME` inside the zone, and `cwd` never
inherited. `CHANNELS` is the measured list of what a fresh directory does *not*
close, and criterion 21's test enumerates it. Criterion 10's test asserts the
hook half plus the declaration half, **never the kernel half**, and `boundary.py`
records the measurement that makes that honest: `Bash{'command': 'python3
reader.py'}` returns ALLOW, because there is no path in the payload to match.

**The mechanism that makes a phase attributable.** *That* a phase must carry an
`agent_id` is this module's requirement and is enforced
(`environment.assert_attributable`). *How* a backend delivers it — subagent,
`fork_session`, `resume`, a second client — is `agent` design O6 and is open, so
`AgentBodyRunner` resolves `validator_executor` by name and fails loudly naming
O6 rather than assuming one.

**The `args` signature check.** There is no signature to read on a shell script
or on a description, so pandera's four-line `inspect.signature` check — shipped
because of a check that validated a frame it should have rejected — is not
available in this shape. Design D6 records it as a real loss. What stands in its
place is that `args` is schema-checked at load and lands in the zone as
`args.json`, so *what was this configured with* is answerable after the fact.
That is less than the check was.

---

## What this package owes, and what it is owed

Open at the last commit. Each is with a named owner, not with nobody.

| | With | State |
|---|---|---|
| **§5.16 — may a reader treat *nothing checked this output* as a fault?** | the **user**, via `main` | A spec question. My side is derived and closed — see `report.py::blocks_the_task` |
| ~~Criterion 9's `bound` row~~ | **closed** — `fe9fd55` + this package's step 3 | Landed as an `agent` key naming a spec, not an inline `environment` block: `agent.schema.json` already has `env`, which is what §8.2 row 1's *"that one"* means, and a second copy would be two writers of one fact. My inline proposal was wrong and `spec-loader` refused it. Four packages, in order — `agent-mod` assents, `spec_loader` adds the key, the field here, `closure`'s catalogue check (open, and blocked below). **The `_PENDING` scaffolding in `test_composite.py` existed only to let the key cross without a red shared suite, and deleted itself on the commit that landed the field.** |
| ~~An accessor over this document's `agent` key, in `spec_loader`~~ | **closed** — `spec_loader.validator_agent_of`, `eff9a18`; this copy withdrawn in `11b034c` | `closure`'s catalogue check needs to read the key and **may import `spec_loader` and nothing else** (§4.5), so the accessor this package built for them was unreachable across the package edge — the module was never the problem. Third instance of `spec-loader`'s own criterion, after `body_of` and `task_of`. It **cannot** be called `agent_of` there: `closure/model.py` already exports one over a `ClosureDoc`, both take a mapping, and neither raises on the wrong document — a collision that is aliasable in a consumer's file and not in the leaf. `validator.agent_of` was deleted rather than kept beside it: it had zero callers here, so keeping both was two writers of one key with no reader on this side |
| **Criterion 9's `consumer` / `producer` rows** | `env_mgr`, and likely `main` | A task's **resolved** configuration has no route to this module. Bigger than a field: it is where a task's configuration lives, and it probably rides the `prepare_validation` seam |
| **§5.15 — what confines a validation *body*** | open, no owner | `prepare_validation` confines nothing by design: `prepare` applies Landlock to its own process and a phase runner is the supervisor. So the producer/validator separation rests on **placement**, not enforcement |
| **§5.8 — which branch materialises a pointer's value** | `handoff` and me, neither inventing | Narrowed, not closed. `materials.json` now names the staged copies in the body's `cwd`, so the *mechanism* exists; *which* branch is still a decision |
| **§5.13's body is stale** | `main` | It still says a script verdict "falls back to the producer's id with `attributed: False`". Both were true this morning; neither is now |
| ~~A map from handoff id to staged path~~ | `env_mgr` — **closed**, `789796d` | `ValidationZone.materials` now carries the association, so `materials.json` is a map. `layout.stage` had the id in hand and was discarding it — `engineer_principle.md` §4.4's second smell, resolved at the end that owned it |
| **Who calls `bind_phase`** | `closure` — **closed**, `check_closures` feeds it | Was the third edge kind recorded and unfed |

Nothing is blocked on any of them.

### One thing owed to whoever reads this next

Six defects in this package were found by other implementers reading a declared
shape and asking a question, and **four were silent** — no exception, plausible
output, wrong answer. None was reachable from `tests/validator`, because a test
asserts what its author thought of.

Two of them had been *written down correctly here* before being implemented
wrongly: `EDGE_KINDS` listed two edge kinds under a comment citing the paper about
listing too few, and `ScriptBodyRunner` said *"a body that reports nothing must
not pass"* and implemented only that half. **A rule written down and
half-implemented looks finished**, because the prose states the whole of it. The
cheap check is to read the comment against the code beneath it — that is
`closure`'s *asymmetry inside one artefact*, and it is spottable by the author
alone.

**The mirror image happened too, and it cost someone else a probe.**
`_configuration_sources`' docstring said *"only `global_`"* for three commits
after `ec5fbba` landed `bound` and `0b64554` landed `producer` —
`CONFIGURATION_SOURCES`, ten lines above it in the same file, was right the whole
time. `demo` asked which §8.2 row their validator body had been given; the answer
came from the prose, was wrong, and they spent a probe measuring
`ConfigSource.PRODUCER` firing all along. So the check runs both ways: **stale
prose beside correct code is a defect with a victim**, and the victim is whoever
reads rather than executes. Fixed in `e163c4f`; the lesson is that landing a
feature includes the paragraph next to it.

### The input phase is a partial backstop, and must not be read as coverage

`task_graph` measured a consumer dispatched against an empty artefact with every
guard reporting valid — the two version counters had diverged, so
`input_versions` named a store directory that was allocated but never written
(`scratch/impl-2026-08/task_graph/probe_consumer_staging.py`). Their path is
**silent**: `allocate` must create `v<N>/content/` for the grant to resolve, and
`layout.stage` skips only when `content/` is *absent*.

That empty input reaches `run_phase(INPUT, …)`, and there are exactly three
outcomes:

| bound to that kind | outcome |
|---|---|
| nothing | the phase folds `empty` with `verdicts_expected`, `blocks_the_task` is true via `unchecked`, and the task **stops** — loudly, but for *nothing checked this input*, not *this input is empty* |
| a body that inspects content | an empty `content/` is what a schema or shape check fails on. This is the case the design intends, and **nothing has built it**: `general_specs`' two bodies still return `true` for every id |
| a body that does not inspect content | it passes, and the silence survives to the consumer |

**So one arm is caught and one is not, and the strength is a property of the
bound body rather than of anything structural here.** Recorded because the first
row looks like coverage to anyone reading the outcome alone, and it is a right
answer for the wrong reason.

#### A body has exactly two routes to its inputs, and the store is not one

`inputs.json` and `materials.json`, both in the body's `cwd`. **There is no
third.** `env_mgr` measured a confined body granted its zone and its inputs'
`content/` (`scratch/impl-2026-08/env_mgr/p11_can_a_body_reach_the_store_root.py`):
the staged copy opens, the store root listing is `EACCES`, another version's
manifest is `EACCES`, against an unconfined control where all four succeed.
Nothing grants the store root.

**This retracts a plan stated in the same breath as the section above.** When
`env_mgr` narrowed `stage()` to `content/` I said the general-spec bodies would
ask the store for a manifest when they stopped being placeholders, on the
strength of `store.get_manifest` verifying a digest where a staged copy does not.
That is true of this package's **own process** — `history.py`'s prior-verdict
path is unaffected — and false of a **body**, which is confined and cannot reach
the store at all. A body that wants a prior verdict or a manifest cannot fetch
one; it must be handed it.

The placeholders would never have found this, because they read `inputs.json` and
stop. That is the shape of what "my evidence about bodies is weak" was hiding.
