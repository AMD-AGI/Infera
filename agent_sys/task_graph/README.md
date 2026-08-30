# task_graph

Task-management substrate for Infera's agent-driven performance-optimization
loop.

An AI agent is treated as a function that is not very procedural. A handoff is
that function's input or output. This system decides **which task runs when**,
and nothing else — it never inspects what a task does.

## Documents

| | |
|---|---|
| [`docs/spec.md`](docs/spec.md) | What the system must do. 54 acceptance criteria (rev. 14) |
| [`docs/design.md`](docs/design.md) | How it is built: files, classes, interfaces, test plan |

Read the spec first. The design implements it and records, in its §13, every
place where implementing it literally did not work.

`task_graph` is one component of the wider agent work system; the whole-system
specification is [`../docs/spec.md`](../docs/spec.md).

## Layout

`task_graph` is one of the components under `agent_sys/`, alongside `env_mgr`;
both are declared by `agent_sys/pyproject.toml`.

```
agent_sys/
├── pyproject.toml       declares env_mgr and task_graph
├── docs/spec.md         the whole-system specification
├── env_mgr/             the sibling component
├── task_graph/          this package
│   ├── docs/            spec.md, design.md
│   └── *.py
└── tests/
    ├── env_mgr/
    └── task_graph/
```

## Status

**All 54 acceptance criteria are implemented and covered**, each by a named test
`docs/design.md` §11 maps it to. Criteria 1–35 were spec rev. 7; 36–54 are
subgraph nesting and the two validation phases (rev. 8–9), task-owned
transitions and cascading cancel (rev. 10), leaf-only resource acquisition
(rev. 11), and the three rev. 12 corrections — `Grant.kind`, the cascade reason
in the report, and `OrderedIdSet` pools rebuilt on resume.

**The debt was paid in one landing, deliberately.** `Model` sets
`extra="forbid"` with `validate_assignment=True`, so assigning an undeclared
field raises and a record written by new code is unreadable by old code — the
seven `Task` fields could not arrive one at a time. The cost was a long red
window in the shared suite; the alternative was five green intermediate states
each needing to be redone.

**Everything after that came from other packages assembling against it**, and
that is where the value was: the composition root gained a `KindSource`, the
`agent_specs → agent_mgr` bridge, the knowledge pass, non-fatal problem
reporting and `set_task` at dispatch — every one of them a seam that was
declared, agreed, and had never been traversed.

```bash
pip install -e agent_sys      # once
pytest agent_sys/tests/task_graph
```

Three things a reader should know before changing anything here:

- **`Model` sets `extra="forbid"` and `validate_assignment=True`.** Adding a
  field is a real change, and a record written by new code is unreadable by old
  code. There is no rollback across a `Task` field addition.
- **Dispatch lands a task in `INPUT_VALIDATING`, not `RUNNING`.** The runner
  advances it through the three phases on one lease. A test that means "this was
  dispatched" says `DISPATCHED`, which `tests/task_graph/conftest.py` defines.
- **The composition root defers every *sibling* import to call time.**
  `interfaces.md` §2 has `build_registry` construct six other packages' types
  while §4.7 permits module scope pydantic, `spec_loader` and this package; a
  function-local guarded import is what satisfies both, and
  `test_bootstrap.py::test_importing_task_graph_reaches_no_sibling_package`
  runs a fresh interpreter to keep it that way. **`spec_loader` is not one of
  the deferred six** — it is a module-scope import in `models.py`, `graph.py`
  and `bootstrap.py`, and it is the only package of ours that any of them names.

## What this package owes other packages

The column nobody had a home for. **Each line was checked against the tree
before being written here**, because on a tree eight people are writing, a
status note is stale on arrival — two entries that were on this list an hour
ago had already been closed by somebody else.

| Owed | State |
|---|---|
| Remove `build_registry`'s default construction of the five spec registries, and the four guarded sibling imports with it | **Waiting on `demo`** to pass `registries=`. Additive until then; omitting it builds exactly what it built before |
| `monitor_for` is registered under a name that is in no table | **Open.** `interfaces.md` §2.7 names it in prose; §2.1's registered-name table does not have a row. Reported |
| `check_knowledge`'s strict mode | **Open, and not mine to close.** The pass runs in its default (warn) mode. `mandatory` is a run-config flag, and `strict_level`, `config_order`, `handoff_root` and `knowledge_root` are the same question asked four times — a run-config object is the shape, and that is a contract decision |
| The `subgraph` key and its entry shape | **A documented convention, not a specification.** No spec fixes either; `spec_loader`'s schema now validates the shape this package defined, which gives the key one reader without promoting the convention to a rule |
| `SUBGRAPH_AGENT_SPEC` — a name in `AgentMgr`'s vocabulary that no package authored | **New, and `docs/interfaces.md` has no row for it.** Main spec §4.8 narrowed to leaf-only, so a non-leaf carries no `agent`; `submit` still gates every task on `is_registered` and `_dispatch_pass` still feeds `instantiate(...).id` into a required `Execution.agent_id`, so the field cannot be `None` without reaching the execution record — the shape §5.13 already has open for `Verdict.agent_id`. `build_registry` registers the name; nothing puts it in `agent_specs`, and the only reader that would resolve it there (`runner.agent_spec_of`) is unreachable for a non-leaf. **Observable**: `cli/main.py` prints `agent=task.agent_spec` for every task, so a run report now says `subgraph` where it used to say the author's agent — which never ran. Reported, not filed |
| ~~`Execution.output_versions` for an output write grant~~ | **BUILT — `interfaces.md` §4.14, and this row was wrong about why it could not be.** It said the scheduler cannot pre-fill because *"allocating a version is `Handoff.open_next`'s"*. That named the wrong allocator: `env_mgr`'s grant resolves `<store>/<hid>/v<N>/`, so the number it wants is the **store** directory version, and `handoff.HandoffStore.allocate` reserves it without touching a slot. Criterion 14 is about `HandoffMgr` — no transition, no `persist` from a scheduler frame — and `Scheduler._pin_outputs` does neither, on the same standing the already-sanctioned `declare` has. §5.12's two allocators are the whole content of the mistake: the row assumed there was one |

### `Execution.input_versions` is slot-space, and `env_mgr` resolves it as a store path

**An open defect in this package, waiting on a three-package ruling.** It has a
section rather than a row because the measurement is the part that will be
argued with, and it is not one line.

`env_mgr/grants.py:49-52` merges both version maps and resolves **both** through
`handoff_version_dir` — a store path. But `scheduler.py:278` fills
`input_versions` from `handoff_mgr.latest(hid).version`, a **slot** number,
while `_pin_outputs` fills `output_versions` from `handoff_store.allocate`, a
**store** number. **Two fields of one `Execution`, in two spaces.** It predates
§4.14; putting a genuine store number beside it is what made it visible.

**The two counters cannot converge, because their reuse rules differ.**
`allocate` takes each number with an `os.mkdir` token, so a burned number stays
burned; `Handoff.open_next` adopts a `CREATED` latest **in place**. Measured
(`scratch/impl-2026-08/task_graph/probe_allocator_reuse.py`) — **one dispatch
that does not write is enough**:

```
attempt 0 dispatched          store_pin=0  slot=0/created
attempt 0 failed, never wrote store_pin=0  slot=0/created
attempt 1 dispatched          store_pin=1  slot=0/created
attempt 1 wrote               store_pin=1  slot=0/valid
store burned [0, 1]; the slot used v0
```

**And the consumer side is silent, which is the part worth keeping**
(`probe_consumer_staging.py`). Every guard reports valid and the body still gets
nothing: `allocate` must create `v<N>/content/` for the ruleset to open it, so
the directory exists and Landlock builds; `layout.stage` skips only an *absent*
`content/`; staging is a `copytree`, not `handoff.copy_out`, so no digest is
checked.

```
consumer dispatched  True   (check_if_latest_valid says the slot is VALID)
input_versions {hid: 0}     producer actually sealed v1
layout.stage(v0) -> mapping returned, dir exists, contents []
layout.stage(v1) -> ['README.md', 'items', 'text.json']
```

**It is silent, but it is diagnosable, and the two are not the same thing.**
Nothing raises — but **an empty staged input is always a hole**, never a
legitimately empty artefact, because `handoff.seal` refuses to publish empty
content (`probe_can_empty_be_sealed.py`):

```
seal on EMPTY content -> "nothing was written to <...>/v0/content. That directory
                          is the agent's grant and it is empty, so this attempt
                          produced no content at all"
after: list_versions=[]  latest=None
```

So a sealed version always has content, and the inference *empty ⟹ never sealed*
holds even though no single component checks it. **The one-line check when a
body reports no input:** compare `Execution.input_versions[hid]` against the
version that has a `manifest.yaml`.

**Why this package will not fix it alone.** That consumer was admitted *because*
`check_if_latest_valid` reported the **slot** VALID. Moving the grant to store
space while the admission gate stays in slot space admits a task on one object's
validity and hands it another — worse than the divergence. **The gate and the
grant have to move together**, which makes it `task_graph` (fills it),
`env_mgr` (resolves it) and `handoff` (owns both counters). §5.12's join,
arriving from the input end.

**A second reason to move them together, and it is stronger — an invariant the
fix must *preserve*, not merely not break.** Today a dispatched task always has
every input pinned, and structurally rather than by luck: `_ready` admits only
on `check_if_latest_valid(hid)`, which needs the handoff to **exist** in
`HandoffMgr`; existence implies `latest(hid)` is non-`None`; and
`input_versions` omits exactly the `None` cases. **The gate's precondition is
strictly stronger than the pin's**, so the two cannot disagree
(`probe_unpinned_input_reachable.py` — every dispatched task, `missing=0`).

`env_mgr` has **three silent skips** waiting on the state where they do
disagree — `layout.py:242`, `layout.py:246`, `grants.py:415` — and `prepare`
then returns a healthy-looking `Prepared` with nothing staged and no
`AGENT_SYS_INPUT_*` exported. **That state is unreachable today and becomes
reachable the moment the pin and the gate are sourced separately.** Measured by
`env-mgr-2` against `prepare()` directly; unreachable through dispatch,
confirmed here.

**Who is waiting: `validator`** — `phase.py:638`'s output branch is the other
slot-space read, and they cannot pick a space while one of ours is inconsistent.
Their path fails loudly; **ours does not.**

**One arm of the silence is caught one step later and one is not**, and the
detail is `validator`'s: `validator/README.md`, *"The input phase is a partial
backstop, and must not be read as coverage"*. **A pointer rather than a summary**
— the outcomes depend on `report.py` and on which validator is bound, neither of
which this package can see, and a copy here would go stale exactly the way two
of this file's own sentences did today.

#### Ruled: store space, and the gate moves with the grant

**`main`'s ruling.** `input_versions` moves to store space, **and the admission
gate moves with it in one change** — a task admitted on one object's validity
and handed another object's directory is worse than the divergence.

**Pinned before anything was built** — `tests/task_graph/test_versioning.py`:
one plain assertion for the allocator asymmetry (correct on both sides, so it
stays an assertion), and `test_a_consumer_is_pinned_to_the_version_its_producer_actually_published`
as `xfail(strict=True)`, which goes red the day the fix lands. Verified with
`--runxfail` that it fails on its own assertion rather than a setup error.

**The ruling named `handoff_store.latest(hid)` as the source for both, and that
half does not survive contact.** It is right for the grant and cannot be the
gate:

| moment | | space |
|---|---|---|
| **before** the gate | `_seal_outputs` → `store.seal` → published | store |
| **after** output validation | `HandoffVersion.seal(VALID\|INVALID)` | slot |

The ordering is deliberate — `agent/runner.py:880`: sealing the slot beside the
store seal *"would make a consumer eligible for an output that nothing has
checked yet."* **Publication is not validity.** A version that failed output
validation is published and has a manifest, so `store.latest` returns it, and
making that the gate inverts criterion 13. Confirmed independently from
`validator`'s side, whose phase sits between the two moments.

**The shape proposed instead: join the two counters on the slot.**
`HandoffVersion` records the store version it corresponds to; the gate stays
`check_if_latest_valid`; `input_versions` is read off the same object that
answered the gate. **One object, both answers** — the ruling's requirement by a
different route. It also explains the defect rather than patching it: the slot's
list **index** was being read as a store number, and the slot has never recorded
one.

**State: every answer is in; not built, because `main` ruled it is not today's
work.** Recorded here so tomorrow starts from the answers rather than the
questions.

| from | answer |
|---|---|
| `handoff` | **No store-side `latest_valid`.** The store holds verdicts and deliberately never folds them: `verdict.py` serialises `result` and nothing reads one to decide anything. The fold — `strength`, `dimension`, an empty phase — is `validator`'s policy, and a store-side answer would reimplement it (§4.4) and be a second writer of a fact whose chain is §4.2's own worked example. `allocate`, `seal`, `latest` are stable and will not move |
| `agent` | **The field is `int \| None`**, and `_seal_model_versions` loops `task.outputs`, so it reaches a hid `_seal_outputs` skipped. Put it on the seal — `seal(status, content=None, store_version=None)` — because a separate setter reopens §1's *must remember to also call*. `None` whenever `INVALID`, and there are **two** ways to get there: refused (the store gave a reason) and skipped (nothing was pinned) |
| `env_mgr` | **A third site**, below. Their `grants.py:58-61` and `handoff_version_dir` do not move. They **declined** a defensive emptiness check in `layout.stage`, and the reason is right: the real discriminator is *unpublished*, not *empty*, and spelling `manifest.yaml` there would make `env_mgr` a second judge of what published means — the same two-answers defect they removed from that function hours earlier |
| `validator` | Holding `phase.py:747` until the **source** settles, not just the space, because the join changes which attribute they read |

**Six sites, not two, and my sweep could only ever have found two of them.**
`env_mgr` grepped rather than recalled. I searched for `.latest` reads feeding a
store path and was right about those; a site that **receives** a number is as
much a fork in the road as one that derives it, and I stated a bounded answer to
an unbounded question.

| site | reads |
|---|---|
| `grants.py:59-60` `_versions` | both maps, merged — feeds `resolve` |
| **`grants.py:356` `output_env`** | `output_versions` — **the second exit, below** |
| `layout.py:241-248` `stage` | the mapping it is handed |
| `layout.py:260` `stage_handoffs` | `input_versions` |
| `prepare.py:482-484` `prepare_validation` | either map, by phase — **and not through `prepare`** |
| `meta.py:106` `from_knowledge` | a version passed as an **argument** |

**`output_env` makes the defect worse than mis-staging.** It turns
`output_versions[hid]` into `AGENT_SYS_OUTPUT_<KIND>` — the path a body is *told
to write to*. A wrong number there does not starve a consumer, it **hands a
producer a wrong place to write**. A consumer reading nothing is recoverable; a
producer writing into someone else's version directory is not.

**`from_knowledge` should be decided by the ruling rather than rediscovered.**
It is store space, called only from tests, and takes its version from a caller
who does not exist yet — the task that produces a knowledge handoff. Whoever
wires it faces the same fork.

**One invariant to pin rather than inherit.** After `agent`'s `b029c80`, a
`VALID` slot implies *this attempt published* — `_seal_model_versions` now
requires membership in `_store_sealed`, which `_seal_outputs` populates only
when the store returned no reason (`runner.py:943`, `:1061`). Positive evidence
rather than absence of a refusal. The read side wants exactly that invariant, so
it should be asserted here rather than assumed from their file.

Two entries came off this list by being checked rather than remembered: the
`_Id = Id` transitional alias (removed — `monitor` had already migrated to the
public `Id`), and the `agent_specs → agent_mgr` bridge (landed; `demo`'s copy
can go whenever, since `AgentMgr.register` overwrites and both running is a
no-op).

## Dependencies

**pydantic v2**, which the repository already installs — `fastapi` pulls it.
Everything else is Python ≥ 3.10 standard library, plus `pytest` for the tests,
already a dev dependency.

The task definition requires researching whether a mature solution exists before
building, and recording the outcome. It does, for parts of this; the table below
is that record. `docs/design.md` §10 carries the same table with the full
reasoning, and `docs/spec.md` §9 records the platform-level rejections that came
out of the prior-art survey.

| Module | Considered | Chosen | Why |
|---|---|---|---|
| `models` | dataclasses, msgspec, attrs | **pydantic v2** | Already installed via `fastapi`, so it costs nothing. `model_dump` / `model_validate` remove the two hand-written deserialisers `dataclasses.asdict` would need — it has no inverse — and which would drift from the models on every field added. `validate_assignment` makes in-place mutation checked, which matters because status is assigned directly. |
| `ids` | bare `str`, `NewType` | `uuid.UUID` subclasses | `NewType` erases at runtime, so two ids of different kinds would still compare equal and collide in one dict. Subclassing gives both static and runtime distinctness. It costs a ten-line `__get_pydantic_core_schema__`: pydantic raises on a `UUID` subclass without one. |
| `registry` | dependency-injector, pluggy, punq | `dict` | All three are built around constructor injection; the spec requires resolve-at-use-time. What is left is a name→instance map: nine lines. |
| `store` | sqlite3, shelve, tinydb, diskcache | `json` + `pathlib` | Records stay readable with `cat` while the schema is still moving. `Path.replace` gives per-record atomicity. **sqlite3 is the named upgrade path** — stdlib, and it would supply the cross-manager transaction the spec leaves open. `StoreMgr` is a Protocol so the swap is one file. |
| `handoff` | content-addressed stores (git, DVC, S3) | own | Versioning here is metadata bookkeeping. Where payloads live is deliberately open (spec §8.2); a content store plugs in behind `Handoff.content`. |
| `resource` | `threading.Semaphore`, Prefect concurrency limits | own | A semaphore cannot express reserve-then-settle for consumables, nor all-or-nothing multi-pool acquisition. Prefect's limits do exactly the right thing but live server-side — adopting a server to obtain one primitive. |
| `runner` | Claude Code / Codex / Cursor CLIs, subprocess | Protocol + a fake | The real implementations are harness-specific and out of scope. What this system owes is the seam. |
| `policy` | graphlib, networkx, OR-Tools | `sorted()` | No graph algorithm is required — the only graph operation is asking whether a task's inputs are valid. `graphlib.TopologicalSorter` additionally cannot accept nodes after `prepare()`, and this graph grows at runtime. |
| `scheduler` | Prefect, Hatchet, Temporal, Ray, Airflow, Slurm | own | Every one is a platform whose scheduling core is not separable. See spec §9. |
| `spec_loader` accessors | own four-line copies of `task_of` / `subgraph_of` | **`spec_loader`'s** | Both were duplicated here because `task_graph` may not import `closure`. They moved to the leaf, which owns the five schema files and therefore owns that the keys exist. `subgraph_of` **split** rather than moved: the leaf hands back entries as written, `subgraph_entries` normalises them into `SubgraphEntry` — the marks mean nothing until an entry is this package's type, and that type is not the leaf's to name. |
| `OnDone` | a `Callable` alias | a **`Protocol`** | `Execution.detail` is "from the runner; for a human" and was empty on every failed task. The field, `on_task_done`'s keyword-only parameter and `close_execution`'s forwarding were all in place — a `Callable[[TaskId, TaskStatus, dict], None]` **cannot express a keyword argument**, so a runner holding an exception had nowhere declared to put it. `Callable[..., None]` would have widened it by giving up the first three as well. The three are positional-only, so an implementation may name them what it likes; `Scheduler.on_task_done` calls its first parameter `tid`. The cost is real: a callback must now tolerate the keyword, and `lambda *a:` does not. |
| `permissions` | a shared type in `env_mgr`, a `TypedDict`, a bare `dict` | two small pydantic models | The type must live where neither package imports the other, which rules out the first. A `dict` would work — this package never interprets it — but `closure`'s load check 6 asks "does this cover that handoff, for reading or writing", and a method on a model is a better home for that question than a convention about keys. |
| `ordered` | `collections.OrderedDict`, `sortedcontainers`, a plain `list` | a wrapper over `dict` | `dict` has preserved insertion order since 3.7 and gives O(1) membership and deletion, the two operations `_move` does most. `OrderedDict` adds a doubly-linked list and `move_to_end` this never needs; a `list` makes `discard` O(n); `sortedcontainers` sorts by a key, which is exactly what an ordered pool exists to stop anyone doing. |
| `graph` | `networkx`, `graphlib` | own | Both were already rejected for scheduling and the rejection holds. The containment check is one pass over declared inputs and outputs grouped by parent — a dict of sets, not a graph algorithm. Importing a graph library would invite modelling the catalogue as a graph object that then has to be kept in sync. **Re-argued for `froms`; see below.** |
| cascade | `graphlib.TopologicalSorter`, a recursive walk | a `deque` | The order required is level-by-level, which is `popleft`. A topological sorter answers a different question and cannot accept nodes after `prepare()`. Recursion produces depth-first order — observably different on any graph that is not a chain — which is why the spec's own "level by level" decides it. |

### `graphlib.TopologicalSorter` for `froms`, re-argued and still rejected

`docs/spec.md` §7 rejected it because it "cannot accept nodes after
`prepare()`", and that reason does **not** apply to `check_graph`: a subgraph's
entry list is static and complete at load. So the rejection was re-derived from
scratch rather than inherited, and two measurements decide it.

**Its `CycleError` is good, and that is not the question.**
`scratch/ui-yaml-2026-08/w5/probe_graphlib_cycleerror.py`:

```
CycleError args : ('nodes are in a cycle', ['b', 'c', 'a', 'b'])
```

The closed walk names every participant, so message quality is not a reason to
reject it. The two reasons that are:

- **The API does not answer the question.** The requirement is *"is **this**
  listing order a valid topological order"*. `TopologicalSorter` offers `add`,
  `prepare`, `get_ready`, `done`, `is_active`, `static_order` — it produces *an*
  order and never judges a candidate one. Adopting it would mean sorting, then
  comparing the result against the listing, and a mismatch would name a
  permutation rather than the edge the author got wrong.
- **After the check, no cycle is representable.** `models.derived_edges` links
  each entry only to an *earlier* producer, so a derived edge points backwards by
  construction; `check_graph` rejects a `froms` that does not. Every remaining
  edge therefore runs low-index to high-index, which is acyclic by definition —
  `CycleError` would be unreachable code, and the graph would have to be rebuilt
  as a `dict[str, set[str]]` to reach it. `tests/task_graph/test_froms.py::test_derived_edges_always_point_backwards`
  pins the half of that which is ours to keep true.

What replaced it is one comparison per declared edge, `index_of[declared] < i`,
which names the entry and the edge in the message because it already holds both.

The short version: pydantic is adopted; for everything else each candidate is
either a platform (adopt the server to get the primitive) or a library for a
problem this system does not have — graph traversal, dependency injection. The
named upgrade path, `sqlite3` for the store, sits behind an interface that
already exists.

Adopted from the prior-art survey as *design* rather than as a dependency:
RCPSP terminology and its two waiting sets, the parallel schedule generation
scheme, the A2A task-state vocabulary, reserve-then-settle for consumable pools,
and "the engine owns routing".

---

## `enter_phase` places a non-leaf's container zone — 2026-08-30

`env_mgr.place_zone` is called from `Task.enter_phase(RUNNING)`, immediately
before the unfold. It used to be called by `agent.TaskAttempt._main`, whose
docstring left the caller open: *"The scheduler at `unfold` and the attempt
before it releases its thread are both plausible … the choice is not this
module's."*

It is this module's, and the reason is that **`enter_phase` submits, and
`Scheduler.submit` dispatches**. Every child is running before `enter_phase`
returns, so the parent's zone was created after its children's — always, for a
root non-leaf as much as for a nested one. What kept it working was a margin, and
the margin is `layout.create`'s `os.walk` of the whole zones tree, which grows
with the run: 11 ms empty, 540 ms over the 1669 directories one full
`examples/demo2/` run accumulated. `grade`, four minutes in, lost both children
to *"declares parent bd890c07, which has no zone"*.

`env_mgr` is **resolved by name**, as `scheduler` and `closures` already are —
no import edge, and a system assembled without it skips the call
(`docs/interfaces.md` §2.4). The verb and its `place_zone`-not-`prepare`
derivation are `env_mgr`'s and are unchanged.

Measurement, probes and the differential transcript:
`scratch/demo2-2026-08/zone-ordering.md`. Regression:
`tests/task_graph/test_subgraph.py::test_a_nested_non_leaf_is_zoned_before_its_subgraph_is_dispatched`.
