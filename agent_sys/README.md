# agent_sys

Task-management substrate for Infera's agent-driven performance-optimization
loop.

An AI agent is treated as a function that is not very procedural. A handoff is
that function's input or output. This system decides **which task runs when**,
and nothing else — it never inspects what a task does.

## Documents

| | |
|---|---|
| [`docs/spec.md`](docs/spec.md) | What the system must do. 25 acceptance criteria |
| [`docs/design.md`](docs/design.md) | How it is built: files, classes, method bodies, test plan |

Read the spec first. The design implements it and records, in its §13, every
place where implementing it literally did not work.

## Status

Design stage. No code yet.

## Dependencies

**None.** Python ≥ 3.10 standard library, plus `pytest` for the tests, which the
repository already carries as a dev dependency.

`mission.md` rule 3 requires researching whether a mature solution exists before
building, and recording the outcome. It does, for parts of this; the table below
is that record. `docs/design.md` §10 carries the same table with the full
reasoning, and `docs/spec.md` §9 records the platform-level rejections that came
out of the prior-art survey.

| Module | Considered | Chosen | Why |
|---|---|---|---|
| `core` | pydantic, msgspec, attrs | `dataclasses` + `enum` | Internal records, not a wire format — no validation or coercion needed. `msgspec` is already an `infera` dependency and is the drop-in if serialisation ever shows up in a profile. |
| `registry` | dependency-injector, pluggy, punq | `dict` | All three are built around constructor injection; the spec requires resolve-at-use-time. What is left is a name→instance map: nine lines. |
| `store` | sqlite3, shelve, tinydb, diskcache | `json` + `pathlib` | Records stay readable with `cat` while the schema is still moving. `Path.replace` gives per-record atomicity. **sqlite3 is the named upgrade path** — stdlib, and it would supply the cross-manager transaction the spec leaves open. `StoreMgr` is a Protocol so the swap is one file. |
| `handoff` | content-addressed stores (git, DVC, S3) | own | Versioning here is metadata bookkeeping. Where payloads live is deliberately open (spec §8.2); a content store plugs in behind `HandoffVersion.content`. |
| `resource` | `threading.Semaphore`, Prefect concurrency limits | own | A semaphore cannot express reserve-then-settle for consumables, nor all-or-nothing multi-pool acquisition. Prefect's limits do exactly the right thing but live server-side — adopting a server to obtain one primitive. |
| `runner` | Claude Code / Codex / Cursor CLIs, subprocess | Protocol + a fake | The real implementations are harness-specific and out of scope. What this system owes is the seam. |
| `policy` | graphlib, networkx, OR-Tools | `sorted()` | No graph algorithm is required — the only graph operation is asking whether a task's inputs are valid. `graphlib.TopologicalSorter` additionally cannot accept nodes after `prepare()`, and this graph grows at runtime. |
| `scheduler` | Prefect, Hatchet, Temporal, Ray, Airflow, Slurm | own | Every one is a platform whose scheduling core is not separable. See spec §9. |

The short version: every candidate is either a platform (adopt the server to get
the primitive) or a library for a problem this system does not have — graph
traversal, dependency injection, schema validation. The two genuine upgrade
paths, `sqlite3` for the store and `msgspec` for serialisation, both sit behind
an interface that already exists.

Adopted from the prior-art survey as *design* rather than as a dependency:
RCPSP terminology and its two waiting sets, the parallel schedule generation
scheme, the A2A task-state vocabulary, reserve-then-settle for consumable pools,
and "the engine owns routing".
