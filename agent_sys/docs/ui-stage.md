# The user-interface stage — plan

| | |
|---|---|
| Status | **The plan.** Rulings in `../CLAUDE.md` are settled; this says who builds what |
| Date | 2026-08-29 |
| Source | `../../refine.task_package.define.md` — the user's request, verbatim |
| Binding | `../engineer_principle.md`; `interfaces.md` §1.1 on seams |

---

## 0. The goal — added by the user, 2026-08-29, mid-stage

> **`examples/demo` converted to the new format and running end to end.**

This is the deliverable. Everything else in this plan is instrumental to it, and
the waves are re-ordered by that: a wave is finished when the demo can use what
it built, not when its own tests pass.

**The acceptance test is a run, not a suite.** `agent-sys` executing the
converted package to completion is what "done" means for this stage. A green
`pytest agent_sys` is necessary and is not sufficient — the demo has criteria of
its own (`cli/docs/spec.md`) and they must still be met by the converted
package, not weakened to make the conversion easier.

**W6 is therefore not the tail.** It is the goal, and W3 / W4 / W5 are judged by
whether the demo can be written against them. Anyone who finds their design does
not survive contact with the demo should say so rather than shipping it and
leaving the collision for W6.

---

## 1. The target, on one page

```
my_package/
├── main.yaml                 MANDATORY. the outermost graph's entry
├── <anything>/**/*.yaml      scanned. an object may live in any of them
└── assets/                   MANDATORY
    ├── collect.readme.md     found by convention against `name` (+ optional `type`)
    ├── collect.entry.sh
    └── check_facts.validator/    a folder, likewise
```

Every object carries a discriminator:

```yaml
module: task            # task | agent | handoff | validator
name: collect
...
```

Four user-facing kinds. **`closure` is not one of them** — a `module: task`
document produces the closure *and* the task spec internally, which is what
`closure/check.py:709` already does by splitting one out of the other.

A file may hold a **list** of objects. An object may be defined **inline** where
it is referenced. **Definition order is load-bearing**: a name referenced before
it is defined is an error.

---

## 2. The seam change — approved, and both sides named

`interfaces.md` §1.1 requires naming both sides. They are **`spec_loader`**
(owner) and **`task_graph/bootstrap.py`**'s `packages=` parameter, plus `demo`.

### Why it has to change

`SpecSource(path, kind)` is "one file, one object, kind claimed by location".
Three of the five requirements break it: several objects per file, kind claimed
by a key rather than a directory, and inline definitions that have no file of
their own.

### The shape

```python
@dataclass(frozen=True)
class SpecDocument:
    kind: str                     # handoff | validator | agent | closure
    doc: Mapping[str, Any]        # parsed, ready to validate
    origin: str                   # "steps/collect.yaml#/2" — path + JSON pointer
    line: int | None = None       # 1-based, for a diagnostic that points at source
    column: int | None = None


class TaskPackage(Protocol):
    root: Path
    def documents(self) -> Sequence[SpecDocument]: ...
```

`load_package` becomes **validate + admit**. It no longer renders, no longer
opens a file, and no longer knows a source format exists.

**This makes main spec §4.4's central promise structural rather than
conventional.** §4.4 says the loader "does not read, audit, or constrain a
package's" source and that "only the result is checked". Today that is an
ordering convention inside `load_package`; after this it is a type boundary.

### What is deleted

`spec_loader/render.py` entirely — `render`, `Runtime`, `select_runtime`,
`available_runtimes`, `FileImportResolver`, `DEFAULTS`, and the `ImportResolver`
Protocol. The `jsonnet` and `rjsonnet` dependencies. `$AGENT_SYS_JSONNET`.

### What must not change

`validate`'s path-free signature (main spec criterion 4 rests on it),
`Problem`, `LoadReport`, `BaseSpecRegistry`'s collision policy, `schema_for`.

---

## 3. Waves and ownership

One owner per file. Two agents must never hold the same file.

| Wave | Owner | Owns these paths | Blocked by |
|---|---|---|---|
| **W1** | `spec-author` | `docs/spec.md`, `closure/docs/spec.md`, `spec_loader/schemas/*.json` | — |
| **W2** | `env-paths` | `env_mgr/**`, `env_mgr/README.md` | — |
| **W3** | `yaml-loader` | `spec_loader/**` (except schemas), `tests/spec_loader/**` | W1's schemas |
| **W4** | `assets` | a new resolver module, `tests/` for it | W1's schemas |
| **W5** | `graph-edges` | `task_graph/**`, `closure/**` | W1's schemas |
| **W6** | `demo-author` | `examples/demo/**`, `cli/**`, `tests/cli/**` | W3, W4, W5 |
| **W7** | all | the remaining 34 test files that mention jsonnet | W3 |

W1 and W2 are independent and start together. W3–W5 start when W1's schemas
land. W6 is last because it consumes all of them.

---

## 4. What each wave must deliver

### W1 — the specification

Amend, do not work around:

- **`docs/spec.md` §4.4** — retitle and rewrite the pipeline as `YAML → schema`.
  The `config` fill's story changes: with no computation there is no `extVar`.
  State what replaces it (a package-level variable set plus schema `default`),
  and keep the "the schema is the only enforcement point" half intact.
- **`docs/spec.md` §4.3** — the table's "the jsonnet sources, organised however
  the package likes" becomes YAML, and must now admit **two mandatory names**:
  `main.yaml` and `assets/`. That narrows "does not interpret a package's
  layout". Say so explicitly rather than letting the two sentences disagree.
- **`docs/spec.md` §7** — the adopt/reject table. jsonnet moves to rejected with
  the measurement that killed it (its power is unused). Kustomize's row gets the
  real reason: Go-only API, no Python binding. Record `ruamel.yaml` as adopted.
- **`docs/spec.md` §4.8** — "Every task has exactly one agent … always, and the
  load checker demands one" becomes leaf-only.
- **`closure/docs/spec.md`** — rev. header, §2 key table, §2.2 (whole section),
  §2.3's `agent_of` "Never `None`", criterion 3.
- **`closure.schema.json`** — `agent` leaves `required`; a conditional makes it
  required iff there is no `task.subgraph`. Check whether the schemas already
  use `if`/`then` before introducing the construct; if not, say so and propose.
- **`task.schema.json`** — the subgraph item gains **`froms`**, required.

### W2 — the path environment-variable system

The user's list, with "romote" read as "remote":

```
task_package_root            agent_workspace_root         agent_handoff_root
agent_playground_root        agent_workspace_root_remote  agent_handoff_root_remote
agent_playground_root_remote my_agent_workspace           my_agent_playground
my_agent_workspace_remote    ...
```

Known before you start:

- `AGENT_SYS_TASK_PACKAGE` already exists and is `task_package_root`.
- The insertion point is `env_mgr/prepare.py:424-447`. It has **five** sources
  and `Prepared.environment` is a read-only `MappingProxyType` (line 479).
  Adding a sixth is a real change — decide whether it is a sixth source or
  folds into an existing one, and **say which and why**.
- Remote exists: `env_mgr/remote/{connection,tools}.py`, ssh / docker exec /
  local behind one Protocol, with a `RemoteMapping` that `meta` persists.
- `${TASK_PACKAGE_ASSERT_DIR}` is the user's token for the assets directory.
  It does not exist yet. **Keep the user's spelling** — it is the interface they
  asked for, and renaming it is not this wave's call to make.

Open, and yours to answer with evidence: **is there a playground today?** If
not, say it does not exist rather than describing what it would be. Same for
each `my_*` slot — name the identifier that indexes it, its generation site, and
whether it is filesystem-safe.

### W3 — the YAML front end

A `YamlPackage` implementing the new `TaskPackage`. Its pipeline:

1. **Scan** every `*.yaml` under the root, excluding `assets/`.
2. **Parse** with `ruamel.yaml` in round-trip mode so `lc.line` / `lc.key(k)` /
   `lc.value(k)` / `lc.item(i)` survive into diagnostics. Syntax errors come
   back as `MarkedYAMLError.problem_mark`. Both are 0-based — add 1.
3. **Discriminate** on `module:`. An unknown or absent value is an error naming
   the file and the line.
4. **Expand inline definitions** — an object written where it is referenced is
   registered under its own name and replaced by that name.
5. **Substitute variables**. The whole measured need is: shared constants, path
   concatenation, and default-if-absent. Do not build more than that.
6. **Order-check**. A reference to a name defined later is an error. This lives
   here, not at the composition root, because the package owns its own file
   order and nothing else can see it.
7. **Emit** `SpecDocument`s in definition order.

`main.yaml` must exist and must hold the outermost graph; its absence is an
error naming the root.

**Library first.** Before writing any of steps 2-6, research what exists and
record the choice in `spec_loader/README.md`. `ruamel.yaml` is the leading
candidate for step 2 and is not yet a dependency. For step 5, look at whether a
config-merge library earns its place or whether this is twenty lines.

### W4 — assets auto-discovery

The convention, from the user:

- readme: `${name}.readme.md`, `readme.${name}.md`, `${name}.md`,
  `${type}.${name}.readme.md`, `${name}.${type}.readme.md`, `${name}.${type}.md`,
  `readme.${name}.${type}.md`, and the rest of the permutations. `type` optional,
  `name` mandatory, `.md` mandatory, the literal `readme` optional.
- entry: `${name}.entry.sh`, same permutation rules.
- folders: `${name}.${type}`, `${name}`, `${type}.${name}`.
- Recursive under `assets/`. **A conflict crashes** — match the existing
  collision policy in `spec_loader/registry.py` rather than inventing an error
  shape.
- A YAML field may be omitted and derived, or bound explicitly — **an explicit
  binding warns**.

Derive the *paths*, never the semantics: `entry.sh` present must not silently
reinterpret what a task is. That rule is `user_interface.ai.draft.md` §4.9's one
survivor and it came from a real failure (opa#6509).

### W5 — `froms` and topological order

Ruling: **option (a)** — `froms` is mandatory on every subgraph entry and is
**cross-checked** against the edges derived from handoff wiring. A mismatch is
an error naming both.

- `task_graph/models.py:560-569` derives `depends_on` today. Keep the
  derivation; it becomes the thing `froms` is checked against.
- `task_graph/scheduler.py:639`'s `_warn_depends_on` becomes reachable
  differently — read its docstring before touching it, it explains why it warns
  rather than rejects.
- **Listing order must be a valid topological order**, and that is ours, not
  adopted: Argo resolves by name and imposes no order. Reject a violation at
  load, naming the entry and the edge that goes backwards.
- `graphlib.TopologicalSorter` is stdlib. Check whether its `CycleError` names
  the cycle usefully before adopting or rejecting it; main spec §7 previously
  rejected it for a different reason (it refuses nodes after `prepare()`), and
  that reason may not apply to a load-time check over a static list.

### W6 — the demo as best practice

Every filename found by convention, nothing bound by hand. The non-leaf `main`
loses its `agent`. The demo must still prove what it proves today — read
`cli/docs/spec.md`'s criteria before moving anything, and do not weaken one to
make the rewrite easier.

### W7 — the tests

**34 test files mention jsonnet.** Split them: those that pin the *format* get
rewritten; those that pin *loader behaviour* should pass unchanged, and one that
does not is a finding worth reporting rather than editing away.

---

## 5. How the team works

- **One owner per file.** If you need a file you do not own, say so and stop.
- **Report a seam, do not change it.** Both sides, by name.
- **Evidence discipline is binding** — see `../CLAUDE.md`. Cite `file:line`.
  "I did not check" is a complete answer; a confident guess is not.
- **Probes stay** in `scratch/ui-yaml-2026-08/<wave>/`. They are the evidence.
- Commit with `git commit -s -F - -- <paths>`, explicit paths only. The worktree
  is shared with eight others.
