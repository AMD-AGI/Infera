# agent_sys — test & code stage

## Task

Implement `agent_sys`, the task-management substrate for Infera's agent-driven
performance-optimization loop. **The design is frozen; this stage writes tests
and code against it.**

- **What to build**: [`agent_sys/docs/design.md`](../agent_sys/docs/design.md) rev. 4 —
  files, classes, interfaces, test plan (§11), implementation order (§12).
- **What "done" means**: [`agent_sys/docs/spec.md`](../agent_sys/docs/spec.md) rev. 6 —
  35 acceptance criteria. Design §11 maps every one to a named test.
- **Where**: `agent_sys/src/agent_sys/`, tests in `agent_sys/tests/`.
  Scratch experiments in `agent_sys/scratch/` (gitignored) and nowhere else.

Run the suite with `pytest agent_sys/tests` from the repository root.

## Method

**Test first, in design §12's order.** Each step green before the next begins:

```
1 ids → 2 models → 3 registry → 4 store → 5 resource
6 handoff → 7 task → 8 agent/runner/policy → 9 bootstrap
10 scheduler(submit+dispatch) → 11 scheduler(lifecycle) → 12 recovery
```

Steps 1–9 are deliberately dull; the design work is all in 10–12. Step 2 carries
more than its size suggests — the state-machine guards live there.

**The design doc is the contract.** If implementing it exposes a contradiction,
record it in design §13 (deviations) or §14 (open questions) — do not silently
paper over it. That is how the last three revisions were produced and it is the
reason the doc is trustworthy.

## Background

`mission.md` (untracked, repository root, deliberately never committed) is the
task definition. Its binding rules:

| | |
|---|---|
| 1 | Everything goes in the `agent_sys` subfolder |
| 2 | Deliver spec → design doc → test & code, pausing for user review at each stage |
| 3 | Research mature solutions before building; record the choice and rationale in a README |
| 5 | Research → plan → sub-workspace → back up and rewrite `CLAUDE.md` → start |
| 6 | Work in English; report to the user in Chinese |

Stages 1 and 2 are reviewed and pushed (`f4a031b`, PR #124). This is stage 3.

## Core principles

Restated from spec §2 because they decide implementation questions daily:

1. **Composition over inheritance.** The one exception is `ResourceMgr`, where
   renewable and consumable genuinely differ in three behaviours.
2. **Resolve at use time, not construction time.** A component holds the
   `Registry` and calls `.get(name)` when it needs a collaborator. No manager
   imports another manager.
3. **Transitions belong to the object, collections belong to the manager.**
   `Handoff.open_next`, `HandoffVersion.seal`, `Task.push_execution` — a manager
   has no `set_status` and no `seal`.
4. **The scheduler decides when, never what.** It never inspects a handoff's
   content and never writes handoff state. `test_authority.py` enforces this
   mechanically.
5. **One writer per invariant.** `Scheduler._move` is the only thing that assigns
   `task.status` or mutates a pool.

## Repository conventions

### DCO sign-off is required on every commit

This repository enforces the [Developer Certificate of Origin](https://developercertificate.org/).
CI blocks any PR containing a commit without a `Signed-off-by:` trailer, so an
unsigned commit is a broken PR, not a style nit.

Commit with `-s`, always:

```bash
git commit -s -m "..."
git commit -s -F -   # when writing a longer message from a heredoc
```

That appends a trailer built from **your own** `user.name` / `user.email`:

```
Signed-off-by: Your Name <you@example.com>
```

**Sign off as yourself.** The DCO is an assertion that *you* have the right to
submit this code, so the trailer must name the person making the commit. Never
copy a colleague's line from an existing commit, and never use a bot or assistant
identity — contributors here sign off under several different addresses, and the
trailer has to match the commit's actual author.

Check what `-s` will produce before your first commit in a fresh clone or
container, where git may have inherited a default from the environment:

```bash
git config user.name && git config user.email
```

If those are empty or wrong, set them (add `--global` outside a container):

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
```

#### Fixing commits that are already missing it

Sign off a range without disturbing commits that already carry the trailer —
rebasing from a point that includes signed commits appends duplicates:

```bash
git rebase --signoff <last-already-signed-commit>
git push --force-with-lease origin <branch>
```

Pick `<last-already-signed-commit>` as the newest commit that already has the
trailer. Check what you are about to touch first:

```bash
git log --format='%h %s | %(trailers:key=Signed-off-by,valueonly)' origin/main..HEAD
```

Cherry-picks do **not** inherit the trailer — `git cherry-pick -s`, or sign off
afterwards. This is the easiest way to reintroduce the problem on a second branch
after fixing it on the first.

### Related trailers

`Signed-off-by` is the DCO assertion and is mandatory. `Co-Authored-By` is
separate, is not a substitute, and some upstreams reject assistant co-author
trailers outright — when contributing to a third-party repository (e.g. ROCm/aiter),
do not add them.

### Style

`ruff` with the repository's settings: line length 100, double quotes,
`target-version = "py310"`. `agent_sys` is not in
`[tool.setuptools.packages.find] include = ["infera*"]` and does not need to be —
`agent_sys/conftest.py` puts `src/` on `sys.path`, which is the two-line editable
install. **Do not edit the repository's `pyproject.toml` for this work.**

## Other notable details

- **Only runtime dependency is pydantic v2**, already installed via `fastapi`.
  Everything else is stdlib plus `pytest`.
- **Python ≥ 3.10.** No `StrEnum` (3.11), no `Self` in annotations without
  `typing_extensions`.
- **`_Id` needs `__get_pydantic_core_schema__`.** pydantic raises
  `PydanticSchemaGenerationError` on a bare `uuid.UUID` subclass. Verified
  against pydantic 2.13; design §3.1 has the working shape.
- Open questions live in spec §10 (system-level) and design §14 (O2, O3, O4, O7,
  O8). Do not close one by implementing it without asking.
