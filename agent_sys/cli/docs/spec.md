# Demo — Specification

| | |
|---|---|
| Status | Draft, revised after review |
| Revision | 7 — 2026-08-29. **The package format is YAML.** §1.1 and criterion 16 said *jsonnet*; the user-interface stage deleted it (`docs/ui-stage.md`). Criterion 16's substance is unchanged — the ordinary task-package path, no privileged import, no schema of its own — and only the format word moves. Criterion 7's *"an agent of `kind: program`"* now describes the two program **leaves** only: a non-leaf declares no agent at all (`closure.schema.json` requires one of a leaf and of nothing else), so the hand-written `compose` spec is deleted rather than converted. (rev. 6: 2026-08-27. Every task has an agent; the program node's agent is `kind: program` rather than absent (§2 item 6, criterion 7). (rev. 5: 2026-08-26. The isolation properties are CI-enforced in tests/env_mgr; the demo additionally shows them against a real agent (§5). (rev. 4: The demo is a task package — the first, and the only one in this repository (§1.1). rev. 3: The e2e test is a separate artefact with its own tasks, not this demo under CI (§5). rev. 2: Review of PR #132: lives in `examples/`; a whole-system CLI is separate and larger; credentials come from config via `env_mgr`; cost is not a constraint. rev. 1: initial)) |
| Date | 2026-08-24 |
| Scope | The runnable proof that the components compose |
| Source | The task definition, goal 2 |
| Part of | [`../../docs/spec.md`](../../docs/spec.md) — the whole-system specification |

---

## 1. Purpose

A command-line demo that runs a task graph end to end. **Simple and fast, but it
must exhibit a task graph and a subgraph** — that pairing is the requirement, and
it is the interesting one: a flat graph would prove less than half of what needs
proving.

The demo is not a tutorial and not a benchmark. It is the **falsifiable claim**
that the seven components fit together, and it is the first thing to break when
one of them drifts.

### 1.1 The demo is a task package, and the only one in this repository

**The demo is a task package** (main spec §4.3) — a self-contained directory of
YAML specs, exactly like the one a real workflow ships. Its handoff kinds, its
validators, its closures are its own; nothing in `agent_sys` imports any of them.

That used to be a discipline this document had to state and defend. Now it is
structural: a task package is outside the system by construction, and the demo is
one.

**It is the only task package that lives in this repository**, and the exception
is deliberate.

There is one qualification, added at rev. 7. `examples/demo-broken/` is a second
directory that is structurally a package, and it exists only so that criterion
11's *"a deliberately broken closure makes it fail with the offending file
path"* can be shown without colliding with criterion 13's *no hand-editing*. It
was a subdirectory of this package for the same purpose; the YAML front end
scans every `*.yaml` under a root, so it could not stay one. It holds one
document that is not intended to load, is reached only by `--with-broken`, and
is part of the demo rather than a second example. The demo is the system's falsifiable claim, and the first thing a
reviewer runs — putting it elsewhere would mean "clone two things to see whether
this works". Every other workflow's package lives outside.

The exception is bounded by one rule: **the demo may use nothing a task package
outside the repository could not use.** It has no privileged import, no private
loader path, no schema of its own. If it needs something the system does not
offer to everyone, that is a missing feature, not a demo detail.

### 1.2 In scope

- What the demo must demonstrate, and why each item is on the list.
- What it must not require.
- Its CLI surface and observable output.

### 1.3 Out of scope

**The whole-system CLI.** A CLI that takes a global task, a config YAML, and some
options, and runs the entire system, is a separate and considerably larger piece
of work — recorded in [`../../docs/TODO.md`](../../docs/TODO.md). The demo has its
own entry point and does not wait for it.

**The reference workflow.** The six-step optimisation loop needs GPUs, a cluster,
and hours. The demo proves the machinery; the workflow proves the value.

---

## 2. What it must demonstrate

Eight things. Each is a seam that could silently not work.

| # | Demonstrates | Why this one |
|---|---|---|
| 1 | **A root task expanding into a subgraph** | A flat graph exercises none of the nesting |
| 2 | **All three phases**, with input validation empty in one task and populated in another | Empty is the normal case and must be shown to be normal, not degenerate |
| 3 | **The validations are invisible to the scheduler** | One dispatch, three phases, no validator in a pool (`task_graph` spec §3.2.1) |
| 4 | **At least one validator actually gating a handoff** | A validator that runs but blocks nothing proves nothing. One handoff left with a failing verdict, one consumer stuck in `WAITING_HANDOFF` because of it |
| 5 | **One `claude-agent-sdk` agent node** | The backend abstraction against a real backend, not a fake |
| 6 | **One program-executed node** | The other half: the executor is genuinely swappable, and an agent need not be an AI |
| 7 | **OS-level isolation actually confining** | An agent attempts a write outside its zone **via a script**, and it fails. The hook alone would not catch this — `env_mgr` spec §4 |
| 8 | **A resume** | Interrupt the run, restart, and the graph continues from persisted state |

Items 4 and 7 are the ones a demo is tempted to skip, because both are failures
and a demo that fails looks broken. They are also the two things the system
exists to provide, so both run on the happy path and both are reported as
expected outcomes.

### 2.1 A shape that satisfies the list

Illustrative, not mandated.

```
system whole task
└── main
    ├── produce            program node — an agent of `kind: program`, no AI
    │   └── output validation
    │       └── check_shape          strong, completeness. PASSES
    ├── describe           claude-agent-sdk node
    │   │                  reads the artefact, writes a summary handoff
    │   │                  also attempts one out-of-zone script write → blocked
    │   └── output validation
    │       └── check_summary        FAILS deliberately
    └── consume            blocked: its input never became valid
```

`produce` is the start entry subtask; `consume` is the end entry subtask and
never runs. The run ends quiescent with one task still waiting — the correct
outcome, and the demo says so.

---

## 3. What it must not require

| | |
|---|---|
| **No GPU** | Nothing in the machinery needs one, and a reviewer must be able to run this |
| **No cluster, no remote host** | The local↔remote mapping degenerates to a strong same-machine mapping, which is a legitimate configuration |
| **No long run** | **Target: under one minute** of wall clock, excluding model latency |
| **No hand-editing to run twice** | Idempotent, or it cleans up after itself |
| **No manual credential setup** | §3.1 |

**Cost is not a concern.** The demo makes a real model call and is not shaped
around minimising it.

### 3.1 Credentials come from config

The API key and endpoint are supplied **in the config file**, and `env_mgr` has a
submodule that sets up the Claude Code SDK from them
([`../../docs/TODO.md`](../../docs/TODO.md)). A reviewer supplies config, not a
setup procedure.

With credentials absent, the demo **fails with a clear message** naming what is
missing — not a stack trace, and **not** a silent fallback to a fake agent, which
would make the run prove less than it appears to.

### 3.2 Isolation is required, not optional

Criterion 7 needs a working sandbox. On a machine with neither `bwrap` nor
Landlock the demo **refuses to start**, the same way any task does (`env_mgr`
spec §4.2) — and that refusal is itself correct behaviour, not a demo failure.

---

## 4. CLI surface and output

### 4.1 Invocation

One entry point, runnable after `pip install -e agent_sys`:

| | |
|---|---|
| **run** | Runs the graph |
| **show** | Prints the graph without running: closures, tasks, phases, validators |
| `--dry-run` on run | Resolves and validates everything, dispatches nothing — proving the load-time checks fire |

### 4.2 Observable output

The output *is* the deliverable — a demo whose result must be inferred from an
exit code proves nothing. It must show, legibly:

- **The graph as loaded**: tasks, parents, phases, handoffs.
- **Each dispatch**, in order, with the task and its bound agent — and visibly
  **one dispatch per task**, not one per validation.
- **Each handoff transition**, with the version and the verdict.
- **The validator verdicts**, each labelled with its dimension and strength — so
  the demo teaches the taxonomy by using it.
- **The isolation block**, as an expected event, naming the mechanism that caught
  it.
- **The final state of every task**, including the one still in
  `WAITING_HANDOFF`, and why.

Machine-readable output alongside the human-readable form, because the acceptance
criteria are assertions over it.

---

## 5. Where it lives, and its relationship to the tests

**`examples/demo/`.** It is an example — the first thing someone reads to
understand the system, and something a person runs by hand. `examples/` is where
a task package sits when it lives in this repository, and the demo is the only
one that does (§1.1).

**The demo is not a test, and CI does not run it.** Tests live in `tests/` and
are their own thing: unit tests for each component, plus an **end-to-end
whole-graph test** that may take a shape similar to the demo's. Two artefacts,
similar shape, different jobs:

| | The demo (`examples/`) | The e2e test (`tests/`) |
|---|---|---|
| Audience | A person reading or evaluating the system | CI |
| Run by | Hand | The test runner |
| Output | Legible narration (§4.2) | Assertions |
| Credentials | Required — it makes a real model call | Whatever its own tasks need; not this demo's |
| Proves | The components compose, against a real backend | The components compose, deterministically and on every commit |

**The e2e test is not this demo, run under CI.** It is written separately, at the
system-level implementation stage, modelled on this demo's *shape* and choosing
its own tasks. Nothing here constrains what those tasks are or what runner they
use.

**The isolation properties do not wait for any of this.** `env_mgr` criteria
2–14 — the scripted bypass, the three prefix defeats, inheritance across `exec`,
fail-closed — are CI-enforced in `tests/env_mgr` on every commit
(`env_mgr` spec §10). They need a subprocess and a filesystem, not a model. What
the demo adds is showing the same block happening to a **real agent**, in a run
a person watches; the property itself is already guarded.

That split dissolves the apparent problem with §3.1: the demo failing without
credentials is correct *because* CI is not the thing running it. A test that
needed an API key would be a bad test for unrelated reasons — non-deterministic,
costly, and failing on a fork.

When the system-level implementation stage arrives, the e2e test is written
there. This document specifies only the demo.

---

## 6. Acceptance criteria

1. `pip install -e agent_sys` then the run verb completes in **under one minute**
   excluding model latency, with no GPU, no cluster, and no manual setup.
2. The loaded graph has a root task with `parent = None` and at least one subtask
   with a non-`None` parent.
3. Some task's input validation phase is empty and runs nothing; another's is
   populated and runs.
4. **One dispatch per task.** A task with both validation phases populated is
   dispatched once, and no validator appears in any pool.
5. **A validator records a failing verdict, and the consumer remains in
   `WAITING_HANDOFF`** at the end of the run — reported as the expected outcome,
   not as a crash. The verdict is written by the validation phase, never by the
   agent that produced the content (`task_graph` spec §3.1).
6. **With no API credentials the run fails with a clear message** naming what is
   missing. It does not fall back to a fake agent.
7. **One node runs on `claude-agent-sdk` and one runs as a program** — an agent
   of `kind: program`, with no AI in it — and the handoff state each produces is
   indistinguishable in kind.
8. **A scripted out-of-zone write is blocked**, the agent is told why, and the
   block appears in the output as an expected event naming the mechanism.
9. On a machine with no sandbox available, the demo refuses to start and says so.
10. Every validator verdict in the output carries its dimension and strength.
11. `--dry-run` resolves every closure, validates every spec, and dispatches
    nothing — and a deliberately broken closure makes it fail with the offending
    file path.
12. Interrupting the run and restarting continues from persisted state.
13. Running twice in succession succeeds without hand-editing anything.
14. The machine-readable output is sufficient to assert criteria 2–10 without
    parsing prose.
15. **The components import nothing from the demo** (§1.1).
16. **The demo loads through the ordinary task-package path**: its specs are
    YAML, discriminated and schema-validated like any package's, with no
    privileged import and no schema of its own (§1.1, main spec §4.3).
17. **Every filename is found by convention and nothing is bound by hand.** No
    document in the package declares a `body` path; every readme and every
    `entry.sh` is resolved from `assets/` against the object's `name` and `type`,
    and the loader reports no `explicit-binding` warning. This is the criterion
    that makes §1's *"best-practice reference"* checkable rather than a claim —
    an explicit binding is legal, so nothing else would fail if the package drifted
    into hand-binding.

---

## 7. Open questions

| Item | Status |
|---|---|
| **What the demo's task actually does** | §2.1 is illustrative, and it is a *package-content* question rather than a system one (§1.1) — the demo package can change its task without any spec here changing. Wanted: something small, verifiable, and not contrived, so a validator has something to be honestly `strong` about |
| **What tasks the e2e test picks** | §5 settles that it is a separate artefact modelled on this demo's shape, choosing its own tasks. Which ones is a system-level implementation-stage decision, not this document's |
| **Which model, and which tools** | Cost is not a constraint, but the SDK's default is Claude Code's full tool set and system prompt. What the demo agent should actually be given is a design-stage choice |
