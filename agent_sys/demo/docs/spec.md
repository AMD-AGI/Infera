# Demo — Specification

| | |
|---|---|
| Status | Draft, pending review |
| Revision | 1 — 2026-08-24 |
| Date | 2026-08-24 |
| Scope | The runnable proof that the components compose |
| Source | The task definition, goal 2 |
| Part of | [`../../docs/spec.md`](../../docs/spec.md) — the whole-system specification |

---

## 1. Purpose

A command-line demo that runs a task graph end to end. **Simple and fast, but it
must exhibit a task graph and a subgraph** — that pairing is the task
definition's own requirement, and it is the interesting one: a flat graph would
prove less than half of what needs proving.

The demo is not a tutorial and not a benchmark. It is the **falsifiable claim**
that the seven components in the main spec actually fit together, and it is the
first thing to break when one of them drifts.

### 1.1 In scope

- What the demo must demonstrate, and why each item is on the list.
- What it must not require.
- The CLI surface and the observable output.

### 1.2 Out of scope

- **The reference workflow.** The six-step optimisation loop is what the system
  is *for*, and it needs GPUs, a cluster, and hours. The demo proves the
  machinery; the workflow proves the value. Conflating them would produce a demo
  nobody can run.

---

## 2. What it must demonstrate

Seven things. Each is here because it is a seam that could silently not work.

| # | Demonstrates | Why this one |
|---|---|---|
| 1 | **A root task expanding into a subgraph** | The rev.8 addition (`task_graph` spec §3.2.1). A flat graph exercises none of it |
| 2 | **All three phases**, with `input_validation` empty in at least one task and populated in another | Empty is the normal case and must be shown to be normal, not degenerate |
| 3 | **The start and end entry subtask markers** | The one observable that says a subgraph began and finished |
| 4 | **At least one leaf validator actually gating a handoff** | A validator that runs but blocks nothing proves nothing. One handoff must be sealed `INVALID` and one consumer must stay in `WAITING_HANDOFF` because of it |
| 5 | **One `claude-agent-sdk` agent node** | The backend abstraction against a real backend, not a fake |
| 6 | **One program-executed node** | The other half: the executor is genuinely swappable (main spec criterion 9) |
| 7 | **A permission block** | An agent attempts a write outside its zone, the `PreToolUse` hook blocks it, and the agent is told why (`env_mgr` spec §4.4) |

Items 4 and 7 are the ones a demo is most tempted to skip, because both are
failures and a demo that fails looks broken. They are also the two the system
exists to provide, so both run on the happy path and both are reported as
expected outcomes rather than errors.

### 2.1 A shape that satisfies the list

Illustrative, not mandated — the demo's own graph is a design-stage choice.

```
system whole task
└── main
    ├── produce            program node. Writes a small artefact
    │   └── output_validation
    │       └── check_shape          leaf validator, strong. PASSES
    ├── describe           claude-agent-sdk node. Reads the artefact,
    │   │                  writes a summary handoff
    │   └── output_validation
    │       └── check_summary        leaf validator. FAILS deliberately
    └── consume            blocked: its input never became VALID
```

`produce` is the start entry subtask; `consume` is the end entry subtask and
never runs. The run ends with the graph quiescent and one task still waiting —
which is the correct outcome and the demo says so.

---

## 3. What it must not require

| | |
|---|---|
| **No GPU** | Nothing in the machinery needs one. A demo that needs one cannot be run by a reviewer |
| **No cluster, no remote host** | `env_mgr`'s local↔remote mapping degenerates to a strong same-machine mapping (`env_mgr` spec §4.5), which is a legitimate configuration and not a special case |
| **No long run** | **Target: under one minute** of wall clock, excluding the `claude-agent-sdk` node's model latency. That node is one small call |
| **No network beyond the model call** | The program node and every validator are local |
| **No hand-editing to run twice** | It is idempotent, or it cleans up after itself |

The one unavoidable external dependency is API access for item 5. The demo must
**fail with a clear message** when credentials are absent — not a stack trace,
and not a silent fallback to a fake, which would make the run prove less than it
appears to.

---

## 4. The CLI surface and output

### 4.1 Invocation

A single entry point, runnable after `pip install -e agent_sys`, with:

| | |
|---|---|
| A **run** verb | Runs the graph |
| A **show** verb | Prints the graph without running it — the closures, the tasks, the phases, the validators |
| A `--dry-run` on run | Resolves and validates everything, dispatches nothing. Proves the load-time checks fire |

### 4.2 Observable output

The output *is* the deliverable — a demo whose result must be inferred from
exit code proves nothing to a reader. It must show, legibly:

- **The graph as loaded**: tasks, their parent, their phase, their handoffs.
- **Each dispatch**, in order, with the task and its bound agent.
- **Each handoff transition**, with the version and the verdict.
- **The validator verdicts**, each labelled `strong` or `weak` (validator spec
  §5.3) — so the demo teaches the taxonomy by using it.
- **The permission block**, as an expected event.
- **The final state of every task**, including the one still in
  `WAITING_HANDOFF`, and why.

Machine-readable output alongside the human-readable form, because the
acceptance criteria below are assertions over it.

---

## 5. Acceptance criteria

1. `pip install -e agent_sys` then the run verb completes in **under one minute**
   excluding model latency, with no GPU, no cluster, and no manual setup.
2. The graph loaded has a root task with `parent = None` and at least one subtask
   with a non-`None` parent (`task_graph` spec criterion 36).
3. Some task's `input_validation` phase is empty and dispatches nothing; another
   task's is populated and its validator runs (criterion 41).
4. The start entry subtask's dispatch and the end entry subtask's completion are
   both visible in the output (criterion 37).
5. **A leaf validator seals a handoff `INVALID`, and its consumer remains in
   `WAITING_HANDOFF` at the end of the run** — reported as the expected outcome,
   not as a crash.
6. **One node runs on `claude-agent-sdk` and one runs as a program**, and the
   handoff state each produces is indistinguishable in kind (main spec criterion 9).
7. **An out-of-zone write is blocked by the `PreToolUse` hook**, the agent
   receives the reason, and the block appears in the output as an expected event.
8. Every validator verdict in the output carries its `strong` / `weak` label.
9. `--dry-run` resolves every closure, validates every spec, and dispatches
   nothing — and a deliberately broken closure makes it fail with the offending
   file path (`closure` spec §5).
10. Running twice in succession succeeds without hand-editing anything.
11. **With no API credentials, the run fails with a clear message** naming what
    is missing. It does not fall back to a fake agent.
12. The machine-readable output is sufficient to assert criteria 2–8 without
    parsing prose.

---

## 6. Open questions

| Item | Status |
|---|---|
| **What the demo's task actually does** | §2.1 is illustrative. Something small, verifiable, and not contrived is wanted — a real check over a real artefact, so the validator has something to be honestly `strong` about |
| **Model cost** | One small `claude-agent-sdk` call per run is the intent, but the SDK's default is Claude Code's own tool set and system prompt, which is not small. Which model, which tools, and which effort level is a design-stage decision with a cost attached |
| **Where the demo lives in the package** | A `demo` package with a console script, a `tests/demo/`, or an `examples/` directory. This affects whether the demo is covered by CI, which decides whether it stays working |
| **Whether the demo doubles as an integration test** | It should — it is the only thing that exercises every component together. But criterion 11 requires it to fail without credentials, and a CI job without credentials would then fail. Splitting it into a credentialed and an uncredentialed half is the obvious answer and has not been specified |
