# Agent — Specification

| | |
|---|---|
| Status | Draft, revised after review |
| Revision | 6 — 2026-08-28. **§4.3: the loop is the agent's, the thread is not.** Rev. 5's "one thread per agent is the alpha's shape" named an owner while no document named one at all — nothing said who created an agent's thread or joined it. The task owns it and the agent borrows it for the main phase (`design.md` §7.5's `TaskAttempt`). The count is unchanged, so this corrects ownership rather than adding a requirement. §4.3's note on the monitor's loop widens with `monitor` spec rev. 14 — that loop now carries the task's planned advances as well as its exceptions, and is still the one thread that may not be shared. (rev. 5: 2026-08-27. **The user-interface brief.** An agent has its own **`mainloop()`** (§4.3): an agent is a live, stateful thing, and `start()` returning immediately raises the question of who is then executing. Every synchronous verb is sugar this layer wraps, not a subclass's to implement. ROADMAP §7.1 records attaching an agent's loop to a shared round-robin thread. (rev. 4: 2026-08-26. An agent spec is a jsonnet source in a task package (§3.1). (rev. 3: Backend order and its config/CLI fallbacks (§3.3); a backend implements the whole interface or raises (§3.3.1). rev. 2: Review of PR #132: two interface levels; permissions moved to the task; six knowledge types; async lifecycle with status; backend ≠ runner; logging moved to o11y. rev. 1: initial))) |
| Date | 2026-08-24 |
| Scope | What wraps a task spec for execution, and the backend abstraction |
| Source | The task definition §6, §7.8; the Claude Code and Cursor SDK references |
| Part of | [`../../docs/spec.md`](../../docs/spec.md) — the whole-system specification |
| Depends on | [`../../task_graph/docs/spec.md`](../../task_graph/docs/spec.md) §3.2.2, §3.3 |

---

## 1. Purpose

**An agent spec is the thing that wraps a task spec.**

A task spec says *what*: inputs, outputs, and a goal template. An agent spec says
*who executes it and how*, and it exists to abstract a small set of common
properties away from any particular harness:

| The agent spec abstracts | |
|---|---|
| **How to interact with the executor** | start, stop, interrupt, instruct, query |
| **Configuration** | rules, hooks, skills |

An executor may be an AI, a human, or a plain program — the system does not care
which, as long as the handoffs are standardised.

**This document specifies the spec and backend layers.** The runtime `Agent`
record — its id, its binding to a task, the `HandoffRef`s it touched — is
specified in `task_graph` spec §3.3 and is not restated here.

### 1.1 Two interface levels

The abstraction has two levels, and conflating them was a mistake in the previous
revision:

```
level 1   AgentSpec          a uniform interface toward the aiopt system's
          ─────────          task runner. Every executor looks the same here,
                             whether it is an AI, a human, or a shell script.

level 2   AiAgentBackend     what every major agent harness provides:
          ──────────────     Claude Code, Cursor, Codex. This level is
                             AI-specific and only exists for AI executors.
```

**Level 1 is what the task runner talks to.** It is deliberately thin: start,
stop, status, result. A program executor implements it directly and never touches
level 2.

**Level 2 is the AI-harness abstraction.** It wraps what the SOTA harnesses
have in common — history, interrupt, message queue, hooks, permissions, sessions
(§5). We start with the basic features and leave extensibility to the framework,
so phase one is not blocked on covering every harness's full surface.

### 1.2 In scope

- The agent spec: what it declares, and what it deliberately does not.
- The six knowledge types.
- The two interface levels, and the lifecycle with its statuses.
- Backends, and how `claude-agent-sdk` satisfies level 2.

### 1.3 Out of scope

- **The `Agent` runtime record** — `task_graph` spec §3.3.
- **Permissions** — a versioned *task* attribute, not an agent one (§3.2).
- **Logging** — the o11y subsystem's, not this one's (§6).
- **An agent's own durability** — its business, not the system's.
- **What happens inside a backend** — §4.2.

---

## 2. Design principles

| # | Principle | Consequence |
|---|---|---|
| 1 | **The executor is interchangeable** | AI, human, or program. The interface is the handoff |
| 2 | **Two levels, kept apart** | §1.1 |
| 3 | **The agent node is coarse** | Whatever a backend organises internally is invisible and unmanaged. §4.2 |
| 4 | **Knowledge is expected, warned about, and optionally enforced** | Not silently mandatory. §3.4 |
| 5 | **Prepared, not negotiated** | An agent spec arrives fully set up. There is no runtime interface for changing its rules. §4.4 |
| 6 | **The specification outlives the implementation** | Backends are consumables; the interface and the knowledge are assets |

---

## 3. The agent spec

A jsonnet source in a **task package**, rendered to YAML and validated against
the agent JSON Schema (main spec §4.3, §4.4). **An agent spec is bound to a task
spec** — the two are declared together in a closure.

### 3.1 What it declares

| Key | Meaning |
|---|---|
| `name` | Unique in the registry |
| `version` | The spec's own revision. **Maintenance metadata only** — nothing at runtime reads it (closure spec §1.2) |
| `description` | What this kind of executor is for |
| `kind` | `ai` \| `program` \| `human` |
| `backends` | For `kind: ai` — a list or dict of backend implementations. §3.3 |
| `env` | Environment requirements, resolved by `env_mgr` |
| `knowledge` | §3.4 |
| `rules` / `hooks` / `skills` | Configuration, stored in canonical form. §4.5 |
| `assets` | **Filled by `spec_loader`, not written.** This agent's own directory under the package's `assets/`, found by the same three folder spellings a body lookup uses — `X`, `X.agent`, `agent.X`. Two matching directories is `SpecInconsistent`; an explicit binding is legal and warns. §4.5a |
| `recipes` | The **agent layer** of three recipe layers; `env_mgr` recipe YAMLs by name or package-relative path. §4.5a |
| `agent_plugins` | Bare names under `agent_sys/agent_plugins/` — the agent plugins this repository ships. §4.5a |

Nine keys became twelve, and the three additions are one thing: **an agent may
now carry components, not only files.** §4.5a is why that needed new keys instead
of a longer `skills` list.

### 3.2 Permissions are not here

**Permissions live on the task, versioned with it** (`task_graph` spec §3.2.2).
The previous revision put them on the agent spec, and that was wrong for two
reasons:

- **Permission is a runtime fact about a particular piece of work**, not a
  property of a kind of executor. The same agent spec running two different tasks
  should reach two different sets of files.
- **A task must reach its whole subgraph.** Since a task is strongly bound to its
  current agent, and its subtasks' storage is nested inside its own, the
  permission set follows the task tree by construction. Deriving it per-agent
  would mean recomputing a subtree's reach every time an agent is minted.

By default a task's executor reaches its own input and output handoffs, its
workspace, its playground, its log location, and everything belonging to its
subtasks — recursively.

### 3.3 Backends are a list, not a singleton

For an AI agent, `backends` is a list or dict of implementations:

```yaml
backends:
  - key: claude_code_sdk
    backend_entry: ...
  - key: cursor_sdk
    backend_entry: ...
```

A leaf task is bound to an agent runtime, and **which backend runs it may be
chosen at run time** among the declared ones. Declaring several is how an agent
spec survives a harness being unavailable, deprecated, or wrong for a particular
job.

**The list is ordered, and the order is the preference.** Three sources decide
which backend runs, in descending precedence:

| | Source | When it applies |
|---|---|---|
| 1 | **A CLI override** | Always wins. Forces one backend for the whole run, so a run can be pinned when reproducing something |
| 2 | **The agent spec's own order** | The normal case: the first declared backend that is available |
| 3 | **A global preference order**, in the whole-system config | The fallback: the spec declares no usable order, or none of its preferred backends is available |

The global order being config rather than code is the point: which harness a site
prefers is a deployment fact, not a system fact.

### 3.3.1 A backend either implements the interface or fails loudly

**Every backend implements the whole of §4.3.** A backend that cannot interrupt,
or has no programmatic hook callback, is not a partially-supported backend — it
is a backend whose adapter is incomplete, and that is the adapter author's
problem, not something the system negotiates around.

So there is no capability negotiation, and no per-capability degradation. An
unimplementable method **raises**, and the system's only responsibility is that
the error goes somewhere useful:

| Who called it | Where the error goes |
|---|---|
| o11y | o11y handles it — it asked, it deals with the answer |
| The monitor | The monitor handles it, the same way it handles any task failure (validator spec §3.4) |
| The task runner, mid-execution | The task fails, and the monitor takes it from there |

The reason to draw it this way rather than as a capability matrix: a matrix makes
every caller branch on what the backend can do, and those branches are untested
in exactly the configuration the site actually runs. An exception is one code
path, and the incompleteness is visible the first time it matters instead of
being silently routed around.

The consequence is stated plainly: **whoever adapts a backend guarantees its own
coherence.** Cursor's hooks are file-based rather than callback-based, so its
adapter's job is to make file-based hooks satisfy §4.3 — not to report that hooks
are unavailable.

### 3.4 Knowledge: six types, extensible

Knowledge arrives as **knowledge handoffs** (handoff spec §4.1) — versioned,
digested, checkable — not as inline prose in a YAML file.

| # | Type | What it is |
|---|---|---|
| 1 | **few shot** | Worked examples. The most effective constraint available |
| 2 | **runnable** | Something the agent can execute to see the mechanism work |
| 3 | **official reference** | First-party documentation, papers, technical reports |
| 4 | **expert experience / suggestion** | Accumulated human judgement: pitfalls, heuristics, what to watch for |
| 5 | **suggested / verifiable resource source** | Where to go to check something — so the agent verifies rather than guesses |
| 6 | **runtime-generated** | Produced during the run: a Serena analysis result, a `compile_commands.json` |

**The list is extensible.** Six is where it stands, not a closed vocabulary.

Type 6 is the interesting one: knowledge that does not exist until the work
starts. It is still a knowledge handoff, so it is versioned and checkable like
the rest — which is what stops it becoming an untracked side-channel.

### 3.5 Knowledge is strongly suggested, not hardcoded mandatory

The previous revision made all knowledge parts a hard load-time requirement.
Instead:

| | |
|---|---|
| **Default** | Missing knowledge produces a **warning**, naming what is absent |
| **CLI option** | A run-config flag makes it **mandatory** — the spec is then rejected |

A hard requirement blocks bring-up, and a silent absence is how zero-shot agents
happen. A warning plus an opt-in gate is the shape that survives both.

Configuration for this — and for everything else in the alpha — is **one global
YAML file with well-classified partitions that everyone reads**. A real config
dispatch and dissemination system is on the roadmap; the alpha does the simple
thing.

### 3.6 The registry

One of the four independent registries. Load-time checks:

1. The YAML validates; the name is unique.
2. Every declared backend resolves.
3. Every knowledge handoff named resolves in the handoff registry.
4. Knowledge coverage is reported — warned by default, fatal under the flag
   (§3.5).

---

## 4. The two levels

### 4.1 Level 1 — toward the task runner

**`TaskRunner` is not a backend, and a backend is not a `TaskRunner`.** The
distinction is the point of having two levels:

| | What it is |
|---|---|
| **`TaskRunner`** | A middle-man between the task, the scheduler, and the agent. A pure function / helper, existing for decoupling. It also runs the three phases (`task_graph` spec §3.2.1) |
| **Backend** | A real thing that runs a defined "agent" — a Claude Code agent, say — **after** the environment, workspace, handoffs, and playground have been deployed |

The runner orchestrates; the backend executes. The runner decides *that* an agent
should start and *in what order* the phases run; the backend knows how to
actually start one.

### 4.2 The agent node is coarse

**Whatever multi-agent structure a backend organises internally is invisible to
this system.** The work handed to a backend was done by one agent, as far as the
system is concerned.

A backend *may* report how many subagents it has, and a harness may make every
subagent's history visible — Claude Code does. That is an **o11y** concern, and
o11y may surface it. What stays true regardless:

> **Only the main backend agent may be interacted with.** Interrupt, instruct,
> query — all of it addresses the main agent. Reaching into a subagent is not
> offered, at any point, however visible the subagent is.

### 4.3 Lifecycle and status

Modelled on what both surveyed SDKs converged on — a durable handle plus a
per-submission unit of work.

```python
class AgentBackend(Protocol):
    status: AgentStatus

    def start_async(self, on_started: Callable[[], None]) -> None: ...
    def wait(self) -> AgentResult: ...
    def start(self) -> AgentResult: ...      # sugar: start_async + wait
    def stop(self) -> None: ...
    def interrupt(self) -> None: ...
    def instruct(self, message: str) -> None: ...
    def query(self) -> AgentHistory: ...
```

**`start_async` is the primitive.** It returns immediately and takes a callback
invoked when the agent *really* starts — deploying an environment and launching a
harness takes long enough that "started" and "asked to start" are different
events, and the difference is exactly what a monitor needs to see.

`wait()` and the synchronous `start()` are sugar over it.

**Every synchronous verb is sugar, and none of them is a subclass's to
implement.** An adapter implements the asynchronous form; this layer wraps it.
That is one rule rather than a per-method convention, and it is what keeps a
backend from shipping a `stop()` that blocks differently from every other
backend's.

**No task parameter.** The task uuid is already in the runtime agent's schema, so
passing it again would be a second source of truth.

#### An agent has its own mainloop

**An agent is a live, stateful thing, and without a loop of its own nobody can
interact with it.** The question that settles it is the concrete one: `start()`
returns immediately — *then who is executing?*

So level 1 is two halves and only one of them is a caller-facing verb:

| | |
|---|---|
| The verbs above | What the task runner, a monitor, or an interactive surface calls |
| **`mainloop()`** | What actually drives the agent. It owns the agent's status, services the message queue, and is what `start_async` hands work to |

An adapter implements `mainloop()`. Nothing above level 1 calls it — the runner
calls `start_async` and gets a callback — and it is specified here rather than
left implicit because an interface of five verbs with nothing behind them is not
an interface, it is a wish.

**The monitor's loop is a different loop with a different job**
(`task_graph` spec §3.5): this one runs *the agent*, that one handles the
**task's events** — its planned phase advances and its exceptions alike
(`../../monitor/docs/spec.md` §2.2, widened 2026-08-28). They are not one
mechanism serving two callers, and unlike the agent's loop, **the monitor's owns
its own thread** — §4.3 below on why that one is not shared.

**The loop is the agent's; the thread is not.** Amended 2026-08-28 — this said
*"one thread per agent is the alpha's shape"*, which named an owner at a time when
**no document named one at all**: nothing anywhere said who created an agent's
thread or who joined it.

**The task owns the thread and the agent borrows it.** One thread is started per
dispatch and the three phases take it in turn; during the main phase, `mainloop()`
is what runs on it, and it is handed back when the phase ends. So an agent still
has a loop of its own — the question this section exists to answer, *`start()`
returns immediately, then who is executing*, still has the same answer — while the
lifetime question now has an owner (`design.md` §5.1.1, §7.5).

**The count is unchanged, and that is why this is a correction and not a new
requirement:** a graph running K leaf tasks has K threads either way, because a
leaf has one agent. What changes is that they are not *two* sets of K.

[`../../docs/ROADMAP.md`](../../docs/ROADMAP.md) §7 records the intended
refinement: a loop may later attach to a shared thread that round-robins over
every attached agent, which is the same trade the global monitor already makes.
**That refinement is now about sharing task threads**, and is otherwise unchanged.

**One thread is not shared, and must not be: the monitor's.** Giving the watched
and the watcher one heartbeat is the failure a watchdog exists to prevent
(`../../monitor/docs/spec.md` §1.1), and it is the one merge this system declines.

**Agent status:**

```
pending → deploying → running → { finished | failed | interrupted }
```

**`Task.status` is a superset of the status of the agent at the top of its
execution stack** (`task_graph` spec §3.2.1). The task adds the states that exist
when no agent is bound — waiting, cancelled — and the phase states.

### 4.4 No runtime configuration interface

**An agent spec arrives fully prepared** — rules, hooks, skills, and environment
set up before the agent starts, by `env_mgr` as part of preparing the
environment. There is **no runtime interface for changing them**.

What is needed instead is a **format transform helper**: an independent module
that converts rules, hooks, and skills between harness formats. Not part of the
agent spec, and not a runtime negotiation.

### 4.5 Material is stored in canonical form

Agent material splits in two, and the split decides how it is stored:

| Kind | Examples | Stored as |
|---|---|---|
| **Backend-format** | plugins, rules, hooks | **Claude Code's format, canonically**, with a transform helper converting to other harnesses |
| **Format-free** | knowledge | Its own form — it is a handoff |

Picking one canonical format matters more than which one is picked: with N
harnesses, storing each in its own format needs N² converters, and storing one
canonical form needs N.

**Which Claude Code surface is canonical: the declarative one.** Design O4 asked,
because *"Claude Code's format"* named two different execution models — the
`.claude/settings.json` tree and the SDK's `ClaudeAgentOptions(hooks={...})`
callbacks — and every surveyed converter targets the first while nobody converts
programmatic callbacks at all. The answer is the first, and it is now what the
code does: `env_mgr` writes `<zone>/config/settings.json` and points
`CLAUDE_CONFIG_DIR` at it. The consequence O4 warned about is therefore not
incurred — criterion 13 rests on the surface the prior art actually covers.

The callback form is not forbidden; a backend config may still carry `hooks`, and
`claude_sdk` passes it through. What it is not is the **stored** form, so nothing
in a package or a component is written that way.

### 4.5a A component is a tree, and that needed three keys

§4.5's three lists are lists of *files*. A Claude Code component is a directory:
a skill is a directory, a plugin marketplace is a directory of directories, and
an MCP server is a process to register rather than a file to place. Naming every
file would make a package author restate a layout the harness already fixes.

Three **origins**, not three levels: what differs between them is who owns the
directory, and the numbering was a vocabulary each document restated slightly
differently while carrying no ordering the table does not already give.

| owner | what | declared how |
|---|---|---|
| upstream | serena, a marketplace plugin, an apt/pip tool | `recipes: [...]` |
| this repository | the agent plugins `agent_sys` ships | `agent_plugins: [...]`, or a recipe item carrying `tags: [internal]` |
| one task package | what it carries for one agent | **undeclared** — `<assets>/.claude/` |

**The last two have one on-disk shape**, in Claude Code's canonical layout:
`settings.json`, `skills/<name>/`, `plugins/` (a local marketplace),
`.mcp.json`, and `tools/*.mcp.py` / `tools/*.tooldef.py`. One shape means one
installer and means promoting a component is moving a directory.

**A package's own material is undeclared on purpose.** A declaration would be a
second statement of what the directory already says, and the two would drift the
first time somebody moved it without editing the YAML.

`env_mgr/agent_assets.py` installs all three; `env_mgr/docs/design.md` §11.5a is
the mechanism, including the measured ordering constraint that decides when
`settings.json` is written, the marketplace copy probe F forced, and why a
recipe runs the shipped machinery as a subprocess. What reaches this package is
`Assignment.mcp_servers` and `Assignment.tools`.

**A component names a binary through `${VAR}`, never through `PATH`.** An
`.mcp.json` entry is expanded against the zone environment before it becomes an
`mcp_servers` entry, and an unresolved name is an error. That is not a
convenience: `PATH` is derived from the granted policy at prepare step 2, and a
directory a recipe installs into does not exist until step 6b — so
`"${UV_TOOL_BIN_DIR}/serena"` is the only spelling that works, and it is the one
measured working.

---

## 5. `claude-agent-sdk` as the first backend

PyPI, requires Python ≥ 3.10, bundles the Claude Code CLI.

### 5.1 The level-2 capability mapping

| Level-2 capability | Claude Agent SDK | Cursor SDK |
|---|---|---|
| durable handle | `ClaudeSDKClient(options)` + `connect()` | `Agent.create(...)` → `agent_id` immediately |
| one submission | `query(prompt, session_id)` | `agent.send(prompt)` → a **`Run`** handle |
| wait | `receive_response()` → `ResultMessage` | `run.wait()` → `RunResult` |
| stream | `receive_messages()` | `run.stream()` / `run.events()` |
| **interrupt** | `interrupt()` | `run.cancel()` |
| **status** | `ResultMessage.subtype`, `.terminal_reason` | `run.status`: running / finished / error / cancelled / expired |
| **history** | `get_session_messages()`, `list_sessions()`, `get_session_info()` | `agent.list_messages()` |
| **instruct** | streaming input: `query(AsyncIterable[dict])` | `agent.send(...)` on the same agent |
| metrics | `ResultMessage`: `duration_ms`, `num_turns`, `total_cost_usd`, `usage`, `model_usage` | `agent.get_usage()`, `run.usage` |
| hooks | `ClaudeAgentOptions(hooks={HookEvent: [HookMatcher]})` | file-based `.cursor/hooks.json` only |
| permission gate | `can_use_tool`, `PreToolUse` hook | Run Modes, allowlist, `local.sandboxOptions` |
| rules / prompt | `system_prompt`, `setting_sources`, `agents`, `skills` | rules, skills, subagents |
| resume | `resume=`, `session_id=`, `fork_session=` | `Agent.resume()` |
| reload config | — | **`agent.reload()`** — re-reads hooks, MCP, subagents without disposing |

**The most important structural lesson is Cursor's Agent/Run split**: a durable
container holding conversation state, and a per-submission unit with its own
status and cancellation. That is what §4.3's status model follows, and it is why
`interrupt` belongs to the submission rather than to the agent.

Two capabilities worth adopting later, neither in the alpha: Cursor's `reload()`
(§4.4 forbids runtime *negotiation*, but re-reading prepared config is different)
and its artifact listing.

### 5.2 Two caveats, both from the reference

**`can_use_tool` does not fire for every call.** The SDK reference is explicit:
it is invoked only when the permission flow resolves to a prompt, and calls
approved by `allowed_tools`, a settings rule, or a permissive mode never reach
it. **To gate every call, use a `PreToolUse` hook.**

**`interrupt()` does not clear the buffer.** Messages already produced by the
interrupted task, including its `ResultMessage`, stay in the stream and must be
drained before a new query's response can be read. A backend that sends a new
query straight after interrupting and reads once gets the *old* task's messages.

### 5.3 The hook is a first gate, not the boundary

`PreToolUse` is where this system's checks are expressed. It is **not** what
enforces them: a hook sees `{"tool_name": "Bash", "command": "python3 x.py"}`
with no file path in it, and confinement is an OS-level mechanism.
[`../../env_mgr/docs/spec.md`](../../env_mgr/docs/spec.md) §4 specifies the
sandbox chain and why the hook alone is insufficient.

### 5.4 Session identity

The SDK's session id and this system's `AgentId` are different things; the
backend records the correspondence. Re-establishing a run's context after a
restart is the agent's own business.

---

## 6. Logging belongs to o11y

**Logging is not specified here.** It is an independent subsystem —
[`../../docs/ROADMAP.md`](../../docs/ROADMAP.md) §1 — with backend-specific
writers producing schema-conformant JSON, decoupled post-process visitors reading
it, and storage managed by `env_mgr` under the ordinary permission rules.

What belongs to the agent spec is only this: an agent has a log tool, its levels
are `debug` / `info` / `warning` / `error`, and three things can require an entry
— system policy, the agent's own rules, or a skill/rule/hook. Where those entries
go, and what is done with them, is o11y's.

Also o11y's, and recorded there: **per-run metrics for every agent spec** — cost,
wall-clock, resume count, failure count.

---

## 7. What the system records about an agent

Only the index, never the content:

| Recorded | Not recorded |
|---|---|
| The `AgentId` and its spec name | The prompt |
| The task it is bound to | The reasoning |
| The `HandoffRef`s it touched | The intermediate tool calls |
| Its execution record | The backend's internal structure |

The agent's history *is* available — through `query()`, from the backend — but it
is the backend's data, fetched on demand, not the system's record.

---

## 8. Acceptance criteria

1. An agent spec naming an unregistered backend, or an unresolvable knowledge
   handoff, is rejected at load with the offending value named.
2. **Missing knowledge warns by default and is fatal under the CLI flag** — the
   same spec loads in one mode and is rejected in the other.
3. An agent spec declaring several backends loads, and **the first available one
   in declared order is used**; with none of them available the global config
   order decides; a CLI override beats both (§3.3).
4. **A backend method that its adapter does not implement raises**, and the
   exception surfaces to whoever called it — o11y or the monitor — rather than
   being reported as a missing capability (§3.3.1).
5. **The agent spec carries no permissions.** Reach is decided by the task's
   versioned permissions; the same agent spec on two tasks reaches two different
   sets.
6. **A backend is not a `TaskRunner`.** The runner drives the three phases and
   calls the backend for the main phase; substituting the backend leaves the
   runner unchanged.
7. `start_async` returns immediately and invokes its callback when the agent
   really starts; `wait()` blocks until a result; `start()` is equivalent to both.
8. Agent status transitions `pending → deploying → running → finished`, and
   `Task.status` is a superset of the stack-top agent's.
9. `interrupt()` stops a running agent, and the backend **drains the interrupted
   submission's buffered messages** before reporting — verified by asserting the
   next query's response is the new one's, not the old one's.
10. `instruct()` reaches a running agent and affects its behaviour without ending
   the run.
11. `query()` returns the agent's history for a completed run, and the session
    corresponds to the recorded `AgentId`.
12. **Only the main backend agent is interactable.** No interface reaches a
    subagent, even where the backend exposes one.
13. Rules, hooks, and skills are stored in Claude Code's canonical format, and
    the transform helper converts them to another harness's format losslessly for
    what both support.
14. **There is no runtime interface for changing an agent's rules or hooks**; a
    spec arrives prepared.
15. **Swapping the backend changes no other component.** The demo runs with the
    SDK backend and with a program executor, and the resulting handoff state is
    identical.
16. The system's record of a run contains no prompt text and no reasoning —
    asserted over the persisted records.

---

## 9. Open questions

| Item | Status |
|---|---|
| **The human executor** | `kind: human` is declared and nothing implements it. What `query()` and `interrupt()` mean for a person is undecided — probably "the notes they took" and "ask them to stop". The deploy-and-wait model is specified (`../../docs/ROADMAP.md` §3); the interface is not |
| **Cost attribution** | Both SDKs report cost per session or per run. The system has a consumable token pool expecting a settled figure at completion. The two disagree about what a "run" is when a session is resumed |
| **Runtime-generated knowledge lifecycle** | Type 6 (§3.4) is produced during a run. Whether it persists past the run, and whether it becomes available to later tasks, is unspecified — and it is the type most likely to be worth keeping |
