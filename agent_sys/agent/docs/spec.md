# Agent — Specification

| | |
|---|---|
| Status | Draft, pending review |
| Revision | 1 — 2026-08-24 |
| Date | 2026-08-24 |
| Scope | What an executor declares, what its knowledge must contain, and the backend abstraction |
| Source | The task definition §6, §7.8; the Infera × Hyperloom kickoff report §2, §5F |
| Depends on | [`../../task_graph/docs/spec.md`](../../task_graph/docs/spec.md) §3.3 — the `Agent` record |

---

## 1. Purpose

An agent is **one executor of one task run**. The system's own definition is
deliberately weak: an agent may be an AI, a human, or a program, and the system
does not care which — *as long as the handoffs are standardised*. That
indifference is the point. It is what makes the executor a swappable part rather
than the system's foundation.

**This document specifies the spec and backend layers.** The runtime `Agent`
record — its id, its binding to a task, the `HandoffRef`s it touched, its
persistence — is already specified in `task_graph` spec §3.3 and is **not**
restated here.

```
task_graph spec §3.3       this document
────────────────────       ────────────────────────────────
Agent      the instance    AgentSpec     the kind
  .id                        name
  .task_id                   permissions
  .handoffs[]                env
  .spec ────────────────────►knowledge
                             backend
                                │
                                ▼
                          AgentBackend  the harness abstraction
                             history · interrupt · queue · hooks
```

### 1.1 In scope

- The YAML agent spec: permissions, environment, knowledge, backend.
- The four mandatory parts of an agent's knowledge.
- The backend abstraction, and why the system's agent node is coarse.
- `claude-agent-sdk` as the first backend, with the capability mapping.
- The logging tool and its levels.

### 1.2 Out of scope

- **The `Agent` runtime record** — `task_graph` spec §3.3.
- **An agent's own durability.** `task_graph` spec §1.2 is explicit: an agent is
  responsible for re-establishing what it was doing after a restart. This system
  records the binding, not the lifecycle.
- **What happens inside a backend.** §4.1.
- **Where the workspace and playground physically live** —
  [`../../env_mgr/docs/spec.md`](../../env_mgr/docs/spec.md).

---

## 2. Design principles

| # | Principle | Consequence |
|---|---|---|
| 1 | **The executor is interchangeable** | AI, human, or program. The interface is the handoff, not the executor |
| 2 | **The agent node is coarse** | Whatever a backend organises internally is invisible and unmanaged. §4.1 |
| 3 | **No zero-shot** | Knowledge is mandatory and has four required parts. §3 |
| 4 | **An agent knows only what it needs and touches only what it may** | The permission list is a whitelist, and the validator boundary depends on it. §5.4 |
| 5 | **Observable and interruptible** | An agent that cannot be watched cannot be repaired; one that cannot be interrupted cannot be steered. §4.2 |
| 6 | **The specification outlives the implementation** | Backends are consumables; the interface and the knowledge are assets. §4.4 |

---

## 3. The agent spec

A YAML file in the predefined-spec folder, constrained by a JSON Schema.
`Task.agent_spec` names one (`task_graph` spec §3.2).

### 3.1 What it declares

| Key | Meaning |
|---|---|
| `name` | The spec name. Unique in the registry |
| `description` | What this kind of executor is for |
| `backend` | Which backend runs it. §4 |
| `permissions` | §3.2 |
| `env` | Environment requirements, resolved by `env_mgr` |
| `knowledge` | §3.3. **Mandatory** |
| `resources` | Default pool names and amounts a task using this spec declares |

### 3.2 Permissions are a whitelist

| Key | Meaning |
|---|---|
| `handoffs.read` | Which handoff kinds this agent may read |
| `handoffs.write` | Which it may produce |
| `zones` | Which filesystem permission zones it may reach. See [`../../env_mgr/docs/spec.md`](../../env_mgr/docs/spec.md) §4 |
| `workspace` | Which workspace, if any |
| `playground` | Which playground |

A whitelist rather than a blacklist, because the validator boundary (validator
spec §8.1) is expressed as *absence*: a producing agent cannot read the checking
standard because its permission list does not name the zone the standard lives
in. A blacklist would make that boundary depend on remembering to add an entry.

### 3.3 Knowledge is mandatory and has four parts

**Zero-shot is forbidden.** The rule comes from the kickoff report's development
requirements, and every agent spec must declare all four:

| # | Part | Why |
|---|---|---|
| 1 | **Few-shot examples** — at least one | Examples are the most effective constraint available. An agent given a worked example of the output shape produces that shape; an agent given a description of it produces an interpretation |
| 2 | **Official first-party references** | Documentation, papers, technical reports. Prevents the agent relying on stale training data — which, for a fast-moving inference engine, is most of what it thinks it knows |
| 3 | **Mechanism-explaining code** | Where a mechanism can be shown in code, show it in code. Code is the most precise specification available, and the agent reads code better than it reads prose about code |
| 4 | **Verification tools** | LSP, Context7, and the like — so the agent can *check* rather than guess. An agent without a way to verify will confabulate, and the confabulation will be fluent |

**Knowledge arrives as knowledge handoffs** (handoff spec §4), not as inline text
in the spec. This is deliberate: knowledge accumulates, is versioned, is
digested, and is checkable — the same properties every other artefact in the
system has. Inline prose in a YAML file has none of them.

A load-time check enforces the four parts. An agent spec declaring no few-shot
example is rejected; §2 principle 3 is not advice.

### 3.4 The registry

One of the four independent registries (main spec §4.1). Name → spec, over the
predefined-spec folder. Load-time checks:

1. The YAML validates against the schema.
2. The name is unique.
3. `backend` resolves to a registered backend.
4. All four knowledge parts are present, and every knowledge handoff named
   resolves in the handoff registry.
5. Every permission zone named exists (`env_mgr` spec §4).

---

## 4. The backend abstraction

### 4.1 The system's agent node is coarse

**Whatever multi-agent structure a backend organises internally is invisible to
this system, and explicitly not managed by it.** The system always considers the
work handed to a backend to have been done by one agent.

Three consequences, all intended:

- **Backends are swappable** because none of them can leak structure into the
  system's model.
- **The system's observability claims stay honest.** It reports that an agent ran
  and what it touched, never what it thought — because for a backend with
  internal subagents, "what it thought" is not a single thing.
- **The audit record stays finite.** One `Execution` per run, whatever happened
  inside.

The cost: a backend that spawns twenty subagents looks, from here, exactly like
one that does not. If that distinction ever needs to be visible, it becomes a
field on the backend's own reporting, not a change to the agent model.

### 4.2 The required interface

A backend is a thin wrapper exposing a uniform surface. The initial
implementation may be naive as long as the decoupling is real — the interface is
the asset, not the wrapper.

| Capability | Required because |
|---|---|
| `get_history()` | Observability (main spec §2 principle 4), and it is one of the three things an agent needs to be resumable (env_mgr spec §5) |
| `interrupt()` | Interventionability (§2 principle 5) |
| `append(message)` | The other half of intervention: steering a running agent without killing it |
| `set_rule(...)` | Injecting the system's rules and the task's constraints |
| `add_hook(event, handler)` | The enforcement mechanism for every boundary in the system that is not a scheduler boundary. §5.4 |
| `start(task, agent)` / `stop(...)` | The `TaskRunner` protocol `task_graph` spec §5 already defines. A backend satisfies it |

The last row matters structurally: **a backend is a `TaskRunner`.** The seam
already exists — `task_graph` specifies `TaskRunner` as a registered `Protocol`
with `start` and `stop`, and ships only a fake. A backend is what fills it, and
nothing in the scheduler changes to accommodate one.

### 4.3 Observation and interaction, stated as the task definition does

The task definition asks for three things by name, and they map onto the above:

| Asked for | Interface |
|---|---|
| Capture history | `get_history()` |
| Interrupt | `interrupt()` |
| Append to the message queue | `append(message)` |

### 4.4 Consumables and assets

The kickoff report's research on harness engineering (§5F) reaches a conclusion
this spec adopts wholesale, because it determines what is worth specifying
carefully:

> Split the architecture in two. **Models, runtimes, and CLI shells are
> consumables** — route every model call through one gateway and every tool
> through a standard protocol, so replacing them is a configuration change.
> **Tool implementations, the knowledge base, skills, and private evaluation
> sets are assets** — these survive across generations. Specifications outlive
> implementations.

Which is why §4.2 specifies an interface and §3.3 makes knowledge mandatory,
while §5 treats a specific SDK as one satisfying implementation.

---

## 5. `claude-agent-sdk` as the first backend

`claude-agent-sdk` (PyPI, 0.2.144, requires Python ≥ 3.10, bundles the Claude
Code CLI) is the first backend. It satisfies every capability in §4.2.

### 5.1 The capability mapping

Verified against the SDK reference at
`https://code.claude.com/docs/en/agent-sdk/python.md` unless another page is
named.

| Required capability | SDK surface |
|---|---|
| `get_history()` | `get_session_messages(session_id, directory=, limit=, offset=)` → `list[SessionMessage]`; `list_sessions(...)` → `list[SDKSessionInfo]`; `get_session_info(session_id)` |
| `interrupt()` | `ClaudeSDKClient.interrupt()` — streaming mode only |
| `append(message)` | Streaming input: `client.query(prompt: AsyncIterable[dict])`, yielding `{"type": "user", "message": {...}}` items; or a further `client.query(str)` on the same session |
| `add_hook(event, handler)` | `ClaudeAgentOptions(hooks={HookEvent: [HookMatcher(matcher=..., hooks=[cb])]})` |
| permission gate | `ClaudeAgentOptions(can_use_tool=...)` for the prompt path; a `PreToolUse` hook to gate **every** call. §5.4 |
| `set_rule(...)` | `system_prompt` (string, `{"type": "preset", "preset": "claude_code", "append": ...}`, or `{"type": "file", "path": ...}`); `setting_sources`; `agents`; `skills` |
| resume | `resume=<session_id>`, `session_id=`, `fork_session=`, `continue_conversation=` |
| workspace scoping | `cwd=`, `add_dirs=` |
| tool restriction | `allowed_tools=`, `disallowed_tools=`, `tools=` |
| custom tools | `@tool` decorator + `create_sdk_mcp_server(...)`, registered through `mcp_servers=` |
| cost and usage | `ResultMessage` fields; `max_budget_usd=`, `max_turns=` |
| structured output | `output_format={"type": "json_schema", "schema": {...}}` |

### 5.2 The hook events available

From `HookEvent` in the Python reference:

```
PreToolUse · PostToolUse · PostToolUseFailure · UserPromptSubmit
Stop · SubagentStop · PreCompact · Notification
SubagentStart · PermissionRequest
```

`PreToolUse` is the one this system depends on. The others are available and
unused in v1.

### 5.3 Two caveats worth carrying forward

Both are stated plainly in the SDK reference and both would otherwise be found
the hard way.

**`can_use_tool` does not fire for every call.** The reference:

> The callback is the SDK replacement for the interactive permission prompt: it's
> invoked only when the permission evaluation flow resolves to a prompt. Tool
> calls already approved by an `allowed_tools` entry, a settings allow rule, or
> the permission mode … never invoke it. **To gate every tool call, use a
> `PreToolUse` hook instead.**

This decides §5.4.

**`interrupt()` does not clear the buffer.** Messages already produced by the
interrupted task, including its `ResultMessage`, remain in the stream and must be
drained with `receive_response()` before a new query's response can be read. A
backend that sends a new query straight after interrupting and reads once gets
the *old* task's messages.

### 5.4 The permission boundary is a `PreToolUse` hook

**Every filesystem and tool boundary this system defines is enforced by a
`PreToolUse` hook, not by `can_use_tool` and not by `allowed_tools`.** §5.3 is
the reason: the other two mechanisms are bypassed by exactly the configurations
an operator is most likely to set.

The hook is what makes two boundaries real rather than intended:

| Boundary | Mechanism |
|---|---|
| An agent may not write outside its zone (env_mgr spec §4) | The hook resolves the target path, matches it against the zone prefix convention, and blocks a write outside the agent's declared zones |
| A producing agent cannot reach the validator's context (validator spec §8.1) | The same hook, denying a read into a zone the agent's permission list does not name |

The hook returns `{"decision": "block"}` with a `systemMessage`, so the agent
learns why rather than silently failing — which matters, because an agent that
does not know why it was blocked will retry creatively.

### 5.5 Session identity

The SDK's session id and this system's `AgentId` are different things, and the
backend records the correspondence. `task_graph` spec §3.3 already establishes
that one agent means one run; the SDK's `session_id`, `fork_session`, and
`resume` are how a backend re-establishes a run's context after a restart —
which is the agent's own business (§1.2), not the system's.

---

## 6. Logging

An agent has a log tool with four levels:

```
debug · info · warning · error
```

**Three things can require a log entry**, and the distinction matters because it
determines who can silence one:

| Source | Example |
|---|---|
| System-level logging policy | Every handoff read is logged at `info` |
| The agent's own logging rules | This agent logs each optimisation attempt |
| A skill, rule, or hook's demand | A hook that blocks a write logs at `warning` |

A system-level requirement cannot be waived by an agent's own rules. That is what
makes the log usable as evidence rather than as the agent's account of itself —
the same reason validator spec §8.1 exists.

### 6.1 The pattern the logs should follow

From the kickoff report's harness research, and adopted because it is
counter-intuitive enough to be worth stating:

> **Silent on success, loud on failure.** When a check passes the agent hears
> nothing; when it fails, the error text is injected back into the loop so the
> agent can correct itself.

An agent told about its successes learns nothing and spends context. An agent
handed its failure verbatim usually fixes it.

---

## 7. What the system records about an agent

Only the index, never the content. `task_graph` spec §7 already states this and
it is repeated here because it is the boundary readers most often expect to be
elsewhere:

| Recorded | Not recorded |
|---|---|
| The `AgentId` and its spec name | The prompt |
| The task it is bound to | The reasoning |
| The `HandoffRef`s it touched | The intermediate tool calls |
| Its execution record: attempt, versions in and out, outcome | The internal structure of the backend's own work |

The agent's history *is* available — through `get_history()`, from the backend —
but it is the backend's data, fetched on demand, not the system's record.

---

## 8. Acceptance criteria

1. An agent spec declaring no few-shot example is rejected at load, and the
   message names the missing knowledge part.
2. An agent spec naming an unregistered backend, an unresolvable knowledge
   handoff, or a nonexistent permission zone is rejected at load; each message
   names the offending value.
3. **A backend satisfies `TaskRunner` unchanged.** The `claude-agent-sdk` backend
   is registered as `runner` in `bootstrap.build_registry()` and the scheduler's
   existing tests pass against it with no scheduler change.
4. `get_history()` returns the agent's messages for a completed run, resolved
   through the backend, and the returned session corresponds to the recorded
   `AgentId`.
5. `interrupt()` stops a running agent, and the backend drains the interrupted
   task's buffered messages before reporting — verified by asserting that the
   next query's response is the new query's, not the old one's.
6. `append(message)` reaches a running agent and affects its subsequent
   behaviour, without ending the run.
7. **A `PreToolUse` hook blocks a write outside the agent's declared zones**, and
   the agent receives a message saying why. Verified with `allowed_tools`
   deliberately naming the tool, to demonstrate that the hook — not the allow
   list — is what gates.
8. **A producing agent cannot read a validator's checking standard.** The same
   hook denies it; the denial is recorded at `warning`.
9. A system-level logging requirement cannot be suppressed by an agent's own
   logging rules.
10. **Swapping the backend changes no other component.** The demo graph
    (`../../demo/docs/spec.md`) runs with the SDK backend and with a program
    executor, and the resulting handoff state is identical.
11. The system's record of a run contains no prompt text and no reasoning —
    asserted over the persisted records, not by inspection.

---

## 9. Open questions

| Item | Status |
|---|---|
| **The human executor** | §2 principle 1 says a human may be an executor and nothing contradicts it, but no backend implements one. What a human backend's `get_history()` and `interrupt()` mean is undecided — the honest answers are probably "the notes they took" and "ask them to stop" |
| **The observer** | Main spec §10 leaves this open system-wide. The agent-level part of it: an outside view of whether *this* agent has drifted from its goal or is looping. `get_history()` supplies the data; nothing consumes it |
| **Backend capability negotiation** | §4.2 lists required capabilities as though every backend has all of them. A backend that cannot interrupt is still useful; whether it declares that, and what the system does when a task needs a capability the backend lacks, is unspecified |
| **Cost attribution** | The SDK reports cost per session (`ResultMessage`, `max_budget_usd`). The system has a consumable token pool (`task_graph` spec §3.4) that expects a settled figure at completion. The mapping between them is a design-stage question, but the two have different notions of what a "run" is when a session is resumed |
| **Knowledge freshness** | §3.3 requires official references, and handoff spec §11 notes that knowledge handoffs outlive runs. Nothing detects that a referenced document has changed, which is precisely the staleness the requirement exists to prevent |
| **Subagent visibility** | §4.1 makes internal structure invisible by design. The SDK exposes it — `SubagentStart`, `SubagentStop`, `forward_subagent_text`, and `parent_tool_use_id` on session messages. Whether the system should stay blind or record a summary is a genuine design choice, not an oversight |
