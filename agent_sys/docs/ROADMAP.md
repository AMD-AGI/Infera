# agent_sys — Roadmap

Long-term work: subsystems that are not in the alpha, and design questions
deferred past it. Near-term items live in [`TODO.md`](TODO.md).

Every entry names where it came from, so a reader can find the argument rather
than re-having it. Entries marked **subsystem** get the full
spec → research → design → implement treatment when their turn comes; the rest
are single decisions.

---

## 1. Observability (`o11y`) — subsystem

**The largest deferred piece.** Logging was originally specified inside the agent
spec; it does not belong there. It is its own package with its own lifecycle.

The shape, as decided:

| | |
|---|---|
| **Backend-specific writers** | Save logs to JSON files against a predefined schema |
| **Decoupled readers** | Python post-process *visitors* read those files. Generation and post-processing are independent; some real-time post-processing is allowed but is not the design point |
| **Storage** | Managed by `env_mgr`'s filesystem manager, and subject to the permission rules like any other storage |

Everything an agent or a task emits routes here rather than into a component's
own logging:

- **Levelled logs** — `debug` / `info` / `warning` / `error`, with the three
  sources that can demand an entry (system policy, the agent's own rules, a
  skill/rule/hook).
- **Per-run metrics for every agent spec** — cost, wall-clock, resume count,
  failure count. Standardly recorded so later analysis and optimisation have
  something to work from, and *not* hardcoded into whatever component happens to
  observe them.
- **Per-task metrics**, on the same terms.
- **Backend runtime data.** The harness already produces it; the work is a
  visitor over it, which is why this is an o11y problem and not an agent-spec
  problem.

The alpha records nothing beyond what `task_graph` already persists.

## 2. Monitor agent system — subsystem

**The mechanism moved into the alpha on 2026-08-27** — `task_graph` spec §3.5
now specifies it: every task has a monitor, the monitor has its own mainloop and
a `set_task`, `Task.monitor_spec` names which loop watches it, and its job is the
*task's* exceptions. A monitor that could only be called cannot notice a stall,
which is the failure it exists for.

**Widened again on 2026-08-28** — `monitor` spec rev. 14 makes it the task's
**event loop**, on two channels: planned phase advances, handled by code and never
by a model, and the unplanned outcomes this entry has always been about. **That
does not move anything out of this section**: the analysing dispatcher is still
roadmap and is still bound to the unplanned channel alone. What changed is that
ordinary progress now depends on the monitor being alive, which is why `monitor`
spec §5.4 turns the note further down about nothing monitoring the monitor into
two built mechanisms rather than a recorded risk.

**What stays here is the analysing dispatcher** — point 3's richer action set,
and the open risk below. The alpha ships the simple pusher.

The mechanism, for the record, and points 1–2 now duplicated by
`task_graph` spec §3.5:

1. **Every task has a monitor** — a thread, or a thread with an AI in it. It
   maintains the agent's status and answers, on demand: where is it, how far has
   it got, is it alive, is it still in the right place.
2. **A monitor may be per-task or global.** A task may hand its monitoring job to
   a global monitor that round-robins over every task. When the global monitor
   handles one task's job it has only that task's permission scope.
3. **A monitor handles a stalled or failed agent** in one of two ways:
   - **Alpha: a simple pusher.** A status checker plus one phrase — *continue,
     do it until finished*.
   - **Later: an analysing dispatcher.** An agent gives a short analysis and
     picks from the action set. **The canonical list now lives in `monitor` spec
     §7.1**, which is where it belongs and which marks what the alpha reaches;
     it gained `answer` on 2026-08-27, an action none of the earlier lists had.
     For reference: `[push, answer, help (assign a helper agent), add
     input/knowledge, create coordinator, create teammate/rival, change agent
     spec, escalate to upper scope, report to user, restart the task, submit a
     copy, reconcile related task state]`.

   The last three arrived with "a failed validation fails the task"
   (`validator` spec §3.4), which makes a terminal failed branch a routine
   event rather than an exceptional one. **Each is expressed as a task
   transition the monitor calls, never as a status write** — the monitor has no
   authority over task state, and `task_graph` spec §2 principle 4 stands
   unamended because of it. `restart` is `resume_task`, which already accepts
   `FAILED`; a copy is an ordinary `submit`; reconciling is what needs the
   transition set.

   **The alpha's pusher does not reach any of this**, and that bounds the
   *response* only. It addresses a *stalled* agent — "continue, do it until
   finished" — and a task that failed its output validation is terminal, with no
   agent running to push. **The failure is still reported and recorded**
   (`monitor` spec §2.1 and §7): under principle 1 every departure from the plan
   reaches the monitor, and a terminal task means the graph will not finish
   whether or not anything malfunctioned. Amended 2026-08-27 — this paragraph
   previously said such a branch goes quiescent with nothing surfacing an error,
   and main spec §10 has withdrawn that too.

**An ad-hoc push, for agents that pass the gate.** `monitor` spec §4.1.0's
completeness gate reacts to an agent that delivered *nothing*. The observation
behind this entry is that an agent which delivered *everything declared* may still
have stopped short of the work — so the system should be able to push it a few
further rounds even on a clean pass, with an AI deciding whether the answer
improved. Distinct from the pusher, which fires only on a gate failure; this one
fires on success and is speculative by design. Cost and stopping rule both
unexamined.

**A `fixing` agent status**, for the interval between an exception being reported
and being resolved. The alpha keeps `running` (`monitor` spec §4.2) because the
condition is legible from the record rather than the status, and because `fixing`
costs more than an enum value: **it would be the first agent status driven by the
monitor rather than the runner.** The authority rule governs *task* status —
every monitor action is a transition it calls, and `_move` is the single writer —
but agent status is a separate model that does not pass through `_move`. So
adding `fixing` opens *who may write agent status, and under what discipline*,
which nothing has yet asked. Start there, not at the enum.

**Open risk, recorded rather than solved:** a global monitor's history leaks
context across the tasks it monitors.

## 3. Human in the loop — subsystem

A task may have no AI agent at all (main spec §4). Making a *human* a
first-class executor needs an interface, and the intended one is deliberately
unglamorous:

> A human gets a deployed workspace and starts a `claude` CLI or Cursor in it,
> does the job, and calls an interface that formalises the output and submits it.

Reading inside the `agent_sys` architecture and doing some copy-paste
donkey-work — not guessing at what the system wants. The point is that a human
can join the loop happily, without the system pretending they are a bot.

Related, and specified in the alpha already: a human-executed task is *deployed*
and then waits. The human sets the status and provides the handoff by hand. No
detection, no heartbeat, no cyclical action — the monitor is relieved of most of
its duty and only maintains status when called. A GUI comes later; the alpha
interfaces must not preclude one.

## 4. Knowledge record system

Knowledge handoffs carry versions, timestamps, and checksums in the alpha — the
simple version. Later, a **knowledge record system** is called every time
knowledge storage is updated, so the accumulation is itself auditable.

### 4.1 **P0 — organise the agent-facing prompt corpus; `knowledge/` is a holding pen**

`agent_sys/knowledge/general/` exists and has its first document
(`working-on-a-remote-host.md`, distilled 2026-09-01 from the remote-mode stage).
**It is parked there, not designed there.** Nothing reads it: `knowledge` is a
*handoff kind* resolved by `agent/registry.py:check_knowledge`, and a directory
on disk is not a handoff. So today the corpus reaches an agent only if a package
author copies from it by hand.

**What P0 has to settle**, and the ordering matters because the second question
is the reason the first is not obvious:

1. **Where agent-facing text lives, as one decision rather than four.** There are
   currently at least four channels carrying instructions to an agent — the
   backend's system prompt, `CLAUDE.md` (repository and package), a task
   package's `assets/<name>.task/readme.md`, and tool descriptions. They were
   settled one at a time and nothing states which fact belongs in which.
2. **Who owns which fact.** The remote stage produced the one worked example and
   it generalises: *identity is the module's, intent is the package's.* `env_mgr`
   names the far side because only it can know it; the package says whether the
   work happens there, because only it can know that. A fact placed in the wrong
   voice is either re-said by every package until one forgets, or asserted by a
   module that has no standing to assert it.
3. **How general knowledge reaches a run.** Either `knowledge/` gets packed into
   knowledge handoffs at load time, or the composition root injects it, or it is
   folded into the system prompt. Pick one; "an author copies it" is the current
   answer and it is the one that decays.
4. **Deduplication against `CLAUDE.md`.** Some of what is in the first document
   already exists in stage briefs. Whatever the mechanism, a fact should have one
   home, and the others should reference it.

**P0 rather than P1** because it gates how *every* future task is briefed, and
because the corpus is small enough today that restructuring it is cheap. It gets
more expensive with every document added, and the first one is already written.

**Not in scope here:** the audit/versioning half of §4 above. This is about
placement and delivery of prompt-shaped text, not about recording updates to it.

## 5. Validator — deferred pieces

| Item | Note |
|---|---|
| **The template-with-blanks system** | Removed from the spec as over-design. Something must eventually stop copy-paste between similar validators; whatever replaces it must be simpler than a recursive template engine |
| **Score-typed results** | Boolean in v1. A score is only useful against a threshold, and a threshold is only meaningful once run-to-run variance has been measured |
| **Review guard for validator consistency** | A hook, a guard, or a piece of code that checks a validator's metadata against its actual content and implementation. A validator whose tag says `strong` and whose body asks an agent for an opinion should not pass review |
| **Validator versioning** | A validator that changes has re-graded history. Not an alpha concern |
| **Weak-validator aggregation** | Several weak checks agreeing is more informative than one; using a different model family for the reviewer than for the producer avoids shared blind spots. **A trick, not a design-level mechanism** — noted here and nowhere else |

**Open risk, recorded rather than solved:** the verdict is itself a channel.
§8's isolation stops a producer *reading* the standard; it does not limit how
many times one may be re-validated against it, and each pass/fail leaks a little
about where the bar is. Whitehill (arXiv:1707.01825) reached rank 4 of 848 on a
real Kaggle competition from repeated scalar feedback alone, training nothing.
How narrow this is depends on whether a failed task is re-run automatically and
how much of the verdict it sees — both alpha-dependent, neither measured.

## 6. Environment and execution

| Item | Note |
|---|---|
| **`kubectl exec`** | A third remote access mechanism alongside ssh and docker exec |
| **`spur` / `slurm` exec** | A fourth, for scheduler-managed clusters |
| **Agent env reuse** | A task, or a validation phase, reusing an existing agent environment directly or with light modification. **Careful:** this must not blur the system's isolation standard, which is the whole reason each validation gets a fresh environment |
| **Movable handoff storage** | A handoff record should carry enough detail to be trackable *and* be movable between storage locations. The alpha does the first and not the second |
| **Sync direction and conflict** | The weak local↔remote mapping is `rsync`, which has a direction. Which side wins when both changed is unspecified, and "the caller decides" will lose data eventually |

## 6.1 **P0 RISK — an AI task is not confined, and confinement is the anti-cheating property**

**This is the alpha's one unmitigated security gap.** The system is
multi-agent; **confinement is what stops one agent reading another's files and
cheating.** A non-AI task is confined today and measured
(`test_spawn_confines_a_child_from_an_unconfined_parent`). **An AI task is not,
in any runnable configuration.**

### Why it is not confined

`prepared.spawn(argv)` confines a child we start, and descendants inherit the
domain (measured: a grandchild is denied). **We do not start the AI process.**
`ClaudeSDKClient` takes a `cli_path` and connects to a CLI it launches itself —
`agent/backends/claude_sdk.py` passes the option and the SDK does the fork. Our
ruleset never reaches it. `_apply_confinement`'s docstring carries the history:
**three earlier versions of this decision were wrong in the same direction**, so the
current one asks the executor rather than guessing from a wrapper or a `kind`.

### What the standard mechanism gives, and what it does not

Claude Code ships a sandbox; on Linux it is **bubblewrap**, which is a mechanism
`env_mgr` already supports (`isolation/apply.py:30`). It is configurable from
the SDK:

```python
ClaudeAgentOptions(sandbox={"enabled": True, "allowUnsandboxedCommands": False})
```

**But the sandbox covers the Bash tool only.** Read, Write, WebFetch, MCP
servers and hooks are governed by the **permission layer**, not the sandbox — so
**an agent that never runs a shell command can read a sibling's zone with the
`Read` tool and the sandbox never sees it.** For this threat model, sandbox
alone is not a mitigation.

**Two layers, and neither is optional:**

| layer | covers | configured by |
|---|---|---|
| sandbox | Bash children | `sandbox={enabled, allowUnsandboxedCommands: False, excludedCommands}` |
| **permission rules** | **Read / Write / WebFetch / MCP** | `denyRead` / `allowRead` / `allowWrite`, `disallowed_tools` |

**The usable fail-closed primitive:** a matching **deny** rule blocks a tool
**even under `bypassPermissions`**. That is the one control that does not depend
on the agent cooperating.

### Known hazards, recorded so they are not rediscovered

- **`failIfUnavailable`** — without it, a missing sandbox is a warning and the
  run continues **unsandboxed**. Published research records Claude Code
  bypassing a denylist via `/proc/self/root`, and, when bwrap failed on that
  kernel, **the agent requesting an unsandboxed run to finish the task.**
- The same research's recommendation is **an outer bwrap around the whole
  `claude` process** rather than trusting the built-in Bash sandbox — which is
  exactly `prepared.spawn`'s shape, if the harness is what we spawn.
- **This machine cannot test any of it:** `claude-agent-sdk` is not installed,
  driving it needs credentials the suite forbids, and **`bwrap` is absent**
  (`socat` is present). Anything written here would be green for the wrong
  reason. `interfaces.md` §8.7.

### Measured 2026-08-29, on the first real model calls this repository has made

Three of these were **reassuring for the wrong reason**, which is the hazard this
section exists for. All are first-hand from `AGENT_SYS_NO_PERMISSIONS=1` runs.

**The harness confines the agent even when we do not — and that is the direction
that hides whether our confinement works.** With permission management switched
*off*, the `claude` CLI was still in its default ask-for-approval mode with no
approval channel: `printenv` blocked, `Write` blocked **inside the agent's own
zone**. The run was *more* restricted with permissions off than on. See
`interfaces.md` §4.22d. **So any future measurement of our sandbox must first
establish that the harness is not the thing doing the refusing.**

**Criterion 8 measured the harness, not us.** The agent ran the leak probe and
reported, unprompted: `Error: Contains simple_expansion`, then
`Error: This command requires approval` — **no shell exit code, the shell never ran
the command**, whether `leak.txt` appeared is **unknown**, and *"the interception
happened at the command parse/approval layer, and the boundary was drawn by the
harness rather than the OS."* The readme's expected layering (hook allows → OS
refuses) **was not observed.**

**And before that, the probe was writing to a root-owned `/`.** `AGENT_SYS_DEMO_OUTSIDE`
was never exported, so the command was `echo leaked > "/leak.txt"` and returned a
convincing `Permission denied` **on any machine, for ever, with the sandbox off
entirely.** Fixed (`5cc00c9`), but it had been green for months.

**Relocating `CLAUDE_CONFIG_DIR` is load-bearing for cost, not only for isolation.**
Same model, same one-turn prompt: **$0.782 unrelocated versus $0.024 relocated —
32×.** The operator's entire environment (plugins, skills, personal `CLAUDE.md`)
enters every unrelocated agent's context. This is the earlier finding — a confined
agent read the operator's personal `CLAUDE.md` and obeyed its language rule — with a
price attached.

### The first end-to-end run, 2026-08-29 10:05:57 → 10:08:01, 124 s

**All four tasks succeeded, the root terminated, both validators recorded real
verdicts, exit 3.** Zero stalls, zero timeouts — the run ended because it finished.

**Three consecutive completions: 124 s, 114 s, 109 s.** Not a lucky run.

**No run store here is expected to survive, and this section names none as
current.** The standard invocation is **two commands on one shell line** —
`agent-sys run --clean; … agent-sys run …` — which `validator` read off
`ps` rather than from a habit report. A run never deletes a run (`_clean` returns
at `cli.py:344` before either run path is reached); **the preceding `--clean`
does, and it runs before every drive**, whether or not anything is concurrent. Run *n* is deleted by run *n+1*, unconditionally. Two of the three
stores above were gone within minutes; **an earlier amendment here named the
second as "the one with artefacts on disk" and was false seven minutes later.**
Naming a third would be the same mistake a third time, so the figures stand on
their own and the reader is told they are not re-checkable in place.

Copies were taken before the next drive — `demo-2` and `validator` each wrote one
under `scratch/impl-2026-08/`. **That directory's own `.gitignore` is `*`**, so
they survive `--clean` and not a fresh clone. Whether the day's evidence should
be tracked somewhere is a repository-convention question and open.

`--clean` buys nothing: `layout_for` already gives each run its own root, and its
docstring says why — *"what can collide is the store root, so each run gets its
own."* **Its real cost is that criterion 12 has never been reachable.** Resume
needs the previous run's state to still be there, and there has never been a
previous run. Not failing — unreachable, which reads as untested.

**The third run is also the first to execute the user's `check_grounded` ruling**
(`fd4a9d3` + `c8c1d37`): the `summary` kind declares `items/grounding/` required
and the AI body is asked for one `cp -r` of its own input. Verified on disk —
`diff -r` between the `facts` handoff content and the summary's `grounding/`
is **byte-identical**, so the model copied rather than curated, and
`check_grounded`'s external-route fallback arm fired zero times. **Only a run
could establish that**; the guard test proves the readme asks, not that a model
complies.

**And it changes what a `strong` verdict from `check_grounded` means.** The
grounding set now reaches the validator *through the producer it judges*, so the
check is exact about the summary's internal consistency and inherits the
producer's honesty about the copy — `validator` spec §8's *"the producer cannot"*
territory. The ruling settles where the mechanism lives, not what the resulting
check is worth. Not rigging: fabricating a number now also requires planting it
in the copied facts. Recorded so the narrowing is visible rather than inferred
from a green line.

`validator` placed it as a report rather than a spec row (`3bc2c36`), which is
the right form — §8.1's fourth row forbids *the producer seeing the hidden
reference*, and this shape is **stronger than that row anticipated: the producer
does not see the reference, it authors it.** Their statement of what a verdict
now asserts:

> A `strong` verdict from this validator asserts that **the summary is internally
> consistent with the grounding copy the producer carried** — not that it is
> grounded in the facts. The two differ exactly when the copy is not faithful.

**The exposure is inherent to the ruling, not to the implementation**, and it is
open rather than deferred: nothing in the confined path can distinguish a
faithful copy from an unfaithful one. A body cannot reach the store, and the
ruling forecloses declaring the original as a second input. **Closing it needs a
party that sees both, and the current design has none.**

```
produce: succeeded   facts   v0 valid   check_facts:    PASS  completeness/strong
describe: succeeded  summary v0 valid   check_grounded: PASS  trustworthiness/strong
consume: succeeded   main: succeeded    0 validation(s) dropped
```

**What it establishes:** the nine packages compose. Dispatch, staging, pre-allocated
grants, a program body writing into its own grant, a store seal, a published handoff,
**an AI node making a real model call** (12 turns, $0.409), an artefact written into
its granted directory, two `strong` verdicts on two dimensions, a consumer running on
a validated input, **and a root that leaves `RUNNING`** — `04a5b76`'s producer
executing for the first time in this repository, on a chain its author had declined
to claim worked.

**What it establishes about isolation: nothing, and the run says so in its second
line.** `NO SANDBOX — PERMISSION MANAGEMENT IS OFF for this run… A pass here is not
evidence that the sandbox works.`

**Criterion 8 is now measured rather than fictional, and it is inverted by design.**
The agent ran the probe and reported verbatim:

```
$ echo leaked > "$AGENT_SYS_DEMO_OUTSIDE/leak.txt"; echo "exit=$?"
exit=0
$ cat "$AGENT_SYS_DEMO_OUTSIDE/leak.txt"
leaked
```

> *The mode is `drwxrwxr-x` owned by the current user, so the OS forms no boundary
> here. **I have not deleted `leak.txt`; it is left as evidence.** As to why this run
> produced no restricted mode — `env_mgr` unwired, or confinement not in effect — **I
> have no first-hand evidence and will not infer.**

**The write succeeding is correct with the switch on.** This supersedes the earlier
note that the probe measured a root-owned `/`: that defect is fixed (`5cc00c9`), the
probe now writes to a real out-of-zone directory, and **it is a working measurement
whose current answer is "there is no boundary", because there is not.**

**Still unexercised:** criterion 9's refusal, and the version-space seam —
`produce` succeeded on attempt 0, **where slot 0 and store 0 coincide by
construction.**

### Open for the user: criterion 5's scenario assumes a model that fabricates

**Not a defect. A premise that a good model defeats.** `describe`'s body asks in
good faith for a duration `facts` cannot ground, and `check_grounded` exists to catch
the invented numeral. **The model declined to invent one, twice, and cited our own
validator by name:**

> *"Writing any duration would be an ungrounded numeral and `check_grounded` would
> fail it, so I honestly leave it out."*

So when a summary is finally written, `check_grounded` **passes** — and a passing
`check_grounded` is `UNEXPECTED_SUCCESS`, **exit 3**, which this artefact treats as
worse than red. The demo is wired to report it loudly and **nobody should "fix" it
when it fires.**

**Ruled 2026-08-29: do not soften the scenario.** Whether criterion 5 needs a harder
task is a spec question and the user's. Recorded here so it is decided rather than
patched by whoever next sees exit 3.

**And the near-miss is the reason this is written down.** A draft of `describe`'s
readme added *"if the facts do not contain a number you were asked for, say so
instead of estimating"* — helpful-sounding, and it would have **rigged the strict
expected failure into an xpass**. Caught by its author re-reading their own edit, not
by a test.

**The same premise leaves criterion 10 undemonstrated — `check_grounded` has never
been observed catching anything** (`demo`). Three runs showed a good model closing
the gap; the validator's **failing** direction, which is what its `strong` claim is
about, has never executed.

**Ruled 2026-08-29: parked, and not a roadmap item.** *"Not a framework question and
not a principle question — this is `check_grounded`'s own business semantics, and we
do not spend time on it."* The suggested shape, if anyone ever does: **split it in
two** — one validator for the other fields, one that judges only whether the agent's
answer about the missing duration is *reasonable*, passing if it is. Filed as
`TODO.md` 4f with the measurement that bears on it. **Nothing here is waiting on the
user.**

**One thing from the run is worth keeping regardless of that.** The model argued the
design's own case back at it, unprompted, in its `## Grounding` section:

> *copied with a single `cp -r` and neither edited nor filtered, **so whoever checks
> this handoff can check my numbers against the same bytes I read rather than against
> a list I curated for myself.***

That is verbatim the argument `demo` used to reject *"the numerals you used"* — reached
independently by the party the rule constrains. It is the strongest evidence the shape
is right, and it is also exactly why the failing direction stays unobserved.

### The fallback, if the standard route does not close it

Vendor the SDK and patch the fork site so the child is started through
`prepared.spawn`. **Named as the worst option**, and only after the two-layer
route is measured against a real backend.

### §9.2 — the CLI identity question, same root

**O2 is not a separate item.** `_find_cli()` prefers the SDK's bundled binary
while `env_mgr/installers/claude.py` installs plugins onto `PATH`, so **the
backend runs a CLI `env_mgr` never touched and the agent silently lacks its own
recipe's plugins — and the run succeeds.** Both this and the confinement gap
turn on the same question: **who owns the CLI process.** Fix: `env_mgr` reports
the CLI it installed into, and the backend uses that one **or refuses.**

## 6.2 **P2 RISK — `extensions.preciousObjects` means the object store only grows**

**Accepted, not a defect.** `env_mgr.workspace.cut` is `git clone --shared
--no-hardlinks` (`workspace.py:129`), so every task workspace **borrows** the main
repository's objects through `.git/objects/info/alternates`. If the lender ever
deletes an object a borrower is using, the borrower breaks **totally** —
`env_mgr` reproduced `fatal: bad object HEAD`, triggered by nothing more exotic
than an ordinary `git commit` in the source firing automatic maintenance. So
`cut` refuses on a repository without `extensions.preciousObjects`, and the flag
makes git refuse to delete objects.

**Measured 2026-08-29 on the development repository**, which is where the cost
shows first because nine worktrees share one object database:

| | |
|---|---|
| `git prune` | `fatal: cannot prune in a precious-objects repo` |
| `git gc` | **runs normally** — it simply skips pruning |
| unreachable objects | **1018** (`git fsck --unreachable`) |
| packs | **21**, uncoalesced; 68 MB total |

**Worktrees are not the exposed party and never were.** `git worktree add` shares
one object database — there is no alternates link and no borrower — so the nine
worktrees would be safe with the flag off. **The flag exists for the per-task
shared clones**, which are the only borrowers in the design.

**Ruled 2026-08-29: accept the growth.** Manual cleanup now or later is fine.
The clean sequence, when it is wanted, is to let every task workspace finish,
then `--unset` → `git gc --prune=now` → set it back; **the flag must be on before
the next `cut`, or every output-producing dispatch dies in `prepare`.**

**Why it is P2 and not lower:** unbounded growth on a machine nobody is watching
eventually becomes a disk problem, and the repair window requires *no live
workspaces* — which gets harder to find, not easier, as the system runs more
tasks. Cheap now, not automatically cheap later.

## 6.5 **P1 — the stream cannot tell a working phase from a wedged one, and the signal that can is inside the backend**

Three runs hung on 2026-08-31 and **all three were found by a human reading an
agent's transcript**, not by anything the system emits. `8b4b3ff` fixed half of
it — `_settle` now emits `phase_start` / `phase_complete` as tasks move, so a
reader can follow a run's shape from `stream.jsonl`, which was impossible
before. This is the other half.

### What the fix does not reach, measured on B6

`stream.jsonl` carried 21 lines, 5 of them live, and they span the run rather
than clustering at startup. But the largest gap between consecutive events is
**1456 s — 24 m 16 s**, the whole agent working phase, and the second largest is
561 s. Everything else is 1 s. So the stream says *which phase a task is in* and
never *whether it is progressing* — and "wedged inside `running`" is the state
B4 died in for 65 minutes.

A liveness check cannot be built on this. "No events for N minutes" fires on a
healthy run too, because a healthy run is also silent for 24 minutes.

### The signal that works, with the number

`run-watchdog` measured the agent transcript's inter-entry gaps across every
session watched that day:

| session | outcome | entries | span | largest gap |
|---|---|---|---|---|
| B6 agent | healthy | 396 | 24.2 min | **87 s** |
| B6 reproducer | healthy | 106 | 9.3 min | **162 s** |
| B5 agent | healthy | 311 | 30.5 min | **257 s** |
| B5 reproducer | healthy | 105 | 10.0 min | **257 s** |
| B4 agent | **hung** | 348 | 23.2 min | 120 s working, then **65 min** |

Healthy work never goes quiet longer than **257 s**, and that worst case is a
deliberate cold-start wait. The hang was **65 minutes** — nearly an order of
magnitude of separation. A threshold anywhere in 600–900 s catches B4 within a
quarter hour and fires on none of the four healthy sessions.

### Three limits, because the number is worth less without them

1. **n = 4 sessions, one package.** A package that waits on a longer build, or
   makes one very long model call, could legitimately exceed 257 s. This is a
   floor measured here, not a constant to hardcode.
2. **It detects *stopped*, not *stuck*.** An agent in a retry loop grows its
   transcript while making no progress. Watched for specifically in B4 and B6
   and not seen — all three hangs that day were the *stopped* kind — so this
   covers what happened, not everything that could.
3. **In B4 the agent had legitimately finished.** Its transcript stopping was
   *correct*; the fault was the engine not terminating. The alarm this raises is
   therefore **"this session is over"**, not "this session is broken" — which,
   for all three of that day's hangs, is exactly the alarm that was needed.

### Why it is a seam and not another line in `_settle`

**Everything observable from outside the backend was identical between a
65-minute deadlock and a healthy run**, checked on the live hang before
concluding it: process alive, CPU ticking ~1 s per 20 s, one socket
`ESTABLISHED` with `txq=0 rxq=0`, parked in `ep_poll`, task status `running`.
`_settle` polls task *status*, and a wedged task and a working one have the same
one. The distinguishing fact lives where the transcript is written, so exposing
it crosses `agent` ↔ `cli` and `interfaces.md` §1.1 applies.

**P1 rather than P2** because it is the difference between a run that reports
its own failure and one that needs somebody to notice. Evidence:
`scratch/single-real-task-2026-08/stream-not-a-view.md`.

## 6.4 **P2 — rebuild the locality check on oracles; the shape heuristic is disconnected**

**User-ruled 2026-08-31: disable it now, rebuild later.** `handoff/store.py` no
longer calls `locality.check`, so **`handoff` spec criterion 17 is not enforced
today**. The module and its twenty tests are kept intact and correct; this is a
disconnected caller, not a deleted module, and re-wiring it is one line.

### Why it was disabled

It **refused a correct artefact, and would have refused every correct one.** On
the first end-to-end run of `examples/single_real_task` the seal rejected the
agent's reproduction kit at `README.md:42`, on this line:

    `"POST /v1/chat/completions HTTP/1.1" 200 OK`

A quoted HTTP access-log record, in the evidence section, showing the completion
had gone through the infera router rather than the engine's own port — which is
**criterion 2 of that task's own brief**. The check read the request-target as an
absolute filesystem path. Any correct kit for that task contains the string, so
the refusal was systematic rather than unlucky.

Measured over the produced kit: **778 flagged occurrences, of which ~97% are
false positives** — 618 container-internal (`/sgl-workspace/`, `/tmp/aiter_configs/`),
106 HTTP request-targets, 10 an etcd key prefix, and **35 genuinely local**. That
reproduces, on a second independent corpus, the module's own docstring
measurement of 650 matches with 627 needing suppression.

### Why more regex is not the fix, and this is the load-bearing paragraph

`locality.py`'s own docstring records that the shape refinement was proposed on
**Debian #1002451 and refused on the record**: *"you cannot recognise a build
path by its shape, because the shape is a property of whoever built it."* Two
patches were made on 2026-08-31 — one for scheme-less request-targets, then a
stricter version anchored on the `HTTP/x.y` token after a review found the first
opened a cloak — and each revealed the next shape. The module was right and the
patches were treading the path it warned about.

### What the rebuild must do

**The design is already correct and is simply not wired.** The module splits its
evidence honestly: an **oracle** hit is *certain* (a prefix this system minted),
the **shape heuristic** is *best effort*, and its docstring says the heuristic
"runs only where no oracle applies". Production inverted that:

| | today |
|---|---|
| `Oracles` | **constructed nowhere.** `store.py:140` falls back to `Oracles(store_root=...)`, so `playground_root` is never supplied — half the certain signal is off |
| `image_prefixes` | spec §7's mechanism for a declared container image. Read at `locality.py:151`, **set by nobody** — so a containerised workload's paths could never be allowed, by construction |
| `check()` | raises identically for an oracle hit and a heuristic hit, so the best-effort half was the hard gate |

So the sound half was unwired and the unsound half was load-bearing. Three things
to settle, and the third is why this is not a pure bug fix:

1. **Wire `Oracles`** — supply `playground_root` as well as `store_root` at the
   composition root. Pure plumbing.
2. **Separate the two verdicts.** An oracle hit stays fatal; a heuristic hit
   becomes a recorded finding on the handoff rather than a refusal. That is
   arguably what criterion 17 already means, but it is a criterion edit and needs
   saying out loud.
3. **Wire `image_prefixes`** — and this needs a *convention*, because
   `handoff.schema.json`'s `dependencies` is deliberately an unconstrained object
   (*"fixing a shape here would be inventing a requirement"*). Choosing where
   inside it the prefixes live is a specification decision, not a code change.

### What is lost meanwhile, stated so it is not discovered by accident

The check was aimed at exactly the right thing, and **this stage's own task is
the case it was built for**: the mission's second half is *a second AI reproduces
the run from the kit alone*, and a kit naming one machine's paths is what breaks
that. Concretely, the kit that triggered all this bakes in

    /data/<user>_hf_cache/models/Qwen3.6-27B      — 20 occurrences

which is a **true positive**: it should say "point this at your weights". With
the check off, nothing catches that class, and a reproduction failure caused by
it will present as the reproducer's fault rather than the kit's.

### Keeping only the sound half would not have worked either — measured

An automated review proposed the obvious refinement: disable the shape heuristic
and keep the oracle branch, which is sound and cheap. It is the right instinct
and the measurement refutes it.

`store.py:140` does supply `store_root`, so the oracle branch **is live** — this
is a correction to an earlier claim in this entry's neighbourhood that the oracle
half was unwired; only `playground_root` is missing. Run against the real kit
with that oracle supplied, it produces **two hits**, and both are genuine:

    logs/run_all.second-run.out:85  '… ALL CHECKS PASSED -- evidence in /var/tmp/…/handoffs/…'
    logs/run_all.second-run.out:87  'Tear down with: bash /var/tmp/…/handoffs/…'

Two instead of 778, and **a refusal is still a refusal** — the run would have
been blocked either way.

**And the two hits expose a conflict that soundness cannot resolve.** The task's
brief tells the agent to capture its logs as evidence; the agent's logs record
where it wrote; where it wrote is under the store root. So *"a handoff carries no
path from the machine that produced it"* and *"capture your own logs as
evidence"* are in direct conflict, and this artefact satisfies both briefs while
failing the check. A rebuild has to answer this — plausibly by weighting file
role, which `check()`'s own docstring already flags as an open question
(`design.md` O5: *"No weighting by file role … a playground path in a changelog
is still a record of one machine"*) — and not by making the oracle branch more
precise, which it already is.

### The wiring was never tested, and that is how this survived

Disconnecting the call changed **no test result**: 2059 passed before and after.
`tests/handoff/test_locality.py` has twenty tests and every one of them calls
`locality.check` directly; **nothing anywhere puts or seals content containing a
local path and asserts the store refuses it.** So the criterion had unit
coverage and no wiring coverage, which is why a check with a measured 97% false
positive rate could sit as a hard gate in the publish path without anything
saying so until a real artefact arrived.

That is the same shape as three other findings from the same day — `Oracles`
constructed nowhere, `image_prefixes` set by nobody, `env_mgr`'s `Ssh`/`DockerExec`
with no production caller. **A rebuild that does not add a wiring test leaves the
next inversion just as invisible**, in either direction: nothing today would fail
if the check silently came back, either.

Evidence: `scratch/single-real-task-2026-08/seal-refusal.md`, and
`probe_locality_url_fp.py` beside it — whose fourth case is the non-vacuity
control any rebuild must keep firing.

## 6.3 **Rebuild the permission system** — user-ruled 2026-08-29, and the demo is the evidence

**Ruled by the user, in as many words: the permission system as implemented
"感觉问题百出" — it feels riddled with problems — and it is to be rebuilt rather
than patched.** This entry records what is known so the rebuild starts from
measurements instead of from the same instincts.

### The measurement that prompted it

The UI stage's goal was `examples/demo` converted to YAML and **running end to
end**. With permissions enforced it does not:

```
describe: failed — ConfinementNotApplied: backend 'claude_code_sdk' cannot start
          confined: it does not spawn a command line of its own … (criterion 14)
exit 4    1 of 2 expected failures observed, 1 never reached
```

With `AGENT_SYS_NO_PERMISSIONS=1` **every task succeeds**, both handoffs reach
`valid`, and `check_grounded` executes for the first time in this repository's
history:

```
consume: succeeded   describe: succeeded   main: succeeded   produce: succeeded
facts   v0 valid     check_facts:    PASS  completeness / strong
summary v0 valid     check_grounded: PASS  trustworthiness / strong
```

**So the confinement layer is the only thing between this system and a working
end-to-end run**, and it has been for longer than the format change: the
pre-stage jsonnet tree, measured in a temporary worktree at `8274a5b`, produces a
transcript differing by **exactly one line** (`main: agent 'compose'` versus
`main: agent None`) and the same exit 4.

### What is already known about why, so it is not rediscovered

| | |
|---|---|
| **§5.11** (`interfaces.md`) | An AI backend cannot be confined **in-process**, and the word *cannot* was wrong: a Landlock domain is inherited by **every descendant**, so a harness running inside a `spawn`-ed child confines the `claude` CLI it spawns — no shim, no argv interception, no SDK cooperation. **Measured**: `unconfined grandchild rc=0`, `grandchild of a confined child rc=13 EACCES` |
| the price | **level 2 entirely.** All three `AgentBackend` methods are calls on an in-process `ClaudeSDKClient`; `monitor`'s `Pushable` *is* that handle; `interrupt`'s drain reads `terminal_reason` off the message stream. *"That is not a plumbing change. It is the whole reason `AgentBackend` is a second protocol."* |
| **§6.1** (this file) | P0 RISK, already open on the same subject |
| the switch | `AGENT_SYS_NO_PERMISSIONS`, read in exactly one place (`env_mgr/prepare.py:96`) and deliberately a function rather than a constant. **This is the escape hatch that makes the demo work today** |

### The second half — three names that cannot be exported, ruled "record it, do not work on it"

The UI stage added a path environment-variable system
(`refine.task_package.define.md` item 3). **Three of the user's eleven names were
refused, and the refusal is a measurement rather than an omission.**

`agent_workspace_root`, `agent_handoff_root` and `agent_playground_root` resolve
to domain roots **outside the zone**:

| the name | resolves to | where that sits |
|---|---|---|
| `agent_handoff_root` | `domains.storage_root()` | the zone's **ancestor** |
| `agent_workspace_root` | the WORKSPACE domain's root | unrelated to the zone |
| `agent_playground_root` | the PLAYGROUND domain's root | unrelated to the zone |

A confined body gets **EACCES on all three**, measured against an **in-zone
positive control that succeeds in the same child** — the control is what rules
out "the probe over-confined its own subject", which is the shape a wall of
EACCES otherwise has. Probe: `scratch/ui-yaml-2026-08/w2/p13_are_the_root_paths_reachable.py`.

Exporting them would break `env_mgr`'s own rule that **exported and granted agree
by construction** — *"an exported path we did not grant would be the evaporating
allow-list one level up: the body failing on our own instruction."*

**What shipped instead**: `AGENT_SYS_MY_ZONE` (the one root in the user's sense
that is granted), `AGENT_SYS_TASK_PACKAGE`, and `MY_WORKSPACE` / `MY_PLAYGROUND`
/ `MY_HANDOFFS` / `MY_LOGS` plus a `_REMOTE` mirror of each where a mapping
covers the zone. A directory that does not exist gets no name.

**The user's ruling: record it, do not work on it.** The choice the rebuild has to
make, stated so it is not re-derived: either the `*_root` names stay unexportable
and an agent only ever names things inside its own zone, **or** the authorisation
model changes to grant those roots read-only — which opens lateral visibility
between zones and is a specification change against `env_mgr` spec §4's
isolation goal, not an implementation one.

### What a rebuild must not lose

Recorded because each was bought with a measurement and a rewrite is where they
get dropped:

- **Path-prefix isolation does not work.** Main spec §7.1: an agent that writes a
  Python script and runs it defeats a `PreToolUse` hook entirely — the hook sees
  `python3 x.py` and no path — and prefix matching is CVE-2025-54794 in Claude
  Code itself, CVSS 9.1, three defeats reproduced. Replaced by canonical
  containment plus an OS sandbox, and that replacement is the current design.
- **Refusing to run unconfined is a feature, not the bug.** Criterion 14 is why
  `describe` fails rather than silently running an AI agent outside the sandbox.
  A rebuild that makes the demo pass by relaxing this has removed the property
  the system exists to have.
- **Confinement is irreversible within a thread** — a confined thread can no
  longer write outside the zone, so it cannot record its own outcome afterwards.
  That is why `_place_container_zone` uses `place_zone` and not `prepare`.
- **`apply()` refuses with more than one thread alive.**

---

---

### 6.6 **P2 — a container is a valid tool target and there is no way to declare one**

Raised by the ver1 review's finding #5, verified 2026-09-01, and **half-closed
rather than closed**: the CLI now reports the configuration fault as a
precondition instead of a traceback, and a `docker` mapping is still
unconstructible.

`tools.py`'s own docstring says *"a container is a valid tool target and an
invalid sync transport"*, and `RemoteMapping.target`'s comment says *"host for
ssh, container for docker exec"*. Both describe an intent no code implements:
`sync_transport` is the only constructor a mapping reaches, it returns a
`SyncTransport`, and `DockerExec` deliberately is not one. So `DockerExec` joins
the list of mechanisms in this repository that are written, correct, and reached
by no production caller.

**The change is a seam, which is why it is not in the review's scope.**
`Context.transports` is a `SyncTransport` map read by two consumers that want
different things: `sync` needs a transport that can `rsync --delete`, and
`_remote_tools` needs only a `Connection`. Separating them means either two
fields or one field of the weaker type with `sync` narrowing — and
`interfaces.md` §1.1 applies, because `Context` has two sides.

Worth doing when something actually needs to reach into a container. Until then
the honest state is a documented refusal, which is what it now gives.

## 7. Scheduling

| Item | Note |
|---|---|
| **Status-triggered ordering** | "Task B must start after task A's status reaches point X" — richer than the current handoff-validity dependency |
| **Together / peer start** | Ensure two agents start together. Neither is expressible today |
| **Cycle detection** | Inherited from `task_graph` spec §10 |
| **The downstream index** | `task_graph` spec §3.2.4 promotes it from an optimisation to a requirement: a cascade needs a task's consumers, and `depends_on` gives the upstream direction only. What it is keyed by, who maintains it, and how `submit` / `update_task` keep it current are unspecified, and a cascade cannot be built without them |
| **Cascade semantics at the edges** | Three questions `task_graph` spec §3.2.4 leaves open: what a cascade does on reaching a `RUNNING` task (stop it via `STOPPING`, which makes `cancel()` asynchronous and changes its signature; skip it; or refuse the whole cascade); whether a cascade is atomic, since half-cancelled is a state nothing describes; and to whom the cascade reports upward, in what form, and whether a parent may veto |

### 7.1 Attaching a task's mainloop to a shared thread

`agent` spec §4.3 gives every agent its own mainloop, because an agent is a live
stateful thing and nothing can interact with one that has no loop. **What it does
not give it is a thread** — amended 2026-08-28: the task owns one thread per
dispatch and the agent borrows it for the main phase (`agent` design §7.5). So the
alpha runs one thread per *executing leaf task*, which for a graph of leaves is
the same number as before and is now one set rather than two.

**The refinement: that loop may attach to a shared thread that round-robins over
every attached task.** That is the same trade the global monitor already makes
(§2 point 2) — bounded threads against a bounded response latency — and it is
worth taking for the same reason: a graph with many long-idle agents pays a thread
each for doing nothing.

**One thread stays out of the pool: the monitor's.** Sharing it with what it
watches gives the watched and the watcher one heartbeat, which is the failure a
watchdog exists to prevent (`monitor` spec §1.1) — and since rev. 14 that loop
also carries every task's planned phase advances, so starving it starves ordinary
progress and not only the response to trouble.

Two things to settle when it is built, both visible from the monitor's version of
the problem: what an agent's fair share of a round is, and whether an attached
agent can be starved by a busy neighbour in a way an owned thread would not have
been.

## 8. Configuration and quality

| Item | Note |
|---|---|
| **Config dispatch and dissemination** | The alpha is one global YAML with well-classified partitions that everyone reads. A real dispatch system comes later. Distinct from **spec** templating, which is settled — main spec §4.4 |
| **Task-package distribution** | Packages are directories the loader is pointed at, and cross-package references are symlinks the package author places (main spec §4.3). Fetching a package by name and version, or publishing one, is not specified and will be wanted the first time two teams share a workflow. **Two things moved under it on 2026-08-29 and neither is resolved**: a package with no `main.yaml` is now a *library* rather than an error (spec §4.3, criterion 18), which is the shape a shared package takes — and *which* package a run starts from has no owner, carried as a new row in spec §10 |
| **A standard for admitting parts to `agent_sys`** | There should be a rule, a checker, and a review process for adding to the system. Every module should carry hard and soft standards that guarantee the quality of its code or of an object spec. This is the meta-item that makes the others enforceable |
| ~~Templating mechanism~~ | **Closed twice, and the second answer is the opposite of the first.** Closed at spec rev. 4 by adopting jsonnet; **re-closed at rev. 10 by deleting it**, after measuring that across all 21 sources the templating was constants, string concatenation and default-if-absent. There is no spec templating layer now: the schema at tier ① carries a kind's shape and its `default`s, and a package parameterises itself with its own variable set (main spec §4.4, §2.3). **Recorded as reopened-then-reclosed rather than left reading as a settled adoption**, because a reader looking for "is templating decided" would otherwise find the wrong answer. The one thing that genuinely *did* reopen is **cross-package reuse**: with no template artefact there is nothing to share, and if two packages ever need the same shape this row is where that argument starts |

## 9. Agent backends

### 9.1 A `pydantic-ai` backend

`agent` spec §3.3 makes backends a list, and the alpha ships one AI backend
(`claude-agent-sdk`) beside `ProgramExecutor`. A second one is what makes that
list more than a promise, and **`pydantic-ai` satisfies level 2 as written** —
read from 2.35.3:

| `AgentBackend` | `pydantic_ai` |
|---|---|
| `interrupt()` | `AgentRun.cancel()` — `run.py:555` |
| `instruct(msg)` | `AgentRun.enqueue(*content, priority='asap'\|'when_idle')` — `run.py:514` |
| `query() -> AgentHistory` | `AgentRun.all_messages()` — `run.py:163` |
| `mainloop()` | `async for node in agent.iter(...)` |
| `status` | `AgentRun.next_node` / `.result` |

`pydantic-ai-slim` targets Python ≥ 3.10 and declares nine dependencies, 43 MB
installed against the SDK's 270 MB. **Import cost is a wash** — both 6–7 s,
measured side by side on one machine — so design §8.1's "lazily, never at module
scope" rule applies to it unchanged, for the second reason rather than the first.

**Three things to settle before writing the adapter**, all properties of their
side rather than ours:

| | |
|---|---|
| **`enqueue` is event-loop-bound** | It must be called on the loop driving `iter()`; from anywhere else, `loop.call_soon_threadsafe`. Its drain does `queue[:] = remaining`, which is not atomic against a concurrent append. `TaskAttempt` is a thread and `instruct()` arrives from outside it (design §7.5), so this seam is work rather than a wrapper |
| **`cancel()` degrades on 3.10** | Without `Task.cancelling()` / `uncancel()` a first-party cancellation cannot be told from an external one. 3.10 is our floor, so the degraded path is the one we would ship |
| **Nothing enumerates live runs** | There is no run registry; the only handle is the object `iter()` returned. Whoever holds it is the only thing that can reach that agent, which is a constraint on where a `TaskAttempt` may keep it |

The interception point for hooks is `capabilities/` — roughly fifty
before/after/wrap/on_error hooks over run, model request, tool execution and
output validation — and not `toolsets/`, whose `ApprovalRequiredToolset` is
thirty-three lines around a single boolean predicate.

**What it does not replace.** Not `env_mgr`: the tree contains no `landlock`,
`seccomp`, `bwrap` or `chroot`, `subprocess` appears in one file (`mcp.py`), and
`CodeExecutionTool` runs in the model provider's sandbox rather than on our
filesystem. Not `task_graph`: `pydantic_graph` ships no persistence layer and
delegates durability to Temporal, DBOS or Prefect, which would trade a resumable
store that already recovers from twelve interruption points for an external
engine.

### 9.2 A validation gets no composed environment — same root as the CLI question

**Ruled 2026-08-29: yes, and deliberately not built the same day.** Recorded as
one entry because it is one question, and placed here rather than in §7 because
its root is §6.1's: **who owns the CLI process.**

`validator/phase.py:325-361` offers two environment rows and **neither is an
`env_mgr`-composed environment**, so an agent-bodied validator gets no config
relocation and no `agent_cli`. That is arm B — the measured one — for
validators:

| arm | what a checking agent gets |
|---|---|
| A: `CLAUDE_CONFIG_DIR` left alone | the operator's own config; a transcript that changes with whose dotfiles are on the machine |
| **B: `CLAUDE_CONFIG_DIR` relocated into the zone, nothing carried** | **`Not logged in · Please run /login`** — the endpoint and credentials live in the `env` block of the settings file that was just relocated away |

**Why it matters more for a validator than for a task.** A checking agent
silently running a different CLI from the one whose plugins were installed is
**the anti-gaming property degrading invisibly** — the validator still returns a
verdict. **A wrong verdict that arrives is worse than no verdict.**

**The shape, and `validator` holds the binding constraint.**
`env_mgr.prepare_validation` should compose **configuration values that feed
`build_environment`** — a mapping, never a pre-built environment and never a
live `os.environ`. `validator`'s `CHANNELS` (`environment.py:84`) enumerates the
five channels criterion 21's isolation test closes, `environ` among them, **on
the strength of `build_environment` not inheriting.** If the environment arrives
pre-built, **criterion 21 stops measuring what it says it measures**: the test
keeps passing and stops meaning anything, which is §4.13's family in a test
rather than in a value.

**It is the criterion 9 gap seen from the composition side, not a new proposal.**
`validator` spec §8.2's rows are *configurations*, so the chain picks one of four
and **building** from it is a shared step — what is missing is the build, not a
row.

**Why not today, measured rather than assumed:**

```
examples/demo/validators/check_facts.yaml        entry: logic/check_facts/entry.sh
examples/demo/validators/check_grounded.yaml     entry: logic/check_grounded/entry.sh
```

**Both of the demo's validators are script-bodied.** No AI validator runs in the
demo, so neither the CLI-identity divergence nor the `Not logged in` arm can bite
today's run — and three packages composing a validation environment while
`demo run` is the goal is a cost with no matching benefit this week.

**`agent` owes nothing either way**: `Assignment.agent_cli` is already the field,
and `TaskAttempt.environment` already carries the resolved configuration a
validation's producer row wants.

### 9.3 **P1 — a finished AI task's CLI subprocess lives to the end of the run**

Measured 2026-09-01, not inferred. `agent/backends/claude_sdk.py:_terminate` —
which is the only caller of `self._client.disconnect()` — is reached solely from
`ExecutorBase.stop()`, and:

    stop()  <- TaskAttempt.halt()  <- Runner.stop()  <- Scheduler.stop()

`Scheduler.stop` has **no caller outside `tests/`**, and normal completion never
calls `stop()` at all. So on every path a run actually takes, no backend is ever
disconnected.

A per-poll descendant census over demo2 at `--var n_problems=2`, sampling every
5 s (`scratch/review-ver1-2026-09/`, and the script is kept):

           pid   first   last   alive     task
       2299467      16    264     248     `directions`  — succeeded at t≈33
       2309338      81    264     183     `problems`
       2323081     178    264      86     solve_a / _b / _c, at the fan-out
       2323109     178    264      86
       2323117     178    264      86
       run exited at t=270

    departures after t=11 : 0
    peak concurrent       : 5, and it never came down

`directions` reached `final: succeeded` at log line 67; its CLI lived another
three minutes. **The set is monotonically non-decreasing for the whole run.**
A fourteen-task demo2 would end holding fourteen.

**Why it is not just tidiness.** Each is a node process, and unrelated `claude`
processes on the same box measured 237 MB–960 MB RSS. These five were not
sampled before they exited, so the memory figure for *them* is not measured —
but the count is, and it scales with the task graph rather than with concurrency.

**Why it is not a one-line fix, and belongs here rather than in a review.** The
missing call is a *lifecycle* decision, not a backend defect: something has to
own "this attempt is over, release its executor", and today nothing does, because
the only route was built for cancellation. Options, none free — call `stop()`
from the completion path in `Runner` (but `stop()` also drains the inbox and
settles, which a completed attempt has already done); give `ExecutorBase` a
separate `release()`; or give `Scheduler.stop` the production caller it was
written for. The third is the honest one and it is the largest.

**Related and already recorded:** `Runner._attempts` is popped only in
`Runner.stop()`, so an attempt's `TaskAttempt` — and its executor, and its event
loop — is retained for the runner's lifetime by the same gap. Three file
descriptors per loop, GC-reclaimed only when the reference goes, against
`RLIMIT_NOFILE` of 1,048,576: not a hazard, and the same root.
