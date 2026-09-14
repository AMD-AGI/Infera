# Monitor — Specification

| | |
|---|---|
| Status | Draft |
| Revision | **14** — 2026-08-28. The monitor becomes the task's event loop: two channels, one planned and one not. **Full history: §12** |
| Date | 2026-08-28 |
| Scope | Every event in a task's life that is not the task's own work: the planned phase advances, every unplanned outcome, who reports each, the loop that handles both, the escalation chain, and where it is recorded |
| Source | `image.how.to.usedb.yuser.md` §2.4; [`../../docs/ROADMAP.md`](../../docs/ROADMAP.md) §2; `task_graph` spec §3.5 and design §8.9; the user's answers of 2026-08-27 |
| Part of | [`../../docs/spec.md`](../../docs/spec.md) — the whole-system specification |

---

## 1. Purpose

**The monitor is a task's event loop.** Everything that happens to a task and is
not the task's own work arrives here: a phase that finished, a subgraph that
completed, an agent that delivered nothing, a validator that could not reach a
verdict, a branch that went quiet.

**Two channels, and the difference between them is the module's shape** (§2.2):

| | | Handled by |
|---|---|---|
| **Planned** | A phase completed and the next one should begin | **Code, always.** A fixed advance, no judgement |
| **Unplanned** | Anything that makes the graph stop finishing as planned | A decision. The alpha's is thin (§7); a later one has an AI in it |

**On the unplanned channel it is the system's single decision-maker** (§2
principle 1) — not the handler of one module's failures. Any module facing a
condition it cannot resolve deterministically reports here rather than inventing a
local recovery. The runner's completeness gate (§4.1.0) is the first and most
frequent reporter, not the only one.

**Why the two live in one module rather than two.** They are the same fact
arriving from the same place: a phase ends, and either it ended as planned or it
did not. Splitting them would mean two loops, two records, and a reporter that has
to decide which door to knock on before it knows what it is reporting. **What is
split is the queue, not the module** — and that split is what keeps the planned
path free of judgement forever (§2.2).

**Rev. 14 widened this.** Through rev. 13 the module was *"the decision-maker for
every unplanned outcome"*, and the planned advances had no owner at all — the
runner was assumed to walk its three phases with nothing said about who wakes it
for the second one. §5.3 is that missing owner.

### 1.1 It is the second thing in the system with a loop of its own

The first is the agent (`agent` spec §4.3). **They are not one mechanism with two
users**, and conflating them gives the watched and the watcher one heartbeat —
an agent whose loop is wedged cannot be the thing that notices its own wedge.

That is the entire reason the monitor is a module rather than a protocol bolted
to the scheduler.

### 1.2 In scope

- What counts as an exception, mechanically (§4)
- The loop, and the two queues it drains (§5)
- **The planned phase advance, for a leaf and for a non-leaf** (§5.3)
- **The monitor's own liveness**, now that the planned path depends on it (§5.4)
- `set_task`, and the permission scope a monitor holds while it works (§6)
- The action set, and how much of it the alpha reaches (§7, §7.1)
- Where an event is recorded (§8)
- What this module requires from the modules that feed it (§9)

### 1.3 Out of scope

| | Owner |
|---|---|
| The verbs a monitor calls — `cancel`, `restart`, `fail`, `replace_with` | `task_graph` spec §3.4. They are the *transition* half of §7.1's action set; several other actions are messages, not transitions |
| That those verbs are safe from a second thread | `task_graph` design §9 |
| Resolving `Task.monitor_spec` by name | `task_graph` design §3.8 |
| Driving the agent, and the three phases | `agent` spec §4.3, design §7.2 |
| **Detecting non-delivery** | the runner. §4.1 — this is a decision, not an omission |
| *Implementing* the analysing dispatcher's roadmap actions | [`../../docs/ROADMAP.md`](../../docs/ROADMAP.md) §2.3. The set itself is specified here (§7.1); what the alpha *ships* is the pusher |

---

## 2. Design principles

**1. The monitor decides every unplanned outcome, and advances every planned
one.** This is the module's definition, not one property among several.
**Anything that makes `task_graph`'s execution depart from the plan is the
monitor's** — and unless a module can absorb a condition deterministically inside
itself, it reports here.

**The two halves of that sentence are not symmetrical, and §2.2 is why.**
"Advances" is mechanical and stays mechanical; "decides" is where the judgement
is. A monitor that blurred them would put a model on the path every task takes.

**The domain is the plan, not component health.** This is the distinction that
matters and the one an earlier revision got wrong:

| | |
|---|---|
| **Wrong test** | *did a component malfunction?* |
| **Right test** | *is the graph still going to finish as planned?* |

**A component can work perfectly and still produce an unplanned outcome.** A
validator that returns a verdict of "fail" did its job exactly right — and the
task is now terminal and the graph will not complete. Nothing malfunctioned;
the plan broke. That is squarely the monitor's.

**The question that settles it: what else would guarantee the graph finishes?**
Nothing does. The scheduler decides *when*, never *what*. The runner drives one
task forward. The validator answers yes or no. **No other component's job is
"execution is still on track"** — so if the monitor does not own every departure
from the plan, no one does, and the graph completing becomes a matter of hope.

**The carve-out is narrow on purpose.** A module absorbs a condition only when the
response is *entailed* — one correct action, no policy, no judgement, and small.
The moment there is a choice — push or resume or give up, retry or escalate, this
branch is dead or is not — it is the monitor's.

**What this prevents is the failure mode §4.1.0 already names once**, generalised:
a runner that retries on its own is a second failure policy the record cannot see,
and so is a validator that quietly re-runs itself, and so is any module that grows
a private recovery path. **Each such path is invisible to the record, unreachable
by the analysing dispatcher, and untunable from outside.** One decision-maker is
worth more than the local cleverness it displaces.

**It is also what makes the analysing dispatcher inevitable rather than
decorative.** With a single reporter the pusher is nearly sufficient; with every
module reporting, the action set has to grow, which is exactly what
[`../../docs/ROADMAP.md`](../../docs/ROADMAP.md) §2.3 holds.

### 2.1 The validator's two bad outcomes — both are reported

They differ in *kind*, never in whether the monitor hears about them:

| | | Reported |
|---|---|---|
| **A verdict of "fail"** | The validator worked; the answer is *no*. A normal product of the system (`validator` spec §3.4) — **and the task is now terminal, so the plan is broken** | **yes** |
| **No verdict reachable** | Its `entry.sh` crashed, its agent died, its own inputs were missing. Nothing was decided | **yes** |

**The first is the one worth being explicit about**, because it is where the wrong
test leads somewhere bad. "The validator worked correctly" is true and irrelevant:
a task that failed its output validation is terminal, its dependents will never
run, and **the graph will not finish.** Whether that branch is genuinely dead, or
should be retried with more knowledge, or handed to a different agent, or
surfaced to a human — that is a decision, and there is no other component whose
job it is.

**This closes a hole the system had already recorded and accepted.** Main spec §10
and [`../../docs/ROADMAP.md`](../../docs/ROADMAP.md) §2 both say that in the alpha
such a branch goes quiescent and *nothing surfaces an error*. Under principle 1 it
does not go quiescent: it is reported, and the record exists whether or not the
alpha can do anything clever about it.

**Reporting is mandatory; the alpha's *reaction* is thin, and that is fine.** The
pusher has nothing to push — the agent is gone and the task is terminal (§7). So
the alpha's monitor records the exception and surfaces it, and the richer
responses wait for the analysing dispatcher. **The point is that the branch is no
longer silent**, which was the whole of the recorded defect.

Both other documents need amending; §9 carries them.

**2. Every monitor action is a task transition it *calls*, never a status it
*assigns*.** This is inherited, not invented here: `task_graph` spec §2 principle
4 and §3.2.3 route every transition through `_move`, the single writer, under the
scheduler's `RLock`. A monitor thread therefore mutates nothing directly — it
observes, and it calls the same verbs an operator would, blocking on the lock for
the duration of a call and holding nothing between calls.

**It is what makes a second loop cheap rather than dangerous**, and it is the
reason this module can be specified without reopening the scheduler's authority
model.

**3. A monitor is not a task.** Decided 2026-08-27. It has no zone, no lease, no
agent spec of the task kind, and it does not appear in the graph. The gap between
a monitor and the task model is too wide to be worth closing — a task is a
function `<handoffs, agent>` with inputs, outputs and validators, and a monitor
has none of those.

**The consequence is stated rather than hidden: nothing monitors the monitor.**
A monitor that dies takes its watch with it, and the alpha accepts that.

**4. It receives; it does not hunt.** The stall is reported to the monitor by the
component that is already standing where the failure is visible (§4.1). A design
in which the monitor discovers everything by polling was the starting assumption
(`image.how.to.usedb.yuser.md` §2.4.3.1) and is not what this spec adopts —
polling remains available for the case §4.3 records as uncovered.

**5. An event is recorded, not only acted on.** A push that did not work is
invisible otherwise, and "the monitor handled it" with no trace is
indistinguishable from "nothing happened". **Rev. 14 widens this from exceptions
to every event**, which is what makes the planned path auditable at no extra cost:
a phase advance that never happened is now as visible as one that failed.

### 2.2 Two channels, and the line between them is permanent

| | Planned | Unplanned |
|---|---|---|
| **What arrives** | a phase completed; a subgraph finished | a gate failure, a budget overrun, either validator outcome (§2.1), a push that did nothing, an escalation |
| **Handling** | **program, always** | a decision |
| **May an AI ever be in it** | **no** | yes — the analysing dispatcher, roadmap |
| **Collapsed by task id** | **no** | **yes** |
| **Recorded** | yes | yes |

**"Program, always" is a requirement, not the alpha's convenience.** The planned
channel is on the path of every task, every phase. A model there would make
ordinary progress depend on a model's mood, and would make the cost of running the
graph a function of how many phases it has. The analysing dispatcher, when it
arrives, is bound to the unplanned channel and to nothing else.

**The collapse rule differs, and it must.** §5.2 rule 2 bounds the unplanned
queue's depth by deduplicating on task id. **Applying that to the planned channel
would lose a phase advance** — "input validation finished" and "the main phase
finished" are two advances of one task, and merging them is a task that never
runs its middle phase. Two queues, two rules; §5.2 states each against its own
queue.

**This is also the answer to "why is the monitor allowed on the happy path".**
It is allowed there because on that path it is a switch, not a judge — and §5.4
is the price of admission.

---

## 3. Two kinds, one interface

| | |
|---|---|
| **Without an agent** | A loop and a fixed reaction. The alpha's pusher (§7) |
| **With an agent** | The same loop, and an agent that *analyses* before choosing. The analysing dispatcher, roadmap scope |

**Both have a mainloop, and both satisfy the same interface.** The second is not
a different module; it is a different body behind the same `handle`.

**The kinds differ only on the unplanned channel.** Both advance a planned phase
the same way, in code (§2.2) — the "agent" in "with an agent" is bound to the
decision, and a monitor with an AI in it runs the planned path with exactly the
same lines as one without.

**Per task, or global.** **Every task has a monitor.** A task may hand its job to
a global monitor that round-robins over the tasks registered with it — the same
trade an agent makes when it attaches its mainloop to a shared thread
([`../../docs/ROADMAP.md`](../../docs/ROADMAP.md) §7.1). §6 is where the global
form costs something.

### 3.1 Escalation: a monitor that cannot resolve it passes it up

**A monitor that decides, and cannot resolve, escalates to the monitor of the
parent task.** Repeatedly, up the task tree, terminating at the root — whose
escalation target is the user.

**Escalation follows the *task* tree, never the monitor topology.** Global
monitors are a flat pool; the tree that matters is the one `task_graph` already
has. So the target is always "the monitor of my task's parent", whether either
monitor is per-task or global.

**This is what makes principle 1 affordable.** Principle 1 sends every departure
from the plan to *a* monitor; without escalation, a monitor that cannot act is
where the plan quietly dies — which is the same defect §2.1 just removed, one
level up. A decision that exceeds one monitor's reach is still a decision; it
belongs to a wider one.

**The scope moves in the right direction, and this is not a coincidence.** A
parent's zone contains its children's (`env_mgr` spec §5.1's nested layout), so
the escalation target always has **at least** the reporter's scope — the same
containment §6's tree-shaped view measured. Escalation never needs a scope the
system cannot grant.

**It does not solve "who watches the watcher".** A monitor that *cannot decide*
escalates; a monitor that *dies* escalates nothing. Principle 3's admission
stands unchanged.

**The alpha's ceiling applies here as everywhere: recording, not reaction.**
`escalate` is one of §7.1's twelve actions and the structure above is what it
will need. What the alpha owes is the chain being *defined* and
the escalation being *recorded*, so a branch that outruns its monitor is visible
rather than silent.

---

## 4. What counts as an exception

**Detection must be mechanical or nothing downstream is trustworthy**, and the
definition below is narrower than "stalled" on purpose. §4.1 is the frequent case
the alpha is built around; principle 1 is what makes it one case among several
rather than the definition of the module.

### 4.1 Non-delivery, not unresponsiveness

**The agent returned, and the goods are not there.** A terminal result message
came back and the delivery does not stand up. §4.1.0 makes that mechanical: four
independent checks, and failing any one of them is the same event.

**This is common, not exceptional.** An agent returning a terminal result without
having actually finished is an ordinary occurrence, and the module is shaped
around that fact: "exception" here names a *kind* of event, not a rare one.

#### 4.1.0 Where it happens: the admission gate before output validation

**The check sits between the main phase and output validation, and it is the
runner's.**

```
RUNNING ──agent returns──▶ [ completeness gate ] ──pass──▶ OUTPUT_VALIDATING ──▶ on_done
                                  │
                             fail │ report()
                                  ▼
                            [ the monitor decides ]
                                  │
                             push │  (or resume, or give up — §7)
                                  ▼
                            agent runs again ──────▶ back to the gate
```

**The runner does not push.** It reports, and the monitor decides — that is the
whole reason this module exists. A runner that retried on its own would be a
second, hidden policy for handling failure, invisible to the record and
unreachable by the analysing dispatcher that replaces the pusher later.
`report()` is therefore not an aside on the retry path; it *is* the retry path.

**It has nothing to do with `on_task_done`**, which happens after output
validation, when the task is being closed. The two instants are separated by the
entire validation phase, and an earlier revision of this section conflated them —
corrected here, and the "who owns the fact" question that rested on the confusion
is void.

**The whole cycle is absorbed below the scheduler.** It does not move task status,
does not reach the scheduler, and does not involve `on_task_done`. A task cycling
through the gate several times is behaving normally, and the graph sees one task
still `RUNNING` throughout.

**A task may not enter output validation until the gate passes.** The gate is
deliberately cheap — it is an admission check, not validation:

| The gate asks | |
|---|---|
| **Is everything the spec requires present?** | Not one declared output missing |
| **Did the agent say it was done?** | `done_by_self_check` on the handoff — §4.1.2 |
| **Is what claims to be executable actually executable?** | A delivered `entry.sh` that cannot run has not been delivered |
| **Is the task still inside its budget?** | Tokens and elapsed time — §4.1.3. Not a property of the delivery, but a gate failure on the same path |

Only then does `OUTPUT_VALIDATING` begin. **The gate is not the validator** — it
does not ask whether the work is *right*, only whether there is something to
check.

#### 4.1.1 What the store already guarantees, and what it leaves to the gate

Measured against the shipped contract rather than assumed. `HandoffStore.put`
(`handoff/protocols.py`) *"runs the README check and the locality check **before**
anything is created, raising `Malformed`; a handoff that reached storage malformed
would need retracting, and nobody anywhere has solved retraction."*

**So for those two properties the gate has nothing to do**: a handoff that is
badly formed in *that* sense never reaches storage, and its absence is the whole
observable —

```
exists(hid) is False
```

**But `put`'s admission checks are exactly two, and executability is not one of
them.** A delivered `entry.sh` that cannot run is *present and wrong*, and it
reaches storage. So §4.1.0's third check is not redundant with the store; it
covers precisely what the store does not:

| Property | Guaranteed by |
|---|---|
| README present and well-formed | `put` — cannot be wrong in storage |
| Locality | `put` — cannot be wrong in storage |
| **Present at all** | the gate, one `exists()` per declared output |
| **Executable if it claims to be** | **the gate. `put` does not check this** |
| The agent's own claim of completeness | the gate. Not an artefact property at all |

`items_schema` is a third thing again: it is verified when the **kind is loaded**
(`handoff` design §3.5), not against delivered content at publish.

**The distinction is still real, and it is visible only where it happens.**
`handoff` design §5.3: *"The producing agent works in its playground and hands the
store a directory."* **The producer calls `put`, from inside its own zone** — so a
`Malformed` is raised inside the agent's execution and never reaches the runner,
which sees only the later absence.

"Never attempted" and "attempted and refused" therefore differ in a fact that
cannot be reconstructed from an absence. If the record (§8) is to carry it, it
must be captured **producer-side, at the moment `put` refuses**. That is
criterion 9's requirement one level down, and it lands on the agent, not on this
module.

#### 4.1.2 `done_by_self_check` — a weak check whose description is the mechanism

**A boolean on the handoff, set by the producing agent when it believes the
package is complete.** The gate reads it; it is a separate signal from the
artefacts existing, because "everything I was asked for is on disk" and "I think
I am finished" are different claims and an agent can be wrong about either.

**The field's own description carries the instruction**, in the shape of:

> *you MUST check three times to ensure the handoff you packaged is … before you
> turn this value to true*

**This is deliberately a weak check and should not be mistaken for validation.**
It verifies nothing. Its purpose is narrower and worth stating plainly: **to cut
the number of round-trips between the main phase and validation**, by making the
agent look once more before it declares itself done. A cheap re-read at the point
of packaging is worth several expensive validation cycles.

**Why a field with a description, rather than a line in a prompt.** Core principle
3: *a rule that lives only in a prompt is not a rule.* As a field it is
mechanically readable by the gate, and the instruction travels attached to the
thing it governs instead of living in whatever prompt happened to be assembled.

**It does not exist yet** — `handoff` has no such field. §9 records it.

#### 4.1.3 Budgets are gate failures too, and they are what bounds the loop

**The runner also carries thresholds** — tokens spent, wall-clock elapsed, and
whatever else it tracks — and **exceeding one is itself a reportable exception**.
The runner does not decide what to do about it. It reports, and the monitor
decides, exactly as for a missing output.

**The data is already returned.** `ResultMessage` carries `duration_ms`,
`num_turns`, `total_cost_usd`, `usage` and `model_usage` (`agent` spec §5.1), so a
budget check at the gate reads values the backend hands over anyway.

**This is what bounds §4.1.0's loop.** Without it, an agent that never satisfies
the gate cycles forever; with it, the cycle has an exit that is a *decision* — the
monitor's — rather than a hard-coded retry count buried in the runner.

**In the alpha the thresholds are one global setting.** Not per task, not per
agent spec. Decided 2026-08-27, and it is the right first move for a reason worth
recording: **nobody yet knows what a normal task costs.** A per-task limit would
have to be authored per task, out of numbers no one has, and would mostly be
copied from a default — which is a global setting with extra steps and a worse
failure mode, since a wrong local value is invisible where a wrong global one is
felt immediately.

Per-task or per-agent-spec limits become answerable once the system has produced
a cost distribution; §11 keeps that.

### 4.2 The agent's status while an exception is open

**Decided 2026-08-27: the alpha keeps `running`, and `fixing` goes to the
roadmap.** `AgentStatus` is unchanged — `pending → deploying → running →
{finished | failed | interrupted}` — so no backend adapter and no superset rule
(`task_graph` spec §3.2.1) has to learn a new state. The condition is legible from
the record (§8), not from the status.

**`fixing` is more expensive than one enum value, which is why it is not free to
add later without thought.** It would be **the first agent status driven by the
monitor rather than the runner.** The authority rule (§2 principle 2) covers
*task* status — every monitor action is a transition it calls, and `_move` is the
single writer — but **agent status is a different model that does not go through
`_move` at all.** Introducing `fixing` therefore opens a question nobody has yet
asked: who may write agent status, and under what discipline.

Recorded so that whoever picks `fixing` up starts from that question rather than
from the enum.

### 4.3 What this path does not cover

**An agent that never returns at all.** §4.1 triggers on a *return*; an agent
whose process is wedged produces no return and no report.

**The alpha builds nothing for it, and makes the seam exist.** Not a heartbeat,
not a deadline, not a probe — a period on the loop's wait and a sweep hook that
does nothing yet. That is a few lines, and it is the difference between a design
that admits a poller and one where adding a poller is a refactor.

**The reason it is a position rather than an evasion**: in the alpha's shape —
one process, and the agent running on the task's own thread (`agent` spec §4.3) —
a wedged agent is a wedged *thread*, and **Python cannot kill a thread**. None of
the three standard mechanisms recovers from that cleanly, so building one would
buy detection without remedy.

**Rev. 14 makes that cost sharper without changing the answer.** The thread the
wedged agent holds is the *task's*, so the task cannot reach its output validation
either — a wedge now stalls a phase advance and not just an agent. It is still one
task's thread and no one else's, so nothing else in the graph is affected, and
**the monitor's own loop is a different thread**, which §5.4 is about.

**If a sweep is added later it should be a liveness probe over
`AgentBackend.status`, not a heartbeat**, and it should take N consecutive
negative probes rather than one, in the shape of a `failureThreshold`.

**The measurement that would change this**, named so it is not merely deferred:
the distribution of wall-clock time between `on_started` and the terminal result,
over real tasks and real backends. If it is tight, a deadline becomes viable and
is far cheaper than a probe. **Nothing in the repository can produce that number
today** — `FakeRunner` never calls `on_done` from `start`, and no real backend is
wired.

---

## 5. The loop, and the two queues it drains

**The simplest form, and the one the alpha adopts:** the monitor holds two
queues; its mainloop takes work from them one item at a time and handles it.

```
                                    ┌──▶ [ planned queue ]  ──┐   FIFO, no collapse
a reporter calls report(record) ────┤    (advance a phase)    ├──▶ monitor mainloop
   (§5.1, several)                  └──▶ [ unplanned buffer ] ─┘   (one consumer)
                                         (decide, §7)              collapse on task id
```

**One call, two queues, and the routing is the module's not the caller's.**
`report()` is still the only inbound call (§5.1); which queue a record lands in
follows from its `kind` and from nothing the caller has to know. A reporter that
had to choose the door would have to classify the event before reporting it, and
classification is exactly the thing this module exists to own.

**The planned queue is served first.** A task waiting to advance is a task doing
nothing, and the unplanned channel's work is by nature slower and sometimes
blocking. Draining planned work first keeps one stuck decision from stalling every
other task's progress. **It cannot starve the unplanned queue in the alpha**,
because a planned advance is a fixed, non-blocking transition and the planned
queue drains to empty in bounded time.

**One consumer, several producers.** A global monitor watching several tasks is
fed by several runners; and under principle 1 a runner is not the only kind of
reporter (§5.1).

### 5.1 `report()` — the inbound call, and the routes that do not exist

**`report()` is how an event gets from the thing that observed it to the
monitor.** It is the concrete form of the missing routes §9 lists: no reporter —
not the runner at the gate, not the validator, not the runner finishing a phase —
has a handle on a monitor or a call to make.

```python
def report(self, record: EventRecord) -> None: ...
```

| | |
|---|---|
| **Who calls it, planned** | The thing that just finished a phase. In the alpha that is the task's own attempt object (§5.3) — after input validation, after the main phase, and after a subgraph's `is_end` completes |
| **Who calls it, unplanned** | **Any module facing a condition it cannot resolve deterministically** (§2 principle 1). The runner at a failed completeness gate (§4.1.0) is the most frequent caller; the validator, when it cannot reach a verdict, is the other one the alpha knows about |
| **What it does** | **Persists the record synchronously, then enqueues it** on the queue its `kind` selects. One call, both steps |
| **When it returns** | After the write. It is a few small files, on no lock and on no scheduler path (§4.1.0) |
| **The inbound surface** | `report()` and `set_task`, and nothing else. **Rev. 14 widened what arrives and did not widen this** — that is the whole reason the routing is internal |

**`EventRecord`, not `ExceptionRecord`.** Renamed in rev. 14: half of what now
travels through this call is a phase finishing normally, and a record type called
`Exception` would make every ordinary advance read as a fault. §8 keeps the
vocabulary; only the noun changed.

**One call, not two, and the persistence is synchronous.** Decided 2026-08-27.
Rule 3 requires the record durable before the buffer sees it; making that one
call's internal ordering means the rule holds **structurally** rather than by the
caller remembering to do two things in order. A caller that forgets the first of
two steps loses the event silently, which is the failure the rule exists to
prevent.

**The cost is affordable because of where the call sits.** §4.1.0 puts it inside
the runner's own gate loop — not under the scheduler's lock, not on the
`on_task_done` path, not holding anything. A synchronous write of a few small
files there delays one task's own cycle and nothing else.

**It is named here because rev. 3 used the name five times without ever
introducing it** — the same defect §9 exists to catch, committed one section after
that section's own instance of it was written up. Left visible for the same
reason.

**`report()` is a call, not a record.** Erlang's "supervisor report" — the source
of §8's field structure — is the *record itself*, a noun. The collision is this
document's own and §8.2.1 resolves it: `report()` is only ever the call, and the
thing recorded is a **record**.

### 5.2 The five rules

**The asynchrony is sound, under five rules.** Each is a requirement, not an
implementation: the design chooses the structure, the spec fixes what must hold.
Kubernetes' `client-go` workqueue is the prior art and the source of the shape
(`scratch/design/findings-monitor-loop.md`).

**Which rule governs which queue**, since rev. 14 there are two:

| | Planned | Unplanned |
|---|---|---|
| 1 — never blocks, never refuses | **yes** | **yes** |
| 2 — bounded | by construction, below | by dedup on task id |
| 3 — record first, enqueue second | **yes** | **yes** |
| 4 — collapse merges, never overwrites | **does not apply** — nothing collapses | **yes** |
| 5 — one task is not handled twice at once | **yes, and it spans both queues** | **yes** |

**The planned queue is bounded without dedup, and the reason is structural.** A
task is in exactly one phase at a time, so it can have at most one outstanding
advance; depth cannot exceed the number of tasks, which is rule 2's bound reached
by a different route. **Deduplicating it would be a defect, not an optimisation**
(§2.2).

**1. `report()` never blocks and never fails.** The buffer is **unbounded** —
the canonical implementation's choice, and the right one here.

**Precisely: it never blocks *on the buffer*, and never refuses a record.** It
does block on its own synchronous write (§5.1), which is intended and bounded.
The distinction matters because the two have different failure modes: a slow write
delays one task's gate cycle, whereas a full queue would make the caller wait on
*the monitor*, coupling a common path to how busy the handler happens to be.

**Rev. 3 justified this by saying the reporter holds the scheduler's lock. It does
not**, and §4.1.0 now says why the question does not arise at all: `report()` is
called from inside the runner's gate loop, which holds no lock and touches no
scheduler path.

**2. It is bounded anyway, by dedup on task id.** Depth cannot exceed the number
of tasks. That is what makes rule 1 safe without a `maxsize`.

**3. Record first, enqueue second.** The exception is **durably recorded before
the buffer sees it** — inside `report()`, which is what makes the ordering
structural rather than the caller's to remember (§5.1, §8). This is the rule that makes everything
else affordable: once the record exists, losing the queued *work* is a
degradation, not a loss of the event. Without it, a dropped queue entry is a
vanished fact.

**4. Collapsing merges; it never overwrites.** Two exceptions for one task
combine into one unit of work, and **both records survive** with a suppression
count. `client-go` does last-wins here and that is its gap, not a pattern to
inherit — criterion 9 requires *every* exception to be recorded.

**5. A task being handled is not handled twice.** An event arriving while the
monitor is inside a transition — blocked on the scheduler's `RLock` — is
re-queued exactly once, after the current handling completes.

**Since rev. 14 this spans both queues**, and it has to: a planned advance for
task T while T's exception is being decided would have the monitor moving a task
forward and repairing it at the same instant. **One task, one handling, whichever
queue it came from.**

**One consumer per monitor.** Rule 5 would make several consumers safe, but
nothing in the alpha needs the throughput, and one consumer keeps §6's
"one task's scope at a time" straightforward.

**Two things the design must not inherit from the standard library**: there is no
`Queue` `maxsize` here (rule 1), and `queue.Queue.shutdown` is **3.13-only** while
the target is 3.10 — shutdown is hand-rolled, and its `immediate=True`
discard-everything semantics are the wrong default. Refuse new reports *loudly*,
drain what is queued, then stop.

### 5.3 The planned advance

**What the monitor does on a planned event is one transition and, when the next
phase needs a thread, one call to get it one.** There is no decision in it.

**A leaf.** Its attempt holds one thread from dispatch to `on_done`; the phases
borrow it in turn and hand it back.

```
scheduler dispatches ──▶ attempt created, one thread started
   ├ INPUT_VALIDATING    runs, returns the thread, report(planned)
   │                        └─▶ monitor: task.enter_phase(RUNNING), wake the thread
   ├ RUNNING             the thread is what the agent's mainloop runs on;
   │                     the completeness gate (§4.1.0) sits at its end
   │                        pass ─▶ report(planned) ─▶ monitor: enter_phase(OUTPUT_VALIDATING)
   │                        fail ─▶ report(unplanned) ─▶ §7
   └ OUTPUT_VALIDATING   runs, then on_done
```

**A non-leaf. It holds no thread while its subgraph runs**, and that is the whole
reason this case is written out:

```
   ├ INPUT_VALIDATING    runs, then unfold, then the thread ENDS.
   │                     The attempt object stays; the task occupies no thread.
   │
   │        … the subgraph runs, for as long as it takes …
   │
   └ is_end subtask completes
        └─▶ the subtask's monitor walks Task.parent and reports a planned
            event to the PARENT's monitor
              └─▶ parent's monitor: parent.enter_phase(OUTPUT_VALIDATING),
                  then asks the runner for a thread; output validation runs
```

**Three properties this is built to preserve, each of which an earlier draft
broke:**

| | |
|---|---|
| **The scheduler never observes another task's state** | It is not involved. `task_graph` §2 principle 2 (content-agnostic) and principle 4 (a transition is the only thing that triggers it) are both untouched, and §3.2.1's "does not treat `is_end` specially at completion" is honoured because the scheduler never sees `is_end` at all |
| **A monitor transitions only its own task** | The subtask's monitor does **not** touch the parent. It reports to the parent's monitor, which owns that transition. §6's scope rule is unchanged, and the `Task.parent` walk is the one escalation (§3.1) already needs |
| **Thread count stays at the executing leaves** | The non-leaf's thread ends at `unfold` and a new one is taken for output validation. An attempt that waited on a condition through its subgraph would hold one thread per ancestor |

**A re-entry is not a new attempt.** One `Execution` spans both threads: the
parent was dispatched once, and its output-validation phase is the same attempt
resuming. Nothing pushes a second execution record and no second agent is bound.

### 5.4 The monitor's own liveness

**Rev. 14 put the planned path through this module, so "nothing monitors the
monitor" stops being a recorded risk and becomes a requirement.** Before, a dead
monitor meant exceptions went unhandled. Now it means **every task stops
advancing, silently** — which is the defect class §2.1 exists to remove, applied
to the whole system.

**Two mechanisms, both small, both alpha scope.**

**1. An uncaught exception must reach a human.** It does not by default, and this
is measured rather than assumed (`scratch/design/probes-monitor/p4_thread_death.py`,
re-run on 3.13.13): a `threading.Thread` whose target raises prints a traceback to
stderr and dies; **the process keeps running, the exit code does not change, and
producers see no error** — further reports are accepted and queue up behind a
consumer that no longer exists.

So the system installs a `threading.excepthook` that turns an escaped exception
into a record and surfaces it. **This is a system-wide rule, not a monitor
feature** — any thread that dies of an exception nobody caught is a fact the user
must be able to see.

**It covers a crash and not a wedge**, which is why there is a second mechanism.

**2. The loop publishes a heartbeat, and something trivial checks it.** The
mainloop stamps a timestamp each time round; a check outside it reports when the
stamp has been stale for **N consecutive periods**. The shape is
`failureThreshold`, which §4.3 already chose over a heartbeat for the agent case
and which is chosen here for the same reason: one slow round is not a death.

**The checker must be trivial enough that nothing needs to watch it.** That is
the answer every surveyed supervision system gives to this question in one form or
another (`scratch/design/findings-arch-super.md`): s6 makes its top-level
supervisor unable to fail — no heap allocation, under 500 lines, a full
deterministic automaton — while systemd hands the same problem to a hardware
watchdog and Ray hands it outward to KubeRay. **Comparing a timestamp is at the
trivial end of that scale**, and it is the reason this is a sound answer rather
than an infinite regress.

**What neither mechanism gives is recovery.** They make a dead or wedged monitor
*visible*. Restarting one, and what happens to the tasks it was watching, is not
specified here — §11 carries it.

---

## 6. `set_task`, and the permission scope

**A monitor is told what to watch; it does not go looking.** `set_task` is the
interface, and it is what makes a global monitor possible at all.

**A monitor's permission needs follow from the task.** Given the task it is
handling, a monitor knows what it needs to reach — the task carries its own
versioned permissions (`task_graph` spec §3.2), and the monitor's scope while
handling that task is that task's scope and no more.

**The alpha needs no environment reach at all, and this is measured rather than
hoped.** Every action §4.1 and §7 require — `set_task`, the buffer,
`AgentBackend.status`, `AgentBackend.instruct`, and the four transition verbs —
is an in-process call on an object the supervisor already holds. The handoff check
belongs to the runner, not to the monitor.

**It writes records, and that is not a counter-example.** §5.1 and §8 put a
synchronous write on the recording path, but it lands in the supervisor's own
record store — **never in a task's zone.** The claim here is precise and should
stay that way: the monitor needs no reach into any *task's* environment.

**A monitor must not enter a task's sandbox**, and there are four independent
reasons, each probed (`scratch/design/findings-monitor-sandbox.md`, M2–M8):

| | |
|---|---|
| There is nothing to enter | On the live rung (Landlock, ABI 3; `bwrap` absent) no syscall applies a ruleset to another task, and a zone is an ordinary host path. **M6**: an unconfined outside process read *and wrote* a confined agent's zone |
| It cannot enter anyway | `setns` works unprivileged, but **a multithreaded process cannot join a user namespace** — M8: single-threaded `rc=0`, two threads `EINVAL`. The supervisor is multithreaded by construction: one `mainloop()` per agent, plus this one |
| Entering would be an **escalation, not a scoping** | The single-threaded child that *can* join lands at `uid=0`, `CapEff=0x1ffffffffff`, carrying **none** of the target's Landlock domain. **M3**: the joiner read and wrote outside the zone the agent itself was denied |
| It is redundant | `/proc/<pid>/root` reaches the private mount namespace from outside |

**The tree-shaped view is real, free, and lives in the layout rather than the
kernel.** `env_mgr` spec §5.1 nests a child's zone inside its parent's, and a
Landlock hierarchy grant cannot be holed. **M4**: one grant on the parent zone
root reached every descendant to depth 4 — *including subtrees created after the
ruleset was built* — while denying the sibling before and after. That last clause
is `env_mgr` criterion 14; **the tree view and criterion 14 are one measurement.**
The cost is one `Granted` entry. It is a *filesystem* view only: no process state,
no live agent state.

**Handing the monitor a zone of its own does not work.** `prepare` cannot serve it
— the signature takes an `Execution`, and a monitor has no attempt — and **M5**
kills the global form outright: confine to A, then `set_task(B)`, and the monitor
reaches *neither*, because `{A} ∩ {B} = ∅` and the restriction is irreversible.
The only workable shape would be **fork-per-exception**, which is what `env_mgr`
§14.3 already built for tests. Not needed by the alpha, and recorded so the next
person does not rediscover it.

### 6.1 What that makes criterion 8, and why it is not a weakening

`env_mgr` design §8.4: *"`prepare()` is called by the supervisor, once per
attempt, and the executor is the supervisor's child."* Executors are therefore
flat children of an **unconfined** supervisor, and the monitor lives in the
supervisor. **A monitor is unconfined by construction**, so no OS mechanism
bounds it.

Criterion 8 is accordingly about **the verbs**: a monitor may transition the task
`set_task` gave it and no other. That is enforceable in-process, and it covers
every §7.1 action that touches task state — the rest are messages, which reach no
zone either.

**This does not sit against `env_mgr` §4.1's rule that a hook alone does not
isolate anything.** That rule governs a component with reach *into what it is
supervising* that proposes to bound itself by convention. The alpha's monitor has
no reach into a task's environment to bound (§6) — the criterion as first drafted
assumed a contact surface that never occurs.

**OS enforcement becomes a live question when a monitor gains an AI that touches
files** — the analysing dispatcher, roadmap scope. At that point the fork-per-
exception shape above is what it would need, and §6 records its cost.

**The known risk, carried forward:** a global monitor's history leaks context
across the tasks it monitors ([`../../docs/ROADMAP.md`](../../docs/ROADMAP.md) §2).
Recorded, not solved.

---

## 7. What the alpha does: the pusher

**A status check plus one phrase: *continue, do it until finished*.**

That is the whole of the alpha's reaction, and §4.1's definition is what makes it
coherent: the agent returned without delivering, so the corrective action is to
tell it to keep going.

**"Keep going" is expressible on a returned agent. Measured, not assumed.**
A `ResultMessage` ends a **turn**, not the session and not the process: the SDK's
`query()` has no already-finished guard, its only precondition being that the
client is connected, and a string prompt is one JSON line written to a live stdin
(claude-agent-sdk 0.2.145, `client.py`). A live probe pushed a returned agent and
got an answer on the **same session id, in the same process, in ~2 s**
(`scratch/design/findings-monitor-push.md` §1–2).

**The agent-directed actions are ordered by cost, and the order is
load-bearing:**

| | Cost | Loses |
|---|---|---|
| **push** | ~2 s | nothing |
| resume | ~5.5 s warm | `permission_mode`, `--mcp-config`, `--settings`, `--add-dir` — i.e. **the per-attempt wiring `env_mgr` prepared** |
| `restart` | full | all context, plus the zone, which is per attempt |

**A monitor must not reach for `restart` first.** For an agent that returned
nearly-finished work and merely failed to publish, it is the most expensive
possible reaction to the cheapest possible fault.

**This imposes one requirement on the backend, recorded in §9**: the client's
lifetime must be the *agent's*, not the *turn's*. A backend that wraps a turn in
`async with ClaudeSDKClient(...)` has destroyed the subprocess before the runner
even checks the handoffs, and leaves the monitor only the lossy resume.

### 7.1 The full action set

**Every action in this table is on the unplanned channel.** None of them is
reachable from a planned advance, which has one fixed behaviour and no action set
at all (§2.2, §5.3). That is what the table means by an alpha column: it bounds
the *response to trouble*, and bounds nothing about ordinary progress.

**The action set belongs here, in full.** It had been carried as a bullet in
[`../../docs/ROADMAP.md`](../../docs/ROADMAP.md) §2.3 and quoted in this section
as seven of its twelve entries — an abbreviation of a normative list, which is how
entries go missing.

| Action | | Alpha |
|---|---|---|
| **push** | *continue, do it until finished*. Content-free | **yes** — §7 |
| **escalate** | to the monitor of the parent task (§3.1) | **chain defined, recorded** |
| **report directly to user** | bypassing the chain, when the decision is a human's | **recorded only** |
| **answer** | the agent asked something and is waiting; the monitor supplies the reply | roadmap |
| **help** | assign a helper agent | roadmap |
| **add context** | more input or knowledge | roadmap |
| **coordinate** | create a coordinator | roadmap |
| **create teammate / rival** | a second agent on the same work | roadmap |
| **change agent spec** | a different agent for the retry | roadmap |
| **restart the task** | `resume_task`, which already accepts `FAILED` | verb exists |
| **submit a copy** | an ordinary `submit` | verb exists |
| **reconcile related task state** | the one needing the transition set | roadmap |

**`answer` is a genuine addition, not a rename.** It was in none of the recorded
lists. It differs from the three nearest neighbours in a way that matters: **push
is content-free**, *add context* is unprompted, and *report to user* hands the
question away — whereas `answer` responds to a question **the agent actually
asked**, with content. That makes it the one action that **cannot be done by the
pusher at all**: replying requires understanding what was asked, so `answer`
belongs to the agent-bearing monitor by construction, not by policy.

### 7.2 `answer` needs no new channel — the question is already in the history

An earlier revision recorded `answer` as unreachable, on the reasoning that
`AgentBackend` has `instruct` (monitor → agent) and no reverse. **That assumed the
wrong shape.** The agent does not have to *push* a question; the monitor *reads*
one.

| | |
|---|---|
| While the agent lives | `AgentBackend.query() -> AgentHistory` (`agent` spec §4.3). The monitor holds the handle, so it holds the history |
| After a restart | `get_session_messages(session_id)` rebuilds it from the session store |

**Both are public API**, measured against claude-agent-sdk 0.2.145:
`get_session_messages`, `get_session_info` and `list_sessions` are in the
package's `__all__`, not under `_internal` only. Cursor's equivalent is
`agent.list_messages()` (`agent` spec §5.1), so the capability is not
Claude-specific.

**So the monitor can always reconstruct what the agent was asking**, including
across a restart, and `answer` is reachable with nothing new built. What it needs
is a monitor that can *read* — which is why §7.1 puts it in the agent-bearing kind.

**This is not a breach of principle 4.** *It receives; it does not hunt* governs
how an exception is **discovered** — the monitor is told, it does not poll to find
out. Reading the history of a task it has already been handed, in order to
**decide**, is the opposite end of the same event.

**Principle 2 does not bend for any of them.** Every one of these, when it
arrives, is a transition the monitor *calls* or a message it sends — never a
status it assigns.

**The pusher has no push for a failed validation, and that does not excuse
silence.** A task that failed its output validation is terminal with no agent
running (`validator` spec §3.4), so none of push / resume / restart applies to the
*agent*. But the exception is still reported and still recorded (§2.1): the alpha's
monitor cannot fix such a branch and **must not leave it unremarked**, because a
dead branch nobody is told about is how a graph stops without anyone noticing.

**Recording is the alpha's action here.** Deciding what to *do* about a dead
branch — retry with more knowledge, reassign, escalate to a human — is the
analysing dispatcher's ([`../../docs/ROADMAP.md`](../../docs/ROADMAP.md) §2.3).
The separation is deliberate: the alpha's ceiling is on the *response*, never on
the *reporting*.

---

## 8. Recording the event

**Somewhere records every event**, whether or not it was handled: a phase that
advanced, a failed gate, a budget overrun, either of the validator's bad outcomes
(§2.1), a push that did not take, an escalation, a monitor that gave up.

**Rev. 14 brought the planned advances in here too**, and it costs nothing: they
travel the same call and take the same shape, and the result is that a task's
whole life — every advance and every fault, in order — is one readable sequence
rather than a fault log beside an unrecorded happy path. **A phase that never
advanced is now as visible as one that failed**, which is what makes §5.4's
wedged-monitor case diagnosable at all.

### 8.1 The carrier was never open

Rev. 1 said *"nothing in this system records an exception today — there is no
existing shape to be consistent with."* **That is true of exceptions and false as
a licence to choose freely.** This repository has already fixed the *carrier*, in
three places and with its reason: a record is **a persisted value written through
`task_graph`'s `StoreMgr`, not a log line.** A test that asserts on `caplog` is
testing the logging configuration, not the behaviour.

So the choice is narrower than it looked: **only the field vocabulary is open**,
and a design that reaches for a logging library and satisfies criterion 9 with a
log capture has gone wrong at the first step.

**Logging is a projection of the record, never the record.** One handler renders
a record to a line at the severity it already carries; tests assert on records.

### 8.2 The vocabulary is a hybrid, and it costs no new dependency

No single standard covers this case, so each supplies the part it is good at:

| From | What it supplies |
|---|---|
| **OpenTelemetry** stable `exception.*` semantic conventions and `SeverityNumber` | the field **names**, as naming only |
| **Erlang/OTP** SASL supervisor record | the field **structure** — see §8.2.1 |
| **Sentry**'s event / fingerprint / issue split | **identity and grouping** — a fingerprint groups, and never discards |

**Net cost: zero new packages.** Adopting the OTel *SDK* would add three and
would not answer the question the monitor actually has, because **OTel is
emit-only by construction** — and the monitor must sometimes *read* what it
recorded, to know that a push already failed once.

#### 8.2.1 The word "report" means two things, and only one of them is ours

**A naming collision this document created, resolved here rather than left to
trip someone.**

| | |
|---|---|
| **`report()`** | A **call** — the runner handing an exception to the monitor (§5.1). This module's own coinage |
| **"supervisor report"** | Erlang's term for **the written record itself** — a structured description of one failure |

They are a verb and a noun sharing a spelling. **This spec uses `report()` only
for the call, and "record" for the thing recorded.** The word "report" is not used
for a record anywhere else in this document.

**What was borrowed is the structure.** Erlang's supervisor record, verbatim from
the SASL documentation, is issued when a supervised child terminates unexpectedly
and carries four items:

| Erlang | Meaning | Ours |
|---|---|---|
| `Supervisor` | name of the reporting supervisor | who reported — any of §5.1's callers, or a named monitor when it escalates |
| **`Context`** | **in which phase the child terminated, from the supervisor's point of view.** A closed enum: `start_error`, `child_terminated`, `shutdown_error` | the record's `kind` |
| `Reason` | termination reason | the OTel `exception.*` payload |
| `Offender` | start specification for the child | the correlation fields — task, attempt, agent, handoff |

**`Context` is the item that earns its place, and the reason is criterion 9.**
Distinguishing "attempted and ineffective" from "never attempted" is a
*phase* distinction — the two are **different kinds of record**, not one kind of
record carrying different prose. A closed enum answers that mechanically; a free-
text message answers it only to a human reading the file. That is why the record's
`kind` is an enum with a value per case, and why none of its values may default to
a benign one.

**Do not build on span events.** They were the obvious shape and are no longer
one: recording exceptions on spans is **Deprecated** as of semconv v1.40.0
(PR #3256, merged 2026-01-28), in favour of the logs conventions, with
`Span.RecordException` and `Span.AddEvent` slated for deprecation under OTEP
4430. Verified against the merged PR, not recalled.

**And the standard declines the half we most wanted from it:** *"Guidance on when
to record exceptions is left to specific semantic conventions authors."* OTel
supplies names; the policy is this document's, which is why §4.1 has to be as
mechanical as it is.

### 8.3 Absence is a signal, so the container must exist

Criterion 9 distinguishes "attempted and ineffective" from "never attempted", and
the second is read as an **absence**. Absence only carries meaning if the thing
that would hold the record reliably exists.

This repository has already solved the identical problem once, on
`validation.yaml` (`handoff/protocols.py`): *"It is created empty at publication
rather than on first verdict: an empty `verdicts:` list says 'nothing has checked
this yet', a missing file says something is wrong."*

**The same rule applies here**: an attempt's record set is created — possibly
empty — when the exception is opened. Without it, "never attempted" and "the
store lost it" are the same observation, which is exactly the confusion criterion
9 exists to prevent.

### 8.4 One field already exists and must be reconciled

`Execution.detail` (`task_graph/models.py`) is written **by the runner, at the
instant §4.1 describes**. It is not a monitor field and this spec does not claim
it, but a design that adds a second writer to it, or that duplicates it, is
creating a two-writer problem rather than a record. The design says which it is.

---

## 9. What this module requires from the modules that feed it

Stated as its own section because the recurring defect of this stage has been a
document declaring that X consumes Y when X's signature cannot receive Y. These
are the routes this spec depends on, and their status as measured on 2026-08-27.

| Required | From | Status |
|---|---|---|
| The verbs — `cancel`, `restart`, `fail`, `replace_with` | `task_graph` §3.4 | **Exists** |
| Those verbs safe from a second thread | `task_graph` design §9 | **Exists** |
| `Task.monitor_spec`, resolved by name | `task_graph` §3.2, design §3.8 | **Exists** |
| `Task.status` readable, `by_status` scannable | `task_graph` design §6.2.1 | **Exists** |
| **A route from the runner to the monitor** to report §4.1 | `agent` | **Does not exist.** The runner has no monitor handle and no reporting call. The largest of the module's missing routes, and still a small one |
| **The same route used for planned events** (§5.3) | `agent` | **Does not exist**, and is the same route one row up. A phase that finished reports through `report()` exactly as a phase that failed does |
| **A per-task attempt object that owns the thread and the executor handle** | `agent` | **Does not exist.** Rev. 14 requires the runner to create one per dispatch. It is what closes the "who holds the live handle" gap, and it is where the thread is created and ended (§5.3) |
| **`resume(task_id)` on the runner** — give an existing attempt a thread again | `agent` | **Does not exist.** The non-leaf re-entry needs it (§5.3). It is not `start`: no new `Execution`, no new agent, the same attempt |
| **`Task.parent`, readable** | `task_graph` | **Designed, not implemented.** `task_graph` design §3.3 declares it; the shipped `models.py` is at design rev. 10 and does not have it. §5.3's non-leaf walk and §3.1's escalation both read it |
| **A `threading.excepthook` that surfaces an escaped exception** | the composition root | **Does not exist**, and the default is measured to be silent (§5.4). System-wide, not this module's alone |
| **`fixing` in the agent status set**, if §4.2 chooses it | `agent` §4.3 | Does not exist. Conditional on a design decision, so not yet a defect |
| **The presence check §4.1.0 needs** | `handoff` | **Exists.** `HandoffStore.exists(hid)`, one call per declared output. §4.1.1: README and locality cannot be wrong in storage, so presence is all the store owes — **executability is the gate's own and `put` does not check it** |
| **The runner holding a store handle and the task's declared outputs** | `agent`, `task_graph` | **To be verified.** `Task.outputs` exists; that the runner is *given* a `HandoffStore` is not established by `agent` design §7 |
| **A producer-side record when `put` raises `Malformed`** | `agent`, `handoff` | **Does not exist.** Needed only for criterion 9's "attempted vs never attempted" — the raise happens inside the agent's zone (§4.1.1), out of this module's reach |
| **A backend client whose lifetime is the agent's, not the turn's** | `agent` | **Not stated.** Design §7 does not say it, and the natural way to write a turn (`async with`) breaks it. Without this the push in §7 degrades to a lossy resume |
| **`instruct` mapped to a form that can actually be pushed** | `agent` | **Defect, reported not edited.** Spec §5.1's `instruct` row maps to `query(AsyncIterable[dict])` — the one form that closes stdin at the first turn boundary. Measured: `CLIConnectionError: ProcessTransport is not ready for writing` |
| **A way to learn what the agent asked** | `agent` | **Exists, and needs no new channel** — §7.2. `AgentBackend.query() -> AgentHistory` while the agent lives, and `get_session_messages(session_id)` across a restart. Both public |
| **A route from the validator to the monitor**, for **both** of §2.1's outcomes | `validator` | **Does not exist**, and is the same shape as the runner's missing route one row up. Widened by principle 1 |
| **Withdrawing the "quiescent branch" limitation** | `docs/spec.md` §10, `docs/ROADMAP.md` §2 | **Both still say a failed validation surfaces no error.** Principle 1 contradicts them: the branch is reported. An amendment to two documents, decided by the user 2026-08-27 and carried here |
| **`done_by_self_check`, with its description** (§4.1.2) | `handoff` | **Does not exist.** Decided by the user on 2026-08-27; a `handoff` spec change, carried in this module's propagation set rather than made here |
| **Budget figures at the gate** — tokens, elapsed (§4.1.3) | `agent` | **Exists.** `ResultMessage` already carries `duration_ms`, `num_turns`, `total_cost_usd`, `usage`, `model_usage` (spec §5.1). What does not exist is a threshold to compare them against |
| **Somewhere to put a record** (§8) | `task_graph` | **Exists.** `StoreMgr` (`task_graph/store.py`), needing a new kind and no new dependency. **This row was absent from rev. 2** — see the note below |
| `Monitor` protocol, and a `PusherMonitor` default | [`../../docs/interfaces.md`](../../docs/interfaces.md) §2 | **Registered by name, defined nowhere.** This module is what defines them |
| **A task with no `monitor_spec` still has a monitor** | `task_graph` design §3.8 | **Exists** — absent takes the default. Rev. 14 makes this load-bearing rather than convenient: a task whose monitor could not be resolved would not merely go unwatched, **it would never advance a phase** |

**On the row that was missing.** §8 required a record and this table, then eight
rows long, listed no place to put one. **The table whose entire purpose is to catch
"X consumes Y, and X cannot receive Y" had that defect itself**, and it was found
by the research rather than by the table. Recorded here rather than quietly fixed,
because the failure mode is the point: a checklist does not check itself, and this
one had been written two days earlier for exactly this class of miss.

---

## 10. Acceptance criteria

1. A task with no `monitor_spec` is watched by the default monitor; a task naming
   one is watched by that one, resolved by name from the component registry.
2. **An unregistered `monitor_spec` name is rejected with the offending value
   named**, at the same point every other by-name collaborator is resolved.
3. A monitor runs its own loop, distinct from any agent's.
4. **A task does not enter `OUTPUT_VALIDATING` until the completeness gate
   passes**, and a gate failure reports an exception the monitor receives
   (§4.1.0). Four things fail it independently: a missing declared output; a
   delivered non-executable that claims to be executable; `done_by_self_check`
   not set; and a budget threshold exceeded (§4.1.3).
   **The cycle stays below the scheduler**: repeating it does not move task
   status, does not reach the scheduler, and does not call `on_done`.
   **And the runner never pushes** — every retry originates in a monitor
   decision, so no failure is handled by a policy the record cannot see.
5. **An agent whose `put` was refused is not silently identical to one that never
   called `put`.** Both leave the output absent (§4.1.1); the record distinguishes
   them, and the distinction is written producer-side or not at all.
6. **Every monitor action is observable as a transition call.** The status-write
   spy of `tests/task_graph/test_authority.py` sees no write originating in the
   monitor — the existing test extends to the new caller without amendment.
7. A monitor calling a transition while the scheduler is mid-pass blocks and then
   proceeds; it holds nothing across the call.
8. **A global monitor may transition only the task `set_task` gave it.** An
   attempt to act on another task is refused rather than silently widened.
   **The scope is the verbs, not the filesystem** — see §6.1 for why that is the
   honest form of this criterion rather than a weakened one.
9. **Every exception is recorded**, including one whose handling failed, and a
   push that produced no change is distinguishable from a push that was never
   attempted.
10. `set_task` is the only way a monitor learns what it watches.
11. **A monitor is not a task**: it holds no lease, occupies no zone of its own,
    and does not appear in the graph.
12. **The alpha's *reaction* is the pusher** — a status check plus one
    instruction — while escalation and recording still work for every exception
    it cannot fix (§7.1). The ceiling is on the response, never on the reporting.
13. **The record is a persisted value, not a log line.** The tests for §8 assert
    on records and **pass with logging disabled** — a suite that needs `caplog`
    is testing the logging configuration (§8.1).
14. **An attempt against which nothing was recorded has an empty record set, not
    a missing one** (§8.3), so "never attempted" and "the store lost it" are
    distinguishable.
15. **`report()` does not block**, and a report arriving while the monitor holds
    the scheduler's lock is handled exactly once after the current handling
    completes (§5 rules 1 and 5).
16. **Both of the validator's bad outcomes reach the monitor** (§2.1) — a verdict
    of "fail" and an unreachable verdict alike — and they are distinguishable in
    the record by `kind`. **A failed output validation leaves a record and does
    not go quiescent**, which is the defect main spec §10 recorded.
17. **A monitor that cannot resolve an exception escalates to the monitor of the
    parent task**, up to the root, and each escalation is recorded (§3.1). The
    chain follows the task tree, so it is the same for per-task and global
    monitors, and an escalation never requires a scope wider than the target
    already has.
18. **No recovery action originates outside a monitor decision.** For each
    reporter the alpha knows about — the runner's gate, the validator — the test
    is the same: the failing path calls `report()` and takes no corrective action
    of its own. A module that retried internally would satisfy every other
    criterion here and still be wrong (§2 principle 1).

### Rev. 14 — the planned channel, threads, and liveness

19. **A planned event advances the phase and does nothing else.** No action set,
    no analysis, and — for both kinds of monitor (§3) — **no model on the path**.
    The test that makes this mechanical: a monitor built with an agent handles a
    planned event through the same code as one built without, and the agent is
    never consulted.
20. **The two queues have different collapse rules, and the planned one does not
    collapse.** Two advances of one task both take effect; deduplicating them
    would skip a phase (§2.2). And **one task is never handled twice at once,
    across both queues** (§5.2 rule 5).
21. **A leaf holds exactly one thread from dispatch to `on_done`**, and its three
    phases borrow that one thread in turn.
22. **A non-leaf holds no thread while its subgraph runs.** Its thread ends at
    `unfold`; a new one is taken for output validation. **The re-entry is the same
    `Execution`** — no second execution record is pushed and no second agent is
    bound (§5.3).
23. **The scheduler is not involved in a non-leaf's re-entry.** It never reads
    `is_end`, never observes another task's status to decide this one's, and never
    dispatches a task that is in a validation phase. `task_graph` §3.2.1's "does
    not treat `is_end` specially at completion" holds unamended.
24. **A monitor transitions only its own task, on the planned channel too.** A
    subtask's monitor reports the subgraph's completion **to the parent's
    monitor**; it does not transition the parent. Criterion 8's scope guard sees
    no new violation (§5.3).
25. **An exception that escapes any thread produces a record and reaches the
    user.** The default does not: measured, a thread's uncaught exception prints
    to stderr, the thread dies, the process continues, and producers see nothing
    (§5.4).
26. **A monitor whose loop has stopped turning is detected**, after N consecutive
    stale periods rather than one, and the detection is reported. **The check
    itself is a timestamp comparison** — small enough that nothing needs to watch
    it (§5.4).

---

## 11. Open questions

What the research and the decisions did not settle. Each is here because guessing
it would be worse than leaving it visible.

| | |
|---|---|
| **When thresholds should stop being global** | §4.1.3 settles the alpha: one global setting. Per-task or per-agent-spec limits need a cost distribution the system has not produced yet, and *who may raise a limit* is unasked |
| **What the alpha does at the top of an escalation chain** | §3.1 terminates escalation at the root task's monitor, whose target is the user. **How** a monitor reaches a user is not specified anywhere in the system |
| **How long the monitor blocks on the `RLock`** | §5.2 bounds the queue's depth, not the monitor's latency. A scheduler pass calls `runner.start` for every dispatchable task while holding the lock. **Not measurable today** — `FakeRunner.start` returns immediately. **Rev. 14 raised the stakes rather than the difficulty**: this latency is now on every phase advance of every task, not only on the exception path, so the same unmeasured number is multiplied by the number of phases in the graph |
| **What happens to the tasks of a monitor that died** | §5.4 makes a dead or wedged monitor *visible*; it does not restart one, and it does not say whether the tasks it was watching can be adopted by another monitor, must be failed, or wait. Every one of those is a different answer to "the graph stopped and someone noticed" |
| **What a fingerprint is made of** | §8.2. Excluding `attempt` groups repeated failures across attempts into one issue; including it makes every fingerprint unique and grouping a no-op. A choice, not a detail |
| **Cold-cache resume cost** | §7's fallback. Warm resume is ~5.5 s; a cold one reprocesses the whole history once, and that was not measured |

### 11.1 Closed by the research step

| Was | Answer | Where |
|---|---|---|
| Can a returned agent be pushed at all | **Yes** — a `ResultMessage` ends a turn, not the session or the process | §7, `findings-monitor-push.md` |
| Can a monitor enter a task's sandbox | **It cannot, and it must not** — four independent reasons, and entering would escalate rather than scope | §6 |
| Is a tree-shaped view over a subgraph available | **Yes, and free** — it is `env_mgr`'s nested layout plus one hierarchy grant, and it is the same measurement as `env_mgr` criterion 14 | §6 |
| Is a "basic schema check" available where §4.1 needs it | **Partly the question dissolved, partly not.** README and locality cannot be wrong in storage, so presence is one `exists()`. Executability is *not* one of `put`'s checks and is genuinely the gate's | §4.1.1 |
| Is the buffer's asynchrony sound | **Yes, under five rules** — unbounded, bounded in practice by dedup, record-before-enqueue, merge-never-overwrite, and no task handled twice | §5.2 |
| Who advances a task from one phase to the next | **This module, on a second queue.** It had no owner through rev. 13 | §5.3 |
| What re-enters a non-leaf after its subgraph finishes | **Its own monitor**, told by the `is_end` subtask's monitor along `Task.parent`. **Not the scheduler** — that would be one task's completion decided by observing another's | §5.3 |
| Does an AI ever end up on the ordinary path | **No, and the queue split is what guarantees it** rather than a convention | §2.2 |
| Does an uncaught exception in a thread reach anyone | **No — measured.** Traceback to stderr, thread dies, exit code unchanged, producers see nothing. It needs a `threading.excepthook` | §5.4 |
| Is "nothing monitors the monitor" still affordable | **Not once the planned path runs through it.** Two mechanisms, both alpha: the excepthook, and a heartbeat checked against N stale periods by something trivial | §5.4 |
| Where the non-delivery check sits | **In the runner, at an admission gate between the main phase and output validation.** Nothing to do with `on_task_done`, which is after validation; an earlier revision conflated the two instants | §4.1.0 |
| Is `report()` one step or two | **One, with synchronous persistence inside it**, so rule 3 holds structurally instead of by the caller's discipline. Affordable because the call sits on no lock and no scheduler path | §5.1 |
| Is non-delivery rare | **No — it is common**, which is why the gate is a loop and not an error path | §4.1 |
| How the agent's claim of completeness is carried | **`done_by_self_check` on the handoff**, a weak check whose *description* is the instruction. Its purpose is cutting main↔validation round-trips, not catching errors | §4.1.2 |
| What bounds the gate loop | **Budget thresholds**, reported like any other gate failure. The exit is the monitor's decision, not a retry count in the runner | §4.1.3 |
| Who pushes on a gate failure | **The monitor, never the runner.** A runner that retried on its own would be a second, hidden failure policy | §4.1.0 |
| What shape does the record take | **The carrier was never open** — a persisted value, not a log line. Only the vocabulary was, and it is a three-source hybrid at zero dependency cost | §8 |
| Does the alpha need a poller for the never-returning agent | **No, and the reason is not "later"** — a wedged agent is a wedged thread, and Python cannot kill a thread, so detection would come without remedy. Build the seam only | §4.3 |
| `running` or `fixing` while an exception is open | **`running`; `fixing` to the roadmap.** It would be the first agent status written by the *monitor* rather than the runner, and agent status does not go through `_move` — so it opens "who may write agent status", which the authority rule does not cover | §4.2 |

---

## 12. Revision history

Newest first. Kept as prose blocks rather than one nested parenthetical: at
thirteen revisions the single-line form had become a 5,000-character string ten
parentheses deep, and keeping it balanced by hand had become a recurring source
of error rather than a record of anything.

### rev. 14 — 2026-08-28

**The monitor becomes the task's event loop.** Rev. 13 owned every *unplanned*
outcome and left the *planned* phase advances with no owner at all — nothing
said what wakes a runner for its second phase, and for a non-leaf nothing said
what happens after the subgraph finishes. **Two channels through one call**
(§2.2): planned advances are handled by code, always, and the unplanned channel
keeps the decision and the action set. The queues differ in their collapse rule
because deduplicating an advance skips a phase. `ExceptionRecord` becomes
`EventRecord`.

**§5.3 writes out both walks.** A leaf holds one thread for its whole dispatch;
a non-leaf holds none while its subgraph runs, and its re-entry goes
subtask-monitor → parent-monitor → `enter_phase` → a thread from the runner —
**never through the scheduler**, which would have meant one task's completion
being decided by observing another's, against `task_graph` §2 principles 2 and 4
and §3.2.1's rule on `is_end`.

**§5.4 turns "nothing monitors the monitor" from a recorded risk into two built
mechanisms**, because the planned path now depends on this module: a
`threading.excepthook` (the default is measured to be silent — a dead thread, an
unchanged exit code, producers none the wiser) and a heartbeat timestamp checked
against N stale periods. The checker is a timestamp comparison, which is the
"make the top trivial" answer s6 gives and systemd and Ray answer differently.

Criteria 19–26. The action set (§7.1) is unchanged and is now explicitly the
unplanned channel's alone.

### rev. 13 — 2026-08-27

**`answer` needs no new channel** (§7.2). Rev. 11 recorded it as unreachable
because `AgentBackend` has `instruct` and no reverse — which assumed the agent
must *push* a question. It does not: the monitor **reads** the history.
`AgentBackend.query()` while the agent lives,
`get_session_messages(session_id)` across a restart, both public in
claude-agent-sdk 0.2.145 and both mirrored by Cursor's `list_messages()`. §9's
row flips from a missing route to an existing one.

### rev. 12 — 2026-08-27

**A consistency pass over eleven revisions**, several of which reversed earlier
ones and left residue. Three real contradictions: §6 claimed the monitor
touches no filesystem while §5.1 puts a synchronous write on the recording path
— corrected to *no reach into a task's zone*, which is the precise and
still-true claim; §9 said `exists()` was the whole check while §4.1.1 says
executability is a second state `put` does not cover; and criterion 12 said the
alpha's action set is a status check plus one instruction while §7.1's alpha
column also carries escalation and recording. The rest was reference drift —
eleven actions vs twelve, "§9's single real gap" when six routes are missing,
"eight routes" in a table now sixteen rows long — plus stale pre-decision
framing in §4.1 and §4.2.

### rev. 11 — 2026-08-27

**The action set moves here in full** (§7.1). It had lived as a roadmap bullet
and been quoted in §7 as seven of its entries — an abbreviated normative list,
which is how entries go missing. **`answer` was missing from every recorded
list** and is added: it replies to a question the agent *actually asked*, which
distinguishes it from content-free `push`, unprompted *add context*, and
*report to user*, and makes it the one action the pusher structurally cannot
perform. §9 gains the consequence — **no channel exists for an agent to ask
anything**.

### rev. 10 — 2026-08-27

**Escalation is specified** (§3.1) — it had been carried only as one line in
the roadmap's eleven-action list, which names an action without giving it a
target. A monitor that cannot resolve passes up **the task tree** to the parent
task's monitor, terminating at the root, whose target is the user; the chain is
the same for per-task and global monitors, and it always moves into a scope
that already contains the reporter's. Without it, principle 1 sends every
departure to a monitor and a monitor that cannot act becomes where the plan
quietly dies. **Alpha thresholds are one global setting** (§4.1.3), because
nobody yet knows what a normal task costs. Criteria 17–18.

### rev. 9 — 2026-08-27

**The test was wrong and is replaced** (§2 principle 1). Rev. 8 asked *did a
component malfunction*; the right question is *is the graph still going to
finish as planned*. A component can work perfectly and still break the plan — a
validator returning "fail" is exactly that — so **both** of §2.1's validator
outcomes are reported, not one. This **closes** the quiescent-branch hole
rather than reopening it as a question: main spec §10 and ROADMAP §2 must
withdraw the limitation, and §9 carries the amendment. §7 says the alpha's
ceiling is on the *response*, never on the *reporting*.

### rev. 8 — 2026-08-27

**Scope, raised to the module's definition** (§1, §2 principle 1): the monitor
is the system's *single decision-maker for all exceptional work*, and any
module that cannot resolve a condition deterministically reports here rather
than growing a private recovery path. `report()`'s callers widen; its interface
does not. Criteria 16–17.

### rev. 7 — 2026-08-27

**The runner never pushes** — it reports, and the monitor decides; rev. 6's
diagram drew the retry as the runner's own edge, which would have made a second
failure policy the record cannot see (§4.1.0). The agent's claim of
completeness is **`done_by_self_check`**, a weak check whose field
*description* carries the instruction, existing to cut main↔validation
round-trips (§4.1.2). **Budget thresholds are gate failures too**, and they are
what bounds the loop — with the exit a monitor decision rather than a retry
count (§4.1.3).

### rev. 6 — 2026-08-27

**The detection point was wrong and is corrected** (§4.1.0): non-delivery is
caught by an **admission gate inside the runner, between the main phase and
output validation** — rev. 4's claim that `on_task_done` already computes it
conflated two instants separated by the whole validation phase, and is
retracted. The gate is a **loop the runner absorbs**, not an error path: this
is a *common* occurrence. It checks three things, and executability is
genuinely its own because `put` does not check it (§4.1.1). **`report()` is one
call with synchronous persistence** (§5.1). The speculative post-pass push goes
to the roadmap.

### rev. 5 — 2026-08-27

Two terms the document used without defining, both raised by the user.
**"report" meant two things** — a call and, in the borrowed Erlang vocabulary,
the record itself; §8.2.1 separates them and this spec now says "record" for
the noun. **`Context` is defined** from the SASL source: the closed enum naming
*in which phase* a child failed, which is why criterion 9 needs an enum and not
prose. Also a correction: rev. 3's justification for an unbounded buffer said
the reporter holds the scheduler's lock, and it does not — `on_task_done`
acquires the lock itself (§5.2 rule 1).

### rev. 4 — 2026-08-27

**`report()` is defined** (§5.1) — rev. 3 used the name five times without ever
introducing it, the same defect §9 exists to catch. `fixing` is decided out of
the alpha (§4.2) with its real cost recorded: it would be the first agent
status written by the monitor, and agent status does not pass through `_move`.
And §4.1 gains a measured correction — **the scheduler already computes
non-delivery** under its lock in `on_task_done` and discards it, so the open
question is ownership, not detection.

### rev. 3 — 2026-08-27

The other two research threads land. **The buffer's asynchrony is sound under
five rules** (§5), unbounded because the reporter holds the scheduler's lock.
**The record's carrier was never open** — a persisted value through `StoreMgr`,
not a log line; only the vocabulary was, and it is an OTel/OTP/Sentry hybrid at
zero dependency cost (§8). Span events for exceptions are deprecated upstream.
§4.3 takes a position on the never-returning agent instead of deferring. §9
gains the store row **it was missing** — the table that exists to catch this
class of omission had one. Criteria 13–15.

### rev. 2 — 2026-08-27

Two of §11's questions close on measurement. **The pusher works**: a
`ResultMessage` ends a turn, not the session, so §7 gains a cost-ordered action
set and a requirement on the backend's client lifetime. **The monitor needs no
environment reach and must not enter a sandbox** (§6, four probed reasons),
which makes criterion 8 a statement about verbs — §6.1 says why that is honest
rather than weakened. §9 gains two `agent`-side rows.

### rev. 1 — 2026-08-27

First revision. The mechanism moves here from
[`../../task_graph/docs/spec.md`](../../task_graph/docs/spec.md) §3.5, which
was written before this module existed; that section keeps the boundary and the
`Task.monitor_spec` field.
