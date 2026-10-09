# `monitor` — the task's event loop

| | |
|---|---|
| Implements | [`docs/spec.md`](docs/spec.md) rev. 14, [`docs/design.md`](docs/design.md) rev. 2 |
| Seam | [`../docs/interfaces.md`](../docs/interfaces.md) §4.9. Imports `task_graph` and **nothing else of ours** |
| Tests | `../tests/monitor/` (98), plus `../tests/interfaces/` — `test_pushable.py`, `test_runner_seam.py`, `test_import_rules.py`, `test_stub_agreement.py`, `test_declaration_conformance.py` |

Everything that happens to a task and is not the task's own work arrives here
through **one call, `report`**, and lands on one of two queues:

| | Handled by | Collapsed by task id |
|---|---|---|
| **planned** — a phase finished, a subgraph finished | **code, always.** No model, ever | **no** — merging two advances skips a phase |
| **unplanned** — a gate failure, a budget overrun, either validator outcome, an escalation | a decision. The alpha's is the pusher | **yes** — that is what bounds the queue |

The routing is `kind in PLANNED`, consulted in one place. **A reporter never
classifies what it is reporting**, which is what lets the completeness gate call
the same method whether it passed or failed.

## Files

| | |
|---|---|
| `protocols.py` / `.pyi` | the frozen seam: `Monitor`, `Recorder`, `UserSink`, `EventRecord`, `EventKind`, `PLANNED`, `Budget`, `BufferClosed`, `ScopeViolation` — plus the three shapes this package *requires* of `agent` and declares locally rather than import: `Pushable`, `Attempt`, `AttemptRunner` |
| `record.py` | `EventId`, the concrete `EventRecord`, `default_fingerprint`, `rekeyed`, and the `Recorder` over `StoreMgr` |
| `buffer.py` | `ExceptionBuffer` (dedup + merge), `PlannedQueue` (FIFO, no collapse), `Unit` |
| `base.py` | `BaseMonitor` — the loop, `report`'s routing, `_advance`, `_escalate`, the scope guard — plus `next_phase`, `monitor_for`, `check_liveness`, `install_excepthook`, and `start_monitors` / `RunningMonitors` |
| `pusher.py` | `PusherMonitor.decide`, the alpha's decision function |

`record.py` may not import `buffer.py`: **the record survives the buffer**
(spec §5.2 rule 3), and the import graph is what makes that structural.

---

## Libraries adopted, and why

**Net new runtime dependencies: none.** Each row below is a candidate that was
considered and settled, per design §10.

| Piece | Decision | Why |
|---|---|---|
| **Kubernetes `client-go/util/workqueue`** — the dedup/merge queue | **copy the shape, ~60 lines** | It is Go. The shape is three collections (`_order`, `_dirty`, `_processing`) and one invariant; taking a dependency on a Kubernetes client to get it would be absurd, and the invariant is one assertion in a test. **One thing is deliberately *not* inherited**: `client-go` collapses last-wins and drops pending items on shutdown, and criterion 9 requires every occurrence to survive |
| **`queue.Queue`** (stdlib) | **no** | `shutdown` is **3.13-only** against a 3.10 target, and its `immediate=True` discards pending items silently. Neither dedup nor merge is expressible on it, and a `maxsize` is the one thing rule 1 forbids |
| **OpenTelemetry SDK** | **no — the names only** | Three packages, and it answers the wrong question: **OTel is emit-only by construction**, and `PusherMonitor.decide` must *read back* what it wrote to know a push already failed. The stable `exception.*` field names and `SeverityNumber` cost nothing to adopt as naming. Also: recording exceptions on **span events is Deprecated** as of semconv v1.40.0 (PR #3256, merged 2026-01-28), so the obvious shape is no longer one |
| **Sentry SDK** | **no — the split only** | The event / fingerprint / issue idea is the value; the client is a network reporter |
| **A logging library** | **no** | Spec §8.1: **the carrier was never open.** A record is a persisted value written through `task_graph`'s `StoreMgr`, and a test that satisfies criterion 9 with `caplog` is testing the logging configuration. Logging is a *projection* of the record — one handler, rendering at the severity the record already carries |
| **pydantic v2** | **yes**, already declared | `EventRecord` subclasses `task_graph.models.Model`, which gives `extra="forbid"`, `validate_assignment=True` and the id serialisation for free, and keeps **one** pydantic configuration in the system rather than two |
| **`threading.Condition`** (stdlib) | **yes** | The buffer needs exactly one wait/notify |
| **`threading.excepthook`** (stdlib) | **yes** | Measured: a thread whose target raises prints a traceback and dies with **the exit code unchanged and every producer seeing nothing**. The hook is the whole of criterion 25 |

---

## Two things a reader will trip over

**`EventRecord` and `Recorder` are each declared twice, on purpose.**
`protocols.py` holds the *seam* — what `agent` and `validator` are written
against — and `record.py` holds the implementation that satisfies it.
`monitor/__init__.py` re-exports the **implementation**, because a reporter has
to *build* a record rather than only accept one.

**Three Protocols duplicate part of `agent`, and that is the design.** The
monitor needs `instruct` on a live agent, an attempt's liveness, and two verbs on
the runner; the runner needs `report` from here. Written the obvious way that is
a package cycle, so the monitor declares the parts it uses locally and `agent`
satisfies them structurally.

| declared here | satisfied by | guarded by |
|---|---|---|
| `Pushable` | `agent.AgentBackend` | `tests/interfaces/test_pushable.py` |
| `Attempt` | `agent.TaskAttempt` | `tests/interfaces/test_runner_seam.py` |
| `AttemptRunner` | `agent.Runner` | `tests/interfaces/test_runner_seam.py` |

The cost is two declarations of one shape with nothing in either package noticing
if they diverge, and the tests are what notice. **They are not a nicety attached
to the decision; they are the decision's price.**

**The second and third rows are there because the gap cost a defect.** `Attempt`
and `AttemptRunner` were undeclared until `_advance` branched on
`attempt_of(tid) is None` — an attempt *survives its thread*, so the branch never
fired, `wake()` set an Event nobody was waiting on, and a non-leaf stalled
silently. Four members across two objects, checked by nothing, at the one seam
`interfaces.md` §4.9 describes only as "resolves `runner`". Found by `closure`'s
cold review, which named the class rather than the instance.

---

## What is built, and what is waiting on a neighbour

| | |
|---|---|
| **Built and green** | the record and its two store kinds, both queues, the loop and its guard, the scope guard, the planned advance, escalation, the pusher's decision function, and both liveness mechanisms |
| **Landed** — `Runner.attempt_of` / `carry_on` | Declared in `agent/protocols.py`, so the seam is checkable rather than duck-typed, and `test_runner_seam.py` pins it from both sides. `live_handle` calls `attempt_of` **directly**: the `getattr(..., None)` it carried while `agent` was declaration-only outlived its reason the day the method landed, and what was left made a renamed accessor indistinguishable from "no live agent" — `interfaces.md` §4.11's first row, which cost `handoff` an unreported admission with three suites green |
| **Withdrawn** — `Runner.resume` | `carry_on` subsumed it and this package stopped requiring it, leaving a published verb nobody invoked — §4.12's family on `agent`'s surface. Removed by agreement rather than unilaterally, since the shrink here is what orphaned it. **A different `resume` is untouched**: the claude-agent-sdk *session* resume of spec §7's cost table, push/resume/restart's middle rung at ~5.5 s warm |
| **Resolved by name, and now bound** | `PHASE_ORDER` names `INPUT_VALIDATING → RUNNING → OUTPUT_VALIDATING` and `next_phase` looks the successor up **on whichever enum the status came from** — which is why the planned channel was testable before `task_graph` rev. 12 existed and bound to the real `TaskStatus` with no change when it landed. All three members are present as of rev. 12, verified; `test_planned.py::test_phase_order_names_are_task_graphs_when_they_exist` asserts **all three or none**, so a rename fails here and in `task_graph` at once |
| **Not this package's, and its tests say so** | the completeness gate (`agent`), the validator's route (`validator`), the thread a `TaskAttempt` owns (`agent`). `test_gate.py`, `test_validator_route.py` and `test_threads.py` each test the half that is here and name the half that is not |

---

## Every acceptance criterion, and the test that holds it

`docs/spec.md` §10, all 26. **Verified against the tree**, not transcribed: every
name below exists in `../tests/monitor/`. 40 test references over 26 criteria; the
suite has 98 tests, so most of it is not in this table — the rest guards the
invariant, the prior art's sharp edges, and the four defects found by review.

| # | What it requires | Test |
|---|---|---|
| 1 | default vs named monitor, resolved by name | `test_registry.py::test_monitor_spec_resolves_by_name` |
| 2 | an unregistered name is rejected, value named | `test_registry.py::test_unknown_monitor_spec_names_the_value` |
| 3 | its own loop, distinct from any agent's | `test_loop.py::test_mainloop_is_not_the_agents` |
| 4 † | the gate blocks `OUTPUT_VALIDATING`; four failures; cycle below the scheduler; the runner never pushes | `test_gate.py::test_four_failures_each_report`, `::test_no_status_move_during_cycle`, `::test_runner_takes_no_corrective_action` |
| 5 † | a refused `put` ≠ never called | `test_record.py::test_absent_output_kind_does_not_claim_malformed` |
| 6 | every action is a transition **call**, never a status assigned | `test_scope.py::test_every_action_is_a_transition_call` |
| 7 | blocks on the lock, then proceeds, holding nothing | `test_concurrency.py::test_transition_blocks_then_proceeds` |
| 8 | transitions only the task `set_task` gave it | `test_scope.py::test_global_monitor_refuses_another_task` |
| 9 | every exception recorded; attempted vs ineffective vs never | `test_record.py::test_push_attempted_ineffective_never`, `test_loop.py::test_handler_exception_does_not_kill_the_loop` |
| 10 | `set_task` is the only way it learns what it watches | `test_scope.py::test_no_discovery_without_set_task` |
| 11 | a monitor is not a task | `test_identity.py::test_monitor_holds_no_lease_no_zone_and_is_not_in_the_graph` |
| 12 | the alpha's reaction is the pusher; recording and escalation still work | `test_pusher.py::test_decision_table_pushes_a_live_agent`, `test_escalation.py::test_unpushable_still_records` |
| 13 | records, not log lines — passes with logging disabled | `test_record.py::test_suite_passes_with_logging_disabled` |
| 14 | an empty record set, not a missing one | `test_record.py::test_open_creates_an_empty_set` |
| 15 | `report()` does not block; handled exactly once | `test_buffer.py::test_add_never_blocks`, `::test_requeued_exactly_once_while_processing` |
| 16 | both validator outcomes arrive, distinguishable by `kind`; a failure is not quiescent | `test_validator_route.py::test_fail_and_unreached_are_different_kinds`, `::test_a_failed_validation_leaves_a_record_and_does_not_go_quiescent` |
| 17 | escalates up the parent chain to the root, each step recorded | `test_escalation.py::test_walks_parent_chain_to_root` |
| 18 † | no recovery originates outside a monitor decision | `test_gate.py::test_runner_takes_no_corrective_action`, `test_validator_route.py::test_validator_does_not_rerun_itself` |
| 19 | a planned event advances and does nothing else; **no model on the path** | `test_planned.py::test_advance_is_one_transition`, `::test_agent_monitor_uses_the_same_advance`, `::test_advance_cannot_be_overridden_by_accident` |
| 20 | the planned queue does not collapse; one task not handled twice across both | `test_buffer.py::test_planned_queue_never_collapses`, `test_loop.py::test_one_task_one_handling_across_queues` |
| 21 † | a leaf's phases borrow one thread in turn | `test_threads.py::test_leaf_holds_one_thread`, `::test_a_parked_leaf_is_woken_not_resumed` |
| 22 † | a non-leaf holds none during its subgraph; the re-entry is the same `Execution` | `test_threads.py::test_non_leaf_holds_no_thread`, `::test_a_released_attempt_is_resumed_not_woken`, `test_planned.py::test_reentry_pushes_no_second_execution` |
| 23 | the scheduler is not in the re-entry | `test_planned.py::test_scheduler_untouched_by_reentry` |
| 24 | a subtask's monitor **reports** the subgraph's completion to the parent's, and does not transition it | `test_planned.py::test_an_is_end_subtask_announces_its_subgraphs_completion`, `::test_a_subtask_that_is_not_the_end_announces_nothing`, `::test_the_root_completing_announces_to_nobody`, `::test_subtask_monitor_does_not_transition_parent` |
| 25 | an escaped thread exception produces a record and reaches the user | `test_liveness.py::test_excepthook_records_and_surfaces` |
| 26 | a stalled loop detected after N stale periods, not one | `test_liveness.py::test_stall_needs_n_consecutive`, `::test_one_slow_round_is_not_a_stall` |

**† — this module's half only.** The gate is `agent`'s, the validator's route is
`validator`'s, and the OS thread a `TaskAttempt` owns is `agent`'s. Each of those
tests names in its docstring which half is here and which is not; **do not read a
green † row as end-to-end coverage.**

**Row 24 was wrong for weeks, and the way it was wrong is worth keeping.** It
named one test — `test_subtask_monitor_does_not_transition_parent` — which
passed throughout while **criterion 24's producing half did not exist**:
`SUBGRAPH_DONE` was declared, consumed and re-emitted, and nothing anywhere
created the first one. That test **constructs the record by hand**, so it was
itself standing in for the missing production. It proved the relay forwards,
which was true, and was structurally incapable of noticing that nobody upstream
ever spoke. Every non-leaf, the root included, sat in `RUNNING` for ever
(`04a5b76`; first green end-to-end run 2026-08-29, `main: succeeded`).

The row now names the producer first. **A criterion with a verb in it needs a
test per verb** — 24 says *reports* and *does not transition*, and only the
second was covered. And the check that finds this class is *"for each event
kind, who writes it?"* — then, of each write site, *"is its input already a
record of that kind?"*, because a re-emission reads as a write
(`scratch/impl-2026-08/monitor/p16_every_kind_has_a_producer.py`; the other
fourteen kinds are clean).

---

## A live spec conflict, recorded where the code is

**`monitor` spec §4.1.1 and `env_mgr` spec §4.5 cannot both be satisfied.** With
the user; unresolved at the time of writing. Both quoted verbatim, checked
against the files:

| | |
|---|---|
| `monitor` §4.1.1 | *"`handoff` design §5.3: 'The producing agent works in its playground and hands the store a directory.' **The producer calls `put`, from inside its own zone** — so a `Malformed` is raised inside the agent's execution and never reaches the runner, which sees only the later absence."* |
| `env_mgr` §4.5 | *"**Write** — A task's executor may not write outside its zones. Local or remote, no exception."* |

`put` writes to the handoff store, which is not the producer's zone. So either
the producer is not the caller, or the store root is in its granted write set —
and §4.5 admits no exception.

**What depends on the answer, and it is criterion 5.** *"An agent whose `put` was
refused is not silently identical to one that never called `put`."* §4.1.1's
reasoning is that the distinction can only be captured **producer-side, at the
moment `put` refuses**, because a `Malformed` raised inside the zone never
reaches the runner and an absence cannot be told from a never-attempt. If the
resolution moves the caller out of the zone, **that reasoning goes with it** —
the raise becomes visible to whoever calls `put`, and criterion 5's positive half
lands on a different component than `agent`.

Nothing in this package changes either way: the negative half — that
`OUTPUT_ABSENT` must not be read as "malformed", and that no `OUTPUT_MALFORMED`
kind exists — is `test_record.py::test_absent_output_kind_does_not_claim_malformed`
and holds under both readings.

## Two other things carried, not resolved

| | |
|---|---|
| **`VALIDATION_UNREACHED` is wider than spec §2.1's wording** | §2.1 lists three examples that all mean *the validator could not decide*; `agent` catches **any** exception out of `run_phase`, so the kind also covers *the system could not set up* (`PrepareRefused`, `OSError`, `SpecNotFound`). Ruled for deliberately — a type split is unavailable (`agent` may not import `validator` or `env_mgr`), the narrow form excludes §2.1's own third example (*"its own inputs were missing"*, a `KeyError` from `handoff_mgr`), and `validator`'s raise inventory went stale within an hour of being written. **If it ever needs separating the discriminator is `exception_type`, already on every record** — no new kind. With the user |
| **`_crash` reports `HANDLING_FAILED`** | That kind means *the monitor's own handler raised*; `agent` uses it for a dying attempt, which is not that. A wrong value flowing on, **left alone knowingly**: the behaviour is defensible (`GiveUp`, and the task is already terminal via `on_done(FAILED)`) and the cost is unmeasured, so a frozen enum is not widened for a naming problem |
