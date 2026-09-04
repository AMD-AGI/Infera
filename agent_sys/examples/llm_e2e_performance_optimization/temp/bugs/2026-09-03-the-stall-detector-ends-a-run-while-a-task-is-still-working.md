# The stall detector ends a run while a task is still working

**The shape, before the line number** — m1's framing, and it is what survives
this call site:

> `blocked` is **permanently true in this package by design** — `main` sits
> awaiting a decision the `NullUserSink` will never answer — so the guard
> reduces to *"nothing has changed for 20 s"*, and **any healthy operation that
> is quiet for 20 s is indistinguishable from a hang.**

> ⚠️ **"Permanently true" is false, measured 2026-09-04 — see the last section.**
> `blocked` is empty in a clean run and becomes non-empty only once something
> escalates. The quote is left standing because the *rest* of it holds: once
> `blocked` is non-empty, a healthy operation quiet for 20 s is indeed
> indistinguishable from a hang, and that is the defect. Only the "permanently"
> is wrong, and it is wrong in the direction that makes the bug look worse than
> it is.

`build_workset` is quiet for minutes by construction. So is an AI node thinking,
which is how this already happened once before with a different bound.
**Both times a bound chosen for the failure case was applied to the success
case.**

**Measured 2026-09-03.** `agent_sys/cli/main.py:1015`:

```python
elif (not holding or blocked) and time.monotonic() - last_change > stall_after:
```

The comment immediately below justifies the guard for one case only:

> **Stalled, not running.** … No status has changed and **no attempt holds a
> thread**: nothing is going to happen.

That reasoning covers `not holding`. **The `or blocked` arm fires even when an
attempt *does* hold a thread** — so a graph in which the root is awaiting a
decision nobody will answer is torn down after `stall_after` seconds *while a
leaf is legitimately executing*.

## What it cost

`build_workset` (m3's stage) runs, per mission 3.2.7's protocol floor, **three
shapes × five groups × ten iterations** inside a container on the node. That is
minutes, and — as m3 measured independently — it is **silent** for its whole
duration: container start, torch import, correctness, then timing.

Every rung-0 attempt ended the same way, ~20 s after `build_workset` began:

```
done  main is waiting on a decision no one will make — the escalation reached
      the top and this entry point installs a sink that records and does not
      answer. Nothing has changed for 20 s; still in a phase:
      build_workset:running, m3_analysis:running, main:running
```

**`--timeout` does not help and the CLI says so itself:** *"A run that stops
making progress ends in seconds regardless; this only bounds one that never
stops."* Passing `--timeout 3000` changed nothing; the run lasted ~3 minutes.

**Two owners lost time to this before it was located.** m1 believed their own
900 s watchdog had cut the stage and told m3 so, in good faith; the same cut
happened at 3000 s. The stage was never near either bound.

## `--resume` does not recover it

```
final  build_workset: failed — HandoffStateError: 657bcbde-… v0 is already
       open by task 356505d8-…
```

The cut leaves the output handoff slot **open**, and the resumed attempt of the
same task cannot re-open it. So the one recovery path the CLI offers is closed
by the same event. A fresh run re-does the eight upstream handoffs and is then
cut at the same place.

## What is *not* the fix

**Shortening the measurement.** Five groups and ten iterations are mission
3.2.7's floor, declared in `steps/m3_analysis.yaml:136` as `min_groups: 5`.
Reducing them to fit the runner's patience would be tuning the evidence to suit
the instrument — and `check_workset_runs` re-measures, so it would have to be
weakened in step. **A number produced to make a run finish is not a measurement.**

## The code contradicts its own docstring, and that is the unambiguous part

`_watch`'s docstring (`main.py:924`) states the rule the loop is supposed to
implement:

> the loop also ends when no task's status or attempt count has moved for
> `stall_after` **and no attempt holds a thread**

The implementation is `(not holding or blocked)`. **`blocked` alone satisfies
it, with a thread held.** So the documented conjunction is a disjunction in
code, and the arm that differs is the one that fired every time today.

`stall_after` is a function parameter defaulting to `20.0` and is **not exposed
on the CLI**, so there is no operator-side knob.

The same docstring records the precedent, which is what makes this a recurrence
rather than a novelty:

> **`timeout` was 300 s and it killed the first real model call in this
> repository**, six seconds after the agent finished writing its answer … The
> run reported *"did not settle within 300 s"*, which reads as a hang and was a
> healthy conversation.

A healthy long operation reported as a stall, by a bound chosen for the failure
case. Then it was an AI node's thinking time; now it is a measurement's.

**Not fixed here.** `agent_sys/cli/` is outside this effort's activity scope
(`.claude/CLAUDE.md`: `examples/llm_e2e_performance_optimization/` plus
root-level notes). Recorded with the line number and the docstring so whoever
owns the runner has everything needed.

## The workaround a reader will try first does not work

**Making the stage talk does not help, and it is worth closing that door
explicitly** because it is the obvious move and I made it before checking.

`last_change` is reset only when `_snapshot` changes, and `_snapshot`
(`main.py:1140-1147`) is:

```python
tuple(sorted((str(t.id), t.status.value, len(t.history)) for t in task_mgr.all()))
```

**Task id, status, attempt count. Nothing else.** So a leaf that prints progress
to stdout every ten seconds resets nothing: its status is `running` throughout
and its attempt count does not move. The detector is not watching output, it is
watching the task table — and a body cannot change the task table from inside
itself.

I committed progress output to `measure_in_container.sh` (`6afa0e9`) reasoning
that *"a watchdog keying on silence cannot tell a slow measurement from a
wedge"*. That reasoning is **wrong for this watchdog**; it keys on task state,
not on stdout. The commit is still worth having for the other reason in its
message — a person reading the log can now tell a slow measurement from a
stopped one — but it does not address this bug and should not be mistaken for a
mitigation.

**So there is no workaround available from inside a task package.** Not a longer
`--timeout`, not progress output, not `--resume`, and not a shorter measurement.
The knob is `stall_after`, it is a function default, and it is not exposed on the
CLI.

## Scope: this gets worse up the ladder, not better

`build_workset` is the *shortest* of the long stages. Every rung above it has a
leaf that runs longer and quieter:

| rung | stage | quiet stretch |
|---|---|---|
| 3 | `build_workset` | minutes — three shapes, five groups, ten iterations |
| 4 | `optimize_kernel` | **a KernelForge campaign — hours** |
| 5 | `integrate_and_verify` | two bring-ups and two load tests |

Whatever is done about this, it has to be settled before rung 4 rather than
after, because the stage that will meet it next is the one that costs the most
to re-run.

## Uncertainty, stated

Whether `holding` was true at the moment of the cut is inferred rather than
read: the run printed `build_workset:running`, and `running` is a phase state
with a subprocess behind it. The `blocked` half is certain — the message only
exists on that branch. What is unambiguous is the **behaviour**: a task in a
running phase, doing real work, was torn down 20 s in.

Recorded per mission rule: bug recorded first, worked around second, fixed only
on unambiguous evidence. Pairs with
`2026-09-03-a-validators-stdout-is-not-kept-anywhere.md` and `todo.md` **T14** —
three ways the runner loses a reason. Here the reason survived, in the run log,
and was the one thing that made this findable in a single grep.

---

## What is actually wrong with the guard, and what I got wrong twice getting here

**Rewritten 2026-09-04 by m4, replacing two earlier appended sections of mine
and one of the leader's.** Both of mine were wrong, in opposite directions, and
a reader should not have to reconcile a chain of qualifications to find that
out. This is the single corrected account.

### The defect, which has not changed

`main.py:1015` is `(not holding or blocked)`. `_watch`'s own docstring
(`main.py:924`) says the loop ends when nothing has moved **"and no attempt
holds a thread"**. That is a conjunction; the code is a disjunction. **So once
`blocked` is non-empty, a leaf that is genuinely executing is torn down 20 s
later** — the documented condition never has to hold.

That is m1's original finding and nothing below touches it. It is worth fixing
on its own terms.

### What `blocked` actually is — and it is not permanent

`blocked = [t for t in live if _awaiting_a_decision(t, registry)]`, and
`_awaiting_a_decision` reads an escalation **record**. Escalations are raised by
`monitor/base.py:731-737`, where `decide()` returns `Escalate` **on a unit
event**: something happens to a task. There is no structural, from-t=0 case.

The string this file leaned on —

```
nothing to push: the attempt holds no executor: it is not in its main phase
```

— is **why the escalation stopped at the root, not why it started.** It
describes the end of the walk. I read it as the cause, and so did the leader.

**The measurement that settles it.** Run `20260904T062414-be315b` (rung 1, real
m1), checked 26 minutes in:

```
started 06:24:14      checked 06:50:11      = 26 minutes
no file in the run tree modified in the last 6 minutes   -> genuinely quiet
grep for an escalation record in the store               -> NOTHING
```

**No escalation record exists at all.** `blocked` is empty, a leaf is holding,
`(not holding or blocked)` is `False`, and the run is untouched. Which is what
the code's own comment claims: *"A healthy run has no such escalation and is
untouched."*

### Whose observation this was

**Not mine unaided, and the file should not imply it.** The leader wrote that I
"went and got the fact that settles it". I queried the store *because they told
me rung 1 had run 21 minutes without a cut*. Left to myself I had already
stopped, satisfied, with the wrong answer.

Their half was symmetrical: they had that run in front of them for 21 minutes
and read it as *"the detector has not fired yet"* rather than as *"this
contradicts what we believe"*. It took their observation to prompt the query and
the query to turn their observation into evidence. **Neither of us got there
alone**, and the transferable lesson is not "check the store" — it is that a
question one person has closed is reopened by someone else's stray observation,
so say the stray thing out loud.

### How I got it wrong, which is the part worth keeping

I had the correct reading from the code **first** — a healthy run has no such
escalation — and then abandoned it.

What made me abandon it: rung 0's logs show the cut arriving with fifteen lines
of normal progress above it and no visible failure. I concluded *"nothing failed,
therefore the escalation is structural."*

**That is absence of evidence read as evidence of absence.** An escalation
trigger does not have to print as a failure in that log. A log that does not
show a trigger is not a log that shows there was none — and the store, which
records escalations directly, was two commands away the whole time and answers
the question the log cannot.

I then reported the wrong conclusion as settled, on the one question the leader
had said they could not decide. **The error was not the hypothesis; it was
treating a log's silence as a measurement.**

### Rung 4: open, and nobody knows

The earlier version of this section claimed a KernelForge campaign would be cut
~20 s in and that this was "settled by reading, not by spending a node". **That
claim is withdrawn.**

What is known:

- the cut requires something to have escalated **earlier in that same run**;
- rung 0's runs had such an escalation by the time `build_workset` started;
- rung 1 has none at 26 minutes.

What is not known: **whether a run that reaches m4 cleanly will have escalated
by then.** Nobody has measured it, and it is the only thing that decides whether
a campaign is exposed. Left explicitly open rather than guessed — the previous
two versions of this section each guessed, in opposite directions.

The cheap way to find out, when someone wants it: check the store for an
escalation record at the moment a run reaches the stage, rather than inferring
from the log.

### What this means for the fix

The earlier "the fix is a conflation, not a threshold" framing was built on the
structural premise and goes with it. With escalation event-triggered, there is
no structural case for `blocked` to be conflated with.

What stands is narrower and does not depend on any of the above: **the code
should implement its own docstring.** `and no attempt holds a thread` is the
documented rule; `or blocked` is not it. A leaf holding a thread and doing work
is not a stalled run, whatever else is true of the graph.

Whether that is the right rule is the runner owner's call. `agent_sys/cli/` is
outside this effort's scope, and this file's job is to hand over the line
number, the docstring, the mechanism, and — now — an honest account of which
parts were measured and which were inferred.

---

## The founding instance is not an instance of the bug

**Appended 2026-09-04 by m4, from the first real use of
`assets/lib/runprobe.py`** — which the leader asked for precisely so this
question could be asked in one command instead of a hand-composed grep. It paid
for itself immediately and against its author's expectations.

Run `20260903T172821-6a3c24`, the rung-0 run this whole file is written about.
The store, at **+179 s**:

```
17:31:21.321  build_workset   output_absent   declared output 657bcbde-… was never delivered
17:31:21.340  build_workset   escalated       nothing to push: the executor is a program body:
                                              there is no agent to instruct
17:31:21.368  m3_analysis     escalated       (hop)
17:31:21.397  main            escalated       (hop)
17:31:21.403  main            escalated       -> target: user
```

**The trigger is `build_workset` failing to deliver its declared output**, and
the escalation follows it by nineteen milliseconds. The monitor had no action
because the executor is a *program body* — there is no agent to instruct — so it
escalated, the walk ran out of task tree, and `blocked` became non-empty.

That is a mock stage failing on a login node with no torch, which is the failure
`RUN-PLAN.md` already documents as **correct and by design**.

**So the sentence this file rests on —** *"a task in a running phase, doing real
work, was torn down 20 s in"* **— is not what the store shows.** The task had
already failed. `blocked` became non-empty *because* of that failure, and the
run ending afterwards is the detector reporting that nothing further was going
to happen, which was true.

**What remains, and it is now smaller and unobserved:** `(not holding or
blocked)` still admits tearing down a leaf that *is* working, because `blocked`
alone satisfies it. That is a real discrepancy against the docstring's
conjunction. **But there is no longer a known instance of it happening.** Every
cut in this file traces to a run in which something had genuinely failed first.

**Left open rather than closed in the other direction, because that is the
mistake this file has now made twice.** It is not established that the defect
cannot bite — only that it has not been seen to. What would settle it: a run
where `runprobe.py` shows `blocked` non-empty while a leaf is legitimately
executing, and the run is cut anyway. Nobody has that.

**And the honest reading of "what it cost" above** is that two owners lost time
to a *correctly failing* mock stage whose failure was reported as a stall — the
detector's *message* named the wrong cause, which is the
`2026-09-03-a-validators-stdout-is-not-kept-anywhere.md` family, not this one.
The cost was real; the attribution was not.
