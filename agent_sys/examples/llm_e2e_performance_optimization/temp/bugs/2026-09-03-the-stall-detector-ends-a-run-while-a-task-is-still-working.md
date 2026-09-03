# The stall detector ends a run while a task is still working

**The shape, before the line number** — m1's framing, and it is what survives
this call site:

> `blocked` is **permanently true in this package by design** — `main` sits
> awaiting a decision the `NullUserSink` will never answer — so the guard
> reduces to *"nothing has changed for 20 s"*, and **any healthy operation that
> is quiet for 20 s is indistinguishable from a hang.**

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
