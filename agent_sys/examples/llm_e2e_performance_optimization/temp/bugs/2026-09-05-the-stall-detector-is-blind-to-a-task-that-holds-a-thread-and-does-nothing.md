# The stall detector is blind to a task that holds a thread and does nothing

**This is the other arm of `main.py:1015`, and it is a different defect from the
one in `2026-09-03-the-stall-detector-ends-a-run-while-a-task-is-still-working.md`.**
That file is entirely about **false positives** — a run cut while a leaf works.
This one is a **false negative**: a run that is genuinely hung is never cut, and
holds a GPU node until `--timeout`.

Recorded 2026-09-05 by m1, from a run of my own that I killed by hand.

## The condition, and which arm fails

```python
elif (not holding or blocked) and time.monotonic() - last_change > stall_after:
```

`holding = [t for t in live if _is_running(runner, t) and not _awaiting_a_decision(t, registry)]`.

**`_is_running` is not wrong; it answers a different question than the guard
needs.** Read rather than assumed — `main.py:1155` is
`attempt is not None and attempt.is_running`, and `runner.py:420` documents
`is_running` as *"whether a thread is currently carrying this attempt"*. For an
agent that has stopped producing, **a thread genuinely is carrying the attempt**;
it is alive and making no progress. The predicate reports that truthfully, and
`holding` therefore stays non-empty for ever.

So the defect is not in `_is_running`. It is that `holding` is used as a proxy
for *progress* while measuring *occupancy*, and the two come apart exactly when
you need them not to. With `blocked` empty (the normal case for a healthy graph,
established by m4 in the companion file), the guard is `not holding`, which is
`False`, and the elapsed-time test is never reached. **`stall_after` has no
effect on this case at any value.**

## Measured, not inferred

Run `20260905T081811-6e8750` — `p4_d`, real `deploy_and_prove` on 088 cards 4–7,
launched 08:18:11 with `--stall-after 900`.

```
08:28:20   agent transcript last write
08:28:20   newest write anywhere in the zone tree (excluding the package snapshot)
08:28:20   last store event, +609 s into the run
09:11      still deploy_kit=generating; 43 minutes with no write of any kind
           orchestrator alive; no cut; --stall-after 900 never fired
```

`runprobe.py` on that run's store:

```
1. escalations reaching the user: 0   (all escalation hops: 0)
   -> `blocked` was EMPTY as of the last record.
3. latest event +609s into the run
   tasks NOT terminal: deploy_and_prove[running], m1_deploy[running],
   m2_profiling[waiting_handoff], m3_analysis[waiting_handoff],
   m4_kernel_opt[waiting_handoff], m5_integration[waiting_handoff], main[running]
```

Seven tasks, no status change for 43 minutes, no escalation. `_snapshot` is
`(id, status, len(history))` per task, so it was constant throughout and
`last_change` was 43 minutes stale — **more than 2.8× the 900 s threshold.**
`blocked` empty, so the only way the guard stayed `False` is `holding` being
non-empty. **The hung task was counted as working.**

**The one residual, stated:** `last_change` is computed from the live task
manager, not from the store, so strictly I measured that no *store event*
occurred, not that `len(t.history)` never moved. History appends are events, so
the two should coincide; I did not verify that they must.

## Why this is not the companion file's bug turned around

The companion file's open question is *"has a leaf that is legitimately
executing ever been cut?"* — nobody has an instance. **This is not an instance
of that and does not bear on it.** Here nothing was executing; the detector's
error is in the opposite direction, and the two arms fail on different inputs:

| | `holding` | `blocked` | elapsed | outcome |
|---|---|---|---|---|
| companion file | non-empty (working) | non-empty | > threshold | cut — **wrong** |
| this file | non-empty (**hung**) | empty | > threshold | not cut — **wrong** |

Fixing one does not fix the other. Making the code implement its own docstring
(`and no attempt holds a thread`) — the companion file's recommendation —
**makes this case strictly worse**, because it removes the `or blocked` arm that
is the only path to a cut when a task is stuck holding.

## What it cost, and what it will cost

Forty-three minutes of four MI355X cards on a node under an 8 h wall, plus the
time to notice. Nothing recovered it: with `--timeout` unset the run would have
sat there for the 4 h default. **On this effort the detector has never once
ended a hung run; every hang so far has been ended by a person watching a log.**

The stage that will meet this next is `optimize_kernel` — hours long and quiet
by construction — where "quiet because working" and "quiet because hung" are
indistinguishable from outside, and where the companion file already notes the
re-run cost is highest.

## What a fix would need, and what it is not

**Not a threshold.** No value of `stall_after` reaches this branch.

**Not progress output.** The companion file already closes that door: the
detector watches the task table, and a body cannot change the task table from
inside itself.

What would actually separate the two cases is a liveness signal `_is_running`
does not currently consult — the attempt's own last-activity timestamp, or the
executor's. That is a runner design question and it is the runner owner's to
answer.

**Not fixed here.** `agent_sys/cli/` is outside this effort's activity scope.
Recorded with the line number, the predicate, the run id, and the probe output
so whoever owns the runner has the whole case.

## The workaround in force

None inside the package. Operationally: a run is polled by hand, and the
signature to look for is **the newest write in the zone tree, excluding
`package/`** — the package snapshot is copied per task at dispatch and its
mtimes are recent on a run that has not written anything for an hour, which is
how a hung run can look busy to `ls -t`.
