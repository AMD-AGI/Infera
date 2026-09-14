# Which of the two stall endings fired is persisted nowhere

**Measured 2026-09-05 10:20–10:26 UTC by the leader, on m5's attempts 3 and 5.**
`main.py:1015` has deliberately been given two different endings, and the code
says why:

```python
# The two endings are different facts and a reader has to be able to
# act on which one happened: *nothing is running* is a graph that
# died, *somebody is waiting on a decision* names a task and a reason
# and says a human was asked for something this entry point cannot
# deliver. Reporting both as "stopped making progress" was how the
# second one read as the first for two runs.
```

The distinction is real and it is correctly drawn — **at the moment of emission.**
It survives nowhere afterwards.

## What was checked

For run `20260905T095400-d2eb3a` (m5's attempt 3), which ended on the escalation
branch:

| where a reader would look | what is there |
|---|---|
| `store/` — any file matching `escalat` | **no hits.** `_awaiting_a_decision(t, registry)` reads the in-memory registry; nothing is written. |
| `store/event/` | **3 files, no match** for either ending string. `RUN_COMPLETE` is not in the event store. |
| run root | no log file at all — `handoffs/ knowledge/ outside/ playground/ store/ zones/`, zero top-level files |
| `store/task/` | `succeeded run_profiling_mode_off`, `running run_profiling_mode_on` — **identical to what a non-escalation stall would leave** |
| the launcher's stdout | the only place. `/tmp/yihou_rung5_287_c.log:52` |

## Why it costs something

m5 read the store, saw `mode_off succeeded / mode_on running`, and concluded
*"not a deadlock, not a missing output — a real line that is quiet for longer
than 900 s"*. That reading is unavailable from the store, because **quiet alone
cannot fire this branch**: the condition is
`(not holding or blocked) and now - last_change > stall_after`, and a genuinely
running line keeps `holding` non-empty. The kill required `blocked`, and the
line that says so was in a file nobody thought to open:

```
main is waiting on a decision no one will make — the escalation reached the top
and this entry point installs a sink that records and does not answer
(nothing to push: the attempt holds no executor: it is not in its main phase).
Nothing has changed for 900 s; still in a phase:
  m2_profiling:running, main:running, run_profiling_mode_off:running
```

They then raised `--stall-after` from 900 to 3600 on the strength of the
diagnosis. **The lever happens to be right and the reason is wrong**, which is
the expensive combination: the next death will print the same escalation
message an hour later, and there is nothing in the artefacts to stop it being
read as another slow line.

## A second thing the same message exposes

The `stalled` list names **`run_profiling_mode_off:running`**, while the store —
read afterwards — says that task succeeded and `mode_on` was running. Threads
keep writing after the orchestrator returns, so **the store is strictly later
than the event**. Any reconstruction of "what was running when it died" from
the store is reconstructing a different instant.

## Shape

This is the recorded class *"a zero that is 'not recorded' rather than 'did not
happen'"*, one level up: not a missing value but a **missing discriminator**.
Both endings leave byte-identical stores. The instrument that can tell them
apart exists, is correct, and is written to a stream that is discarded unless
the operator happened to redirect it — which is a property of how someone typed
the launch line, not of the framework.

## Workaround, in force now

Every live line redirects stdout to a file (checked via `/proc/<pid>/fd/1`:
five of five). Keep doing that. To read which ending a dead run got:

```sh
grep -m1 -oE "is waiting on a decision no one will make.{0,160}|the graph stopped making progress [0-9]+ s ago" <launch-log>
```

**Do not grep the run tree for those strings.** Tried first and it matched every
run: a task that clones the repository stages `agent_sys/cli/main.py` and its
`README.md` into its workspace, so the source of the message is inside the
artefacts. Ten of fourteen recent runs "matched both endings", which is how the
contamination was caught — the answer was absurd rather than merely wrong.

## Fix, if it is ever taken

Emit the ending into the run root — `RUN_COMPLETE` already carries
`stalled_tasks` and `awaiting_decision` as structured fields, and
`awaiting_decision` being empty or not *is* the discriminator. Persisting that
one event would make the whole question a file read. Not attempted: `agent_sys/cli/`
is outside this effort's activity scope.
