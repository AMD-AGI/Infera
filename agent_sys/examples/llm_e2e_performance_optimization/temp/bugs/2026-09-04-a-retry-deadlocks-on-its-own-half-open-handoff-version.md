# A retry deadlocks on its own half-open handoff version, and the monitor has no action for it

Found 2026-09-04 by the checkpoint writer, in **yesterday's** data, while
checking an unrelated claim. Filed as its own record at the leader's request,
because three distinct failures at the `build_workset` closure had already been
merged into one narrative once and the merge cost a day of wrong attribution.

**This is the third failure at that closure.** The other two are
`2026-09-03-the-stall-detector-ends-a-run-while-a-task-is-still-working.md`
(now carrying m4's retraction) and the escalation that reaches the top and finds
nobody. This one is separate from both and has not been diagnosed by anyone.

## What happens

Run `20260903T172821-6a3c24`, rung 0, task `356505d8` (`build_workset`,
declared output `657bcbde` = `operator_workset`). Read with
`assets/lib/read_events.py <run> --task 356505d8`:

```
17:31:07.532  phase_done       INPUT_VALIDATING finished
17:31:21.321  output_absent    declared output 657bcbde was never delivered
17:31:21.340  escalated        nothing to push: the executor is a program body: there is no agent to instruct
17:34:21.594  phase_done       INPUT_VALIDATING finished          <- retry, 3 min later
17:34:40.980  handling_failed  657bcbde v0 is already open by task 356505d8-be03-4d2c-a988-183fe44c7f59
17:34:40.988  monitor_gave_up  the pusher has no action for handling_failed
```

The retry is blocked **by the first attempt's own leftover**: `v0` of the output
handoff is still open, held by the same task id that is now retrying. The
monitor then reports it has **no action for `handling_failed`** and stops. So
the recovery path exists, fires, and terminates without recovering.

## The state underneath it

The on-disk handoff tree and the store index **disagree**, and that is the part
worth fixing.

```
handoffs/657bcbde-…/          store/handoff/657bcbde-….json
  v0/  claim/ content/ (0 files)   versions: [ {version 0, status "generating",
  v1/  claim/ content/ (39 files)                timestamp 17:31:20.488955Z,
  v2/  claim/ content/ (0 files)                 producer_task 356505d8,
  v3/  claim/ content/ (0 files)                 producer_agent 89f40d2c…} ]
```

**Four version directories on disk; one version in the store.** The store's
single entry is `v0`, stuck in `generating` — which is exactly the thing the
retry then trips over.

## What is established, and what is not

**Established.**

- The retry is deadlocked by a half-open `v0` left by its own previous attempt,
  and `monitor_gave_up` says there is no handler for that condition.
- The store records one version where the filesystem has four.
- `v1/content/` is populated — 39 files, item layout `codes/ env/ result/
  script/ watchout/`, the shape of a `code` handoff. So the body ran **at least
  far enough to open `v0`, create `v1`, and populate it.** "The body did
  nothing" is not consistent with this tree.

**Not established, and I am not asserting it.**

- **Whether the body authored `v1` in this run.** Its `content/` is mtime
  `Sep 3 10:57` and `README.md` is `Sep 2 12:31`, both **before the run started
  at 17:28**, and the directory is `drwxrwxrwx` — the signature of the historical
  `chmod -R 777` that `temp/leader/repair_modes.py` exists to undo. That is what
  a `cp -a` of sealed mock material looks like, not what freshly written output
  looks like. So `v1` is most likely the mock workset copied in with metadata
  preserved. **It shows the body executed its adapter; it does not show the body
  produced a workset of its own.**
- **Why the store never advanced past `v0`.** Unknown.
- **Whether `output_absent` at 13.8 s is caused by this divergence** or is
  independent of it. The timing is suggestive — `v0` opened, `v1` populated at
  17:31, store frozen at `generating` 17:31:20.488, absent declared 17:31:21.321,
  under a second later — but suggestive is not measured.

## Why it was missed

Everyone reading this closure, including me, was reading the **runner's stdout
log**, which shows a stall message and nothing else. All six events above exist
only in `store/event/`, and the store is one JSON object per file rather than a
`.jsonl` stream, so the obvious scan returns zero and reads as "no events".
`assets/lib/read_events.py` was written the same day to remove that barrier.

**And one control matters for anyone who picks this up:** the task zone's
`logs/`, `playground/` and `tmp/` are **empty for every task in every run
examined**, including tasks that produced valid handoffs. They are not a
liveness signal and cannot be used to decide whether a body ran. That check was
tried here first and had to be thrown away.

## Not worked around

Nothing is changed by this record. Rung 0 reaches this closure and stops; the
retry deadlocks; the run ends. Owner is whoever owns `build_workset` (m3) for
the body half, and it is framework-level for the `handling_failed` half —
**`monitor_gave_up: the pusher has no action for handling_failed`** belongs
beside T14 and the discarded-validator-stdout record, as a third instance of
*the machinery detects the condition correctly and then has nowhere to take it*.
