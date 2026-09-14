# An escalation reaches the top and finds nobody, and the run says so to nobody

**Measured 2026-09-04**, rung 1's run `20260904T041742-8431df`. Read out of the
event store, which is the only place it exists.

```
05:08:56.746  output_absent  declared output 6c5b43da-… was never delivered   (×3)
05:08:56.762  escalated      nothing to push: the executor is a program body:
                             there is no agent to instruct
05:08:56.787  escalated      nothing to push: the attempt holds no executor:
                             it is not in its main phase                      (×2)
05:08:56.821  escalated      target: user — same reason
```

`6c5b43da` is `operator_workset`, still `generating`. **The run ended with a
declared output undelivered, escalated four times, and every attempt answered
that there was nobody to escalate to.** One of the four says why in as many
words: *the executor is a program body: there is no agent to instruct.*

## It is the third face of one seam, and m1 is the one who joined them

| | what exists | where it goes |
|---|---|---|
| `…validators-stdout-is-not-kept…` | a validator's careful diagnosis | **discarded** |
| `todo.md` **T14** | a broken validator's verdict | **never written** |
| **this** | a program body's full refusal | **kept, escalated, and read by nobody** |

**Corrected by m3 after this record was first filed, and the correction matters
because it splits the family.** The original text put this in the same class as
the discarded stdout. It is not: **the detail was kept the whole time**, in the
`output_absent` event's `attributes.detail`, carrying the body's entire stderr
including the sentence naming the cause. Nothing was lost. What is missing is a
**reader** — the escalation had no recipient because the failing executor was a
program body, and `NullUserSink` records without answering.

So the first two are *information destroyed*; this one is *information
preserved and unrouted*. Filing them as one thing was the leader's error, made
while writing a record about explanations having no route — **from a store the
leader had not read carefully enough to notice that this explanation had been
kept.**

m1's framing, and it is better than three separate entries:

> **Every one of them is the system having the explanation and no route for it.**

That is why this is filed rather than shrugged at. The machinery **worked** —
it detected an absent output, it escalated, it recorded the escalation, it
recorded that the escalation had no recipient. Four correct steps ending in
silence. Nothing here is a crash and nothing here is a wrong answer; the
information simply has no consumer.

## What it cost, concretely

The leader spent the afternoon attributing **every** `build_workset` stop to
the stall-detector bug, and told the team and the user that. Rung 0's stop was
the detector — its log says *"Nothing has changed for 20 s"*. **Rung 1's was
not**: `build_workset` entered running at ~05:08:45 and `output_absent` fired
at 05:08:56, **eleven seconds**, having produced nothing.

**Two different failures at one closure, merged into one cause and broadcast**,
because the only artefact anybody read was the console line saying
`build_workset: running` — and the console does not carry `output_absent`.
Found by `checkpoint` opening the event store, which nobody had done.

## Not proposing a fix

Where the escalation *should* go when the executor is a program body is a
runner design question, and `agent_sys/cli/` is outside this effort's activity
scope. What is unambiguous is the behaviour and its cost.

**Cheap mitigation available to us, and it is not a fix:** the event store is
readable and nobody was reading it. Any claim of the form *"the run completed"*
should be checked against `output_absent` in the store rather than against the
console's last line. `checkpoint` has that in their standing checks now.
