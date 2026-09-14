# `output_absent` states a cause that is false, and the true one is a sibling attribute

**Measured 2026-09-04, twice, by two owners independently.**

An `output_absent` event's `message` reads:

```
declared output <hid> was never delivered
```

**It was delivered.** In the instance below, 35 files. What happened is that the
**seal refused it**, and the reason is in a different attribute of the same
event:

```json
"detail":       "exit 0",
"exit_status":  "finished",
"message":      "declared output 11d8db19-… was never delivered",
"seal_refused": ".../v1/content/README.md: every handoff opens with a README.md
                 (spec §3.1). An artefact only a program can open is a blob,
                 and blobs do not get reviewed"
```

Three fields, and **the two a reader meets first both point away from the
cause**. `message` says the output is missing. `detail` says `exit 0`, which
invites the obvious next inference — *the body succeeded and the framework lost
its work.* Only `seal_refused`, read last if at all, says the artefact arrived
and was rejected on its merits.

**A wrong cause stated as fact is worse than no cause**, because it points the
next reader at the wrong repair. "The framework loses a leaf's output" and "your
artefact is missing a required file" send you to different codebases.

## Two instances, two owners, one attribute

**m2, 2026-09-04 10:09.** `deploy_and_prove` on a replayed `deploy_kit`. Clean
A/B, one variable — same body, same adapter, two mock roots:

```
cheat_for_mock/stage1-deploy/deploy_kit/content/   README.md  items   -> sealed valid
mockroot_realkit/…/deploy_kit/content/                        items   -> output_absent
```

The kit supplied had no `content/README.md`. The seal was right, the message was
wrong, and the investigation went to the framework first because that is where
the message pointed.

**The leader's four-run stall study, same day.** Four runs, and the
`seal_refused` attribute was **identical in all four** —
`README.md: required section 'Interface' is missing`. That study's whole
question was whether the stall detector was ending healthy leaves; the answer
was in an attribute nobody was reading, and it took opening raw JSON to see it.

**Both owners had to `cat` an event file.** That convergence is what makes this
a bug rather than a preference: two independent investigations, the same
attribute, the same manual step to reach it.

## Why the readers did not help

`assets/lib/read_events.py` prints `message` and not the other attributes, so
the tool built for reading the event store reproduces the misdirection. m4 hit
the same gap from the other side and fixed **their** reader — `runprobe` now
prints every attribute of a triggering event — which is what turned a lost
reason into a one-command answer three times today. `read_events.py` has not had
that change. **Routed to checkpoint, its author, rather than edited here**
(`todo.md` T39).

## What is *not* wrong

`seal_refused` is excellent: it names the file, the rule and the reason, and it
is machine-readable. The seal itself behaved correctly in every instance — it
refused an artefact that genuinely violated spec §3.1. **Nothing here is an
argument for weakening the seal.** The defect is entirely in which of the three
fields a reader meets first.

## Workaround, which is what to do today

Read the attributes, not the message:

```sh
python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(json.dumps(d["attributes"],indent=1))' \
  <run>/store/event/<task>%23*%23*.json
```

and treat `message` on an `output_absent` as *"the runner did not receive an
acceptable output"* rather than as a statement about delivery.

## Not fixed here, and one reason to be careful about fixing it

`agent_sys/agent/` is outside this effort's activity scope. And the standing
rule is record first, work around second, fix only on unambiguous evidence —
which this is not, quite: **the message may be literally true in the runner's
own vocabulary**, where "delivered" could mean "accepted into the store" rather
than "written to the slot". If so the wording is defensible internally and still
misleads every reader from outside, and the fix is a clarification rather than a
correction. Whoever owns the runner can tell which; we cannot from here.

The smallest useful change is not to `message` at all: **surface `seal_refused`
wherever `message` is surfaced.** That is one line in each reader and it makes
the wording question moot.
