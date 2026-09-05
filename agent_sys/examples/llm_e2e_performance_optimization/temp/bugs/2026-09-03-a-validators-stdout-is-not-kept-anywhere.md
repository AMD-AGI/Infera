# A validator's stdout is not kept anywhere

**Measured 2026-09-03.** A program validator's findings go to `stdout`. Nothing
in the run keeps them: not the streamed console output, not the run tree, not
the zone. What survives is `verdict.json`, which holds a boolean per handoff, and
`validation.yaml`, which holds the same boolean plus provenance.

So a validator that explains itself carefully explains itself to nothing.

## What it cost

Rung 0 refused three times at `check_deploy_serves`. All three refusals produced
exactly this much evidence:

```yaml
- validator: check_deploy_serves
  result: false
  strength: strong
  dimension: usability
```

The validator had in fact printed:

```
check_deploy_serves: <hid>: FAIL: scripts/deploy.sh failed (rc=1); last output:
Error: failed to connect to controller
Caused by:
    0: transport error
    1: tcp connect error
    3: Connection refused (os error 111)
```

which names the cause outright. **Recovering that line took about an hour** and
two wrong attributions — first to GPU contention with another module's live
deployment, then to a missing `local` branch in `remote.sh`. Neither was the
cause. The actual cause was a missing `--var transport_env`, and the correct
incantation is written in a comment in the file that declares the parameter
(`steps/m1_deploy.yaml:128`).

**Every step of that hour was avoidable by one line of retained output.**

## How it was recovered, since the same method will be needed again

The zone survives the run. Copy it, and run the validator by hand *under the
environment the framework gives it* — not under the developer's shell, which has
`SPUR_CONTROLLER_ADDR`, a full `PATH` and everything else the policy environment
strips:

```sh
S=/shared_nfs/yihou/.../dsrepro.$(date +%H%M%S)      # never reuse; never rm -rf
mkdir -p "$S" && cp -r "<zone>/." "$S/"
cd "$S" && env -i PATH=/usr/local/bin:/usr/local/sbin:/usr/bin:/bin \
    HOME="$S" AGENT_SYS_TASK_PACKAGE=<pkg> python3 <pkg>/assets/<v>.validator/check.py
```

**`env -i` is the load-bearing part.** Run from a normal shell the same
validator *passes*, because the developer's environment supplies what the
policy environment does not. A by-hand run that inherits your shell is a
different experiment from the one that failed, and it will tell you the subject
is fine — `CONTRACT.md` §4.4, face 1, arriving during the debugging of face 2.

## Why this is worth fixing rather than working around

The workaround is real and is in use: a validator can write its findings into
the zone as a file beside `verdict.json`. Several already print to `stderr` as
well, which is likewise not retained.

But the property that makes this a framework issue rather than a package
convention is that **the failure is silent and shaped like success**: a
validator author writes a careful diagnostic, tests it by hand where it appears,
and never learns that in a run it goes nowhere. Nothing warns. The verdict is
still correct — only the reason is lost — so every check of the *result* passes.

## It is one half of a pair, and `todo.md` T14 is the other

Connection made by `checkpoint`, and it is the reason this is worth more than
the run it blocked.

| | what is lost | what the phase shows |
|---|---|---|
| **this** | the **reason** for a refusal | a refusal, no reason |
| **T14** | the **verdict itself** — a broken interpreter writes no `verdict.json` | indistinguishable from a bad handoff |

**Same seam, two ways of losing the same thing.** In both, the validator's own
account of what happened does not survive to the reader, and in both the run
still looks well-formed: T14 turns "my validator is broken" into "your handoff is
bad", and this one turns "you forgot an argument" into "the deployment failed".

A validator's job is to produce a *judgement with a reason*. The framework
currently persists the judgement and drops the reason, and — when the body dies
early — persists neither while still attributing the outcome to the subject.

Recording per mission rule: bug recorded first, worked around second, fixed only
on unambiguous evidence. The evidence here is unambiguous about the behaviour;
whether the fix belongs in the runner or in a package convention is not.

---

## Correction, 2026-09-04 — the title is right, and the way it has been cited is not

**Left standing above rather than edited away**, because the over-broad reading
is the part worth seeing: this file was cited five times in one day — by m3,
m4, m1, m2 and the leader — and every citation treated it as *"stdout is kept
nowhere"*, a claim about the framework rather than about validators.

**Measured 2026-09-04, from the event store of two runs:**

| producer | event | attributes | output kept? |
|---|---|---|---|
| **validator** | `validation_failed` | `evidence, from_task, message` | **no** |
| **task body, exit ≠ 0** | `output_absent` | `detail, exit_status, message, seal_refused` | **yes**, in `detail` |
| **task body, exit = 0** | — | — | **no** |

**So the title is accurate for validators and was never a claim about task
bodies.** For a task body the truth is different and narrower, and the narrow
version is the more useful one:

> A task body's output is kept **only on the failure path the runner already
> recognises**. A body that exits **0** has its output discarded entirely.

**That is the worse half, because it is the silent one.** A body that fails
loudly is already explained; a body that exits 0 having done part of its work
looks exactly like one that succeeded. It cost the 047 investigation two runs
and most of an hour: `build_workset` sealed a workset with no `workset.yaml`,
then one with no `evidence`, exiting 0 both times with nothing retained. The
cause was found by a two-node differential (`a4b6dd4`), not by any log —
**this file's absence is why there was nothing to read, not why the answer was
eventually found.**

### Which fix covers which half

| half | fix | scope |
|---|---|---|
| validator findings | `dff2bcb` — `workset_io.write_report`, always, before `write_verdict` | m3's two workset validators; adopted since by m4 and m1 in their own |
| validator **crash** vs refusal | `4b4c9ce` — `validator_crash.txt` + `THIS VALIDATOR DID NOT RUN` | m3's, same limitation: `verdict.json` is `dict[str, bool]` (`todo.md` T29) |
| task body exiting 0 | `cff4571` — `entry.sh` tees to `logs/build_workset.body.log` | **`build_workset` only** |
| task body exiting ≠ 0 | none needed | the runner already keeps it in `detail` |

**Nothing central.** Six owners have now fixed this in their own stage, each
with their own file name and their own placement. That is the signature
`checkpoint` named for a class that is named but never swept (`todo.md` T30,
T31) — and it is the argument for the framework fix this file already asks
for, restated with a year's worth of instances in one day.

### The `seal_refused` addendum

`output_absent` also carries `seal_refused`, which names the file and the
missing section when a seal is refused. **It was read by nobody for two runs
across two days** while four owners investigated the stall detector, the card
and the body. So the task-body half of this bug has a second face: not only
*"the reason is not kept"* but *"the reason is kept in an attribute nobody
prints"*. The first needs a framework fix; the second needs only that
something surface it, which is why it was routed to `runprobe.py`.
