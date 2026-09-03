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
