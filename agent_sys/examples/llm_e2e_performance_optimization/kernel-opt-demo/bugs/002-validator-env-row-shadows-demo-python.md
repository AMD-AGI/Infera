# BUG 002 — the PRODUCER env row shadows `AGENT_SYS_DEMO_PYTHON`

Found 2026-09-01 by teammate `env` while running `agent_sys/examples/demo`.
Recorded per series task book rule 1.1.

**Status:** real bug. Not hit by this package, for a reason worth writing down.

## Symptom

`examples/demo`'s `check_facts` validator exits 1 and records **no verdict** at
all, so the phase folds as a fault rather than as a failure.

## Cause

`validator/environment.py:136-142`. An **output**-phase validator body takes
§8.2's PRODUCER row, whose source is the producing task's resolved config. That
row **shadows** the GLOBAL row rather than merging with it — and the GLOBAL row
is the only one that carries `AGENT_SYS_DEMO_PACKAGE`, `AGENT_SYS_DEMO_STORE`
and `AGENT_SYS_DEMO_PYTHON`.

The template's entry-script idiom is:

```sh
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" "…/check.py"
```

On the output phase `AGENT_SYS_DEMO_PYTHON` is therefore unset and the fallback
fires — `python3` off `PATH`, which is **not** the interpreter the supervisor is
running under. In the container used here that is `/usr/bin/python3`, which has
no `pyyaml`, so the body dies on import.

**The rows being disjoint rather than nested is documented and intended**
(`validator/docs/spec.md` §8.2). What is not intended is that the documented
fallback silently selects a *different interpreter* — the failure reads as a
missing dependency, not as an environment-row problem, and that is why it costs
an afternoon.

## This package DOES hit it — I was wrong, and it cost three campaigns

**The first version of this file claimed immunity. That claim was false and it
is the most expensive mistake of the session.** It is left here, corrected in
place, because the reasoning error is more instructive than the bug.

What I wrote: *"all three validator bodies import stdlib only, so which
`python3` wins does not matter."* Both halves are true. The conclusion does not
follow.

`check_speedup_substantiated` imports stdlib — and then **shells out to
`measure_baseline.py`, which imports `torch`**, using `sys.executable` as the
interpreter. Inside a validator body `sys.executable` is `/usr/bin/python3`,
exactly as this bug describes, and that interpreter has no torch. So:

```
measure_baseline.py exited 1: ModuleNotFoundError: No module named 'torch'
```

The body reported that faithfully as "measurement failed" and folded it into a
FAIL. **Measured 2026-09-01 on three separate campaigns** (`short_1`, `short_2`,
`long_2`): every one recorded
`check_speedup_substantiated: FAIL  trustworthiness / weak`, in **0.1 s**. The
same handoffs pass when the validator is re-run by hand with the venv
interpreter — `short_1` re-measures 2.834× against its claimed 2.857×.

The 0.1 s was the tell and I did not read it: a body that genuinely runs ten
measurement rounds cannot return in a tenth of a second.

**Immunity to a missing import is not immunity to picking the wrong
interpreter.** A body is not "stdlib only" if it launches something that is not.

### The fix in this package

`check_speedup_substantiated` now **probes** candidates — `$KFO_PYTHON`,
`$AGENT_SYS_DEMO_PYTHON`, `sys.executable`, `/opt/venv/bin/python3`, `python3` —
runs `import torch` in each, uses the first that works, and prints which one it
chose. If none works it fails with a message naming every candidate it tried
instead of one that reads like a measurement disagreement.

`KFO_PYTHON` is set on the agent spec's `env:` block, which is the PRODUCER row
and therefore the one environment an output-phase validator actually receives.

The other two bodies here really are stdlib-only and start no subprocess, so
they were unaffected — but that is now a property to check when adding one, not
an assumption.

## Proposed fix

Either merge the GLOBAL row underneath the PRODUCER row for the three
`AGENT_SYS_DEMO_*` names, or export the supervisor's `sys.executable` under a
name that is present in **every** row. The second is smaller and removes the
class of bug rather than one instance.

**Not applied here** — `validator/environment.py` is outside this task's scope.

## Related container drift

To prove the diagnosis, `env` ran `pip install pyyaml` into `kf-mi300b`'s
`/usr/bin/python3`. **That container now differs from the pristine image by one
package.** Recorded because `environment.md` in any packup produced from this
container would otherwise be wrong. It does not affect this package's
validators, which use no third-party imports.
