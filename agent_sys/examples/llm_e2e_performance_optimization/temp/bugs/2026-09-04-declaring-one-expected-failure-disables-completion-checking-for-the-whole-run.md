# Declaring one expected failure switches off completion checking for the whole run

**Read 2026-09-04 in `agent_sys/cli/`, not run — this is a code read, and the
two decisive lines are quoted verbatim below.** Filed under principle 6 because
the route into it is short and inviting: a package needs to say *"this validator
is expected to refuse on this input"*, finds `cli/expectations.py`, reads a
well-argued implementation of pytest's `xfail(strict=True)`, and uses it.

**The framework already knows about every part of this.** That is what makes it
worth a file rather than a fix: the gaps are recorded in the source, in
docstrings written by whoever built it, and a reader who skims will take the
mechanism for a finished feature.

## What is there, and it is good

`cli/expectations.py` gives a package four outcomes and a fifth concept:

| outcome | event | exit |
|---|---|---|
| promised failure observed | `EXPECTED_FAILURE` | 0 |
| not observed, run could have tested it | `UNEXPECTED_SUCCESS` | 3 — *a property stopped holding* |
| not observed, run never reached it | `EXPECTATION_UNREACHED` | 4 — *untested, not broken* |
| no promises, something unfinished | `RUN_COMPLETE ok:false` | 5 |

plus `dropped`, a *deliberately not run* declaration emitted as
`VALIDATION_DROPPED`, because *"a dropped check that simply vanishes from the
output is indistinguishable from one that was never declared."*

The three-way split between **observed**, **did not happen** and **never
reached** is better than what a package would invent, and `main.py:1310` is
explicit that green must never be reachable by absence.

## The defect: the blast radius is the run, not the declared pair

`cli/main.py:1408-1409`, verbatim:

```python
# **Only when nothing was promised** — see the docstring's fourth row.
gaps = sorted(unfinished) if nothing_promised else []
```

`unfinished` is `_completion_gaps` — *every task `SUCCEEDED`, no handoff
`INVALID`*. **It decides the exit code only while the package promises
nothing.** Declare one promise and it stops deciding anything, for every task
and every handoff in the run.

So the natural use — *declare the one refusal we expect, keep everything else
graded as before* — is not what happens. What happens is: **completion checking
is switched off for all fifteen handoffs in exchange for declaring one
refusal.** That is one dict entry away from any reader who finds the file.

**The docstring says so, at `main.py:1344`:**

> *"And it is applied only to the empty set, which is a gap and not an
> oversight. … **What is missing is the package saying which of those it
> meant**, which is the same package-declared expectation
> `cli/expectations.py`'s docstring records as out of scope."*

The reasoning is sound for `examples/demo`, whose *own ending* is a task
deliberately left in `WAITING_HANDOFF` and a handoff deliberately never made
valid — a generic completion rule there would contradict the package's
specification. It does not generalise to a package with fifteen handoffs and one
declared refusal.

## Two more blocks, either of which is independently sufficient

**It is not package-declarable.** `cli/expectations.py:246`:

```python
_BY_PACKAGE: dict[str, ExpectationSet] = {"demo": DEMO}
```

Keyed on directory name, hardcoded in the CLI. `e2e-flow` is absent and gets
`EMPTY`. Declaring anything means editing framework code. The module's own
opening docstring calls this out — *"**a package name appearing in the CLI is a
leak**, and enumerating it here does not stop being one just because it is now
in a file of its own"* — and names the fix as out of scope.

**A promise cannot pin a reason, because the sealed verdict has none.** Measured
on the refusal in run `20260904T114914-0a0cdd`:

```json
{"validator": "check_no_regression", "result": false, "strength": "strong",
 "dimension": "trustworthiness", "task_id": "1a2f39cc-…", "agent_id": null,
 "environment": {"source": "producer", "zone": "…/validation-s57x3sv2"},
 "at": "2026-09-04T11:56:19.773496+00:00"}
```

No reason field. So a matcher can only key on `(validator, kind, result is
False)` — which is exactly what the demo's `_demo_verdict` does. **Any** refusal
from that validator on that kind satisfies the promise, including a real
regression appearing next week. An expected-refusal that silently accepts a
different refusal reason is strictly worse than no mechanism: it masks a real
finding behind an approved one.

## What this package does instead

Nothing in `agent_sys`. The acceptance claim is a named file and a failing
condition — `assets/lib/accept_mock.py`, and the `RUN-PLAN` section *"Mock e2e
green is a file and a condition"*. It pins the reason by reading the four
`PROBLEM:` lines out of the refusing validator's `validator_report.txt`, which
exists only because the shared `workset_io.write_report` helper puts the reason
on disk in structured form. The run still exits 5; nothing is inverted.

## The honest upstream path, out of scope and not blocked by anything we own

Three changes, and it takes all three — any one alone leaves a hole:

1. **`promises:` in `main.yaml`**, loaded by `spec_loader`, replacing the
   `_BY_PACKAGE` dict. This is `cli/expectations.py`'s own plan, quoted in its
   docstring, including the shape: `[{name, description, observed_when,
   judged_when}]`.
2. **Per-promise scoping of `_completion_gaps`**, so declaring one expectation
   exempts the handoff it names and nothing else — the *"which of those it
   meant"* the docstring is missing.
3. **A reason field on the sealed verdict**, so a promise can pin one. Without
   this, (1) and (2) produce a safe-looking mechanism that still accepts the
   wrong refusal.

Recorded so nobody reads this as impossible. It is not ours, it is not blocked,
and it is not urgent — but a package that reaches for the mechanism before (2)
exists will silently lose its completion check, and that is the part worth
knowing today.
