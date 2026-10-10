# `validator` — a handoff is only a contract if something checks it

Validators are the **sole standard** by which a handoff is judged. This package
is where a check plugs in, what it may claim, and what stops the producer from
grading its own work.

| | |
|---|---|
| Specification | [`docs/spec.md`](docs/spec.md) — 21 acceptance criteria |
| Design | [`docs/design.md`](docs/design.md) |
| Seam | [`../docs/interfaces.md`](../docs/interfaces.md) §4.3 — normative |
| Contract | [`protocols.py`](protocols.py) + `protocols.pyi`, declarations only |
| Tests | `../tests/validator/` |
| General specs | [`general_specs/`](general_specs/) — the workflow-independent validators this repository ships |

## Who uses it

`agent`'s runner enters a validation phase and hands it the targets. `handoff`
persists the `Verdict` this package produces. `cli` reports it. `monitor`
receives a failed phase as an unplanned outcome. A **task package** supplies the
validators themselves, as bodies — this package runs them and judges nothing
itself.

## The interface

```python
from validator import (
    ValidatorSpec, ValidatorSpecRegistry,   # what a validator is, and where they live
    Validator, Body, LogicSource,           # how a check is implemented
    PhaseRunner, PhaseKind, PhaseOutcome,   # running a validation phase
    Verdict, VerdictRecord, Evidence,       # what comes out
    Dimension, Strength, Tags, Cost,        # what a verdict may claim
    Composite, NestedComposite, Reducer, REDUCERS, get_reducer,
    StrictLevel, RunRecord, RunState, SkipRecord,
    check_separation, SeparationViolation, ValidatorInvalid,
)
```

`Verdict` is **`handoff`'s** and is re-exported rather than re-declared: the
module that persists a record is the module that has to keep it readable, and
two records of one fact is `engineer_principle.md` §1's failure.

## What is inside

| | |
|---|---|
| `protocols.py` / `.pyi` | the frozen, importable half of the contract |
| `spec.py` | the admission model — what makes a validator spec real |
| `phase.py` | the validation phase: target selection, running bodies, the outcome |
| `boundary.py` | what a body may see |
| `environment.py` | which environment a validation runs in — spec §8.2's selection chain |
| `composite.py`, `reducers.py` | several checks folded into one verdict |
| `separation.py` | the anti-gaming rule: a producer does not grade itself |
| `history.py` | prior verdicts, read back from the store |
| `report.py`, `registry.py` | reporting, and the spec registry |

## Four properties worth knowing before changing anything

1. **A failure binds at every strength.** Strength qualifies a *pass*, never a
   fail — a weak check that fails still fails the handoff.
2. **One fails, all fail.** A composite is not a vote.
3. **A validation environment is rebuilt, never reused.** A check that inherited
   the producer's environment would be checking the producer's assumptions.
4. **An unchecked output is a fault, and it blocks.** Not a warning, not a pass —
   silence is the outcome this package exists to prevent.

## Two things a reader should not mistake

**The input phase is a partial backstop, not coverage.** It runs over what the
consumer declares, on the consumer's side. It does not re-check what the
producer's output phase already judged, and reading it as a second line of
defence will overstate what is checked.

**Which interpreter runs a body is not settled.** The selection chain always has
a producer environment on an output phase, so the one configuration row that
names an interpreter is never consulted there.
[`../docs/TODO.md`](../docs/TODO.md) item 5 has the shape of the fix and says
which of two answers is still a design call.
