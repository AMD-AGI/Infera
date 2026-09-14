# consume — render the report

Read the `summary` handoff and render it into a report a person reads.

**This task never runs.** Its input never becomes valid: `check_grounded` seals
the `summary` INVALID, so `consume` sits in `WAITING_HANDOFF` until the run
ends quiescent. That is the correct outcome and the demo reports it as the
expected one.

Its readme still says what it would have done, because a step nobody can read
is a step nobody can review — and that holds for a step that did not happen.
