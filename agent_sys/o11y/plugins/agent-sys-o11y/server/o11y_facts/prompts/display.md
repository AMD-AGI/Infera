You extract dashboard facts for an LLM inference-optimisation project from the newest part of its main
Claude Code session (lines start with `@<byte offset>`), the task book excerpt, and the run ledger tail.
Fill ONLY what the ledger does not already give (the ledger wins when both exist). Keep strings short
(<= 80 chars). Never invent numbers; use null when unknown.

- `task`: title, metric name + unit + direction (higher|lower), baseline value, target value.
- `current_run`: what is being run or explored right now: theme, point (explore|validate|baseline|repro|
  diagnostic|calibration|profile), machines (node names), started_at if visible.
- `recent_runs`: up to 3 most recent finished runs/explorations: run_id (if any), theme, result (short text), status.
- `next_plan`: up to 3 next planned points: theme, why.
- `evidence_offsets`: the `@offset` values you relied on.
