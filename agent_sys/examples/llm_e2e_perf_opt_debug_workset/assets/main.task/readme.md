# main — the single-task debug harness

One leaf, `publish_workset`. No agent, no GPU, no credentials, seconds to run.
It exists so the `workset` contract can be exercised on its own, away from the
three-hour campaign that normally consumes it.

`main` is a non-leaf: it carries this readme, no `entry.sh`, and no `agent`.
