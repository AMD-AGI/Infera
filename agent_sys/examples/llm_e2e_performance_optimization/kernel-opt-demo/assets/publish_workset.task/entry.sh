#!/bin/sh
# The body of `publish_workset`, a `kind: program` leaf.
#
# Run by `agent/backends/program.py` as `/bin/sh <entry>`, with `cwd` set to the
# task's zone and the environment being **`os.environ` merged with
# `Prepared.environment`** — a real difference from a validator body, which gets
# a closed block with `os.environ` deliberately not inherited. So `python3` is
# on the ambient `PATH` here and needs no `AGENT_SYS_DEMO_PYTHON` fallback.
#
# `AGENT_SYS_TASK_PACKAGE` is the **staged** copy of this package inside the
# zone. It is always set for a task body; the `:?` makes its absence a named
# failure rather than a path that silently resolves to `/assets/...`.
set -eu
exec python3 "${AGENT_SYS_TASK_PACKAGE:?the task runner exports this}/assets/publish_workset.task/publish.py"
