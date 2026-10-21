#!/usr/bin/env bash
# Queue step: the kit's engine/down.sh (saves logs and inspect output into the run directory, removes the kit's
# containers on both nodes), then stop this run's log followers and copy the run directory's logs back.
# Usage: kit_down.sh RUN=<id>
set -uo pipefail
for a in "$@"; do export "$a"; done
: "${RUN:?}"
source "$(dirname "$0")/kit_env.sh"
OUT="$PR_RUNS/$RUN"
kit_on "$P_NODE" "bash engine/down.sh" > "$OUT/down.log" 2>&1 || say "down.sh rc=$? (see $OUT/down.log)"
if [[ -f "$OUT/logs/followers.pid" ]]; then
    while read -r pid; do kill "$pid" 2>/dev/null; done < "$OUT/logs/followers.pid"
    rm -f "$OUT/logs/followers.pid"
fi
ssh "$P_NODE" "tar -C /scratch/liyingli_2p1d/pr210/work/runs/$RUN -cf - logs 2>/dev/null" | tar -C "$OUT" -xf - 2>/dev/null
say "down $RUN"
