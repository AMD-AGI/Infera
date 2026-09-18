#!/usr/bin/env bash
# Purpose: T2 -- run the AgentX concurrency sweep sequentially against the ONE
#   live T2 stack. agentx_bench.sh talks to the live router, so the sweep needs
#   a single bring-up, not one per point.
# Usage: ./sweep.yihou.sh [CONC ...]     (default: 40 56 72 96 128)
# Artifacts: <workspace>/t2-agentx-c<N>/ per point, plus t2-sweep.log here.
#
# DELIBERATELY NOT `set -e` ON THE BENCH CALL. The 128 point is expected to be
# able to fail on memory, and the mission says that is an acceptable end. A
# failure is recorded and the sweep stops; it is not retried and not tuned
# around.
#
# HiCache release lags the request stream, so a gap is left between points for
# the host pool to drain. Slow memory return here is expected and is NOT a leak.
set -uo pipefail

W="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$W/../.." && pwd)"
CONCS=("${@:-}")
[[ -n "${CONCS[0]:-}" ]] || CONCS=(40 56 72 96 128)
SETTLE="${SETTLE:-180}"

echo "sweep start $(date -u +%FT%TZ) points: ${CONCS[*]}"
for conc in "${CONCS[@]}"; do
    out="$W/t2-agentx-c$conc"
    echo "=== CONC=$conc start $(date -u +%FT%TZ) -> $out"
    if "$BENCH/agentx_bench.sh" CONC="$conc" \
        CONFIG="$W/config.yihou.full.sh" \
        TOPOLOGY="$W/topology.yihou.tsv" \
        OUT_DIR="$out" >"$W/t2-agentx-c$conc.log" 2>&1; then
        echo "=== CONC=$conc PASS $(date -u +%FT%TZ)"
    else
        rc=$?
        echo "=== CONC=$conc FAIL rc=$rc $(date -u +%FT%TZ) -- stopping sweep"
        echo "    last 20 lines of $W/t2-agentx-c$conc.log:"
        tail -20 "$W/t2-agentx-c$conc.log" | sed 's/^/    /'
        break
    fi
    # Let the HiCache host pool drain before loading the next point, so the
    # following run does not start against a still-evicting cache.
    echo "--- settling ${SETTLE}s for HiCache release"
    sleep "$SETTLE"
done
echo "sweep end $(date -u +%FT%TZ)"
