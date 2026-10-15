#!/usr/bin/env bash
# HiCache functional gate on the aggregated stack: the 3P1D kit's hicache_check.py (a prefix pushed
# out of the GPU pool to host and loaded back must give the same first token and logprob as when
# computed fresh) against the mixed worker(s). The router already runs with session affinity on
# mixed workers, so the probes return to their DP rank.
# Usage: hicache_check_agg.sh ROUND=<r> TAG=<suffix>   Exit: 0 pass, 1 fail, 2 inconclusive.
# Output: runtime/runs/<ROUND>/launch/hicache_check.<TAG>.{json,txt}
set -uo pipefail
for a in "$@"; do
    [[ "$a" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "expected KEY=VALUE, got '$a'" >&2; exit 2; }
    export "$a"
done
: "${ROUND:?}" "${TAG:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
set -a; source "$HERE/../config/config.agg.sh"; set +a
L="$RUNS/launch"
control_ip="$(python3 "$BENCH_DIR/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
mapfile -t metrics < <(python3 "$BENCH_DIR/tools/topology.py" rows "$TOPOLOGY" |
    awk -F'\t' -v b="$ENGINE_PORT_BASE" -v s="$ENGINE_PORT_STRIDE" '{print "http://" $5 ":" b + $1 * s "/metrics"}')
python3 "$R/scripts/hicache_check.py" "http://$control_ip:$ROUTER_PORT" "$SERVED_MODEL" \
    "$L/hicache_check.$TAG.json" "${metrics[@]}" 2>&1 | tee "$L/hicache_check.$TAG.txt"
exit "${PIPESTATUS[0]}"
