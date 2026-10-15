#!/usr/bin/env bash
# Gate after a prefill HiCache change: restart the router with prefill session affinity (so the
# check's probes return to their DP rank), run the KV-transfer smoke, then hicache_check.py.
# Fails (non-zero) on any failure or an inconclusive check, which stops the tuning queue.
# Usage: hicache_check.sh ROUND=<r> TAG=<suffix>
# Output: runtime/runs/<ROUND>/launch/{transfer_smoke,hicache_check}.<TAG>.{txt,json}
set -euo pipefail
for a in "$@"; do
    [[ "$a" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "expected KEY=VALUE, got '$a'" >&2; exit 2; }
    export "$a"
done
: "${ROUND:?}" "${TAG:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
set -a; source "$R/config/config.3p1d.sh"; set +a
L="$RUNS/launch"
say() { echo "$(date -u +%FT%TZ) hicache_check: $*"; }
control_ip="$(python3 "$BENCH_DIR/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
router="http://$control_ip:$ROUTER_PORT"
best="INFERA_PD_PREFILL_GUARD_RELEASE=completion INFERA_SESSION_AFFINITY=prefill"
best+=" INFERA_SESSION_AFFINITY_TTL_SECS=3600 INFERA_R2_DECODE_DEMAND=on"
bash "$HERE/restart_router.sh" "ROUND=$ROUND" AFFINITY=false "TAG=hc-$TAG" "ROUTER_ENV=$best" > "$L/router.hc-$TAG.log" 2>&1 ||
    { say "router restart failed (see $L/router.hc-$TAG.log)"; exit 1; }
say "router restarted with session affinity"
python3 "$HERE/transfer_smoke.py" "$router" "$SERVED_MODEL" "$L/server-logs/$DECODE_LOG" 192 8000 \
    > "$L/transfer_smoke.$TAG.txt" 2>&1 || { say "transfer smoke failed: $(tail -1 "$L/transfer_smoke.$TAG.txt")"; exit 1; }
say "transfer smoke: $(tail -1 "$L/transfer_smoke.$TAG.txt")"
mapfile -t metrics < <(python3 "$BENCH_DIR/tools/topology.py" rows "$TOPOLOGY" | awk -F'\t' -v b="$ENGINE_PORT_BASE" \
    -v s="${ENGINE_PORT_STRIDE:-1}" '$3 == "prefill" {print "http://" $5 ":" b + $1 * s "/metrics"}')
say "prefill metrics: ${metrics[*]}"
python3 "$HERE/hicache_check.py" "$router" "$SERVED_MODEL" "$L/hicache_check.$TAG.json" "${metrics[@]}"
