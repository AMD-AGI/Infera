#!/usr/bin/env bash
# Restart decode and all prefills of a live stack under a new overlay, decode first and prefills second so both
# sides start with fresh transfer state (a decode-only restart left some prefill->decode channels broken), then
# verify KV transfer with the 192-request smoke.
# Usage: restart_all_engines.sh ROUND=<r> OVERLAY=<config/...sh> TAG=<suffix>
#   OVERLAY must set DECODE_LOG=decode-0.after-<TAG>.log (the name restart_decode.sh writes).
set -euo pipefail
for a in "$@"; do
    [[ "$a" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "expected KEY=VALUE, got '$a'" >&2; exit 2; }
    export "$a"
done
: "${ROUND:?}" "${OVERLAY:?}" "${TAG:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
[[ "$OVERLAY" == /* ]] || OVERLAY="$R/$OVERLAY"
grep -qx "DECODE_LOG=decode-0.after-$TAG.log" "$OVERLAY" ||
    { echo "$OVERLAY must set DECODE_LOG=decode-0.after-$TAG.log" >&2; exit 2; }
say() { echo "$(date -u +%FT%TZ) restart_all_engines: $*"; }
round_file="$R/config/round.$ROUND.sh"
[[ -e "$round_file" ]] && cp -p "$round_file" "$round_file.before-$TAG"
cp "$OVERLAY" "$round_file"
say "installed $(basename "$OVERLAY") as $(basename "$round_file")"
set -a; source "$R/config/config.3p1d.sh"; set +a
live_prefills="$(curl -fsS --max-time 10 "http://$(python3 "$BENCH_DIR/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE"):$ROUTER_PORT/v1/workers" |
    python3 -c 'import json,sys; p=json.load(sys.stdin); w=p if isinstance(p,list) else (p.get("workers") or p.get("data") or []); print(sum(1 for x in w if x.get("disagg_mode")=="prefill"))')"
EXPECT_WORKERS=$(( live_prefills + 1 )) bash "$HERE/restart_decode.sh" "ROUND=$ROUND" SIM=3.61 "LAUNCH_DIR=$RUNS/launch" "TAG=$TAG"
say "decode restarted"
bash "$HERE/restart_prefills.sh" "ROUND=$ROUND" "OVERLAY=$OVERLAY" "TAG=${TAG}p"
say "prefills restarted"
control_ip="$(python3 "$BENCH_DIR/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
python3 "$HERE/transfer_smoke.py" "http://$control_ip:$ROUTER_PORT" "$SERVED_MODEL" \
    "$RUNS/launch/server-logs/$DECODE_LOG" 192 8000 || { say "transfer smoke FAILED"; exit 1; }
say "done: transfer smoke ok"
