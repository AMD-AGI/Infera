#!/usr/bin/env bash
# Bring-up for a (re)started allocation, as a queue step: preflight -> up with decode at simulated acceptance 3.61 ->
# census -> one smoke request -> the 192 x 8k-token KV transfer smoke. The known-answer gate at real acceptance already
# passed on AUS with this image and config (b1, 10-19 10:56: 48/48 and 37 + 11 + 0, as on crsuse2); skipping it also
# avoids the decode-only restart, which left some prefill->decode channels broken in 3P1D. Decode log: decode-0.log.
# Usage: bringup_fast.sh ROUND=<r>
set -uo pipefail
for a in "$@"; do export "$a"; done
: "${ROUND:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
bash "$HERE/run_3p1d.sh" preflight || exit 1
DECODE_SIMULATE_ACC_LEN=3.61 bash "$HERE/run_3p1d.sh" up || exit 1
set -a; source "$R/config/config.3p1d.sh"; set +a
status() { printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "${2:-}" | tee -a "$RUNS/status.tsv"; }
sleep 30   # the login-host followers (follow_supervisor.sh) pick up new logs within 20 s
bash "$HERE/census.sh" "$RUNS/launch/server-logs" "" > "$RUNS/census.bringup.txt" 2>&1
status census.bringup "rc=$? $(grep -c FATAL "$RUNS/census.bringup.txt") fatal-lines"
control_ip="$(python3 "$BENCH_DIR/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
curl -fsS --max-time 300 "http://$control_ip:$ROUTER_PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"max_tokens\":32,\"temperature\":0}" \
    > "$RUNS/smoke.json" || { status smoke FAIL; exit 1; }
status smoke ok
python3 "$HERE/transfer_smoke.py" "http://$control_ip:$ROUTER_PORT" "$SERVED_MODEL" \
    "$RUNS/launch/server-logs/decode-0.log" 192 8000 > "$RUNS/transfer_smoke.txt" 2>&1
rc=$?
status transfer_smoke "rc=$rc $(tail -1 "$RUNS/transfer_smoke.txt")"
printf '%s\tall\tdone\n' "$(date -u +%FT%TZ)" >> "$RUNS/status.tsv"
exit $rc
