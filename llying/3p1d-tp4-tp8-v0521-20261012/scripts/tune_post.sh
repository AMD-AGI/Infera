#!/usr/bin/env bash
# After-run half of a tuning round (also used to finish a round whose driver died):
# census, flush check, router metrics / warnings, memory snapshot, window analyses, publish
# to results/<VARIANT>/c<CONC>/, summary, status line.
# Usage: tune_post.sh VARIANT=<name> CONC=<n> [AGENTX_RC=<rc>] [ROUND=r2]
# Reads runtime/runs/<ROUND>/tune/<VARIANT>-c<CONC>/{bench.start,flushed_at,round.env}.
set -uo pipefail
for a in "$@"; do
    [[ "$a" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "expected KEY=VALUE, got '$a'" >&2; exit 2; }
    export "$a"
done
: "${VARIANT:?}" "${CONC:?}"
export ROUND="${ROUND:-r2}"
rc="${AGENTX_RC:-0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
set -a; source "$R/config/config.3p1d.sh"; set +a
LOGS="$RUNS/launch/server-logs"
name="$VARIANT-c$CONC"
OUT="$RUNS/tune/$name"
RES="$R/results/$VARIANT/c$CONC"
DECODE_LOG="$(sed -n 's/^DECODE_LOG=//p' "$OUT/round.env")"
start="$(cat "$OUT/bench.start")"
flushed_at="$(cat "$OUT/flushed_at")"
read -r -a ssh_args <<< "$SSH_OPTS"
q() { local out="" i; for i in "$@"; do printf -v i '%q' "$i"; out+="${out:+ }$i"; done; echo "$out"; }
on() { local n="$1"; shift; ssh "${ssh_args[@]}" "$n" "$(q "$@")" </dev/null; }
status() { printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$name" "$1" "${2:-}" | tee -a "$RUNS/tune/status.tsv"; }
die() { status "$1" "FAIL ${2:-}"; exit 1; }
noansi() { sed -E 's/\x1b\[[0-9;]*m//g'; }
# Publishing writes to the shared /home; wait out a momentarily full volume instead of failing midway.
bash "$HERE/wait_home_space.sh" 5 7200 || true
control_ip="$(python3 "$BENCH_DIR/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
router_url="http://$control_ip:$ROUTER_PORT"
router="$CONTAINER_PREFIX-router"
result="$OUT/agentx-$name/agentx_conc$CONC.json"

snapshot() {
    local node
    for node in $(python3 "$BENCH_DIR/tools/topology.py" rows "$TOPOLOGY" | cut -f4 | sort -u); do
        echo "== $node"
        on "$node" bash -c 'rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2; free -g | head -2'
    done > "$OUT/mem.$1.txt" 2>&1
}

bash "$HERE/census.sh" "$LOGS" "$start" > "$OUT/census.txt" 2>&1
census_rc=$?
python3 "$HERE/flush_check.py" "$LOGS" "$flushed_at" > "$OUT/flush_check.txt" 2>&1
flush_rc=$?
curl -fsS --max-time 10 "$router_url/metrics" > "$OUT/router.metrics.txt" 2>/dev/null
on "$CONTROL_NODE" docker logs --since "${start}Z" "$router" 2>&1 | noansi |
    grep -E ' (WARN|ERROR) ' > "$OUT/router.warn.txt"
snapshot end
window="$(python3 -c '
import json,sys
from datetime import datetime,timedelta
s=datetime.fromisoformat(json.load(open(sys.argv[1]))["start_time"])
print(s.strftime("%Y-%m-%dT%H:%M:%S"), (s+timedelta(seconds=int(sys.argv[2]))).strftime("%Y-%m-%dT%H:%M:%S"))' \
    "$OUT/agentx-$name/aiperf_artifacts/profile_export_aiperf.json" "$AGENTX_DURATION" 2>/dev/null)"
if [[ -n "$window" ]]; then
    read -r w0 w1 <<< "$window"
    python3 "$HERE/prefill_balance.py" "$LOGS" "$w0" "$w1" > "$OUT/prefill_balance.txt" 2>&1
    python3 "$HERE/decode_balance.py" "$LOGS/$DECODE_LOG" "$w0" "$w1" > "$OUT/decode_balance.txt" 2>&1
    python3 "$HERE/engine_pace.py" "$LOGS" "$DECODE_LOG" "$w0" "$w1" > "$OUT/engine_pace.txt" 2>&1
fi
status bench "rc=$rc census_rc=$census_rc flush_rc=$flush_rc router_warn=$(wc -l < "$OUT/router.warn.txt") $(tail -1 "$OUT/agentx.log" | cut -c1-160)"
[[ -s "$result" ]] || die bench "no AgentX result (rc=$rc)"

src="$OUT/agentx-$name"
mkdir -p "$RES"
cp -p "$result" "$src"/runtime.env "$src"/benchmark_command.txt "$src"/workload_distribution_summary.txt \
    "$src"/metrics_plots.png "$RES"/ 2>/dev/null
cp -p "$src"/aiperf_artifacts/profile_export_aiperf.json "$src"/aiperf_artifacts/profile_export_aiperf.csv \
    "$RES"/ 2>/dev/null
[[ -d "$src/service" ]] && cp -rp "$src/service" "$RES/service"
cp -p "$OUT"/round.env "$OUT"/census.txt "$OUT"/flush.txt "$OUT"/flush_check*.txt "$OUT"/router.controls.txt \
    "$OUT"/router.warn.txt "$OUT"/mem.*.txt "$OUT"/*_balance.txt "$OUT"/engine_pace.txt "$RES"/ 2>/dev/null
grep -E '^infera_router_session' "$OUT/router.metrics.txt" > "$RES/router.session_metrics.txt" 2>/dev/null
python3 "$HERE/summarize.py" > /dev/null 2>&1
status done "$(python3 -c '
import json,sys
j=json.load(open(sys.argv[1])); t=j["request_metrics"]["throughput"]["per_gpu"]
print("%.0f tok/s/GPU, out %.1f, ok %d" % (t["total_tput_tps"], t["output_tput_tps"], j["num_requests_successful"]))' "$result")"
(( flush_rc == 0 )) || die flush_check "first prefill batches after the flush were not all empty (see flush_check.txt)"
# A partial result (client aborted, e.g. failed-request threshold) is published for diagnosis
# but must not count as a finished round: fail so the queue stops.
(( rc == 0 )) || die bench "AgentX exited rc=$rc; partial result only (see agentx.log)"
exit 0
