#!/usr/bin/env bash
# One AgentX point on the live aggregated stack, the worker left running:
#   memory snapshot -> /flush_cache every worker -> router restarted with ROUTER_ENV (affinity
#   checked) -> engine gauge sampler -> AgentX at CONC -> publish to results/<VARIANT>/c<CONC>/
#   with a summary (agg_report.py).
# The client always sends X-Dynamo-Session-ID; the router ignores it unless affinity is on.
# Usage: agg_round.sh ROUND=<r> VARIANT=<name> CONC=<n> [DURATION=<s>] [ROUTER_ENV="INFERA_X=v ..."]
# Output: runtime/runs/<ROUND>/tune/<VARIANT>-c<CONC>/, published to results/<VARIANT>/c<CONC>/.
set -uo pipefail
for a in "$@"; do
    [[ "$a" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "expected KEY=VALUE, got '$a'" >&2; exit 2; }
    export "$a"
done
: "${ROUND:?}" "${VARIANT:?}" "${CONC:?}"
[[ "$VARIANT" =~ ^[a-z0-9][a-z0-9.-]*$ ]] || { echo "bad VARIANT '$VARIANT'" >&2; exit 2; }
[[ "$CONC" =~ ^[1-9][0-9]*$ ]] || { echo "bad CONC '$CONC'" >&2; exit 2; }
HERE="$(cd "$(dirname "$0")" && pwd)"
set -a; source "$HERE/../config/config.agg.sh"; set +a
name="$VARIANT-c$CONC"
OUT="$RUNS/tune/$name"
RES="$AGG/results/$VARIANT/c$CONC"
read -r -a ssh_args <<< "$SSH_OPTS"
q() { local out="" i; for i in "$@"; do printf -v i '%q' "$i"; out+="${out:+ }$i"; done; echo "$out"; }
on() { local n="$1"; shift; ssh "${ssh_args[@]}" "$n" "$(q "$@")" </dev/null; }
status() { printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$name" "$1" "${2:-}" | tee -a "$RUNS/tune/status.tsv"; }
die() { status "$1" "FAIL ${2:-}"; exit 1; }
control_ip="$(python3 "$BENCH_DIR/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
router_url="http://$control_ip:$ROUTER_PORT"

mkdir -p "$RUNS/tune"
[[ ! -e "$OUT" && ! -e "$RES" ]] || { echo "$OUT or $RES already exists" >&2; exit 2; }
bash "$R/scripts/wait_home_space.sh" 20 10800 || die pre "/home has no room"
mkdir -p "$OUT"
{ printf 'VARIANT=%s\nCONC=%s\nROUND=%s\nROUTER_ENV=%s\nDURATION=%s\n' \
    "$VARIANT" "$CONC" "$ROUND" "$ROUTER_ENV" "${DURATION:-$AGENTX_DURATION}"
  set | grep -E '^(MIXED_|HICACHE_|KV_OVERLAP_WEIGHT=|AGENTX_WARMUP)' ; } > "$OUT/round.env"
status start "ROUTER_ENV='$ROUTER_ENV' duration=${DURATION:-$AGENTX_DURATION}"

snapshot() {
    on "$CONTROL_NODE" bash -c 'rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2; free -g | head -2' \
        > "$OUT/mem.$1.txt" 2>&1
}
clients="$(on "$CONTROL_NODE" docker ps --format '{{.Names}}' | grep -c -- "^$CONTAINER_PREFIX-agentx-client-" || true)"
[[ "$clients" == 0 ]] || die pre "$clients AgentX client container(s) still running"
snapshot before

mapfile -t urls < <(python3 "$BENCH_DIR/tools/topology.py" rows "$TOPOLOGY" |
    awk -F'\t' -v b="$ENGINE_PORT_BASE" -v s="$ENGINE_PORT_STRIDE" '{print "http://" $5 ":" b + $1 * s}')
for url in "${urls[@]}"; do
    code=000
    for i in $(seq 1 90); do
        code="$(curl -s -o "$OUT/flush.body" -w '%{http_code}' --max-time 120 -X POST "$url/flush_cache")"
        [[ "$code" == 200 ]] && break
        sleep 10
    done
    printf '%s %s try=%s code=%s %s\n' "$(date -u +%FT%TZ)" "$url" "$i" "$code" \
        "$(tr '\n' ' ' < "$OUT/flush.body" | cut -c1-120)" >> "$OUT/flush.txt"
    [[ "$code" == 200 ]] || die flush "$url still $code after $i tries"
done
status flush "ok ${#urls[@]} worker(s)"

bash "$HERE/router_agg.sh" "ROUND=$ROUND" "ROUTER_ENV=$ROUTER_ENV" "TAG=$name-$(date -u +%H%M%S)" \
    > "$OUT/router.restart.log" 2>&1 || die router "restart failed (see router.restart.log)"
status router "$(tail -1 "$OUT/router.restart.log")"

python3 "$HERE/agg_sampler.py" "$OUT/samples.jsonl" 10 "${urls[@]}" 2>> "$OUT/sampler.log" &
sampler=$!
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/bench.start"
status bench "start C$CONC"
# Through a pipe: agentx_bench.sh runs under set -e, and a failed write to a full /home on its own
# stdout would kill it mid-run; tee only warns and keeps draining.
bash "$BENCH_DIR/agentx_bench.sh" "CONFIG=$AGG/config/config.agg.sh" "ROUND=$ROUND" "CONC=$CONC" \
    "OUT_DIR=$OUT/agentx-$name" ${DURATION:+"DURATION=$DURATION"} \
    AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true 2>&1 |
    tee --output-error=warn "$OUT/agentx.log" > /dev/null
rc=${PIPESTATUS[0]}
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/bench.end"
kill "$sampler" 2>/dev/null
snapshot after
curl -fsS --max-time 10 "$router_url/metrics" > "$OUT/router.metrics.txt" 2>/dev/null
on "$CONTROL_NODE" docker logs "$CONTAINER_PREFIX-router" 2>&1 | sed -E 's/\x1b\[[0-9;]*m//g' |
    grep -E ' (WARN|ERROR) ' | tail -200 > "$OUT/router.warn.txt"

src="$OUT/agentx-$name"
mkdir -p "$RES"
cp -p "$src"/agentx_conc*.json "$src"/runtime.env "$src"/benchmark_command.txt \
    "$src"/workload_distribution_summary.txt "$src"/metrics_plots.png "$RES"/ 2>/dev/null
cp -p "$src"/aiperf_artifacts/profile_export_aiperf.json "$src"/aiperf_artifacts/profile_export_aiperf.csv "$RES"/ 2>/dev/null
[[ -d "$src/service" ]] && cp -rp "$src/service" "$RES/service"
cp -p "$OUT"/round.env "$OUT"/flush.txt "$OUT"/mem.*.txt "$OUT"/router.warn.txt "$OUT"/samples.jsonl "$RES"/ 2>/dev/null
grep -E '^infera_router_session' "$OUT/router.metrics.txt" > "$RES/router.session_metrics.txt" 2>/dev/null
python3 "$HERE/agg_report.py" "$RES" > "$RES/summary.txt" 2>&1
cat "$RES/summary.txt"
(( rc == 0 )) || die bench "agentx_bench rc=$rc (partial results published)"
status done "$(head -1 "$RES/summary.txt")"
