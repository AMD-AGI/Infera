#!/usr/bin/env bash
# 3P(TP4)1D(TP8) pipeline: preflight -> up (decode at REAL acceptance) -> census -> KV gate
# -> switch decode to simulated acceptance -> smoke -> AgentX points (one stack, in order).
# Run detached on the control node:
#   ROUND=r1 POINTS="144 160 176" setsid nohup bash run_3p1d.sh all > <log> 2>&1 < /dev/null &
# Stages can be run alone: preflight | up | gate | switch | bench (POINTS) | stop.
# The stack is left running at the end (and on failure) for inspection; `stop` tears it down.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
CONFIG="$R/config/config.3p1d.sh"
: "${ROUND:?set ROUND}"
POINTS="${POINTS:-144 160 176}"
export ROUND
set -a; source "$CONFIG"; set +a
H="$BENCH_DIR"
LAUNCH="$RUNS/launch"
mkdir -p "$RUNS"
read -r -a ssh_args <<< "$SSH_OPTS"
q() { local out="" i; for i in "$@"; do printf -v i '%q' "$i"; out+="${out:+ }$i"; done; echo "$out"; }
# </dev/null: ssh must not consume the stdin of a surrounding `while read` loop.
on() { local n="$1"; shift; ssh "${ssh_args[@]}" "$n" "$(q "$@")" </dev/null; }
status() { printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "${2:-}" | tee -a "$RUNS/status.tsv"; }
die() { status "$1" "FAIL ${2:-}"; exit 1; }
control_ip="$(python3 "$H/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
router_url="http://$control_ip:$ROUTER_PORT"

# GPUs this deployment uses on each node (from the same per-instance overrides launch.sh reads).
node_gpus() {
    python3 "$H/tools/topology.py" rows "$TOPOLOGY" | while IFS=$'\t' read -r index inst role node ip; do
        var="GPUS_${inst//-/_}"; def="$PREFILL_GPU_DEVICES"; [[ "$role" == decode ]] && def="$DECODE_GPU_DEVICES"
        printf '%s\t%s\n' "$node" "${!var:-$def}"
    done | awk -F'\t' '{g[$1]=g[$1] (g[$1]?",":"") $2} END {for (n in g) print n "\t" g[n]}'
}

preflight() {
    local node gpus id used
    while IFS=$'\t' read -r node gpus; do
        id="$(on "$node" docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null)"
        [[ "$id" == "$EXPECTED_IMAGE_ID" ]] || die preflight "$node image $id"
        used="$(on "$node" bash -c "rocm-smi --showmeminfo vram --json" | python3 -c '
import json,sys
d=json.load(sys.stdin)
out=[]
for i in sys.argv[1].split(","):
    out.append("%s:%d" % (i, int(d["card" + i]["VRAM Total Used Memory (B)"]) // 1048576))
print(" ".join(out))' "$gpus")"
        for kv in $used; do
            (( ${kv#*:} < 1024 )) || die preflight "$node GPU ${kv%:*} holds ${kv#*:} MiB"
        done
        on "$node" docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_PREFIX-" &&
            die preflight "$node already has $CONTAINER_PREFIX-* containers"
        status preflight "$node image ok, GPUs $used MiB"
    done < <(node_gpus)
}

up() {
    [[ ! -e "$LAUNCH" ]] || die up "$LAUNCH exists"
    status up "start (DECODE_SIMULATE_ACC_LEN='${DECODE_SIMULATE_ACC_LEN}')"
    bash "$H/launch.sh" "CONFIG=$CONFIG" "ROUND=$ROUND" \
        "DECODE_SIMULATE_ACC_LEN=$DECODE_SIMULATE_ACC_LEN" "OUT_DIR=$LAUNCH" \
        > "$RUNS/launch.log" 2>&1 || die up "launch.sh rc=$? (see launch.log)"
    status up ok
}

census() {  # census <label> [since]
    bash "$HERE/census.sh" "$LAUNCH/server-logs" "${2:-}" > "$RUNS/census.$1.txt" 2>&1
    local rc=$?
    status "census.$1" "rc=$rc $(grep -c FATAL "$RUNS/census.$1.txt") fatal-lines"
    return $rc
}

# Two tiers (r2, 2026-10-12): easy documents (~1.7k / ~2.7k tokens, dense and sparse DSA
# paths) must be >= 95% exact; on hard ones (~41k / ~123k tokens) a correct deployment
# still misses some retrievals (the answer is another record's code), so there only
# fabricated / garbled answers (corrupted-KV signature) fail the gate, plus a 60% floor.
gate() {
    local rc_easy rc_hard summary='import json,sys; s=json.load(open(sys.argv[1]))["summary"]; print(s["counts"], s["correct_rate"], s["prompt_tokens_min_max"])'
    status gate start
    python3 "$HERE/kv_gate.py" --url "$router_url" --model "$SERVED_MODEL" \
        --short 32 --short-records 200 --long 16 --long-records 120 \
        --min-correct 0.95 --max-fabricated 0 \
        --out "$RUNS/gate.easy.json" > "$RUNS/gate.easy.log" 2>&1
    rc_easy=$?
    status gate.easy "rc=$rc_easy $(python3 -c "$summary" "$RUNS/gate.easy.json" 2>/dev/null)"
    python3 "$HERE/kv_gate.py" --url "$router_url" --model "$SERVED_MODEL" \
        --short 40 --long 8 --min-correct 0.60 --max-fabricated 0 \
        --out "$RUNS/gate.hard.json" > "$RUNS/gate.hard.log" 2>&1
    rc_hard=$?
    status gate.hard "rc=$rc_hard $(python3 -c "$summary" "$RUNS/gate.hard.json" 2>/dev/null)"
    (( rc_easy == 0 && rc_hard == 0 ))
}

switch() {
    status switch "decode -> SIMULATE_ACC_LEN=3.61"
    bash "$HERE/restart_decode.sh" "ROUND=$ROUND" SIM=3.61 "LAUNCH_DIR=$LAUNCH" TAG=realacc \
        > "$RUNS/switch.log" 2>&1 || die switch "rc=$? (see switch.log)"
    status switch ok
}

smoke() {
    local out
    out="$(curl -fsS --max-time 300 "$router_url/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$SERVED_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"max_tokens\":32,\"temperature\":0}")" ||
        die smoke "request failed"
    echo "$out" > "$RUNS/smoke.json"
    status smoke "ok (text is garbled by design under simulated acceptance)"
}

publish() {  # copy the small per-point artifacts into results/<round>/c<CONC>/
    local c="$1" src="$RUNS/agentx-c$1" dst="$R/results/$ROUND/c$1"
    mkdir -p "$dst"
    cp -p "$src"/agentx_conc*.json "$src"/runtime.env "$src"/benchmark_command.txt \
        "$src"/workload_distribution_summary.txt "$src"/metrics_plots.png "$dst"/ 2>/dev/null
    cp -p "$src"/profile_export_aiperf.json "$src"/profile_export_aiperf.csv "$dst"/ 2>/dev/null
    [[ -d "$src/service" ]] && cp -rp "$src/service" "$dst/service"
    cp -p "$RUNS/census.c$1.txt" "$dst"/ 2>/dev/null
    return 0
}

bench() {
    local c start rc
    for c in $POINTS; do
        curl -fsS --max-time 10 "$router_url/health" >/dev/null || die "bench.c$c" "router unhealthy before start"
        start="$(date -u +%Y-%m-%dT%H:%M:%S)"
        status "bench.c$c" start
        bash "$H/agentx_bench.sh" "CONFIG=$CONFIG" "ROUND=$ROUND" "CONC=$c" \
            "OUT_DIR=$RUNS/agentx-c$c" > "$RUNS/agentx-c$c.log" 2>&1
        rc=$?
        census "c$c" "$start"
        status "bench.c$c" "rc=$rc $(tail -1 "$RUNS/agentx-c$c.log" | cut -c1-200)"
        [[ "$rc" == 0 || -s "$RUNS/agentx-c$c/agentx_conc$c.json" ]] ||
            die "bench.c$c" "no AgentX result (rc=$rc); not running later points"
        publish "$c"
        curl -fsS --max-time 10 "$router_url/health" >/dev/null || die "bench.c$c" "router unhealthy after point"
    done
    status bench done
}

stop_stack() {
    bash "$H/stop.sh" "CONFIG=$CONFIG" "ROUND=$ROUND" > "$RUNS/stop.log" 2>&1
    status stop "rc=$?"
}

after_up() {
    # CENSUS_SINCE (docker timestamp) skips log lines from before a manual recovery.
    census bringup "${CENSUS_SINCE:-}" || die census.bringup "fatal patterns at bring-up"
    gate || die gate "known-answer gate failed (stack left up)"
    census gate || die census.gate "fatal patterns during the gate"
    export DECODE_SIMULATE_ACC_LEN=3.61
    switch
    # Stopping the old decode legitimately logs session failures on the prefills;
    # judge only what happens after the new decode and router are up.
    local switched
    switched="$(date -u +%Y-%m-%dT%H:%M:%S)"
    smoke
    census switch "$switched" || die census.switch "fatal patterns after the switch"
    bench
}

case "${1:-all}" in
    preflight) preflight ;;
    up) up ;;
    gate) gate ;;
    switch) switch ;;
    smoke) smoke ;;
    bench) bench ;;
    stop) stop_stack ;;
    after-up)
        # Resume on a stack that is already up with decode at real acceptance.
        status all "resume after-up ROUND=$ROUND POINTS='$POINTS'"
        export DECODE_SIMULATE_ACC_LEN=""
        after_up
        status all done
        ;;
    all)
        status all "start ROUND=$ROUND POINTS='$POINTS'"
        preflight
        export DECODE_SIMULATE_ACC_LEN=""
        up
        after_up
        status all done
        ;;
    *) echo "unknown stage: $1" >&2; exit 2 ;;
esac
