#!/usr/bin/env bash
# Restart the three prefills with a config overlay on a live stack; decode stays up (warm).
# Usage: restart_prefills.sh ROUND=<r> OVERLAY=<config/...sh> TAG=<suffix> [INSTANCES="prefill-N ..."]
#   1. retire the router and the prefills (stopped, renamed <name>-<TAG>, kept for logs) and
#      move their server logs to <instance>.before-<TAG>.log, so the live log keeps its name;
#   2. wait for the prefill GPUs to drain and for enough host memory for the new pools;
#   3. install OVERLAY as config/round.<ROUND>.sh (everything that sources the config after
#      this point, including agentx_env.py validation, sees the new settings);
#   4. start the prefills (same-node ones one after another), wait healthy, check the new
#      KV / host pool sizes against the overlay, restart the router, smoke one request.
set -euo pipefail
for a in "$@"; do
    [[ "$a" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "expected KEY=VALUE, got '$a'" >&2; exit 2; }
    export "$a"
done
: "${ROUND:?}" "${OVERLAY:?}" "${TAG:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
CONFIG="$R/config/config.3p1d.sh"
[[ "$OVERLAY" == /* ]] || OVERLAY="$R/$OVERLAY"
[[ -r "$OVERLAY" ]] || { echo "overlay not readable: $OVERLAY" >&2; exit 2; }
set -a; source "$CONFIG"; set +a
H="$BENCH_DIR"
LAUNCH="$RUNS/launch"; LOGS="$LAUNCH/server-logs"
read -r -a ssh_args <<< "$SSH_OPTS"
q() { local out="" i; for i in "$@"; do printf -v i '%q' "$i"; out+="${out:+ }$i"; done; echo "$out"; }
on() { local n="$1"; shift; ssh "${ssh_args[@]}" "$n" "$(q "$@")" </dev/null; }
say() { echo "$(date -u +%FT%TZ) restart_prefills: $*"; }
control_ip="$(python3 "$H/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
router="$CONTAINER_PREFIX-router"
mapfile -t rows < <(python3 "$H/tools/topology.py" rows "$TOPOLOGY" | awk -F'\t' '$3=="prefill"')
(( ${#rows[@]} >= 1 )) || { echo "no prefill rows" >&2; exit 2; }
# INSTANCES="prefill-2 ..." restarts only those prefills (e.g. one that hung); default all.
if [[ -n "${INSTANCES:-}" ]]; then
    mapfile -t rows < <(printf '%s\n' "${rows[@]}" | awk -F'\t' -v want=" $INSTANCES " 'index(want, " " $2 " ")')
    (( ${#rows[@]} >= 1 )) || { echo "INSTANCES '$INSTANCES' matched no prefill row" >&2; exit 2; }
fi
say "ROUND=$ROUND OVERLAY=$OVERLAY TAG=$TAG"

# --- 1. retire
retire() {  # retire <node> <container>
    if on "$1" docker inspect "$2" >/dev/null 2>&1; then
        on "$1" docker stop --time 120 "$2" >/dev/null
        on "$1" docker rename "$2" "$2-$TAG"
        say "retired $2 on $1 as $2-$TAG"
    fi
}
[[ "$router" == *llying* ]] || exit 2
retire "$CONTROL_NODE" "$router"
pids=()
for row in "${rows[@]}"; do
    IFS=$'\t' read -r index instance role node ip <<< "$row"
    [[ "$CONTAINER_PREFIX-$instance" == *llying* ]] || exit 2
    retire "$node" "$CONTAINER_PREFIX-$instance" & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
for row in "${rows[@]}"; do
    IFS=$'\t' read -r index instance role node ip <<< "$row"
    c="$CONTAINER_PREFIX-$instance"
    for i in $(seq 1 60); do  # the docker-logs follower ends with its container
        n_follow="$(on "$node" bash -c "ps -eo args | grep -c '^[d]ocker logs .* $c\$' || true")"
        [[ "$n_follow" == 0 ]] && break
        sleep 5
    done
    [[ -e "$LOGS/$instance.log" ]] && mv "$LOGS/$instance.log" "$LOGS/$instance.before-$TAG.log"
done
say "prefills and router retired; logs moved to *.before-$TAG.log"

# --- 2. wait for GPUs and host memory
declare -A node_gpus node_need
for row in "${rows[@]}"; do
    IFS=$'\t' read -r index instance role node ip <<< "$row"
    gv="GPUS_${instance//-/_}"
    node_gpus[$node]="${node_gpus[$node]:+${node_gpus[$node]},}${!gv:-$PREFILL_GPU_DEVICES}"
done
# Host need per node from the overlay: KV pool GB x ranks x 1.23 (DSA indexer region) + 300 GB.
need_gb() {  # need_gb <node>
    (
        set -a; source "$CONFIG"; source "$OVERLAY"; set +a
        total=0
        for row in "${rows[@]}"; do
            IFS=$'\t' read -r index instance role node ip <<< "$row"
            [[ "$node" == "$1" ]] || continue
            sv="HICACHE_SIZE_${instance//-/_}"; size="${!sv:-${PREFILL_HICACHE_SIZE:-0}}"
            (( size > 0 )) || size=153
            total=$(( total + size * PREFILL_DP * 123 / 100 ))
        done
        echo $(( total + 300 ))
    )
}
for node in "${!node_gpus[@]}"; do
    need="$(need_gb "$node")"
    for i in $(seq 1 360); do
        used="$(on "$node" bash -c "rocm-smi --showmeminfo vram --json" | python3 -c '
import json,sys
d=json.load(sys.stdin)
print(max(int(d["card"+g]["VRAM Total Used Memory (B)"])//1048576 for g in sys.argv[1].split(",")))' "${node_gpus[$node]}")"
        avail="$(on "$node" bash -c "free -g | awk '/^Mem:/{print \$7}'")"
        # A GPU that hung can keep ~2 GiB of driver-level residue with no owning process.
        (( used < 4096 && avail >= need )) && break
        (( i % 12 == 1 )) && say "$node waiting: GPU max used ${used} MiB, host avail ${avail} GB (need ${need})"
        (( i < 360 )) || { say "$node not ready after 60 min (GPU ${used} MiB, host ${avail}/${need} GB)"; exit 1; }
        sleep 10
    done
    say "$node ready: GPU max used ${used} MiB, host avail ${avail} GB (need ${need})"
done

# --- 3. install the overlay
round_file="$R/config/round.$ROUND.sh"
[[ -e "$round_file" ]] && cp -p "$round_file" "$round_file.before-$TAG"
cp "$OVERLAY" "$round_file"
set -a; source "$CONFIG"; set +a
say "installed $(basename "$OVERLAY") as $(basename "$round_file"): PREFILL_MEM_FRACTION=$PREFILL_MEM_FRACTION"

# --- 4. start, verify
start_one() {  # start_one <row>
    local index instance role node ip gv gpus engine
    IFS=$'\t' read -r index instance role node ip <<< "$1"
    gv="GPUS_${instance//-/_}"; gpus="${!gv:-$PREFILL_GPU_DEVICES}"
    engine=$((ENGINE_PORT_BASE + index * ENGINE_PORT_STRIDE))
    on "$node" bash "$H/engine.sh" prefill "$instance" "$ip" "$gpus" "$engine" \
        $((BOOTSTRAP_PORT_BASE + index)) $((KV_EVENT_PORT_BASE + index * KV_EVENT_PORT_STRIDE)) \
        $((SNAPSHOT_PORT_BASE + index)) "$CONTAINER_PREFIX-$instance" "$control_ip:$ETCD_PORT" \
        "CONFIG=$CONFIG" "ROUND=$ROUND" "SERVER_LOG=$LOGS/$instance.log" > "$LAUNCH/start.$instance.after-$TAG.log" 2>&1
    python3 "$H/tools/wait_healthy.py" --ssh-options "$SSH_OPTS" \
        --timeout "$READY_TIMEOUT" --interval "$HEALTH_INTERVAL" --probe-timeout 5 \
        --failure-dir "$LAUNCH/failures" --summary "$LAUNCH/$instance-health.after-$TAG.json" \
        --target "$instance" "$node" "$CONTAINER_PREFIX-$instance" "http://$ip:$engine/health" \
        >> "$LAUNCH/start.$instance.after-$TAG.log" 2>&1
    curl -fsS --max-time 30 "http://$ip:$engine/get_server_info" > "$LAUNCH/server-info/$instance.after-$TAG.json"
}
# First instance on every node in parallel, then the remaining same-node instances.
declare -A seen
first=() rest=()
for row in "${rows[@]}"; do
    IFS=$'\t' read -r index instance role node ip <<< "$row"
    if [[ -z "${seen[$node]:-}" ]]; then seen[$node]=1; first+=("$row"); else rest+=("$row"); fi
done
for batch in first rest; do
    declare -n list="$batch"
    pids=()
    for row in "${list[@]}"; do start_one "$row" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p" || { say "a prefill failed to start (see $LAUNCH/start.*.after-$TAG.log)"; exit 1; }; done
    unset -n list
done
say "prefills healthy"

for row in "${rows[@]}"; do
    IFS=$'\t' read -r index instance role node ip <<< "$row"
    sv="HICACHE_SIZE_${instance//-/_}"; want="${!sv:-${PREFILL_HICACHE_SIZE:-0}}"
    kv="$(grep -m1 -o 'KV Cache is allocated.*#tokens: [0-9]*' "$LOGS/$instance.log" | grep -o '[0-9]*$' || true)"
    host="$(grep -m1 -o 'KV host pool: [0-9]* tokens, [0-9.]* GB' "$LOGS/$instance.log" | grep -o '[0-9.]* GB$' || true)"
    say "$instance: GPU KV ${kv:-?} tokens/rank, host pool ${host:-?} (overlay size ${want})"
    [[ -n "$kv" && -n "$host" ]] || { say "$instance: cannot read pool sizes from its log"; exit 1; }
    if (( want > 0 )); then
        awk -v h="${host% GB}" -v w="$want" 'BEGIN { exit !(h > w * 0.98 && h < w * 1.02) }' ||
            { say "$instance: host pool ${host} != ${want} GB"; exit 1; }
    fi
done

bash "$HERE/restart_router.sh" "ROUND=$ROUND" AFFINITY=false "TAG=unused-$TAG" > "$LAUNCH/router.after-$TAG.log" 2>&1 ||
    { say "router restart failed"; exit 1; }
curl -fsS --max-time 300 "http://$control_ip:$ROUTER_PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"max_tokens\":16,\"temperature\":0}" \
    > "$LAUNCH/smoke.after-$TAG.json" || { say "smoke request failed"; exit 1; }
say "done: router sees 4 workers, smoke ok"
