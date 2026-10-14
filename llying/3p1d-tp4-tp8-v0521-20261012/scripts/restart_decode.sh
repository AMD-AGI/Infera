#!/usr/bin/env bash
# Restart only decode-0 with another acceptance mode, then restart the router, on a
# live stack. Used to run the known-answer gate at real acceptance and then switch
# decode to the simulated acceptance the benchmark uses, without tearing down the
# prefills (whose HiCache host pools are slow to release).
# Usage: restart_decode.sh ROUND=<r> SIM=<3.61|> LAUNCH_DIR=<launch out dir> TAG=<retired-suffix>
# The retired decode and router containers are stopped and renamed <name>-<TAG>
# (kept for their logs); the new ones take the original names.
set -euo pipefail
for a in "$@"; do
    [[ "$a" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "expected KEY=VALUE, got '$a'" >&2; exit 2; }
    export "$a"
done
: "${ROUND:?}" "${LAUNCH_DIR:?}" "${TAG:?}"
[[ -v SIM ]] || { echo "set SIM (3.61 or empty)" >&2; exit 2; }
HERE="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$HERE/../config/config.3p1d.sh"
export DECODE_SIMULATE_ACC_LEN="$SIM"
set -a; source "$CONFIG"; set +a
H="$BENCH_DIR"
read -r -a ssh_args <<< "$SSH_OPTS"
q() { local out="" i; for i in "$@"; do printf -v i '%q' "$i"; out+="${out:+ }$i"; done; echo "$out"; }
on() { local n="$1"; shift; ssh "${ssh_args[@]}" "$n" "$(q "$@")" </dev/null; }

IFS=$'\t' read -r index instance role node ip < <(python3 "$H/tools/topology.py" rows "$TOPOLOGY" | awk -F'\t' '$3=="decode"' | head -1)
gpu_var="GPUS_${instance//-/_}"; gpus="${!gpu_var:-$DECODE_GPU_DEVICES}"
engine=$((ENGINE_PORT_BASE + index * ENGINE_PORT_STRIDE))
bootstrap=$((BOOTSTRAP_PORT_BASE + index))
kv=$((KV_EVENT_PORT_BASE + index * KV_EVENT_PORT_STRIDE))
snapshot=$((SNAPSHOT_PORT_BASE + index))
container="$CONTAINER_PREFIX-$instance"
router="$CONTAINER_PREFIX-router"
control_ip="$(python3 "$H/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
[[ "$container" == *llying* && "$router" == *llying* ]] || { echo "refusing: names not ours" >&2; exit 2; }
echo "$(date -u +%FT%TZ) restart $instance on $node (gpus=$gpus port=$engine) SIM='${SIM}'"

# Either container may be absent (e.g. a bring-up that died before the router started).
retire() {  # retire <node> <container> <stop timeout>
    if on "$1" docker inspect "$2" >/dev/null 2>&1; then
        on "$1" docker stop --time "$3" "$2" >/dev/null
        on "$1" docker rename "$2" "$2-$TAG"
        echo "retired $2 on $1 as $2-$TAG"
    else
        echo "$2 not present on $1"
    fi
}
retire "$CONTROL_NODE" "$router" 60
retire "$node" "$container" 120

# Wait until the decode GPUs have released their memory.
for i in $(seq 1 120); do
    used="$(on "$node" bash -c "rocm-smi --showmeminfo vram --json" | python3 -c '
import json,sys
d=json.load(sys.stdin); g=[int(x) for x in sys.argv[1].split(",")]
print(max(int(d[f"card{i}"]["VRAM Total Used Memory (B)"])//1048576 for i in g))' "$gpus")"
    (( used < 2048 )) && break
    (( i < 120 )) || { echo "decode GPUs still hold ${used} MiB after 20 min" >&2; exit 1; }
    sleep 10
done
echo "$(date -u +%FT%TZ) decode GPUs free (max ${used} MiB)"

on "$node" bash "$H/engine.sh" decode "$instance" "$ip" "$gpus" "$engine" "$bootstrap" \
    "$kv" "$snapshot" "$container" "$control_ip:$ETCD_PORT" \
    "CONFIG=$CONFIG" "ROUND=$ROUND" "DECODE_SIMULATE_ACC_LEN=$SIM" \
    "SERVER_LOG=$LAUNCH_DIR/server-logs/$instance.after-$TAG.log" </dev/null
python3 "$H/tools/wait_healthy.py" --ssh-options "$SSH_OPTS" \
    --timeout "$READY_TIMEOUT" --interval "$HEALTH_INTERVAL" --probe-timeout 5 \
    --failure-dir "$LAUNCH_DIR/failures" --summary "$LAUNCH_DIR/decode-health.after-$TAG.json" \
    --target "$instance" "$node" "$container" "http://$ip:$engine/health"

# Router: same command as launch.sh.
affinity=false; [[ "$PD_DP_RANK_AFFINITY" == 1 ]] && affinity=true
key="${CONTROL_NODE##*-}"
on "$CONTROL_NODE" docker run -d --init --name "$router" --network host \
    -e "INFERA_PD_DP_RANK_AFFINITY=$affinity" \
    -v "$WS:$WS:ro" -v "$NODE_RW_ROOT:$NODE_RW_ROOT" -v "$MODEL:$MODEL:ro" \
    --tmpfs "$NODE_RW_ROOT/tmp/$key/router:rw,exec,size=2g" \
    -e "TMPDIR=$NODE_RW_ROOT/tmp/$key/router" \
    -e "HF_HOME=$NODE_RW_ROOT/cache/$key/router" \
    -e "XDG_CACHE_HOME=$NODE_RW_ROOT/cache/$key/router" -e PYTHONDONTWRITEBYTECODE=1 \
    "$IMAGE" python3 -m infera.server --host 0.0.0.0 --port "$ROUTER_PORT" \
    --router-backend rust --discovery-backend etcd --etcd-endpoint "$control_ip:$ETCD_PORT" \
    --request-transport http --kv-event-transport zmq --router-tokenizer-path "$MODEL" \
    --router-policy kv-aware --kv-prefill-overlap-weight "$KV_PREFILL_OVERLAP_WEIGHT" \
    --kv-decode-overlap-weight "$KV_DECODE_OVERLAP_WEIGHT" >/dev/null
python3 "$H/tools/wait_healthy.py" --ssh-options "$SSH_OPTS" --timeout 300 --interval 5 \
    --probe-timeout 5 --failure-dir "$LAUNCH_DIR/failures" \
    --summary "$LAUNCH_DIR/router-health.after-$TAG.json" \
    --target router "$CONTROL_NODE" "$router" "http://$control_ip:$ROUTER_PORT/health"
expected=$(( $(python3 "$H/tools/topology.py" count "$TOPOLOGY" prefill) + 1 ))
for i in $(seq 1 30); do
    n="$(curl -fsS --max-time 10 "http://$control_ip:$ROUTER_PORT/v1/workers" | python3 -c '
import json,sys
p=json.load(sys.stdin); w=p if isinstance(p,list) else (p.get("workers") or p.get("data") or [])
print(len(w))')"
    [[ "$n" == "$expected" ]] && break
    (( i < 30 )) || { echo "router sees $n workers, expected $expected" >&2; exit 1; }
    sleep 5
done
curl -fsS --max-time 10 "http://$control_ip:$ROUTER_PORT/v1/workers" | python3 -m json.tool \
    > "$LAUNCH_DIR/workers.after-$TAG.json"
curl -fsS --max-time 30 "http://$ip:$engine/get_server_info" > "$LAUNCH_DIR/server-info/$instance.after-$TAG.json"
echo "$(date -u +%FT%TZ) decode + router restarted; router sees $n workers"
