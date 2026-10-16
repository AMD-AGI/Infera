#!/usr/bin/env bash
# Bring up the aggregated stack on CONTROL_NODE: etcd -> the topology's mixed worker(s) -> router,
# then a one-request smoke through the router.
# Usage: launch_agg.sh ROUND=<new tag>
# Output: runtime/runs/<ROUND>/launch/ (engine commands, server logs, smoke response).
set -euo pipefail
for a in "$@"; do
    [[ "$a" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "expected KEY=VALUE, got '$a'" >&2; exit 2; }
    export "$a"
done
HERE="$(cd "$(dirname "$0")" && pwd)"
set -a; source "$HERE/../config/config.agg.sh"; set +a
read -r -a ssh_args <<< "$SSH_OPTS"
q() { local out="" i; for i in "$@"; do printf -v i '%q' "$i"; out+="${out:+ }$i"; done; echo "$out"; }
on() { local n="$1"; shift; ssh "${ssh_args[@]}" "$n" "$(q "$@")" </dev/null; }
LAUNCH="$RUNS/launch"
[[ ! -e "$LAUNCH" ]] || { echo "$LAUNCH exists; use a new ROUND" >&2; exit 2; }
mkdir -p "$LAUNCH/server-logs"
control_ip="$(python3 "$BENCH_DIR/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
etcd_endpoint="$control_ip:$ETCD_PORT"
mapfile -t rows < <(python3 "$BENCH_DIR/tools/topology.py" rows "$TOPOLOGY")

# --- preconditions: our ports free, no container of this prefix, the GPUs idle
ports=("$ETCD_PORT" "$ETCD_PEER_PORT" "$ROUTER_PORT")
for row in "${rows[@]}"; do
    IFS=$'\t' read -r index _ _ _ _ <<< "$row"
    ports+=("$((ENGINE_PORT_BASE + index * ENGINE_PORT_STRIDE))" "$((SNAPSHOT_PORT_BASE + index))")
done
busy="$(on "$CONTROL_NODE" ss -ltnH | awk '{print $4}' | sed 's/.*://' | sort -u |
    grep -xF -f <(printf '%s\n' "${ports[@]}") || true)"
[[ -z "$busy" ]] || { echo "ports in use on $CONTROL_NODE: $busy" >&2; exit 1; }
existing="$(on "$CONTROL_NODE" docker ps -a --format '{{.Names}}' | grep -c -- "^$CONTAINER_PREFIX-" || true)"
[[ "$existing" == 0 ]] || { echo "$existing container(s) named $CONTAINER_PREFIX-* exist" >&2; exit 1; }
for row in "${rows[@]}"; do
    IFS=$'\t' read -r _ instance _ node _ <<< "$row"
    gpus_var="GPUS_${instance//-/_}"
    on "$node" rocm-smi --showmemuse --json | python3 -c '
import json, sys
used = {int(k[4:]): float(v.get("GPU Memory Allocated (VRAM%)", 0)) for k, v in json.load(sys.stdin).items() if k.startswith("card")}
busy = [g for g in map(int, sys.argv[1].split(",")) if used.get(g, 100) > 1]
sys.exit(f"GPUs {busy} on {sys.argv[2]} are in use" if busy else 0)' "${!gpus_var}" "$node"
done

# --- etcd
on "$CONTROL_NODE" docker run -d --init --name "$CONTAINER_PREFIX-etcd" --network host \
    "${ETCD_IMAGE:-quay.io/coreos/etcd:v3.5.14}" etcd \
    --advertise-client-urls "http://$etcd_endpoint" \
    --listen-client-urls "http://0.0.0.0:$ETCD_PORT" \
    --listen-peer-urls "http://127.0.0.1:$ETCD_PEER_PORT" \
    --initial-advertise-peer-urls "http://127.0.0.1:$ETCD_PEER_PORT" \
    --initial-cluster "default=http://127.0.0.1:$ETCD_PEER_PORT" >/dev/null
for i in $(seq 1 60); do
    on "$CONTROL_NODE" docker exec "$CONTAINER_PREFIX-etcd" etcdctl --endpoints "http://127.0.0.1:$ETCD_PORT" \
        endpoint health >/dev/null 2>&1 && break
    (( i < 60 )) || { echo "etcd unhealthy" >&2; exit 1; }
    sleep 1
done
echo "etcd $etcd_endpoint ok"

# --- worker(s)
for row in "${rows[@]}"; do
    IFS=$'\t' read -r index instance role node ip <<< "$row"
    [[ "$role" == mixed ]] || { echo "$instance: only mixed workers are supported here" >&2; exit 2; }
    on "$node" env "ROUND=$ROUND" "SERVER_LOG=$LAUNCH/server-logs/$instance.log" \
        bash "$HERE/engine_mixed.sh" "$instance" "$ip" \
        "$((ENGINE_PORT_BASE + index * ENGINE_PORT_STRIDE))" "$((KV_EVENT_PORT_BASE + 8 * index))" \
        "$((SNAPSHOT_PORT_BASE + index))" "$CONTAINER_PREFIX-$instance" "$etcd_endpoint" \
        > "$LAUNCH/$instance.engine.txt" 2>&1 || { cat "$LAUNCH/$instance.engine.txt" >&2; exit 1; }
done
for row in "${rows[@]}"; do
    IFS=$'\t' read -r index instance _ node ip <<< "$row"
    url="http://$ip:$((ENGINE_PORT_BASE + index * ENGINE_PORT_STRIDE))"
    for i in $(seq 1 $((READY_TIMEOUT / 10))); do
        curl -fsS --max-time 5 "$url/health" >/dev/null 2>&1 && break
        state="$(on "$node" docker inspect -f '{{.State.Running}}' "$CONTAINER_PREFIX-$instance" 2>/dev/null || echo gone)"
        [[ "$state" == true ]] || { echo "$instance exited before it was healthy" >&2; exit 1; }
        (( i % 30 == 0 )) && echo "$(date -u +%T) waiting for $instance ($((i * 10)) s)"
        sleep 10
    done
    curl -fsS --max-time 5 "$url/health" >/dev/null || { echo "$instance not healthy after $READY_TIMEOUT s" >&2; exit 1; }
    curl -fsS --max-time 10 "$url/server_info" > "$LAUNCH/$instance.server_info.json" ||
        curl -fsS --max-time 10 "$url/get_server_info" > "$LAUNCH/$instance.server_info.json"
    echo "$instance healthy at $url"
done

# --- router, smoke
bash "$HERE/router_agg.sh" "ROUND=$ROUND" | tee "$LAUNCH/router.txt"
curl -fsS --max-time 300 "http://$control_ip:$ROUTER_PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 * 23? Answer with the number only.\"}],\"max_tokens\":2048,\"temperature\":0}" \
    > "$LAUNCH/smoke.json"
python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["choices"][0]["message"]
c = json.load(open(sys.argv[1]))["choices"][0]
print("smoke:", repr((m.get("content") or "")[-80:]), "| reasoning chars:", len(m.get("reasoning_content") or ""),
      "| finish:", c.get("finish_reason"))' "$LAUNCH/smoke.json"
