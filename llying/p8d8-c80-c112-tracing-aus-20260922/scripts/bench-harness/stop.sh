#!/usr/bin/env bash
# Purpose: Stop benchmark containers derived from config.sh and topology.tsv.
# Usage: ./stop.sh [CONTAINER_STOP_TIMEOUT=SECONDS] [KEY=VALUE ...]
# Artifacts: terminal cleanup status and residual GPU state; no files.
# Artifact paths: not applicable.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

for assignment in "$@"; do
    [[ "$assignment" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] ||
        { echo "expected KEY=VALUE, got '$assignment'" >&2; exit 2; }
    export "$assignment"
done
CONFIG="${CONFIG:-$DIR/config.sh}"
[[ -r "$CONFIG" ]] || { echo "config is not readable: $CONFIG" >&2; exit 1; }
set -a
source "$CONFIG"
set +a
TOPOLOGY="${TOPOLOGY:-$DIR/topology.tsv}"
: "${CONTROL_NODE:?set CONTROL_NODE}"
: "${CONTAINER_PREFIX:?set CONTAINER_PREFIX}"
read -r -a ssh_args <<< "${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10}"
remote_command() {
    local result="" item quoted
    for item in "$@"; do
        printf -v quoted '%q' "$item"
        result+="${result:+ }$quoted"
    done
    echo "$result"
}
ssh_run() {
    local node="$1"
    shift
    ssh -n "${ssh_args[@]}" "$node" "$(remote_command "$@")"
}
rows() {
    python3 "$DIR/tools/topology.py" rows "$TOPOLOGY"
}

stop_one() {
    local node="$1" container="$2" state
    state="$(ssh_run "$node" docker inspect --format '{{.State.Status}}' "$container" 2>&1)" || {
        [[ "${state,,}" == *"no such"* ]] && { echo "$container absent on $node"; return; }
        echo "cannot inspect $container on $node: $state" >&2
        return 1
    }
    if [[ "$state" == running ]]; then
        echo "stopping $container on $node (timeout=${CONTAINER_STOP_TIMEOUT:-300}s)"
        if ! ssh_run "$node" docker stop --time "${CONTAINER_STOP_TIMEOUT:-300}" "$container"; then
            echo "graceful stop timed out; forcing $container on $node" >&2
            ssh_run "$node" docker rm -f "$container"
            return
        fi
    fi
    ssh_run "$node" docker rm "$container" >/dev/null 2>&1 ||
        ssh_run "$node" docker rm -f "$container" >/dev/null
}

stop_prefix() {
    local node="$1" prefix="$2" name names
    names="$(
        ssh_run "$node" docker ps -a --format '{{.Names}}' |
            awk -v wanted="$prefix" 'index($0,wanted)==1'
    )" || return 1
    while IFS= read -r name; do
        [[ -n "$name" ]] && stop_one "$node" "$name"
    done <<<"$names"
}

topology_rows="$(rows)"
status=0
stop_prefix "$CONTROL_NODE" "$CONTAINER_PREFIX-agentx-client-" || status=1
stop_prefix "$CONTROL_NODE" "$CONTAINER_PREFIX-gsm8k-client-" || status=1
stop_one "$CONTROL_NODE" "$CONTAINER_PREFIX-router" || status=1
while IFS=$'\t' read -r index instance role node ip; do
    stop_one "$node" "$CONTAINER_PREFIX-$instance" || status=1
done <<<"$topology_rows"
stop_one "$CONTROL_NODE" "$CONTAINER_PREFIX-etcd" || status=1

while IFS=$'\t' read -r index instance role node ip; do
    echo "residual GPU state: $instance on $node"
    ssh_run "$node" rocm-smi --showpids --showmemuse || status=1
done <<<"$topology_rows"
(( status == 0 )) || { echo "cleanup completed with errors" >&2; exit 1; }
echo "topology containers are stopped"
