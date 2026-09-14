#!/usr/bin/env bash
# Gracefully stop only containers derived from config.sh and topology.tsv.
set -euo pipefail
COMPONENT=stop
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/lib/common.sh"

load_config "$@"
init_ssh
start_log
cleanup_rc=0

stop_one() {
    local node="$1" container="$2" state rc remove_output remove_rc attempt
    set +e
    state="$(ssh_exec "$node" docker inspect \
        --format '{{.State.Status}}' "$container" </dev/null 2>&1)"
    rc=$?
    set -e
    if (( rc != 0 )); then
        if [[ "${state,,}" == *"no such object"* ||
            "${state,,}" == *"no such container"* ]]; then
            log "$container is absent on $node"
            return
        fi
        log "cannot inspect $container on $node: $state"
        cleanup_rc=1
        return
    fi

    if [[ "$state" == running ]]; then
        log "gracefully stopping $container on $node (timeout=${CONTAINER_STOP_TIMEOUT}s)"
        if ! ssh_exec "$node" docker stop --time "$CONTAINER_STOP_TIMEOUT" \
            "$container" </dev/null; then
            log "stop failed; leaving $container in place for manual inspection"
            cleanup_rc=1
            return
        fi
    fi
    set +e
    remove_output="$(ssh_exec "$node" docker rm "$container" </dev/null 2>&1)"
    remove_rc=$?
    set -e
    if (( remove_rc != 0 )); then
        if [[ "${remove_output,,}" == *"no such object"* ||
            "${remove_output,,}" == *"no such container"* ]]; then
            return
        fi
        if [[ "${remove_output,,}" == *"removal of container"* &&
            "${remove_output,,}" == *"already in progress"* ]]; then
            for ((attempt = 1; attempt <= 30; attempt++)); do
                ssh_exec "$node" docker inspect "$container" \
                    >/dev/null 2>&1 || return 0
                sleep 1
            done
        fi
        log "failed to remove stopped container $container on $node: $remove_output"
        cleanup_rc=1
    fi
}

stop_matching() {
    local node="$1" prefix="$2" name
    while IFS= read -r name; do
        [[ -n "$name" ]] && stop_one "$node" "$name"
    done < <(
        ssh_exec "$node" docker ps -a --format '{{.Names}}' </dev/null |
            awk -v prefix="$prefix" 'index($0, prefix) == 1'
    )
}

stop_matching "$CONTROL_NODE" "$(service_container agentx-client)-"
stop_matching "$CONTROL_NODE" "$(service_container gsm8k-client)-"
while IFS=$'\x1f' read -r instance role node ip gpus engine bootstrap kv snapshot container; do
    stop_matching "$node" "$CONTAINER_PREFIX-preflight-"
done < <(topology_rows)
stop_one "$CONTROL_NODE" "$(service_container router)"
while IFS=$'\x1f' read -r instance role node ip gpus engine bootstrap kv snapshot container; do
    stop_one "$node" "$container"
done < <(topology_rows)
stop_one "$CONTROL_NODE" "$(service_container etcd)"

while IFS=$'\x1f' read -r instance role node ip gpus engine bootstrap kv snapshot container; do
    log "residual GPU state: $instance on $node"
    ssh_exec "$node" rocm-smi --showpids --showmemuse </dev/null || cleanup_rc=1
done < <(topology_rows)

(( cleanup_rc == 0 )) || die "cleanup completed with errors"
log "PASS: topology containers are stopped"
