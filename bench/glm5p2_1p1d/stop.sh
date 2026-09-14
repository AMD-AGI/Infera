#!/usr/bin/env bash
# Stop this experiment's containers and report residual GPU state.
# Usage: bash stop.sh  # FORCE_STOP=1 bypasses an active run lock
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/config.sh"

if [[ "${FORCE_STOP:-0}" == "1" ]]; then
    echo "[stop] FORCE_STOP=1; bypassing lifecycle lock"
else
    acquire_bench_lock
fi

cleanup_rc=0
remove_containers() {
    local node="$1" container state removed attempt force_remove
    local -a remove_args
    shift
    for container in "$@"; do
        if state="$(ssh $SSH_OPTS "$node" docker inspect \
            -f 'status={{.State.Status}},exit_code={{.State.ExitCode}},oom_killed={{.State.OOMKilled}}' \
            "$container" 2>/dev/null)"; then
            echo "[stop] removing $container on $node (RUN_ID=$RUN_ID; $state)"
            force_remove=0
            if [[ "$state" == status=running,* ]]; then
                echo "[stop] gracefully stopping $container (timeout=${CONTAINER_STOP_TIMEOUT}s)"
                if ! ssh $SSH_OPTS "$node" docker stop \
                    --time "$CONTAINER_STOP_TIMEOUT" "$container" >/dev/null; then
                    echo "[stop] graceful stop failed for $container on $node; falling back to force" >&2
                    force_remove=1
                fi
            fi
            removed=0
            for attempt in 1 2 3; do
                remove_args=(docker rm)
                (( force_remove == 1 )) && remove_args+=(-f)
                remove_args+=("$container")
                if ssh $SSH_OPTS "$node" "${remove_args[@]}" >/dev/null; then
                    removed=1
                    break
                fi
                force_remove=1
                (( attempt < 3 )) && {
                    echo "[stop] remove attempt $attempt/3 failed for $container on $node; retrying" >&2
                    sleep 10
                }
            done
            if (( removed == 0 )); then
                echo "[stop] failed to remove $container on $node; continuing cleanup" >&2
                cleanup_rc=1
            fi
        fi
    done
}

remove_containers "$PREFILL_NODE" \
    glm52-agentx-client glm52-lm-eval-client \
    glm52-pd-router glm52-pd-prefill glm52-pd-etcd
remove_containers "$DECODE_NODE" glm52-pd-decode

sleep 15
for node in "$PREFILL_NODE" "$DECODE_NODE"; do
    echo "===== $node residual GPU state ====="
    ssh $SSH_OPTS "$node" 'rocm-smi --showpids; rocm-smi --showmemuse' || true
done

exit "$cleanup_rc"
