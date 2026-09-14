#!/usr/bin/env bash
# Run the official InferenceX GSM8K check on the router's host.
set -euo pipefail
COMPONENT=gsm8k
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$DIR/../../.." && pwd)"
source "$DIR/lib/common.sh"

load_config "$@"
init_ssh
OUT_DIR="${OUT_DIR:-$DIR/results/$(date -u +%Y%m%dT%H%M%SZ)-gsm8k}"
[[ "$OUT_DIR" == /* ]] || OUT_DIR="$DIR/$OUT_DIR"
if [[ -d "$OUT_DIR" && -n "$(ls -A "$OUT_DIR")" ]]; then
    die "output directory is not empty: $OUT_DIR"
fi
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
RUN_LOG="$OUT_DIR/runner.log"
export OUT_DIR RUN_LOG
start_log

curl -fsS --max-time 30 "$(router_url)/health" >/dev/null ||
    die "router is not healthy at $(router_url)"
ssh_exec "$CONTROL_NODE" test -r "$MODEL/tokenizer_config.json" ||
    die "tokenizer is not readable at $MODEL on $CONTROL_NODE"

while IFS=$'\x1f' read -r instance role node ip gpus engine bootstrap kv snapshot container; do
    [[ "$role" == decode ]] || continue
    simulation="$(ssh_exec "$node" docker inspect --format \
        '{{range .Config.Env}}{{println .}}{{end}}' "$container" </dev/null |
        awk -F= '$1 == "SGLANG_SIMULATE_ACC_LEN" {print $2}')"
    [[ -z "$simulation" ]] ||
        die "$instance uses simulated MTP acceptance ($simulation); relaunch with DECODE_SIMULATE_ACC_LEN="
done < <(topology_rows)

run_token="$(date -u +%Y%m%dT%H%M%S)-$$"
client_container="$(service_container gsm8k-client)-$run_token"

mounts=(-v "$WORKSPACE_ROOT:$WORKSPACE_ROOT")
[[ "$MODEL" == "$WORKSPACE_ROOT"/* ]] || mounts+=(-v "$MODEL:$MODEL:ro")
[[ "$OUT_DIR" == "$WORKSPACE_ROOT"/* ]] || mounts+=(-v "$OUT_DIR:$OUT_DIR")

client_started=0
cleanup_client() {
    (( client_started == 1 )) || return 0
    ssh_exec "$CONTROL_NODE" docker stop --time 30 "$client_container" \
        >/dev/null 2>&1 || true
    ssh_exec "$CONTROL_NODE" docker rm -f "$client_container" \
        >/dev/null 2>&1 || true
}
trap cleanup_client EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
client_started=1
ssh_exec "$CONTROL_NODE" docker run --rm \
    --name "$client_container" \
    --network host --ipc host --shm-size "${CLIENT_SHM_SIZE:-32g}" \
    "${mounts[@]}" \
    -e "PORT=$ROUTER_PORT" \
    -e "MODEL=$MODEL" \
    -e "MODEL_NAME=$SERVED_MODEL" \
    -e MODEL_PREFIX=glm5.2 \
    -e "EVAL_CONCURRENT_REQUESTS=$EVAL_CONCURRENCY" \
    -e "EVAL_LIMIT=$EVAL_LIMIT" \
    -e "INFMAX_CONTAINER_WORKSPACE=$INFERENCEX_DIR" \
    -e "HOST_UID=$(id -u)" \
    -e "HOST_GID=$(id -g)" \
    "$IMAGE" bash -c '
        set -euo pipefail
        out="$1"
        trap "chown -R $HOST_UID:$HOST_GID \"$out\"" EXIT
        source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"
        run_lm_eval \
            --port "$PORT" \
            --task "$INFMAX_CONTAINER_WORKSPACE/utils/evals/gsm8k.yaml" \
            --results-dir "$out"
        result="$(find "$out" -type f -name "results*.json" -print -quit)"
        test -n "$result"
        python3 "$INFMAX_CONTAINER_WORKSPACE/utils/evals/validate_scores.py" \
            --model-prefix "$MODEL_PREFIX" \
            --results-glob "$result"
    ' _ "$OUT_DIR"
client_started=0
trap - EXIT INT TERM

log "PASS: $OUT_DIR"
