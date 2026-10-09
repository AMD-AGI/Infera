#!/usr/bin/env bash
# Purpose: Run and validate the official InferenceX GSM8K evaluation.
# Usage: ./eval/gsm8k.sh [OUT_DIR=PATH] [EVAL_CONCURRENCY=N] [KEY=VALUE ...]
# Artifacts: runner.log and InferenceX/lm-eval results*.json files.
# Artifact paths: OUT_DIR defaults to results/<UTC>-gsm8k.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$DIR/../.." && pwd)"

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
INFERENCEX_DIR="${INFERENCEX_DIR:-$DIR/.cache/InferenceX}"
python3 "$DIR/tools/ensure_inferencex.py" \
    --directory "$INFERENCEX_DIR" --repository "$INFERENCEX_REPOSITORY" \
    --ref "$INFERENCEX_REF"
INFERENCEX_DIR="$(cd "$INFERENCEX_DIR" && pwd)"

rows() {
    python3 "$DIR/tools/topology.py" rows "$TOPOLOGY"
}
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
control_ip="$(python3 "$DIR/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
router_url="http://$control_ip:$ROUTER_PORT"
curl -fsS --max-time 30 "$router_url/health" >/dev/null ||
    { echo "router is not healthy at $router_url" >&2; exit 1; }
ssh_run "$CONTROL_NODE" test -r "$MODEL/tokenizer_config.json" ||
    { echo "tokenizer is not readable at $MODEL on $CONTROL_NODE" >&2; exit 1; }
while IFS=$'\t' read -r index instance role node ip; do
    [[ "$role" == decode ]] || continue
    simulation="$(ssh_run "$node" docker inspect --format \
        '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER_PREFIX-$instance" |
        awk -F= '$1=="SGLANG_SIMULATE_ACC_LEN"{print $2}')"
    [[ -z "$simulation" ]] ||
        { echo "$instance uses simulated acceptance; relaunch with DECODE_SIMULATE_ACC_LEN=" >&2; exit 1; }
done < <(rows)

OUT_DIR="${OUT_DIR:-$DIR/results/$(date -u +%Y%m%dT%H%M%SZ)-gsm8k}"
[[ "$OUT_DIR" == /* ]] || OUT_DIR="$DIR/$OUT_DIR"
[[ ! -e "$OUT_DIR" ]] || { echo "output path already exists: $OUT_DIR" >&2; exit 1; }
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
name="$CONTAINER_PREFIX-gsm8k-client-$(date -u +%Y%m%dT%H%M%S)-$$"
cleanup() {
    ssh_run "$CONTROL_NODE" docker rm -f "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mounts=(-v "$REPO:$REPO" -v "$INFERENCEX_DIR:$INFERENCEX_DIR:ro")
[[ "$MODEL" == "$REPO"/* ]] || mounts+=(-v "$MODEL:$MODEL:ro")
[[ "$OUT_DIR" == "$REPO"/* ]] || mounts+=(-v "$OUT_DIR:$OUT_DIR")
ssh_run "$CONTROL_NODE" docker run --rm --name "$name" \
    --network host --ipc host --shm-size "${CLIENT_SHM_SIZE:-32g}" \
    "${mounts[@]}" \
    -e "PORT=$ROUTER_PORT" -e "MODEL=$MODEL" -e "MODEL_NAME=$SERVED_MODEL" \
    -e MODEL_PREFIX=glm5.2 \
    -e "EVAL_CONCURRENT_REQUESTS=$EVAL_CONCURRENCY" -e "EVAL_LIMIT=$EVAL_LIMIT" \
    -e "INFMAX_CONTAINER_WORKSPACE=$INFERENCEX_DIR" \
    -e "HOST_UID=$(id -u)" -e "HOST_GID=$(id -g)" \
    "$IMAGE" bash -c '
        set -euo pipefail
        out="$1"
        trap "chown -R $HOST_UID:$HOST_GID \"$out\"" EXIT
        source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"
        run_lm_eval --port "$PORT" \
            --task "$INFMAX_CONTAINER_WORKSPACE/utils/evals/gsm8k.yaml" \
            --results-dir "$out"
        result="$(find "$out" -type f -name "results*.json" -print -quit)"
        test -n "$result"
        python3 "$INFMAX_CONTAINER_WORKSPACE/utils/evals/validate_scores.py" \
            --model-prefix "$MODEL_PREFIX" --results-glob "$result"
    ' _ "$OUT_DIR" 2>&1 | tee "$OUT_DIR/runner.log"
trap - EXIT INT TERM
echo "GSM8K passed: $OUT_DIR"
