#!/usr/bin/env bash
# Sourced by an engine launcher after it declares its defaults and engine_configure.
set -euo pipefail

SUITE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
MODEL_IN_CONTAINER=/model
SERVED_MODEL=deepseek-ai/DeepSeek-V4.1-Flash

die() { printf '%s\n' "$*" >&2; exit 2; }

usage() {
    cat <<USAGE
Usage: $ENGINE/run.sh plan|run TAG [-- ENGINE_FLAGS...]

Environment:
  MODEL_DIR         Local DeepSeek-V4.1-Flash checkpoint (required for run)
  RUN_ROOT          Dedicated directory on a data or NFS mount (required for run)
  IMAGE             Docker image already available (default ${DEFAULT_IMAGE:-unset})
  GPUS              Comma-separated AMD GPU IDs (default $DEFAULT_GPUS)
  PORT              Loopback API port (default $DEFAULT_PORT)
  ISL / OSL         Exact prompt / output token lengths (default 10000 / 500)
  NUM_PROMPTS       Measured requests per concurrency point (default 128)
  WARMUP            Warmup requests per point (default 16, 0 disables)
  CONCURRENCIES     Comma-separated concurrency points (default 16,32)
  CONTEXT_LENGTH    Server context limit (default ISL + OSL + 32)
  SEED              Common random token seed (default 42)
  STARTUP_TIMEOUT   Readiness deadline in seconds (default 2400)
  REQUEST_TIMEOUT   HTTP request deadline in seconds (default 900)
  PID_HOST          Set to 1 on hosts allowing Docker host PID mode (default 0)
  MIN_FREE_GIB      Minimum free GiB at RUN_ROOT (default 10)
USAGE
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

positive() {
    [[ $2 =~ ^[1-9][0-9]{0,8}$ ]] || die "$1 must be a positive decimal integer."
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    usage
    exit 0
fi
[[ $# -ge 2 ]] || { usage >&2; exit 2; }
action=$1 tag=$2
shift 2
[[ $action == plan || $action == run ]] || die 'Expected plan or run.'
[[ $tag =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,47}$ ]] || die 'Invalid run tag.'
ENGINE_EXTRA_ARGS=()
if (( $# )); then
    [[ $1 == -- ]] || die 'Place additional engine flags after --.'
    shift
    ENGINE_EXTRA_ARGS=("$@")
fi

IMAGE=${IMAGE:-$DEFAULT_IMAGE}
GPU_CSV=${GPUS:-$DEFAULT_GPUS}
PORT=${PORT:-$DEFAULT_PORT}
ISL=${ISL:-10000}
OSL=${OSL:-500}
NUM_PROMPTS=${NUM_PROMPTS:-128}
WARMUP=${WARMUP:-16}
CONCURRENCIES=${CONCURRENCIES:-16,32}
SEED=${SEED:-42}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-2400}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-900}
PID_HOST=${PID_HOST:-0}
MIN_FREE_GIB=${MIN_FREE_GIB:-10}
for key in PORT ISL OSL NUM_PROMPTS STARTUP_TIMEOUT REQUEST_TIMEOUT MIN_FREE_GIB; do
    positive "$key" "${!key}"
done
[[ $WARMUP =~ ^(0|[1-9][0-9]{0,8})$ ]] || die 'WARMUP must be nonnegative.'
[[ $SEED =~ ^(0|[1-9][0-9]{0,8})$ ]] || die 'SEED must be nonnegative.'
[[ $PID_HOST == 0 || $PID_HOST == 1 ]] || die 'PID_HOST must be 0 or 1.'
(( PORT >= 1024 && PORT <= 65535 )) || die 'PORT must be in 1024..65535.'
(( ISL >= 2 && OSL >= 2 )) || die 'ISL and OSL must each be at least 2.'
CONTEXT_LENGTH=${CONTEXT_LENGTH:-$((ISL + OSL + 32))}
positive CONTEXT_LENGTH "$CONTEXT_LENGTH"
(( CONTEXT_LENGTH >= ISL + OSL + 2 )) || die 'CONTEXT_LENGTH must cover ISL + OSL + 2.'
[[ $GPU_CSV =~ ^(0|[1-9][0-9]?)(,(0|[1-9][0-9]?))*$ ]] || die 'GPUS must be a CSV of numeric GPU IDs.'
IFS=, read -ra gpu_ids <<< "$GPU_CSV"
declare -A seen=()
for gpu in "${gpu_ids[@]}"; do
    [[ ! ${seen[$gpu]+set} ]] || die "Duplicate GPU ID $gpu."
    if [[ $action == run && ${ROCR_VISIBLE_DEVICES:-} =~ ^[0-9,]+$ ]]; then
        [[ ,$ROCR_VISIBLE_DEVICES, == *,"$gpu",* ]] || die "GPU $gpu is outside ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES."
    fi
    seen[$gpu]=1
done
GPU_COUNT=${#gpu_ids[@]}

[[ $CONCURRENCIES =~ ^[1-9][0-9]{0,8}(,[1-9][0-9]{0,8})*$ ]] || die 'CONCURRENCIES must be a CSV of positive decimal integers.'
IFS=, read -ra points <<< "$CONCURRENCIES"
seen=()
MAX_CONCURRENCY=0
for point in "${points[@]}"; do
    [[ ! ${seen[$point]+set} ]] || die "Duplicate concurrency $point."
    seen[$point]=1
    (( point <= NUM_PROMPTS )) || die 'NUM_PROMPTS must cover the largest concurrency.'
    if (( point > MAX_CONCURRENCY )); then MAX_CONCURRENCY=$point; fi
done

[[ $action == plan || -n ${MODEL_DIR:-} ]] || die 'Set MODEL_DIR to the local checkpoint directory.'
[[ $action == plan || -n ${RUN_ROOT:-} ]] || die 'Set RUN_ROOT to a dedicated directory on a checked data or NFS mount.'
[[ $action == plan || -n $IMAGE ]] || die 'Set IMAGE to a compatible local Docker image for this engine.'
MODEL_DIR=$(realpath -m -- "${MODEL_DIR:-/path/to/DeepSeek-V4.1-Flash}")
RUN_ROOT=$(realpath -m -- "${RUN_ROOT:-$SUITE_DIR/.runtime}")
[[ $RUN_ROOT != / && $RUN_ROOT != /tmp && $RUN_ROOT != /tmp/* ]] ||
    die 'RUN_ROOT must use a dedicated directory on a data or NFS mount.'
run_dir=$RUN_ROOT/$ENGINE/$tag
[[ $(realpath -m -- "$run_dir") == "$run_dir" ]] || die 'Runtime path components under RUN_ROOT must be real directories.'
[[ "$run_dir$MODEL_DIR$SUITE_DIR" != *:* ]] || die 'Docker mount paths cannot contain a colon.'
name=dsv41-fast-$ENGINE-$tag

ENGINE_CMD=()
ENGINE_ENV=()
ENGINE_DOCKER=()
engine_configure
(( ${#ENGINE_CMD[@]} )) || die 'Engine command is empty.'

server_image=${IMAGE:-'<local-engine-image>'}
if [[ $action == run ]]; then
    [[ -r $MODEL_DIR/config.json && -r $MODEL_DIR/model.safetensors.index.json ]] ||
        die "Checkpoint config or safetensors index is missing in $MODEL_DIR."
    grep -Eq '"model_type"[[:space:]]*:[[:space:]]*"deepseek_v41"' "$MODEL_DIR/config.json" ||
        die 'MODEL_DIR must contain a DeepSeek V4.1 checkpoint.'
    [[ -c /dev/kfd && -d /dev/dri ]] || die 'AMD GPU device nodes are unavailable.'
    [[ ! -e $run_dir ]] || die "Run already exists at $run_dir. Use a new tag."
    port_listeners=$(ss -ltnH "sport = :$PORT")
    [[ -z $port_listeners ]] || die "TCP port $PORT is already in use."
    server_image=$(docker image inspect --format '{{.Id}}' "$IMAGE") || die "Stage image $IMAGE before running."
fi

cidfile=$run_dir/container-id.txt
docker_run=(docker run --rm --pull=never --name "$name" --cidfile "$cidfile"
    --label "dsv41.fast.engine=$ENGINE" --label "dsv41.fast.tag=$tag"
    --network host --ipc host --device /dev/kfd --device /dev/dri
    --group-add video --security-opt seccomp=unconfined
    --ulimit nofile=1048576:1048576 --ulimit memlock=-1
    -v "$SUITE_DIR:/suite:ro" -v "$MODEL_DIR:$MODEL_IN_CONTAINER:ro"
    -v "$run_dir/cache:/cache" -v "$run_dir/tmp:/tmp" -v "$run_dir/home:/root"
    -e "ROCR_VISIBLE_DEVICES=$GPU_CSV"
    -e HOME=/root -e TMPDIR=/tmp -e HF_HOME=/cache/hf
    -e XDG_CACHE_HOME=/cache/xdg -e TRITON_CACHE_DIR=/cache/triton
    -e AITER_JIT_DIR=/cache/aiter -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor
    -e VLLM_CACHE_ROOT=/cache/vllm -e PYTHONDONTWRITEBYTECODE=1
    -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1)
if [[ $PID_HOST == 1 ]]; then docker_run+=(--pid host); fi
for entry in "${ENGINE_ENV[@]}"; do docker_run+=(-e "$entry"); done
docker_run+=("${ENGINE_DOCKER[@]}" --entrypoint "${ENGINE_CMD[0]}" "$server_image" "${ENGINE_CMD[@]:1}")

client=(python3 /suite/common/bench.py --engine "$ENGINE" --port "$PORT"
    --model "$SERVED_MODEL" --model-dir "$MODEL_IN_CONTAINER" --gpu-count "$GPU_COUNT"
    --isl "$ISL" --osl "$OSL" --prompts "$NUM_PROMPTS" --warmup "$WARMUP"
    --seed "$SEED" --timeout "$REQUEST_TIMEOUT" --concurrencies "$CONCURRENCIES")
smoke=("${client[@]}"
    --isl "$(( ISL < 256 ? ISL : 256 ))" --osl "$(( OSL < 8 ? OSL : 8 ))"
    --prompts "$GPU_COUNT" --warmup 0 --concurrencies "$GPU_COUNT")

if [[ $action == plan ]]; then
    print_command "${docker_run[@]}"
    print_command docker exec "$name" "${smoke[@]}"
    print_command docker exec "$name" "${client[@]}"
    exit 0
fi

# Check the nearest existing parent before creating any runtime directories.
mount_path=$RUN_ROOT
while [[ ! -d $mount_path ]]; do mount_path=$(dirname -- "$mount_path"); done
mount_target=$(findmnt -n -o TARGET -T "$mount_path")
[[ $mount_target != / ]] || die "RUN_ROOT is on the system disk ($mount_target)."
available_kib=$(df -Pk -- "$mount_path" | awk 'NR == 2 {print $4}')
(( available_kib >= MIN_FREE_GIB * 1024 * 1024 )) || die "RUN_ROOT needs $MIN_FREE_GIB GiB free."
mkdir -p -- "$RUN_ROOT/$ENGINE"
mkdir -- "$run_dir"
mkdir -- "$run_dir/cache" "$run_dir/tmp" "$run_dir/home"
{
    printf 'started_utc=%s\nengine=%s\nimage=%s\nimage_id=%s\ngpus=%s\nmount=%s\n' \
        "$(date -u +%FT%TZ)" "$ENGINE" "$IMAGE" "$server_image" "$GPU_CSV" "$mount_target"
    printf 'model_dir=%s\n' "$MODEL_DIR"
} > "$run_dir/run.env"
print_command "${docker_run[@]}" > "$run_dir/server-command.txt"
print_command docker exec "$name" "${smoke[@]}" > "$run_dir/smoke-command.txt"
print_command docker exec "$name" "${client[@]}" > "$run_dir/benchmark-command.txt"

server_pid=
container_id=
cleanup() {
    local status=$?
    if [[ -s $cidfile ]]; then
        container_id=$(cat -- "$cidfile")
        if ! timeout 45s docker stop --time 20 "$container_id" >/dev/null 2>&1 &&
            docker inspect --format '{{.State.Running}}' "$container_id" 2>/dev/null | grep -Fxq true; then
            printf 'Decoder is still running (%s).\n' "$container_id" >&2
            kill -TERM "$server_pid" 2>/dev/null || true
            status=1
        fi
    elif [[ -n $server_pid ]]; then
        kill -TERM "$server_pid" 2>/dev/null || true
    fi
    if [[ -n $server_pid ]]; then wait "$server_pid" 2>/dev/null || true; fi
    printf 'exit_code=%s\nfinished_utc=%s\n' "$status" "$(date -u +%FT%TZ)" >> "$run_dir/run.env"
    printf 'Run artifacts: %s\n' "$run_dir" >&2
    trap - EXIT
    exit "$status"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

"${docker_run[@]}" > "$run_dir/server.log" 2>&1 &
server_pid=$!
printf 'Waiting for %s on port %s (%s)\n' "$ENGINE" "$PORT" "$run_dir" >&2

deadline=$((SECONDS + STARTUP_TIMEOUT))
ready=0
base_url=http://127.0.0.1:$PORT
while (( SECONDS < deadline )); do
    kill -0 "$server_pid" 2>/dev/null || die 'Decoder exited during startup. See server.log.'
    if [[ -s $cidfile ]] &&
        curl --noproxy '*' --silent --fail --max-time 3 "$base_url/health" >/dev/null &&
        curl --noproxy '*' --silent --fail --max-time 3 "$base_url/v1/models" > "$run_dir/models.json" &&
        grep -Fq -- "$SERVED_MODEL" "$run_dir/models.json"; then
        container_id=$(cat -- "$cidfile")
        ready=1
        break
    fi
    sleep 5
done
(( ready )) || die "Decoder readiness timed out after $STARTUP_TIMEOUT seconds."

# Readiness and measured requests use the same loopback server.
docker exec "$container_id" "${smoke[@]}" 2> "$run_dir/smoke.log" > "$run_dir/smoke.jsonl"
docker exec "$container_id" "${client[@]}" 2> "$run_dir/client.log" > "$run_dir/benchmark.jsonl"
grep -F '"event": "summary"' "$run_dir/benchmark.jsonl"
