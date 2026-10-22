#!/usr/bin/env bash
# Purpose: Run one AgentX concurrency point against the live multi-P/D service.
# Usage: ./agentx_bench.sh CONC=N [DURATION=SECONDS] [OUT_DIR=PATH] [KEY=VALUE ...]
# Artifacts: runtime.env, service snapshots, runner log, AIPerf data, aggregate JSON.
# Artifact paths: OUT_DIR defaults to results/<UTC>-agentx-c<CONC>; the primary
#   result is OUT_DIR/agentx_conc<CONC>.json; AGENTX_CACHE_DIR holds client caches.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
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
export INFERENCEX_DIR
: "${CONC:?set CONC to one positive integer}"
DURATION="${DURATION:-$AGENTX_DURATION}"
[[ "$CONC" =~ ^[1-9][0-9]*$ ]] || { echo "CONC must be positive" >&2; exit 2; }
[[ "$DURATION" =~ ^[1-9][0-9]*$ ]] || { echo "DURATION must be positive" >&2; exit 2; }
python3 "$DIR/tools/ensure_inferencex.py" \
    --directory "$INFERENCEX_DIR" --repository "$INFERENCEX_REPOSITORY" \
    --ref "$INFERENCEX_REF"
INFERENCEX_DIR="$(cd "$INFERENCEX_DIR" && pwd)"
export INFERENCEX_DIR

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
    ssh "${ssh_args[@]}" "$node" "$(remote_command "$@")"
}
control_ip="$(python3 "$DIR/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
router_url="http://$control_ip:$ROUTER_PORT"
curl -fsS --max-time 30 "$router_url/health" >/dev/null ||
    { echo "router is not healthy at $router_url" >&2; exit 1; }

OUT_DIR="${OUT_DIR:-$DIR/results/$(date -u +%Y%m%dT%H%M%SZ)-agentx-c$CONC}"
[[ "$OUT_DIR" == /* ]] || OUT_DIR="$DIR/$OUT_DIR"
[[ ! -e "$OUT_DIR" ]] || { echo "output path already exists: $OUT_DIR" >&2; exit 1; }
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
cache="${AGENTX_CACHE_DIR:-$DIR/.cache/agentx}"
mkdir -p "$cache/aiperf" "$cache/hf"

python3 "$DIR/tools/agentx_env.py" \
    --topology "$TOPOLOGY" --router-url "$router_url" --output-dir "$OUT_DIR" \
    --runtime-dir "$cache/aiperf" --hf-home "$cache/hf" \
    --concurrency "$CONC" --duration "$DURATION" \
    --ssh-options "${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10}"
runtime_model="$(awk -F= '$1=="MODEL"{sub("^[^=]*=",""); print; exit}' "$OUT_DIR/runtime.env")"
[[ -n "$runtime_model" ]] || { echo "runtime.env has no MODEL" >&2; exit 1; }
ssh_run "$CONTROL_NODE" test -r "$runtime_model/tokenizer_config.json" ||
    { echo "tokenizer is not readable on $CONTROL_NODE" >&2; exit 1; }
orphans="$(ssh_run "$CONTROL_NODE" bash -lc "pgrep -cf '(^|/)aiperf profile ' || true")"
[[ "$orphans" == 0 ]] ||
    { echo "$orphans orphaned aiperf processes found on $CONTROL_NODE" >&2; exit 1; }

mounts=(-v "$REPO:$REPO")
[[ "$INFERENCEX_DIR" == "$REPO"/* ]] ||
    mounts+=(-v "$INFERENCEX_DIR:$INFERENCEX_DIR:ro")
[[ "$runtime_model" == "$REPO"/* ]] ||
    mounts+=(-v "$runtime_model:$runtime_model:ro")
[[ "$OUT_DIR" == "$REPO"/* ]] || mounts+=(-v "$OUT_DIR:$OUT_DIR")
# SOURCE_MODEL only matters when MODEL is a symlink overlay onto it.
[[ "$SOURCE_MODEL" == "$runtime_model" || "$SOURCE_MODEL" == "$REPO"/* ]] ||
    mounts+=(-v "$SOURCE_MODEL:$SOURCE_MODEL:ro")
[[ "$cache" == "$REPO"/* ]] || mounts+=(-v "$cache:$cache")

name="$CONTAINER_PREFIX-agentx-client-$(date -u +%Y%m%dT%H%M%S)-$$"
printf '%s\n' "$name" > "$OUT_DIR/client.container"
record_exit() {
    local rc=$?
    printf '%s\n' "$rc" > "$OUT_DIR/client.shell-exit-code"
    ssh_run "$CONTROL_NODE" docker inspect --format '{{.State.ExitCode}}' "$name" \
        > "$OUT_DIR/client.container-exit-code" 2>/dev/null || true
}
trap record_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo "running AgentX C$CONC for ${DURATION}s via $router_url"
ssh_run "$CONTROL_NODE" docker run --name "$name" \
    --network host --ipc host --shm-size "${CLIENT_SHM_SIZE:-32g}" \
    "${mounts[@]}" \
    --tmpfs "$WS/tmp/${YIHOU_NODE##*-}/client:rw,exec,size=16g" \
    -e "TMPDIR=$WS/tmp/${YIHOU_NODE##*-}/client" \
    -e "HOME=$WS/cache/${YIHOU_NODE##*-}/client-home" \
    -e "XDG_CACHE_HOME=$WS/cache/${YIHOU_NODE##*-}/client-xdg" \
    -e "PYTHONPYCACHEPREFIX=$WS/cache/${YIHOU_NODE##*-}/client-pycache" \
    -e "UV_PYTHON_INSTALL_DIR=$WS/cache/${YIHOU_NODE##*-}/uv-python" \
    --env-file "$OUT_DIR/runtime.env" \
    "$IMAGE" bash -c '
        set -euo pipefail
        out="$1"
        # Unix socket addresses must fit sockaddr_un; the alias resolves to workspace tmpfs.
        ln -s "$TMPDIR" /yihou-client-tmp
        export TMPDIR=/yihou-client-tmp
        trap "chown -R $HOST_UID:$HOST_GID \"$out\" \"$AIPERF_RUNTIME_DIR\" \"$HF_HOME\"" EXIT
        source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"
        install_agentic_deps
        resolve_trace_source
        build_replay_cmd "$out"
        run_agentic_replay_and_write_outputs "$out"
    ' _ "$OUT_DIR" 2>&1 | tee "$OUT_DIR/runner.log"
trap - INT TERM
result="$OUT_DIR/agentx_conc$CONC.json"
[[ -s "$result" ]] || { echo "AgentX result is missing: $result" >&2; exit 1; }
echo "AgentX passed: $result"
