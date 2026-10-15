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

# yihou v0521dbg: NODE_RW_ROOT set = the node mounts /shared_nfs read-only. The client then
# writes its output, aiperf runtime and HF cache under NODE_RW_ROOT (runtime.node.env points
# there) and the output is copied back to OUT_DIR afterwards. $REPO stays the read side.
node_rw="${NODE_RW_ROOT:-}"
client_rw="${node_rw:-$WS}"
client_out="$OUT_DIR"
env_file="$OUT_DIR/runtime.env"
if [[ -n "$node_rw" ]]; then
    client_out="$node_rw/runs/${YIHOU_ROUND:?}/$(basename "$OUT_DIR")"
    node_cache="$node_rw/cache/${YIHOU_NODE##*-}/agentx"
    sed -e "s#^AGENTIC_OUTPUT_DIR=.*#AGENTIC_OUTPUT_DIR=$client_out#" \
        -e "s#^AIPERF_RUNTIME_DIR=.*#AIPERF_RUNTIME_DIR=$node_cache/aiperf#" \
        -e "s#^HF_HOME=.*#HF_HOME=$node_cache/hf#" \
        "$OUT_DIR/runtime.env" > "$OUT_DIR/runtime.node.env"
    env_file="$OUT_DIR/runtime.node.env"
    ssh_run "$CONTROL_NODE" mkdir -p "$client_out" "$node_cache/aiperf" "$node_cache/hf"
fi
mounts=(-v "$REPO:$REPO${node_rw:+:ro}")
[[ -z "$node_rw" ]] || mounts+=(-v "$node_rw:$node_rw")
[[ "$INFERENCEX_DIR" == "$REPO"/* ]] ||
    mounts+=(-v "$INFERENCEX_DIR:$INFERENCEX_DIR:ro")
[[ "$runtime_model" == "$REPO"/* ]] ||
    mounts+=(-v "$runtime_model:$runtime_model:ro")
[[ "$OUT_DIR" == "$REPO"/* ]] || mounts+=(-v "$OUT_DIR:$OUT_DIR")
# SOURCE_MODEL only matters when MODEL is a symlink overlay onto it.
[[ "$SOURCE_MODEL" == "$runtime_model" || "$SOURCE_MODEL" == "$REPO"/* ]] ||
    mounts+=(-v "$SOURCE_MODEL:$SOURCE_MODEL:ro")
[[ -n "$node_rw" || "$cache" == "$REPO"/* ]] || mounts+=(-v "$cache:$cache")

name="$CONTAINER_PREFIX-agentx-client-$(date -u +%Y%m%dT%H%M%S)-$$"
printf '%s\n' "$name" > "$OUT_DIR/client.container" || true
copied_back=0
copy_back() {
    # NODE_RW_ROOT: copy the node-local client output back (base64: the spur shim mangles
    # binary stdout). Runs on success and, via record_exit, on failure too.
    [[ -n "$node_rw" && "$copied_back" == 0 ]] || return 0
    copied_back=1
    # server_metrics_export.json (~3.5 GB per hour-long run) stays on the node: the shared
    # /home volume filled up on 2026-10-13, and agentx_conc*.json already carries its summary.
    # The volume refuses writes while it is momentarily full, so retry for up to two hours.
    local attempt
    for attempt in $(seq 1 120); do
        ssh_run "$CONTROL_NODE" bash -c "tar -C $(printf '%q' "$client_out") --exclude=./aiperf_artifacts/server_metrics_export.json -cf - . | base64 -w0" |
            base64 -d | tar -C "$OUT_DIR" -xf - && return 0
        echo "copy-back of $client_out from $CONTROL_NODE failed (attempt $attempt), retrying in 60 s" >&2
        sleep 60
    done
    echo "copy-back of $client_out from $CONTROL_NODE failed; the output stays on the node" >&2
}
record_exit() {
    local rc=$?
    copy_back || true
    printf '%s\n' "$rc" > "$OUT_DIR/client.shell-exit-code"
    ssh_run "$CONTROL_NODE" docker inspect --format '{{.State.ExitCode}}' "$name" \
        > "$OUT_DIR/client.container-exit-code" 2>/dev/null || true
}
trap record_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo "running AgentX C$CONC for ${DURATION}s via $router_url"
set +e
ssh_run "$CONTROL_NODE" docker run --name "$name" \
    --network host --ipc host --shm-size "${CLIENT_SHM_SIZE:-32g}" \
    "${mounts[@]}" \
    --tmpfs "$client_rw/tmp/${YIHOU_NODE##*-}/client:rw,exec,size=16g" \
    -e "TMPDIR=$client_rw/tmp/${YIHOU_NODE##*-}/client" \
    -e "HOME=$client_rw/cache/${YIHOU_NODE##*-}/client-home" \
    -e "XDG_CACHE_HOME=$client_rw/cache/${YIHOU_NODE##*-}/client-xdg" \
    -e "PYTHONPYCACHEPREFIX=$client_rw/cache/${YIHOU_NODE##*-}/client-pycache" \
    -e "UV_PYTHON_INSTALL_DIR=$client_rw/cache/${YIHOU_NODE##*-}/uv-python" \
    --env-file "$env_file" \
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
    ' _ "$client_out" 2>&1 | tee --output-error=warn "$OUT_DIR/runner.log"
client_rc=${PIPESTATUS[0]}
set -e
# A failed write to runner.log (full /home) must not end the run: only the client's status counts.
(( client_rc == 0 )) || exit "$client_rc"
trap - INT TERM
copy_back
result="$OUT_DIR/agentx_conc$CONC.json"
[[ -s "$result" ]] || { echo "AgentX result is missing: $result" >&2; exit 1; }
echo "AgentX passed: $result"
