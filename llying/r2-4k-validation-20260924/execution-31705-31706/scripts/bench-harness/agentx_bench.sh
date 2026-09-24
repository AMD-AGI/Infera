#!/usr/bin/env bash
# Purpose: Run one AgentX concurrency point against the live multi-P/D service.
# Usage: ./agentx_bench.sh CONC=N [DURATION=SECONDS] [OUT_DIR=PATH] [KEY=VALUE ...]
# Artifacts: runtime.env, service snapshots, runner log, AIPerf data, aggregate JSON.
# Artifact paths: OUT_DIR defaults to results/<UTC>-agentx-c<CONC>; the primary
#   result is OUT_DIR/agentx_conc<CONC>.json; AGENTX_CACHE_DIR holds client caches.
set -euo pipefail
R2_HARNESS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$R2_HARNESS_DIR/../.." && pwd)"

for assignment in "$@"; do
    [[ "$assignment" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] ||
        { echo "expected KEY=VALUE, got '$assignment'" >&2; exit 2; }
    export "$assignment"
done
CONFIG="${CONFIG:-$R2_HARNESS_DIR/config.sh}"
[[ -r "$CONFIG" ]] || { echo "config is not readable: $CONFIG" >&2; exit 1; }
set -a
source "$CONFIG"
set +a
REPO="$(cd "$R2_HARNESS_DIR/../.." && pwd)"
python3 "$TRACE_RUNTIME/scripts/require_performance_approval.py"
TOPOLOGY="${TOPOLOGY:-$R2_HARNESS_DIR/topology.tsv}"
INFERENCEX_DIR="${INFERENCEX_DIR:-$R2_HARNESS_DIR/.cache/InferenceX}"
export INFERENCEX_DIR
: "${CONC:?set CONC to one positive integer}"
DURATION="${DURATION:-$AGENTX_DURATION}"
[[ "$CONC" =~ ^[1-9][0-9]*$ ]] || { echo "CONC must be positive" >&2; exit 2; }
[[ "$DURATION" =~ ^[1-9][0-9]*$ ]] || { echo "DURATION must be positive" >&2; exit 2; }
python3 "$R2_HARNESS_DIR/tools/ensure_inferencex.py" \
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
control_ip="$(python3 "$R2_HARNESS_DIR/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
router_url="http://$control_ip:$ROUTER_PORT"
curl -fsS --max-time 30 "$router_url/health" >/dev/null ||
    { echo "router is not healthy at $router_url" >&2; exit 1; }

OUT_DIR="${OUT_DIR:-$R2_HARNESS_DIR/results/$(date -u +%Y%m%dT%H%M%SZ)-agentx-c$CONC}"
[[ "$OUT_DIR" == /* ]] || OUT_DIR="$R2_HARNESS_DIR/$OUT_DIR"
[[ ! -e "$OUT_DIR" ]] || { echo "output path already exists: $OUT_DIR" >&2; exit 1; }
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
cp "$0" "$OUT_DIR/executed-agentx-bench.sh"
cp "$TRACE_RUNTIME/scripts/patch_client.py" "$OUT_DIR/executed-client-patch.py"
cache="${AGENTX_CACHE_DIR:-$R2_HARNESS_DIR/.cache/agentx}"
mkdir -p "$cache/aiperf" "$cache/hf"

python3 "$R2_HARNESS_DIR/tools/agentx_env.py" \
    --topology "$TOPOLOGY" --router-url "$router_url" --output-dir "$OUT_DIR" \
    --runtime-dir "$cache/aiperf-$RUN_ID-c$CONC" --hf-home "$cache/hf" \
    --concurrency "$CONC" --duration "$DURATION" \
    --ssh-options "${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10}"
if [[ -n "${AGENTX_RUNTIME_VALIDATOR:-}" ]]; then
    python3 "$AGENTX_RUNTIME_VALIDATOR" "$OUT_DIR"
fi
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
[[ "$cache" == "$REPO"/* ]] || mounts+=(-v "$cache:$cache")

name="$CONTAINER_PREFIX-agentx-client-$(date -u +%Y%m%dT%H%M%S)-$$"
cleanup() {
    ssh_run "$CONTROL_NODE" docker rm -f "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo "running AgentX C$CONC for ${DURATION}s via $router_url"
ssh_run "$CONTROL_NODE" docker run --rm --name "$name" \
    --network host --ipc host --shm-size "${CLIENT_SHM_SIZE:-32g}" \
    "${mounts[@]}" --env-file "$OUT_DIR/runtime.env" \
    -e "AGENTX_DATASET_DOWNLOAD_OFFLINE=${AGENTX_DATASET_DOWNLOAD_OFFLINE:-0}" \
    "$IMAGE" bash -c '
        set -euo pipefail
        out="$1"
        trap "chown -R $HOST_UID:$HOST_GID \"$out\" \"$AIPERF_RUNTIME_DIR\" \"$HF_HOME\"" EXIT
        source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"
        install_agentic_deps
        "$AIPERF_PYTHON" "$2"
        if [[ -n "$3" ]]; then "$AIPERF_PYTHON" "$3"; fi
        export PYTHONPATH="/tmp/aus-client-overlay${PYTHONPATH:+:$PYTHONPATH}"
        if [[ "$AGENTX_DATASET_DOWNLOAD_OFFLINE" == 1 ]]; then
            HF_HUB_OFFLINE=1 resolve_trace_source
        else
            resolve_trace_source
        fi
        build_replay_cmd "$out"
        run_agentic_replay_and_write_outputs "$out"
    ' _ "$OUT_DIR" "$TRACE_RUNTIME/scripts/patch_client.py" "${AGENTX_DATASET_PINNER:-}" 2>&1 | tee "$OUT_DIR/runner.log"
trap - EXIT INT TERM
result="$OUT_DIR/agentx_conc$CONC.json"
[[ -s "$result" ]] || { echo "AgentX result is missing: $result" >&2; exit 1; }
echo "AgentX passed: $result"
