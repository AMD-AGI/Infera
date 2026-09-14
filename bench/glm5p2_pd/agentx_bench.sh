#!/usr/bin/env bash
# Run one AgentX point against an already-running multi-P/D service.
set -euo pipefail
COMPONENT=agentx
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$DIR/../../.." && pwd)"
source "$DIR/lib/common.sh"

load_config "$@"
init_ssh
CONC="${CONC:?set CONC=<one positive integer>}"
DURATION="${DURATION:-$AGENTX_DURATION}"
[[ "$CONC" =~ ^[1-9][0-9]*$ ]] || die "CONC must be one positive integer"
[[ "$DURATION" =~ ^[1-9][0-9]*$ ]] || die "DURATION must be a positive integer"

OUT_DIR="${OUT_DIR:-$DIR/results/$(date -u +%Y%m%dT%H%M%SZ)-agentx-c$CONC}"
[[ "$OUT_DIR" == /* ]] || OUT_DIR="$DIR/$OUT_DIR"
if [[ -d "$OUT_DIR" && -n "$(ls -A "$OUT_DIR")" ]]; then
    die "output directory is not empty: $OUT_DIR"
fi
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
RUN_LOG="$OUT_DIR/runner.log"
export CONC DURATION OUT_DIR RUN_LOG
start_log

router_base_url="$(router_url)"
curl -fsS --max-time 30 "$router_base_url/health" >/dev/null ||
    die "router is not healthy at $router_base_url"

run_token="$(date -u +%Y%m%dT%H%M%S)-$$"
client_container="$(service_container agentx-client)-$run_token"
orphans="$(ssh_exec "$CONTROL_NODE" bash -lc \
    "pgrep -cf '(^|/)aiperf profile ' || true")"
[[ "$orphans" == 0 ]] ||
    die "$orphans orphaned aiperf processes found on $CONTROL_NODE"

AGENTX_CACHE_DIR="${AGENTX_CACHE_DIR:-$DIR/.cache/agentx}"
runtime_dir="$AGENTX_CACHE_DIR/aiperf"
hf_home="$AGENTX_CACHE_DIR/hf"
mkdir -p "$runtime_dir" "$hf_home"

python3 "$DIR/tools/agentx_env.py" \
    --topology "$TOPOLOGY_FILE" \
    --router-url "$router_base_url" \
    --output-dir "$OUT_DIR" \
    --runtime-dir "$runtime_dir" \
    --hf-home "$hf_home" \
    --concurrency "$CONC" \
    --duration "$DURATION" \
    --ssh-options "$SSH_OPTS"

runtime_model="$(awk -F= '$1 == "MODEL" {sub("^[^=]*=", ""); print; exit}' "$OUT_DIR/runtime.env")"
[[ -n "$runtime_model" ]] || die "runtime.env has no live MODEL path"
ssh_exec "$CONTROL_NODE" test -r "$runtime_model/tokenizer_config.json" ||
    die "tokenizer is not readable at $runtime_model on $CONTROL_NODE"

mounts=(-v "$WORKSPACE_ROOT:$WORKSPACE_ROOT")
[[ "$runtime_model" == "$WORKSPACE_ROOT"/* ]] ||
    mounts+=(-v "$runtime_model:$runtime_model:ro")
[[ "$OUT_DIR" == "$WORKSPACE_ROOT"/* ]] || mounts+=(-v "$OUT_DIR:$OUT_DIR")
[[ "$AGENTX_CACHE_DIR" == "$WORKSPACE_ROOT"/* ]] ||
    mounts+=(-v "$AGENTX_CACHE_DIR:$AGENTX_CACHE_DIR")

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
log "C$CONC for ${DURATION}s via $router_base_url"
ssh_exec "$CONTROL_NODE" docker run --rm \
    --name "$client_container" \
    --network host --ipc host --shm-size "${CLIENT_SHM_SIZE:-32g}" \
    "${mounts[@]}" \
    --env-file "$OUT_DIR/runtime.env" \
    "$IMAGE" bash -c '
        set -euo pipefail
        out="$1"
        trap "chown -R $HOST_UID:$HOST_GID \"$out\" \"$AIPERF_RUNTIME_DIR\" \"$HF_HOME\"" EXIT
        source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"
        install_agentic_deps
        resolve_trace_source
        build_replay_cmd "$out"
        run_agentic_replay_and_write_outputs "$out"
    ' _ "$OUT_DIR"
client_started=0
trap - EXIT INT TERM

result="$OUT_DIR/agentx_conc$CONC.json"
[[ -s "$result" ]] || die "AgentX aggregate result is missing: $result"
log "PASS: $result"
