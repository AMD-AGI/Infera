#!/usr/bin/env bash
# Run one AgentX point against the running service, from a client container on
# the control node. The aggregate lands in <current run>/agentx/agentx_conc<N>.json.
# PREPARE_ONLY=1 installs dependencies and saves the replay command without
# contacting the inference service or using GPUs.
# Usage: scripts/agentx.sh [CONC=48] [DURATION=3600] [PREPARE_ONLY=1] [KEY=VALUE ...]
source "$(dirname "$0")/common.sh" "$@"

PREPARE_ONLY="${PREPARE_ONLY:-0}"
if [[ "$PREPARE_ONLY" == 1 ]]; then out="$TMP_DIR/prepared/c$CONC/agentx"
else out="$(current_run)/agentx"; fi
ix="$CACHE_DIR/InferenceX"
cache="$CACHE_DIR/agentx"
mkdir -p "$out/tmp" "$cache"
if [[ ! -d "$ix/.git" ]]; then
    git clone --filter=blob:none "$INFERENCEX_REPO" "$ix"
    git -C "$ix" checkout --detach "$INFERENCEX_REF"
    git -C "$ix" submodule update --init --recursive
fi
[[ "$(git -C "$ix" rev-parse HEAD)" == "$INFERENCEX_REF" ]] || {
    echo "InferenceX cache does not match INFERENCEX_REF: $ix" >&2; exit 1;
}

router="http://$CONTROL_IP:$ROUTER_PORT"
metrics="$router/metrics"
for i in "${!PREFILL_GROUPS[@]}"; do metrics+=",http://$PREFILL_IP:$((PREFILL_PORT + 2 * i))/metrics"; done
metrics+=",http://$DECODE_IP:$DECODE_PORT/metrics"
# InferenceX metadata: the prefill's LMCache CPU tier is DRAM offload.
if (( PREFILL_OFFLOAD_GB > 0 )); then
    offload=$'KV_OFFLOADING=dram\nKV_OFFLOAD_BACKEND=lmcache\n'
    offload+="KV_OFFLOAD_BACKEND_METADATA={\"name\":\"lmcache\",\"version\":\"$LMCACHE_VERSION\"}"
else offload="KV_OFFLOADING=none"; fi
# TMPDIR stays short: AIPerf's Unix socket paths must fit in 107 bytes.
cat > "$out/runtime.env" <<EOF
INFMAX_CONTAINER_WORKSPACE=$ix
INFERENCEX_COMMIT=$INFERENCEX_REF
MODEL=$MODEL
SERVED_MODEL_NAME=$MODEL
MODEL_PREFIX=glm5.2
IMAGE=$IMAGE
FRAMEWORK=atom
PRECISION=fp4
RUNNER_TYPE=mi355x
PREFILL_HARDWARE=mi355x
DECODE_HARDWARE=mi355x
PORT=$ROUTER_PORT
AIPERF_SERVER_URL=$router
AIPERF_SERVER_METRICS_URLS=$metrics
AIPERF_REQUIRED_SERVER_METRIC_PREFIX=atom:
AIPERF_FAILED_REQUEST_THRESHOLD=0.10
AIPERF_WARMUP_REQUESTS_PER_LANE=$WARMUP_PER_LANE
AIPERF_HTTP_TCP_USER_TIMEOUT=900000
CONC=$CONC
DURATION=$DURATION
IS_AGENTIC=1
SCENARIO_TYPE=agentic-coding
IS_MULTINODE=true
DISAGG=true
ENABLE_AGENTX_POWER=0
TP=1
EP_SIZE=1
DP_ATTENTION=false
PREFILL_NUM_WORKERS=${#PREFILL_GROUPS[@]}
PREFILL_TP=1
PREFILL_PP_SIZE=4
PREFILL_EP=1
PREFILL_DP_ATTN=false
DECODE_NUM_WORKERS=1
DECODE_TP=4
DECODE_EP=1
DECODE_DCP_SIZE=4
DECODE_DP_ATTN=false
SPEC_DECODING=mtp
SIMULATE_ACC_LEN=$MTP_AL
$offload
TOTAL_CPU_DRAM_GB=$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo)
RESULT_FILENAME=agentx_conc$CONC
AGENTIC_OUTPUT_DIR=$out
AIPERF_RUNTIME_DIR=$cache/aiperf
AIPERF_DATASET_MMAP_CACHE_DIR=$cache/dataset_mmap
MPLCONFIGDIR=$cache/matplotlib
HF_HOME=$cache/hf
UV_PYTHON_INSTALL_DIR=$cache/python
XDG_CACHE_HOME=$cache/xdg
UV_NO_MODIFY_PATH=1
TMPDIR=/ax-tmp
PYTHONNOUSERSITE=1
EOF

echo "AgentX C$CONC for ${DURATION}s via $router -> $out"
docker run --rm --name "$PREFIX-agentx" --network host --ipc host --shm-size 32g \
    --user "$(id -u):$(id -g)" -v "$KIT_DIR:$KIT_DIR:ro" \
    -v "$TMP_DIR:$TMP_DIR" -v "$CACHE_DIR:$CACHE_DIR" \
    -v "$MODEL:$MODEL:ro" -v "$out/tmp:/ax-tmp" --env-file "$out/runtime.env" "$IMAGE" bash -c '
        set -euo pipefail
        source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"
        install_agentic_deps
        resolve_trace_source
        build_replay_cmd "$1"
        REPLAY_CMD+=" --apply-chat-template"
        printf "%s\n" "$REPLAY_CMD" > "$1/replay-command.txt"
        if [[ "$2" == 1 ]]; then
            echo "AgentX preparation complete; no inference requests sent."
            exit 0
        fi
        run_agentic_replay_and_write_outputs "$1"
    ' _ "$out" "$PREPARE_ONLY" 2>&1 | tee "$out/runner.log"
[[ "$PREPARE_ONLY" != 1 ]] || exit 0
test -s "$out/agentx_conc$CONC.json"
