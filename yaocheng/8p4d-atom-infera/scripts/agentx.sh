#!/usr/bin/env bash
# Run one AgentX point against the running service, from a client container on
# the control node. The aggregate lands in <current run>/agentx/agentx_conc<N>.json.
# Usage: scripts/agentx.sh [CONC=16] [DURATION=3600] [KEY=VALUE ...]
source "$(dirname "$0")/common.sh" "$@"

out="$(current_run)/agentx"
ix="$TMP_DIR/cache/InferenceX"
cache="$TMP_DIR/cache/agentx"
mkdir -p "$out/tmp" "$cache"
if [[ ! -d "$ix/.git" ]]; then
    git clone --filter=blob:none "$INFERENCEX_REPO" "$ix"
    git -C "$ix" checkout --detach "$INFERENCEX_REF"
    git -C "$ix" submodule update --init --recursive
fi

router="http://$CONTROL_IP:$ROUTER_PORT"
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
AIPERF_SERVER_METRICS_URLS=$router/metrics,http://$PREFILL_IP:$PREFILL_PORT/metrics,http://$DECODE_IP:$DECODE_PORT/metrics
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
TP=8
EP_SIZE=1
DP_ATTENTION=true
PREFILL_NUM_WORKERS=1
PREFILL_TP=8
PREFILL_EP=1
PREFILL_DP_ATTN=true
DECODE_NUM_WORKERS=1
DECODE_TP=4
DECODE_EP=1
DECODE_DCP_SIZE=4
DECODE_DP_ATTN=false
SPEC_DECODING=mtp
SIMULATE_ACC_LEN=$MTP_AL
KV_OFFLOADING=none
TOTAL_CPU_DRAM_GB=$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo)
RESULT_FILENAME=agentx_conc$CONC
AGENTIC_OUTPUT_DIR=$out
AIPERF_RUNTIME_DIR=$cache/aiperf
HF_HOME=$cache/hf
UV_PYTHON_INSTALL_DIR=$cache/python
XDG_CACHE_HOME=$cache/xdg
HOME=/ax-tmp
TMPDIR=/ax-tmp
PYTHONNOUSERSITE=1
EOF

echo "AgentX C$CONC for ${DURATION}s via $router -> $out"
docker run --rm --name "$PREFIX-agentx" --network host --ipc host --shm-size 32g \
    --user "$(id -u):$(id -g)" -v "$KIT_DIR:$KIT_DIR" -v "$MODEL:$MODEL:ro" \
    -v "$out/tmp:/ax-tmp" --env-file "$out/runtime.env" "$IMAGE" bash -c '
        set -euo pipefail
        source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"
        install_agentic_deps
        resolve_trace_source
        build_replay_cmd "$1"
        REPLAY_CMD+=" --apply-chat-template"
        run_agentic_replay_and_write_outputs "$1"
    ' _ "$out" 2>&1 | tee "$out/runner.log"
test -s "$out/agentx_conc$CONC.json"
