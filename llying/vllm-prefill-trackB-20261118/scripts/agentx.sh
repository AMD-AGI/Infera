#!/usr/bin/env bash
# One InferenceX AgentX point (agentic-coding trace replay, AIPerf) against the router, the same harness as the sglang
# kit (llying/dp-graph-variant-fix-20261103/kit/engine/agentx/agentx.sh, InferenceX 918524ff): only the server metric
# prefix (vllm:) and the framework / topology labels differ.
#   agentx.sh RUN_ID
# Env: CONC (104), DURATION (3600), WARMUP_PER_LANE (10), N_GPUS (8: tok/s/GPU denominator is set by the replay from
#      PREFILL_NUM_WORKERS x PREFILL_TP + DECODE_TP), PREFILL_TP (4), DECODE_TP (4), SIM_ACCEPT_LEN (2.99).
set -euo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-trackB-20261118}
run=${1:?run id}
RUN_DIR=$R/runs/$run
# PREP_ONLY=1: clone InferenceX, install the replay dependencies and fetch the trace into the node's cache, no replay.
if [[ "${PREP_ONLY:-0}" == 1 ]]; then mkdir -p "$RUN_DIR"; echo http://127.0.0.1:1 > "$RUN_DIR/router_url"; fi
ROUTER_URL=$(cat "$RUN_DIR/router_url")
IMAGE=${IMAGE:-infera-vllm:v0340-0c5e5e89}
MODEL=${MODEL:-/mnt/m2m_nobackup/liyingli_vpd/models/GLM-5.2-MXFP4}
WORK=/mnt/m2m_nobackup/liyingli_vpd/work
IX=$WORK/cache/InferenceX
INFERENCEX_REF=918524ff94045b3f091115f1051c22a8588edf2b
CONC=${CONC:-104} DURATION=${DURATION:-3600} WARMUP_PER_LANE=${WARMUP_PER_LANE:-10}
KIT=$R/kit
if [[ ! -d "$IX/.git" ]]; then
    git clone -q --filter=blob:none https://github.com/SemiAnalysisAI/InferenceX.git "$IX"
    git -C "$IX" checkout -q --detach "$INFERENCEX_REF"
    git -C "$IX" submodule update -q --init --recursive
fi
[[ "$(git -C "$IX" rev-parse HEAD)" == "$INFERENCEX_REF" ]] || { echo "InferenceX not at $INFERENCEX_REF"; exit 1; }
out=$RUN_DIR/agentx-c$CONC
cache=$WORK/cache/agentx
mkdir -p "$out/tmp" "$cache"
workers=$(curl -sf -m 10 "$ROUTER_URL/v1/workers" || echo '{"workers": []}')
metrics="$ROUTER_URL/metrics"
for url in $(python3 -c 'import json, sys; print(*sorted({w["url"] for w in json.load(sys.stdin)["workers"]}))' <<< "$workers"); do
    metrics+=",$url/metrics"
done
cat > "$out/runtime.env" <<EOF
INFMAX_CONTAINER_WORKSPACE=$IX
INFERENCEX_COMMIT=$INFERENCEX_REF
MODEL=$MODEL
SERVED_MODEL_NAME=glm5.2-mxfp4
MODEL_PREFIX=glm5.2
IMAGE=$IMAGE
FRAMEWORK=vllm
PRECISION=fp4
RUNNER_TYPE=mi355x
PREFILL_HARDWARE=mi355x
DECODE_HARDWARE=mi355x
PORT=${ROUTER_URL##*:}
AIPERF_SERVER_URL=$ROUTER_URL
AIPERF_SERVER_METRICS_URLS=$metrics
AIPERF_REQUIRED_SERVER_METRIC_PREFIX=vllm:
AIPERF_FAILED_REQUEST_THRESHOLD=0.10
AIPERF_WARMUP_REQUESTS_PER_LANE=$WARMUP_PER_LANE
AIPERF_HTTP_TCP_USER_TIMEOUT=900000
AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true
CONC=$CONC
DURATION=$DURATION
IS_AGENTIC=1
SCENARIO_TYPE=agentic-coding
IS_MULTINODE=true
DISAGG=true
ENABLE_AGENTX_POWER=0
PREFILL_NUM_WORKERS=1
PREFILL_TP=${PREFILL_TP:-4}
PREFILL_EP=1
PREFILL_DP_ATTN=false
DECODE_NUM_WORKERS=1
DECODE_TP=${DECODE_TP:-4}
DECODE_EP=1
DECODE_DP_ATTN=true
SPEC_DECODING=mtp
SIMULATE_ACC_LEN=${SIM_ACCEPT_LEN:-2.99}
KV_OFFLOADING=${KV_OFFLOADING:-none}
$( [[ "${KV_OFFLOADING:-none}" == dram ]] && printf 'KV_OFFLOAD_BACKEND=%s\nKV_OFFLOAD_BACKEND_METADATA={"name":"%s"}\n' "${KV_OFFLOAD_BACKEND:-vllm-native}" "${KV_OFFLOAD_BACKEND:-vllm-native}" )
KV_P2P_TRANSFER=mooncake
TOTAL_CPU_DRAM_GB=$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo)
RESULT_FILENAME=agentx_conc$CONC
AGENTIC_OUTPUT_DIR=$out
AIPERF_RUNTIME_DIR=$cache/aiperf
AIPERF_DATASET_MMAP_CACHE_DIR=$cache/dataset_mmap
MPLCONFIGDIR=$cache/matplotlib
HF_HOME=$cache/hf
UV_PYTHON_INSTALL_DIR=$cache/python
UV_CONSTRAINT=$KIT/constraints.txt
XDG_CACHE_HOME=$cache/xdg
XDG_CONFIG_HOME=$cache/xdg-config
UV_NO_MODIFY_PATH=1
TMPDIR=/ax-tmp
PYTHONNOUSERSITE=1
EOF
echo "AgentX C$CONC ${DURATION}s via $ROUTER_URL -> $out ($(date -u +%T))"
docker rm -f vpd-agentx > /dev/null 2>&1 || true
docker run --name vpd-agentx --network host --ipc host --shm-size 32g --entrypoint bash \
    --user "$(id -u):$(id -g)" -v "$KIT:$KIT:ro" -v "$WORK:$WORK" -v "$R:$R" \
    -v "$MODEL:$MODEL:ro" -v "$out/tmp:/ax-tmp" --env-file "$out/runtime.env" "$IMAGE" \
    -c '
        set -euo pipefail
        source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"
        install_agentic_deps
        resolve_trace_source
        [[ "$2" == 1 ]] && { echo "prep only: deps and trace ready"; exit 0; }
        build_replay_cmd "$1"
        printf "%s\n" "$REPLAY_CMD" > "$1/replay-command.txt"
        run_agentic_replay_and_write_outputs "$1"
    ' _ "$out" "${PREP_ONLY:-0}" 2>&1 | tee "$out/runner.log" | tail -40
docker rm -f vpd-agentx > /dev/null 2>&1 || true
