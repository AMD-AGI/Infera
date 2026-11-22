#!/usr/bin/env bash
# One InferenceX AgentX point through InferenceX's own client (inferencex-e2e/benchmarks/srt_agentic.sh: agentx-harness,
# AIPerf replay of semianalysisai/cc-traces-weka-062126, aggregation and validation), run in a container on this node.
#   agentx.sh OUT_DIR
# Env: CONC, FAST (0 = formal: DURATION 3600 s and WARMUP_PER_LANE 10 warmup requests per lane, as InferenceX's
#      srt-slurm agentx.sh; 1 = agentx-fast, debugging only), URL (router), METRICS_URLS, SIM_ACCEPT_LEN (golden AL the
#      server runs with; metadata), KV_OFFLOADING / TOTAL_CPU_DRAM_GB (metadata), CLIENT_IMAGE.
#      MODE=agg: AGG_GPUS, DP_ATTENTION.  MODE=pd: PREFILL_GPUS / DECODE_GPUS per worker (1P1D), *_DP_ATTN.
#      InferenceX takes the tok/s/GPU denominator from the PREFILL_* / DECODE_* fields only when IS_MULTINODE=true.
set -uo pipefail
source "$(dirname "$0")/env.sh"
out=${1:?out dir}
CONC=${CONC:?CONC} FAST=${FAST:-0}
mkdir -p "$out" "$HFC/hub" && chmod 777 "$out"
cat > "$out/runtime.env" <<EOF
INFMAX_CONTAINER_WORKSPACE=$IX/inferencex-e2e
MODEL=$MODEL
TOKENIZER=$MODEL
SERVED_MODEL_NAME=$SERVED
MODEL_PREFIX=dsv41flash
FRAMEWORK=vllm
PRECISION=fp4
IMAGE=${CLIENT_IMAGE:-$BASE_IMAGE}
RUNNER_TYPE=mi355x
CONC=$CONC
DURATION=${DURATION:-3600}
AIPERF_EXPERIMENTAL_FAST=$FAST
AIPERF_WARMUP_REQUESTS_PER_LANE=${WARMUP_PER_LANE:-10}
RESULT_DIR=$out/result
RESULT_FILENAME=agentx_conc$CONC
AGENTIC_OUTPUT_DIR=$out
EVAL_ONLY=false
IS_AGENTIC=1
SCENARIO_TYPE=agentic-coding
ENABLE_AGENTX_POWER=0
REQUIRE_POWER=0
$(if [ "${MODE:-pd}" = agg ]; then
    printf 'IS_MULTINODE=false\nDISAGG=false\nTP=%s\nEP_SIZE=1\nDP_ATTENTION=%s\n' "${AGG_GPUS:-2}" "${DP_ATTENTION:-true}"
else
    printf 'IS_MULTINODE=true\nDISAGG=true\nPREFILL_NUM_WORKERS=1\nPREFILL_TP=%s\nPREFILL_EP=1\nPREFILL_DP_ATTN=%s\n' "${PREFILL_GPUS:-2}" "${PREFILL_DP_ATTN:-true}"
    printf 'DECODE_NUM_WORKERS=1\nDECODE_TP=%s\nDECODE_EP=1\nDECODE_DP_ATTN=%s\nPREFILL_HARDWARE=mi355x\nDECODE_HARDWARE=mi355x\n' "${DECODE_GPUS:-2}" "${DECODE_DP_ATTN:-true}"
    echo "KV_P2P_TRANSFER=mooncake"
fi)
SPEC_DECODING=mtp
$( [ -n "${SIM_ACCEPT_LEN:-}" ] && echo "SIMULATE_ACC_LEN=$SIM_ACCEPT_LEN" )
KV_OFFLOADING=${KV_OFFLOADING:-none}
$( [ "${KV_OFFLOADING:-none}" != none ] && printf 'KV_OFFLOAD_BACKEND=%s\nKV_OFFLOAD_BACKEND_METADATA={"name":"%s"}\n' "${KV_OFFLOAD_BACKEND:-umbp}" "${KV_OFFLOAD_BACKEND:-umbp}" )
TOTAL_CPU_DRAM_GB=${TOTAL_CPU_DRAM_GB:-$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo)}
AIPERF_SERVER_URL=${URL:-http://127.0.0.1:$ROUTER_PORT}
AIPERF_SERVER_METRICS_URLS=${METRICS_URLS:?METRICS_URLS}
AIPERF_REQUIRED_SERVER_METRIC_PREFIX=vllm:
AIPERF_FAILED_REQUEST_THRESHOLD=0.10
AIPERF_LIVE_FAILED_REQUEST_THRESHOLD=0.10
AIPERF_TRACE_IDLE_GAP_CAP_SECONDS=300
AGENTIC_WARMUP_GRACE_PERIOD=1800
AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS=3600
AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING=0
AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=false
AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true
AIPERF_HTTP_TCP_USER_TIMEOUT=900000
AIPERF_DRAIN_TIMEOUT_SECONDS=1800
AIPERF_RUNTIME_DIR=$NODE_CACHE/agentx-runtime
HF_HOME=$HFC
HF_HUB_CACHE=$HFC/hub
MPLCONFIGDIR=$NODE_CACHE/agentx-mpl
XDG_CACHE_HOME=$NODE_CACHE/agentx-xdg
PYTHONFAULTHANDLER=1
EOF
name="${CTR_PREFIX}agentx-c$CONC"
docker rm -f "$name" >/dev/null 2>&1
docker run --rm --name "$name" --network host --ipc host --shm-size 32g \
    -v /perf_apps:/perf_apps -v "$NODE_CACHE:$NODE_CACHE" --env-file "$out/runtime.env" --entrypoint bash \
    "${CLIENT_IMAGE:-$BASE_IMAGE}" -c 'cd "$INFMAX_CONTAINER_WORKSPACE" && bash benchmarks/srt_agentic.sh' > "$out/runner.log" 2>&1
rc=$?
echo "agentx rc=$rc" >> "$out/runner.log"
exit $rc
