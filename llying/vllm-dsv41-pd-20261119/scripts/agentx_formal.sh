#!/usr/bin/env bash
# One InferenceX AgentX point against the infera router on this node, through InferenceX's own client
# (inferencex-e2e/benchmarks/srt_agentic.sh -> infx.bench agentic: agentx-harness in a uv venv, AIPerf replay of
# semianalysisai/cc-traces-weka-062126, aggregation and validation). FAST=1 is the workflow's agentx-fast
# (AIPERF_EXPERIMENTAL_FAST=1: one warmup request per lane and a 1200 s profile).
#   agentx_formal.sh OUT_DIR   (agentx_run.sh plus the formal warmup: WARMUP_PER_LANE, default 10, as InferenceX's
#   srt-slurm agentx.sh sets AIPERF_WARMUP_REQUESTS_PER_LANE; use FAST=0 DURATION=3600 for a scored point)
# Env: CONC (32), FAST (1), DURATION (3600 when FAST=0), MODE (pd | agg), METRICS_URLS (router + engines),
#      SIM_ACCEPT_LEN (unset: real acceptance). pd: PREFILL_WORKERS, PREFILL_GPUS (per worker) / DECODE_GPUS, PREFILL_DP_ATTN / DECODE_DP_ATTN;
#      InferenceX takes the tok/s/GPU denominator from the PREFILL_* / DECODE_* fields only when IS_MULTINODE=true.
#      agg: AGG_TP (2), the one worker's GPUs.
set -uo pipefail
source /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/env.sh
out=${1:?out dir}
IX=/perf_apps/liyingli/bench_agentx/InferenceX
CONC=${CONC:-32} FAST=${FAST:-1}
HFC=$W/hf-cache
mkdir -p "$out" "$HFC/hub" && chmod 777 "$out"
ds=datasets--semianalysisai--cc-traces-weka-062126
[ -d "$HFC/hub/$ds" ] || cp -a "/perf_apps/data/models/$ds" "$HFC/hub/"
chmod -R a+rwX "$HFC" 2>/dev/null
cat > "$out/runtime.env" <<EOF
INFMAX_CONTAINER_WORKSPACE=$IX/inferencex-e2e
MODEL=$MODEL
TOKENIZER=$MODEL
SERVED_MODEL_NAME=$SERVED
MODEL_PREFIX=dsv41flash
FRAMEWORK=vllm
PRECISION=fp4
IMAGE=$IMAGE
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
    printf 'IS_MULTINODE=false\nDISAGG=false\nTP=%s\nEP_SIZE=1\nDP_ATTENTION=false\n' "${AGG_TP:-2}"
else
    printf 'IS_MULTINODE=true\nDISAGG=true\nPREFILL_NUM_WORKERS=%s\nPREFILL_TP=%s\nPREFILL_EP=1\nPREFILL_DP_ATTN=%s\n' "${PREFILL_WORKERS:-1}" "${PREFILL_GPUS:-2}" "${PREFILL_DP_ATTN:-true}"
    printf 'DECODE_NUM_WORKERS=1\nDECODE_TP=%s\nDECODE_EP=1\nDECODE_DP_ATTN=%s\nPREFILL_HARDWARE=mi355x\nDECODE_HARDWARE=mi355x\n' "${DECODE_GPUS:-4}" "${DECODE_DP_ATTN:-true}"
fi)
SPEC_DECODING=mtp
$( [ -n "${SIM_ACCEPT_LEN:-}" ] && echo "SIMULATE_ACC_LEN=$SIM_ACCEPT_LEN" )
KV_OFFLOADING=none
KV_P2P_TRANSFER=mooncake
TOTAL_CPU_DRAM_GB=$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo)
AIPERF_SERVER_URL=http://127.0.0.1:$ROUTER_PORT
AIPERF_SERVER_METRICS_URLS=${METRICS_URLS:-http://127.0.0.1:$ROUTER_PORT/metrics,http://127.0.0.1:30000/metrics,http://127.0.0.1:30001/metrics}
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
docker rm -f "${CTR_PREFIX}agentx" >/dev/null 2>&1
docker run --rm --name "${CTR_PREFIX}agentx" --network host --ipc host --shm-size 32g \
    -v /perf_apps:/perf_apps -v "$NODE_CACHE:$NODE_CACHE" --env-file "$out/runtime.env" --entrypoint bash "$IMAGE" \
    -c 'cd "$INFMAX_CONTAINER_WORKSPACE" && bash benchmarks/srt_agentic.sh' > "$out/runner.log" 2>&1
rc=$?
echo "agentx rc=$rc" >> "$out/runner.log"
exit $rc
