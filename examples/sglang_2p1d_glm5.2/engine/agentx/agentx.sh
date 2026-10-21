#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# One InferenceX AgentX (agentic-coding trace replay) point against the running
# service, from a client container in the same image on the prefill node, at
# concurrency CONC. Throughput is per GPU over all of the deployment's GPUs
# (PREFILL_WORKERS x 4 prefill + 8 decode). Output:
# <run>/agentx/agentx_conc<CONC>.json plus the AIPerf exports, runtime.env and
# the replay command.
#
# The reference numbers used DURATION=3600, WARMUP_PER_LANE=10 and a
# deployment started with SIM_ACCEPT_LEN=3.61, freshly for every point.
#
# Usage: bash engine/agentx/agentx.sh      (DURATION=3600 WARMUP_PER_LANE=10)
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../common.sh
source "$KIT/engine/common.sh"
RUN_DIR="$(run_dir)"
# shellcheck source=/dev/null
source "$RUN_DIR/engine-config.env"
inferencex_checkout

DURATION="${DURATION:-3600}"
WARMUP_PER_LANE="${WARMUP_PER_LANE:-10}"
out="$RUN_DIR/agentx"
cache="$WORK_DIR/cache/agentx"
mkdir -p "$out/tmp" "$cache"

# Server metrics from the router and every worker it lists, decode node included.
workers="$(curl -sf -m 10 "$ROUTER_URL/v1/workers")" || die "cannot list workers: $ROUTER_URL/v1/workers"
metrics="$ROUTER_URL/metrics"
for url in $(python3 -c 'import json, sys; print(*(w["url"] for w in json.load(sys.stdin)["workers"]))' <<<"$workers"); do
    metrics+=",$url/metrics"
done

# TMPDIR stays short: AIPerf's Unix socket paths must fit in 107 bytes.
cat >"$out/runtime.env" <<EOF
INFMAX_CONTAINER_WORKSPACE=$IX
INFERENCEX_COMMIT=$INFERENCEX_REF
MODEL=$MODEL
SERVED_MODEL_NAME=$SERVED_MODEL
MODEL_PREFIX=glm5.2
IMAGE=$IMAGE
FRAMEWORK=sglang
PRECISION=fp4
RUNNER_TYPE=mi355x
PREFILL_HARDWARE=mi355x
DECODE_HARDWARE=mi355x
PORT=$ROUTER_PORT
AIPERF_SERVER_URL=$ROUTER_URL
AIPERF_SERVER_METRICS_URLS=$metrics
AIPERF_REQUIRED_SERVER_METRIC_PREFIX=sglang:
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
PREFILL_NUM_WORKERS=$PREFILL_WORKERS
PREFILL_TP=4
PREFILL_EP=1
PREFILL_DP_ATTN=true
DECODE_NUM_WORKERS=1
DECODE_TP=8
DECODE_EP=1
DECODE_DP_ATTN=true
SPEC_DECODING=mtp
SIMULATE_ACC_LEN=$SIM_ACCEPT_LEN
KV_OFFLOADING=dram
KV_OFFLOAD_BACKEND=hicache
KV_OFFLOAD_BACKEND_METADATA={"name":"hicache"}
KV_P2P_TRANSFER=mooncake
TOTAL_CPU_DRAM_GB=$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo)
RESULT_FILENAME=agentx_conc$CONC
AGENTIC_OUTPUT_DIR=$out
AIPERF_RUNTIME_DIR=$cache/aiperf
AIPERF_DATASET_MMAP_CACHE_DIR=$cache/dataset_mmap
MPLCONFIGDIR=$cache/matplotlib
HF_HOME=$cache/hf
UV_PYTHON_INSTALL_DIR=$cache/python
UV_CONSTRAINT=$KIT/engine/agentx/constraints.txt
XDG_CACHE_HOME=$cache/xdg
XDG_CONFIG_HOME=$cache/xdg-config
UV_NO_MODIFY_PATH=1
TMPDIR=/ax-tmp
PYTHONNOUSERSITE=1
EOF

echo "AgentX C$CONC, ${DURATION}s via $ROUTER_URL -> $out"
# shellcheck disable=SC2016  # expanded inside the container
docker run --rm --name "$CTR-agentx" --network host --ipc host --shm-size 32g \
    --user "$(id -u):$(id -g)" -v "$KIT:$KIT:ro" -v "$WORK_DIR:$WORK_DIR" \
    -v "$MODEL:$MODEL:ro" -v "$out/tmp:/ax-tmp" --env-file "$out/runtime.env" "$IMAGE" \
    bash -c '
        set -euo pipefail
        source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"
        install_agentic_deps
        resolve_trace_source
        build_replay_cmd "$1"
        printf "%s\n" "$REPLAY_CMD" >"$1/replay-command.txt"
        run_agentic_replay_and_write_outputs "$1"
    ' _ "$out" 2>&1 | tee "$out/runner.log"
