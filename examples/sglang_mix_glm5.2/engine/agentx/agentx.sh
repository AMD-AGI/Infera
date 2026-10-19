#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# One InferenceX AgentX (agentic-coding trace replay) point against the running
# service, from a client container in the same image, at concurrency CONC.
# Output: <run>/agentx/agentx_conc<CONC>.json plus the AIPerf exports,
# runtime.env and the replay command.
#
# The reference numbers used DURATION=3600, WARMUP_PER_LANE=10 and an engine
# started with SIM_ACCEPT_LEN=3.61.
#
# Usage: bash engine/agentx/agentx.sh      (DURATION=3600 WARMUP_PER_LANE=10)
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../common.sh
source "$KIT/engine/common.sh"
RUN_DIR="$(run_dir)"
# shellcheck source=/dev/null
source "$RUN_DIR/engine-config.env"

INFERENCEX_REPO="${INFERENCEX_REPO:-https://github.com/SemiAnalysisAI/InferenceX.git}"
INFERENCEX_REF="${INFERENCEX_REF:-918524ff94045b3f091115f1051c22a8588edf2b}"
DURATION="${DURATION:-3600}"
WARMUP_PER_LANE="${WARMUP_PER_LANE:-10}"

out="$RUN_DIR/agentx"
ix="$WORK_DIR/cache/InferenceX"
cache="$WORK_DIR/cache/agentx"
mkdir -p "$out/tmp" "$cache"
if [[ ! -d "$ix/.git" ]]; then
    git clone --filter=blob:none "$INFERENCEX_REPO" "$ix"
    git -C "$ix" checkout --detach "$INFERENCEX_REF"
    git -C "$ix" submodule update --init --recursive
fi
[[ "$(git -C "$ix" rev-parse HEAD)" == "$INFERENCEX_REF" ]] ||
    die "InferenceX checkout $ix is not at $INFERENCEX_REF"

# TMPDIR stays short: AIPerf's Unix socket paths must fit in 107 bytes.
cat >"$out/runtime.env" <<EOF
INFMAX_CONTAINER_WORKSPACE=$ix
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
AIPERF_SERVER_METRICS_URLS=$ROUTER_URL/metrics,$ENGINE_URL/metrics
AIPERF_REQUIRED_SERVER_METRIC_PREFIX=sglang:
AIPERF_FAILED_REQUEST_THRESHOLD=0.10
AIPERF_WARMUP_REQUESTS_PER_LANE=$WARMUP_PER_LANE
AIPERF_HTTP_TCP_USER_TIMEOUT=900000
AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true
CONC=$CONC
DURATION=$DURATION
IS_AGENTIC=1
SCENARIO_TYPE=agentic-coding
IS_MULTINODE=false
DISAGG=false
ENABLE_AGENTX_POWER=0
TP=4
EP_SIZE=1
DP_ATTENTION=true
SPEC_DECODING=mtp
SIMULATE_ACC_LEN=$SIM_ACCEPT_LEN
KV_OFFLOADING=dram
KV_OFFLOAD_BACKEND=hicache
KV_OFFLOAD_BACKEND_METADATA={"name":"hicache"}
KV_P2P_TRANSFER=none
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
