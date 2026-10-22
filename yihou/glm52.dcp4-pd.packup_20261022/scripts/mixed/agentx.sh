#!/bin/bash
# agentx.sh TAG PORT CONC DURATION [DCP_SIZE] -> mixrun/agentx/<TAG>/ ; AgentX (InferenceX @918524ff) client against a
# plain sglang endpoint (mixed, non-PD). Scoring: InferenceX per_gpu = total / TP(4) (IS_MULTINODE=false).
# DURATION < 900 => InferenceX adds --unsafe-override (smoke only, not a valid submission).
set -euo pipefail
# install_agentic_deps rm -rf's $AIPERF_RUNTIME_DIR/venv: one runtime dir per TAG so parallel clients never share it.
TAG=$1; PORT=$2; CONC=$3; DUR=$4; DCP=${5:-1}
WS=/shared_nfs/yihou_glm52/dcp_pd_20261021; M=$WS/mixrun
IX=/shared_nfs/yihou_glm52/agentx44_20261009/cache/InferenceX
IMG=lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020
OUT=$M/agentx/$TAG; [[ ! -e $OUT ]] || { echo "exists: $OUT"; exit 1; }
CA=$WS/cache/${NODE:-054}/agentx; mkdir -p $OUT $CA/aiperf.$TAG $WS/cache/agentx-hf $WS/cache/${NODE:-054}/client-home $WS/tmp/${NODE:-054}
cat > $OUT/runtime.env <<EOT
AGENTIC_OUTPUT_DIR=$OUT
AIPERF_FAILED_REQUEST_THRESHOLD=0.10
AIPERF_HTTP_TCP_USER_TIMEOUT=900000
AIPERF_LIVE_FAILED_REQUEST_THRESHOLD=0.10
AIPERF_REQUIRED_SERVER_METRIC_PREFIX=sglang:
AIPERF_RUNTIME_DIR=$CA/aiperf.$TAG
AIPERF_SERVER_URL=http://127.0.0.1:$PORT
AIPERF_WARMUP_REQUESTS_PER_LANE=${WARMUP:-10}
CONC=$CONC
DCP_SIZE=$DCP
DISAGG=false
DP_ATTENTION=false
DSA_DECODE_BACKEND=triton
DSA_PREFILL_BACKEND=triton
DURATION=$DUR
ENABLE_AGENTX_POWER=0
EP_SIZE=4
FRAMEWORK=sglang
HF_HOME=$WS/cache/agentx-hf
HOST_GID=1000
HOST_UID=50112975
IMAGE=$IMG
INFERENCEX_COMMIT=918524ff94045b3f091115f1051c22a8588edf2b
INFMAX_CONTAINER_WORKSPACE=$IX
IS_AGENTIC=1
IS_MULTINODE=false
KV_OFFLOADING=none
MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
MODEL_NAME=amd/GLM-5.2-MXFP4
MODEL_PREFIX=glm5.2
PORT=$PORT
PRECISION=fp4
PYTHONNOUSERSITE=1
RESULT_FILENAME=agentx_conc$CONC
RUNNER_TYPE=mi355x
SCENARIO_TYPE=agentic-coding
SERVED_MODEL_NAME=amd/GLM-5.2-MXFP4
SIMULATE_ACC_LEN=3.61
SPEC_DECODING=mtp
TP=4
EOT
docker run -d --rm --name yihou-dcp-agentx-$TAG --network host --ipc host --shm-size 32g \
  -v $WS:$WS -v $IX:$IX:ro -v /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4:/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4:ro \
  --tmpfs $WS/tmp/${NODE:-054}/client:rw,exec,size=16g -e TMPDIR=$WS/tmp/${NODE:-054}/client \
  -e HOME=$WS/cache/${NODE:-054}/client-home -e XDG_CACHE_HOME=$WS/cache/${NODE:-054}/client-xdg \
  -e UV_PYTHON_INSTALL_DIR=$WS/cache/${NODE:-054}/uv-python \
  --env-file $OUT/runtime.env --entrypoint bash $IMG -c '
    set -euo pipefail
    out="$1"
    exec > "$out/runner.log" 2>&1
    ln -s "$TMPDIR" /yihou-client-tmp; export TMPDIR=/yihou-client-tmp
    source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"
    install_agentic_deps
    resolve_trace_source
    build_replay_cmd "$out"
    run_agentic_replay_and_write_outputs "$out"
    echo "AGENTX_CLIENT_DONE rc=$?"
  ' _ "$OUT" > $OUT/container.id
echo "started yihou-dcp-agentx-$TAG -> $OUT"
