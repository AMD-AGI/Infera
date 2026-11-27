#!/usr/bin/env bash
# One AgentX point against a running deployment with InferenceX's own client and scoring (InferenceX main d490919c:
# `python3 -m infx.bench agentic` = agentx-harness replay of semianalysis_cc_traces_weka_062126, seed 42, 10 warmup
# requests per lane, 3600 s, failed-request threshold 0.10; then process_agentic_result / validate_agentic_result).
# tok/s/GPU = (input + output tokens of the successful profiled requests) / span / 8 GPUs (4 prefill + 4 decode).
#   agentx.sh OUT_DIR ROUTER_URL CONC "ENGINE_METRICS_URLS"
# Env: IMAGE, FAST (1 = InferenceX fast mode: 1200 s, 1 warmup per lane; a preflight, never a score), MTP_N (4).
set -uo pipefail
K=$(cd "$(dirname "$0")" && pwd)
. "$K/env.sh"
out=${1:?out dir} url=${2:?router url} conc=${3:?conc} murls=${4:-}
mkdir -p "$out/cache" && chmod -R 777 "$out"
metrics="$url/metrics"; for m in $murls; do metrics+=",$m/metrics"; done
cat > "$out/runtime.env" <<EOF
MODEL=$MODEL
TOKENIZER=$MODEL
SERVED_MODEL_NAME=$SERVED
MODEL_PREFIX=glm5.2
FRAMEWORK=vllm
PRECISION=fp4
IMAGE=$IMAGE
RUNNER_TYPE=mi355x
PREFILL_HARDWARE=mi355x
DECODE_HARDWARE=mi355x
CONC=$conc
DURATION=3600
AIPERF_EXPERIMENTAL_FAST=${FAST:-0}
AIPERF_SERVER_URL=$url
AIPERF_SERVER_METRICS_URLS=$metrics
AIPERF_REQUIRED_SERVER_METRIC_PREFIX=vllm:
AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true
AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=false
AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING=false
AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS=0
IS_AGENTIC=1
SCENARIO_TYPE=agentic-coding
EVAL_ONLY=false
IS_MULTINODE=false
DISAGG=true
ENABLE_AGENTX_POWER=0
REQUIRE_POWER=0
PREFILL_NUM_WORKERS=1
PREFILL_TP=1
PREFILL_PP_SIZE=4
PREFILL_EP=1
PREFILL_DP_ATTN=false
PREFILL_GPUS_PER_WORKER=4
DECODE_NUM_WORKERS=1
DECODE_TP=4
DECODE_EP=1
DECODE_DP_ATTN=true
DECODE_GPUS_PER_WORKER=4
SPEC_DECODING=mtp
SIMULATE_ACC_LEN=$(golden_al "${MTP_N:-4}")
KV_OFFLOADING=dram
KV_OFFLOAD_BACKEND=vllm-native
KV_P2P_TRANSFER=mooncake
TOTAL_CPU_DRAM_GB=$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo)
RESULT_DIR=$out
RESULT_FILENAME=agentx_conc$conc
AGENTIC_OUTPUT_DIR=$out
HOME=$out/cache
XDG_CACHE_HOME=$out/cache/xdg
HF_HOME=/perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/cache/hf
UV_CACHE_DIR=/perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/cache/uv
TMPDIR=/ax-tmp
PYTHONSAFEPATH=1
PYTHONPATH=$IXM/inferencex-e2e
EOF
mkdir -p /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/cache/{hf,uv}
echo "$(ts) AgentX C$conc ($( [[ "${FAST:-0}" == 1 ]] && echo fast || echo full)) via $url -> $out"
# AIPerf binds ZMQ ipc:// sockets under TMPDIR: keep it short (unix socket paths are limited to ~107 bytes) and node-local.
axtmp=$(mktemp -d /tmp/lyl-axtmp.XXXXXX) && chmod 777 "$axtmp"
docker rm -f vpd-agentx > /dev/null 2>&1
docker run --name vpd-agentx --rm --network host --ipc host --shm-size 32g --user "$(id -u):$(id -g)" \
    -v "$IXM:$IXM:ro" -v "$MODEL:$MODEL:ro" -v "$out:$out" -v /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/cache:/perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/cache \
    -v "$axtmp:/ax-tmp" --env-file "$out/runtime.env" -w "$IXM/inferencex-e2e" --entrypoint python3 "$IMAGE" -m infx.bench agentic \
    > "$out/runner.log" 2>&1
rc=$?
python3 - "$out" <<'EOF' | tee "$out/score.txt"
import glob, json, sys
for f in sorted(glob.glob(f"{sys.argv[1]}/agentx_conc*.json")):
    d = json.load(open(f))
    pg, ia = d.get("per_gpu", {}), d.get("interactivity", d)
    print(f"{f.rsplit('/', 1)[-1]}: tok/s/GPU {pg.get('total_tput_tps')} keys {sorted(d)[:12]}")
EOF
echo "$(ts) AgentX rc=$rc"
exit $rc
