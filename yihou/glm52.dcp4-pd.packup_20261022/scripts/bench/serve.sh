#!/bin/bash
# Launch one GLM-5.2 mixed (non-PD) sglang server in a detached container (copied from mixrun/serve.sh for benchrun).
# Usage: serve.sh NAME GPUS PORT DCP(1|4) GRAPH(0|1) SIM_ACC(""|3.61) MAXRUN GRAPHBS [EXTRA_ARGS...]
set -euo pipefail
NAME=$1; GPUS=$2; PORT=$3; DCP=$4; GRAPH=$5; SIM=$6; MAXRUN=$7; GBS=$8; shift 8
IMG=lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020
WS=/shared_nfs/yihou_glm52/dcp_pd_20261021
SHA=${SHA:-4792dc0e52}
NODE=${NODE:-054}
SRC=$WS/src/sglang-$SHA/python/sglang
MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
C=$WS/cache/$NODE
TAG=${NAME#yihou-dcp-}
# aiter JIT locks are PID-based: never share one JIT dir between concurrently running containers
# (2026-10-21 R1: two servers sharing one dir left a 0-byte module_custom_all_reduce.so).
AJ=$C/aiter-jit/v0522-20261020/g${GPUS//,/}
LOGDIR=$WS/bench/logs; mkdir -p $LOGDIR $AJ $C/triton/$TAG $C/hf $C/xdg/$TAG
LOG=$LOGDIR/$TAG.$NODE.$(date -u +%Y%m%dT%H%M%SZ).log
args=(--model-path $MODEL --served-model-name amd/GLM-5.2-MXFP4 --host 0.0.0.0 --port $PORT
  --trust-remote-code --tp 4 --ep 4 --kv-cache-dtype fp8_e4m3
  --dsa-prefill-backend triton --dsa-decode-backend triton
  --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6
  --chunked-prefill-size 32768 --mem-fraction-static 0.85
  --max-running-requests $MAXRUN --cuda-graph-max-bs-decode $GBS
  --tool-call-parser glm47 --reasoning-parser glm45 --watchdog-timeout 1800 --enable-metrics --enable-cache-report)
[[ $DCP -gt 1 ]] && args+=(--dcp-size $DCP)
[[ $GRAPH == 0 ]] && args+=(--disable-cuda-graph)
# JSON_OVERRIDE_FILE: file holding the --json-model-override-args JSON (avoids nested shell quoting).
[[ -n "${JSON_OVERRIDE_FILE:-}" ]] && args+=(--json-model-override-args "$(cat $JSON_OVERRIDE_FILE)")
args+=("$@")
# EXTRA_ENV: space-separated KEY=VALUE added to the container env (e.g. SAFETENSORS_FAST_GPU=1: v3-045 weight-load hang).
envs=(-e SGLANG_OPT_USE_TOPK_V2=true -e PYTHONNOUSERSITE=1 -e SGLANG_TIMEOUT_KEEP_ALIVE=900
  -e SGLANG_USE_AITER=1 -e PYTHONDONTWRITEBYTECODE=1 -e HIP_VISIBLE_DEVICES=$GPUS
  -e AITER_JIT_DIR=/aiter-jit -e TRITON_CACHE_DIR=$C/triton/$TAG -e HF_HOME=$C/hf -e XDG_CACHE_HOME=$C/xdg/$TAG)
for kv in ${EXTRA_ENV:-}; do envs+=(-e $kv); done
if [[ -n "$SIM" ]]; then
  envs+=(-e SGLANG_SIMULATE_ACC_LEN=$SIM -e SGLANG_SIMULATE_ACC_METHOD=match-expected -e SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token)
fi
# Two containers with private PID namespaces on one node: ROCm/CLR IPC fails ("HIP failure: invalid argument" at RCCL
# init, 2026-10-21 R2) when an exporter pid equals an importer pid. --pid host is denied by spur-authz, so burn
# PID_OFFSET pids before exec (kit engine.sh R022). Use a different PID_OFFSET per concurrently running container.
PID_OFFSET=${PID_OFFSET:-0}
burn="i=0; while [ \$i -lt $PID_OFFSET ]; do ( : ); i=\$((i+1)); done;"
qargs=$(printf "%q " "${args[@]}")
echo "LOG=$LOG"
{ echo "# $(date -u +%FT%TZ) image=$IMG sglang_src=$SHA name=$NAME gpus=$GPUS pid_offset=$PID_OFFSET node=$NODE cpuset=${CPUSET:-} sha=$SHA"; echo "# env: ${envs[*]}"; echo "# args: ${args[*]}"; } > $LOG
cpu_args=()
# CPUSET: pin the container to a CPU set (e.g. its GPUs' NUMA node) and turn off sglang's gpu_id-based
# affinity, which gives two 4-GPU servers on one node the SAME CPU set.
[[ -n "${CPUSET:-}" ]] && { cpu_args=(--cpuset-cpus $CPUSET); envs+=(-e SGLANG_SET_CPU_AFFINITY=0); }
docker run -d --rm --init --name $NAME "${cpu_args[@]}" --network host --ipc host --shm-size 32g \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
  -v $SRC:/sgl-workspace/sglang/python/sglang:ro -v $MODEL:$MODEL:ro -v $WS:$WS \
  -v $AJ:/aiter-jit "${envs[@]}" -w /tmp \
  --entrypoint bash $IMG -c "$burn cd /tmp && python3 -c 'import sglang;print(\"SGLANG_FILE\",sglang.__file__)' >> $LOG 2>&1 && exec python3 -m sglang.launch_server $qargs >> $LOG 2>&1" >> $LOG 2>&1
