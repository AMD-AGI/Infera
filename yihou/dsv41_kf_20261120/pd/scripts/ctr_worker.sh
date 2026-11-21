#!/usr/bin/env bash
# Inside the engine container: one vLLM worker through infera.engine.vllm, which spawns `vllm serve` and registers
# the worker in etcd. Usage: ctr_worker.sh CONFIG   (CONFIG sets GPUS, PORT, WORKER_ENV=(K=V ...), VLLM_ARGS=(...))
set -eo pipefail
source /perf_apps/yihou/dsv41_kf_20261120/pd/scripts/env.sh
source "${1:?config}"
: "${ETCD_IP:=$(node_ip)}"
export "${WORKER_ENV[@]}"
# EXTRA_WORKER_ENV: space-separated K=V for one experiment (diagnostics), applied over the config.
[ -z "${EXTRA_WORKER_ENV:-}" ] || export $EXTRA_WORKER_ENV
# kf: calibrated-profiler window (scripts/calib/hook). The fork SDK must win over /opt/rocm's copy.
if [ "${CALIB_ENABLE:-0}" = 1 ]; then
    export LD_LIBRARY_PATH="${CALIB_SDK_BUILD:?}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="${CALIB_HOOK:?}${PYTHONPATH:+:$PYTHONPATH}"
fi
export HIP_VISIBLE_DEVICES="$GPUS" CUDA_VISIBLE_DEVICES="$GPUS"
# The MI355X vLLM baseline leaves --kv-cache-dtype at vLLM's default; infera 0.3.3 would otherwise add fp8_e4m3.
export INFERA_DEFAULT_KV_FP8="${INFERA_DEFAULT_KV_FP8:-0}"
export INFERA_ENGINE_READY_TIMEOUT="${INFERA_ENGINE_READY_TIMEOUT:-7200}"
export PYTHONHASHSEED=0
echo "$(ts) worker $1 on GPUs $GPUS port $PORT"
env | grep -E '^(VLLM_|AITER|HIP_|CUDA_VISIBLE|INFERA_|OMP_|MORI|NCCL_|RCCL_|PYTHONHASHSEED)' | sort
exec python3 -m infera.engine.vllm \
    --discovery-backend etcd --etcd-endpoint "$ETCD_IP:$ETCD_PORT" \
    --request-transport http --kv-event-transport zmq --advertise-host "$(node_ip)" \
    --host 0.0.0.0 --port "$PORT" "${VLLM_ARGS[@]}"
