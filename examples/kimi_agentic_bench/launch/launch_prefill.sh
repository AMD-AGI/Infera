#!/bin/bash
# Launch the PREFILL engine (Infera PD, kv_producer role) on the prefill node.
# Run inside the Infera container:  bash launch/launch_prefill.sh
#
# Env toggles:
#   TP=4     tensor-parallel size for prefill (default 4)
#   KVD=1    enable KV cache offload (kvd L3 + Mooncake MultiConnector). Requires a
#            kvd daemon on $KVD_SOCK and $KVD_L3 on a single NVMe. KVD=0 (default) is
#            Mooncake-only.
#   STATS=1  expose /metrics (prefix_cache_* counters used to read the cache-hit rate).
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; source "$HERE/env.sh"
KVD="${KVD:-0}"; STATS="${STATS:-1}"; TP="${TP:-4}"; PORT="${PORT:-30000}"

export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
       HSA_NO_SCRATCH_RECLAIM=1 PYTHONHASHSEED=0
export VLLM_HOST_IP=$PREFILL_IP MC_GID_INDEX=1 NCCL_IB_GID_INDEX=1 RDMAV_FORK_SAFE=1
export VLLM_MOONCAKE_BOOTSTRAP_PORT=${MC_PORT:-8998}

STATFLAG="--disable-log-stats"; [ "$STATS" = "1" ] && STATFLAG=""

if [ "$KVD" = "1" ]; then
  # INFERA_KVD_AIS=1 uses the hipFile GPU-direct path (single-NVMe L3). Set AIS=0 to
  # use the POSIX mmap read path. LOAD_WORKERS=auto parallelizes L3 reads.
  export INFERA_KVD_SOCKET=$KVD_SOCK \
         INFERA_KVD_AIS=${INFERA_KVD_AIS:-1} \
         INFERA_KVD_LOAD_WORKERS=${INFERA_KVD_LOAD_WORKERS:-auto} \
         INFERA_KVD_HIPFILE_ROOTS=long=$KVD_L3 INFERA_KVD_LOG_L3=1
  KVCFG='{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"InferaKvdConnector","kv_role":"kv_both","kv_connector_module_path":"infera.engine.vllm.kvd_connector"},{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}]}}'
else
  KVCFG='{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
fi

cd /tmp
exec python3 -m infera.engine.vllm \
  --model "$MODEL" --served-model-name "$SERVED" \
  --port $PORT --host 0.0.0.0 --advertise-host $PREFILL_IP --etcd-endpoint $ETCD_EP \
  --tensor-parallel-size $TP --enable-expert-parallel \
  --max-model-len 131072 --max-num-seqs 256 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization ${GMU:-0.6} --kv-cache-dtype fp8 --block-size 16 \
  --enable-prefix-caching --trust-remote-code $STATFLAG \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq \
  --compilation-config '{"cudagraph_mode":"NONE"}' \
  --kv-transfer-config "$KVCFG"
