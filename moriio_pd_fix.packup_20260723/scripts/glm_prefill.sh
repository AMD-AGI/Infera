#!/bin/bash
# GLM-5.1-FP8 MoRIIO PREFILL (kv_producer) INSIDE container glm_pd on chi2879 (.10). TP4 gpu0-3.
set -u
IP=10.2.122.10; ROUTER=10.2.122.10; ETCD=10.2.122.10:2379; PREFIX=/infera/pd-glm/
MODEL=/mnt/vast/xiaobo/models/GLM-5.1-FP8
LOG=/mnt/vast/c_huggingface/vllm_patch_verify/glm_prefill.log
pkill -9 -f infera.engine.vllm 2>/dev/null; sleep 1; : > "$LOG"
cd /opt/infera
export VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ENGINE_READY_TIMEOUT_S=3600
export AITER_BF16_FP8_MOE_BOUND=0 PYTHONHASHSEED=0 VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
export VLLM_HOST_IP=$IP MC_GID_INDEX=1 MORI_IB_GID_INDEX=1 HIP_VISIBLE_DEVICES=0,1,2,3
KVCFG=$(cat <<JSON
{"kv_connector":"MoRIIOConnector","kv_role":"kv_producer","kv_connector_extra_config":{"host_ip":"$IP","proxy_ip":"$ROUTER","proxy_ping_port":36000,"http_port":30001,"handshake_port":36100,"notify_port":36200,"backend":"rdma","tp_size":"4"}}
JSON
)
nohup python3 -m infera.engine.vllm \
  --model "$MODEL" --served-model-name GLM-5.1-FP8 --port 30001 --host 0.0.0.0 --advertise-host "$IP" \
  --discovery-backend etcd --etcd-endpoint "$ETCD" --etcd-prefix "$PREFIX" \
  --request-transport http --no-enable-kv-events --trust-remote-code \
  --tensor-parallel-size 4 --distributed-executor-backend mp \
  --kv-cache-dtype fp8 --moe-backend aiter \
  --reasoning-parser glm45 \
  --no-enable-prefix-caching --gpu-memory-utilization 0.85 \
  --max-model-len 9472 --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --kv-transfer-config "$KVCFG" \
  > "$LOG" 2>&1 &
echo "prefill_pid=$!"
