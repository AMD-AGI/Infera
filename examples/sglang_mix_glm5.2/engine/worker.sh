#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Start the engine container: one TP4 + DP4-attention GLM-5.2 worker with EAGLE
# MTP and a host-memory HiCache tier. This file carries the tuned recipe; site
# values come from env.sh. Normally called by up.sh.
#
# Usage: bash engine/worker.sh
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$KIT/engine/common.sh"

# NCCL / gloo / SGLang bind to the interface that carries NODE_IP.
nic="$(ip -o -4 addr show | awk -v ip="$NODE_IP" '$4 ~ ("^" ip "/") {print $2; exit}')"
[[ -n "$nic" ]] || die "no local interface has $NODE_IP"

env=(
    "HIP_VISIBLE_DEVICES=$GPUS"
    "SGLANG_HOST_IP=$NODE_IP" "HOST_IP=$NODE_IP"
    "SGLANG_LOCAL_IP_NIC=$nic" "GLOO_SOCKET_IFNAME=$nic" "NCCL_SOCKET_IFNAME=$nic"
    NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1
    AITER_USE_FLYDSL_MOE_SORTING=1 PYTHONHASHSEED=0 SAFETENSORS_FAST_GPU=1
    SGLANG_DP_USE_GATHERV=1 SGLANG_OPT_USE_TOPK_V2=false SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=0
    # Per-call hipHostRegister limit: keeps each 270 GB host pool in one
    # registration (the 256 GB default would split it).
    SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512
    # AgentX keeps connections idle between the turns of a session.
    SGLANG_TIMEOUT_KEEP_ALIVE=900
    INFERA_ENGINE_READY_TIMEOUT=3600
)
if [[ -n "$SIM_ACCEPT_LEN" ]]; then
    env+=("SGLANG_SIMULATE_ACC_LEN=$SIM_ACCEPT_LEN"
        SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token)
fi
env_args=()
for kv in "${env[@]}"; do env_args+=(-e "$kv"); done

# Caches persist per image in named volumes: /root/.cache (Triton, HF) and
# AITER's module directory, which a new volume copies from the image.
image_id="$(docker image inspect -f '{{.Id}}' "$IMAGE")"
mounts=(-v "$MODEL:$MODEL:ro"
    -v "$CTR-cache-${image_id:7:12}:/root/.cache"
    -v "$CTR-aiter-jit-${image_id:7:12}:/sgl-workspace/aiter/aiter/jit")
# Host libionic, so the entrypoint can match the host ionic kernel ABI.
[[ ! -e /lib/x86_64-linux-gnu/libionic.so ]] ||
    mounts+=(-v /lib/x86_64-linux-gnu/libionic.so:/host-libionic/libionic.so:ro)

read -ra extra <<<"$ENGINE_EXTRA_ARGS"
docker run -d --name "$CTR-engine" --network host --ipc host --shm-size 32g \
    --device /dev/kfd --device /dev/dri --device /dev/infiniband \
    --group-add video --group-add render --cap-add SYS_PTRACE --cap-add IPC_LOCK \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
    "${mounts[@]}" "${env_args[@]}" "$IMAGE" \
    python3 -m infera.engine.sglang \
    --model-path "$MODEL" --served-model-name "$SERVED_MODEL" --trust-remote-code \
    --host "$NODE_IP" --port "$ENGINE_PORT" --advertise-host "$NODE_IP" \
    --discovery-backend etcd --etcd-endpoint "$NODE_IP:$ETCD_PORT" \
    --request-transport http --kv-event-transport zmq \
    --enable-kv-events --kv-events on --kv-events-bind "tcp://0.0.0.0:$KV_EVENTS_PORT" \
    --kv-snapshot-port "$KV_SNAPSHOT_PORT" \
    --tp-size 4 --ep-size 1 --dp-size 4 --enable-dp-attention --enable-dp-lm-head \
    --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton --dsa-decode-backend triton \
    --mem-fraction-static 0.85 --chunked-prefill-size 32768 --max-prefill-tokens 8192 \
    --max-running-requests 128 --cuda-graph-max-bs-decode 64 \
    --enable-prefill-delayer --prefill-delayer-max-delay-passes 60 \
    --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 --json-model-override-args '{"index_share_for_mtp_iteration":false}' \
    --enable-hierarchical-cache --hicache-size 270 --hicache-ratio 1.5 \
    --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first \
    --enable-fused-qk-norm-rope --enable-aiter-allreduce-fusion \
    --reasoning-parser glm45 --tool-call-parser glm47 \
    --enable-metrics --enable-cache-report --watchdog-timeout 3600 \
    "${extra[@]}" >/dev/null
echo "started $CTR-engine on GPUs $GPUS, port $ENGINE_PORT"
