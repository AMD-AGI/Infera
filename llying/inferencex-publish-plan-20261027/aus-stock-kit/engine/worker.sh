#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Start one engine container on THIS node: a TP1 x PP4 prefill with a
# host-memory HiCache tier, or the TP + DP-attention decode (8 or 4 GPUs); both
# legs carry EAGLE MTP.
# This file carries the tuned recipe; site values come from env.sh. Normally
# called by up.sh (the decode over ssh).
#
# Usage: bash engine/worker.sh prefill 0|1
#        bash engine/worker.sh decode
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$KIT/engine/common.sh"

role="${1:-}" idx="${2:-0}"
case "$role" in
prefill)
    [[ "$idx" == 0 || "$idx" == 1 ]] || die "usage: worker.sh prefill 0|1"
    gpus_var="PREFILL${idx}_GPUS"
    gpus="${!gpus_var}" node_ip="$PREFILL_NODE_IP" name="$CTR-prefill$idx"
    port=$((PREFILL_PORT + 256 * idx))
    ;;
decode)
    gpus="$DECODE_GPUS" node_ip="$DECODE_NODE_IP" name="$DECODE_CONTAINER" port="$DECODE_PORT"
    ;;
*) die "usage: worker.sh prefill 0|1 | decode" ;;
esac
n_gpus=$(($(tr -cd , <<<"$gpus" | wc -c) + 1))
[[ "$role" != prefill || "$n_gpus" == 4 ]] || die "a prefill is TP1 x PP4: $gpus_var needs 4 GPUs"

# NCCL / gloo / SGLang bind to the interface that carries this node's IP.
nic="$(ip -o -4 addr show | awk -v ip="$node_ip" '$4 ~ ("^" ip "/") {print $2; exit}')"
[[ -n "$nic" ]] || die "no local interface has $node_ip; run worker.sh $role on its own node"

env=(
    "HIP_VISIBLE_DEVICES=$gpus"
    "SGLANG_HOST_IP=$node_ip" "HOST_IP=$node_ip"
    "SGLANG_LOCAL_IP_NIC=$nic" "GLOO_SOCKET_IFNAME=$nic" "NCCL_SOCKET_IFNAME=$nic"
    NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1 HIP_FORCE_DEV_KERNARG=1
    AITER_USE_FLYDSL_MOE_SORTING=1 PYTHONHASHSEED=0 SAFETENSORS_FAST_GPU=1
    SGLANG_DP_USE_GATHERV=1 SGLANG_OPT_USE_TOPK_V2=true SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=0
    # Mooncake, as the InferenceX recipe sets it on the image's own Mooncake
    # (78566d45): no HIP IPC across nodes, each transfer on the same-named NIC
    # at both ends. MC_ENABLE_DEST_LOCAL_RAIL / MC_IB_TIMEOUT belong to the
    # rebuilt Mooncake of the Infera image and do not exist here.
    "MC_GID_INDEX=$MC_GID_INDEX"
    "MOONCAKE_DISABLE_HIP_DMABUF=$MOONCAKE_DISABLE_HIP_DMABUF"
    MC_DISABLE_HIP=1 MC_ENABLE_DEST_DEVICE_AFFINITY=1 RDMAV_FORK_SAFE=1
    # KV buffers are registered in chunks of at most this size: ionic refuses
    # a 2 GiB chunk of an allocation that starts 1 GiB-aligned (README note 8).
    MC_MAX_MR_SIZE=1073741824
    # The legs load in parallel, and under load a request can wait minutes for
    # its KV; the 300 s defaults give up first.
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800
    # AgentX keeps connections idle between the turns of a session.
    SGLANG_TIMEOUT_KEEP_ALIVE=900
    INFERA_ENGINE_READY_TIMEOUT=3600
)
args=(
    --model-path "$MODEL" --served-model-name "$SERVED_MODEL" --trust-remote-code
    --host "$node_ip" --port "$port" --advertise-host "$node_ip"
    --discovery-backend etcd --etcd-endpoint "$PREFILL_NODE_IP:$ETCD_PORT"
    --request-transport http --kv-event-transport zmq
    --disaggregation-mode "$role" --disaggregation-transfer-backend mooncake
    --disaggregation-ib-device "$RDMA_DEVICES"
    --ep-size 1 --disable-custom-all-reduce
    --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton --dsa-decode-backend triton
    --max-running-requests 256
    --json-model-override-args '{"index_share_for_mtp_iteration":false}'
    --enable-fused-qk-norm-rope --enable-aiter-allreduce-fusion
    --reasoning-parser glm45 --tool-call-parser glm47
    --enable-metrics --enable-cache-report --watchdog-timeout 3600
    # The prefill needs the decode's EAGLE flags too: without its NextN layer
    # the transfer carries no draft KV, and the decode's draft layer reads
    # uninitialised KV for every prompt token (README note 10).
    --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1
    --speculative-num-draft-tokens 6
)
if [[ "$role" == prefill ]]; then
    # Per-call hipHostRegister limit: keeps each 250 GB host pool in one
    # registration (the 256 GB default would split it).
    env+=(SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512)
    # GLM-5.2 has 78 layers; the last stage also holds the LM head and the
    # NextN draft layer, so it takes two fewer.
    env+=(SGLANG_PP_LAYER_PARTITION=20,20,20,18)
    args+=(
        # Pipeline parallelism needs the overlap scheduler off. 8k chunks keep
        # a stage short, so a small prefill does not wait behind a long one
        # through all four stages (README note 11).
        --tp-size 1 --pp-size 4 --disable-overlap-schedule
        --chunked-prefill-size 8192 --max-prefill-tokens 8192
        --mem-fraction-static 0.88 --schedule-policy shortest-prefill-first
        --disaggregation-bootstrap-port $((BOOTSTRAP_PORT + idx))
        --enable-kv-events --kv-events on
        --kv-events-bind "tcp://0.0.0.0:$((KV_EVENTS_PORT + 8 * idx))"
        --kv-snapshot-port $((KV_SNAPSHOT_PORT + idx))
        --enable-hierarchical-cache --hicache-size 250 --hicache-ratio 1.5
        --hicache-write-policy write_back --hicache-io-backend kernel --hicache-mem-layout page_first
    )
    read -ra extra <<<"$PREFILL_EXTRA_ARGS"
else
    if [[ -n "$SIM_ACCEPT_LEN" ]]; then
        env+=("SGLANG_SIMULATE_ACC_LEN=$SIM_ACCEPT_LEN"
            SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token)
    fi
    args+=(
        --tp-size "$n_gpus" --dp-size "$n_gpus" --enable-dp-attention --chunked-prefill-size 32768
        --mem-fraction-static 0.92 --cuda-graph-max-bs-decode 256 --tokenizer-worker-num 4
        --no-enable-kv-events --kv-events off
    )
    read -ra extra <<<"$DECODE_EXTRA_ARGS"
fi
args+=("${extra[@]}")
env_args=()
for kv in "${env[@]}"; do env_args+=(-e "$kv"); done

cmd=(bash /kit/engine/stock_entry.sh python3 -m infera.engine.sglang "${args[@]}")
# Both prefills run with --ipc host on one node, and RCCL names shared memory
# after process ids, which start low in every container: prefill 1 burns
# 30,000 pids first so its ranks never collide with prefill 0's (README note 4).
if [[ "$role" == prefill && "$idx" == 1 ]]; then
    cmd=(bash -c 'i=0; while [ $i -lt 30000 ]; do ( : ); i=$((i + 1)); done; exec "$@"' _ "${cmd[@]}")
fi

# Caches persist per container and image in named volumes: /root/.cache
# (Triton, HF) and AITER's module directory, which a new volume copies from
# the image.
image_id="$(docker image inspect -f '{{.Id}}' "$IMAGE")"
mounts=(-v "$MODEL:$MODEL:ro" -v "$KIT:/kit:ro"
    -v "$name-cache-${image_id:7:12}:/root/.cache"
    -v "$name-aiter-jit-${image_id:7:12}:/sgl-workspace/aiter/aiter/jit")
# The host's ionic verbs provider over the image's (54.0-149), the same mount
# the InferenceX kimik3 recipe uses: the provider must match the host kernel ABI.
if [[ "$MOUNT_HOST_IONIC" == 1 ]]; then
    [[ -e "$HOST_IONIC_PROVIDER" ]] || die "MOUNT_HOST_IONIC=1 but $HOST_IONIC_PROVIDER is absent on $(hostname -s)"
    mounts+=(-v "$HOST_IONIC_PROVIDER:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro")
fi

docker run -d --name "$name" --network host --ipc host --shm-size 32g \
    --device /dev/kfd --device /dev/dri --device /dev/infiniband \
    --group-add video --group-add render --cap-add SYS_PTRACE --cap-add IPC_LOCK \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
    "${mounts[@]}" "${env_args[@]}" "$IMAGE" "${cmd[@]}" >/dev/null
echo "started $name on $(hostname -s) GPUs $gpus, port $port"
