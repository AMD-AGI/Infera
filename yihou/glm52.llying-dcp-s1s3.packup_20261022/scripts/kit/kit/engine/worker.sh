#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Start one engine container on THIS node: a TP4 + DP4-attention prefill with
# a host-memory HiCache tier, or the TP8 + DP8-attention decode; both legs
# carry EAGLE MTP.
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
    # Mooncake: every rank sees all rails; each destination GPU is pinned to
    # one HCA on both hosts, and the RC ACK timeout outlasts NIC stalls.
    "MC_TE_FILTERS=${MC_TE_FILTERS_OVERRIDE:-$RDMA_DEVICES}" "MC_GID_INDEX=$MC_GID_INDEX"
    "MOONCAKE_DISABLE_HIP_DMABUF=$MOONCAKE_DISABLE_HIP_DMABUF"
    MC_DISABLE_HIP_TRANSPORT=1 MC_ENABLE_DEST_DEVICE_AFFINITY=1 MC_ENABLE_DEST_LOCAL_RAIL=1
    MC_IB_TIMEOUT=18 RDMAV_FORK_SAFE=1
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
    --disaggregation-ib-device "${DISAGG_IB_DEVICE:-$RDMA_DEVICES}"
    --ep-size 1
    # yihou dcp_pd: custom all-reduce ON (user directive 2026-10-22 10:12Z).
    --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton --dsa-decode-backend triton
    --chunked-prefill-size 32768 --max-running-requests 256
    --json-model-override-args "{\"index_share_for_mtp_iteration\":${INDEX_SHARE:-true}}"
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
    # PP4 x TP1 prefill (as GB300 Dynamo+SGLang): the last stage also holds
    # the LM head and the NextN draft layer, so it takes two fewer layers.
    env+=(SGLANG_PP_LAYER_PARTITION=20,20,20,18)
    args+=(
        --tp-size 1 --pp-size "$n_gpus" --disable-overlap-schedule
        --chunked-prefill-size "${PP4_CHUNK:-16384}" --max-prefill-tokens "${PP4_CHUNK:-16384}"
        --mem-fraction-static 0.88 --schedule-policy shortest-prefill-first
        --disaggregation-bootstrap-port $((BOOTSTRAP_PORT + idx))
        --enable-kv-events --kv-events on
        --kv-events-bind "tcp://0.0.0.0:$((KV_EVENTS_PORT + 8 * idx))"
        --kv-snapshot-port $((KV_SNAPSHOT_PORT + idx))
    )
    # PP4_HICACHE=0 while debugging: without the pinned host pools a failed
    # start releases its VRAM in seconds instead of tens of minutes.
    if [[ "${PP4_HICACHE:-1}" == 1 ]]; then
        args+=(
            --enable-hierarchical-cache --hicache-size 250 --hicache-ratio 1.5
            --hicache-write-policy write_back --hicache-io-backend kernel --hicache-mem-layout page_first
        )
    fi
    read -ra extra <<<"$PREFILL_EXTRA_ARGS"
else
    if [[ -n "$SIM_ACCEPT_LEN" ]]; then
        env+=("SGLANG_SIMULATE_ACC_LEN=$SIM_ACCEPT_LEN"
            SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token)
    fi
    case "${DECODE_LAYOUT:-dpa4}" in
    dpa4)   # llying: TP4 + DP4 attention, EP1
        args+=(
            --tp-size "$n_gpus" --dp-size "$n_gpus" --enable-dp-attention
            --mem-fraction-static 0.92 --cuda-graph-max-bs-decode 256 --tokenizer-worker-num 4
            --no-enable-kv-events --kv-events off
        ) ;;
    dcp4)   # yihou r6b decode: TP4 EP4 + DCP4, no DP attention (attn_tp % dcp == 0). Graph bs = max running = 96
            # (r6b: 64/64; raised to admit C80). --ep-size / --max-running-requests override the common ones (last wins).
        env+=(SGLANG_DP_USE_GATHERV=0)
        args+=(
            --tp-size "$n_gpus" --ep-size "$n_gpus" --dcp-size "$n_gpus"
            --mem-fraction-static 0.85 --cuda-graph-max-bs-decode 96 --max-running-requests 96
            --tokenizer-worker-num 4 --no-enable-kv-events --kv-events off
        ) ;;
    *) die "DECODE_LAYOUT must be dpa4|dcp4" ;;
    esac
    read -ra extra <<<"$DECODE_EXTRA_ARGS"
fi
args+=("${extra[@]}")
# yihou dcp_pd: per-leg extra container env, space-separated KEY=VALUE (e.g. the DCP4 knobs).
if [[ "$role" == prefill ]]; then read -ra env_extra <<<"${PREFILL_ENV:-}"; else read -ra env_extra <<<"${DECODE_ENV:-}"; fi
env+=("${env_extra[@]}")
env_args=()
for kv in "${env[@]}"; do env_args+=(-e "$kv"); done

cmd=(python3 -m infera.engine.sglang "${args[@]}")
# Both prefills run with --ipc host on one node, and RCCL names shared memory
# after process ids, which start low in every container: prefill 1 burns
# 30,000 pids first so its ranks never collide with prefill 0's (README note 4).
if [[ "$role" == prefill && "$idx" == 1 ]] || [[ "$role" == decode && "${DECODE_PID_BURN:-1}" == 1 ]]; then
    cmd=(bash -c 'i=0; while [ $i -lt 30000 ]; do ( : ); i=$((i + 1)); done; exec "$@"' _ "${cmd[@]}")
fi

# Caches persist per container and image in named volumes: /root/.cache
# (Triton, HF) and AITER's module directory, which a new volume copies from
# the image.
image_id="$(docker image inspect -f '{{.Id}}' "$IMAGE")"
mounts=(-v "$MODEL:$MODEL:ro"
    -v "$name-cache-${image_id:7:12}:/root/.cache"
    -v "$name-aiter-jit-${image_id:7:12}:/sgl-workspace/aiter/aiter/jit")
# yihou dcp_pd: llying's deepseek_v2.py file mount is replaced by our branch overlay (the PP fix is a commit
# there), mounted read-only over the image's sglang on BOTH legs.
[[ -z "${SGLANG_OVERLAY:-}" ]] || mounts+=(-v "$SGLANG_OVERLAY:/sgl-workspace/sglang/python/sglang:ro")
# Host libionic, so the entrypoint can match the host ionic kernel ABI.
[[ ! -e /lib/x86_64-linux-gnu/libionic.so ]] ||
    mounts+=(-v /lib/x86_64-linux-gnu/libionic.so:/host-libionic/libionic.so:ro)

docker run -d --init --name "$name" --network host --ipc host --shm-size 32g \
    --device /dev/kfd --device /dev/dri --device /dev/infiniband \
    --group-add video --group-add render --cap-add SYS_PTRACE --cap-add IPC_LOCK \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
    "${mounts[@]}" "${env_args[@]}" "$IMAGE" "${cmd[@]}" >/dev/null
echo "started $name on $(hostname -s) GPUs $gpus, port $port"
