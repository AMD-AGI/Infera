#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Start ONE engine container, the prefill or the decode leg. This file carries
# the tuned recipe; site values come from env.sh. Normally called by up.sh.
#
# Usage: bash engine/worker.sh prefill|decode
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$KIT/engine/common.sh"

role="${1:-}"
case "$role" in prefill | decode) ;; *) die "usage: bash engine/worker.sh prefill|decode" ;; esac
case "$PREFILL_PP" in 1 | 4) ;; *) die "PREFILL_PP must be 1 or 4, not $PREFILL_PP" ;; esac
resolve_model

# Experts of layers 0-77 stay MXFP4 and the rest is quantized online to
# per-token FP8, except the MTP layer 78: BF16 as in the checkpoint, or with
# MTP_EXPERTS=mxfp4 its routed experts quantized online to MXFP4 at load.
case "$MTP_EXPERTS" in
    bf16) mtp_quant="" mtp_exclude='"model.layers.78.*"' ;;
    mxfp4)
        mtp_quant=',"layer_quant_config":{"model.layers.78.mlp.experts*":"mxfp4"}'
        mtp_exclude='"re:^model\\.layers\\.78\\.(?!mlp\\.experts)"'
        ;;
    *) die "MTP_EXPERTS must be bf16 or mxfp4, not $MTP_EXPERTS" ;;
esac
QUANT_CONFIG='{"global_quant_config":"ptpc_fp8"'"$mtp_quant"',"exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate","model.layers.[0-9].mlp.*expert*","model.layers.[1-6][0-9].mlp.*expert*","model.layers.7[0-7].mlp.*expert*",'"$mtp_exclude"']}'

env=(
    AITER_QUICK_REDUCE_QUANTIZATION=INT4 AITER_USE_FLYDSL_MOE_SORTING=1 AITER_LOG_LEVEL=WARNING
    PYTHONHASHSEED=0 ATOM_MLA_PAGE_SIZE=1 ATOM_ONLINE_QUANT_STREAMING=0
    ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB=2047 ATOM_USE_TRITON_MLA=0
    MC_ENABLE_DEST_DEVICE_AFFINITY=1 "MC_GID_INDEX=$MC_GID_INDEX"
    "MOONCAKE_DISABLE_HIP_DMABUF=$MOONCAKE_DISABLE_HIP_DMABUF"
    # P and D see disjoint GPUs, which breaks the base Mooncake's same-host HIP
    # IPC path ("invalid device context"); send KV over the RDMA rails instead.
    MC_DISABLE_HIP=1
    "ATOM_HOST_IP=$NODE_IP" "INFERA_ATOM_READY_TIMEOUT=$READY_TIMEOUT"
    TRITON_CACHE_DIR=/root/.cache/triton TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor
)

# Graph capture bounded by the concurrency: 1, 2, 4, 8, then every
# DECODE_GRAPH_STEP up to 2 x CONC. Left unset, a DCP prefill would capture up
# to batch 512; a PP prefill runs eager.
cudagraph="[1,2,4,8"
for ((size = 8 + DECODE_GRAPH_STEP; size <= 2 * CONC; size += DECODE_GRAPH_STEP)); do cudagraph+=",$size"; done
cudagraph+="]"

# kv_mooncake KV_ROLE HANDSHAKE_PORT HTTP_PORT
kv_mooncake() {
    printf '{"kv_role":"%s","kv_connector":"mooncake","protocol":"rdma","handshake_port":%s,"http_port":%s,"proxy_ip":"%s"}' \
        "$1" "$2" "$3" "$NODE_IP"
}

if [[ "$role" == prefill ]]; then
    gpus="$PREFILL_GPUS" port="$PREFILL_PORT" util="$PREFILL_MEM_UTIL"
    env+=("INFERA_READINESS_PORT=$PREFILL_READINESS_PORT" OFFLOAD_PROFILE=0
        # The fabric routes same-numbered rails only: write on the rail of the
        # decode rank being served.
        ATOM_MOONCAKE_MATCHED_RAILS=auto)
    kv="$(kv_mooncake kv_producer "$PREFILL_HANDSHAKE_PORT" "$PREFILL_PORT")"
    # Append the send-worker count to that JSON object.
    [[ -z "$PREFILL_SEND_WORKERS" ]] || kv="${kv%\}},\"num_worker_threads\":$PREFILL_SEND_WORKERS}"
    if ((PREFILL_OFFLOAD_GB > 0)); then
        # The four pools, one per TP rank or PP stage, are GTT, whose cap is
        # machine-wide (gotcha 4): check it now, not minutes into the cold
        # start. Every card reports the cap as its total; 64 GiB covers the
        # engine's other pinned buffers (~42 GiB).
        # shellcheck disable=SC2016  # expanded on the docker host
        gtt_free="$(on_docker_host 'cd /sys/class/drm && used=0 &&
            for f in card*/device/mem_info_gtt_used; do used=$((used + $(cat "$f"))); done &&
            total=$(cat card*/device/mem_info_gtt_total | sort -n | tail -1) &&
            echo $(((total - used) >> 30))')" || die "cannot read the GTT use on the docker host"
        gtt_need=$((4 * PREFILL_OFFLOAD_GB + 64))
        ((gtt_need <= gtt_free)) ||
            die "PREFILL_OFFLOAD_GB=$PREFILL_OFFLOAD_GB needs $gtt_need GiB of GTT, $gtt_free GiB free: lower it (README gotcha 4)"
        # Mooncake producer plus LMCache CPU tier. Infera registers a "multi"
        # connector that contains a kv_producer as the PD prefill.
        kv="{\"kv_connector\":\"multi\",\"connectors\":[$kv,{\"kv_connector\":\"lmcache_offload\",\"kv_role\":\"offload\"}]}"
        # Chunk 256 must be a multiple of the block size x DCP4. DCP ranks and
        # PP stages hold distinct KV, so lookups query all four local ranks.
        # shellcheck disable=SC2054  # the commas are part of the value
        env+=(LMCACHE_LOCAL_CPU=True "LMCACHE_MAX_LOCAL_CPU_SIZE=$PREFILL_OFFLOAD_GB"
            LMCACHE_CHUNK_SIZE=256 "OFFLOAD_MIN_LOAD_TOKENS=$OFFLOAD_MIN_LOAD_TOKENS" LMCACHE_NUMA_MODE=auto
            LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3
            # The LMCache pool must be GTT; see patch_lmcache_rocm_pinned_alloc.py.
            HSA_USERPTR_FOR_PAGED_MEM=0)
    fi
    if ((PREFILL_PP > 1)); then
        # TP1 x PP4 as in ATOM's GLM-5.2 AgentX recipe; PP runs without CUDA graphs.
        env+=("VLLM_PP_LAYER_PARTITION=$PREFILL_PP_PARTITION")
        args=(-tp 1 -pp "$PREFILL_PP" --enforce-eager)
    else
        args=(-tp 4 --decode-context-parallel-size 4 --cudagraph-capture-sizes "$cudagraph")
    fi
    args+=(--max-num-batched-tokens "$PREFILL_MAX_BATCHED_TOKENS" --max-num-seqs 512)
    read -ra extra <<<"$PREFILL_EXTRA_ARGS"
else
    gpus="$DECODE_GPUS" port="$DECODE_PORT" util="$DECODE_MEM_UTIL"
    env+=("INFERA_READINESS_PORT=$DECODE_READINESS_PORT")
    kv="$(kv_mooncake kv_consumer "$DECODE_HANDSHAKE_PORT" "$DECODE_PORT")"
    args=(-tp 4 --decode-context-parallel-size 4 --cudagraph-capture-sizes "$cudagraph"
        --max-num-batched-tokens 16384 --max-num-seqs $((2 * CONC)))
    [[ -z "$MTP_AL" ]] || args+=(--spec-decode-acceptance-length "$MTP_AL")
    read -ra extra <<<"$DECODE_EXTRA_ARGS"
    read -ra extra_env <<<"$DECODE_EXTRA_ENV"
    env+=("${extra_env[@]}")
fi

env_args=()
for kv_pair in "${env[@]}"; do env_args+=(-e "$kv_pair"); done
mounts=(-v "$MODEL:$MODEL:ro")
# Checked on the docker host, which DOCKER_HOST=ssh:// can make another node.
if on_docker_host "test -e '$HOST_LIBIONIC'"; then
    mounts+=(-v "$HOST_LIBIONIC:/host-libionic/libionic.so:ro")
fi
# JIT caches persist per role and image in a named volume.
image_id="$(docker image inspect -f '{{.Id}}' "$IMAGE")"
mounts+=(-v "$CTR-cache-$role-${image_id:7:12}:/root/.cache")
# AITER builds its native modules inside its package, not under /root/.cache.
# A new named volume starts as a copy of the image's directory.
mounts+=(-v "$CTR-aiter-jit-$role-${image_id:7:12}:/app/aiter-test/aiter/jit")

docker run -d --name "$CTR-$role" --network host --ipc host --shm-size 32g \
    --device /dev/kfd --device /dev/dri --device /dev/infiniband \
    --group-add video --group-add render --cap-add SYS_PTRACE --cap-add IPC_LOCK \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
    "${mounts[@]}" -e "HIP_VISIBLE_DEVICES=$gpus" "${env_args[@]}" "$IMAGE" \
    python3 -m infera.engine.atom --etcd-endpoint "$ETCD_IP:$ETCD_PORT" --advertise-host "$NODE_IP" \
    --model "$MODEL" --host 0.0.0.0 --server-port "$port" --trust-remote-code \
    --kv_cache_dtype fp8 --block-size "$BLOCK_SIZE" \
    --enable_prefix_caching --online_quant_config "$QUANT_CONFIG" \
    --method mtp --num-speculative-tokens "$MTP_K" --gpu-memory-utilization "$util" \
    --kv-transfer-config "$kv" "${args[@]}" "${extra[@]}" >/dev/null
echo "started $CTR-$role on GPUs $gpus, port $port"
