#!/usr/bin/env bash
# Start one aggregated (mixed: prefill + decode in one engine, no PD) SGLang worker container.
# Runs on the worker's node. Docker side as the 3P1D kit's engine.sh (devices, caches, PID
# offset, runtime-mount patches); engine side from MIXED_* in config.agg.sh.
# Usage: engine_mixed.sh INSTANCE IP ENGINE_PORT KV_EVENT_PORT SNAPSHOT_PORT CONTAINER ETCD_ENDPOINT
#   with ROUND set; DRY_RUN=1 prints the command only.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
(( $# == 7 )) || { echo "usage: $0 INSTANCE IP ENGINE_PORT KV_EVENT_PORT SNAPSHOT_PORT CONTAINER ETCD_ENDPOINT" >&2; exit 2; }
instance="$1"; node_ip="$2"; engine_port="$3"; kv_event_port="$4"; snapshot_port="$5"
container="$6"; etcd_endpoint="$7"
[[ "$container" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*llying[A-Za-z0-9_.-]*$ ]] || { echo "unsafe container name: $container" >&2; exit 2; }
set -a; source "$HERE/../config/config.agg.sh"; set +a

key="${instance//-/_}"
gpus_var="GPUS_$key"; pid_var="PID_OFFSET_$key"
gpus="${!gpus_var:?set $gpus_var}"
pid_offset="${!pid_var:-0}"
node_key="$(hostname -s)"; node_key="${node_key##*-}"
nic="$(ip -o -4 addr show | awk -v wanted="$node_ip" '$4 ~ ("^" wanted "/") {print $2; exit}')"
[[ -n "$nic" ]] || { echo "cannot resolve NIC for $node_ip" >&2; exit 1; }
image_id="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
[[ "$image_id" == "$EXPECTED_IMAGE_ID" ]] || { echo "unexpected image: $image_id" >&2; exit 1; }
aiter_jit_cache="$NODE_RW_ROOT/cache/aiter/$instance/${image_id#sha256:}"
# Seed a new AITER JIT cache from another instance's on this node (same image): modules are
# shape-independent .so files, so this only saves compile time. The cache tree is root-owned
# (docker creates missing bind sources as root), so the copy runs in a container.
if [[ ! -d "$aiter_jit_cache" && "${DRY_RUN:-0}" != 1 ]]; then
    docker run --rm --entrypoint bash -v "$NODE_RW_ROOT:$NODE_RW_ROOT" "$IMAGE" -c \
        'mkdir -p "$(dirname "$1")" && if [ -d "$2" ]; then cp -a "$2" "$1"; else mkdir -p "$1"; fi' \
        _ "$aiter_jit_cache" "${AITER_JIT_SEED:-/nonexistent}/${image_id#sha256:}"
fi
rw="$NODE_RW_ROOT"

docker_args=(
    docker run -d --init --name "$container"
    --network host --ipc host --shm-size "${SHM_SIZE:-32g}"
    --device /dev/kfd --device /dev/dri --device /dev/infiniband
    --group-add video --group-add render
    --cap-add SYS_PTRACE --cap-add IPC_LOCK
    --security-opt seccomp=unconfined --security-opt label=disable
    --ulimit memlock=-1:-1 --ulimit nofile=65536:65536
    -v "$WS:$WS:ro" -v "$MODEL:$MODEL:ro" -v "$rw:$rw"
    -v "$aiter_jit_cache:/aiter-jit" -e AITER_JIT_DIR=/aiter-jit
    --tmpfs "$rw/tmp/$node_key/$instance:rw,exec,size=${ENGINE_TMPFS_SIZE:-8g}"
    -e "TMPDIR=$rw/tmp/$node_key/$instance"
    -e "TRITON_CACHE_DIR=$rw/cache/$node_key/triton/$instance"
    -e "HF_HOME=$rw/cache/$node_key/hf/$instance"
    -e "XDG_CACHE_HOME=$rw/cache/$node_key/xdg/$instance"
    -e PYTHONDONTWRITEBYTECODE=1
)
while IFS=$'\t' read -r source destination; do
    [[ -n "$source" && -r "$source" ]] || { echo "invalid source mount: $source" >&2; exit 1; }
    docker_args+=(-v "$source:$destination:ro")
done < "$YIHOU_MOUNTS"
[[ -n "${HOST_RDMA_LIB:-}" ]] && docker_args+=(-v "$HOST_RDMA_LIB:${HOST_RDMA_MOUNT:-/host-libionic/libionic.so}:ro")
docker_args+=(
    -e "HIP_VISIBLE_DEVICES=$gpus"
    -e "SGLANG_HOST_IP=$node_ip" -e "HOST_IP=$node_ip"
    -e "SGLANG_LOCAL_IP_NIC=$nic" -e "GLOO_SOCKET_IFNAME=$nic" -e "NCCL_SOCKET_IFNAME=$nic"
    -e "NCCL_IB_DISABLE=$NCCL_IB_DISABLE" -e NCCL_IGNORE_CPU_AFFINITY=1
    -e "HSA_NO_SCRATCH_RECLAIM=$HSA_NO_SCRATCH_RECLAIM"
    -e SGLANG_USE_AITER=1
    -e "SGLANG_OPT_USE_TOPK_V2=$SGLANG_OPT_USE_TOPK_V2"
    -e "SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=$SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK"
    -e PYTHONNOUSERSITE=1
    -e "SGLANG_TIMEOUT_KEEP_ALIVE=$SGLANG_TIMEOUT_KEEP_ALIVE"
    -e AITER_USE_FLYDSL_MOE_SORTING=1 -e SAFETENSORS_FAST_GPU=1
    -e HIP_FORCE_DEV_KERNARG=1 -e PYTHONHASHSEED=0
    -e "SGLANG_DP_USE_GATHERV=$MIXED_DPA"
    -e "INFERA_ENGINE_READY_TIMEOUT=$READY_TIMEOUT"
    -e "INFERA_NODEPORT_RANGE=$INFERA_NODEPORT_RANGE"
    -e "SGLANG_ENABLE_FAILED_SESSION_PROBE=$SGLANG_ENABLE_FAILED_SESSION_PROBE"
    -e "SGLANG_FAILED_SESSION_PROBE_INTERVAL_S=$SGLANG_FAILED_SESSION_PROBE_INTERVAL_S"
)
[[ -n "${HICACHE_HOST_REGISTER_CHUNK_GB:-}" ]] &&
    docker_args+=(-e "SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=$HICACHE_HOST_REGISTER_CHUNK_GB")
if [[ "$MIXED_MTP" == 1 && -n "${MIXED_SIMULATE_ACC_LEN:-}" ]]; then
    docker_args+=(
        -e "SGLANG_SIMULATE_ACC_LEN=$MIXED_SIMULATE_ACC_LEN"
        -e SGLANG_SIMULATE_ACC_METHOD=match-expected
        -e SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
    )
fi
for item in ${MIXED_EXTRA_ENV:-}; do
    [[ "$item" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "invalid MIXED_EXTRA_ENV item: $item" >&2; exit 2; }
    docker_args+=(-e "$item")
done

engine_args=(
    python3 -m infera.engine.sglang
    --model-path "$MODEL" --served-model-name "$SERVED_MODEL"
    --host "$node_ip" --port "$engine_port" --advertise-host "$node_ip"
    --etcd-endpoint "$etcd_endpoint" --discovery-backend etcd
    --request-transport http --kv-event-transport zmq
    --tp-size "$MIXED_TP" --ep-size "$MIXED_EP" --trust-remote-code
    --dsa-prefill-backend "${DSA_PREFILL_BACKEND:-tilelang}"
    --dsa-decode-backend "${DSA_DECODE_BACKEND:-tilelang}"
    --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}"
    --mem-fraction-static "$MIXED_MEM_FRACTION"
    --chunked-prefill-size "$MIXED_CHUNK_SIZE"
    --max-prefill-tokens "$MIXED_MAX_PREFILL_TOKENS"
    --cuda-graph-max-bs-decode "$MIXED_GRAPH_MAX_BS"
    --max-running-requests "$MIXED_MAX_RUNNING"
    --watchdog-timeout "${WATCHDOG_TIMEOUT:-3600}"
    --reasoning-parser "${REASONING_PARSER:-glm45}"
    --tool-call-parser "${TOOL_CALL_PARSER:-glm47}"
    --enable-fused-qk-norm-rope
    --enable-cache-report --enable-metrics
    --enable-kv-events --kv-events on
    --kv-events-bind "tcp://0.0.0.0:$kv_event_port"
    --kv-snapshot-port "$snapshot_port"
)
[[ "${MIXED_PREFILL_DELAYER:-1}" == 1 ]] &&
    engine_args+=(--enable-prefill-delayer --prefill-delayer-max-delay-passes "$MIXED_PREFILL_DELAYER_MAX_PASSES")
(( MIXED_DP > 1 )) && engine_args+=(--dp-size "$MIXED_DP")
[[ "$MIXED_DPA" == 1 ]] && engine_args+=(--enable-dp-attention)
[[ "$MIXED_DP_LM_HEAD" == 1 ]] && engine_args+=(--enable-dp-lm-head)
[[ -n "${DSA_TOPK_BACKEND:-}" ]] && engine_args+=(--dsa-topk-backend "$DSA_TOPK_BACKEND")
[[ -n "${JSON_MODEL_OVERRIDE_ARGS:-}" ]] && engine_args+=(--json-model-override-args "$JSON_MODEL_OVERRIDE_ARGS")
if [[ "$MIXED_MTP" == 1 ]]; then
    engine_args+=(
        --speculative-algorithm EAGLE
        --speculative-num-steps "$DECODE_SPEC_STEPS"
        --speculative-eagle-topk "$DECODE_SPEC_TOPK"
        --speculative-num-draft-tokens "$DECODE_SPEC_DRAFT_TOKENS"
    )
fi
if [[ "$MIXED_HICACHE" == 1 ]]; then
    engine_args+=(
        --enable-hierarchical-cache --hicache-ratio "$MIXED_HICACHE_RATIO"
        --hicache-write-policy "$MIXED_HICACHE_WRITE_POLICY"
        --hicache-io-backend "$MIXED_HICACHE_IO_BACKEND"
        --hicache-mem-layout "$MIXED_HICACHE_MEM_LAYOUT"
    )
    [[ "$MIXED_HICACHE_SIZE" != 0 ]] && engine_args+=(--hicache-size "$MIXED_HICACHE_SIZE")
fi
[[ "${AITER_ALLREDUCE_FUSION:-1}" == 1 ]] && engine_args+=(--enable-aiter-allreduce-fusion)
(( MIXED_TOKENIZER_WORKERS > 1 )) && engine_args+=(--tokenizer-worker-num "$MIXED_TOKENIZER_WORKERS")
[[ -n "${MIXED_EXTRA_ARGS:-}" ]] && { read -r -a extra <<< "$MIXED_EXTRA_ARGS"; engine_args+=("${extra[@]}"); }
# Burn pids so ranks do not share pids with other tenants' containers on this node
# (ROCr IPC binds the abstract socket "\0hsa<pid>" in the shared host netns).
if [[ "$pid_offset" != 0 ]]; then
    engine_args=(bash -c 'i=0; while [ $i -lt '"$pid_offset"' ]; do ( : ); i=$((i+1)); done; exec "$@"' _ "${engine_args[@]}")
fi
echo "$instance ip=$node_ip port=$engine_port GPUs=$gpus TP=$MIXED_TP DP=$MIXED_DP DPA=$MIXED_DPA"
printf ' + %q' "${docker_args[@]}" "$IMAGE" "${engine_args[@]}"; echo
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0
"${docker_args[@]}" "$IMAGE" "${engine_args[@]}"
if [[ -n "${SERVER_LOG:-}" ]]; then
    mkdir -p "$(dirname "$SERVER_LOG")"
    : >"$SERVER_LOG"
    setsid nohup bash "$BENCH_DIR/follow_log.sh" "$SERVER_LOG" "$container" >/dev/null 2>&1 </dev/null &
    echo "server log: $SERVER_LOG"
fi
