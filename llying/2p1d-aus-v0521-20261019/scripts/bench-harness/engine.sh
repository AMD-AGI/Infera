#!/usr/bin/env bash
# Purpose: Start one role-specific SGLang Prefill or Decode worker container.
# Usage: ./engine.sh ROLE INSTANCE IP GPUS ENGINE_PORT BOOTSTRAP_PORT \
#   KV_EVENT_PORT SNAPSHOT_PORT CONTAINER ETCD_ENDPOINT [KEY=VALUE ...]
# Artifacts: detached worker container, persistent AITER JIT cache, optional log.
# Artifact paths: AITER cache defaults to /tmp/aiter-jit-<uid>/<image-id>;
#   SERVER_LOG selects the persistent log path.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

(( $# >= 10 )) || {
    echo "usage: $0 ROLE INSTANCE IP GPUS ENGINE_PORT BOOTSTRAP_PORT KV_EVENT_PORT SNAPSHOT_PORT CONTAINER ETCD_ENDPOINT [KEY=VALUE]" >&2
    exit 2
}
role="$1"; instance="$2"; node_ip="$3"; gpus="$4"
engine_port="$5"; bootstrap_port="$6"; kv_event_port="$7"; snapshot_port="$8"
container="$9"; etcd_endpoint="${10}"
shift 10
[[ "$role" == prefill || "$role" == decode ]] ||
    { echo "invalid role: $role" >&2; exit 2; }
[[ "$container" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] ||
    { echo "unsafe container name: $container" >&2; exit 2; }

for assignment in "$@"; do
    [[ "$assignment" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] ||
        { echo "expected KEY=VALUE, got '$assignment'" >&2; exit 2; }
    export "$assignment"
done
CONFIG="${CONFIG:-$DIR/config.sh}"
[[ -r "$CONFIG" ]] || { echo "config is not readable: $CONFIG" >&2; exit 1; }
set -a
source "$CONFIG"
set +a
: "${IMAGE:?set IMAGE}"
: "${MODEL:?set MODEL}"
: "${SERVED_MODEL:?set SERVED_MODEL}"
: "${RDMA_DEVICE:?set RDMA_DEVICE}"

case "$role" in
    prefill)
        tp="$PREFILL_TP"; ep="$PREFILL_EP"; dp="$PREFILL_DP"; dpa="$PREFILL_DPA"
        chunk="$PREFILL_CHUNK_SIZE"; max_running="$PREFILL_MAX_RUNNING"
        graph_bs="$PREFILL_GRAPH_MAX_BS"; mem_fraction="$PREFILL_MEM_FRACTION"
        graph_bs_option="--cuda-graph-max-bs-prefill"
        hicache="$PREFILL_HICACHE"; hicache_ratio="$PREFILL_HICACHE_RATIO"
        hicache_policy="$PREFILL_HICACHE_WRITE_POLICY"
        hicache_backend="$PREFILL_HICACHE_IO_BACKEND"
        hicache_layout="$PREFILL_HICACHE_MEM_LAYOUT"
        scratch_reclaim="$PREFILL_HSA_NO_SCRATCH_RECLAIM"
        extra_args="${PREFILL_EXTRA_ARGS:-}"
        extra_env="${PREFILL_EXTRA_ENV:-}"
        ;;
    decode)
        tp="$DECODE_TP"; ep="$DECODE_EP"; dp="$DECODE_DP"; dpa="$DECODE_DPA"
        chunk="$DECODE_CHUNK_SIZE"; max_running="$DECODE_MAX_RUNNING"
        graph_bs="$DECODE_GRAPH_MAX_BS"; mem_fraction="$DECODE_MEM_FRACTION"
        graph_bs_option="--cuda-graph-max-bs-decode"
        hicache="$DECODE_HICACHE"; hicache_ratio="$DECODE_HICACHE_RATIO"
        hicache_policy="$DECODE_HICACHE_WRITE_POLICY"
        hicache_backend="$DECODE_HICACHE_IO_BACKEND"
        hicache_layout="$DECODE_HICACHE_MEM_LAYOUT"
        scratch_reclaim="$DECODE_HSA_NO_SCRATCH_RECLAIM"
        extra_args="${DECODE_EXTRA_ARGS:-}"
        extra_env="${DECODE_EXTRA_ENV:-}"
        ;;
esac
# Per-instance overrides (dashes as underscores), e.g. a larger HiCache on the prefill
# that has a node's host memory to itself. HiCache size is GB per rank (> 0 overrides the
# ratio); the register chunk is SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB, which must cover
# the whole pool in one hipHostRegister (pools > 256 GiB fault otherwise). agentx_env.py
# checks the live containers against the same variables.
inst_key="${instance//-/_}"
size_var="HICACHE_SIZE_$inst_key"; chunk_var="HICACHE_HOST_REGISTER_CHUNK_GB_$inst_key"
args_var="EXTRA_ARGS_$inst_key"; env_var="EXTRA_ENV_$inst_key"
if [[ "$role" == prefill ]]; then
    hicache_size="${!size_var:-${PREFILL_HICACHE_SIZE:-0}}"
    hicache_chunk_gb="${!chunk_var:-${HICACHE_HOST_REGISTER_CHUNK_GB:-}}"
else
    hicache_size="${!size_var:-${DECODE_HICACHE_SIZE:-0}}"
    hicache_chunk_gb=""
fi
[[ "$hicache_size" =~ ^[0-9]+$ ]] || { echo "invalid HiCache size: $hicache_size" >&2; exit 2; }
[[ -z "$hicache_chunk_gb" || "$hicache_chunk_gb" =~ ^[1-9][0-9]*$ ]] ||
    { echo "invalid HiCache register chunk: $hicache_chunk_gb" >&2; exit 2; }
extra_args="${extra_args:+$extra_args }${!args_var:-}"
extra_env="${extra_env:+$extra_env }${!env_var:-}"
for integer in tp ep dp chunk max_running graph_bs; do
    [[ "${!integer}" =~ ^[1-9][0-9]*$ ]] ||
        { echo "$integer must be a positive integer" >&2; exit 2; }
done
[[ "$dpa" == 0 || "$dpa" == 1 ]] || { echo "DPA must be 0 or 1" >&2; exit 2; }
[[ "$hicache" == 0 || "$hicache" == 1 ]] ||
    { echo "HiCache must be 0 or 1" >&2; exit 2; }
[[ "$scratch_reclaim" == 0 || "$scratch_reclaim" == 1 ]] ||
    { echo "HSA_NO_SCRATCH_RECLAIM must be 0 or 1" >&2; exit 2; }
if [[ "$role" == decode && "$hicache" == 1 && "$DECODE_MTP" == 1 ]]; then
    echo "Decode MTP and HiCache cannot both be enabled" >&2
    exit 2
fi

nic="${DATA_NIC:-}"
if [[ -z "$nic" ]]; then
    nic="$(ip -o -4 addr show | awk -v wanted="$node_ip" '$4 ~ ("^" wanted "/") {print $2; exit}')"
fi
[[ -n "$nic" ]] || { echo "cannot resolve NIC for $node_ip" >&2; exit 1; }

# Node-scoped caches keep parallel same-node deployments from sharing JIT state.
# engine.sh runs on the worker node itself, so the key comes from its hostname.
node_key="$(hostname -s)"; node_key="${node_key##*-}"
[[ -n "$node_key" ]] || { echo "cannot derive node key from hostname" >&2; exit 2; }
# IB_DEVICES_<node key> (e.g. IB_DEVICES_135) replaces RDMA_DEVICE and MC_TE_FILTERS
# on that node only, for nodes with a dead or absent rail.
ib_override="IB_DEVICES_${node_key}"
if [[ -n "${!ib_override:-}" ]]; then
    RDMA_DEVICE="${!ib_override}"
    MC_TE_FILTERS="${!ib_override}"
fi
# DRY_RUN=1 renders the docker + engine command lines and exits before starting
# anything. It tolerates a not-yet-present image and an unfilled EXPECTED_IMAGE_ID
# (warns instead of failing) so the command can be reviewed before the build lands.
if ! image_id="$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null)"; then
    [[ "${DRY_RUN:-0}" == 1 ]] || { echo "image not present: $IMAGE" >&2; exit 1; }
    echo "DRY_RUN: image $IMAGE not present on $(hostname); using a placeholder ID" >&2
    image_id="sha256:$(printf '0%.0s' {1..64})"
fi
if [[ "$image_id" != "$EXPECTED_IMAGE_ID" ]]; then
    [[ "${DRY_RUN:-0}" == 1 ]] || { echo "unexpected image: $image_id" >&2; exit 1; }
    echo "DRY_RUN: image ID $image_id != EXPECTED_IMAGE_ID '${EXPECTED_IMAGE_ID:-}'" >&2
fi
image_key="${image_id#sha256:}"
[[ "$image_key" =~ ^[0-9a-f]{64}$ ]] ||
    { echo "cannot derive image ID for AITER cache: $image_id" >&2; exit 1; }
aiter_jit_cache="${AITER_JIT_CACHE_ROOT:-/tmp/aiter-jit-$(id -u)}/$image_key"
[[ "${DRY_RUN:-0}" == 1 ]] || mkdir -p "$aiter_jit_cache"

# yihou v0521dbg: NODE_RW_ROOT = node-local writable root for everything the node writes
# (tmp, triton/hf/xdg caches; AITER and SERVER_LOG via the config/driver). Needed where
# /shared_nfs is mounted read-only (136/138). Empty = the old $WS behaviour.
rw="${NODE_RW_ROOT:-$WS}"
[[ "$rw" == /* ]] || { echo "NODE_RW_ROOT must be absolute" >&2; exit 2; }
docker_args=(
    docker run -d --init --name "$container"
    --network host --ipc host --shm-size "${SHM_SIZE:-32g}"
    --device /dev/kfd --device /dev/dri --device /dev/infiniband
    --group-add video --group-add render
    --cap-add SYS_PTRACE --cap-add IPC_LOCK
    --security-opt seccomp=unconfined
    --ulimit memlock=-1:-1 --ulimit nofile=65536:65536
    -v "$WS:$WS${NODE_RW_ROOT:+:ro}"   # yihou: read side only when NODE_RW_ROOT is set
    -v "$MODEL:$MODEL:ro"
    -v "$aiter_jit_cache:/aiter-jit"
    -e AITER_JIT_DIR=/aiter-jit
    --tmpfs "$rw/tmp/$node_key/$instance:rw,exec,size=${ENGINE_TMPFS_SIZE:-8g}"
    -e "TMPDIR=$rw/tmp/$node_key/$instance"
    -e "TRITON_CACHE_DIR=$rw/cache/$node_key/triton/$instance"
    -e "HF_HOME=$rw/cache/$node_key/hf/$instance"
    -e "XDG_CACHE_HOME=$rw/cache/$node_key/xdg/$instance"
    -e PYTHONDONTWRITEBYTECODE=1
)
# yihou v0521dbg: same-node IPC. ROCr's default IPC binds the abstract socket "\0hsa<pid>"
# (namespace pid) in the shared host netns; two private PID namespaces collide -> the
# "RCCL race" (hipIpcGetMemHandle invalid argument). --pid host makes the pids unique.
# Both knobs come from the shared config, so both legs always get the same value.
[[ "${ENGINE_PID_HOST:-0}" == 1 && -n "${YIHOU_JOB:-}" ]] &&
    { echo "ENGINE_PID_HOST=1 refused on spur node (spur-authz denies --pid=host)" >&2; exit 2; }
[[ "${ENGINE_PID_HOST:-0}" == 1 ]] && docker_args+=(--pid host)
[[ "${IPC_MODE_LEGACY:-0}" == 1 ]] && docker_args+=(-e HSA_ENABLE_IPC_MODE_LEGACY=1)
# yihou v0521dbg (R045): per-leg NUMA placement. {PREFILL,DECODE}_CPUSET_{CPUS,MEMS} -> docker
# --cpuset-cpus/--cpuset-mems; ENGINE_SET_CPU_AFFINITY -> SGLANG_SET_CPU_AFFINITY on both legs
# (image default 1 pins scheduler ranks by local gpu_id). All empty by default = no flag/env.
if [[ "$role" == prefill ]]; then
    cpuset_cpus="${PREFILL_CPUSET_CPUS:-}"; cpuset_mems="${PREFILL_CPUSET_MEMS:-}"
else
    cpuset_cpus="${DECODE_CPUSET_CPUS:-}"; cpuset_mems="${DECODE_CPUSET_MEMS:-}"
fi
for v in cpuset_cpus cpuset_mems; do
    [[ -z "${!v}" || "${!v}" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]] ||
        { echo "invalid $role $v: ${!v}" >&2; exit 2; }
done
[[ -n "$cpuset_cpus" ]] && docker_args+=(--cpuset-cpus "$cpuset_cpus")
[[ -n "$cpuset_mems" ]] && docker_args+=(--cpuset-mems "$cpuset_mems")
if [[ -n "${ENGINE_SET_CPU_AFFINITY:-}" ]]; then
    [[ "$ENGINE_SET_CPU_AFFINITY" == 0 || "$ENGINE_SET_CPU_AFFINITY" == 1 ]] ||
        { echo "ENGINE_SET_CPU_AFFINITY must be 0 or 1" >&2; exit 2; }
    docker_args+=(-e "SGLANG_SET_CPU_AFFINITY=$ENGINE_SET_CPU_AFFINITY")
fi
if [[ -n "${NODE_RW_ROOT:-}" ]]; then
    [[ "${DRY_RUN:-0}" == 1 ]] || mkdir -p "$NODE_RW_ROOT"
    docker_args+=(-v "$NODE_RW_ROOT:$NODE_RW_ROOT")
fi
# SOURCE_MODEL is only needed when MODEL is a symlink overlay onto it; docker
# rejects a duplicate mount point when the two are the same path.
[[ "$SOURCE_MODEL" == "$MODEL" ]] || docker_args+=(-v "$SOURCE_MODEL:$SOURCE_MODEL:ro")
while IFS=$'\t' read -r source destination; do
    [[ -n "$source" && -r "$source" ]] || { echo "invalid source mount: $source" >&2; exit 1; }
    docker_args+=(-v "$source:$destination:ro")
done < "${YIHOU_MOUNTS:-$WS/patches/mounts.yihou.tsv}"
# yihou v0521dbg (R050): role-only extra mount rows (same TSV format), e.g. decode-only patches.
# {PREFILL,DECODE}_MOUNTS_ADD = TSV path; empty (default) = none.
if [[ "$role" == prefill ]]; then mounts_add="${PREFILL_MOUNTS_ADD:-}"; else mounts_add="${DECODE_MOUNTS_ADD:-}"; fi
if [[ -n "$mounts_add" ]]; then
    [[ -r "$mounts_add" ]] || { echo "invalid $role MOUNTS_ADD: $mounts_add" >&2; exit 1; }
    while IFS=$'\t' read -r source destination; do
        [[ -z "$source" ]] && continue
        [[ -r "$source" ]] || { echo "invalid source mount: $source" >&2; exit 1; }
        docker_args+=(-v "$source:$destination:ro")
    done < "$mounts_add"
fi
if [[ "$role" == prefill && -n "${YIHOU_ASM_TRACE_DIR:-}" ]]; then
    [[ "$YIHOU_ASM_TRACE_DIR" == "$WS"/* ]] || exit 2
    docker_args+=(
        -v "$WS/source/aiter-diagnostic/patched/tuned_gemm.py:/sgl-workspace/aiter/aiter/tuned_gemm.py:ro"
        -e "YIHOU_ASM_TRACE_DIR=$YIHOU_ASM_TRACE_DIR"
    )
fi
if [[ -n "${HOST_RDMA_LIB:-}" ]]; then
    [[ -r "$HOST_RDMA_LIB" ]] ||
        { echo "RDMA provider is not readable: $HOST_RDMA_LIB" >&2; exit 1; }
    docker_args+=(-v "$HOST_RDMA_LIB:${HOST_RDMA_MOUNT:-/host-libionic/libionic.so}:ro")
fi
docker_args+=(
    -e "HIP_VISIBLE_DEVICES=$gpus"
    -e "SGLANG_HOST_IP=$node_ip" -e "HOST_IP=$node_ip"
    -e "SGLANG_LOCAL_IP_NIC=$nic" -e "GLOO_SOCKET_IFNAME=$nic"
    -e "NCCL_SOCKET_IFNAME=$nic"
    -e "MC_GID_INDEX=$MC_GID_INDEX" -e "MC_TE_FILTERS=$MC_TE_FILTERS"
    -e "MC_DISABLE_HIP_TRANSPORT=$MC_DISABLE_HIP_TRANSPORT"
    -e "MC_ENABLE_DEST_DEVICE_AFFINITY=$MC_ENABLE_DEST_DEVICE_AFFINITY"
    -e "MOONCAKE_DISABLE_HIP_DMABUF=$MOONCAKE_DISABLE_HIP_DMABUF"
    -e "RDMAV_FORK_SAFE=$RDMAV_FORK_SAFE"
    -e "NCCL_IB_DISABLE=$NCCL_IB_DISABLE" -e NCCL_IGNORE_CPU_AFFINITY=1
    -e "HSA_NO_SCRATCH_RECLAIM=$scratch_reclaim"
    -e SGLANG_USE_AITER=1
    -e "SGLANG_OPT_USE_TOPK_V2=$SGLANG_OPT_USE_TOPK_V2"
    -e "SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=$SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK"
    -e PYTHONNOUSERSITE=1
    -e "SGLANG_TIMEOUT_KEEP_ALIVE=$SGLANG_TIMEOUT_KEEP_ALIVE"
    -e AITER_USE_FLYDSL_MOE_SORTING=1 -e SAFETENSORS_FAST_GPU=1
    -e HIP_FORCE_DEV_KERNARG=1 -e PYTHONHASHSEED=0
    -e "SGLANG_DP_USE_GATHERV=$dpa"
    -e "INFERA_ENGINE_READY_TIMEOUT=$READY_TIMEOUT"
    -e "INFERA_NODEPORT_RANGE=$INFERA_NODEPORT_RANGE"
    -e "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=${BOOTSTRAP_TIMEOUT:-1800}"
    -e "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=${BOOTSTRAP_TIMEOUT:-1800}"
    -e "SGLANG_ENABLE_FAILED_SESSION_PROBE=$SGLANG_ENABLE_FAILED_SESSION_PROBE"
    -e "SGLANG_FAILED_SESSION_PROBE_INTERVAL_S=$SGLANG_FAILED_SESSION_PROBE_INTERVAL_S"
)
[[ -n "$hicache_chunk_gb" ]] && docker_args+=(-e "SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=$hicache_chunk_gb")
# Role-scoped escape hatch for one-off container env during bring-up and
# debugging, so a config that differs from the default recipe does not need this
# script edited. Space-separated KEY=VALUE items; empty by default, so unused it
# is a no-op.
if [[ -n "${extra_env:-}" ]]; then
    read -r -a extra_env_items <<< "$extra_env"
    for item in "${extra_env_items[@]}"; do
        [[ "$item" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] ||
            { echo "invalid EXTRA_ENV item: $item" >&2; exit 2; }
        docker_args+=(-e "$item")
    done
fi
# yihou v0521dbg: ROCm debug agent, lean mode (names the faulting kernel, also under
# CUDA-graph replay; ~1.00x measured). No --output: the report goes to stderr, i.e. into
# the server log next to "Memory access fault", with docker timestamps. Dedicated -e
# (not EXTRA_ENV) so a future multi-word option value survives.
if [[ "${DEBUG_AGENT:-0}" == 1 ]]; then
    docker_args+=(
        -e "HSA_TOOLS_LIB=${DEBUG_AGENT_LIB:-/opt/rocm/lib/librocm-debug-agent.so.2}"
        -e HSA_ENABLE_DEBUG=1
        -e "ROCM_DEBUG_AGENT_OPTIONS=${DEBUG_AGENT_OPTIONS:---disable-linux-signals}"
    )
fi
if [[ "$role" == decode && -n "${DECODE_SIMULATE_ACC_LEN:-}" ]]; then
    docker_args+=(
        -e "SGLANG_SIMULATE_ACC_LEN=$DECODE_SIMULATE_ACC_LEN"
        -e SGLANG_SIMULATE_ACC_METHOD=match-expected
        -e SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
    )
fi

engine_args=(
    python3 -m infera.engine.sglang
    --model-path "$MODEL" --served-model-name "$SERVED_MODEL"
    --host "$node_ip" --port "$engine_port" --advertise-host "$node_ip"
    --etcd-endpoint "$etcd_endpoint" --discovery-backend etcd
    --request-transport http --kv-event-transport zmq
    --tp-size "$tp" --ep-size "$ep" --trust-remote-code
    --dsa-prefill-backend "${DSA_PREFILL_BACKEND:-tilelang}"
    --dsa-decode-backend "${DSA_DECODE_BACKEND:-tilelang}"
    --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}"
    --mem-fraction-static "$mem_fraction"
    --chunked-prefill-size "$chunk"
    "$graph_bs_option" "$graph_bs"
    --max-running-requests "$max_running"
    --watchdog-timeout "${WATCHDOG_TIMEOUT:-3600}"
    --reasoning-parser "${REASONING_PARSER:-glm45}"
    --tool-call-parser "${TOOL_CALL_PARSER:-glm47}"
    --enable-fused-qk-norm-rope
    --enable-cache-report --enable-metrics
    --disaggregation-mode "$role"
    --disaggregation-transfer-backend "$KV_P2P_TRANSFER"
)
[[ "$dp" -gt 1 ]] && engine_args+=(--dp-size "$dp")
[[ "$dpa" == 1 ]] && engine_args+=(--enable-dp-attention)
[[ -n "${DSA_TOPK_BACKEND:-}" ]] &&
    engine_args+=(--dsa-topk-backend "$DSA_TOPK_BACKEND")
[[ -n "${JSON_MODEL_OVERRIDE_ARGS:-}" ]] &&
    engine_args+=(--json-model-override-args "$JSON_MODEL_OVERRIDE_ARGS")
[[ -n "$RDMA_DEVICE" ]] && engine_args+=(--disaggregation-ib-device "$RDMA_DEVICE")
if [[ "$role" == prefill ]]; then
    engine_args+=(--disaggregation-bootstrap-port "$bootstrap_port")
    if [[ "$ENABLE_KV_AWARE" == 1 ]]; then
        engine_args+=(
            --enable-kv-events --kv-events on
            --kv-events-bind "tcp://0.0.0.0:$kv_event_port"
            --kv-snapshot-port "$snapshot_port"
        )
    else
        engine_args+=(--no-enable-kv-events --kv-events off)
    fi
else
    engine_args+=(--no-enable-kv-events --kv-events off)
fi
if [[ "$role" == decode && "$DECODE_MTP" == 1 ]]; then
    engine_args+=(
        --speculative-algorithm EAGLE
        --speculative-num-steps "$DECODE_SPEC_STEPS"
        --speculative-eagle-topk "$DECODE_SPEC_TOPK"
        --speculative-num-draft-tokens "$DECODE_SPEC_DRAFT_TOKENS"
    )
fi
if [[ "$hicache" == 1 ]]; then
    engine_args+=(
        --enable-hierarchical-cache --hicache-ratio "$hicache_ratio"
        --hicache-write-policy "$hicache_policy"
        --hicache-io-backend "$hicache_backend"
        --hicache-mem-layout "$hicache_layout"
    )
    [[ "$hicache_size" != 0 ]] && engine_args+=(--hicache-size "$hicache_size")
fi
# AITER all-reduce fusion. Emitted by default, so this is a no-op unless a config
# sets AITER_ALLREDUCE_FUSION=0. Made conditional because sglang's CLI exposes
# only the positive `--enable-aiter-allreduce-fusion` (store_true, no `--no-`
# form), so there is otherwise no way to A/B it; the only code path that turns it
# off is --enable-deterministic-inference, which changes much more besides.
[[ "${AITER_ALLREDUCE_FUSION:-1}" == 1 ]] &&
    engine_args+=(--enable-aiter-allreduce-fusion)
[[ -n "${CONTEXT_LENGTH:-}" ]] && engine_args+=(--context-length "$CONTEXT_LENGTH")
# Role-scoped escape hatch for one-off engine flags during bring-up and
# debugging, so an experiment does not have to edit this script. Word-split on
# whitespace; quote nothing fancier than that. Empty by default.
if [[ -n "${extra_args:-}" ]]; then
    read -r -a extra_arg_items <<< "$extra_args"
    engine_args+=("${extra_arg_items[@]}")
fi

# yihou v0521dbg (R022): DECODE_PID_OFFSET=K>0 burns K pids inside the decode container before
# exec'ing the engine, so decode ranks get pids >= K. The CLR hipIpcOpenMemHandle guard returns
# 201 when exporter pid == importer pid across private PID namespaces (e1a: DP0 973 == 973),
# and --pid host is denied by spur-authz on spur nodes. The image ENTRYPOINT
# (infera-inject-host-ionic) ends in `exec "$@"`, so the wrapper runs under it and under --init.
# PID_OFFSET_<instance> (dashes as underscores) applies the same wrapper to any
# instance, so two containers on one node (e.g. two prefills) get disjoint rank
# PIDs; DECODE_PID_OFFSET remains the decode default.
pid_override="PID_OFFSET_${instance//-/_}"
pid_offset="${!pid_override:-}"
[[ -n "$pid_offset" || "$role" != decode ]] || pid_offset="${DECODE_PID_OFFSET:-0}"
pid_offset="${pid_offset:-0}"
if [[ "$pid_offset" != 0 ]]; then
    [[ "$pid_offset" =~ ^[1-9][0-9]*$ ]] || { echo "$pid_override must be a positive integer" >&2; exit 2; }
    engine_args=(bash -c 'i=0; while [ $i -lt '"$pid_offset"' ]; do ( : ); i=$((i+1)); done; exec "$@"' _ "${engine_args[@]}")
fi
echo "$instance role=$role ip=$node_ip port=$engine_port GPUs=$gpus TP=$tp EP=$ep DP=$dp DPA=$dpa HiCache=$hicache"
echo "AITER JIT cache: $aiter_jit_cache"
printf ' + %q' "${docker_args[@]}" "$IMAGE" "${engine_args[@]}"
echo
[[ "${DRY_RUN:-0}" == 1 ]] && { echo "DRY_RUN: not starting $container"; exit 0; }
"${docker_args[@]}" "$IMAGE" "${engine_args[@]}"
if [[ -n "${SERVER_LOG:-}" ]]; then
    mkdir -p "$(dirname "$SERVER_LOG")"
    : >"$SERVER_LOG"
    if [[ "${FOLLOW_ON_NODE:-1}" == 1 ]]; then
        setsid nohup bash "$DIR/follow_log.sh" "$SERVER_LOG" "$container" >/dev/null 2>&1 </dev/null &
    else
        # The login host follows it (scripts/follow_supervisor.sh picks up this marker).
        printf '%s\t%s\n' "$(hostname -s)" "$container" > "$SERVER_LOG.follow"
    fi
    echo "server log: $SERVER_LOG"
fi
