#!/bin/bash
# GLM-5.2-MXFP4 cross-node PD leg over mooncake RDMA, DP-attention on both legs,
# launched through the **infera wrapper** so kv-aware routing and kvd can be switched on.
#
# Derived verbatim from the verified 4/4 recipe
# glm5.2.mxfp4.packup_20260727/07_pd_mooncake_dpa_sweep/scripts/pd_leg_dpa.sh.
# The ONLY deliberate deltas:
#   1. entry point `python3 -m infera.engine.sglang` instead of `sglang.launch_server`
#      (kvaware/kvd wiring lives in the infera wrapper; 07 bypasses it entirely),
#   2. etcd discovery + advertise-host so the infera router can pair the legs,
#   3. KVAWARE / KVD switches (the thing under test).
# Everything else — DSA-ROCm envs, mooncake RDMA envs, DPA args, gmu, chunk, ctx — is 07's.
set -u
ROLE="${ROLE:?ROLE=prefill|decode}"
MY_IP="${MY_IP:?MY_IP=data-plane rail IP}"
ETCD_IP="${ETCD_IP:?}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
SERVED="${SERVED:-glm5.2-mxfp4}"
PORT="${PORT:-30000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
ETCD_PORT="${ETCD_PORT:-2379}"
TP="${TP:-8}"
# Which GPUs this leg owns. Default = the whole node (TP8). Set BASE_GPU to pack
# two TP4 workers onto one node (needed to give the kv-aware scorer >=2 workers
# per role to choose between).
BASE_GPU="${BASE_GPU:-0}"
GPUS="${GPUS:-$(seq -s, "$BASE_GPU" $((BASE_GPU + TP - 1)))}"
DPA="${DPA:-1}"
CTX="${CTX:-32768}"
MAX_RUNNING="${MAX_RUNNING:-2048}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-128}"
DELAYER="${DELAYER:-1}"
PREFILL_DELAY_MS="${PREFILL_DELAY_MS:-5000}"
DMABUF="${DMABUF:-0}"
MAX_TOTAL="${MAX_TOTAL:-}"
# --- knobs under test ---
KVAWARE="${KVAWARE:-1}"          # publish engine KV events (router kv-aware source)
KVD="${KVD:-1}"                  # wire the infera-kvd HiCacheStorage backend
KVD_SOCK="${KVD_SOCK:-/tmp/kvd/kvd.sock}"
# Absolute host pool. The default --hicache-ratio 2.0 sizes it off max_total_num_tokens
# and blew up to 355 GB *per DP rank* in the Qwen3 MVP. Keep it bounded.
HICACHE_GB="${HICACHE_GB:-16}"
KV_PUB_PORT="${KV_PUB_PORT:-5557}"   # infera's own publisher socket; default collides if shared
# infera's per-worker snapshot HTTP server. ANOTHER default (8801) that is
# identical on every worker -> two workers on one host fail with
# "[Errno 98] address already in use ('0.0.0.0', 8801)" AFTER the engine is
# already up, so the leg says "ready to roll" and then never registers in etcd.
KV_SNAP_PORT="${KV_SNAP_PORT:-8801}"
if [ "$ROLE" = "prefill" ]; then GMU="${GMU:-0.88}"; else GMU="${GMU:-0.85}"; fi
LOG="${LOG:-/mnt/vast/c_huggingface/glm52_kvexp/pd_${ROLE}_${PORT}.log}"
mkdir -p "$(dirname "$LOG")"

ISL="${ISL:-8192}"
if [ "$DPA" = "1" ]; then CHUNK="${CHUNK:-$((ISL * TP))}"; else CHUNK="${CHUNK:-8192}"; fi

IB_DEVICES=$(for d in /sys/class/infiniband/*; do
    [ -d "$d" ] || continue; n=$(basename "$d")
    s=$(cat "$d/ports/1/state" 2>/dev/null || echo "")
    drv=$(basename "$(readlink -f "$d/device/driver" 2>/dev/null || echo x)")
    [[ "$s" == *ACTIVE* && "$drv" == ionic ]] && echo "$n"
  done | sort -V | paste -sd,)
[ -z "$IB_DEVICES" ] && { echo "no active ionic NICs" >&2; exit 1; }

if [ "$DMABUF" = "1" ]; then export MOONCAKE_DISABLE_HIP_DMABUF=0; else export MOONCAKE_DISABLE_HIP_DMABUF=1; fi

# mooncake RDMA env — real RDMA across the two nodes, NOT MC_FORCE_TCP.
export MC_GID_INDEX=1 MC_DISABLE_HIP_TRANSPORT=1
unset MC_ENABLE_HIP_TRANSPORT
export RDMAV_FORK_SAFE=1
export NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1
export SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP"
NIC=$(ip -o -4 addr show | awk -v ip="$MY_IP" '$4 ~ ("^" ip "/") {print $2; exit}')
export SGLANG_LOCAL_IP_NIC="$NIC" GLOO_SOCKET_IFNAME="$NIC"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800

# GLM-5.2 DSA-ROCm recipe (mandatory on gfx950).
export SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0
export SGLANG_OPT_USE_TILELANG_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_USE_JIT_NORM=0
export SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1

[ "$DPA" = "1" ] && export SGLANG_DP_USE_GATHERV=1
# Stable block hashes -> stable kvd keys across restarts.
export PYTHONHASHSEED=0

ROLE_ARGS=(--disaggregation-mode "$ROLE" --disaggregation-transfer-backend mooncake \
           --disaggregation-ib-device "$IB_DEVICES")
[ "$ROLE" = "prefill" ] && ROLE_ARGS+=(--disaggregation-bootstrap-port "$BOOTSTRAP_PORT")

DP_ARGS=()
if [ "$DPA" = "1" ]; then
  DP_ARGS+=(--dp-size "$TP" --enable-dp-attention --ep-size "$TP")
  if [ "$ROLE" = "prefill" ] && [ "$DELAYER" = "1" ]; then
    DP_ARGS+=(--enable-prefill-delayer --prefill-delayer-max-delay-ms "$PREFILL_DELAY_MS")
  fi
fi

# infera wrapper args: etcd discovery so the infera router pairs the legs.
INFERA_ARGS=(--advertise-host "$MY_IP" --etcd-endpoint "$ETCD_IP:$ETCD_PORT"
             --discovery-backend etcd --request-transport http --kv-event-transport zmq)

# kv-aware: KV events default ON in the infera wrapper.
if [ "$KVAWARE" = "1" ]; then
  INFERA_ARGS+=(--kv-events-bind "tcp://0.0.0.0:$KV_PUB_PORT"
                --kv-snapshot-port "$KV_SNAP_PORT")
else
  INFERA_ARGS+=(--no-enable-kv-events)
fi

# kvd: probes the daemon, then appends --enable-hierarchical-cache +
# --hicache-storage-backend dynamic + the InferaKvdBackend extra-config.
[ "$KVD" = "1" ] && INFERA_ARGS+=(--infera-kvd-socket "$KVD_SOCK" --hicache-size "$HICACHE_GB")

EXTRA_ARGS=()
[ -n "$MAX_TOTAL" ] && EXTRA_ARGS+=(--max-total-tokens "$MAX_TOTAL")

echo "[glm52-kvexp] role=$ROLE ip=$MY_IP:$PORT nic=$NIC tp=$TP gpus=$GPUS dpa=$DPA kvaware=$KVAWARE kvd=$KVD gmu=$GMU chunk=$CHUNK ctx=$CTX ib=$IB_DEVICES -> $LOG"
HIP_VISIBLE_DEVICES="$GPUS" python3 -m infera.engine.sglang \
  --model-path "$MODEL" --served-model-name "$SERVED" --tp-size "$TP" --trust-remote-code \
  --host "$MY_IP" --port "$PORT" \
  --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static "$GMU" --context-length "$CTX" \
  --chunked-prefill-size "$CHUNK" --cuda-graph-max-bs "$CUDA_GRAPH_BS" \
  --max-running-requests "$MAX_RUNNING" --watchdog-timeout 3600 \
  "${INFERA_ARGS[@]}" "${DP_ARGS[@]}" "${ROLE_ARGS[@]}" "${EXTRA_ARGS[@]}" > "$LOG" 2>&1
