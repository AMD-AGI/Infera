#!/bin/bash
# GLM-5.2-MXFP4 cross-node PD leg over mooncake RDMA, DP-attention on both legs,
# launched through the **infera wrapper** so kv-aware routing and kvd can be switched on.
#
# Derived verbatim from the verified 4/4 recipe
# a SEPARATE, earlier deliverable: glm5.2.mxfp4.packup_20260727 (its
# 07_pd_mooncake_dpa_sweep/scripts/pd_leg_dpa.sh). That numbering is unrelated
# to this folder's; nothing here depends on that packup being present.
# The ONLY deliberate deltas:
#   1. entry point `python3 -m infera.engine.sglang` instead of `sglang.launch_server`
#      (kvaware/kvd wiring lives in the infera wrapper; 07 bypasses it entirely),
#   2. etcd discovery + advertise-host so the infera router can pair the legs,
#   3. KVAWARE / KVD switches (the thing under test).
# Everything else — DSA-ROCm envs, mooncake RDMA envs, DPA args, gmu, chunk, ctx —
# is carried over byte-for-byte from that verified recipe.
#
# MERGE DELTA (work.merge_20260731): MTP and the custom-all-reduce switch are
# added from the converged PD+DPA+MTP kit's pd_leg_exp.sh, both defaulting OFF so
# that with MTP=0 this script produces the SAME command line as the kvaware/kvd
# baseline it came from -- G0 has to be a byte-level replay of that result before
# G1 changes anything.
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
# --- MTP (EAGLE spec-dec). Decode leg only unless PREFILL_MTP=1. ------------
# From the converged kit: every deadlock measured was under decode-only MTP.
# Default OFF so G0 replays the kvaware/kvd baseline exactly.
MTP="${MTP:-0}"
# The aiter custom all-reduce kernel deadlocks on gfx942/gfx950 during EAGLE
# verify at high concurrency (sglang GLM-5.1 cookbook; #28815/#31071/PR #31478).
# Kept OUTSIDE the MTP block on purpose: folding it in would make "MTP on vs off"
# a two-variable comparison. Default here follows MTP so G0 keeps the baseline's
# command line byte-identical; set CUSTOM_AR explicitly to override either way.
CUSTOM_AR="${CUSTOM_AR:-$([ "$MTP" = "1" ] && echo 0 || echo 1)}"
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
#
# BENCH DELTA (work.bench_20260801): MC_GID_INDEX is DISCOVERED, not hardcoded.
#
# The kit hardcoded MC_GID_INDEX=1, which is true on chi2879 and FALSE on
# chi2867. Both nodes expose two RoCE v2 GIDs per ionic port -- a link-local
# fe80:: one and a routable fd93:: one -- but at different indices:
#
#     chi2879  idx0 fe80::(link-local)  idx1 fd93::(routable)
#     chi2867  idx0 fe80::(link-local)  idx1 EMPTY  idx2 fd93::(routable)
#
# Consistent across all 8 NICs on each node. With the hardcoded 1, chi2867 reads
# an empty GID slot and every device fails:
#     rdma_context.cpp:1132 GID is NULL, please check your GID index
#     rdma_context.cpp:200  Failed to open device ionic_N
#     -> "Mooncake Transfer Engine initialization failed" on all 8 DP ranks,
#        scheduler dies during init, leg never reports ready.
# The link-local idx0 is not a workaround: it is not routable across the fabric
# and crashes MoRI at ionic.cpp:414 (see the slurm-cluster skill's rdma.md).
#
# So: pick the first index whose GID is neither empty nor fe80::. Falls back to
# the hardcoded value only if discovery finds nothing, so behaviour on a node
# where discovery fails is unchanged rather than silently different.
_gid_discover() {
  local d idx g
  d=$(echo "$IB_DEVICES" | cut -d, -f1)
  for idx in 0 1 2 3 4 5 6 7; do
    g=$(cat "/sys/class/infiniband/$d/ports/1/gids/$idx" 2>/dev/null) || continue
    case "$g" in
      ""|0000:0000:0000:0000:0000:0000:0000:0000|fe80*) continue ;;
      *) echo "$idx"; return 0 ;;
    esac
  done
  return 1
}
MC_GID_INDEX="${MC_GID_INDEX:-$(_gid_discover || echo 1)}"
export MC_GID_INDEX MC_DISABLE_HIP_TRANSPORT=1
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
# EP_DECOUPLE: --ep-size is passed UNCONDITIONALLY, not only under DP-attention.
# Rationale: the DPA-off latency experiment must vary attention sharding ALONE.
# Left inside the DPA branch, DPA=0 drops the flag and sglang resolves
# ep_size 8 -> 1, collapsing MoE expert-parallelism too -- a second variable
# that would make the measured delta unattributable. With DPA=1 this emits the
# exact same flag as before, so the DPA-on command line is unchanged.
DP_ARGS+=(--ep-size "$TP")
if [ "$DPA" = "1" ]; then
  DP_ARGS+=(--dp-size "$TP" --enable-dp-attention)
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
# BENCH DELTA (work.bench_20260801): --enable-cache-report populates
# usage.prompt_tokens_details.cached_tokens. Without it BOTH benches read a
# cache-hit rate of 0 -- sglang bench_serving's --cache-report column and the
# agentic driver's observed hit rate -- so Case A's 88-90% target becomes
# uncheckable. Default ON; CACHE_REPORT=0 to drop it.
[ "${CACHE_REPORT:-1}" = "1" ] && EXTRA_ARGS+=(--enable-cache-report)

# ---- MTP: EAGLE steps=3 (decode leg by default; PREFILL_MTP=1 adds prefill) --
MTP_ARGS=()
if [ "$MTP" = "1" ] && { [ "$ROLE" = "decode" ] || [ "${PREFILL_MTP:-0}" = "1" ]; }; then
  MTP_ARGS=(--speculative-algorithm EAGLE --speculative-num-steps "${SPEC_STEPS:-3}" \
            --speculative-eagle-topk 1 --speculative-num-draft-tokens "${SPEC_DRAFT:-4}" \
            --num-reserved-decode-tokens "${RESERVED_TOK:-256}")
fi

CAR_ARGS=()
[ "$CUSTOM_AR" = "1" ] || CAR_ARGS+=(--disable-custom-all-reduce)

echo "[glm52-kvexp] role=$ROLE ip=$MY_IP:$PORT nic=$NIC tp=$TP gpus=$GPUS dpa=$DPA mtp=$MTP car=$CUSTOM_AR kvaware=$KVAWARE kvd=$KVD gmu=$GMU chunk=$CHUNK ctx=$CTX ib=$IB_DEVICES -> $LOG"
HIP_VISIBLE_DEVICES="$GPUS" python3 -m infera.engine.sglang \
  --model-path "$MODEL" --served-model-name "$SERVED" --tp-size "$TP" --trust-remote-code \
  --host "$MY_IP" --port "$PORT" \
  --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static "$GMU" --context-length "$CTX" \
  --chunked-prefill-size "$CHUNK" --cuda-graph-max-bs "$CUDA_GRAPH_BS" \
  --max-running-requests "$MAX_RUNNING" --watchdog-timeout 3600 \
  "${INFERA_ARGS[@]}" "${DP_ARGS[@]}" "${MTP_ARGS[@]}" "${CAR_ARGS[@]}" \
  "${ROLE_ARGS[@]}" "${EXTRA_ARGS[@]}" > "$LOG" 2>&1
