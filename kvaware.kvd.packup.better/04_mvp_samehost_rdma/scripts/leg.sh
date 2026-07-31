#!/bin/bash
# ---------------------------------------------------------------------------
# Experiment 04 — one infera-sglang PD leg, Qwen3-1.7B, single node.
#
# This is the round-r3 leg. Everything the previous two rounds found is now
# fixed, so BOTH legs come up and register:
#   * --hicache-size (absolute GB)        -- r1's host-pool blow-up
#   * legs' --port spaced 1000 apart      -- r1's derived-ZMQ collision
#   * patched free_tcp_port_block         -- r2's --kv-events-config collision
#   * each leg its own --kv-events-bind   -- 5557 / 5657, infera's OWN publisher
#
# What r3 exposes is the next layer down: the two legs are on ONE host, so the
# mooncake KV transfer has to loop back across RDMA rails (ionic_0 -> ionic_4),
# and this fabric will not do it. FORCE_TCP selects which side you see.
#
#   FORCE_TCP=0  -> real mooncake RDMA. Legs come up, completions return
#                   HTTP 500 "Failed to get kvcache from prefill instance".
#   FORCE_TCP=1  -> MC_FORCE_TCP=1, the correct-but-slower path. Completions
#                   succeed. (Their CONTENT is a separate matter — see the
#                   differential experiment.)
#
# Runs INSIDE the container. Entry point is `infera.engine.sglang` (NOT
# sglang.launch_server) because the kvaware/kvd wiring lives in the infera
# wrapper.
# ---------------------------------------------------------------------------
set -u
ROLE="${ROLE:?ROLE=prefill|decode}"
MY_IP="${MY_IP:?}"
ETCD_IP="${ETCD_IP:?}"
MODEL="${MODEL:?}"
SERVED="${SERVED:-qwen3}"
PORT="${PORT:-30000}"
BASE_GPU="${BASE_GPU:-0}"
TP="${TP:-4}"
DPA="${DPA:-1}"                 # DP-attention, symmetric on both legs
KVD="${KVD:-1}"
KVAWARE="${KVAWARE:-1}"
KVD_SOCK="${KVD_SOCK:-/tmp/kvd/kvd.sock}"
ETCD_PORT="${ETCD_PORT:-2379}"
BOOTSTRAP="${BOOTSTRAP:-8998}"
CTX="${CTX:-8192}"
CONC="${CONC:-64}"
CHUNK="${CHUNK:-4096}"
HICACHE_GB="${HICACHE_GB:-8}"

LOG="${LOG:-/tmp/kvexp_${ROLE}_${PORT}.log}"

NIC=$(ip -o -4 addr show | awk -v ip="$MY_IP" '$4 ~ ("^" ip "/") {print $2; exit}')
IB_DEVICES=$(for d in /sys/class/infiniband/*; do
    [ -d "$d" ] || continue; n=$(basename "$d")
    s=$(cat "$d/ports/1/state" 2>/dev/null || echo "")
    drv=$(basename "$(readlink -f "$d/device/driver" 2>/dev/null || echo x)")
    [[ "$s" == *ACTIVE* && "$drv" == ionic ]] && echo "$n"
  done | sort -V | paste -sd,)

export MC_GID_INDEX=1 MC_DISABLE_HIP_TRANSPORT=1 MOONCAKE_DISABLE_HIP_DMABUF=1

# --- the knob this experiment is about -------------------------------------
# FORCE_TCP=0: leave mooncake on RDMA. Both legs share one host, so the
#   transfer must loop back across rails. On this ionic fabric that fails with
#     worker_pool.cpp:408 ... local_nic: ionic_0, peer_nic: ...@ionic_4:
#                             transport retry counter exceeded
#     rdma_endpoint.cpp:472  Invalid argument: received packet mismatch
#   and the router returns HTTP 500 "Failed to get kvcache from prefill instance".
# FORCE_TCP=1: this repo's known correct-but-slower transport. Works same-host.
if [ "${FORCE_TCP:-1}" = "1" ]; then
  export MC_FORCE_TCP=1
  echo "[leg] MC_FORCE_TCP=1 (same-host workaround; slower)" >&2
else
  echo "[leg] !! MC_FORCE_TCP unset: real RDMA. Same-host transfer is EXPECTED to fail." >&2
fi

export RDMAV_FORK_SAFE=1 NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1
export HSA_NO_SCRATCH_RECLAIM=1
export SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP"
export SGLANG_LOCAL_IP_NIC="$NIC" GLOO_SOCKET_IFNAME="$NIC"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800
export PYTHONHASHSEED=0          # stable block hashes -> stable kvd keys
[ "$DPA" = "1" ] && export SGLANG_DP_USE_GATHERV=1

ARGS=(--model-path "$MODEL" --served-model-name "$SERVED" --tp-size "$TP" --trust-remote-code
      --host "$MY_IP" --port "$PORT" --advertise-host "$MY_IP" --base-gpu-id "$BASE_GPU"
      --etcd-endpoint "$ETCD_IP:$ETCD_PORT" --discovery-backend etcd
      --request-transport http --kv-event-transport zmq
      --mem-fraction-static 0.60 --context-length "$CTX" --max-running-requests "$CONC"
      --chunked-prefill-size "$CHUNK" --watchdog-timeout 3600
      --disaggregation-mode "$ROLE" --disaggregation-transfer-backend mooncake)
[ -n "$IB_DEVICES" ] && ARGS+=(--disaggregation-ib-device "$IB_DEVICES")
[ "$ROLE" = "prefill" ] && ARGS+=(--disaggregation-bootstrap-port "$BOOTSTRAP")

# kv-aware. --kv-events-bind is infera's OWN publisher socket; its default 5557
# is identical on every leg, so two co-located legs must be given different
# values (5557 / 5657 here). Distinct from the sglang --kv-events-config port,
# which infera derives from free_tcp_port_block() and which this flag does NOT
# control.
if [ "$KVAWARE" = "1" ]; then
  ARGS+=(--kv-events-bind "tcp://0.0.0.0:${KV_PUB_PORT:-5557}")
else
  ARGS+=(--no-enable-kv-events)
fi

[ "$KVD" = "1" ] && ARGS+=(--infera-kvd-socket "$KVD_SOCK" --hicache-size "$HICACHE_GB")

if [ "$DPA" = "1" ]; then ARGS+=(--dp-size "$TP" --enable-dp-attention --ep-size "$TP"); fi

echo "[leg] role=$ROLE ip=$MY_IP tp=$TP base=$BASE_GPU dpa=$DPA kvaware=$KVAWARE kvd=$KVD"
echo "[leg] force_tcp=${FORCE_TCP:-1} kv_pub_port=${KV_PUB_PORT:-5557} port=$PORT nic=$NIC ib=$IB_DEVICES -> $LOG"
echo "[leg] argv: python3 -m infera.engine.sglang ${ARGS[*]}" > "$LOG"
python3 -m infera.engine.sglang "${ARGS[@]}" >> "$LOG" 2>&1
