#!/bin/bash
# ---------------------------------------------------------------------------
# Experiment 03 — one infera-sglang PD leg, Qwen3-1.7B, single node.
#
# This is the round-r2 leg: the previous round's hicache lesson is already
# baked in (--hicache-size, absolute GB), so the only thing left in the way is
# the --kv-events-config port collision that this experiment is about.
#
# The port bug does NOT live in this script. It lives in
# infera/common/net.py:free_tcp_port_block(), which the infera wrapper calls
# when kv-events are enabled. Which version of net.py is inside the container
# decides whether the decode leg lives or dies — see scripts/run.sh.
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
DPA="${DPA:-1}"                 # DP-attention on both legs. dp_size>1 is what
                                # makes the collision fatal: sglang binds one
                                # KV-event publisher per DP rank at base+rank,
                                # so an identical base overlaps at base+1.
KVD="${KVD:-1}"                 # 1 = wire the infera-kvd hicache backend
KVAWARE="${KVAWARE:-1}"         # 1 = publish kv events. free_tcp_port_block is
                                # reached ONLY on this path (worker.py:77), so
                                # KVAWARE=0 makes the bug unreachable.
KVD_SOCK="${KVD_SOCK:-/tmp/kvd/kvd.sock}"
ETCD_PORT="${ETCD_PORT:-2379}"
BOOTSTRAP="${BOOTSTRAP:-8998}"
CTX="${CTX:-8192}"
CONC="${CONC:-64}"
CHUNK="${CHUNK:-4096}"
# Absolute GB per rank. The default --hicache-ratio 2.0 sizes the host pool off
# max_total_num_tokens and on this small model asked for 354.94 GB per DP rank.
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
# Both legs on ONE host. Same-host mooncake RDMA cannot loop back across rails
# on this fabric; force TCP. Irrelevant to r2 (the decode leg never starts, so
# no KV ever moves) but held constant so the rounds stay comparable.
[ "${FORCE_TCP:-1}" = "1" ] && export MC_FORCE_TCP=1
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

# kv-aware. NOTE the distinction that cost a round:
#   --kv-events-bind      = infera's OWN publisher socket. Settable here.
#   --kv-events-config    = the endpoint handed to SGLANG. NOT settable here;
#                           infera computes its port from free_tcp_port_block()
#                           and worker.py:77 ignores --kv-events-bind for it.
# Changing KV_PUB_PORT therefore does NOT work around the collision. It fixes a
# different, adjacent collision (both legs defaulting to 5557).
if [ "$KVAWARE" = "1" ]; then
  ARGS+=(--kv-events-bind "tcp://0.0.0.0:${KV_PUB_PORT:-5557}")
else
  ARGS+=(--no-enable-kv-events)
fi

# kvd: the infera wrapper probes the daemon, then appends
#   --enable-hierarchical-cache --hicache-storage-backend dynamic --hicache-...-extra-config
[ "$KVD" = "1" ] && ARGS+=(--infera-kvd-socket "$KVD_SOCK" --hicache-size "$HICACHE_GB")

# DP-attention (symmetric on both legs)
if [ "$DPA" = "1" ]; then ARGS+=(--dp-size "$TP" --enable-dp-attention --ep-size "$TP"); fi

echo "[leg] role=$ROLE ip=$MY_IP tp=$TP base=$BASE_GPU dpa=$DPA kvaware=$KVAWARE kvd=$KVD"
echo "[leg] hicache_gb=$HICACHE_GB port=$PORT nic=$NIC ib=$IB_DEVICES -> $LOG"
echo "[leg] argv: python3 -m infera.engine.sglang ${ARGS[*]}" > "$LOG"
python3 -m infera.engine.sglang "${ARGS[@]}" >> "$LOG" 2>&1
