#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Launch one PD-disaggregated SGLang engine (prefill OR decode) for
# gpt-oss-120b inside a persistent container on the current node. KV-cache
# transfer between prefill and decode goes over MoRI-IO (AINIC RDMA) through
# the node's 8 ionic NICs.
#
# Adapted from the proven DeepSeek-R1 1P1D mooncake harness, with three
# changes for this run:
#   1. transport: mooncake -> mori  (--disaggregation-transfer-backend mori)
#      + MoRI needs ALL ionic NICs passed (it pairs NIC<->NIC by GID subnet)
#      + MORI_IB_GID_INDEX=1 (ULA GID, not link-local fe80::)
#   2. model:     DeepSeek-R1 (MLA) -> gpt-oss-120b (MoE + GQA) -> drop the
#      EAGLE speculative-decode flags (those are MLA/DSR1 specific).
#   3. kvd L3 (optional, decode-side): when INFERA_KVD_L3=1, the decode
#      engine offloads evicted KV to the kvd HiCacheStorage backend so a
#      repeated prefix is served from the durable L3 tier instead of being
#      re-pushed P->D. This is the kvd-in-PD pairing.
#
# Required env:
#   ROLE            prefill | decode
#   DATA_PLANE_IP   this node's data-plane IP (MoRI/bootstrap bind)
#   ETCD_ENDPOINT   host:port of etcd on the prefill node
#
# Optional:
#   MODEL_PATH      default /PATH/TO/gpt-oss-120b
#   TP_SIZE         default 8
#   SGL_PORT        default 30000
#   BOOTSTRAP_PORT  default 8998 (prefill only)
#   CONTAINER       default pd_mori_sgl
#   INFERA_SRC    default: repo root derived from script location (PYTHONPATH)
#   INFERA_KVD_L3 0|1 — decode-side kvd L3 offload (default 0)
#   KVD_SOCKET      kvd UDS socket (default /tmp/infera-kvd/kvd.sock)

set -euo pipefail

: "${ROLE:?ROLE must be prefill or decode}"
: "${DATA_PLANE_IP:?DATA_PLANE_IP must be the local data-plane IP}"
: "${ETCD_ENDPOINT:?ETCD_ENDPOINT must be set, e.g. <prefill-ip>:12379}"

case "$ROLE" in
    prefill|decode) ;;
    *) echo "[launch_engine] ROLE must be prefill|decode, got '$ROLE'" >&2; exit 64 ;;
esac

CONTAINER="${CONTAINER:-pd_mori_sgl}"
MODEL_PATH="${MODEL_PATH:-/PATH/TO/gpt-oss-120b}"
TP_SIZE="${TP_SIZE:-8}"
SGL_PORT="${SGL_PORT:-30000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-512}"
MAX_RUNNING_REQS="${MAX_RUNNING_REQS:-512}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
PAGE_SIZE="${PAGE_SIZE:-64}"
# MoE runner: 'auto' picks aiter ck which JIT-needs an fp4x2 fused kernel that
# isn't built in this image for gpt-oss MXFP4 (ModuleNotFoundError ->
# scheduler dies). 'triton' is the portable ROCm fallback.
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-auto}"
# Auto-detect the live GPU arch (e.g. gfx942/gfx950) so the same script drives
# aiter/MoRI JIT correctly on any Instinct GPU. Uses infera's torch-based
# detector inside the container (amd-smi text/JSON is an unstable interface
# across ROCm releases). Override via GPU_ARCH.
GPU_ARCH="${GPU_ARCH:-$(docker exec "$CONTAINER" python3 -m infera.common.arch)}"
# Repo root and this bench dir, auto-derived from the script location
# (bench/pd_mori_1p1d/ -> 2 levels up). Override INFERA_SRC / WORKSPACE
# when the source is mounted elsewhere inside the container.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERA_SRC="${INFERA_SRC:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
INFERA_KVD_L3="${INFERA_KVD_L3:-0}"
KVD_SOCKET="${KVD_SOCKET:-/tmp/infera-kvd/kvd.sock}"
# Shared NATS bus on the prefill node. The infera engine wrapper opens a
# NATS client during bring-up regardless of request transport; without a
# reachable broker its async main blocks before etcd registration (worker
# never appears in /v1/workers). Point every node at the one broker.
NATS_SERVER="${NATS_SERVER:?set NATS_SERVER, e.g. nats://<prefill-ip>:4222}"
REQUEST_TRANSPORT="${REQUEST_TRANSPORT:-nats}"

WORKSPACE="${WORKSPACE:-$SCRIPT_DIR}"
LOG_FILE="${LOG_FILE:-${WORKSPACE}/logs/pd_${ROLE}_$(hostname -s)_$(date +%Y%m%d_%H%M%S).log}"

# Auto-detect ACTIVE ionic NICs (MoRI wants them ALL — it pairs NIC<->NIC by
# GID subnet, the opposite of Mooncake where you drop the flag).
if [[ -z "${IB_DEVICES:-}" ]]; then
    IB_DEVICES=$(docker exec "$CONTAINER" bash -c '
        for d in /sys/class/infiniband/*; do
            [[ -d "$d" ]] || continue
            name=$(basename "$d")
            state=$(cat "$d/ports/1/state" 2>/dev/null || echo "")
            drv=$(basename "$(readlink -f "$d/device/driver" 2>/dev/null || echo unknown)")
            [[ "$state" == *"ACTIVE"* && "$drv" == "ionic" ]] && echo "$name"
        done | sort -V | paste -sd,
    ')
fi
[[ -z "$IB_DEVICES" ]] && { echo "[launch_engine] no active ionic NICs found" >&2; exit 1; }

echo "[launch_engine] role=${ROLE} model=${MODEL_PATH} TP=${TP_SIZE}"
echo "[launch_engine] data_ip=${DATA_PLANE_IP} etcd=${ETCD_ENDPOINT}"
echo "[launch_engine] mori ib_devices=${IB_DEVICES}  gid_index=1"
echo "[launch_engine] kvd_l3=${INFERA_KVD_L3}  log=${LOG_FILE}"

ROLE_FLAGS="--disaggregation-mode ${ROLE}"
if [[ "$ROLE" == "prefill" ]]; then
    ROLE_FLAGS+=" --disaggregation-bootstrap-port ${BOOTSTRAP_PORT}"
fi

# Decode-side kvd L3 offload: store evicted KV to the kvd HiCacheStorage
# backend so a repeated prefix is a warm L3 hit, not a P->D re-push. We keep
# the radix cache ON (NEVER disable it — it's the L1 the L3 backs).
#
# The seam is a SINGLE flag, --infera-kvd-socket: the engine's kvd_wiring
# (a) PROBES the daemon and refuses to start if it's down, and (b) auto-appends
# --enable-hierarchical-cache --hicache-storage-backend dynamic + the
# infera-kvd backend extra config. So we must start the kvd DAEMON first
# (it's a separate process), then pass the socket — NOT hand-write the hicache
# flags. The backend reaches the SGLang worker via its built-in `dynamic`
# backend loader (kvd_wiring forwards the class path on the worker argv).
# INFERA_KVD_L3=1 enables it on BOTH roles when KVD_L3_DIR is a SHARED mount
# (content-addressed chunk keys → prefill's writes are visible to decode and to
# a later prefill): prefill gets recompute-avoidance, decode gets wire-transfer
# avoidance, one store, no double-write. Each node runs its own kvd daemon
# (local UDS) but both --long-path the same shared dir. Set INFERA_KVD_L3=decode
# for the decode-only variant.
KVD_FLAGS=""
if [[ "$INFERA_KVD_L3" == "1" || ( "$ROLE" == "decode" && "$INFERA_KVD_L3" == "decode" ) ]]; then
    KVD_L3_DIR="${KVD_L3_DIR:-/mnt/kvd-l3}"
    KVD_RAM_BYTES="${KVD_RAM_BYTES:-$((32 * 1024 * 1024 * 1024))}"     # 32 GiB host-pinned
    KVD_LONG_BYTES="${KVD_LONG_BYTES:-$((512 * 1024 * 1024 * 1024))}"  # 512 GiB L3
    echo "[launch_engine] starting kvd daemon: socket=${KVD_SOCKET} l3=${KVD_L3_DIR}"
    # Remove any STALE socket from a prior run first — otherwise `test -S` (or a
    # connect) can pass on the dead file before the new daemon is listening, and
    # the engine's wiring probe hard-fails with Connection refused.
    docker exec "$CONTAINER" rm -f "${KVD_SOCKET}" 2>/dev/null || true
    docker exec -d "$CONTAINER" bash -lc "
        cd /tmp
        export PYTHONPATH=${INFERA_SRC}:\${PYTHONPATH:-}
        mkdir -p \$(dirname ${KVD_SOCKET}) ${KVD_L3_DIR}
        exec > ${WORKSPACE}/logs/kvd_\$(hostname -s)_\$(date +%H%M%S).log 2>&1
        python3 -m infera.kvd \
            --socket ${KVD_SOCKET} \
            --max-bytes ${KVD_RAM_BYTES} \
            --no-shared-arena-pin \
            --long-path ${KVD_L3_DIR} --long-bytes ${KVD_LONG_BYTES}
    "
    # Wait until the daemon is actually CONNECTABLE (not just that the socket
    # file exists) — the engine's wiring probe hard-fails otherwise. Connect via
    # the real KvdClient so this matches what the engine will do.
    ok=0
    for i in {1..60}; do
        if docker exec -e PYTHONPATH=${INFERA_SRC} "$CONTAINER" python3 -c "
import asyncio,sys
from infera.kvd.client import KvdClient
async def m():
    c=KvdClient('${KVD_SOCKET}', client_id='launch-probe')
    await asyncio.wait_for(c.connect(), timeout=3); await c.close()
asyncio.run(m())
" >/dev/null 2>&1; then ok=1; break; fi
        sleep 1
    done
    [ "$ok" = 1 ] || { echo "[launch_engine] kvd daemon at ${KVD_SOCKET} not connectable" >&2; exit 1; }
    echo "[launch_engine] kvd daemon connectable"
    KVD_FLAGS="--infera-kvd-socket ${KVD_SOCKET}"
    # On the DECODE side of a PD pair, SGLang forces the KV manager to
    # chunk-cache ("KV cache is forced as chunk cache for decode server"),
    # which is mutually exclusive with --enable-hierarchical-cache (the flag
    # the kvd wiring appends). Re-enable the decode radix cache so the L1 the
    # kvd L3 backs exists — this is exactly the kvd-in-PD pairing.
    if [[ "$ROLE" == "decode" ]]; then
        KVD_FLAGS+=" --disaggregation-decode-enable-radix-cache"
        # SGLang gates --disaggregation-decode-enable-radix-cache to
        # transfer-backend ∈ {nixl,mooncake} (pd_disaggregation_hook.py). The
        # guard is conservative — enabling the flag only flips
        # disable_radix_cache=False; nothing else is transport-specific. Add
        # 'mori' to the allowlist so the decode engine boots. EXPERIMENTAL:
        # MoRI's decode KV-receive path must integrate with the radix layout
        # (vs chunk-cache) — VERIFY coherent output, not just that it boots.
        # Opt out with INFERA_KVD_NO_MORI_RADIX_PATCH=1.
        if [[ "${INFERA_KVD_NO_MORI_RADIX_PATCH:-0}" != "1" ]]; then
            docker exec "$CONTAINER" bash -lc '
              f=/sgl-workspace/sglang/python/sglang/srt/arg_groups/pd_disaggregation_hook.py
              grep -q "\"nixl\", \"mooncake\", \"mori\"" "$f" || \
                sed -i "s/not in (\"nixl\", \"mooncake\")/not in (\"nixl\", \"mooncake\", \"mori\")/" "$f"
            ' 2>/dev/null && echo "[launch_engine] patched SGLang decode-radix allowlist to include mori"
        fi
    fi
fi

mkdir -p "$(dirname "$LOG_FILE")"

docker exec -d "$CONTAINER" bash -lc "
    cd ${INFERA_SRC}
    exec >${LOG_FILE} 2>&1
    export PYTHONPATH=${INFERA_SRC}:\${PYTHONPATH:-}
    # MoRI + sglang must bind the data-plane IP, not the default-route NIC.
    export SGLANG_HOST_IP=${DATA_PLANE_IP}
    export HOST_IP=${DATA_PLANE_IP}
    export MORI_IB_GID_INDEX=1
    export SGLANG_USE_AITER=1
    export RCCL_MSCCL_ENABLE=0
    # aiter JIT must build the MXFP4 fp4x2 fused-MoE kernel for THIS arch
    # (gfx950 / MI355X); without the arch set the prebuilt set misses it and
    # the ck2stages fp4 module is ModuleNotFound.
    export PYTORCH_ROCM_ARCH=${GPU_ARCH}
    export GPU_ARCHS=${GPU_ARCH}
    export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export INFERA_KVD_SOCKET=${KVD_SOCKET}
    # cold NFS weight loads + cross-node bootstrap need generous timeouts.
    export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
    export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200

    python3 -m infera.engine.sglang \
        --discovery-backend etcd \
        --etcd-endpoint ${ETCD_ENDPOINT} \
        --advertise-host ${DATA_PLANE_IP} \
        --request-transport ${REQUEST_TRANSPORT} \
        --nats-server ${NATS_SERVER} \
        --kv-events off \
        --model-path ${MODEL_PATH} \
        --host 0.0.0.0 \
        --port ${SGL_PORT} \
        --tp-size ${TP_SIZE} \
        --trust-remote-code \
        --attention-backend aiter \
        --moe-runner-backend ${MOE_RUNNER_BACKEND} \
        --kv-cache-dtype fp8_e4m3 \
        --mem-fraction-static ${MEM_FRACTION_STATIC} \
        --chunked-prefill-size 131072 \
        --max-prefill-tokens 131072 \
        --cuda-graph-max-bs ${CUDA_GRAPH_MAX_BS} \
        --page-size ${PAGE_SIZE} \
        --num-continuous-decode-steps 8 \
        --max-running-requests ${MAX_RUNNING_REQS} \
        --stream-interval 30 \
        --scheduler-recv-interval 10 \
        ${ROLE_FLAGS} \
        ${KVD_FLAGS} \
        --disaggregation-transfer-backend mori \
        --disaggregation-ib-device ${IB_DEVICES}
"

sleep 2
echo "[launch_engine] kicked off (cold start ~8-12 min: weights + cuda graph)"
echo "[launch_engine] tail: docker exec ${CONTAINER} tail -f ${LOG_FILE}"
