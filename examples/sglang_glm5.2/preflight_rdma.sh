#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# RDMA preflight. PD moves the KV cache over RoCE on every request, so a broken
# fabric does not fail loudly — it silently makes the pair slower than one node.
#
#   bash preflight_rdma.sh                       # local: devices + GID + rail IP (run on both nodes)
#   bash preflight_rdma.sh server                # decode node: bandwidth listener
#   PEER=<decode-rail-ip> bash preflight_rdma.sh client   # prefill node: bandwidth test
set -euo pipefail

IB_DEVICE="${IB_DEVICE:-rdma0}"
MC_GID_INDEX="${MC_GID_INDEX:-3}"
MODE="${1:-local}"

RAIL_NIC="$(ls "/sys/class/infiniband/${IB_DEVICE}/device/net/" 2>/dev/null | head -1)"
RAIL_IP="$(ip -o -4 addr show "${RAIL_NIC:-none}" 2>/dev/null | awk '{print $4}' | cut -d/ -f1)"

if [[ "$MODE" == "local" ]]; then
    ACTIVE="$(ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE || true)"
    echo "[preflight] active RDMA ports: $ACTIVE   (expect one per rail, not 0)"
    [[ "$ACTIVE" != "0" ]] || { echo "[preflight] FAIL: no usable verbs device — the pod has no RDMA allocation"; exit 1; }

    # MC_GID_INDEX picks which GID Mooncake dials with. Broadcom bnxt_re exposes
    # RoCEv1/v2 for both the link-local IPv6 and the IPv4-mapped address; only the
    # RoCEv2 IPv4 entry is routable between nodes. Picking wrong is a silent hang.
    GID="$(cat "/sys/class/infiniband/${IB_DEVICE}/ports/1/gids/${MC_GID_INDEX}" 2>/dev/null || true)"
    TYPE="$(cat "/sys/class/infiniband/${IB_DEVICE}/ports/1/gid_attrs/types/${MC_GID_INDEX}" 2>/dev/null || true)"
    echo "[preflight] ${IB_DEVICE} gid[${MC_GID_INDEX}] = ${GID}  type=${TYPE}  ndev=${RAIL_NIC}"
    [[ "$TYPE" == "RoCE v2" && "$GID" == *":ffff:"* ]] \
        || { echo "[preflight] FAIL: gid[$MC_GID_INDEX] is not the routable RoCEv2 IPv4 GID — try another MC_GID_INDEX"; exit 1; }

    echo "[preflight] memlock: $(ulimit -l)   (must be unlimited)"
    echo "[preflight] OK — this node's ${IB_DEVICE} rail IP is ${RAIL_IP}"
    echo "[preflight] next: run 'bash preflight_rdma.sh server' on the decode node,"
    echo "            then 'PEER=<its rail IP> bash preflight_rdma.sh client' here."
    exit 0
fi

# ib_write_bw talks the same RoCEv2 path Mooncake will use, so a good number here
# is the real proof that the two pods can move KV over the fabric.
case "$MODE" in
    server) echo "[preflight] listening on ${IB_DEVICE} (${RAIL_IP}) — start the client now"
            exec ib_write_bw -d "$IB_DEVICE" -x "$MC_GID_INDEX" -F --report_gbits ;;
    client) : "${PEER:?set PEER to the rail IP printed by the peer node local run}"
            exec ib_write_bw -d "$IB_DEVICE" -x "$MC_GID_INDEX" -F --report_gbits "$PEER" ;;
    *)      echo "usage: bash preflight_rdma.sh [local|server|client]" >&2; exit 2 ;;
esac
