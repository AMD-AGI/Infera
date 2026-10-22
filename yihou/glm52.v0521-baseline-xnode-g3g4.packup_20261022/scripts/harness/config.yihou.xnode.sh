#!/usr/bin/env bash
# Overlay (task g4tune xnode, 2026-10-21): CROSS-NODE 1P1D P4D4. Everything = the chained base (default
# config.yihou.idlefix.sh, the 10-21 baseline) except placement and KV transport:
#   prefill on YIHOU_NODE / YIHOU_IP, GPUs 0-3 (also etcd, router, AgentX client = CONTROL_NODE)
#   decode  on XNODE_DECODE_NODE / XNODE_DECODE_IP, GPUs 0-3
#   KV over Mooncake RDMA: the ionic rails are physically isolated (ionic_i reaches only ionic_i), so both
#   legs pin local GPU i -> ionic_i (RDMA_DEVICE JSON map, unchanged from the base) and MC_TE_FILTERS lists the
#   same four rails; PD_DP_RANK_AFFINITY=1 (base) keeps prefill DP rank i -> decode DP rank i, i.e. rail i -> rail i.
#   (v0.5.19 135<->138 packup glm52-1p1d-samerail-c32-c40 notes 1: auto NIC choice -> unreachable rails.)
#   Mooncake routes a cross-host target to rdma on its own (multi_transport.cpp isHipReachableTarget), so
#   MC_DISABLE_HIP stays absent as in the base.
# Required (pass in YIHOU_OVERRIDES so engine.sh / agentx_bench.sh / audit re-source the same values):
#   XNODE_DECODE_NODE  e.g. crsuse2-m2m-136      XNODE_DECODE_IP  its ens3 IP
#   XNODE_BASE_CONFIG  chain to source (default config.yihou.idlefix.sh)
# Writable root: pass NODE_RW_ROOT= (empty) -> everything on $WS (NFS, rw on 135/136 since 10-19), nothing on
# node-local disks except docker's own store.
: "${XNODE_DECODE_NODE:?set XNODE_DECODE_NODE}"
: "${XNODE_DECODE_IP:?set XNODE_DECODE_IP}"
[[ "$XNODE_DECODE_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "config.yihou.xnode.sh: bad XNODE_DECODE_IP" >&2; return 2 2>/dev/null || exit 2; }
[[ "$XNODE_DECODE_NODE" != "$YIHOU_NODE" ]] || { echo "config.yihou.xnode.sh: decode node == prefill node" >&2; return 2 2>/dev/null || exit 2; }
_xn_base="${XNODE_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.idlefix.sh}"
[[ -r "$_xn_base" ]] || { echo "config.yihou.xnode.sh: base not readable: $_xn_base" >&2; return 2 2>/dev/null || exit 2; }
source "$_xn_base"

# Topology: prefill row on YIHOU_NODE, decode row on XNODE_DECODE_NODE (bench-harness tools are topology-aware).
TOPOLOGY="$WS/harness/topology.yihou.${NODE_KEY}x${XNODE_DECODE_NODE##*-}.tsv"
_xn_topo="$(printf 'role\tnode\tdata_ip\nprefill\t%s\t%s\ndecode\t%s\t%s\n' \
    "$YIHOU_NODE" "$YIHOU_IP" "$XNODE_DECODE_NODE" "$XNODE_DECODE_IP")"
if [[ "$(cat "$TOPOLOGY" 2>/dev/null)" != "$_xn_topo" ]]; then
    printf '%s\n' "$_xn_topo" > "$TOPOLOGY.tmp.$$" && mv -f "$TOPOLOGY.tmp.$$" "$TOPOLOGY"
fi
DECODE_GPU_DEVICES=0,1,2,3
[[ "$PREFILL_GPU_DEVICES" == 0,1,2,3 ]] || { echo "config.yihou.xnode.sh: PREFILL_GPU_DEVICES=$PREFILL_GPU_DEVICES" >&2; return 2 2>/dev/null || exit 2; }
[[ "$RDMA_DEVICE" == '{"0":"ionic_0","1":"ionic_1","2":"ionic_2","3":"ionic_3"}' ]] ||
    { echo "config.yihou.xnode.sh: unexpected RDMA_DEVICE=$RDMA_DEVICE" >&2; return 2 2>/dev/null || exit 2; }
MC_TE_FILTERS=ionic_0,ionic_1,ionic_2,ionic_3
[[ "$PD_DP_RANK_AFFINITY" == 1 ]] || { echo "config.yihou.xnode.sh: PD_DP_RANK_AFFINITY must be 1" >&2; return 2 2>/dev/null || exit 2; }

# The image was docker-loaded on each node separately; the containerd store may re-ID it (137, 10-12), so
# the expected ID follows the host this file is sourced on (engine.sh sources it on the node).
case "$(hostname -s)" in
    "$XNODE_DECODE_NODE") [[ -n "${XNODE_DECODE_IMAGE_ID:-}" ]] && EXPECTED_IMAGE_ID="$XNODE_DECODE_IMAGE_ID" ;;
esac
export EXPECTED_IMAGE_ID
YIHOU_EXP="${YIHOU_EXP:-E}+xnode-${NODE_KEY}x${XNODE_DECODE_NODE##*-}"
unset _xn_base _xn_topo
