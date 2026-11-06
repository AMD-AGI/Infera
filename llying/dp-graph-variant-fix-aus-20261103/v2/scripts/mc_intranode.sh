#!/bin/bash
# Intra-node Mooncake KV-transfer bandwidth, GPU VRAM -> GPU VRAM on one node (Infera's mooncakeperf workers: 1 GiB in
# 32 MiB writes, queue depth 32, >= 3 s), under the kit's rail settings and the alternatives. Why one-node 4+4 moves
# KV at ~1 GB/s per transfer (prefill per-stage kv_transfer_speed, 11-05) against 1.3-1.7 GB/s across two nodes.
# Runs inside the infera image (its entrypoint swaps in the host libionic), all 8 GPUs visible.
# yihou/samenode-kv-transfer-slow.brief.yihou.md: on one host Mooncake (faae8dd4) routes to its HIP transport (XGMI peer
# copies; hip > rdma > tcp), MC_DISABLE_HIP_TRANSPORT / MOONCAKE_DISABLE_HIP_DMABUF are read nowhere, MC_DISABLE_HIP=1
# drops hip; RDMA between two different ionic devices of one host fails silently, same-device loopback works (~40 GB/s).
set -u
ip=$(ip -4 -o addr show dev fenic | awk '{print $4}' | cut -d/ -f1)
NICS=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
MP="python3 -m infera.tools.preflight.network.mooncakeperf"
reset() {
    unset MC_DISABLE_HIP MC_DISABLE_HIP_TRANSPORT MC_ENABLE_HIP_TRANSPORT MC_ENABLE_DEST_DEVICE_AFFINITY MC_ENABLE_DEST_LOCAL_RAIL
    export MC_GID_INDEX=1 MC_TE_FILTERS=$NICS MOONCAKE_DISABLE_HIP_DMABUF=1 MC_MAX_MR_SIZE=1073741824 MC_IB_TIMEOUT=18 RDMAV_FORK_SAFE=1
}
# one SRC->DST write; prints "GB/s verified reason"
one() {  # SRC DST [NIC: both ends on this one device]
    local src=$1 dst=$2 dev=${3:-$NICS} sig tp
    sig=$(mktemp -d /tmp/mc.XXXX)
    MC_TE_FILTERS=$dev $MP "{\"role\":\"target\",\"sig\":\"$sig\",\"hostname\":\"$ip:0\",\"host\":\"$(hostname -s)\",\"protocol\":\"rdma\",\"operation\":\"write\",\"loc\":\"gpu\",\"gpu\":$dst,\"dev\":\"$dev\"}" >"$sig/target.log" 2>&1 &
    tp=$!
    MC_TE_FILTERS=$dev timeout 150 $MP "{\"role\":\"initiator\",\"sig\":\"$sig\",\"protocol\":\"rdma\",\"operation\":\"write\",\"loc\":\"gpu\",\"gpu\":$src,\"dev\":\"$dev\"}" >"$sig/init.log" 2>&1
    touch "$sig/done"; wait "$tp"
    python3 -c "import json; r = json.load(open('$sig/result.json')); print(r.get('gb_s'), r.get('verified'), r.get('reason'))" 2>/dev/null || echo "no result ($(tail -c 300 "$sig/init.log" | tr '\n' ' '))"
}
# LABEL then pairs "src:dst[:nic]" run concurrently
run() {
    local label=$1 p out=() pids=() i=0 f a
    shift
    for p in "$@"; do
        f=$(mktemp); out+=("$f")
        IFS=: read -ra a <<<"$p"
        one "${a[0]}" "${a[1]}" "${a[2]:-}" >"$f" &
        pids+=($!)
    done
    wait "${pids[@]}"
    for p in "$@"; do echo "$label $p: $(cat "${out[$i]}")"; i=$((i + 1)); done
}
echo "host $(hostname -s) ip $ip $(date -u +%T)"
# The kit's env (its two DISABLE_HIP variables are no-ops): same host -> HIP transport, idle.
reset; export MC_DISABLE_HIP_TRANSPORT=1 MC_ENABLE_DEST_DEVICE_AFFINITY=1 MC_ENABLE_DEST_LOCAL_RAIL=1
run "hip(kit env)" 0:4
run "hip(kit env) x4" 0:4 1:5 2:6 3:7
# RDMA loopback: MC_DISABLE_HIP=1, both ends on the same ionic device; destination-side rail, then source-side rail.
reset; export MC_DISABLE_HIP=1 MC_ENABLE_DEST_DEVICE_AFFINITY=1
run "rdma-loop dst-rail" 0:4:ionic_4
run "rdma-loop dst-rail x4" 0:4:ionic_4 1:5:ionic_5 2:6:ionic_6 3:7:ionic_7
run "rdma-loop src-rail" 0:4:ionic_0
run "rdma-loop src-rail x4" 0:4:ionic_0 1:5:ionic_1 2:6:ionic_2 3:7:ionic_3
echo "end $(date -u +%T)"
