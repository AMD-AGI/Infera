#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Start the Mooncake Store the prefill offloads its KV to (PREFILL_CPU_TIER=store),
# laid out as InferenceX PR #4045's 1P1D: a master on NUMA node 1 and two owners,
# owner i holding STORE_OWNER_GB of NUMA node i's DRAM, both registered on the
# prefill's rails. Checks THP and the free memory of both NUMA nodes first, and
# waits until both owners serve. Normally called by up.sh.
#
# Usage: bash engine/store.sh
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$KIT/engine/common.sh"

# Each owner faults in its segment as 2 MiB pages, which needs THP and that much
# free memory on its NUMA node (gotcha 14): fail now, not minutes into the mount.
thp="$(on_docker_host 'cat /sys/kernel/mm/transparent_hugepage/enabled')" ||
    die "cannot read the THP setting on the docker host"
[[ "$thp" != *"[never]"* ]] || die "THP is off on the docker host ($thp); the Store owners need madvise or always"
for i in 0 1; do
    free_gb="$(on_docker_host "awk '/MemFree/ {print int(\$4 / 1048576)}' /sys/devices/system/node/node$i/meminfo")" ||
        die "cannot read the free memory of NUMA node $i on the docker host"
    ((free_gb >= STORE_OWNER_GB)) ||
        die "NUMA node $i has $free_gb GiB free, its Store owner takes STORE_OWNER_GB=$STORE_OWNER_GB: drop the page cache or lower STORE_OWNER_GB (README gotcha 14)"
done

# Bind a process's CPUs and memory to one NUMA node; set_mempolicy needs SYS_NICE.
numa_exec=/app/ATOM/.github/scripts/atomesh/numa_exec.py
# No GPU: a Mooncake process that sees one probes every buffer through HIP.
common=(--network host --cap-add SYS_NICE -e HIP_VISIBLE_DEVICES=-1 -e CUDA_VISIBLE_DEVICES=-1
    -e MC_NUM_QP_PER_EP=1 -e "MC_GID_INDEX=$MC_GID_INDEX")

docker run -d --name "$CTR-store-master" "${common[@]}" "$IMAGE" \
    python3 "$numa_exec" 1 mooncake_master --rpc_port="$STORE_MASTER_PORT" --rpc_thread_num=32 \
    --enable_http_metadata_server=true --http_metadata_server_host=0.0.0.0 \
    --http_metadata_server_port="$STORE_METADATA_PORT" --metrics_port="$STORE_METRICS_PORT" \
    --default_kv_lease_ttl=10000 --eviction_high_watermark_ratio=0.90 --eviction_ratio=0.05 \
    --allocation_strategy=random --memory_allocator=offset --client_ttl=10 >/dev/null
wait_health "http://$NODE_IP:$STORE_METRICS_PORT/metrics" "$CTR-store-master"

mounts=()
if on_docker_host "test -e '$HOST_LIBIONIC'"; then
    mounts+=(-v "$HOST_LIBIONIC:/host-libionic/libionic.so:ro")
fi
# glibc's THP malloc backs the segment with 2 MiB pages, which the NICs register
# in 64 GiB regions; 4 KiB pages they refuse (gotcha 14).
for i in 0 1; do
    docker run -d --name "$CTR-store-owner$i" "${common[@]}" "${mounts[@]}" --device /dev/infiniband \
        --cap-add IPC_LOCK --ulimit memlock=-1:-1 --security-opt seccomp=unconfined \
        -e GLIBC_TUNABLES=glibc.malloc.hugetlb=1 -e MC_MAX_MR_SIZE=68719476736 -e "MC_TCP_BIND_ADDRESS=$NODE_IP" \
        "$IMAGE" python3 "$numa_exec" "$i" mooncake_client --host="$NODE_IP" --port=$((STORE_OWNER_PORT + i)) \
        --master_server_address="$NODE_IP:$STORE_MASTER_PORT" \
        --metadata_server="http://$NODE_IP:$STORE_METADATA_PORT/metadata" \
        --protocol=rdma --device_names="$STORE_OWNER_DEVICES" --global_segment_size=$((STORE_OWNER_GB << 30)) \
        --local_buffer_size=0 --threads=4 >/dev/null
done

# An owner serves once its segment is faulted in and registered: minutes.
deadline=$((SECONDS + READY_TIMEOUT))
for i in 0 1; do
    c="$CTR-store-owner$i"
    until docker logs "$c" 2>&1 | grep 'Starting real client service' >/dev/null; do
        [[ "$(docker inspect --type container -f '{{.State.Running}}' "$c" 2>/dev/null)" == true ]] ||
            die "$c stopped while mounting its segment; see: docker logs $c"
        ((SECONDS < deadline)) || die "timeout after ${READY_TIMEOUT}s waiting for $c"
        sleep 10
    done
    echo "$c: $STORE_OWNER_GB GiB on NUMA $i, $(docker exec "$c" grep AnonHugePages /proc/1/smaps_rollup | tr -s ' ')"
done
