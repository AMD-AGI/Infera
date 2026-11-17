#!/bin/bash
# DRIVER619-20261113, 11-17: tests for the review round of kvcache-ai/Mooncake#5584 / #5587 on the builds of 98, GPUs 2-3
# and ionic_2 of n02-25 (another session's transfer_engine_bench holds GPU 1). The existing tests run on upstream base
# and on #5584 alike, so a feature the PR touched shows up as a difference: KFD pin anchor (#4657), HIP transport across
# two GPUs (the copy-device path the PR changed), HIP stream pool under each MC_HIP_STREAM_PRIORITY, TCP transport,
# transport unit tests. #5584 adds hip_gem_pin_test and hip_gem_allocator_test; #5587 runs hip_gem_allocator_test
# through TENT with RDMA only.
B=$(readlink -f $K/out/mc-review-latest)
OUT=$B/tests
mkdir -p "$OUT"
TENT_CONF='{"transports":{"tcp":{"enable":false},"io_uring":{"enable":false}}}'
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 --exclude=$NODE616 bash -c 'rocm-smi --showmemuse | grep -E "GPU\[[23]\].*VRAM%"' \
    > "$OUT/gpus-before.txt" 2>&1
run() {  # TREE NAME "ENV..." COMMAND...
    local tree=$1 name=$2 envs=$3; shift 3
    srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 --exclude=$NODE616 docker run --rm --network host --ipc host \
        --device /dev/kfd --device /dev/dri --device /dev/infiniband --group-add video --group-add render \
        --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
        -v /lib/x86_64-linux-gnu/libionic.so:/host-libionic/libionic.so:ro -v /perf_apps:/perf_apps \
        -e ROCR_VISIBLE_DEVICES=2,3 -e MC_TEST_NICS=ionic_2 -e MC_GID_INDEX=1 -e MC_HIP_GEM_ALLOCATOR=$B/$tree/hip_gem_allocator.so \
        -e MC_METADATA_SERVER=P2PHANDSHAKE -e MC_LOCAL_SERVER_NAME=127.0.0.1:17777 $envs \
        e1ab-sglang:1112 bash -c "export LD_LIBRARY_PATH=$B/$tree:\$LD_LIBRARY_PATH; cd $B/$tree; timeout 600 $*" \
        > "$OUT/$tree-$name.log" 2>&1
    local rc=$?
    printf '%-5s %-30s rc=%-3s %s\n' "$tree" "$name" "$rc" \
        "$(grep -hoE '\[  (PASSED|FAILED|SKIPPED) +\] [0-9]+ tests?' "$OUT/$tree-$name.log" | tr '\n' ' ')"
}
for tree in base main; do
    run $tree kfd_pin "" ./rdma_hip_kfd_pin_test
    run $tree hip_transport "" ./hip_transport_test
    for p in unset default high invalid; do
        e="-e MC_HIP_STREAM_PRIORITY=$p"; [ $p = unset ] && e=""
        run $tree hip_stream_pool_$p "$e" ./hip_stream_pool_test
    done
    run $tree tcp_transport "" ./tcp_transport_test
    run $tree transport_uint "" ./transport_uint_test
done
run main hip_gem_pin "" ./hip_gem_pin_test
run main hip_gem_allocator "" ./hip_gem_allocator_test
run tent hip_gem_pin "" ./hip_gem_pin_test
run tent hip_gem_allocator_tent "-e MC_USE_TENT=1 -e MOONCAKE_LOCAL_HOSTNAME=127.0.0.1 -e MC_TENT_CONF=$TENT_CONF" \
    ./hip_gem_allocator_test
grep -h "FAILED  \]\|Failure" "$OUT"/*.log | sort | uniq -c | head -20
