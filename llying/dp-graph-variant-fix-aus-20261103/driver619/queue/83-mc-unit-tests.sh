#!/bin/bash
# DRIVER619-20261113 formal delivery: Mooncake's AMD-GPU tests from 81 (nightly image), GPU 0 / ionic_0, on the 6.19 node.
#   main  PR 1, classic transfer engine: hip_gem_allocator_test (new) and rdma_hip_kfd_pin_test
#   tent  PR 2, hip_gem_allocator_test through TENT (MC_USE_TENT=1) with only RDMA enabled, so a GEM buffer TENT could
#         not register through its dma-buf fails the test instead of falling back to TCP
OUT=$K/out/mc-unit-$(date -u +%m%dT%H%M)
mkdir -p "$OUT"
B=$(readlink -f $K/out/mc-tests-latest)
TENT_CONF='{"transports":{"tcp":{"enable":false},"io_uring":{"enable":false}}}'
on() { local n=$1; shift; local x=$NODE616; [ "$n" = "$NODE616" ] && x=$NODE619; srun --jobid=$JOB --overlap -N1 -n1 -w $n --exclude=$x "$@"; }
run() {  # NAME BUILD "TESTS" [docker -e args...]
    local name=$1 b=$B/$2 tests=$3; shift 3
    on $NODE619 bash -c "docker image inspect e1ab-sglang:1112 >/dev/null 2>&1 || { echo 'no nightly image here'; exit 0; }
    docker run --rm --network host --ipc host --device /dev/kfd --device /dev/dri --device /dev/infiniband \
        --group-add video --group-add render --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
        -v /lib/x86_64-linux-gnu/libionic.so:/host-libionic/libionic.so:ro -v /perf_apps:/perf_apps \
        -e ROCR_VISIBLE_DEVICES=0 -e MC_TEST_NICS=ionic_0 -e MC_GID_INDEX=1 -e MC_HIP_GEM_ALLOCATOR=$b/hip_gem_allocator.so \
        $* e1ab-sglang:1112 bash -c 'echo amdgpu \$(cat /sys/module/amdgpu/version); export LD_LIBRARY_PATH=$b:\$LD_LIBRARY_PATH
            for t in $tests; do echo \"== \$t\"; $b/\$t > $OUT/$name-\$t.full.log 2>&1; rc=\$?
                grep -E \"^\[|Failure|Expected|error|which is|dma-buf|GEM\" $OUT/$name-\$t.full.log | tail -40; echo \"rc \$rc\"; done'" \
        > "$OUT/$name.log" 2>&1
}
run main main "hip_gem_allocator_test rdma_hip_kfd_pin_test"
run tent tent "hip_gem_allocator_test" -e MC_USE_TENT=1 -e MOONCAKE_LOCAL_HOSTNAME=127.0.0.1 -e "MC_TENT_CONF='$TENT_CONF'"
for f in "$OUT"/main.log "$OUT"/tent.log; do echo "#### $f"; cat "$f"; done
