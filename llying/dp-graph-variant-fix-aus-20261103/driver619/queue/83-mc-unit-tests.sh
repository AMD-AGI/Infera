#!/bin/bash
# DRIVER619-20261113 formal delivery: Mooncake's AMD-GPU tests from 81 (main + PR, nightly image), GPU 0 / ionic_0, on the
# 6.19 node, and on the 6.16 node if it has the nightly image: hip_gem_allocator_test (new) and rdma_hip_kfd_pin_test.
OUT=$K/out/mc-unit-$(date -u +%m%dT%H%M)
mkdir -p "$OUT"
B=$(readlink -f $K/out/mc-tests-latest)/main
on() { local n=$1; shift; local x=$NODE616; [ "$n" = "$NODE616" ] && x=$NODE619; srun --jobid=$JOB --overlap -N1 -n1 -w $n --exclude=$x "$@"; }
for n in $NODE619 $NODE616; do
    on $n bash -c "docker image inspect e1ab-sglang:1112 >/dev/null 2>&1 || { echo 'no nightly image here'; exit 0; }
    docker run --rm --network host --ipc host --device /dev/kfd --device /dev/dri --device /dev/infiniband \
        --group-add video --group-add render --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
        -v /lib/x86_64-linux-gnu/libionic.so:/host-libionic/libionic.so:ro -v /perf_apps:/perf_apps \
        -e ROCR_VISIBLE_DEVICES=0 -e MC_TEST_NICS=ionic_0 -e MC_GID_INDEX=1 -e MC_HIP_GEM_ALLOCATOR=$B/hip_gem_allocator.so \
        e1ab-sglang:1112 bash -c 'echo amdgpu \$(cat /sys/module/amdgpu/version)
            for t in hip_gem_allocator_test rdma_hip_kfd_pin_test; do echo \"== \$t\"; $B/\$t 2>&1 | grep -E \"^\[|Failure|Expected|error|which is\" | tail -30; echo \"rc \${PIPESTATUS[0]}\"; done'" \
        > "$OUT/$n.log" 2>&1 &
done
wait
for f in "$OUT"/*.log; do echo "#### $f"; cat "$f"; done
