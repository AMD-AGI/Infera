#!/bin/bash
# DRIVER619-20261113 formal delivery, nightly only: SGLang's new CPU unit test and the neighbouring EFA allocator test on the
# PR branch (a copy on /perf_apps, kfd619/sglang-pr), as the Python environment the nightly image e1ab-sglang:1112; the
# image's ROCm torch needs the GPU devices to import (63 failed without them).
OUT=$K/out/sglang-unit-$(date -u +%m%dT%H%M)
mkdir -p "$OUT"
T=test/registered/unit/disaggregation
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 --exclude=$NODE616 docker run --rm --network none --device /dev/kfd \
    --device /dev/dri --group-add video --group-add render --entrypoint bash -v /perf_apps:/perf_apps \
    -e PYTHONDONTWRITEBYTECODE=1 e1ab-sglang:1112 -c "
cd $K/sglang-pr && cat REV
PYTHONPATH=python python3 -m pytest -q -p no:cacheprovider $T/test_mooncake_amd_gem_pool.py $T/test_mooncake_efa_allocator.py 2>&1 | grep -v -E 'Warning|warn|^\s*\$' | tail -15
" > "$OUT/run.log" 2>&1
cat "$OUT/run.log"
