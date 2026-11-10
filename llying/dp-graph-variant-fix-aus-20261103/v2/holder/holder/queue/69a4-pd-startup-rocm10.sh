#!/usr/bin/env bash
# REPORT-20261110 section 6, option 1: harness check of the upstream ROCm 10 image on a 6.16 node (69a5 could not pull it: the
# nodes' docker token is expired; pd_startup_test.sh now pulls anonymously).
echo "$(date -u +%FT%TZ) start on $(hostname), amdgpu $(cat /sys/module/amdgpu/version)"
bash /perf_apps/liyingli/bench_agentx/kopt-20261109/bench/pd_startup_test.sh rocm10 8 12
echo "$(date -u +%FT%TZ) done"
