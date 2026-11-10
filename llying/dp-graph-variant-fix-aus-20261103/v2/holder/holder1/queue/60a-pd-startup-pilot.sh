#!/usr/bin/env bash
# REPORT-20261110 section 6, option 1 on an amdgpu 6.19 node (pinned job): SGLang PD prefill start-up (Mooncake, TP 8, DeepSeek-V3-5layer)
# with our image (graphs on = production; graphs off = same condition as the ROCm 10 run) and the upstream ROCm 10 image (graphs off).
B=/perf_apps/liyingli/bench_agentx/kopt-20261109/bench
echo "$(date -u +%FT%TZ) start on $(hostname), amdgpu $(cat /sys/module/amdgpu/version)"
bash $B/pd_startup_test.sh infera 8 12 graph
bash $B/pd_startup_test.sh infera 8 12 nograph
bash $B/pd_startup_test.sh rocm10 8 12 nograph
echo "$(date -u +%FT%TZ) done"
