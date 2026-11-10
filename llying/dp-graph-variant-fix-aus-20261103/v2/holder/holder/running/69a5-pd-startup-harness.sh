#!/usr/bin/env bash
# REPORT-20261110 section 6, option 1: harness check on a 6.16 node (batch host) before the 6.19 test: SGLang PD prefill start-up
# (Mooncake, TP 8, DeepSeek-V3-5layer) with our image and with the upstream ROCm 10 image (pulled if missing).
B=/perf_apps/liyingli/bench_agentx/kopt-20261109/bench
I=lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261021
echo "$(date -u +%FT%TZ) start on $(hostname), amdgpu $(cat /sys/module/amdgpu/version)"
docker image inspect $I >/dev/null 2>&1 || timeout 2400 docker pull $I | tail -2
bash $B/pd_startup_test.sh infera 8 12
bash $B/pd_startup_test.sh rocm10 8 12
echo "$(date -u +%FT%TZ) done"
