#!/bin/bash
# INFX4629-20261113 (not part of the 6.19 work): single-GPU compile test of SGLang's absorb-BMM hand
# config (sglang#46829) in the ROCm 7.2 and ROCm 10 nightlies, on the 6.16 node's GPU 0. About
# 15-30 min, most of it pulling the ROCm 10 image. Files: /perf_apps/liyingli/bench_agentx/absorb-bmm-rocm10-20261113/
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE616 bash /perf_apps/liyingli/bench_agentx/absorb-bmm-rocm10-20261113/run_on_node.sh
