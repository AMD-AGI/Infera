#!/usr/bin/env bash
# REPORT-20261110 section 6: control for the 6.19 root cause on this (6.16) node: the same small-model TP8 prefill PD
# start-up with the VRAM probe around Mooncake's registration at 0.8 (expected: ready, free VRAM unchanged by the
# registration; on 6.19 it drops by exactly the registered size and the start-up segfaults). ~4 min.
echo "$(date -u +%FT%TZ) start on $(hostname), amdgpu $(cat /sys/module/amdgpu/version)"
bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/clear_foreign_containers.sh "$SLURM_JOB_ID"
MEMPROBE=1 MEMFRAC=0.8 bash /perf_apps/liyingli/bench_agentx/kopt-20261109/bench/out/pd_startup/pd_startup_test.run-616ctl.sh infera 8 12 nograph
echo "$(date -u +%FT%TZ) done"
