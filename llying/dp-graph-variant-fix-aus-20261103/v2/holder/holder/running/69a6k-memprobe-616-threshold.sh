#!/usr/bin/env bash
# REPORT-20261110 section 6: the 6.16 control (69a6h) segfaulted like 6.19 at 0.8 (free VRAM 86 -> 0 at registration), so
# the small-model test does not tell the drivers apart. Threshold on 6.16 to compare with 6.19 (0.5 ready, 0.65 segfault):
# MEMPROBE at 0.5, 0.65 and 0.72. ~8 min.
echo "$(date -u +%FT%TZ) start on $(hostname), amdgpu $(cat /sys/module/amdgpu/version)"
bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/clear_foreign_containers.sh "$SLURM_JOB_ID"
for mf in 0.5 0.65 0.72; do MEMPROBE=1 MEMFRAC=$mf bash /perf_apps/liyingli/bench_agentx/kopt-20261109/bench/out/pd_startup/pd_startup_test.run-616ctl.sh infera 8 12 nograph; done
echo "$(date -u +%FT%TZ) done"
