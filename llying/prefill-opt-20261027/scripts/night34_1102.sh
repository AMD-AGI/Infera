#!/usr/bin/env bash
# 11-02 Pareto point 1 on the current best (E10, K3 SIM 2.99): C72 (model prediction: p90 ~64, ~39k tok/s/GPU; model is crude,
# this point tests it), 138 -> 140, right after night33. Then analyze_run.sh.
# Usage: night34_1102.sh LOG_ROOT
set -uo pipefail
L="$1"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
R=$P/prefill-opt-20261027/results
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
while pgrep -f "night33_110[2]" >/dev/null; do sleep 30; done
bash $S/disk_check.sh crsuse2-m2m-140 95 "before Pareto C72" || exit 0
LS=llr37-140-e10-c72
for k in 1 2; do
    out="$(cd "$KIT" && bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-140-e10-c72.txt 2>&1)"
    echo "$out"
    grep -q "up failed" <<<"$out" || break
    mv "$L/perf/$LS/c72.log" "$L/perf/$LS/c72.log.failed$k" 2>/dev/null
done
(cd $P/prefill-opt-20261027 && bash scripts/analyze_run.sh $LS 72 > $R/run-$LS-c72.txt 2>&1)
grep -E "^\| C72" $R/run-$LS-c72.txt
echo "$(date -u +%T) night34 done"
