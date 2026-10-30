#!/usr/bin/env bash
# 10-30: E9 diagnostic short run (instrumented, 20 min) on 138 -> 135 after night23 ends; prints the per-rank E9 stats.
set -uo pipefail
L="$1"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
R=$P/prefill-opt-20261027/results
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
while pgrep -f "night23_103[0]" >/dev/null; do sleep 60; done
bash $S/disk_check.sh crsuse2-m2m-135 95 "before E9 diag" || exit 0
LS=llr37-135-e9-diag
for k in 1 2; do
    out="$(cd "$KIT" && bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-135-e9-diag.txt 2>&1)"
    echo "$out"
    grep -q "up failed" <<<"$out" || break
    mv "$L/perf/$LS/c104.log" "$L/perf/$LS/c104.log.failed$k"
done
(cd $P/prefill-opt-20261027 && bash scripts/analyze_run.sh $LS 104 > $R/run-$LS-c104.txt 2>&1)
grep -E "^\| C104|^pp[0-3]:" $R/run-$LS-c104.txt
RUN=$(grep -o 'run=[^ ]*' $L/perf/$LS/c104.log | head -1 | cut -d= -f2)
grep -aE "diverged|AssertionError" $RUN/logs/glm52_2p1d-prefill0.log | head -3
for r in 0 1 2 3; do grep -a "PP$r\] E9 prewb stats" $RUN/logs/glm52_2p1d-prefill0.log | tail -1 | cut -c1-400; done
bash $S/disk_check.sh crsuse2-m2m-135 95 "after E9 diag"
echo "$(date -u +%T) night24 done"
