#!/usr/bin/env bash
# 11-03 overnight, 138 -> 140:
#  1. E10 + decode custom all-reduce OFF at C104 (E10 with it on: 51,374). Correctness evidence for off: bad start 0/200 in 3 gates.
#  2. Rule: if 1. >= 50,860 (within 1% of E10) the Pareto points use CAR off, else CAR on (E10 as is).
#  3. Pareto points C88, C56, C44, C32 on the chosen config (C72 on CAR on = 43,627 @ p90 62.6 is already measured; with CAR off it is re-run).
# Usage: night35_1103.sh LOG_ROOT
set -uo pipefail
L="$1"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
R=$P/prefill-opt-20261027/results
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
point() {  # LABEL SITE CONC
    local label="$1" site="$2" conc="$3" k out
    echo "$label $site $conc" > "$C/plan-$label.txt"
    for k in 1 2; do
        out="$(cd "$KIT" && bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" "$C/plan-$label.txt" 2>&1)"
        echo "$out" | tail -3
        grep -q "up failed" <<<"$out" || break
        mv "$L/perf/$label/c$conc.log" "$L/perf/$label/c$conc.log.failed$k" 2>/dev/null
    done
    (cd $P/prefill-opt-20261027 && bash scripts/analyze_run.sh "$label" "$conc" > "$R/run-$label-c$conc.txt" 2>&1)
    grep -E "^\| C$conc " "$R/run-$label-c$conc.txt"
}
bash $S/disk_check.sh crsuse2-m2m-140 95 "before night35" || exit 0
point llr37-140-e10-nocar $C/site-140-e10-nocar.env 104
tput=$(grep -E "^\| C104 " $R/run-llr37-140-e10-nocar-c104.txt | awk -F'|' '{gsub(/[ ,]/,"",$3); print $3}')
echo "$(date -u +%T) night35: nocar C104 = ${tput:-none}"
if [[ -n "${tput:-}" && "${tput%.*}" -ge 50860 ]]; then
    site=$C/site-140-e10-nocar.env; tag=nocar; extra="72"
else
    site=$C/site-llr35-v3-evict512k-bq16.env; tag=car; extra=""
fi
echo "$(date -u +%T) night35: Pareto points on $tag"
for c in $extra 88 56 44 32; do
    bash $S/disk_check.sh crsuse2-m2m-140 95 "before $tag C$c" || exit 0
    point llr37-140-e10-$tag-c$c "$site" $c
done
echo "$(date -u +%T) night35 done"
