#!/usr/bin/env bash
# After the running r2 bench chain (C144 -> C160 -> C176) ends, extend the sweep toward
# the throughput peak on the same stack: C128; then C112 only if C128 beat C144 (the
# peak would lie below 128). C144 18,001 > C160 17,754 tok/s/GPU showed the peak at or
# below 144. Run detached on crsuse2-m2m-137.
set -uo pipefail
R=/home/liyingli/bench_agentx/baseline/Infera/llying/3p1d-tp4-tp8-v0521-20261012
ST="$R/runtime/runs/r2/status.tsv"
start_lines="$(wc -l < "$ST")"
echo "$(date -u +%FT%TZ) waiting for the current bench chain (status line $start_lines onward)"
while :; do
    new="$(tail -n +"$((start_lines + 1))" "$ST")"
    if grep -q $'\tFAIL' <<<"$new"; then echo "$(date -u +%FT%TZ) chain failed; not extending"; exit 1; fi
    grep -q $'\tbench\tdone' <<<"$new" && break
    sleep 60
done
tput() {  # tok/s/GPU of one finished point
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["request_metrics"]["throughput"]["per_gpu"]["total_tput_tps"])' \
        "$R/results/r2/c$1/agentx_conc$1.json"
}
echo "$(date -u +%FT%TZ) chain done; running C128"
ROUND=r2 POINTS=128 bash "$R/scripts/run_3p1d.sh" bench || exit 1
c128="$(tput 128)"; c144="$(tput 144)"
echo "$(date -u +%FT%TZ) C128=$c128 C144=$c144"
if python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)' "$c128" "$c144"; then
    echo "$(date -u +%FT%TZ) C128 > C144: running C112"
    ROUND=r2 POINTS=112 bash "$R/scripts/run_3p1d.sh" bench
else
    echo "$(date -u +%FT%TZ) C128 <= C144: peak bracketed by 128 and 160; stopping"
fi
