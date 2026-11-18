#!/usr/bin/env bash
# On the node: score rows and 5-min prefill hit timelines of the track-B P4D4 runs, into results/compare.txt.
#   compare_runs.sh RUN...
R=${R:-/home/liyingli/bench_agentx/vllm-trackB-20261118}
cd "$R" || exit 1
IP=$(ip -o -4 route get 8.8.8.8 | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}')
out=results/compare.txt
: > "$out"
for run in "$@"; do
    d=runs/$run
    {
        echo "=== $run"
        grep -h "score $run " logs/summary.out | tail -1 | cut -c1-420
        grep -m1 -E "Phase profiling \(profiling\) started" "$d/agentx-c104/runner.log" 2>/dev/null | cut -c1-12 | sed 's/^/measurement start /'
        python3 scripts/hit_timeline.py "$d/samples.jsonl" "http://$IP:27201" 300 2>/dev/null
        echo "prefill MEMORY_FAULT lines: $(cat "$d/prefill-faults.txt" 2>/dev/null)"
    } >> "$out"
done
cat "$out"
