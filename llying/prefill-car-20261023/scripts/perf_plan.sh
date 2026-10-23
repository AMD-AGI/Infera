#!/usr/bin/env bash
# AgentX points from a plan file, one fresh deployment each (../../2p1d-v0522-20261020/scripts/pd_perf_chain.sh:
# waits for both nodes' VRAM, SIM_ACCEPT_LEN=3.61, 3600 s, guard, /metrics before down.sh). A plan line is
# "LABEL SITE_ENV CONC [KEY=VALUE ...]"; KEY=VALUE pairs override the kit / site settings for that point (a value
# with spaces uses '|' for them). Lines are read one at a time, so the rest of the plan can be edited while it runs;
# a line starting with '#' is skipped. Results go to LOG_ROOT/LABEL/.
# Usage: perf_plan.sh KIT LOG_ROOT PLAN_FILE
set -uo pipefail
kit="$1" log_root="$2" plan="$3"
chain="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../2p1d-v0522-20261020/scripts" && pwd)/pd_perf_chain.sh"
n=1
while line="$(sed -n "${n}p" "$plan")" && [[ -n "$line" ]]; do
    n=$((n + 1))
    [[ "$line" == \#* ]] && continue
    read -ra f <<<"$line"
    label="${f[0]}" site="${f[1]}" conc="${f[2]}"
    over=()
    for kv in "${f[@]:3}"; do over+=("${kv//|/ }"); done
    echo "$(date -u +%T) $label: $(basename "$site") C$conc ${over[*]}"
    env "${over[@]}" bash "$chain" "$kit" "$site" "$log_root/$label" "$conc"
    echo "$(date -u +%T) $label done"
done
echo "$(date -u +%T) plan done"
