#!/usr/bin/env bash
# Score C + PTPC with #45800 only after a clean gate with working drafts (acceptance >= 3, garbled 0, bad_start <= 1).
o=/perf_apps/liyingli/vfix-logs/opt/opt.out
lab="$(grep -aoE 'arm-C-prod-ptpcfix-[0-9]{4}T[0-9]{4} probe' "$o" | tail -1 | cut -d' ' -f1)"
p="$(grep -a "$lab probe:" "$o" | tail -1)" acc="$(grep -a "acc-ptpcfix accept $lab" "$o" | tail -1 | grep -oE 'mean [0-9.]+' | cut -d' ' -f2)"
if [[ -z "$lab" || "$p" != *" garbled=0 "* || ! "$p" =~ bad_start=[01]\  ]] || ! awk -v a="${acc:-0}" 'BEGIN {exit !(a >= 3)}'; then
    echo "$(date -u +%T) opt Cptpcfix skipped: gate not clean or drafts still rejected (${lab:-no gate}; acc ${acc:-?}; $p)" >> "$o"; exit 0
fi
exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item2.sh Cptpcfix 95 score arm-C-prod-ptpcfix.env 104
