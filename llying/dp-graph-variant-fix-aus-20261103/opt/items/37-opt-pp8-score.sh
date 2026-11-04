#!/usr/bin/env bash
# 1P(PP8)+1D C208 only after a clean PP8 gate (smoke 6/6, garbled 0, bad_start <= 1, acceptance >= 3).
o=/perf_apps/liyingli/vfix-logs/opt/opt.out
lab="$(grep -aoE 'arm-C-1p1d-pp8-[0-9]{4}T[0-9]{4} probe' "$o" | tail -1 | cut -d' ' -f1)"
p="$(grep -a "$lab probe:" "$o" | tail -1)" s="$(grep -a "$lab smoke:" "$o" | tail -1)"
acc="$(grep -a "accept $lab" "$o" | tail -1 | grep -oE 'mean [0-9.]+' | cut -d' ' -f2)"
if [[ -z "$lab" || "$s" != *6/6* || "$p" != *" garbled=0 "* || ! "$p" =~ bad_start=[01]\  ]] || ! awk -v a="${acc:-0}" 'BEGIN {exit !(a >= 3)}'; then
    echo "$(date -u +%T) opt pp8-score skipped: gate not clean (${lab:-no gate}; acc ${acc:-?}; $s; $p)" >> "$o"; exit 0
fi
exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item3.sh pp8 125 score arm-C-1p1d-pp8.env 208
