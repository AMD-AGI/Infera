#!/usr/bin/env bash
# Dense-layer online FP8 at 4+4 C104 (vs C + PTPC/#45800 53,666), only after a clean production suite: smoke 6/6,
# garbled 0, bad_start <= 1, probe accuracy >= 0.960, acceptance >= 3.
o=/perf_apps/liyingli/vfix-logs/opt/opt.out
lab="$(grep -aoE 'arm-C-prod-ptpcfix-densefp8-[0-9]{4}T[0-9]{4} probe' "$o" | tail -1 | cut -d' ' -f1)"
p="$(grep -a "$lab probe:" "$o" | tail -1)" s="$(grep -a "$lab smoke:" "$o" | tail -1)"
acc="$(grep -a "accept $lab" "$o" | tail -1 | grep -oE 'mean [0-9.]+' | cut -d' ' -f2)"
q="$(grep -oE '\(0\.[0-9]+\)' <<<"$p" | tr -d '()')"
if [[ -z "$lab" || "$s" != *6/6* || "$p" != *" garbled=0 "* || ! "$p" =~ bad_start=[01]\  ]] ||
    ! awk -v a="${acc:-0}" -v q="${q:-0}" 'BEGIN {exit !(a >= 3 && q >= 0.960)}'; then
    echo "$(date -u +%T) opt densefp8-score skipped: gate not clean (${lab:-no gate}; acc ${acc:-?}; $s; $p)" >> "$o"; exit 0
fi
exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item3.sh densefp8 100 score arm-C-prod-ptpcfix-densefp8.env 104
