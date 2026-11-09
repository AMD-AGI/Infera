#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: C208 score of arm-C-1p1d-pp8-megatp (TP8 decode MoE as MegaMoeTP on top of
# kopt-nocap), only after a clean 64f gate; 64i scores kopt-nocap on the same allocation right after (64c: 56,595).
o=/perf_apps/liyingli/vfix-logs/opt/opt.out
lab="$(grep -aoE "arm-C-1p1d-pp8-megatp-[0-9]{4}T[0-9]{4} probe" "$o" | tail -1 | cut -d" " -f1)"
p="$(grep -a "$lab probe:" "$o" | tail -1)" s="$(grep -a "$lab smoke:" "$o" | tail -1)"
acc="$(grep -a "accept $lab" "$o" | tail -1 | grep -oE "mean [0-9.]+" | cut -d" " -f2)"
if [[ -z "$lab" || "$s" != *6/6* || "$p" != *" garbled=0 "* || ! "$p" =~ bad_start=[01]\  ]] || ! awk -v a="${acc:-0}" "BEGIN {exit !(a >= 3.4)}"; then
    echo "$(date -u +%T) opt megatp-pp8-score2 skipped: gate not clean (${lab:-no gate}; acc ${acc:-?}; $s; $p)" >> "$o"; exit 0
fi
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh megatp-pp8-score2 125 score arm-C-1p1d-pp8-megatp.env 208
