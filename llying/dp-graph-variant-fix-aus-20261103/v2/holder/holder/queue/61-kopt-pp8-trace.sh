#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 #1-#3: TP8 decode trace on 1P(PP8)+1D C208 (two 40-step profiles), only after a clean kopt gate.
# Compare per-call kernel times with opt-trace2p1d-c208 at equal graph batch sizes (ana/kbybs.py).
o=/perf_apps/liyingli/vfix-logs/opt/opt.out
lab="$(grep -aoE "arm-C-1p1d-pp8-kopt-[0-9]{4}T[0-9]{4} probe" "$o" | tail -1 | cut -d" " -f1)"
p="$(grep -a "$lab probe:" "$o" | tail -1)" s="$(grep -a "$lab smoke:" "$o" | tail -1)"
acc="$(grep -a "accept $lab" "$o" | tail -1 | grep -oE "mean [0-9.]+" | cut -d" " -f2)"
if [[ -z "$lab" || "$s" != *6/6* || "$p" != *" garbled=0 "* || ! "$p" =~ bad_start=[01]\  ]] || ! awk -v a="${acc:-0}" "BEGIN {exit !(a >= 3.4)}"; then
    echo "$(date -u +%T) opt kopt-pp8-trace skipped: gate not clean (${lab:-no gate}; acc ${acc:-?}; $s; $p)" >> "$o"; exit 0
fi
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh kopt-pp8-trace 70 trace arm-C-1p1d-pp8-kopt-trace.env 208
