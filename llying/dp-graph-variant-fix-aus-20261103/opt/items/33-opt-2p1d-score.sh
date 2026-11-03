#!/usr/bin/env bash
# Only after a clean 2P1D gate: smoke 6/6, garbled 0, bad_start <= 1 on the latest arm-C-2p1d deployment.
o=/perf_apps/liyingli/vfix-logs/opt/opt.out
lab="$(grep -aoE 'arm-C-2p1d-[0-9]{4}T[0-9]{4} probe' "$o" | tail -1 | cut -d' ' -f1)"
p="$(grep -a "$lab probe:" "$o" | tail -1)" s="$(grep -a "$lab smoke:" "$o" | tail -1)"
if [[ -z "$lab" || "$s" != *6/6* || "$p" != *" garbled=0 "* || ! "$p" =~ bad_start=[01]\  ]]; then
    echo "$(date -u +%T) opt 2p1d-score skipped: gate not clean (${lab:-no gate}; $s; $p)" >> "$o"; exit 0
fi
exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item.sh 2p1d 160 score arm-C-2p1d.env 208
