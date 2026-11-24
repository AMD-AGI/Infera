#!/bin/bash
# Usage (on the crusoe node): cru_seq_swap.sh "<old-tag> <cfg> <gpus> <off> <conc> <prefix>" ...
# In order: wait for <old-tag> CHAIN_DONE (max 3 h), print its ledger row, stop it, launch the next load; keep >= MIN_GAP s
# between two launches (concurrent server starts stalled crusoe nodes 3x on 2026-11-21/22).
set -u
R=/shared_nfs/yihou/dsv41_vllm_pd_20261119
MIN_GAP=${MIN_GAP:-480}
last=${LAST_START:-0}
for spec in "$@"; do
  set -- $spec; OLD=$1; shift
  for i in $(seq 1 1080); do grep -q CHAIN_DONE "$R/launch/$OLD.point.log" 2>/dev/null && break; sleep 10; done
  echo "$(date -u +%H:%M:%S) $OLD $(tail -n1 "$R/launch/$OLD.point.log")"
  grep "$OLD" "$R/results/ledger.csv" | cut -d, -f1-12
  bash "$R/ws/scripts/kit/stop.sh" "$OLD" > "$R/launch/$OLD.stop.log" 2>&1
  now=$(date +%s); wait=$(( last + MIN_GAP - now )); [ "$wait" -gt 0 ] && sleep "$wait"
  bash "$R/ws/scripts/kit/cru_launch_point.sh" "$@"
  last=$(date +%s)
done
