#!/bin/bash
# Usage (on the crusoe node): cru_seq_launch.sh "<cfg> <gpus> <off> <conc> <prefix>" ...
# Launch points one by one; start the next only after the previous server logs "ready" (max 20 min each).
set -u
R=/shared_nfs/yihou/dsv41_vllm_pd_20261119
for spec in "$@"; do
  out=$(bash "$R/ws/scripts/kit/cru_launch_point.sh" $spec); echo "$out"
  T=$(echo "$out" | awk '/^NEW /{print $2}')
  for i in $(seq 1 120); do grep -q "ready after" "$R/launch/$T.serve.log" 2>/dev/null && break; grep -qiE "error|failed" "$R/launch/$T.serve.log" 2>/dev/null && break; sleep 10; done
  echo "$(date -u +%H:%M:%S) $T: $(tail -n1 "$R/launch/$T.serve.log" | cut -c1-80)"
done
