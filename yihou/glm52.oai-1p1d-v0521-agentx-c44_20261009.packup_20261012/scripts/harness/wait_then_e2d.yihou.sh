#!/usr/bin/env bash
# Wait for the leader's C40 bench on the e2c stack to post its result, then launch e2d
# (PRE_STOP e2c). Polls every 60 s, gives up at 20:00 UTC. Log: runs/e2d.waiter.log.
set -uo pipefail
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness; R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
LOG=$R/e2c.bench-c40-flushed.log; OUT=$R/e2c/agentx-c40-e2cstack-flushed
ts(){ date -u +%FT%TZ; }
while :; do
  line=$(grep -aE $'\tbench\t(ok|FAIL)' "$LOG" | tail -1)
  if [[ -n "$line" ]]; then
    echo "$(ts) bench result: $line"
    if [[ "$line" == *ok* ]]; then
      j=$(ls "$OUT"/agentx_conc*.json 2>/dev/null | head -1)
      [[ -s "$j" ]] && python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$j" ||
        { echo "$(ts) result JSON not complete yet ($j); waiting"; sleep 60; continue; }
      s1=$(stat -c %s "$j"); sleep 30; [[ "$s1" == "$(stat -c %s "$j")" ]] || continue
      echo "$(ts) result JSON complete: $j ($s1 bytes)"
    fi
    break
  fi
  [[ $(date -u +%H%M) > 2000 ]] && { echo "$(ts) gave up waiting"; exit 1; }
  sleep 60
done
cd "$H" && source env.e2d-102.yihou.sh && PRE_STOP_ENV=$H/env.e2c-102.yihou.sh exec bash pipeline.yihou.sh > "$R/e2d.pipeline.log" 2>&1
