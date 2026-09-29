#!/usr/bin/env bash
set -euo pipefail
ROOT=/perf_apps/liyingli/bench_agentx/router-capacity-20260928
trap 'date -u --iso-8601=seconds > "$ROOT/capacity-sequence-failed.txt"' ERR
python3 - <<'PY'
import json,time
from pathlib import Path
r=Path('/perf_apps/liyingli/bench_agentx/router-capacity-20260928');run=r/'runs/campaign-b6-2p8d8-retry1'
while True:
    if (r/'STOP_AUTORUN').exists():raise SystemExit('Stop marker set')
    ready=run/'review-ready.json'
    if ready.exists():
        try:d=json.loads(ready.read_text())
        except json.JSONDecodeError:time.sleep(1);continue
        assert d['checks_passed'];break
    status=(run/'STATUS').read_text() if (run/'STATUS').exists() else ''
    log=(r/'performance-b6-retry1.log').read_text(errors='replace')
    if 'NEEDS_REVIEW' in status or 'Traceback (most recent call last)' in log:raise RuntimeError('B6 retry needs review')
    time.sleep(15)
PY
[[ ! -e "$ROOT/STOP_AUTORUN" ]]
bash "$ROOT/scripts/run-capacity-point.sh" b7 "$ROOT/runs/campaign-b6-2p8d8-retry1" > "$ROOT/performance-b7.log" 2>&1
[[ ! -e "$ROOT/STOP_AUTORUN" ]]
python3 "$ROOT/scripts/release_retired_allocation.py" --run "$ROOT/runs/campaign-b5-p8d4-real" --topology "$ROOT/config/b5-real.json" --job 32077 --node smci355-ccs-aus-n05-29 > "$ROOT/release-extra-32077.log" 2>&1 &
retirement_pid=$!
bash "$ROOT/scripts/run-capacity-point.sh" b5 "$ROOT/runs/campaign-b7-4p4d8" > "$ROOT/performance-b5.log" 2>&1
wait "$retirement_pid"
bash "$ROOT/scripts/render_campaign_matrix.sh" /home/liyingli/code/bench_agentx/Infera/llying/router-capacity-campaign-20260928 --plot
date -u --iso-8601=seconds > "$ROOT/CAPACITY_MEASUREMENTS_COMPLETE"
