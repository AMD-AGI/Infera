#!/usr/bin/env bash
set -euo pipefail
ROOT=/perf_apps/liyingli/bench_agentx/router-capacity-20260928
python3 - <<'PY'
import json,time
from pathlib import Path
p=Path('/perf_apps/liyingli/bench_agentx/router-capacity-20260928/recovery-allocation.json')
while True:
    d=json.loads(p.read_text())
    if d['state']=='PREPARED':break
    if d['state']=='FAILED':raise RuntimeError(d)
    time.sleep(10)
PY
export CONFIG=$ROOT/config/r0-real.sh
set -a; source "$CONFIG"; set +a
python3 "$ROOT/scripts/bootstrap_placement.py"
python3 "$ROOT/scripts/placement_answer_probe.py"
export CONFIG=$ROOT/config/r0.sh
set -a; source "$CONFIG"; set +a
python3 "$ROOT/scripts/transition_placement.py"
python3 "$ROOT/scripts/preflight_placement.py"
python3 "$ROOT/scripts/run_placement_performance.py"
