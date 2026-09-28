#!/usr/bin/env bash
set -euo pipefail
export CONFIG=/perf_apps/liyingli/bench_agentx/router-capacity-20260928/config/b1.sh
set -a; source "$CONFIG"; set +a
python3 - <<'PY'
import json,os,time
from pathlib import Path
r=Path(os.environ['RUN']);end=time.monotonic()+3600
while time.monotonic()<end:
 p=r/'preflight.json'
 if p.exists() and json.loads(p.read_text()).get('passed'):
  if (r/'STATUS').exists() and 'READY_FOR_C80' in (r/'STATUS').read_text():break
 time.sleep(5)
else:raise SystemExit('Performance preflight not completed; preserve state')
PY
bash "$TRACE_RUNTIME/scripts/run-performance.sh"
