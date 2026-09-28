#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import json,time
from pathlib import Path
r=Path('/perf_apps/liyingli/bench_agentx/router-capacity-20260928/runs/campaign-b2-radix-real')
end=time.monotonic()+3600
while time.monotonic()<end:
 p=r/'STATUS'
 if p.exists() and 'RADIX_GATE_PASSED' in p.read_text():
  assert json.loads((r/'radix-local-reuse-check.json').read_text())['passed']
  break
 time.sleep(5)
else:raise SystemExit('No passed radix gate; do not begin performance')
PY
CONFIG=/perf_apps/liyingli/bench_agentx/router-capacity-20260928/config/b2.sh bash /perf_apps/liyingli/bench_agentx/router-capacity-20260928/scripts/run-two-node-point.sh
