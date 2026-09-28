#!/usr/bin/env bash
set -euo pipefail
export CONFIG=/perf_apps/liyingli/bench_agentx/router-capacity-20260928/config/b1-smoke.sh
set -a; source "$CONFIG"; set +a
python3 - <<'PY'
import os,time,urllib.request
from pathlib import Path
end=time.monotonic()+1800
while time.monotonic()<end:
 try:
  with urllib.request.urlopen('http://'+os.environ['PREFILL_IP']+':28000/health',timeout=3) as f:
   if f.status==200 and (Path(os.environ['RUN'])/'launch/workers.json').exists():break
 except Exception:pass
 time.sleep(5)
else:raise SystemExit('Router readiness timeout; preserve services for review')
PY
python3 "$TRACE_RUNTIME/scripts/smoke_sessions.py" --mode prefill --ttl 3600
python3 "$TRACE_RUNTIME/scripts/check_scoring.py"
printf 'SMOKE_PASSED\n' > "$RUN/STATUS"
