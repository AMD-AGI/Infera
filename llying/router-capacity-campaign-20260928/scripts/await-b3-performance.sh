#!/usr/bin/env bash
set -euo pipefail
ROOT=/perf_apps/liyingli/bench_agentx/router-capacity-20260928
REPO=/home/liyingli/code/bench_agentx/Infera
RUN=$ROOT/runs/campaign-b2-radix
python3 - "$RUN" <<'PY'
import json,sys,time
from pathlib import Path
r=Path(sys.argv[1]);deadline=time.monotonic()+7200
while time.monotonic()<deadline:
 p=r/'STATUS';native=r/'analysis/decode-local-prefix.json'
 if p.exists() and 'COMPLETE_REVIEW_PENDING' in p.read_text() and native.exists():
  x=json.loads(native.read_text());assert x['event_coverage']==1.0,x
  break
 time.sleep(10)
else:raise SystemExit('B2 not complete with full D-prefix evidence; do not start B3')
PY
python3 "$ROOT/scripts/compare_matched_requests.py" "$ROOT/../session-affinity-31999-20260928/runs/session-affinity-31999-performance" "$RUN" --input-tolerance 8 --output "$RUN/analysis/matched-baseline"
python3 "$ROOT/scripts/archive_point.py" "$RUN" "$REPO/llying/router-capacity-campaign-20260928/b2"
python3 - "$REPO" <<'PY'
import datetime,json,sys
from pathlib import Path
root=Path(sys.argv[1])/'llying/router-capacity-campaign-20260928'
x=json.loads((root/'b2/REVIEW.json').read_text());assert x['checks_passed'],x['checks']
p=root/'CURRENT.json';s=json.loads(p.read_text());s.update(stage='B3_STARTING',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run='/perf_apps/liyingli/bench_agentx/router-capacity-20260928/runs/campaign-b3-decode-affinity',config='/perf_apps/liyingli/bench_agentx/router-capacity-20260928/config/b3.sh',already_running=['B2 quality checks passed; B3 controller-only transition and C80 starting','allocation watcher active']);s['containers'].update(router='llying-campaign-b3-router',collector='llying-campaign-b3-collector');p.write_text(json.dumps(s,indent=2)+'\n')
PY
git -C "$REPO" add llying/router-capacity-campaign-20260928/b2 llying/router-capacity-campaign-20260928/CURRENT.json
git -C "$REPO" commit -m 'Complete isolated D radix evaluation and begin decode-affinity comparison'
GIT_SSH_COMMAND='ssh -F /dev/null -o BatchMode=yes' git -C "$REPO" push origin dev/pd_opt/glm_5.2_agentx || echo PUSH_NEEDS_REVIEW
CONFIG=$ROOT/config/b3.sh bash "$ROOT/scripts/run-two-node-point.sh"
