#!/usr/bin/env bash
set -euo pipefail
ROOT=/perf_apps/liyingli/bench_agentx/router-capacity-20260928
python3 - <<'PY'
import json,time
from pathlib import Path
r=Path('/perf_apps/liyingli/bench_agentx/router-capacity-20260928')
while True:
    try:
        main=json.loads((r/'recovery-allocation-32076.json').read_text())
        extra=json.loads((r/'extra-node.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):time.sleep(10);continue
    assert main['job']==32076 and extra['job']==32077
    if 'FAILED' in [main['state'],extra['state']]:raise RuntimeError((main,extra))
    if main['state']==extra['state']=='PREPARED':break
    time.sleep(10)
combined={'state':'PREPARED','nodes':main['nodes']+[extra['node']],'jobs':[{'id':32076,'nodes':main['nodes']},{'id':32077,'nodes':[extra['node']]}]}
(r/'recovery-placement-32076-32077.json').write_text(json.dumps(combined,indent=2)+'\n')
PY
export CONFIG=$ROOT/config/b6-retry1-real.sh
set -a; source "$CONFIG"; set +a
python3 "$ROOT/scripts/bootstrap_placement.py"
python3 "$ROOT/scripts/placement_answer_probe.py"
export CONFIG=$ROOT/config/b6-retry1.sh
set -a; source "$CONFIG"; set +a
python3 "$ROOT/scripts/transition_placement.py"
python3 "$ROOT/scripts/preflight_placement.py"
python3 "$ROOT/scripts/run_placement_performance.py"
python3 "$ROOT/scripts/review_point.py" "$RUN" /home/liyingli/code/bench_agentx/Infera/llying/router-capacity-campaign-20260928/b6
