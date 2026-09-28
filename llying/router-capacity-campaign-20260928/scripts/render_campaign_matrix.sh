#!/usr/bin/env bash
set -euo pipefail
export MPLCONFIGDIR=/tmp/bench-campaign-matplotlib
# Keep the distro matplotlib/NumPy pair separate from benchmark dependencies.
python3 -S - "$@" <<'PY'
import sys,runpy
sys.path.insert(0,'/usr/lib/python3/dist-packages')
script='/perf_apps/liyingli/bench_agentx/router-capacity-20260928/scripts/campaign_matrix.py'
sys.argv=[script,*sys.argv[1:]]
runpy.run_path(script,run_name='__main__')
PY
