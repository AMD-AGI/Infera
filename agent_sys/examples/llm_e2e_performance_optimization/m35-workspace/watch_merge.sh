#!/usr/bin/env bash
# Exit as soon as merge_profiling_evidence leaves waiting_handoff, or a
# profiling_evidence handoff appears, or _on refuses. Read-only. No 2>/dev/null
# on anything whose zero is trusted.
set -uo pipefail
R="${1:?run dir}"
DEADLINE=$(( $(date +%s) + 780 ))
while :; do
  st=$(python3 - "$R" <<'PY'
import json,glob,os,sys
run=sys.argv[1]; st={}
for f in glob.glob(run+"/store/task/**/*", recursive=True):
    if not os.path.isfile(f): continue
    try: j=json.load(open(f))
    except Exception: continue
    for e in (j if isinstance(j,list) else [j]):
        if isinstance(e,dict) and e.get("closure"):
            st[e["closure"]]=e.get("state") or e.get("status") or "?"
print(f"{st.get('run_profiling_mode_on','?')}|{st.get('merge_profiling_evidence','?')}|{st.get('m3_analysis','?')}")
PY
)
  on="${st%%|*}"; rest="${st#*|}"; merge="${rest%%|*}"; m3="${rest##*|}"
  if [ "$merge" != "waiting_handoff" ] || [ "$m3" != "waiting_handoff" ]; then
    echo "MERGE MOVED at $(date -u +%H:%M:%SZ): _on=$on merge=$merge m3=$m3"; exit 0
  fi
  case "$on" in
    succeeded|failed|*validation*) : ;;
  esac
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "TIMEOUT at $(date -u +%H:%M:%SZ): _on=$on merge=$merge m3=$m3"; exit 1
  fi
  sleep 15
done
