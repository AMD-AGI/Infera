#!/bin/bash
# idle_watch v2 -- 2026-09-05. Supersedes p8/p9_idle_watch.sh; older copies
# report on the WRONG STAGE and should be replaced, not just left running.
#
# WHAT IT DETECTS: a kind:ai task whose agent has FINISHED -- wrote its report,
# parked in ep_poll -- while the task stays `running`. Symptom-level checks
# cannot see it: the artefacts exist, the phase line says running, the node is
# quiet, and all of that is equally true of a healthy agent mid-thought.
#
# WHAT v1 GOT WRONG (found by m4): it took the newest .jsonl ANYWHERE in the run
# tree. On a mixed graph -- ai at stages 1 and 4, `runner` at stage 2 -- the
# stuck stage is a program body with no transcript, so the probe silently
# adopted a DIFFERENT stage's transcript and reported confidently about a stage
# that had already succeeded. Every word true, the subject wrong.
#   Silence is detectable. A substituted subject is not.
# So v2 names the transcript's owning stage and refuses to compare across
# stages: if the running stage has no transcript, it says so.
#
# A HIT IS NOT A DIAGNOSIS. Read the task's last `store/event` first --
# output_absent + handling_failed means the agent's work is intact and the run
# died at the seal, not in the agent.
R="${1:?usage: idle_watch.sh <run-dir> <orchestrator-log>}"
LOG="${2:?}"
STALE="${STALE:-420}"
echo "idle_watch v2  run=$R  log=$LOG  stale=${STALE}s"
while true; do
  sleep 120
  grep -qE "^      done  run complete" "$LOG" && { echo "$(date -u +%H:%M:%S) run complete; exiting"; break; }
  ph=$(grep -E "^     phase" "$LOG" | tail -1 | sed 's/^ *phase  //')
  cur=${ph%%:*}
  newest=$(/usr/bin/find "$R" -name "*.jsonl" -path "*config/projects*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1)
  if [ -z "$newest" ]; then echo "$(date -u +%H:%M:%S) no transcript in this run yet | $ph"; continue; fi
  t=${newest%% *}; f=${newest#* }
  age=$(( $(date +%s) - ${t%.*} ))
  # Which stage does that transcript belong to? The deepest task.<uuid> in its
  # path, resolved through store/task to a closure name.
  uuid=$(echo "$f" | grep -oE 'task\.[0-9a-f-]{36}' | tail -1 | cut -d. -f2)
  # `cur` may name a NON-LEAF (m1_deploy) whose work happens in a child
  # (deploy_and_prove), so an exact match is too strict and would report BLIND
  # on a healthy run. Walk the transcript owner's `parent` chain and accept any
  # ancestor. Erring toward BLIND is the safe direction -- a missed detection is
  # visible as silence; a substituted subject is not -- but a probe that cries
  # BLIND every cycle gets ignored, which costs the same.
  owner=$(python3 -c "
import json
R,u='$R','$uuid'
def load(x):
    with open(f'{R}/store/task/{x}.json') as fh: return json.load(fh)
try:
    chain, d = [], load(u)
    while True:
        chain.append(d['closure'])
        par = d.get('parent')
        if not par: break
        d = load(par)
    print(' '.join(chain))
except Exception: print('unknown')" 2>/dev/null)
  case " $owner " in *" $cur "*) match=1 ;; *) match=0 ;; esac
  if [ "$match" = "0" ]; then
    echo "$(date -u +%H:%M:%S) BLIND: running stage '$cur' has no transcript (program body?); newest belongs to '${owner%% *}' (${age}s) -- NOT a verdict on '$cur'"
  elif [ "$age" -gt "$STALE" ]; then
    echo "$(date -u +%H:%M:%S) *** IDLE-AGENT SUSPECT on '$cur': its transcript is ${age}s stale while still running ***"
    echo "    read store/event for this task BEFORE concluding: $f"
  else
    echo "$(date -u +%H:%M:%S) ok  '$cur' transcript ${age}s"
  fi
done
