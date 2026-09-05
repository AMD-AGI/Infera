#!/bin/bash
# Stop one line completely: agents, then orchestrator, then what it brought up.
#
#     sh assets/lib/stop_line.sh <orchestrator-pid> <jobid> <container-prefix>
#     sh assets/lib/stop_line.sh 287927 112862 yihou_w17m1f
#
# **THREE steps. Every teardown on 2026-09-05 did two.** Mine did, and so did the
# leader's when they stopped a run of mine that had a live engine at the time —
# that one got away with it only because the container went down with the run.
# *That a given teardown happened to create nothing does not make the omission
# safe.* The step that is missing is always the third.
#
#     1. agents        by PROCESS PARENTAGE
#     2. orchestrator
#     3. CONTAINERS    docker stop -t 10, never rm -f
#     4. re-check      orphan agents whose parent already died
#
# ## Why parentage, and never a name or a grep
#
# Every agent's command line is identical, so `pgrep -f 'claude/versions'` cannot
# tell one owner's from another's. The leader measured the other tempting
# discriminator too: grepping the candidate run directories for a run tag matched
# **all three candidates for both tags**, because every run tree contains a copy
# of the package and the package mentions every tag. `pgrep -P <orchestrator>` is
# the only thing here that is about ownership.
#
# ## Why agents before containers, which is not cosmetic
#
# **An agent outlives its orchestrator and will create NEW containers after you
# think you have finished.** Measured by m5, 2026-09-05: an orphaned agent
# re-created an arm at 15:00:09, one minute after its containers were stopped.
# Stopping containers first is a step the agent undoes.
#
# ## The guard on line `PREFIX`, which is somebody else's cost
#
# m2, 2026-09-05, in their own words after their teardown script sent SIGTERM to
# every agent on the login node — four agents, one intended:
#
#     "I let a computed value flow into a destructive predicate without checking
#      it was non-empty -- [ -n "$R" ] || exit was one line and I did not write it."
#
# Their `ls -dt .../runs/20260905T193*/` missed because their own run was stamped
# `1929`; the empty result turned `case "$c" in *"$(basename ...)"*)` into `* *`,
# a match-all. **Here the equivalent is `PREFIX`: empty would make the container
# filter `grep '^'`, which matches every container on the node, including other
# owners' engines.** So it is required, checked non-empty, and required to carry
# `yihou` — the same substring the deletion rule uses to decide what is ours.
#
# ## What "clean" at the end does and does not mean
#
# m4, 2026-09-05: **zero containers is equally consistent with "torn down" and
# "not yet brought up".** This script therefore reports *nothing of this run is
# left*, which is what it can support. It does **not** report that the node is
# idle, and the card readings it prints are context, not a claim of ownership —
# another line's engine may be in either state behind them.
set -uo pipefail

PID="${1:?usage: stop_line.sh <orchestrator-pid> <jobid> <container-prefix>}"
JOBID="${2:?jobid — required, this script never guesses which node}"
PREFIX="${3:?container prefix, e.g. yihou_w17m1f — see the guard note above}"

# Belt and braces over `${3:?}`: an argument that is present but empty, or that
# does not name us, must stop the script rather than widen its blast radius.
[ -n "$PREFIX" ] || { echo "stop_line: empty container prefix — refusing"; exit 2; }
case "$PREFIX" in
  *yihou*) : ;;
  *) echo "stop_line: prefix '$PREFIX' does not contain 'yihou' — refusing, this script only stops our own containers"; exit 2 ;;
esac
case "$PID" in
  ''|*[!0-9]*) echo "stop_line: pid '$PID' is not numeric — refusing"; exit 2 ;;
esac

echo "=== 1/4 agents (children of $PID) ==="
KIDS=$(pgrep -P "$PID" 2>/dev/null || true)
if [ -z "$KIDS" ]; then
  echo "  none — note this is a MEASUREMENT, not a null result: an orchestrator"
  echo "  with no agent child means the agent already died, which is how you tell"
  echo "  an external kill from a run that ended on its own."
else
  for k in $KIDS; do echo "  SIGTERM agent $k"; kill "$k" 2>/dev/null || true; done
  sleep 5
  for k in $KIDS; do [ -d "/proc/$k" ] && echo "  STILL ALIVE: $k" || echo "  gone: $k"; done
fi

echo "=== 2/4 orchestrator $PID ==="
if [ -d "/proc/$PID" ]; then
  kill "$PID" 2>/dev/null || true
  sleep 5
  [ -d "/proc/$PID" ] && echo "  STILL ALIVE — investigate before escalating to -9" || echo "  gone: $PID"
else
  echo "  already gone"
fi

echo "=== 3/4 containers matching ^${PREFIX} on job ${JOBID} ==="
# `docker stop -t 10`, never `rm -f` (standing rule 1).
spur exec "$JOBID" bash -c "
  names=\$(docker ps --format '{{.Names}}' | grep '^${PREFIX}' || true)
  if [ -z \"\$names\" ]; then
    echo '  none running under this prefix'
  else
    for n in \$names; do echo \"  docker stop -t 10 \$n\"; docker stop -t 10 \"\$n\" >/dev/null; done
  fi
  echo '  --- after ---'
  left=\$(docker ps --format '{{.Names}}' | grep '^${PREFIX}' || true)
  if [ -z \"\$left\" ]; then echo '  nothing of this run is left (NOT a claim that the node is idle)'; else echo \"\$left\"; fi
  echo '  --- cards, as context only ---'
  rocm-smi --showmemuse --csv 2>/dev/null | awk -F, '/^card/{print \"  \"\$1\"=\"\$2}'
"

echo "=== 4/4 orphan agents anywhere under a run tree ==="
# Step 1 cannot see an agent whose parent already died — it is no longer anyone's
# child. `cwd` is the discriminator for those, and it must be read rather than
# assumed. This lists EVERY owner's agents, so a hit here is a question, not a
# target: check the run directory against your own before acting.
found=0
for p in $(pgrep -f 'claude/versions' 2>/dev/null); do
  cwd=$(readlink "/proc/$p/cwd" 2>/dev/null || true)
  case "$cwd" in
    *agent_sys_runroot/runs/*) echo "  candidate $p -> $cwd"; found=1 ;;
  esac
done
[ "$found" = 0 ] && echo "  none"
exit 0
