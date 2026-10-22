#!/usr/bin/env bash
# Advance this PID namespace's PID counter by forking K throwaway subshells, then run "$@" as a CHILD
# (exec would keep this shell's PID, e.g. 1; in production the PIDs that matter are the rank
# processes forked later, which an exec'd launcher still advances).
# usage: pidoffset_exec.yihou.sh <K> <cmd...>
K=${1:?K}; shift
t0=$(date +%s.%N); i=0
while [ $i -lt "$K" ]; do ( : ); i=$((i+1)); done
echo "OFFSET_LOOP_S=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}') K=$K next_pid_after=$$" >&2
"$@" & child=$!; echo "CHILD_PID=$child" >&2; wait $child
