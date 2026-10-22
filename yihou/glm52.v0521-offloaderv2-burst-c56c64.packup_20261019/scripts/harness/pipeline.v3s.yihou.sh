#!/usr/bin/env bash
# (pipeline.v3, task v0521cap2: run.v3 = audit v3; no spur shim when running on the node itself)
# Straight-through round: [wait] -> up -> census -> smoke -> audit -> bench, stopping at the first
# failure. Stage results go to runs/<round>/status.tsv (via run.v2) and this script's stdout
# (redirect to runs/<round>.pipeline.log).
# Usage: source env.<round>.yihou.sh
#        WAIT_PREFIXES="glm52-sanity-yihou-136 yihou-ipc" GO_FILE=/path bash pipeline.yihou.sh
#   WAIT_PREFIXES  start only when no RUNNING container on the node has one of these name prefixes
#   GO_FILE        start only when this file exists (checked on the login host)
#   PRE_STOP_ENV   env file of a previous round on the same node: `run.v2 stop` it first
set -uo pipefail
H="$(cd "$(dirname "$0")" && pwd)"
RUNS=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
D="bash $H/run.v3s.yihou.sh"
ts(){ date -u +%FT%TZ; }
if [[ -n "$YIHOU_JOB" && "$(hostname -s)" != "$YIHOU_NODE" ]]; then   # spur exec shim, only off-node (v3: drivers may run ON the node with the local ssh shim)
  export PATH="/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH"
  export SPUR_SSH_JOBS="${SPUR_SSH_JOBS:-$YIHOU_NODE:$YIHOU_JOB}"
  SSH=(ssh)
else
  SSH=(ssh -o BatchMode=yes -o ClearAllForwardings=yes)
fi

if [[ -n "${PRE_STOP_ENV:-}" ]]; then
  echo "$(ts) stage pre-stop ($PRE_STOP_ENV)"
  # stop.sh also exits 1 when its residual `rocm-smi --showpids` probe returns non-zero (seen on
  # 102), so judge by what is still running instead of by its exit code.
  ( source "$PRE_STOP_ENV" && $D stop ) || echo "$(ts) pre-stop: stop.sh rc!=0, checking containers"
  left=$( source "$PRE_STOP_ENV" && "${SSH[@]}" -n "$YIHOU_NODE" \
    "docker ps --format '{{.Names}}' | grep -c '^glm52-v0521-yihou-${YIHOU_NODE##*-}-$YIHOU_ROUND-'" )
  [[ "$left" == 0 ]] || { echo "$(ts) STOP at pre-stop: ${left:-?} containers of the old round still running"; exit 1; }
fi
if [[ -n "${WAIT_PREFIXES:-}${GO_FILE:-}" ]]; then
  pat="$(tr ' ' '\n' <<< "${WAIT_PREFIXES:-}" | grep . | sed 's/^/^/' | paste -sd'|')"
  for i in $(seq 1 480); do   # up to 8 h
    n=0
    [[ -n "$pat" ]] && n=$("${SSH[@]}" -n "$YIHOU_NODE" "docker ps --format '{{.Names}}' | grep -cE '$pat'" 2>/dev/null)
    g=yes; [[ -z "${GO_FILE:-}" || -e "$GO_FILE" ]] || g=no
    [[ "$n" == 0 && "$g" == yes ]] && { echo "$(ts) wait: done (no running [$pat], go-file=$g)"; break; }
    echo "$(ts) wait: running [$pat]=${n:-?} go-file=$g"; sleep 60
  done
  [[ "$n" == 0 && "$g" == yes ]] || { echo "$(ts) wait: gave up"; exit 1; }
fi
echo "$(ts) stage up";     $D up     || { echo "$(ts) STOP at up"; exit 1; }
c="$RUNS/$YIHOU_ROUND/census.txt"
echo "$(ts) stage census"; $D census > "$c" 2>&1; cat "$c"
if grep -qE '"Memory access fault"=[1-9]|kernel_code_entry=[1-9]|OOM/Traceback  : [1-9]|OOM/Traceback  : [0-9]+/[1-9]' "$c"; then
  echo "$(ts) STOP at census: fault/OOM/Traceback lines (see $c)"; exit 1; fi
# R020/R022: exporter pid == importer pid -> CLR hipIpcOpenMemHandle 201. Hard stop before smoke.
if ! grep -q 'rank-PID overlap prefill/decode: none' "$c"; then
  echo "$(ts) STOP at census: scheduler rank PIDs overlap between legs (or not found): $(grep -a 'rank-PID' "$c")"; exit 1; fi
echo "$(ts) stage smoke";  $D smoke  || { echo "$(ts) STOP at smoke"; exit 1; }
echo "$(ts) stage audit";  $D audit > "$RUNS/$YIHOU_ROUND/audit.txt" 2>&1; rc=$?; tail -1 "$RUNS/$YIHOU_ROUND/audit.txt"
[[ $rc == 0 ]] || { echo "$(ts) STOP at audit"; exit 1; }
echo "$(ts) stage bench";  $D bench  || { echo "$(ts) STOP at bench"; exit 1; }
echo "$(ts) PIPELINE DONE"
