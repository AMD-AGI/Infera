#!/bin/bash
# Capture Python thread stacks from a wedged sglang engine.
#
# WHY THIS EXISTS: on 2026-09-06 the _on engine stopped generating at 18:35:24
# with 31 requests in flight; three of four TP ranks sat at 100 % GFX with zero
# memory bandwidth. The measurement that would have named the cause is a stack
# of the engine's Python threads *while wedged*, and it is unrecoverable after
# teardown. Run this BEFORE stopping anything.
#
# It reads. It starts one throwaway container of its own and removes nothing.
#
# Usage:  bash capture_engine_stack.sh [container-name-substring]
#         default substring: yihou_e2e_sgl
#
# The instrument was verified 2026-09-06 19:09Z against a host process with a
# planted frame name; py-spy printed that frame. --privileged is required:
# --cap-add SYS_PTRACE alone returns "Permission denied" here (yama
# ptrace_scope=1). The privileged container is ours and is --rm.
set -uo pipefail

IMG="${E2E_PYSPY_IMAGE:-lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260903}"
# ${1-default}, not ${1:-default}: an explicitly empty argument must reach the
# guard below. With :- it silently became the default and the guard was
# unreachable code — caught by running the control, not by reading it.
MATCH="${1-yihou_e2e_sgl}"

[ -n "$MATCH" ] || { echo "ABORT: empty container match"; exit 1; }

CTR=$(docker ps --format '{{.Names}}' | grep -a "$MATCH" | head -1)
[ -n "$CTR" ] || { echo "ABORT: no running container matching '$MATCH'"; exit 1; }
echo "container: $CTR"

NOW=$(date -u +%Y%m%dT%H%M%SZ)
OUT="/data/yihou/e2e_verify_20260906/m2/engine_stack_${NOW}"
case "$OUT" in *yihou*) : ;; *) echo "ABORT: out path lacks yihou"; exit 1 ;; esac
mkdir -p "$OUT" || exit 1
echo "out: $OUT"

docker inspect "$CTR" > "$OUT/container.inspect.json" 2>&1

# Host-side pids of the engine's ranks. docker top gives host pids in col 2.
docker top "$CTR" > "$OUT/docker_top.txt" 2>&1
PIDS=$(awk 'NR>1 && $0 ~ /python|sglang/ {print $2}' "$OUT/docker_top.txt")
if [ -z "$PIDS" ]; then
  echo "no python/sglang rows in docker top; falling back to every pid in the container"
  PIDS=$(awk 'NR>1 {print $2}' "$OUT/docker_top.txt")
fi
[ -n "$PIDS" ] || { echo "ABORT: no pids"; exit 1; }
echo "pids: $(echo "$PIDS" | tr '\n' ' ')"

for p in $PIDS; do
  case "$p" in ''|*[!0-9]*) echo "  skip non-numeric '$p'"; continue ;; esac
  echo "--- pid $p ---"
  {
    echo "### /proc/$p/status"; cat "/proc/$p/status" 2>&1 | head -20
    echo "### /proc/$p/wchan";  cat "/proc/$p/wchan"  2>&1; echo
    echo "### /proc/$p/stack";  cat "/proc/$p/stack"  2>&1
    echo "### thread count";    ls "/proc/$p/task" 2>/dev/null | wc -l
  } > "$OUT/proc.$p.txt" 2>&1
  timeout 120 docker run --rm --pid=host --privileged --user 0 \
    --entrypoint /opt/venv/bin/py-spy "$IMG" dump --pid "$p" \
    > "$OUT/pyspy.$p.txt" 2>&1
  echo "  py-spy rc=$? bytes=$(wc -c <"$OUT/pyspy.$p.txt")"
done

rocm-smi --showuse --showmemuse --showpids > "$OUT/rocm-smi.txt" 2>&1

{
  echo "captured_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "container=$CTR"
  echo "image=$IMG"
  echo "note: mtimes of copied files describe their sources, not this capture"
  echo "note: py-spy needs --privileged here; --cap-add SYS_PTRACE alone fails (yama=1)"
} > "$OUT/GENERATED.txt"

echo "=== done: $OUT ==="
grep -l 'Thread' "$OUT"/pyspy.*.txt 2>/dev/null | sed 's/^/  has stacks: /' || echo "  NO stacks captured - read the pyspy.*.txt for the error"
