#!/usr/bin/env bash
# Poll a launched container: health, fault, progress. Prints a verdict and exits.
set -uo pipefail
NAME="${NAME:-glm52-v0516-aiter}"
PORT="${PORT:-30000}"
TAG="${TAG:-aiter}"
MAXWAIT="${MAXWAIT:-1500}"
OUT=/mnt/vast/c_huggingface/glm52_dsa_v0516
mkdir -p "$OUT"
t0=$(date +%s)
while :; do
  now=$(date +%s); el=$((now-t0))
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "[${el}s] VERDICT=CONTAINER_EXITED"
    docker logs "$NAME" > "$OUT/${TAG}.log" 2>&1
    tail -60 "$OUT/${TAG}.log"; exit 2
  fi
  if docker logs "$NAME" 2>&1 | grep -qE "Memory access fault|HSA_STATUS_ERROR|GPU core dump|cannot get heuristic kernel"; then
    echo "[${el}s] VERDICT=GPU_FAULT"
    docker logs "$NAME" > "$OUT/${TAG}.log" 2>&1
    grep -nE "Memory access fault|HSA_STATUS_ERROR|cannot get heuristic kernel|Traceback|Error" "$OUT/${TAG}.log" | tail -30
    exit 3
  fi
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[${el}s] VERDICT=HEALTHY"
    docker logs "$NAME" > "$OUT/${TAG}.log" 2>&1
    exit 0
  fi
  if [ "$el" -gt "$MAXWAIT" ]; then
    echo "[${el}s] VERDICT=TIMEOUT"
    docker logs "$NAME" > "$OUT/${TAG}.log" 2>&1
    tail -40 "$OUT/${TAG}.log"; exit 4
  fi
  if [ $((el % 120)) -lt 20 ]; then
    echo "[${el}s] alive, loglines=$(docker logs "$NAME" 2>&1 | wc -l) $(docker logs "$NAME" 2>&1 | tail -1 | cut -c1-110)"
  fi
  sleep 20
done
