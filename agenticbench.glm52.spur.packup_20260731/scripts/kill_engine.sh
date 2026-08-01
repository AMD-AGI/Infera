#!/bin/bash
# Kill ONE leg's sglang engine inside the agbench container, leaving the
# container, the kvd daemon and (on the prefill node) etcd untouched.
#
# WHY NOT RESTART THE CONTAINER: the ROCm host-alloc patch and the mooncake
# wait_event patch live in the container's site-packages, applied at runtime.
# Recreating the container silently drops both. CLAUDE.md records that the
# wedged server releases all VRAM on kill -9 of the launcher + schedulers, so a
# recreate is never needed.
#
# WHY NOT pkill -f: a broad pattern matches this very shell. Kill by explicit
# PID only.
#
# WHY THE KVD DAEMON MUST SURVIVE: this is the whole point of the replay
# experiment. The daemon owns the L3 store (22.87 GB, 12,942 entries). Killing
# it would empty the store and the replay would prove nothing.
set -u
CTR="${CTR:-agbench}"

pids=$(docker exec "$CTR" bash -c \
  "ps -eo pid,args --no-headers | grep -E 'infera\.engine\.sglang|sglang::' | grep -v grep | awk '{print \$1}'" \
  2>/dev/null | tr -d '\r')

if [ -z "$pids" ]; then
  echo "[kill_engine] no engine processes found"
else
  echo "[kill_engine] killing: $(echo $pids | tr '\n' ' ')"
  for p in $pids; do docker exec "$CTR" kill -9 "$p" 2>/dev/null || true; done
fi

# Prove the kvd daemon and etcd were NOT collateral damage.
sleep 12
echo "[kill_engine] survivors (want kvd daemon present):"
docker exec "$CTR" bash -c \
  "ps -eo pid,args --no-headers | grep -E 'infera\.kvd|etcd' | grep -v grep" 2>/dev/null \
  | sed 's/^/    /' || echo "    (none)"

echo "[kill_engine] remaining engine procs (want none):"
docker exec "$CTR" bash -c \
  "ps -eo pid,args --no-headers | grep -E 'infera\.engine\.sglang|sglang::' | grep -v grep | wc -l" \
  2>/dev/null | sed 's/^/    /'
