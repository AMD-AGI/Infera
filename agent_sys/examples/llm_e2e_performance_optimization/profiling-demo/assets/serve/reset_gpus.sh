#!/usr/bin/env bash
# Per-iteration GPU reset gate. RUNS ON THE NODE. Run BEFORE every bring-up.
#
# A hard gate, not a best-effort sleep: starting the worker while the previous
# round still holds VRAM aborts the distributed bootstrap with a misleading
# "memory capacity is unbalanced" error.
#
# DIVERGES FROM examples/glm53flash-demo/scripts/reset_gpus.sh, on purpose.
# That version does `kill -9` on every pid `rocm-smi --showpids` reports. On a
# Slurm GPU node that set includes **slurmstepd**, which holds a KFD handle for
# the step's cgroup — observed on smci355-ccs-aus-n04-33, where the original
# script killed pid 2305766 (slurmstepd). Killing it can take down a job step
# that has nothing to do with this experiment. The demo's node is a bare host
# with no scheduler, so the difference never showed up there.
#
# So: kill only process names that are plausibly a leftover inference engine,
# never anything else. Whatever is left is reported and the VRAM floor below is
# what decides pass/fail — an unrecognised process holding VRAM is a reason to
# stop and look, not a reason to shoot.
set -u
TIMEOUT="${TIMEOUT:-180}"

# Engine leftovers we are willing to kill. `pt_main_thread` is what torch's
# spawned workers rename themselves to.
KILLABLE_RE='^(python3?|pt_main_thread|ray|sglang.*)$'
# Never, under any circumstance.
PROTECTED_RE='^(slurmstepd|slurmd|slurmctld|kubelet|containerd|dockerd|systemd)'

kfd_pids() {
  rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print $1}'
}
comm_of() { ps -o comm= -p "$1" 2>/dev/null | tr -d ' '; }

echo "[reset] KFD processes: $(for p in $(kfd_pids); do printf '%s(%s) ' "$p" "$(comm_of "$p")"; done)"

skipped=0
for pid in $(kfd_pids); do
  comm="$(comm_of "$pid")"
  [ -z "$comm" ] && continue
  if printf '%s' "$comm" | grep -qE "$PROTECTED_RE"; then
    echo "[reset] PROTECTED, not killing: $pid ($comm)"
    skipped=$((skipped + 1)); continue
  fi
  if ! printf '%s' "$comm" | grep -qE "$KILLABLE_RE"; then
    echo "[reset] unrecognised, not killing: $pid ($comm)"
    skipped=$((skipped + 1)); continue
  fi
  echo "[reset] killing $pid ($comm)"
  kill -9 "$pid" 2>/dev/null || sudo kill -9 "$pid" 2>/dev/null || true
done

# The gate is VRAM, not the process list: a protected process holding a KFD
# handle with no memory allocated is harmless, and that is the common case for
# slurmstepd.
deadline=$(( $(date +%s) + TIMEOUT ))
while :; do
  busy=$(rocm-smi --showmemuse 2>/dev/null | awk -F': ' '/GPU Memory Allocated \(VRAM%\)/ {if ($2+0 > 2) c++} END {print c+0}')
  if [ "$busy" -eq 0 ]; then
    echo "[reset] all GPUs at VRAM baseline (skipped $skipped protected/unrecognised process(es))"
    echo "[reset] clean"
    exit 0
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "[reset] TIMEOUT: $busy GPU(s) still above VRAM baseline" >&2
    rocm-smi --showpids 2>/dev/null | sed -n '3,20p' >&2
    exit 1
  fi
  sleep 2
done
