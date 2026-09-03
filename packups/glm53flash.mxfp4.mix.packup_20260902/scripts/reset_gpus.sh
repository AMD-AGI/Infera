#!/usr/bin/env bash
# Per-iteration GPU reset. Run BEFORE every bring-up.
#
# WHY THIS EXISTS: `docker rm -f` returns as soon as the container is gone, not
# when the kernel has reclaimed the GPU allocations. Starting the next worker too
# early makes sglang abort during distributed bootstrap with
#   "The memory capacity is unbalanced. Some GPUs may be occupied by other
#    processes. pre_model_load_memory=194.9 ..."
# which reads like a config or model problem and is really a stale process from
# the previous round. That cost a whole debug round once, so the gate is real.
#
# ─────────────────────────────────────────────────────────────────────────────
# THIS SCRIPT ONLY EVER KILLS OUR OWN PROCESSES.
#
# The version this was adapted from killed EVERY KFD process on the node. That
# is wrong here and was always wrong: this is a shared host. It currently also
# carries another user's `torchtitan-job27029`, up for days. A blanket kill
# destroys someone else's multi-day run and nothing in the filesystem or the
# scheduler stops it.
#
# So ownership is established per PID, from /proc/<pid>/cgroup, and:
#   * a PID inside a container whose name matches $OWN_CTR_RE  -> killed
#   * ANY other PID                                           -> never touched;
#     we wait for it, and if it is still holding a GPU we need when the budget
#     runs out, we ABORT and say whose it is.
# An abort here is the correct outcome. It means the node is busy, not that the
# script failed.
# ─────────────────────────────────────────────────────────────────────────────
set -u
TIMEOUT="${TIMEOUT:-180}"
# Containers this experiment owns. Anchored, so `glm53_mix` does not match a
# container merely containing that text.
OWN_CTR_RE="${OWN_CTR_RE:-^(glm53_mix|glm53_mixb|glm53_standalone|glm53_[a-z0-9_]*)$}"
# GPUs this bring-up will actually use. Only these gate the VRAM check, so a
# TP4 run on 0-3 is not blocked by a neighbour legitimately using 4-7.
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

kfd_pids() { rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print $1}'; }

# Container name for a pid, or "" when it is not in a docker container.
ctr_of() {
  local pid="$1" id
  id=$(sed -nE 's#.*/docker[-/]([0-9a-f]{12,64})(\.scope)?$#\1#p' \
         "/proc/$pid/cgroup" 2>/dev/null | head -1)
  [ -n "$id" ] || return 0
  docker inspect --format '{{.Name}}' "$id" 2>/dev/null | sed 's#^/##'
}

ours=(); foreign=()
for pid in $(kfd_pids); do
  name=$(ctr_of "$pid")
  if [[ -n "$name" && "$name" =~ $OWN_CTR_RE ]]; then
    ours+=("$pid")
  else
    foreign+=("$pid|${name:-<host>}")
  fi
done

echo "[reset] our KFD processes:     ${ours[*]:-none}"
echo "[reset] foreign KFD processes: ${foreign[*]:-none}  (these are NOT touched)"

for pid in ${ours[@]+"${ours[@]}"}; do
  echo "[reset] killing our $pid ($(ps -o comm= -p "$pid" 2>/dev/null || echo gone))"
  kill -9 "$pid" 2>/dev/null || true
done

# Gate 1: our processes are gone. A foreign process never blocks this gate --
# it is accounted for in gate 2, and only for the GPUs we asked for.
deadline=$(( $(date +%s) + TIMEOUT ))
while :; do
  left=""
  for pid in ${ours[@]+"${ours[@]}"}; do
    [ -d "/proc/$pid" ] && left="$left $pid"
  done
  [ -z "$left" ] && { echo "[reset] none of our KFD processes remain"; break; }
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "[reset] TIMEOUT: our own PIDs still alive:$left" >&2
    exit 1
  fi
  sleep 2
done

# Gate 2: the GPUs WE want are back to baseline. A dead process can hold memory
# for a moment while the driver tears down its DMA buffers -- and a live foreign
# process holds it indefinitely, which is a reason to stop, not to kill.
deadline=$(( $(date +%s) + TIMEOUT ))
want=",${GPUS},"
while :; do
  busy=$(rocm-smi --showmemuse 2>/dev/null | awk -F': ' -v want="$want" '
    /GPU\[[0-9]+\].*GPU Memory Allocated \(VRAM%\)/ {
      match($0, /GPU\[([0-9]+)\]/, m)
      if (index(want, "," m[1] ",") && $2+0 > 2) printf "%s ", m[1]
    }')
  [ -z "$busy" ] && { echo "[reset] GPUs $GPUS at VRAM baseline"; break; }
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "[reset] TIMEOUT: GPU(s) $busy still above VRAM baseline." >&2
    if [ "${#foreign[@]}" -gt 0 ]; then
      echo "[reset] Foreign KFD processes are present: ${foreign[*]}" >&2
      echo "[reset] They are NOT ours to kill. Either wait, or run on the free" >&2
      echo "[reset] GPUs by setting GPUS=<subset> (and TP to match)." >&2
    fi
    exit 1
  fi
  sleep 2
done

echo "[reset] clean"
