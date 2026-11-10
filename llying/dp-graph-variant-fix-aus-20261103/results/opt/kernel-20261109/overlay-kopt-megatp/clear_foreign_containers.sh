#!/bin/bash
# On a node of our allocation: stop (not remove) running containers that are not ours and hold GPU memory (sum of
# /sys/class/kfd/kfd/proc/<pid>/vram_* over the container's processes, matched by the container id in /proc/<pid>/cgroup).
# Foreign containers with GPU devices but no GPU memory are left running (user decision 11-09 17:04). They are leftovers
# of a job that ended or was preempted (n04-33 held yihou's for 6 h on 11-04, 8 GPUs at ~299 GB); our own engine
# (glm52_2p1d-*, glm52_agentx-*) is down.sh's, and our llying-* bench containers are handled (and logged) like foreign ones.
# Every decision is appended to $LOGF so stops can be checked
# against later root cancellations (STATUS 11-07 06:10). Usage: clear_foreign_containers.sh [SLURM_JOB_ID]
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
LOGF=/apps/liyingli/holder/foreign_containers.log
MIN_VRAM=$((256 << 20))
declare -A vram
for p in /sys/class/kfd/kfd/proc/*; do
    [[ -d "$p" ]] || continue
    id="$(grep -o '[0-9a-f]\{64\}' "/proc/${p##*/}/cgroup" 2>/dev/null | head -1)"
    [[ -n "$id" ]] || continue
    v=0; for f in "$p"/vram_*; do v=$((v + $(cat "$f" 2>/dev/null || echo 0))); done
    vram[$id]=$(( ${vram[$id]:-0} + v ))
done
stop=()
while read -r name; do
    [[ -n "$name" && "$name" != glm52_2p1d-* && "$name" != glm52_agentx-* ]] || continue
    info="$(docker inspect -f '{{.Id}} {{range .HostConfig.Devices}}{{.PathOnHost}},{{end}} {{.HostConfig.Privileged}} {{.Config.Image}} {{.Created}}' "$name" 2>/dev/null)"
    read -r id devs priv image created <<< "$info"
    [[ "$devs" == */dev/kfd* || "$devs" == */dev/dri* || "$priv" == true ]] || continue
    v=${vram[$id]:-0}
    if (( v >= MIN_VRAM )); then act=stop; stop+=("$name"); else act=keep; fi
    printf '%s %s job=%s %s %s image=%s created=%s vram_gib=%s\n' "$(date -u +%FT%TZ)" "$(hostname -s)" "${1:-?}" "$act" \
        "$name" "$image" "${created%%.*}" "$(awk -v v="$v" 'BEGIN{printf "%.1f", v / 2^30}')" >> "$LOGF"
    [[ $act == keep ]] && echo "$(hostname -s): left running $name (no GPU memory)"
done < <(docker ps --format '{{.Names}}')
((${#stop[@]})) || exit 0
echo "$(hostname -s): stopping ${stop[*]} (holding GPU memory)"
docker stop -t 30 "${stop[@]}" >/dev/null 2>&1
