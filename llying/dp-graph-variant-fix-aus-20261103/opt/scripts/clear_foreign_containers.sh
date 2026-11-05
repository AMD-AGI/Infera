#!/bin/bash
# On a node of our allocation: stop (not remove) every running container that is not ours and holds GPU devices. They
# are leftovers of a job that ended or was preempted (n04-33 held yihou's for 6 h on 11-04, 8 GPUs at ~299 GB), and the
# node is ours now; our own (glm52_2p1d-*, glm52_agentx-*) are down.sh's. Prints what it stopped.
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
stop=()
while read -r name; do
    [[ -n "$name" && "$name" != glm52_2p1d-* && "$name" != glm52_agentx-* ]] || continue
    devs="$(docker inspect -f '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}{{.HostConfig.Privileged}}' "$name" 2>/dev/null)"
    [[ "$devs" == */dev/kfd* || "$devs" == */dev/dri* || "$devs" == *true ]] && stop+=("$name")
done < <(docker ps --format '{{.Names}}')
((${#stop[@]})) || exit 0
echo "$(hostname -s): stopping ${stop[*]}"
docker stop -t 30 "${stop[@]}" >/dev/null 2>&1
