#!/usr/bin/env bash
# The e2e points need amdgpu 6.16 on both nodes (6.19.14 double-charges pinned VRAM after Mooncake
# registers it). Any other driver: add the node to the exclude list, put this check back in the queue
# and stop the holder; resubmit_watch.sh submits a new holder with the longer exclude list.
H=/apps/liyingli/holder-infx
EX=/apps/liyingli/jobs/exclude_nodes.txt
bad=()
for n in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    v="$(srun --overlap -N1 -n1 -w "$n" cat /sys/module/amdgpu/version 2>/dev/null)"
    echo "$n amdgpu $v"
    case "$v" in 6.16.*) ;; *) bad+=("$n") ;; esac
done
[ ${#bad[@]} -eq 0 ] && exit 0
for n in "${bad[@]}"; do grep -qx "$n" "$EX" || echo "$n" >> "$EX"; done
echo "not 6.16: ${bad[*]}; excluded, asking for a new holder"
cp "$H/running/88-require-6.16.sh" "$H/queue/88-require-6.16.sh"
touch "$H/NEED_RESUBMIT" "$H/STOP"
exit 1
