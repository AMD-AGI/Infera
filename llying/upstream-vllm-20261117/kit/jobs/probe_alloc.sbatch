#!/usr/bin/env bash
#SBATCH -J lyl-up-probe
#SBATCH -p amd-spur-v2
#SBATCH -N 1
#SBATCH -t 00:05:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/upstream-20261119/logs/probe-%j.out
# What a non-exclusive allocation looks like on this cluster: which GPUs the job got (env), how they map to /dev/dri
# render nodes and PCI addresses, cgroup limits, and whether docker works. Prints only; uses no GPU.
echo "== $(date -u '+%F %T') $(hostname -s) job $SLURM_JOB_ID"
env | grep -E '^(SLURM_(JOB_GPUS|STEP_GPUS|GPUS|CPUS|MEM|JOB_CPUS|CPUS_ON_NODE|TRES)|CUDA_VISIBLE|ROCR_VISIBLE|HIP_VISIBLE|GPU_DEVICE_ORDINAL)' | sort
echo "-- nproc $(nproc); cgroup: $(cat /proc/self/cgroup | head -3 | tr '\n' ' ')"
cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null; cat /sys/fs/cgroup/memory.max 2>/dev/null
echo "-- render nodes -> PCI"
for r in /dev/dri/renderD*; do n=$(basename "$r"); echo "$n $(readlink -f /sys/class/drm/$n/device | xargs basename)"; done
echo "-- KFD nodes (gpu_id, location)"
for n in /sys/class/kfd/kfd/topology/nodes/*; do
    id=$(cat "$n/gpu_id" 2>/dev/null); [[ "$id" == 0 ]] && continue
    echo "$(basename "$n") gpu_id=$id domain/bdf=$(grep -E 'location_id|domain' "$n/properties" | tr '\n' ' ')"
done
echo "-- other jobs on this node: $(squeue -h -w "$(hostname -s)" -o '%i %u %b %C' | tr '\n' ';')"
echo "-- docker: $(timeout 60 docker run --rm --network none busybox:1.36 echo ok 2>&1 | tail -1)"
exit 0
