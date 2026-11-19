#!/usr/bin/env bash
#SBATCH -J lyl-up-shmclean
#SBATCH -p amd-spur-v2
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 00:10:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/upstream-20261119/logs/shmclean-%j.out
# Remove the U19 measurement's region files from /dev/shm on this node (left behind if the step was killed), as root
# through docker, and print Shmem before / after. Submit with -w <node>.
echo "== $(date -u '+%F %T') $(hostname -s) Shmem before: $(awk '/^Shmem:/ {print int($2 / 1048576)}' /proc/meminfo) GiB"
ls -la /dev/shm | grep -c vllm_offload_u19probe
docker run --rm -v /dev/shm:/dev/shm --entrypoint sh busybox:1.36 -c 'rm -f /dev/shm/vllm_offload_u19probe*'
sleep 20
echo "== Shmem after: $(awk '/^Shmem:/ {print int($2 / 1048576)}' /proc/meminfo) GiB"
exit 0
