#!/usr/bin/env bash
#SBATCH -J lyl-dsv41-kfdprobe
#SBATCH -p amd-spur-v2
#SBATCH -A amd-hyperloom
#SBATCH -q amd-hyperloom-qos
#SBATCH -N 1
#SBATCH --gres gpu:4
#SBATCH -c 118
#SBATCH --mem 1200G
#SBATCH -t 00:40:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/job-%j.out
# Which host allocations make KFD evict a process's queues when their pages move or are write-protected
# (scripts/kfd_migrate_probe.py): anonymous + hipHostRegister, hipHostMalloc, HIP VMM host memory. One GPU of the
# job's four, in the engine image with docker's default isolation plus CAP_SYS_NICE (move_pages of its own pages).
# Before that it builds instr/umbp_vmm_pool.so in the image and runs its fake-HIP tests (instr/test/); after, it repeats
# the "register" method (anonymous mmap + hipHostRegister, what UMBP does) with that shim preloaded.
#   sbatch -x "$(paste -sd, exclude_nodes.txt)" [--export=ALL,SIZE_GIB=64,PAGES=4096,METHODS=register,malloc,vmm] probe_kfd.sbatch
W=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123
cd "$W" || exit 0
source scripts/env.sh
OUT=$W/logs/kfdprobe-$SLURM_JOB_ID
mkdir -p "$OUT" && chmod 777 "$OUT"
GPU=${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}; GPU=${GPU%%,*}
echo "== $(ts) job $SLURM_JOB_ID on $(hostname -s): KFD migration probe on GPU $GPU"
{
    echo "numa nodes: $(ls -d /sys/devices/system/node/node[0-9]* | wc -l)"
    for f in /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag \
             /sys/kernel/mm/transparent_hugepage/shmem_enabled /proc/sys/vm/compact_unevictable_allowed \
             /proc/sys/kernel/numa_balancing; do echo "$f: $(cat $f 2>/dev/null)"; done
    echo "amdgpu: $(cat /sys/module/amdgpu/version 2>/dev/null)"
} > "$OUT/host.txt"
docker image inspect "$IMAGE" > /dev/null 2>&1 || timeout 1800 docker load -i "$IMAGE_TAR" > /dev/null 2>&1
docker image inspect "$IMAGE" > /dev/null 2>&1 || { echo "== no $IMAGE on this node"; exit 0; }
gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [ -n "$gid" ] && gids+=(--group-add "$gid"); done
RUN=(docker run --rm --device=/dev/kfd --device=/dev/dri "${gids[@]}" --cap-add=SYS_NICE -e HIP_VISIBLE_DEVICES="$GPU"
     -v "$W/scripts:$W/scripts:ro" -v "$W/instr:$W/instr:ro" -v "$OUT:$OUT")
timeout 600 "${RUN[@]}" --entrypoint bash "$IMAGE" "$W/instr/test/run_shim_tests.sh" "$OUT" > "$OUT/shim-tests.log" 2>&1
echo "== $(ts) shim tests rc=$?: $(tail -1 "$OUT/shim-tests.log")"
timeout 1800 docker run --rm --device=/dev/kfd --device=/dev/dri "${gids[@]}" --cap-add=SYS_NICE \
    -e HIP_VISIBLE_DEVICES="$GPU" -v "$W/scripts:$W/scripts:ro" \
    --entrypoint python3 "$IMAGE" "$W/scripts/kfd_migrate_probe.py" "${SIZE_GIB:-64}" "${PAGES:-4096}" \
    "${METHODS:-register,malloc,vmm}" > "$OUT/probe.log" 2>&1
echo "== $(ts) probe rc=$?"
grep -a 'RESULT\|allocated\|pid \|Error\|error' "$OUT/probe.log" | sed 's/^/   /'
if [[ -f $OUT/umbp_vmm_pool.so ]]; then
    timeout 900 "${RUN[@]}" -e LD_PRELOAD="$OUT/umbp_vmm_pool.so" --entrypoint python3 "$IMAGE" \
        "$W/scripts/kfd_migrate_probe.py" "${SIZE_GIB:-64}" "${PAGES:-4096}" register > "$OUT/probe-shim.log" 2>&1
    echo "== $(ts) probe with the VMM shim (register method) rc=$?"
    grep -a 'RESULT\|allocated\|pid \|umbp_vmm_pool\|Error\|error' "$OUT/probe-shim.log" | sed 's/^/   /'
fi
exit 0
