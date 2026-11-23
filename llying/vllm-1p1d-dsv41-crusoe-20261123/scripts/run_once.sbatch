#!/usr/bin/env bash
#SBATCH -J lyl-dsv41-c640
#SBATCH -p amd-spur-v2
#SBATCH -A amd-hyperloom
#SBATCH -q amd-hyperloom-qos
#SBATCH -N 1
#SBATCH --gres gpu:4
#SBATCH -c 118
#SBATCH --mem 1200G
#SBATCH -t 06:00:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/job-%j.out
# One-node run-and-exit job on half a node (4 GPUs, half the CPUs and memory; crusoe does not allow holding idle
# nodes): node check -> node_prep.sh (weights to NVMe, engine image) -> pipeline.sh (up, smoke, accuracy, kv-aware,
# golden, AgentX CONC) -> remove our containers -> exit 0 (spur ends the
# job on a non-zero step; nothing is resubmitted). Prefill takes the two lower GPUs the job was given, decode the two
# higher; the containers see every GPU, so the GPU lists are always explicit.
#   sbatch -x "$(paste -sd, exclude_nodes.txt)" [--export=ALL,CONC=640] run_once.sbatch
W=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123
cd "$W" || { echo "node $(hostname -s) does not see $W"; exit 0; }
source scripts/env.sh
JOB=$SLURM_JOB_ID
echo "== $(ts) job $JOB on $(hostname -s): ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-}"
mapfile -t g < <(tr ',' '\n' <<<"${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-${GPU_DEVICE_ORDINAL:-}}}" | grep -E '^[0-7]$' | sort -n)
(( ${#g[@]} == 4 )) || { echo "== the job did not name 4 GPUs (${g[*]:-none}); ending the job"; exit 0; }
export PREFILL_GPUS="${g[0]},${g[1]}" DECODE_GPUS="${g[2]},${g[3]}"
echo "-- prefill GPUs $PREFILL_GPUS, decode GPUs $DECODE_GPUS; co-tenants: $(squeue -w "$(hostname -s)" -h -o '%i:%u:%b:%C' 2>/dev/null | tr '\n' ' ')"
# Some nodes do not give this account its docker / video / render groups (v3-157 on 11-23).
id -nG | tr ' ' '\n' | grep -qx docker || { echo "== $(id -nG): no docker group on this node; ending the job"; exit 0; }
dk=$( (timeout 180 docker pull -q busybox:1.36 && timeout 60 docker run --rm --network none busybox:1.36 true && echo run-ok) 2>&1)
case "$dk" in *run-ok*|*toomanyrequests*) dk=ok ;; *) dk=fail ;; esac
memavail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
shmfree=$(df -BG --output=avail /dev/shm | tail -1 | tr -dc 0-9)
r="amdgpu=$(cat /sys/module/amdgpu/version) docker=$(curl -s -m 15 -o /dev/null -w %{http_code} --unix-socket /var/run/docker.sock http://localhost/images/json) run=$dk memavail_gb=$memavail shm_free_gb=$shmfree"
echo "-- $r"
[[ "$r" == *docker=200* && "$r" == *run=ok* ]] || { echo "== node check failed; ending the job"; exit 0; }
# UMBP 449 GB + two engines' Engram tables (2 x 189 GiB in /dev/shm) + engine and client processes.
(( memavail >= 1100 && shmfree >= 450 )) || { echo "== not enough free host memory / shm for this run; ending the job"; exit 0; }
rm_ours() { docker ps -a --format '{{.Names}}' | grep -E '^lyl-dsv41' | xargs -r docker rm -f > /dev/null 2>&1; rm -rf /dev/shm/lyl-dsv41-umbp; }
trap rm_ours EXIT TERM INT

bash scripts/node_prep.sh > "logs/job-$JOB-prep.log" 2>&1
tail -1 "logs/job-$JOB-prep.log"
grep -q "^prep ok" "logs/job-$JOB-prep.log" || { echo "== node_prep failed; ending the job"; exit 0; }

CONC=${CONC:-640} bash scripts/pipeline.sh > "logs/job-$JOB-pipeline.log" 2>&1
echo "== $(ts) pipeline rc=$?"

echo "== $(ts) done"
exit 0
