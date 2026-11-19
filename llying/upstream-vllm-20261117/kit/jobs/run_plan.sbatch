#!/usr/bin/env bash
#SBATCH -J lyl-up-plan
#SBATCH -p amd-spur-v2
#SBATCH -A amd-primus-cicd
#SBATCH -q amd-primus-cicd-qos
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 02:00:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/upstream-20261119/logs/plan-%j.out
# One run-and-exit job per plan (no holding, 11-19): health check -> prep for what the plan needs (PREP, list joined
# by + since sbatch --export splits on commas: v034, nightly, py310, py312, rust, model, p4d4 = the P4D4 image + model)
# -> jobs/runner.sh over plans/$PLAN.plan (parallel steps; resumes: steps that passed with the same key are skipped)
# -> DRAIN_S idle for appended steps (default 0) -> remove our containers -> exit 0. The batch script is the only
# session on the node.
# Resources follow the plan (11-19 GPU-utilisation rule: only the GPUs that are used; no node exclusivity unless all 8
# GPUs are used; CPU-only work asks for CPUs only). The defaults above are a CPU-only job; pass JOB_MEM_G equal to
# --mem so containers (outside the job's cgroup) stay within it. Submit from the login node, e.g.:
#   CPU only:  sbatch -x "$(paste -sd, jobs/exclude_nodes.txt)" -c 16 --mem 64G -J lyl-up-<plan> -t HH:MM:SS \
#                  --export=ALL,PLAN=<plan>,PREP=...,JOB_MEM_G=64 jobs/run_plan.sbatch
#   2 GPUs:    ... --gres gpu:2 -c 32 --mem 256G --export=ALL,PLAN=<plan>,PREP=...,JOB_MEM_G=256 jobs/run_plan.sbatch
#   8 GPUs:    ... --gres gpu:8 --exclusive --export=ALL,PLAN=<plan>,PREP=... jobs/run_plan.sbatch
# QoS: -A amd-agentx -q amd-burst-qos (preemptible, starts at once) or -q amd-agentx-qos / -A amd-primus-cicd.
U=/home/liyingli/bench_agentx/upstream-20261119
cd "$U" || exit 0
export JOB=$SLURM_JOB_ID NODE=$(hostname -s) PLAN=${PLAN:?PLAN} PREP=${PREP:-v034}
export JOB_CPUS=${SLURM_CPUS_ON_NODE:-$(nproc)} JOB_MEM_G=${JOB_MEM_G:-}
limit=$(squeue -h -j "$JOB" -o %L 2>/dev/null)
secs=$(awk -F'[-:]' '{n = NF; s = $n + 60 * $(n - 1); if (n > 2) s += 3600 * $(n - 2); if (n > 3) s += 86400 * $(n - 3); print s}' <<< "${limit:-1:00:00}")
export END=$(( $(date +%s) + secs ))
echo "== $(date -u '+%F %T') job $JOB on $NODE: plan $PLAN, prep $PREP, ${secs} s, GPUs $(ls /dev/dri/renderD* 2> /dev/null | wc -l), CPUs $JOB_CPUS, mem cap ${JOB_MEM_G:-none} G"
dk=$( (timeout 180 docker pull -q busybox:1.36 && timeout 60 docker run --rm --network none busybox:1.36 true && echo run-ok) 2>&1)
dk_out=$dk
case "$dk" in *run-ok*|*toomanyrequests*) dk=ok ;; *) dk=fail ;; esac
r="amdgpu=$(cat /sys/module/amdgpu/version) docker=$(curl -s -m 15 -o /dev/null -w %{http_code} --unix-socket /var/run/docker.sock http://localhost/images/json) run=$dk gpus=$(ls /dev/dri | grep -c renderD) shmem_gb=$(awk '/^Shmem:/ {print int($2 / 1048576)}' /proc/meminfo) memavail_gb=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)"
echo "-- $r"
[[ "$r" == *docker=200* && "$r" == *run=ok* ]] || { echo "== node check failed; ending the job"; echo "$dk_out" | tail -5; exit 0; }
rm_ours() {
    # Only this job's containers (the node may be shared, also by another of our jobs); a whole-node job also clears
    # leftovers by name.
    docker ps -aq --filter "label=up.job=$JOB" | xargs -r docker rm -f > /dev/null 2>&1
    if (( $(ls /dev/dri/renderD* 2> /dev/null | wc -l) == 8 )); then
        docker ps -a --format '{{.Names}}' | grep -E '^(up-|vpd-)' | xargs -r docker rm -f > /dev/null 2>&1
    fi
    # Shared-region files a killed measurement left in /dev/shm (root-owned).
    docker run --rm -v /dev/shm:/dev/shm --entrypoint sh busybox:1.36 -c 'rm -f /dev/shm/vllm_offload_u19probe*' > /dev/null 2>&1
}
trap 'rm_ours' EXIT TERM INT
bash jobs/prep_node.sh > "logs/prep-$JOB.log" 2>&1 &
cp jobs/runner.sh "/tmp/up-runner-$JOB.sh"
bash "/tmp/up-runner-$JOB.sh"
echo "== $(date -u '+%F %T') runner finished; $(grep -c . logs/plan-$PLAN-$JOB.log 2>/dev/null) log lines"
exit 0
