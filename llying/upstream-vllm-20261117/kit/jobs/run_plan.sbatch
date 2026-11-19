#!/usr/bin/env bash
#SBATCH -J lyl-up-plan
#SBATCH -p amd-spur-v2
#SBATCH -A amd-primus-cicd
#SBATCH -q amd-primus-cicd-qos
#SBATCH -N 1
#SBATCH --gres gpu:8
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/upstream-20261119/logs/plan-%j.out
# One run-and-exit job per plan (no holding, 11-19): health check -> prep for what the plan needs (PREP, list joined by + (sbatch --export splits on commas):
# v034, nightly, rust, model, p4d4 = the P4D4 image + model via p4d4/scripts/node_prep.sh) -> jobs/runner.sh over
# plans/$PLAN.plan (resumes: steps that passed with the same key are skipped) -> at most DRAIN_S (600 s, user-approved
# 11-19, pending the lead's confirmation) idle for appended steps -> remove our containers -> exit 0.
# The batch script is the only session on the node. Submit from the login node, with -t sized to the plan:
#   sbatch -x "$(paste -sd, /home/liyingli/bench_agentx/pd-repro-20261116/exclude_nodes.txt)" -J lyl-up-<plan> \
#       -t HH:MM:SS [-A amd-agentx -q amd-agentx-qos] --export=ALL,PLAN=<plan>,PREP=v034+nightly+... jobs/run_plan.sbatch
U=/home/liyingli/bench_agentx/upstream-20261119
cd "$U" || exit 0
export JOB=$SLURM_JOB_ID NODE=$(hostname -s) PLAN=${PLAN:?PLAN} PREP=${PREP:-v034}
limit=$(squeue -h -j "$JOB" -o %L 2>/dev/null)
secs=$(awk -F'[-:]' '{n = NF; s = $n + 60 * $(n - 1); if (n > 2) s += 3600 * $(n - 2); if (n > 3) s += 86400 * $(n - 3); print s}' <<< "${limit:-1:00:00}")
export END=$(( $(date +%s) + secs ))
echo "== $(date -u '+%F %T') job $JOB on $NODE: plan $PLAN, prep $PREP, ${secs} s"
dk=$( (timeout 180 docker pull -q busybox:1.36 && timeout 60 docker run --rm --network none busybox:1.36 true && echo run-ok) 2>&1)
case "$dk" in *run-ok*|*toomanyrequests*) dk=ok ;; *) dk=fail ;; esac
r="amdgpu=$(cat /sys/module/amdgpu/version) docker=$(curl -s -m 15 -o /dev/null -w %{http_code} --unix-socket /var/run/docker.sock http://localhost/images/json) run=$dk gpus=$(ls /dev/dri | grep -c renderD) shmem_gb=$(awk '/^Shmem:/ {print int($2 / 1048576)}' /proc/meminfo) memavail_gb=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)"
echo "-- $r"
[[ "$r" == *docker=200* && "$r" == *run=ok* ]] || { echo "== node check failed; ending the job"; exit 0; }
rm_ours() { docker ps -a --format '{{.Names}}' | grep -E '^(up-|vpd-)' | xargs -r docker rm -f > /dev/null 2>&1; }
trap 'rm_ours' EXIT TERM INT
bash jobs/prep_node.sh > "logs/prep-$JOB.log" 2>&1 &
cp jobs/runner.sh "/tmp/up-runner-$JOB.sh"
bash "/tmp/up-runner-$JOB.sh"
echo "== $(date -u '+%F %T') runner finished; $(grep -c . logs/plan-$PLAN-$JOB.log 2>/dev/null) log lines"
exit 0
