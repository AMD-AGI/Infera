#!/usr/bin/env bash
#SBATCH -J lyl-vpd-once
#SBATCH -p amd-spur-v2
#SBATCH -A amd-agentx
#SBATCH -q amd-agentx-qos
#SBATCH -N 1
#SBATCH --gres gpu:8
#SBATCH --exclusive
#SBATCH -t 04:00:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/vllm-pd-20261117/logs/once-%j.out
# One-node run-and-exit job (11-19: holding nodes with idle jobs is not allowed): health check -> node_prep.sh (model
# staging, image build if missing; ONCE_PREP=0 skips) -> bash once/<ONCE>.sh from R with NODE0 / JOB set -> remove our
# containers -> exit. Put the round and its analyses in the once script so nothing needs the node afterwards.
# The step exits 0 whatever happens (spur ends the job on a non-zero step). No resubmission.
#   sbatch -x "$(paste -sd, /home/liyingli/bench_agentx/pd-repro-20261116/exclude_nodes.txt)" [-J NAME] [-t HH:MM:SS] \
#       --export=ALL,ONCE=<name> run_once.sbatch
R=/home/liyingli/bench_agentx/vllm-pd-20261117
cd "$R" || { echo "node $(hostname -s) does not see $R"; exit 0; }
f=$R/once/${ONCE:?ONCE}.sh
[ -r "$f" ] || { echo "no $f"; exit 0; }
export NODE0=$(hostname -s) JOB=$SLURM_JOB_ID
echo "== $(date -u '+%F %T') job $JOB on $NODE0: once/$ONCE.sh"
dk=$( (timeout 180 docker pull -q busybox:1.36 && timeout 60 docker run --rm --network none busybox:1.36 true && echo run-ok) 2>&1)
case "$dk" in *run-ok*|*toomanyrequests*) dk=ok ;; *) dk=fail ;; esac
r="amdgpu=$(cat /sys/module/amdgpu/version) docker=$(curl -s -m 15 -o /dev/null -w %{http_code} --unix-socket /var/run/docker.sock http://localhost/images/json) run=$dk shmem_gb=$(awk '/^Shmem:/ {print int($2 / 1048576)}' /proc/meminfo)"
echo "-- $r"
[[ "$r" == *docker=200* && "$r" == *run=ok* ]] || { echo "== node check failed; ending the job"; exit 0; }
rm_ours() { docker ps -a --format '{{.Names}}' | grep -E '^(vpd-|mcx-|up-)' | xargs -r docker rm -f > /dev/null 2>&1; }
trap 'rm_ours' EXIT TERM INT
if [[ "${ONCE_PREP:-1}" != 0 ]]; then
    bash scripts/node_prep.sh > "logs/once-$JOB-prep.log" 2>&1
    tail -1 "logs/once-$JOB-prep.log"
    grep -q "prep ok" "logs/once-$JOB-prep.log" || { echo "== node_prep failed; ending the job"; exit 0; }
fi
bash "$f" > "logs/once-$JOB-$ONCE.log" 2>&1
echo "== $(date -u '+%F %T') once/$ONCE.sh rc=$?"
exit 0
