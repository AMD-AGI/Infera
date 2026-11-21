#!/usr/bin/env bash
#SBATCH -J lyl-ix5138-q
#SBATCH -p amd-spur-v2
#SBATCH -A amd-agentx
#SBATCH -q amd-agentx-qos
#SBATCH -N 1
#SBATCH --gres gpu:8
#SBATCH --exclusive
#SBATCH -t 05:30:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/ix5138-crusoe-20261120/logs/queue-%j.out
# Runs the one-node points of a run list one after another, then ends. Each list line: LABEL CONFIG CONC fast|full
# PLACE... (point.sh arguments after the port base). A point already recorded ok in runs/points.jsonl is skipped, so a
# resubmitted list resumes; a point is not started unless the job's time left covers it (100 min full, 50 min fast).
# Spur ends the job when any step exits non-zero, so the script exits 0.
# Usage: sbatch -x <exclude> queue.sbatch LIST_FILE
source /home/liyingli/bench_agentx/ix5138-crusoe-20261120/harness/lib.sh
list=$1
NODE_HOSTS=("$(hostname -s)")
say "queue job $SLURM_JOB_ID on ${NODE_HOSTS[0]}: $list"
drv=$(cat /sys/module/amdgpu/version)
gpus=$(tr ',' '\n' <<<"${SPUR_JOB_GPUS:-}" | grep -c .)
docker_ok=$(curl -s -m 15 -o /dev/null -w '%{http_code}' --unix-socket /var/run/docker.sock http://localhost/images/json)
echo "amdgpu $drv, job GPUs $gpus, docker $docker_ok, VRAM % $(vram_pct 0)"
ok=1
[[ "$drv" == 6.16.13 && "$gpus" == 8 && "$docker_ok" == 200 && -r "$list" ]] || ok=0
((ok)) && { wait_idle 0 120 || ok=0; }
if ((!ok)); then
    resubmit_bad_node "$list.resubmits" 3 "$R/harness/queue.sbatch" "$list"
    exit 0
fi
pull_images 0 || { say "image pull failed"; exit 0; }
stage_model 0 || { say "model staging failed"; exit 0; }
end=$(date -d "$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'EndTime=\K\S+')" +%s)
while read -r label config conc mode places; do
    [[ -z "$label" || "$label" == \#* ]] && continue
    if grep -q "\"label\": \"$label\", .*\"status\": \"ok\"" "$R/runs/points.jsonl" 2>/dev/null; then
        say "$label already done, skipped"
        continue
    fi
    need=$([[ "$mode" == full ]] && echo 6000 || echo 3000)
    if (($(date +%s) + need > end)); then
        say "$label skipped: $(((end - $(date +%s)) / 60)) min left, needs $((need / 60))"
        continue
    fi
    # shellcheck disable=SC2086  # places are separate point.sh arguments
    NODE_HOSTS="${NODE_HOSTS[*]}" bash "$R/harness/point.sh" "$label" "$config" "$conc" "$mode" 30000 $places </dev/null
    tail -1 "$R/runs/points.jsonl"
done <"$list"
say "queue job done"
exit 0
