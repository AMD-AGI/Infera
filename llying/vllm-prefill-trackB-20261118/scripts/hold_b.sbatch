#!/usr/bin/env bash
#SBATCH -J lyl-trackB-n4
#SBATCH -p amd-spur-v2
#SBATCH -A amd-agentx
#SBATCH -q amd-burst-qos
#SBATCH -N 1
#SBATCH --gres gpu:8
#SBATCH --exclusive
#SBATCH -t 08:00:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/vllm-trackB-20261118/logs/hold-b-%j.out
# Track B's own node: the queue runner lives in the batch script and runs queue-b/*.sh (from qb4.sh) in the track-B
# work directory. Every step runs from /tmp and exits 0 (spur ends the job on a non-zero step). Start-up check: the
# work directory is visible and docker answers within 15 s, otherwise the job just ends. On exit our containers go.
R=/home/liyingli/bench_agentx/vllm-trackB-20261118
Q=$R/queue-b
me=$(hostname -s)
echo "== $(date -u '+%F %T') job $SLURM_JOB_ID on $me"
r="amdgpu=$(cat /sys/module/amdgpu/version) home=$(test -r $R/qb.sh && echo ok || echo missing) docker=$(curl -s -m 15 -o /dev/null -w %{http_code} --unix-socket /var/run/docker.sock http://localhost/images/json)"
echo "-- $r"
if [[ "$r" != *home=ok* || "$r" != *docker=200* ]]; then echo "== bad node $me"; exit 0; fi
cleanup() { touch "$Q/STOP"; docker ps -a --format '{{.Names}}' | grep -E '^vpd-' | xargs -r docker rm -f; }
trap cleanup EXIT TERM INT
mkdir -p "$Q"; rm -f "$Q/STOP"
echo "runner job $SLURM_JOB_ID on $me since $(date -u '+%F %T')" > "$Q/RUNNER"
while [ ! -e "$Q/STOP" ]; do
    for f in "$Q"/*.sh; do
        [ -e "$f" ] || continue
        id=${f%.sh}
        mv "$f" "$id.run" || continue
        ( cd "$R" && bash "$id.run" > "$id.log" 2>&1; echo $? > "$id.rc" ) &
    done
    touch "$Q/RUNNER.hb"
    sleep 2
    ls "$Q" > /dev/null
done
wait
exit 0
