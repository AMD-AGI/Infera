#!/usr/bin/env bash
#SBATCH -J lyl-vpd-n3
#SBATCH -p amd-spur-v2
#SBATCH -A amd-agentx
#SBATCH -q amd-agentx-qos
#SBATCH -N 1
#SBATCH --gres gpu:8
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/vllm-pd-20261117/logs/hold-n3-%j.out
# Node for the main vLLM PD line after 294199: the queue runner lives in the batch script (queue-n3/, items from q3.sh), so it
# does not depend on a login-node session. Every step runs from /tmp and exits 0 (spur ends the job on a non-zero step).
# Start-up check: sees $R and docker answers images/json within 15 s; a failing node is excluded and the job resubmitted
# (at most 4 times). On exit our containers are removed.
R=/home/liyingli/bench_agentx/vllm-pd-20261117
Q=$R/queue-n3
EXC=/home/liyingli/bench_agentx/pd-repro-20261116/exclude_nodes.txt
me=$(hostname -s)
echo "== $(date -u '+%F %T') job $SLURM_JOB_ID on $me"
r="amdgpu=$(cat /sys/module/amdgpu/version) home=$(test -r $R/q.sh && echo ok || echo missing) docker=$(curl -s -m 15 -o /dev/null -w %{http_code} --unix-socket /var/run/docker.sock http://localhost/images/json)"
echo "-- $r"
if [[ "$r" != *home=ok* || "$r" != *docker=200* ]]; then
    grep -qx "$me" "$EXC" || echo "$me" >> "$EXC"
    k=$(cat "$R/.resub_n3" 2>/dev/null || echo 0)
    if ((k < 4)); then echo $((k + 1)) > "$R/.resub_n3"; echo "== bad node; resubmit: $(sbatch -x "$(paste -sd, "$EXC")" "$R/hold_n3.sbatch" 2>&1)"; fi
    exit 0
fi
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
