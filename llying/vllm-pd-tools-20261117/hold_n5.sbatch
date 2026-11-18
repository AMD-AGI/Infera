#!/usr/bin/env bash
#SBATCH -J lyl-vpd-n5
#SBATCH -p amd-spur-v2
#SBATCH -A amd-agentx
#SBATCH -q amd-burst-qos
#SBATCH -N 1
#SBATCH --gres gpu:8
#SBATCH --exclusive
#SBATCH -t 16:00:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/vllm-pd-20261117/logs/hold-n5-%j.out
# Preemptible (amd-burst-qos) fallback node for the main vLLM PD line while the agentx-qos jobs wait: the queue runner lives in the batch script (queue-n5/, items from q5.sh), so it
# does not depend on a login-node session. Every step runs from /tmp and exits 0 (spur ends the job on a non-zero step).
# Start-up check: sees $R, docker answers images/json within 15 s, and docker can pull and start a new container (v3-106,
# 11-18: the NVMe volume was mounted over /var/lib/containerd after dockerd started, so images/json answered but every
# pull / build / run failed); a failing node is excluded and the job resubmitted (at most 4 times). A registry rate limit
# does not count as a failure. On exit our containers are removed.
R=/home/liyingli/bench_agentx/vllm-pd-20261117
Q=$R/queue-n5
EXC=/home/liyingli/bench_agentx/pd-repro-20261116/exclude_nodes.txt
me=$(hostname -s)
echo "== $(date -u '+%F %T') job $SLURM_JOB_ID on $me"
dk=$( (timeout 180 docker pull -q busybox:1.36 && timeout 60 docker run --rm --network none busybox:1.36 true && echo run-ok) 2>&1)
case "$dk" in *run-ok*|*toomanyrequests*) dk=ok ;; *) echo "-- docker: $dk"; dk=fail ;; esac
r="amdgpu=$(cat /sys/module/amdgpu/version) home=$(test -r $R/q.sh && echo ok || echo missing) docker=$(curl -s -m 15 -o /dev/null -w %{http_code} --unix-socket /var/run/docker.sock http://localhost/images/json) run=$dk"
echo "-- $r"
if [[ "$r" != *home=ok* || "$r" != *docker=200* || "$r" != *run=ok* ]]; then
    grep -qx "$me" "$EXC" || echo "$me" >> "$EXC"
    k=$(cat "$R/.resub_n5" 2>/dev/null || echo 0)
    if ((k < 4)); then echo $((k + 1)) > "$R/.resub_n5"; echo "== bad node; resubmit: $(sbatch -x "$(paste -sd, "$EXC")" "$R/hold_n5.sbatch" 2>&1)"; fi
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
