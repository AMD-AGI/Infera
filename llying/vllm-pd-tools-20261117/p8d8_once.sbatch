#!/usr/bin/env bash
#SBATCH -J lyl-vpd-p8d8
#SBATCH -p amd-spur-v2
#SBATCH -A amd-agentx
#SBATCH -q amd-agentx-qos
#SBATCH -N 2
#SBATCH --gres gpu:8
#SBATCH --exclusive
#SBATCH -t 04:00:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/vllm-pd-20261117/logs/p8d8-once-%j.out
# One P8D8 scoring round as a run-and-exit job (11-19: holding nodes with idle jobs is not allowed). NODE0 runs this
# script (etcd, prefill PP8, router, AgentX); NODE1 runs exactly one step, scripts/job_runner.sh over queue-j<JOB>/,
# which executes the decode-side items p8d8_run.sh drops there. Order: health check on both nodes -> node_prep on both
# (model staging, image build if missing) -> p8d8_run.sh -> analyses (scripts/p8d8_finish.sh) -> release. Every step
# runs from /tmp and exits 0 (spur ends the job on a non-zero step). No resubmission: a failed check ends the job.
#   sbatch --export=ALL,P8D8_TAG=<tag> p8d8_once.sbatch     (p8d8-conf/<tag>.env: K=V lines for p8d8_run.sh)
R=/home/liyingli/bench_agentx/vllm-pd-20261117
J=$SLURM_JOB_ID
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
me=$(hostname -s)
[[ "$me" == "${nodes[0]%%.*}" ]] || exit 0
NODE1=${nodes[1]%%.*}
cd "$R" || { echo "batch node $me does not see $R"; exit 0; }
QD=queue-j$J
Q=$R/$QD
tag=${P8D8_TAG:?P8D8_TAG}
conf=$R/p8d8-conf/$tag.env
[ -r "$conf" ] || { echo "no $conf"; exit 0; }
mkdir -p "$Q"
echo "== $(date -u '+%F %T') job $J: NODE0=$me NODE1=$NODE1 tag=$tag"
srun --overlap -D /tmp -N1 -n1 -w "$NODE1" --gres=gpu:8 /bin/bash -c "bash $R/scripts/job_runner.sh $Q; exit 0" \
    < /dev/null >> "$R/logs/p8d8-once-$J-node1.log" 2>&1 &
step=$!
rm_ours() { docker ps -a --format '{{.Names}}' | grep -E '^vpd-' | xargs -r docker rm -f > /dev/null 2>&1; }
cleanup() { echo "== $(date -u '+%F %T') cleanup"; touch "$Q/STOP"; rm_ours; }
trap cleanup EXIT TERM INT
# Run a command on NODE1 through its runner and wait for it, at most TIMEOUT s (prints its log).
#   on1 NAME TIMEOUT COMMAND...
on1() {
    local id="$Q/$(date -u +%m%d-%H%M%S)-$1" tmo=$2 t=0; shift 2
    { echo "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=$HOME USER=$USER R=$R"
      echo "cd $R; $*"; } > "$id.tmp" && mv "$id.tmp" "$id.sh"
    while [ ! -e "$id.rc" ]; do
        sleep 5; t=$((t + 5)); ls "$Q" > /dev/null
        ((t > tmo)) && { echo "on1 $id: no rc after $tmo s"; return 1; }
    done
    cat "$id.log"
}
check='dk=$( (timeout 180 docker pull -q busybox:1.36 && timeout 60 docker run --rm --network none busybox:1.36 true && echo run-ok) 2>&1); case "$dk" in *run-ok*|*toomanyrequests*) dk=ok ;; *) dk=fail ;; esac; echo "amdgpu=$(cat /sys/module/amdgpu/version) home=$(test -r '$R'/q.sh && echo ok || echo missing) docker=$(curl -s -m 15 -o /dev/null -w %{http_code} --unix-socket /var/run/docker.sock http://localhost/images/json) run=$dk shmem_gb=$(awk "/^Shmem:/ {print int(\$2 / 1048576)}" /proc/meminfo)"'
r0=$(bash -c "$check")
r1=$(on1 check 600 "$check")
echo "-- $me: $r0"; echo "-- $NODE1: $r1"
for r in "$r0" "$r1"; do
    if [[ "$r" != *home=ok* || "$r" != *docker=200* || "$r" != *run=ok* ]]; then echo "== node check failed; ending the job"; exit 0; fi
done
( bash scripts/node_prep.sh > "$Q/prep-node0.log" 2>&1 ) &
p0=$!
p1=$(on1 prep 3600 "bash scripts/node_prep.sh")
wait $p0
tail -2 "$Q/prep-node0.log"; echo "$p1" | tail -2
if ! grep -q "prep ok" "$Q/prep-node0.log" || [[ "$p1" != *"prep ok"* ]]; then echo "== node_prep failed; ending the job"; exit 0; fi
echo "== $(date -u '+%F %T') p8d8_run.sh $tag with $conf: $(grep -v '^#' "$conf" | tr '\n' ' ')"
( set -a; . "$conf"; set +a; QD="$QD" bash scripts/snap.sh p8d8_run.sh "$tag" )
bash scripts/p8d8_finish.sh "p8d8-$tag" "p8d8-$tag" "$( (. "$conf"; echo "${CONC:-208}") )"
touch "$Q/STOP"
wait $step
echo "== $(date -u '+%F %T') done"
exit 0
