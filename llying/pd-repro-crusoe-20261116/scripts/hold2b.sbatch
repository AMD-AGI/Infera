#!/usr/bin/env bash
#SBATCH -J lyl-pdrepro-p8
#SBATCH -p amd-spur-v2
#SBATCH -A amd-primus
#SBATCH -q amd-primus-qos
#SBATCH -N 2
#SBATCH --gres gpu:8
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/pd-repro-20261116/logs/hold-p8-%j.out
# Holds two MI355X nodes for the P8D8 reproduction (queue3/, items from q3.sh): NODE0 = prefill, NODE1 = decode.
# Spur ends the whole job when any step exits non-zero, and a step starts in the submit directory unless told otherwise,
# so every step here runs from /tmp and exits 0 (crusoe 11-16: nodes without /home ended jobs 291051 / 291150 / 291162).
# Start-up check per node: sees $R, and docker answers images/json within 15 s (a hung containerd: v3-058, v3-220).
# A failing node goes into exclude_nodes.txt and a replacement job excluding the list is submitted (at most MAX_RESUB).
R=/home/liyingli/bench_agentx/pd-repro-20261116
Q=$R/queue3
MAX_RESUB=6
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
me=$(hostname -s)
if [[ "$me" != "${nodes[0]%%.*}" ]]; then sleep infinity; fi
export NODE0="${nodes[0]%%.*}" NODE1="${nodes[1]%%.*}" JOB="$SLURM_JOB_ID"
export SHIM_JOB="$SLURM_JOB_ID" PATH="$R/pkg/tools/sshshim:$PATH"
cd "$R" || { echo "batch node $me does not see $R"; exit 0; }
step() { srun --overlap -D /tmp -N1 -n1 -w "$1" --gres=gpu:8 /bin/bash -c "$2; exit 0" < /dev/null 2>&1 | grep -v '^srun: dispatched'; }
rm_ours='docker ps -a --format "{{.Names}}" | grep -E "^(glm52_2p1d-|llying-pdr-)" | xargs -r docker rm -f'
echo "== $(date -u '+%F %T') job $JOB: NODE0=$NODE0 NODE1=$NODE1"
bad=""
for n in "$NODE0" "$NODE1"; do
    r="$(step "$n" "echo amdgpu \$(cat /sys/module/amdgpu/version) home=\$(test -r $R/q.sh && echo ok || echo missing) \
docker=\$(curl -s -m 15 -o /dev/null -w %{http_code} --unix-socket /var/run/docker.sock http://localhost/images/json) \
vram=\$(rocm-smi --showmemuse 2>/dev/null | awk '/VRAM%/ {printf \"%s,\", \$NF}')")"
    echo "-- $n: $r"
    [[ "$r" == *home=ok* && "$r" == *docker=200* ]] || bad="$bad $n"
done
if [[ -n "$bad" ]]; then
    for n in $bad; do grep -qx "$n" "$R/exclude_nodes.txt" || echo "$n" >> "$R/exclude_nodes.txt"; done
    k=$(cat "$R/.resub_count" 2>/dev/null || echo 0)
    if ((k < MAX_RESUB)); then
        echo $((k + 1)) > "$R/.resub_count"
        echo "== bad node(s):$bad; replacement: $(sbatch -x "$(paste -sd, "$R/exclude_nodes.txt")" "$R/hold2b.sbatch" 2>&1)"
    else
        echo "== bad node(s):$bad; resubmit limit $MAX_RESUB reached, not resubmitting"
    fi
    exit 0
fi
echo 0 > "$R/.resub_count"
# Guard on NODE1: removes our containers there when the step is signalled (preemption / end of the allocation).
srun --overlap -D /tmp -N1 -n1 -w "$NODE1" --gres=gpu:8 /bin/bash -c \
    "trap '$rm_ours; exit 0' TERM INT; while sleep 30; do :; done; exit 0" < /dev/null >> "$R/logs/guard-$JOB.log" 2>&1 &
guard=$!
cleanup() {
    echo "== $(date -u '+%F %T') cleanup"
    touch "$Q/STOP"
    bash -c "$rm_ours" 2>&1
    timeout 60 srun --overlap -D /tmp -N1 -n1 -w "$NODE1" --gres=gpu:8 /bin/bash -c "$rm_ours; exit 0" < /dev/null 2>&1
    kill "$guard" 2>/dev/null
}
trap cleanup EXIT TERM INT
rm -f "$Q/STOP"
echo "runner job $JOB on $NODE0 (decode $NODE1) since $(date -u '+%F %T')" > "$Q/RUNNER"
while [ ! -e "$Q/STOP" ]; do
    for f in "$Q"/*.sh; do
        [ -e "$f" ] || continue
        id=${f%.sh}
        mv "$f" "$id.run" || continue
        ( cd "$R" && bash "$id.run" > "$id.log" 2>&1; echo $? > "$id.rc" ) &
    done
    sleep 3
    ls "$Q" > /dev/null
done
echo "runner stopped $(date -u '+%F %T')" >> "$Q/RUNNER"
wait
