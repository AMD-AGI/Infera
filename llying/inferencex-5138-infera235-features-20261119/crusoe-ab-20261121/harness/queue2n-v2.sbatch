#!/usr/bin/env bash
#SBATCH -J lyl-ix5138-q
#SBATCH -p amd-spur-v2
#SBATCH -A amd-agentx
#SBATCH -q amd-agentx-qos
#SBATCH -N 2
#SBATCH --gres gpu:8
#SBATCH --exclusive
#SBATCH -t 04:30:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/ix5138-crusoe-20261120/logs/queue2n-v2-%j.out
# Two-node counterpart of queue.sbatch: runs the list's points one after another, then ends. Node 0 holds the prefills,
# etcd, the router and the AgentX client; node 1 the decode. List lines: LABEL CONFIG CONC fast|full PLACE... (point.sh
# arguments after the port base, nodes 0 and 1). A point already recorded ok in runs/points.jsonl is skipped; a point
# starts only if the job's time left covers it (120 min). Spur may start this script on every node: the others wait
# for node 0's done marker. Every step exits 0. Usage: sbatch -x <exclude> queue2n.sbatch LIST_FILE
source /home/liyingli/bench_agentx/ix5138-crusoe-20261120/harness/lib.sh
list=$1
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes=("${nodes[@]%%.*}")
marker=$R/.done-$SLURM_JOB_ID
if [[ "$(hostname -s)" != "${nodes[0]}" ]]; then
    while [[ ! -e "$marker" ]]; do sleep 10; done
    exit 0
fi
trap 'touch "$marker"' EXIT
NODE_HOSTS=("${nodes[0]}" "${nodes[1]}")
say "queue2n job $SLURM_JOB_ID: node 0 ${NODE_HOSTS[0]}, node 1 ${NODE_HOSTS[1]}: $list"
bad=()
for n in 0 1; do
    facts=$(on "$n" 'echo "$(cat /sys/module/amdgpu/version) $(ls -d /sys/class/drm/card*/device/mem_info_vram_total | wc -l) \
        $(curl -s -m 15 -o /dev/null -w %{http_code} --unix-socket /var/run/docker.sock http://localhost/images/json)"')
    echo "node $n ${NODE_HOSTS[$n]}: amdgpu, GPUs, docker = $facts; VRAM % $(vram_pct "$n")"
    read -r drv gpus dock <<<"$facts"
    if [[ "$drv" != 6.16.13 || "$gpus" != 8 || "$dock" != 200 || ! -r "$list" ]] || ! wait_idle "$n" 120; then
        bad+=("${NODE_HOSTS[$n]}")
    fi
done
if ((${#bad[@]})); then
    for h in "${bad[@]}"; do grep -qx "$h" "$R/exclude_nodes.txt" || echo "$h" >>"$R/exclude_nodes.txt"; done
    k=$(cat "$list.resubmits" 2>/dev/null || echo 0)
    if ((k < 3)); then
        echo $((k + 1)) >"$list.resubmits"
        say "unusable: ${bad[*]}; resubmitted: $(cd /tmp && sbatch -x "$(paste -sd, "$R/exclude_nodes.txt")" "$R/harness/queue2n-v2.sbatch" "$list" 2>&1)"
    else
        say "unusable: ${bad[*]}; resubmit limit reached"
    fi
    exit 0
fi
t0=$SECONDS
(pull_images 0 && stage_model 0) >"$R/logs/queue2n-v2-$SLURM_JOB_ID.node0-prep.log" 2>&1 &
p0=$!
(pull_images 1 && stage_model 1) >"$R/logs/queue2n-v2-$SLURM_JOB_ID.node1-prep.log" 2>&1 &
p1=$!
wait $p0
r0=$?
wait $p1
r1=$?
say "images and model: node 0 rc=$r0, node 1 rc=$r1, $((SECONDS - t0)) s"
((r0 == 0 && r1 == 0)) || { tail -3 "$R"/logs/queue2n-v2-"$SLURM_JOB_ID".node*-prep.log; exit 0; }
end=$(date -d "$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'EndTime=\K\S+')" +%s)
while read -r label config conc mode places; do
    [[ -z "$label" || "$label" == \#* ]] && continue
    if grep -q "\"label\": \"$label\", .*\"status\": \"ok\"" "$R/runs/points.jsonl" 2>/dev/null; then
        say "$label already done, skipped"
        continue
    fi
    if (($(date +%s) + 7200 > end)); then
        say "$label skipped: $(((end - $(date +%s)) / 60)) min left, needs 120"
        continue
    fi
    # shellcheck disable=SC2086  # places are separate point.sh arguments
    NODE_HOSTS="${NODE_HOSTS[*]}" bash "$R/harness/point3.sh" "$label" "$config" "$conc" "$mode" 30000 $places </dev/null
    tail -1 "$R/runs/points.jsonl"
done <"$list"
say "queue2n job done"
exit 0
