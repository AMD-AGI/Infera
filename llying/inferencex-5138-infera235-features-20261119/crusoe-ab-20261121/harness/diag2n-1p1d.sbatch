#!/usr/bin/env bash
#SBATCH -J lyl-ix5138-diag2n
#SBATCH -p amd-spur-v2
#SBATCH -A amd-agentx
#SBATCH -q amd-burst-qos
#SBATCH -N 2
#SBATCH --gres gpu:8
#SBATCH --exclusive
#SBATCH -t 00:45:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/ix5138-crusoe-20261120/logs/diag2n-1p1d-%j.out
# Follow-up to diag2n.sbatch: 1P1D with MC_RDMA_NOTIFY_ENABLED=0, since 1P1D as #5138 has it did not register there.
# Deployment + checks + smoke + Mooncake error counts, no AgentX. Node 0 holds the prefill, node 1 the TP8 decode.
# Spur may start this script on every node: the others wait for node 0's done marker. Every step exits 0.
source /home/liyingli/bench_agentx/ix5138-crusoe-20261120/harness/lib.sh
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes=("${nodes[@]%%.*}")
marker=$R/.done-$SLURM_JOB_ID
if [[ "$(hostname -s)" != "${nodes[0]}" ]]; then
    while [[ ! -e "$marker" ]]; do sleep 10; done
    exit 0
fi
trap 'touch "$marker"' EXIT
NODE_HOSTS=("${nodes[0]}" "${nodes[1]}")
say "diag job $SLURM_JOB_ID: node 0 ${NODE_HOSTS[0]}, node 1 ${NODE_HOSTS[1]}"
bad=()
for n in 0 1; do
    facts=$(on "$n" 'echo "$(cat /sys/module/amdgpu/version) $(ls -d /sys/class/drm/card*/device/mem_info_vram_total | wc -l) \
        $(curl -s -m 15 -o /dev/null -w %{http_code} --unix-socket /var/run/docker.sock http://localhost/images/json)"')
    echo "node $n ${NODE_HOSTS[$n]}: amdgpu, GPUs, docker = $facts; VRAM % $(vram_pct "$n")"
    read -r drv gpus dock <<<"$facts"
    if [[ "$drv" != 6.16.13 || "$gpus" != 8 || "$dock" != 200 ]] || ! wait_idle "$n" 120; then
        bad+=("${NODE_HOSTS[$n]}")
    fi
done
if ((${#bad[@]})); then
    for h in "${bad[@]}"; do grep -qx "$h" "$R/exclude_nodes.txt" || echo "$h" >>"$R/exclude_nodes.txt"; done
    k=$(cat "$R/.diag2n1p_resubmits" 2>/dev/null || echo 0)
    if ((k < 3)); then
        echo $((k + 1)) >"$R/.diag2n1p_resubmits"
        say "unusable: ${bad[*]}; resubmitted: $(cd /tmp && sbatch -x "$(paste -sd, "$R/exclude_nodes.txt")" "$R/harness/diag2n-1p1d.sbatch" 2>&1)"
    else
        say "unusable: ${bad[*]}; resubmit limit reached"
    fi
    exit 0
fi

t0=$SECONDS
(pull_images 0 && stage_model 0) >"$R/logs/diag2n-$SLURM_JOB_ID.node0-prep.log" 2>&1 &
p0=$!
(pull_images 1 && stage_model 1) >"$R/logs/diag2n-$SLURM_JOB_ID.node1-prep.log" 2>&1 &
p1=$!
wait $p0
r0=$?
wait $p1
r1=$?
say "images and model: node 0 rc=$r0, node 1 rc=$r1, $((SECONDS - t0)) s"
((r0 == 0 && r1 == 0)) || { tail -3 "$R"/logs/diag2n-"$SLURM_JOB_ID".node*-prep.log; exit 0; }

export ROUTER_WAIT=900
run_up() {  # label config places...
    local label=$1 config=$2
    shift 2
    NODE_HOSTS="${NODE_HOSTS[*]}" bash "$R/harness/point2.sh" "$label" "$config" 0 up 30000 "$@" </dev/null
    say "$label: $(grep -h '^mooncake decode0\|^FAIL\|router:' "$R/runs/$label/notes.txt" 2>/dev/null | tr '\n' ' ')"
}
run_up diag-1p1d-n0 1p1d-notify0 prefill:0:0,1,2,3 decode:1:0,1,2,3,4,5,6,7
say "diag job done"
exit 0
