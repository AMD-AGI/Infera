#!/usr/bin/env bash
#SBATCH -J lyl-dsv41-pair
#SBATCH -p amd-spur-v2
#SBATCH -A amd-hyperloom
#SBATCH -q amd-hyperloom-qos
#SBATCH -N 1
#SBATCH --gres gpu:4
#SBATCH -c 118
#SBATCH --mem 1200G
#SBATCH -t 06:00:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/job-%j.out
# Paired A/B on one node (EXPERIMENT 4.11: node / co-tenant differences are ~3%, as large as the candidates): node checks
# -> node_prep.sh -> scripts/pipeline12.sh with ARM_A and A_ENV -> remove our containers -> the same with ARM_B and B_ENV.
# A_ENV / B_ENV: space-separated VAR=value pairs for that run only. Everything else (CONC, GATE, PROFILE_AT, BASELINE,
# UMBP_*, THP_OFF, UVA_VMM, HOSTALLOC_*) applies to both. About 2 h per run.
#   sbatch -x "$(paste -sd, exclude_nodes.txt)" --export=ALL,ARM_A=arms/<a>.sh,ARM_B=arms/<b>.sh[,A_ENV=..,B_ENV=..],... run_pair2.sbatch
W=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123
cd "$W" || { echo "node $(hostname -s) does not see $W"; exit 0; }
source scripts/env.sh
source scripts/node_check.sh
JOB=$SLURM_JOB_ID
echo "== $(ts) job $JOB ($SLURM_JOB_NAME, resubmission ${RESUB:-0}) on $(hostname -s): A=${ARM_A:-} [${A_ENV:-}] B=${ARM_B:-} [${B_ENV:-}] CONC=${CONC:-} GATE=${GATE:-} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-}"
[[ -n "${ARM_A:-}" && -n "${ARM_B:-}" ]] || { echo "== ARM_A and ARM_B are required; ending the job"; exit 0; }
mapfile -t g < <(tr ',' '\n' <<<"${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-${GPU_DEVICE_ORDINAL:-}}}" | grep -E '^[0-7]$' | sort -n)
(( ${#g[@]} == 4 )) || { echo "== the job did not name 4 GPUs (${g[*]:-none}); ending the job"; exit 0; }
export PREFILL_GPUS="${g[0]},${g[1]}" DECODE_GPUS="${g[2]},${g[3]}"
echo "-- prefill GPUs $PREFILL_GPUS, decode GPUs $DECODE_GPUS; co-tenants: $(squeue -w "$(hostname -s)" -h -o '%i:%u:%b:%C' 2>/dev/null | tr '\n' ' ')"
if ! node_ok; then
    resubmit "$W/run_pair2.sbatch"
    echo "== node refused; ending the job"; exit 0
fi
rm_ours() { docker ps -a --format '{{.Names}}' | grep -E '^lyl-dsv41' | xargs -r docker rm -f > /dev/null 2>&1; rm -rf /dev/shm/lyl-dsv41-umbp; }
trap rm_ours EXIT TERM INT

bash scripts/node_prep.sh > "logs/job-$JOB-prep.log" 2>&1
tail -1 "logs/job-$JOB-prep.log"
grep -q "^prep ok" "logs/job-$JOB-prep.log" || { echo "== node_prep failed; ending the job"; exit 0; }

for side in A B; do
    arm_var=ARM_$side env_var=${side}_ENV
    echo "== $(ts) run $side: ${!arm_var} [${!env_var:-}]"
    env ARM="${!arm_var}" CONC=${CONC:-704} GATE=${GATE:-1} ${!env_var:-} bash scripts/pipeline12.sh > "logs/job-$JOB-pipeline-$side.log" 2>&1
    echo "== $(ts) run $side pipeline rc=$?: $(grep -a 'AgentX c[0-9]* rc' "logs/job-$JOB-pipeline-$side.log" | tail -1 | grep -oE 'total tok/s/GPU [0-9,]+')"
    rm_ours
done
echo "== $(ts) done"
exit 0
