#!/usr/bin/env bash
#SBATCH -J lyl-dsv41-opt
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
# One-node run-and-exit job on half a node (4 GPUs, half the CPUs and memory; crusoe does not allow holding idle
# nodes): node checks (scripts/node_check.sh; a refused node is excluded and the job resubmitted) -> node_prep.sh
# (weights to NVMe, engine image) -> scripts/pipeline4.sh with ARM (an arm of arms/; arms/base.sh = the baseline)
# -> remove our containers -> exit 0 (spur ends the job on a non-zero step). Prefill takes the two lower GPUs the job
# was given, decode the two higher; the containers see every GPU, so the GPU lists are always explicit.
#   sbatch -J lyl-dsv41-<arm> -x "$(paste -sd, exclude_nodes.txt)" \
#       --export=ALL,ARM=arms/<arm>.sh,CONC=704,GATE=1[,PROFILE_AT=3300,BASELINE=<artifact dir>:P,D] run_opt2.sbatch
W=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123
cd "$W" || { echo "node $(hostname -s) does not see $W"; exit 0; }
source scripts/env.sh
source scripts/node_check.sh
JOB=$SLURM_JOB_ID
echo "== $(ts) job $JOB ($SLURM_JOB_NAME, resubmission ${RESUB:-0}) on $(hostname -s): ARM=${ARM:-} CONC=${CONC:-} GATE=${GATE:-} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-}"
[[ -n "${ARM:-}" ]] || { echo "== no ARM given; ending the job"; exit 0; }
mapfile -t g < <(tr ',' '\n' <<<"${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-${GPU_DEVICE_ORDINAL:-}}}" | grep -E '^[0-7]$' | sort -n)
(( ${#g[@]} == 4 )) || { echo "== the job did not name 4 GPUs (${g[*]:-none}); ending the job"; exit 0; }
export PREFILL_GPUS="${g[0]},${g[1]}" DECODE_GPUS="${g[2]},${g[3]}"
echo "-- prefill GPUs $PREFILL_GPUS, decode GPUs $DECODE_GPUS; co-tenants: $(squeue -w "$(hostname -s)" -h -o '%i:%u:%b:%C' 2>/dev/null | tr '\n' ' ')"
if ! node_ok; then
    resubmit "$W/run_opt2.sbatch"
    echo "== node refused; ending the job"; exit 0
fi
rm_ours() { docker ps -a --format '{{.Names}}' | grep -E '^lyl-dsv41' | xargs -r docker rm -f > /dev/null 2>&1; rm -rf /dev/shm/lyl-dsv41-umbp; }
trap rm_ours EXIT TERM INT

bash scripts/node_prep.sh > "logs/job-$JOB-prep.log" 2>&1
tail -1 "logs/job-$JOB-prep.log"
grep -q "^prep ok" "logs/job-$JOB-prep.log" || { echo "== node_prep failed; ending the job"; exit 0; }

CONC=${CONC:-704} GATE=${GATE:-1} bash scripts/pipeline4.sh > "logs/job-$JOB-pipeline.log" 2>&1
echo "== $(ts) pipeline rc=$?"
# FALLBACK_ARM: when the arm cannot come up at all (an untested backend), run this arm on the same allocation.
if [[ -n "${FALLBACK_ARM:-}" ]] && tail -5 "logs/job-$JOB-pipeline.log" | grep -q "up FAILED"; then
    echo "== $(ts) $ARM did not come up; running FALLBACK_ARM $FALLBACK_ARM"
    ARM=$FALLBACK_ARM CONC=${FALLBACK_CONC:-${CONC:-704}} GATE=${GATE:-1} bash scripts/pipeline4.sh > "logs/job-$JOB-pipeline-fallback.log" 2>&1
    echo "== $(ts) fallback pipeline rc=$?"
fi
echo "== $(ts) done"
exit 0
