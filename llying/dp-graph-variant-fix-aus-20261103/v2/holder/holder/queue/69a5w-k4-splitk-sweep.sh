#!/usr/bin/env bash
# KERNEL-PLAN-20261109 7.4 item 1 (K4): TP8 decode dense projections (fused_qkv_a, q_b_proj, o_proj; bf16) at per-rank
# rows 32-128, production tgemm.mm (production tuned CSVs mounted as on the decode) against aiter's Triton split-K
# (reduce kernel) and atomic split-K kernels, cold weights, CUDA graphs; config sweep split over the batch host's 8 GPUs
# (bench/k4_splitk_sweep.py), ~15 min. Runs after the two synthetic traces and before the 3600 s score.
B=/perf_apps/liyingli/bench_agentx/kopt-20261109/bench
K=/perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/kit/aiter_csv
MC=/sgl-workspace/aiter/aiter/configs/model_configs
D=$B/out/k4_splitk
O=$B/out/k4_splitk.log
C=llying-kopt-bench
cleanup() { docker rm -f $C >/dev/null 2>&1; kill "$g" 2>/dev/null; }
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
echo "$(date -u +%FT%TZ) start on $(hostname)" > $O
srun --jobid="$SLURM_JOB_ID" --overlap -N1 -n1 -w "$(hostname -s)" --export=NONE \
    /bin/bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/guard_containers.sh llying-kopt- \
    >> /perf_apps/liyingli/vfix-logs/opt/guard.log 2>&1 &
g=$!
# The previous item's engines were just removed; the driver frees their VRAM 10-40 min later.
for i in $(seq 30); do
    rocm-smi --showmemuse 2>/dev/null | awk '/VRAM%/ {if ($NF + 0 > 50) b = 1} END {exit b}' && break
    echo "$(date -u +%T) waiting for VRAM" >> $O; sleep 60
done
docker rm -f $C >/dev/null 2>&1
docker run -d --name $C --network host --ipc host --shm-size 32g --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 -v /perf_apps:/perf_apps \
  -v $K/yihou_glm52_dpa_bf16_tuned_gemm.csv:$MC/yihou_glm52_dpa_bf16_tuned_gemm.csv:ro \
  -v $K/yihou_glm52_ptpc_bf16_tuned_gemm.csv:$MC/yihou_glm52_ptpc_bf16_tuned_gemm.csv:ro \
  -v llying-kopt-cache:/root/.cache -v llying-kopt-aiter-jit:/sgl-workspace/aiter/aiter/jit \
  --entrypoint sleep infera-sglang:v0522-fdbafb60 infinity >/dev/null || exit 1
rm -rf $D; mkdir -p $D
for i in 0 1 2 3 4 5 6 7; do
  docker exec -e HIP_VISIBLE_DEVICES=$i $C bash -lc "cd $B && timeout 2400 python3 k4_splitk_sweep.py --part $i/8 --out $D" \
    > $D/part$i.log 2>&1 &
done
wait
docker exec $C bash -lc "cd $B && python3 k4_splitk_sweep.py --merge $D" >> $O 2>&1
grep -h -E "Error|Traceback" $D/part*.log | sort | uniq -c | head -5 >> $O
echo "$(date -u +%FT%TZ) done" >> $O
cat $O
