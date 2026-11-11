#!/usr/bin/env bash
# lm_head GEMM (N 19360, K 6144, bf16; production torch.matmul / hipBLASLt default): aiter a16w16 tuner on M 96-768, then
# torch.matmul vs aiter tgemm with the tuned table (CUDA graph, 1 GPU). Microbench only.
B=/perf_apps/liyingli/bench_agentx/kopt-20261109/bench
O=$B/out/lmhead.log
C=llying-kopt-bench
cleanup() { docker rm -f $C >/dev/null 2>&1; kill "$g" 2>/dev/null; }
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
echo "$(date -u +%FT%TZ) start on $(hostname)" > $O
srun --jobid="$SLURM_JOB_ID" --overlap -N1 -n1 -w "$(hostname -s)" --export=NONE \
    /bin/bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/guard_containers.sh llying-kopt- \
    >> /perf_apps/liyingli/vfix-logs/opt/guard.log 2>&1 &
g=$!
for i in $(seq 30); do
    rocm-smi --showmemuse 2>/dev/null | awk '/VRAM%/ {if ($NF + 0 > 50) b = 1} END {exit b}' && break
    echo "$(date -u +%T) waiting for VRAM" >> $O; sleep 60
done
docker rm -f $C >/dev/null 2>&1
docker run -d --name $C --network host --ipc host --shm-size 32g --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 -v /perf_apps:/perf_apps \
  -v llying-kopt-cache:/root/.cache -v llying-kopt-aiter-jit:/sgl-workspace/aiter/aiter/jit \
  --entrypoint sleep infera-sglang:v0522-fdbafb60 infinity >/dev/null || exit 1
T=/perf_apps/liyingli/bench_agentx/kopt-20261109/tune
rm -f $T/tuned_lmhead.csv $T/profile_lmhead.csv
docker exec $C bash -lc "cd /sgl-workspace/aiter && timeout 1500 python3 csrc/gemm_a16w16/gemm_tuner.py -i $T/untuned_lmhead.csv -o $T/tuned_lmhead.csv -o2 $T/profile_lmhead.csv --mp 8" \
  > $T/tune_lmhead.log 2>&1
echo "$(date -u +%T) tuner rc=$?" >> $O
cat $T/tuned_lmhead.csv >> $O 2>&1
docker exec -e HIP_VISIBLE_DEVICES=0 -e AITER_CONFIG_GEMM_BF16=$T/tuned_lmhead.csv $C bash -lc \
  "cd $B && timeout 600 python3 lmhead_bench.py --ms 96,192,384,576,768" 2>&1 | grep -v "^\[aiter\]" >> $O
echo "$(date -u +%FT%TZ) done" >> $O
tail -30 $O
