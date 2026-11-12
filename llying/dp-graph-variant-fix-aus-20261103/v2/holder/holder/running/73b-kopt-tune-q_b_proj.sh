#!/usr/bin/env bash
# kopt 73b (microbench only): aiter a16w16 tuner for q_b_proj at the 17 remaining TP8 decode verify M values (4-128),
# for a decode GEMM table covering every captured graph batch. 8 GPUs.
B=/perf_apps/liyingli/bench_agentx/kopt-20261109/bench
O=$B/out/kopt73b.log
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
  --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 -v /perf_apps:/perf_apps -v /perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/kit/overlay-kopt:/yihou_overlay:ro \
  -v /perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/kit/aiter_csv/yihou_glm52_dpa_bf16_tuned_gemm.csv:/sgl-workspace/aiter/aiter/configs/model_configs/yihou_glm52_dpa_bf16_tuned_gemm.csv:ro -v /perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/kit/aiter_csv/yihou_glm52_ptpc_bf16_tuned_gemm.csv:/sgl-workspace/aiter/aiter/configs/model_configs/yihou_glm52_ptpc_bf16_tuned_gemm.csv:ro -v llying-kopt-cache:/root/.cache -v llying-kopt-aiter-jit:/sgl-workspace/aiter/aiter/jit \
  --entrypoint sleep infera-sglang:v0522-fdbafb60 infinity >/dev/null || exit 1
T=/perf_apps/liyingli/bench_agentx/kopt-20261109/tune
rm -f $T/tuned_decode_q_b_proj.csv $T/profile_decode_q_b_proj.csv
docker exec $C bash -lc "cd /sgl-workspace/aiter && timeout 2400 python3 csrc/gemm_a16w16/gemm_tuner.py -i $T/untuned_decode_q_b_proj.csv -o $T/tuned_decode_q_b_proj.csv -o2 $T/profile_decode_q_b_proj.csv --mp 8" \
  > $T/tune_decode_q_b_proj.log 2>&1
echo "$(date -u +%T) tuner q_b_proj rc=$?" >> $O
wc -l $T/tuned_decode_q_b_proj.csv >> $O 2>&1
echo "$(date -u +%FT%TZ) done" >> $O
tail -30 $O
