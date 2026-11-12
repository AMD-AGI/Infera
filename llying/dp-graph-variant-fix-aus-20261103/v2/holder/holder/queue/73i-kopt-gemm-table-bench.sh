#!/usr/bin/env bash
# kopt 73i (microbench only): combine the decode GEMM tuning results (73a merge shapes + 73b-73h) into one table and time every
# shape at all 20 verify M values with the production tables vs production + the new table. 1 GPU.
B=/perf_apps/liyingli/bench_agentx/kopt-20261109/bench
O=$B/out/kopt73i.log
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
docker exec -i $C python3 - <<'PY' >> $O 2>&1
import glob, pandas as pd
fs = ['/perf_apps/liyingli/bench_agentx/kopt-20261109/tune/tuned_merge.csv'] + sorted(glob.glob('/perf_apps/liyingli/bench_agentx/kopt-20261109/tune/tuned_decode_*.csv'))
df = pd.concat([pd.read_csv(f) for f in fs if f.endswith('.csv')], ignore_index=True)
keys = ['gfx','cu_num','M','N','K','bias','dtype','outdtype','scaleAB','bpreshuffle']
df = df.sort_values('us').drop_duplicates(keys).sort_values(['N','K','M'])
df.to_csv('/perf_apps/liyingli/bench_agentx/kopt-20261109/tune/tuned_decode_all.csv', index=False)
print('combined', len(df), 'rows from', len(fs), 'files')
PY
docker exec -e HIP_VISIBLE_DEVICES=0 $C bash -lc "cd $B && timeout 1200 python3 gemm_table_bench.py --label production --out out/gemm_table_prod.json" 2>&1 | grep -v "^\[aiter\]" >> $O
docker exec -e HIP_VISIBLE_DEVICES=0 -e AITER_CONFIG_GEMM_BF16=$T/tuned_decode_all.csv $C bash -lc "cd $B && timeout 1200 python3 gemm_table_bench.py --label tuned --out out/gemm_table_tuned.json" 2>&1 | grep -v "^\[aiter\]" >> $O
docker exec $C bash -lc "cd $B && python3 gemm_table_bench.py --compare out/gemm_table_prod.json out/gemm_table_tuned.json" >> $O 2>&1
echo "$(date -u +%FT%TZ) done" >> $O
tail -30 $O
